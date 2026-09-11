"""Agent orchestration: LlamaIndex ``FunctionAgent`` replacing ReasonerAgent.

Ticket: T6 (Agent 替换). Depends on T2 (``DrbrainLLM``), T4 (fusion
retrievers). Tools = drbrain's 8 graph tools (``extractor/agent_tools.py``,
``execute_tool`` as the execution body) + an optional fused-retrieval tool.

0.14.23 API note: the classic ``FunctionCallingAgentWorker``/``AgentRunner``
are gone from ``llama_index.core.agent``; the successor is the workflow-based
:class:`FunctionAgent` (with ``ReActAgent`` as the non-function-calling
fallback). ``FunctionAgent.take_step`` hard-requires
``llm.metadata.is_function_calling_model`` and calls ``llm.achat_with_tools``
— ``DrbrainLLM`` (rag/llm.py, T2-owned, untouched) advertises ``False`` and
never forwards ``tools``. So this module defines :class:`_AgentFunctionLLM`, a
``FunctionCallingLLM`` glue over ``DrbrainLLM`` that adds exactly those two
capabilities (mirroring the native LlamaIndex function-calling
reference implementation) while delegating every completion to the same drbrain
fallback chain / ApiCache / metrics.

Session persistence is minimal but two-way: when ``session_id`` is given,
prior turns are restored from the ``agent_sessions``/``agent_messages`` tables
and injected as ``chat_history`` (T9 read recovery + SessionAgent-style
compression, see :func:`load_session_history`), and the new run is appended in
a shape ``SessionAgent.load_session`` can read back.

Everything degrades gracefully when llama-index is not installed:
``build_agent`` returns ``None`` and ``reason_llamaindex`` returns an error
dict, so the CLI and existing tests never break.
"""

from __future__ import annotations

from drbrain.rag.agent_tools import (
    _resolve_papers_dir as _resolve_papers_dir,
    _schema_to_model as _schema_to_model,
    _resolve_durable_policy as _resolve_durable_policy,
    _normalize_side_effect as _normalize_side_effect,
    _durable_tool_definition as _durable_tool_definition,
    _make_graph_tool as _make_graph_tool,
    _make_validate_tool as _make_validate_tool,
    _build_retrieval_tool as _build_retrieval_tool,
    _load_plugin_tools as _load_plugin_tools,
    _string_tuple as _string_tuple,
    _mcp_tool_definition as _mcp_tool_definition,
    _load_mcp_tools as _load_mcp_tools,
)

from drbrain.rag.agent_llm import (
    AgentFunctionLLM as AgentFunctionLLM,
)

from drbrain.rag.agent_sessions import (
    _history_summary as _history_summary,
    _session_principal_matches as _session_principal_matches,
    load_session_history as load_session_history,
    _persist_reason_session as _persist_reason_session,
)

from drbrain.rag.retrieval import (
    _retrieval_rows as _retrieval_rows,
    retrieve_documents as retrieve_documents,
)

import asyncio
import json
import logging
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

from drbrain.config import ApiConfig, Config, DBConfig, DirsConfig, EmbedConfig, LLMConfig
from drbrain.extractor.agent_tools import TOOL_DEFINITIONS, execute_tool
from drbrain.rag.evidence import (
    INSUFFICIENT_EVIDENCE_MESSAGE,
    INSUFFICIENT_EVIDENCE_STATUS,
    build_evidence_record,
    evidence_ids_from_records,
    has_retrieved_evidence,
)
from drbrain.rag.llm import DrbrainLLM
from drbrain.rag.status import RetrievalStatus, RetrievalUnavailableError
from drbrain.security import public_model_configs, redact_sensitive

try:
    from llama_index.core.agent import FunctionAgent
    from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
    from llama_index.core.llms.function_calling import FunctionCallingLLM
    from llama_index.core.llms.llm import ToolSelection
    from llama_index.core.tools import BaseTool, FunctionTool

    _LLAMA_INDEX_AVAILABLE = True
except ImportError:  # pragma: no cover - envs without llama-index
    FunctionAgent = None  # type: ignore[assignment,misc]
    ChatMessage = None  # type: ignore[assignment,misc]
    ChatResponse = None  # type: ignore[assignment,misc]
    MessageRole = None  # type: ignore[assignment,misc]
    FunctionCallingLLM = None  # type: ignore[assignment,misc]
    ToolSelection = None  # type: ignore[assignment,misc]
    BaseTool = None  # type: ignore[assignment,misc]
    FunctionTool = None  # type: ignore[assignment,misc]
    _LLAMA_INDEX_AVAILABLE = False

log = logging.getLogger(__name__)

__all__ = [
    "_LLAMA_INDEX_AVAILABLE",
    "AgentFunctionLLM",
    "build_agent",
    "load_session_history",
    "reason_llamaindex",
]

#: Base system prompt — identical to ReasonerAgent's (T6 keeps behavior parity).
BASE_SYSTEM_PROMPT = (
    "You are a knowledge graph reasoning assistant. "
    "Use the provided tools to explore the graph and answer questions. "
    "Explain your reasoning step by step."
)

#: Agent-loop LLM settings matching the legacy ReasonerAgent loop.
AGENT_TEMPERATURE = 0.3
AGENT_MAX_TOKENS = 1024

#: Result summary length cap in the returned tool trajectory.
MAX_RESULT_SUMMARY_CHARS = 800

#: Token budget for session-history compression (mirrors
#: ``SessionAgent.DEFAULT_TOKEN_BUDGET``; estimated = ``len(content)//4``).
SESSION_TOKEN_BUDGET = 8000
#: Recent messages kept verbatim when a long history is compressed
#: (mirrors ``SessionAgent._maybe_compress``'s ``keep``).
SESSION_KEEP_RECENT = 6

#: Canonical OpenAI-format tool specs, keyed by tool name — the exact dicts the
#: legacy ReasonerAgent sent over the wire, so tool schemas are byte-identical.
CANONICAL_TOOL_SPECS: dict[str, dict[str, Any]] = {
    d["function"]["name"]: d for d in TOOL_DEFINITIONS
}
GRAPH_TOOL_NAMES: list[str] = list(CANONICAL_TOOL_SPECS)

_JSON_TYPE_MAP: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


# ── small config helpers (dict & typed Config both accepted) ───────────────


def _coerce_cfg(cfg: Config | dict[str, Any]) -> Config:
    """Compatibility alias for the shared boundary conversion."""
    from drbrain.rag.config import coerce_config

    return coerce_config(cfg)


def _llamaindex_from_dict(raw: dict) -> Any:
    from drbrain.config import LlamaIndexConfig

    return LlamaIndexConfig.from_dict(raw)


def _cfg_models(cfg: Any) -> list[dict]:
    """Return the LLM fallback-chain model list from a Config or dict."""
    if isinstance(cfg, dict):
        llm = cfg.get("llm", {})
        return list(llm.get("models", [])) if isinstance(llm, dict) else []
    llm = getattr(cfg, "llm", None)
    if llm is None:
        return []
    models = getattr(llm, "models", None)
    return list(models) if models is not None else []


def _model_version(cfg: Any) -> str:
    """``provider/model`` of the first entry in the LLM fallback chain.

    Matches :attr:`DrbrainLLM.model_name` (and thus the agent loop's
    ``AgentFunctionLLM``), so the recorded version lines up with the model that
    actually produced the answer.
    """
    models = _cfg_models(cfg)
    if not models:
        return "drbrain/unknown"
    first = models[0] or {}
    return f"{first.get('provider', 'openai')}/{first.get('model', 'unknown')}"


def _parse_tool_result(summary: str) -> list[dict[str, Any]]:
    """Parse a tool ``result_summary`` (JSON string) back into rows.

    Retrieval tools return ``json.dumps([{paper_id, node_id, ...}])``. Handles
    a truncated tail (``result_summary`` is capped to
    :data:`MAX_RESULT_SUMMARY_CHARS`) by falling back to a regex scan.
    """
    if not summary:
        return []
    try:
        data = json.loads(summary)
    except (ValueError, TypeError, json.JSONDecodeError):
        data = None
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        return [data]

    # Truncated JSON: pull paper_id/node_id pairs out of the raw text.
    import re

    paper_ids = re.findall(r'"paper_id"\s*:\s*"([^"]*)"', summary)
    node_ids = re.findall(r'"node_id"\s*:\s*"([^"]*)"', summary)
    rows: list[dict[str, Any]] = []
    for i in range(max(len(paper_ids), len(node_ids))):
        row: dict[str, Any] = {}
        if i < len(paper_ids):
            row["paper_id"] = paper_ids[i]
        if i < len(node_ids):
            row["node_id"] = node_ids[i]
        if row:
            rows.append(row)
    return rows


def _evidence_ids_from_tool_calls(tool_calls: list[dict[str, Any]]) -> list[str]:
    """Extract ``paper_id:node_id`` evidence identifiers from retrieval calls.

    Only ``search_documents`` (fused LlamaIndex retrieval) and ``search_tree``
    (cross-paper collapsed tree) contribute evidence; graph tools
    (``search_concepts``, ``get_neighbors``, …) are reasoning steps, not
    answer evidence.
    """
    records: list[dict[str, Any]] = []
    for tc in tool_calls or []:
        name = str(tc.get("name") or "").strip()
        if name not in ("search_documents", "search_tree"):
            continue
        for row in _parse_tool_result(str(tc.get("result_summary") or "")):
            pid = str(row.get("paper_id") or "").strip()
            nid = str(row.get("node_id") or "").strip()
            if pid or nid:
                records.append(row)
    return evidence_ids_from_records(records)


def _evidence_records_from_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return query-engine-shaped evidence rows from allowed retrieval tools only."""
    records: list[dict[str, Any]] = []
    for tc in tool_calls or []:
        if str(tc.get("name") or "").strip() not in ("search_documents", "search_tree"):
            continue
        for row in _parse_tool_result(str(tc.get("result_summary") or "")):
            if str(row.get("paper_id") or "").strip() or str(row.get("node_id") or "").strip():
                raw_sources = row.get("sources")
                if isinstance(raw_sources, list):
                    sources = raw_sources
                else:
                    source = row.get("source")
                    sources = [source] if source else []
                records.append(
                    {
                        "paper_id": str(row.get("paper_id") or "").strip(),
                        "node_id": str(row.get("node_id") or "").strip(),
                        "title": str(row.get("title") or ""),
                        "score": row.get("score", 0.0),
                        "sources": sources,
                    }
                )
    return records


def _record_answer(
    cfg: Any,
    db: Any,
    question: str,
    answer: str,
    tool_calls: list[dict[str, Any]],
    session_id: str | None,
) -> None:
    """Best-effort persist of an answer + its evidence (never raises)."""
    record = getattr(db, "record_answer", None)
    if record is None:
        return
    try:
        record(
            question,
            answer,
            session_id=session_id,
            evidence_ids=_evidence_ids_from_tool_calls(tool_calls),
            model_version=_model_version(cfg),
            retriever_version="llamaindex-agent",
        )
    except Exception:  # pragma: no cover - persistence must not break the answer
        log.warning("[rag] record_answer failed (reason_llamaindex)", exc_info=True)
































def build_agent(
    cfg: Config,
    db: Any = None,
    session_id: str | None = None,
    *,
    graph: Any = None,
    closure_context: str = "",
    temperature: float = AGENT_TEMPERATURE,
    max_tokens: int = AGENT_MAX_TOKENS,
    include_retrieval: bool = True,
    plugins_dir: str | Path | None = None,
    mcp_servers: list[dict[str, Any]] | None = None,
    require_trusted_mcp: bool = False,
    tool_broker: Any = None,
    tool_policy: Any = None,
    workflow_step: str | None = None,
    rag_generation: str | None = None,
    models_override: list[dict] | None = None,
) -> Any | None:
    """Assemble the LlamaIndex :class:`FunctionAgent`.

    Tools: the 7 drbrain graph tools (FunctionTool over ``execute_tool``),
    plus ``kg_validate`` (KG consistency check — T9, only when ``graph`` is
    given) and, when the LlamaIndex index exists on disk, a fused-retrieval
    ``search_documents`` tool. ``system_prompt`` keeps ReasonerAgent's
    closure_context injection. Returns ``None`` when llama-index is
    unavailable (callers fall back to legacy). ``session_id`` is accepted for
    interface parity; history read/restore happens in
    :func:`reason_llamaindex` via :func:`load_session_history`.

    ``tool_broker`` is an additive durable-loop hook. When present,
    ``workflow_step`` selects a policy-filtered surface and every exposed tool
    runs through the broker; absent it, construction and direct execution keep
    their historic behavior.
    """
    if not _LLAMA_INDEX_AVAILABLE:
        return None
    cfg = _coerce_cfg(cfg)
    papers_dir = _resolve_papers_dir(cfg, db)
    policy = _resolve_durable_policy(tool_broker, tool_policy)
    if tool_broker is not None and not workflow_step:
        raise ValueError("brokered agents require workflow_step")

    tools = []
    for name in GRAPH_TOOL_NAMES:
        graph_tool = _make_graph_tool(
            name,
            db,
            graph,
            papers_dir,
            tool_broker=tool_broker,
            tool_policy=policy,
            workflow_step=workflow_step,
        )
        if graph_tool is not None:
            tools.append(graph_tool)
    # T9: kg_validate (KG consistency check) as the 8th tool, only with a graph.
    vt = _make_validate_tool(
        db,
        graph,
        tool_broker=tool_broker,
        tool_policy=policy,
        workflow_step=workflow_step,
    )
    if vt is not None:
        tools.append(vt)
    if include_retrieval:
        rt = _build_retrieval_tool(
            cfg,
            db,
            graph,
            tool_broker=tool_broker,
            tool_policy=policy,
            workflow_step=workflow_step,
            rag_generation=rag_generation,
        )
        if rt is not None:
            tools.append(rt)
    if plugins_dir:
        tools.extend(
            _load_plugin_tools(
                plugins_dir,
                tool_broker=tool_broker,
                tool_policy=policy,
                workflow_step=workflow_step,
            )
        )
    if mcp_servers:
        tools.extend(
            _load_mcp_tools(
                mcp_servers,
                require_trusted=require_trusted_mcp,
                tool_broker=tool_broker,
                tool_policy=policy,
                workflow_step=workflow_step,
            )
        )

    system_prompt = BASE_SYSTEM_PROMPT
    if closure_context:
        system_prompt += (
            "\n\nInferred relations from logical closure "
            "(distinguished by --[inferred: ...]-->):\n" + closure_context
        )

    llm = AgentFunctionLLM(
        cfg,
        temperature=temperature,
        max_tokens=max_tokens,
        models_override=models_override,
    )
    agent = FunctionAgent(
        name="drbrain-reasoner",
        description="Knowledge graph reasoning assistant with graph tools",
        system_prompt=system_prompt,
        tools=tools,
        llm=llm,
        streaming=False,  # DrbrainLLM streams single-chunk; achat path is real
        early_stopping_method="generate",
    )
    return agent


# ── session persistence (write-only; read/restore + compression → T9) ───────










def reason_llamaindex(
    cfg: Config,
    db: Any = None,
    question: str = "",
    max_turns: int = 5,
    session_id: str | None = None,
    *,
    graph: Any = None,
    closure_context: str = "",
    principal: str | None = None,
) -> dict[str, Any]:
    """Run the LlamaIndex FunctionAgent over the drbrain graph tools.

    Returns ``{answer, tool_calls: [{name, args, result_summary}], turns,
    engine: "llamaindex"}`` (plus ``session_id`` when persisted). On any
    failure — llama-index unavailable, session missing, LLM exhaustion — the
    dict's ``answer`` carries the message and ``tool_calls`` is empty, so
    callers never crash.
    """
    cfg = _coerce_cfg(cfg)
    try:
        return asyncio.run(
            _areason_llamaindex(
                cfg,
                db,
                question,
                max_turns=max_turns,
                session_id=session_id,
                graph=graph,
                closure_context=closure_context,
                principal=principal,
            )
        )
    except PermissionError as exc:
        message = str(exc)
        return {
            "answer": message,
            "message": message,
            "status": RetrievalStatus.PERMISSION_DENIED.value,
            "sources": [],
            "evidence_ids": [],
            "tool_calls": [],
            "turns": 0,
            "engine": "llamaindex",
        }
    except Exception as exc:
        log.exception("[rag] reason_llamaindex failed")
        return {
            "answer": f"Reasoning error: {exc}",
            "tool_calls": [],
            "turns": 0,
            "engine": "llamaindex",
        }


async def _areason_llamaindex(
    cfg: Config,
    db: Any,
    question: str,
    *,
    max_turns: int,
    session_id: str | None,
    graph: Any,
    closure_context: str,
    principal: str | None,
) -> dict[str, Any]:
    if principal is not None and not str(principal).strip():
        message = "Session access denied: principal must be non-empty"
        return {
            "answer": message,
            "message": message,
            "status": RetrievalStatus.PERMISSION_DENIED.value,
            "evidence_ids": [],
            "tool_calls": [],
            "turns": 0,
            "engine": "llamaindex",
        }

    # T9 read recovery: an existing session's prior turns are restored from
    # agent_messages and injected as chat_history (the system prompt itself is
    # re-injected by build_agent). "new" sessions start with empty history.
    chat_history: list[ChatMessage] = []
    if session_id and session_id != "new":
        if db is None:
            return {
                "answer": "Session not found: " + str(session_id),
                "tool_calls": [],
                "turns": 0,
                "engine": "llamaindex",
            }
        row = db.conn.execute(
            "SELECT 1 FROM agent_sessions WHERE session_id = ? AND status != 'deleted'",
            (session_id,),
        ).fetchone()
        if row is None:
            return {
                "answer": "Session not found: " + str(session_id),
                "tool_calls": [],
                "turns": 0,
                "engine": "llamaindex",
            }
        if principal is not None and not _session_principal_matches(db, session_id, principal):
            message = "Session access denied: " + str(session_id)
            return {
                "answer": message,
                "message": message,
                "status": RetrievalStatus.PERMISSION_DENIED.value,
                "evidence_ids": [],
                "tool_calls": [],
                "turns": 0,
                "engine": "llamaindex",
            }
        chat_history = load_session_history(db, session_id, principal=principal)

    agent = build_agent(
        cfg,
        db,
        session_id,
        graph=graph,
        closure_context=closure_context,
        require_trusted_mcp=bool(getattr(cfg.llamaindex, "mcp_require_trusted", False)),
    )
    if agent is None:
        return {
            "answer": "LlamaIndex is not available (llamaindex.enabled=false or "
            "llama_index not installed). Use --engine legacy.",
            "tool_calls": [],
            "turns": 0,
            "engine": "llamaindex",
        }

    handler = agent.run(
        user_msg=question,
        chat_history=chat_history,
        max_iterations=max(1, int(max_turns)),
    )
    result = await handler

    answer = ""
    response = getattr(result, "response", None)
    if response is not None:
        answer = (response.content or "") if response.content else ""
    if not answer:
        answer = "No answer generated."

    tool_calls: list[dict[str, Any]] = []
    for tc in getattr(result, "tool_calls", None) or []:
        name = getattr(tc, "tool_name", None) or ""
        args = getattr(tc, "tool_kwargs", None) or {}
        summary = ""
        tool_out = getattr(tc, "tool_output", None)
        if tool_out is not None:
            try:
                summary = tool_out.content or str(tool_out) or ""
            except Exception:
                summary = ""
            if getattr(tool_out, "is_error", False):
                summary = f"[error] {summary}"
        tool_calls.append(
            {
                "name": name,
                "args": dict(args) if isinstance(args, dict) else args,
                "result_summary": summary[:MAX_RESULT_SUMMARY_CHARS],
            }
        )

    turns = 0
    try:
        turns = int(await handler.ctx.store.get("num_iterations", default=0) or 0)
    except Exception:
        turns = 0

    out: dict[str, Any] = {
        "answer": answer,
        "tool_calls": tool_calls,
        "turns": turns,
        "engine": "llamaindex",
    }

    evidence_records = _evidence_records_from_tool_calls(tool_calls)
    evidence_ids = evidence_ids_from_records(evidence_records)
    if not has_retrieved_evidence(evidence_records):
        answer = INSUFFICIENT_EVIDENCE_MESSAGE
        out.update(
            {
                "answer": answer,
                "message": answer,
                "status": INSUFFICIENT_EVIDENCE_STATUS,
                "sources": [],
                "evidence_ids": [],
            }
        )
    else:
        out["sources"] = evidence_records
        out["evidence_ids"] = evidence_ids

    resolved_session_id: str | None = None
    if session_id:
        resolved_session_id = _persist_reason_session(
            cfg,
            db,
            session_id,
            question,
            BASE_SYSTEM_PROMPT
            + (
                "\n\nInferred relations from logical closure "
                "(distinguished by --[inferred: ...]-->):\n" + closure_context
                if closure_context
                else ""
            ),
            answer,
            tool_calls,
            _cfg_models(cfg),
            principal=principal,
        )
        out["session_id"] = resolved_session_id

    # Bind the answer to its evidence (best-effort; never affects the result).
    _record_answer(cfg, db, question, answer, tool_calls, resolved_session_id)
    return out
