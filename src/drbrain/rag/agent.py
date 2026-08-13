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
capabilities (mirroring the installed ``llama-index-llms-litellm`` LiteLLM
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

import asyncio
import json
import logging
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from drbrain.config import ApiConfig, Config, DBConfig, DirsConfig, EmbedConfig, LLMConfig
from drbrain.extractor.agent_tools import TOOL_DEFINITIONS, execute_tool
from drbrain.rag.llm import DrbrainLLM

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
#: legacy ReasonerAgent sent to litellm, so tool schemas are byte-identical.
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


def _coerce_cfg(cfg: Any) -> Config:
    """Normalize a dict config to a typed :class:`Config`.

    The CLI's ``ctx.obj["config"]`` is already a typed Config in production;
    tests pass plain dicts. Only the sections the agent touches are mapped.
    """
    if isinstance(cfg, Config) or not isinstance(cfg, dict):
        return cfg
    return Config(
        llm=LLMConfig(**cfg.get("llm", {})),
        db=DBConfig(**cfg.get("db", {})),
        dirs=DirsConfig(**cfg.get("dirs", {})),
        api=ApiConfig(**cfg.get("api", {})),
        embed=EmbedConfig(**cfg.get("embed", {})),
        llamaindex=_llamaindex_from_dict(cfg.get("llamaindex", {})),
    )


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
    ids: list[str] = []
    for tc in tool_calls or []:
        name = str(tc.get("name") or "").strip()
        if name not in ("search_documents", "search_tree"):
            continue
        for row in _parse_tool_result(str(tc.get("result_summary") or "")):
            pid = str(row.get("paper_id") or "").strip()
            nid = str(row.get("node_id") or "").strip()
            if pid or nid:
                ids.append(f"{pid}:{nid}" if pid and nid else (pid or nid))
    seen: set[str] = set()
    out: list[str] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


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


def _resolve_papers_dir(cfg: Any, db: Any) -> Path | None:
    """Resolve the papers data directory.

    Prefers ``cfg.dirs.papers`` when it resolves to an existing path (the CLI
    runs with CWD matching the config); falls back to ``db.path.parent/papers``
    (legacy ReasonerAgent behavior).
    """
    papers: Any = None
    if isinstance(cfg, dict):
        papers = (cfg.get("dirs", {}) or {}).get("papers")
    else:
        dirs = getattr(cfg, "dirs", None)
        papers = getattr(dirs, "papers", None) if dirs is not None else None
    if papers:
        p = Path(papers)
        if p.is_absolute() or p.exists():
            return p
    if db is not None and getattr(db, "path", None):
        return Path(db.path).parent / "papers"
    return None


# ── JSON-schema → pydantic model (for FunctionTool schemas) ─────────────────


def _schema_to_model(name: str, schema: dict) -> Any:
    """Convert a JSON-schema ``parameters`` dict into a pydantic model.

    Used so each FunctionTool's ``metadata.fn_schema`` round-trips the original
    TOOL_DEFINITIONS schema (enums via ``Literal``, optional fields keep their
    declared defaults). ``FunctionTool.acall`` does not validate against it —
    the schema only feeds the OpenAI tool spec.
    """
    from typing import Literal

    from pydantic import Field, create_model

    props = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    fields: dict[str, Any] = {}
    for pname, pinfo in props.items():
        jt = pinfo.get("type", "string")
        if pinfo.get("enum"):
            # ``Literal`` is parameterized with a runtime tuple of literal
            # values, which mypy rejects as an invalid type expression
            # (``valid-type``) — the flattened ``Literal[...]`` is still built
            # correctly at runtime.
            ptype = Literal[tuple(pinfo["enum"])]  # type: ignore[valid-type]
        else:
            ptype = _JSON_TYPE_MAP.get(jt, str)
        if pname in required:
            fields[pname] = (ptype, Field(description=pinfo.get("description", "")))
        else:
            fields[pname] = (
                ptype,
                Field(default=pinfo.get("default", None), description=pinfo.get("description", "")),
            )
    return create_model(f"Tool_{name}", **fields)


# ── graph tools: FunctionTool wrappers over execute_tool ────────────────────


def _make_graph_tool(name: str, db: Any, graph: Any, papers_dir: Path | None) -> Any:
    """Build one ``FunctionTool`` backed by ``agent_tools.execute_tool``.

    ``execute_tool(name, args, db=, graph=, papers_dir=)`` dispatches to the
    canonical handler for the tool — the tool logic is never rewritten here.
    The wrapper returns a JSON string so the LLM sees clean JSON (the legacy
    loop ``json.dumps``-ed tool results the same way).
    """
    spec = CANONICAL_TOOL_SPECS[name]
    fn_spec = spec["function"]

    async def _exec(**kwargs: Any) -> str:
        result = execute_tool(name, kwargs, db=db, graph=graph, papers_dir=papers_dir)
        return json.dumps(result, ensure_ascii=False, default=str)

    return FunctionTool.from_defaults(
        fn=_exec,
        name=fn_spec["name"],
        description=fn_spec["description"],
        fn_schema=_schema_to_model(name, fn_spec["parameters"]),
    )


def _make_validate_tool(db: Any, graph: Any) -> Any | None:
    """Optional ``kg_validate`` tool (T9 decision: ADD as the 8th tool).

    ``kg_validate`` is drbrain's KG-consistency check (TBox/RBox violations +
    graph patterns: debates/gaps) — a semantic asset the graph-backed agent
    should be able to call mid-reasoning to self-validate a hypothesis
    (mirrors ``SessionAgent.reason_bidirectional``'s propose→validate→revise
    loop). It is registered ONLY when a graph is available (it is a no-op
    without one) and wrapped directly — ``agent_tools.TOOL_DEFINITIONS`` is
    deliberately untouched so the legacy ReasonerAgent/SessionAgent tool spec
    stays byte-identical (T6 invariant).
    """
    if graph is None:
        return None
    from drbrain.extractor.agent_tools import kg_validate

    async def _validate(**kwargs: Any) -> str:
        result = kg_validate(str(kwargs.get("hypothesis") or ""), db=db, graph=graph)
        return json.dumps(result, ensure_ascii=False, default=str)

    return FunctionTool.from_defaults(
        fn=_validate,
        name="kg_validate",
        description=(
            "Check a hypothesis or proposed answer against the knowledge graph "
            "for consistency (TBox/RBox violations and graph patterns such as "
            "debates and gaps). Call this after forming a conclusion to verify "
            "it is supported by the graph."
        ),
        fn_schema=_schema_to_model(
            "kg_validate",
            {
                "type": "object",
                "properties": {
                    "hypothesis": {
                        "type": "string",
                        "description": "The hypothesis text to validate against the KG",
                    }
                },
                "required": ["hypothesis"],
            },
        ),
    )


def _build_retrieval_tool(cfg: Config, db: Any, graph: Any) -> Any | None:
    """Optional fused-retrieval tool (``search_documents``) over LlamaIndex legs.

    Only registered when the LlamaIndex index/legs actually exist on disk —
    ``get_retrievers`` returns ``{}`` and ``build_fusion_retriever`` returns
    ``None`` when there is nothing to fuse, in which case this returns ``None``
    and the agent keeps the 8 graph tools. Any failure is swallowed (the tool
    is a bonus, never a blocker).
    """
    try:
        from llama_index.core.schema import QueryBundle

        from drbrain.rag.fusion import build_fusion_retriever, get_retrievers
    except Exception:
        return None
    try:
        legs = get_retrievers(cfg, db, graph)
        if not legs:
            return None
        fused = build_fusion_retriever(
            cfg,
            vector_index=legs.get("vector"),
            bm25_retriever=legs.get("bm25"),
            custom_retrievers={k: v for k, v in legs.items() if k not in ("bm25", "vector")},
        )
        if fused is None:
            return None
    except Exception as exc:  # pragma: no cover - depends on on-disk index state
        log.warning("[rag] retrieval tool unavailable (%s); keeping 8 graph tools", exc)
        return None

    async def _search(**kwargs: Any) -> str:
        query = kwargs.get("query", "")
        nodes = fused.retrieve(QueryBundle(query_str=query))
        rows = []
        for nws in nodes[:5]:
            md = dict(nws.node.metadata or {})
            rows.append(
                {
                    "paper_id": md.get("paper_id", ""),
                    "node_id": md.get("node_id", ""),
                    "title": md.get("title", ""),
                    "source": md.get("source", ""),
                    "score": round(float(nws.score), 4) if nws.score is not None else None,
                    "text": (nws.node.get_content() or "")[:500],
                }
            )
        return json.dumps(rows, ensure_ascii=False, default=str)

    return FunctionTool.from_defaults(
        fn=_search,
        name="search_documents",
        description=(
            "Search papers and sections via fused BM25 + vector retrieval over the "
            "LlamaIndex index (paper_id/node_id/title + section text)."
        ),
    )


# ── LLM glue: FunctionCallingLLM over DrbrainLLM ────────────────────────────


class AgentFunctionLLM(DrbrainLLM, FunctionCallingLLM):
    """``FunctionCallingLLM`` adapter over ``DrbrainLLM`` for the agent loop.

    ``DrbrainLLM`` (rag/llm.py) is T2-owned and deliberately not touched: it
    advertises ``is_function_calling_model=False`` and drops ``tools``. This
    subclass adds exactly the FunctionAgent contract — advertises function
    calling, forwards OpenAI-format tool specs to litellm via the same drbrain
    fallback chain (``llm_client.acall_with_messages``), and round-trips
    assistant ``tool_calls`` / tool ``tool_call_id`` messages — mirroring the
    installed ``llama-index-llms-litellm`` ``LiteLLM`` implementation.
    """

    def __init__(
        self,
        cfg: Config,
        temperature: float = AGENT_TEMPERATURE,
        max_tokens: int = AGENT_MAX_TOKENS,
        **kwargs: Any,
    ) -> None:
        # Explicit so mypy uses this signature instead of synthesizing one from
        # the (multiple-inheritance) pydantic base's fields.
        super().__init__(cfg, temperature=temperature, max_tokens=max_tokens, **kwargs)

    @property
    def metadata(self) -> Any:
        md = super().metadata
        md.is_function_calling_model = True
        return md

    def _prepare_chat_with_tools(
        self,
        tools: Sequence[BaseTool],
        user_msg: str | ChatMessage | None = None,
        chat_history: list[ChatMessage] | None = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        tool_required: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # Prefer the canonical drbrain OpenAI-format specs (identical to what
        # the legacy ReasonerAgent sent); fall back to the tool's own schema.
        tool_specs = [
            CANONICAL_TOOL_SPECS.get(tool.metadata.name or "")
            or tool.metadata.to_openai_tool(skip_length_check=True)
            for tool in tools
        ]
        if isinstance(user_msg, str):
            user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)
        messages = list(chat_history or [])
        if user_msg is not None:
            messages.append(user_msg)
        return {"messages": messages, "tools": tool_specs or None}

    def get_tool_calls_from_response(
        self,
        response: ChatResponse,
        error_on_no_tool_call: bool = True,
        **kwargs: Any,
    ) -> list[ToolSelection]:
        """Parse ``ChatResponse.message.additional_kwargs["tool_calls"]``."""
        tool_calls = response.message.additional_kwargs.get("tool_calls", [])
        if len(tool_calls) < 1:
            if error_on_no_tool_call:
                raise ValueError(f"Expected at least one tool call, but got {len(tool_calls)}.")
            return []
        selections: list[ToolSelection] = []
        for tool_call in tool_calls:
            if tool_call.get("type") != "function" or "function" not in tool_call:
                raise ValueError(f"Invalid tool call of type {tool_call.get('type')}")
            fn = tool_call.get("function", {})
            arguments = fn.get("arguments")
            try:
                argument_dict = json.loads(arguments) if arguments else {}
            except (ValueError, TypeError, json.JSONDecodeError):
                argument_dict = {}
            if fn.get("name"):
                selections.append(
                    ToolSelection(
                        tool_id=tool_call.get("id") or f"call_{len(selections)}",
                        tool_name=fn.get("name"),
                        tool_kwargs=argument_dict,
                    )
                )
        if not selections and error_on_no_tool_call:
            raise ValueError("No valid tool calls found.")
        return selections

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> Any:
        """Forward ``tools`` (OpenAI-format) to the drbrain fallback chain."""
        tools = kwargs.pop("tools", None)
        if not tools:
            return await super().achat(messages, **kwargs)
        from drbrain.extractor.llm_client import acall_with_messages

        result = await acall_with_messages(
            self._to_litellm_messages(messages),
            self._models,
            tools=tools,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            _cache=self._get_cache(),
        )
        return self._chat(result)

    @staticmethod
    def _to_litellm_messages(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
        """litellm dicts preserving tool protocol fields + block content."""
        out: list[dict[str, Any]] = []
        for msg in messages:
            role = getattr(msg.role, "value", str(msg.role))
            content = msg.content
            if content is None:  # tool-result messages carry ContentBlocks
                parts = []
                for block in getattr(msg, "blocks", None) or []:
                    if hasattr(block, "text"):
                        parts.append(block.text)
                content = "\n".join(parts)
            item: dict[str, Any] = {"role": role, "content": content or ""}
            ak = getattr(msg, "additional_kwargs", None) or {}
            if ak.get("tool_calls"):
                item["tool_calls"] = ak["tool_calls"]
            if ak.get("tool_call_id"):
                item["tool_call_id"] = ak["tool_call_id"]
            out.append(item)
        return out


# ── agent assembly ──────────────────────────────────────────────────────────


def _load_plugin_tools(plugins_dir: str | Path) -> list:
    """Discover plugins from ``plugins_dir`` and bridge them to LlamaIndex tools.

    Graceful by design: any discovery/bridge failure returns ``[]`` so the
    agent still assembles with the built-in graph tools. drbrain never imports
    a concrete plugin here — external plugins register themselves via
    ``PluginRegistry.discover``.
    """
    try:
        from drbrain.rag.plugins.registry import PluginRegistry
    except ImportError:
        return []
    try:
        registry = PluginRegistry()
        n = registry.discover(plugins_dir)
        tools = registry.to_llamaindex_tools()
        log.info("[rag] loaded %d plugin(s) from %s → %d tool(s)", n, plugins_dir, len(tools))
        return tools
    except Exception as exc:  # noqa: BLE001 — plugin failure must not break assembly
        log.warning("[rag] plugin discovery failed for %s: %s", plugins_dir, exc)
        return []


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
    """
    if not _LLAMA_INDEX_AVAILABLE:
        return None
    cfg = _coerce_cfg(cfg)
    papers_dir = _resolve_papers_dir(cfg, db)

    tools = [_make_graph_tool(name, db, graph, papers_dir) for name in GRAPH_TOOL_NAMES]
    # T9: kg_validate (KG consistency check) as the 8th tool, only with a graph.
    vt = _make_validate_tool(db, graph)
    if vt is not None:
        tools.append(vt)
    if include_retrieval:
        rt = _build_retrieval_tool(cfg, db, graph)
        if rt is not None:
            tools.append(rt)
    if plugins_dir:
        tools.extend(_load_plugin_tools(plugins_dir))

    system_prompt = BASE_SYSTEM_PROMPT
    if closure_context:
        system_prompt += (
            "\n\nInferred relations from logical closure "
            "(distinguished by --[inferred: ...]-->):\n" + closure_context
        )

    llm = AgentFunctionLLM(cfg, temperature=temperature, max_tokens=max_tokens)
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


def _history_summary(messages: list[dict]) -> str:
    """Plain-text summary of a message list for context compression.

    Mirrors ``SessionAgent._build_summary_text``: tool results are reduced to
    a char count, assistant tool-call messages to the tool names, everything
    else to a 200-char content preview.
    """
    parts = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content", "")
        if role == "tool":
            parts.append(f"[Tool result: {len(content)} chars]")
        elif role == "assistant" and m.get("tool_calls"):
            names = [tc.get("function", {}).get("name", "?") for tc in m["tool_calls"]]
            parts.append(f"Assistant called: {', '.join(names)}")
        elif content:
            parts.append(f"[{role}] {content[:200]}")
    return "\n".join(parts)


def load_session_history(
    db: Any,
    session_id: str,
    token_budget: int = SESSION_TOKEN_BUDGET,
) -> list[ChatMessage]:
    """Restore a session's prior turns as LlamaIndex ``ChatMessage``s (T9).

    Reads ``agent_messages`` (the shape both ``SessionAgent`` and
    ``_persist_reason_session`` write) in ``seq`` order and converts each row
    to a :class:`ChatMessage` for injection as ``agent.run(chat_history=...)``:

    * ``system`` rows are skipped — ``build_agent`` re-injects the same
      system prompt, so including the stored copy would duplicate it;
    * assistant tool-call rows keep ``additional_kwargs["tool_calls"]`` (the
      stored single dict is normalized to a list) and tool rows keep
      ``tool_call_id``/``name``, so the drbrain fallback chain receives a
      valid OpenAI tool loop;
    * long histories are compressed exactly like ``SessionAgent._maybe_compress``
      — once the estimated token count (``len(content)//4``) exceeds
      ``token_budget``, everything but the last :data:`SESSION_KEEP_RECENT`
      messages is collapsed into a leading ``[Context summary]`` system
      message.

    Returns ``[]`` for sessions with no (non-system) messages.
    """
    if db is None:
        return []
    rows = db.conn.execute(
        "SELECT role, content, tool_calls_json, tool_call_id, tool_name "
        "FROM agent_messages WHERE session_id = ? AND role != 'system' ORDER BY seq",
        (session_id,),
    ).fetchall()

    messages: list[dict[str, Any]] = []
    for r in rows:
        msg: dict[str, Any] = {"role": r[0], "content": r[1] or ""}
        if r[0] == "assistant" and r[2]:
            try:
                msg["tool_calls"] = json.loads(r[2])
            except (ValueError, TypeError):
                pass
        if r[0] == "tool" and r[3]:
            msg["tool_call_id"] = r[3]
            if r[4]:
                msg["name"] = r[4]
        messages.append(msg)

    if not messages:
        return []

    # SessionAgent._maybe_compress parity: keep the recent tail, summarize
    # the middle once the estimated size exceeds the budget.
    if len(messages) >= 8:
        estimated = sum(len(m.get("content", "")) // 4 for m in messages)
        if estimated >= token_budget:
            keep = min(SESSION_KEEP_RECENT, len(messages) - 3)
            recent = messages[-keep:] if keep > 0 else messages
            middle = messages[:-keep] if keep > 0 else []
            summary = _history_summary(middle)
            messages = [{"role": "system", "content": f"[Context summary]\n{summary}"}] + recent

    out: list[ChatMessage] = []
    for m in messages:
        role = m["role"]
        if role == "system":
            out.append(ChatMessage(role=MessageRole.SYSTEM, content=m["content"]))
        elif role == "user":
            out.append(ChatMessage(role=MessageRole.USER, content=m["content"]))
        elif role == "assistant":
            ak: dict[str, Any] = {}
            if m.get("tool_calls"):
                tc = m["tool_calls"]
                ak["tool_calls"] = tc if isinstance(tc, list) else [tc]
            out.append(
                ChatMessage(role=MessageRole.ASSISTANT, content=m["content"], additional_kwargs=ak)
            )
        elif role == "tool":
            ak = {"tool_call_id": m.get("tool_call_id", "")}
            if m.get("name"):
                ak["name"] = m["name"]
            out.append(
                ChatMessage(role=MessageRole.TOOL, content=m["content"], additional_kwargs=ak)
            )
    return out


def _persist_reason_session(
    cfg: Config,
    db: Any,
    session_id: str | None,
    question: str,
    system_prompt: str,
    answer: str,
    tool_calls: list[dict[str, Any]],
    models: list[dict],
) -> str | None:
    """Append one reasoning run to ``agent_sessions``/``agent_messages``.

    Write-only, best-effort, in a shape ``SessionAgent.load_session`` can read
    back (assistant messages carry ``tool_calls_json``, tool messages carry
    ``tool_call_id`` + ``tool_name``). ``"new"`` creates a session; an unknown
    existing id raises ``ValueError``. Read/restore + compression live in
    :func:`load_session_history` (T9).
    """
    if db is None or session_id is None:
        return None

    created = False
    if session_id == "new":
        session_id = "sess-" + uuid.uuid4().hex[:8]
        db.insert_agent_session(
            session_id,
            title="reason",
            system_prompt=system_prompt,
            model_config=json.dumps(models, ensure_ascii=False),
        )
        created = True
    else:
        row = db.conn.execute(
            "SELECT 1 FROM agent_sessions WHERE session_id = ? AND status != 'deleted'",
            (session_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Session not found: {session_id}")

    seq_row = db.conn.execute(
        "SELECT COALESCE(MAX(seq), -1) + 1 FROM agent_messages WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    seq = seq_row[0] if seq_row else 0

    if created:
        db.insert_agent_message(session_id, seq, "system", content=system_prompt)
        seq += 1
    db.insert_agent_message(session_id, seq, "user", content=question)
    seq += 1
    for tc in tool_calls:
        call_id = f"call_{seq}"
        tcall = {
            "id": call_id,
            "type": "function",
            "function": {
                "name": tc.get("name", ""),
                "arguments": json.dumps(tc.get("args") or {}, ensure_ascii=False),
            },
        }
        db.insert_agent_message(
            session_id, seq, "assistant", content="", tool_calls_json=json.dumps(tcall)
        )
        seq += 1
        db.insert_agent_message(
            session_id,
            seq,
            "tool",
            content=(tc.get("result_summary") or "")[:MAX_RESULT_SUMMARY_CHARS],
            tool_call_id=call_id,
            tool_name=tc.get("name", ""),
        )
        seq += 1
    db.insert_agent_message(session_id, seq, "assistant", content=answer or "")
    db.touch_session(session_id)
    db.commit()
    return session_id


# ── public entry point ──────────────────────────────────────────────────────


def reason_llamaindex(
    cfg: Config,
    db: Any = None,
    question: str = "",
    max_turns: int = 5,
    session_id: str | None = None,
    *,
    graph: Any = None,
    closure_context: str = "",
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
            )
        )
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
) -> dict[str, Any]:
    agent = build_agent(cfg, db, session_id, graph=graph, closure_context=closure_context)
    if agent is None:
        return {
            "answer": "LlamaIndex is not available (llamaindex.enabled=false or "
            "llama_index not installed). Use --engine legacy.",
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
        chat_history = load_session_history(db, session_id)

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
        )
        out["session_id"] = resolved_session_id

    # Bind the answer to its evidence (best-effort; never affects the result).
    _record_answer(cfg, db, question, answer, tool_calls, resolved_session_id)
    return out
