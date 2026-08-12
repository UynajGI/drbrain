"""T6 tests: ``rag/agent.py`` — LlamaIndex FunctionAgent + reason_llamaindex.

Covers :func:`~drbrain.rag.agent.build_agent` (FunctionAgent assembly, 7
drbrain graph tools registered as FunctionTools over ``execute_tool``,
closure-context system prompt, ``None`` fallback without llama-index),
:func:`~drbrain.rag.agent.reason_llamaindex` (output dict shape, tool
trajectory, session persistence, graceful fallback) and the CLI ``reason
--engine`` routing/fallback.

Mocked unit tests are offline (the LLM fallback chain is stubbed via
``llm_client.acall_with_messages``); the one live-call test is marked
``integration`` (opencode test key from ``test-run/``).
"""

import importlib.util
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from drbrain.config import Config, LlamaIndexConfig

_HAS_LLAMA_INDEX = importlib.util.find_spec("llama_index") is not None

if _HAS_LLAMA_INDEX:
    from llama_index.core.base.llms.types import ChatMessage, MessageRole
    from llama_index.core.tools import FunctionTool

    from drbrain.rag.agent import AgentFunctionLLM, build_agent, reason_llamaindex

pytestmark = pytest.mark.skipif(not _HAS_LLAMA_INDEX, reason="llama_index not installed")

from drbrain.extractor.agent_tools import TOOL_DEFINITIONS  # noqa: E402
from drbrain.storage.database import Database  # noqa: E402

MODELS = [{"provider": "openai", "model": "gpt-4o", "api_key": "k", "base_url": None}]

runner = CliRunner()


def _cfg(tmp_path=None, enabled=True) -> Config:
    c = Config(llamaindex=LlamaIndexConfig(enabled=enabled))
    c.llm.models = list(MODELS)
    c.api.cache_ttl = 0
    if tmp_path is not None:
        c.dirs.cache = str(tmp_path)
        c.dirs.papers = str(tmp_path)
        c.llamaindex.storage_dir = str(tmp_path / "li")
    else:
        # Never resolve to a real on-disk index (the repo's data/llamaindex
        # must not leak into unit tests via the default storage_dir).
        c.llamaindex.storage_dir = "/nonexistent-rag-index"
    return c


def _scripted_llm(monkeypatch, script, capture=None):
    """Stub ``llm_client.acall_with_messages`` with a script of results.

    ``script`` is a list of ``{"text": str, "tool_calls": list|None}`` steps;
    the last step repeats if the loop needs more calls. When ``capture`` is a
    list, each call's litellm messages are appended to it.
    """

    async def fake(messages, models, tools=None, max_tokens=1024, temperature=0.3, **kw):
        if capture is not None:
            capture.append(messages)
        step = script[min(fake.calls, len(script) - 1)]
        fake.calls += 1
        return dict(step)

    fake.calls = 0
    monkeypatch.setattr("drbrain.extractor.llm_client.acall_with_messages", fake)
    return fake


_TOOL_CALL_SEARCH = [
    {
        "id": "call_1",
        "type": "function",
        "function": {"name": "search_concepts", "arguments": '{"query": "perovskite"}'},
    }
]


# ── build_agent ──────────────────────────────────────────────────────────────


def test_build_agent_assembles_graph_tools():
    agent = build_agent(_cfg(), db=None, graph=None)
    assert agent is not None
    names = [t.metadata.name for t in agent.tools]
    expected = [fn["name"] for fn in (d["function"] for d in TOOL_DEFINITIONS)]
    assert names == expected, f"agent tools {names} != TOOL_DEFINITIONS {expected}"
    # every graph tool is a FunctionTool whose execution body is execute_tool
    for tool in agent.tools:
        assert tool.metadata.description
        schema = tool.metadata.fn_schema.model_json_schema()
        assert schema.get("type") == "object"


def test_build_agent_llm_is_function_calling_glue():
    agent = build_agent(_cfg(), db=None, graph=None)
    assert isinstance(agent.llm, AgentFunctionLLM)
    assert agent.llm.metadata.is_function_calling_model is True
    assert agent.llm.temperature == 0.3  # legacy agent-loop temperature
    assert agent.llm.max_tokens == 1024  # legacy agent-loop cap


def test_build_agent_system_prompt_keeps_closure_context():
    agent = build_agent(
        _cfg(),
        db=None,
        graph=None,
        closure_context="A --[inferred: subsumes]--> B",
    )
    assert "Inferred relations from logical closure" in agent.system_prompt
    assert "A --[inferred: subsumes]--> B" in agent.system_prompt


def test_build_agent_without_llamaindex_returns_none(monkeypatch):
    import drbrain.rag.agent as ra

    monkeypatch.setattr(ra, "_LLAMA_INDEX_AVAILABLE", False)
    assert build_agent(_cfg(), db=None) is None


def test_build_agent_extra_retrieval_tool_absent_without_index():
    """With no index on disk, get_retrievers has no legs → still 7 tools."""
    agent = build_agent(_cfg(), db=None, graph=None)
    assert [t.metadata.name for t in agent.tools] == [
        fn["name"] for fn in (d["function"] for d in TOOL_DEFINITIONS)
    ]


# ── T9: kg_validate as the 8th graph tool ───────────────────────────────────


def test_build_agent_with_graph_adds_kg_validate_tool():
    """graph present → kg_validate joins the 7 graph tools (T9 decision)."""
    agent = build_agent(_cfg(), db=None, graph=object())  # any truthy graph
    names = [t.metadata.name for t in agent.tools]
    assert "kg_validate" in names
    assert len(names) == 8
    tool = next(t for t in agent.tools if t.metadata.name == "kg_validate")
    schema = tool.metadata.fn_schema.model_json_schema()
    assert schema["type"] == "object"
    assert "hypothesis" in schema["properties"]


async def test_kg_validate_tool_executes_through_agent_tools(monkeypatch, tmp_path):
    """The wrapper executes the real kg_validate (mocked graph, empty result)."""
    from drbrain.extractor import agent_tools as at

    captured = {}

    def fake_kg_validate(hypothesis, db=None, graph=None):
        captured["hypothesis"] = hypothesis
        return {"consistent": True, "violations": [], "patterns": []}

    monkeypatch.setattr(at, "kg_validate", fake_kg_validate)
    agent = build_agent(_cfg(tmp_path), db=None, graph=object())
    tool = next(t for t in agent.tools if t.metadata.name == "kg_validate")
    out = await tool.acall(hypothesis="Perovskite solar cells dominate the market.")
    assert json.loads(out.content) == {"consistent": True, "violations": [], "patterns": []}
    assert captured["hypothesis"] == "Perovskite solar cells dominate the market."


# ── tool execution via execute_tool ─────────────────────────────────────────


async def test_graph_tool_executes_through_execute_tool():
    agent = build_agent(_cfg(), db=None, graph=None)
    by_name = {t.metadata.name: t for t in agent.tools}
    # search_concepts with db=None → execute_tool handler returns []
    out = await by_name["search_concepts"].acall(query="nothing")
    assert json.loads(out.content) == []
    # missing required arg is tolerated by the wrapper (handler applies defaults)
    out2 = await by_name["get_neighbors"].acall(node="nonexistent")
    assert isinstance(json.loads(out2.content), list)


# ── AgentFunctionLLM glue ───────────────────────────────────────────────────


async def test_agent_llm_forwards_tools_in_canonical_openai_format(monkeypatch):
    captured: dict = {}

    async def fake(messages, models, tools=None, max_tokens=1024, temperature=0.3, **kw):
        captured["tools"] = tools
        captured["temperature"] = temperature
        return {"text": "hi", "tool_calls": None, "usage": None}

    monkeypatch.setattr("drbrain.extractor.llm_client.acall_with_messages", fake)
    llm = AgentFunctionLLM(_cfg(), temperature=0.3, max_tokens=256)

    async def _t(**kwargs):  # noqa: ANN202
        return "[]"

    tool = FunctionTool.from_defaults(fn=_t, name="search_concepts", description="d")
    resp = await llm.achat_with_tools(
        [tool],
        chat_history=[ChatMessage(role=MessageRole.USER, content="q")],
    )
    # canonical TOOL_DEFINITIONS spec is used (exact legacy schema with params)
    spec = captured["tools"][0]
    assert spec["function"]["name"] == "search_concepts"
    assert "parameters" in spec["function"]
    assert spec["function"]["parameters"]["required"] == ["query"]
    assert captured["temperature"] == 0.3
    assert resp.message.content == "hi"


# ── reason_llamaindex ───────────────────────────────────────────────────────


def test_reason_llamaindex_output_structure(monkeypatch):
    captured: list = []
    fake = _scripted_llm(
        monkeypatch,
        [
            {"text": "", "tool_calls": _TOOL_CALL_SEARCH, "usage": None},
            {"text": "Perovskite solar cells dominate.", "tool_calls": None, "usage": None},
        ],
        capture=captured,
    )
    result = reason_llamaindex(_cfg(), None, "What is perovskite?", max_turns=5)

    assert set(result) == {"answer", "tool_calls", "turns", "engine"}
    assert result["engine"] == "llamaindex"
    assert result["answer"] == "Perovskite solar cells dominate."
    assert result["turns"] == 2  # one tool step + final answer
    assert fake.calls == 2

    assert len(result["tool_calls"]) == 1
    tc = result["tool_calls"][0]
    assert set(tc) == {"name", "args", "result_summary"}
    assert tc["name"] == "search_concepts"
    assert tc["args"] == {"query": "perovskite"}
    assert isinstance(tc["result_summary"], str)

    # second LLM call round-trips assistant tool_calls + tool result messages
    second = captured[1]
    roles = [m["role"] for m in second]
    assert "tool" in roles
    tool_msg = next(m for m in second if m["role"] == "tool")
    assert tool_msg.get("tool_call_id") == "call_1"


def test_reason_llamaindex_direct_answer_no_tools(monkeypatch):
    fake = _scripted_llm(
        monkeypatch,
        [{"text": "Direct answer.", "tool_calls": None, "usage": None}],
    )
    result = reason_llamaindex(_cfg(), None, "hi", max_turns=5)
    assert result["answer"] == "Direct answer."
    assert result["tool_calls"] == []
    assert result["turns"] == 1
    assert fake.calls == 1


def test_reason_llamaindex_error_result_when_llamaindex_unavailable(monkeypatch):
    import drbrain.rag.agent as ra

    monkeypatch.setattr(ra, "_LLAMA_INDEX_AVAILABLE", False)
    result = reason_llamaindex(_cfg(), None, "hi", max_turns=5)
    assert result["engine"] == "llamaindex"
    assert result["tool_calls"] == []
    assert "llamaindex" in result["answer"]  # explains the fallback


def test_reason_llamaindex_session_new_persists_to_agent_tables(tmp_path, monkeypatch):
    _scripted_llm(
        monkeypatch,
        [
            {"text": "", "tool_calls": _TOOL_CALL_SEARCH, "usage": None},
            {"text": "Answer with tools.", "tool_calls": None, "usage": None},
        ],
    )
    db = Database(str(tmp_path / "t.db"))
    try:
        result = reason_llamaindex(_cfg(tmp_path), db, "question?", max_turns=5, session_id="new")
    finally:
        db.close()
    sid = result["session_id"]
    assert sid and sid.startswith("sess-")

    db = Database(str(tmp_path / "t.db"))
    try:
        rows = db.conn.execute(
            "SELECT role, content, tool_calls_json, tool_call_id, tool_name "
            "FROM agent_messages WHERE session_id = ? ORDER BY seq",
            (sid,),
        ).fetchall()
    finally:
        db.close()
    roles = [r[0] for r in rows]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert rows[1][1] == "question?"  # user question
    assert json.loads(rows[2][2])["function"]["name"] == "search_concepts"  # tool_call
    assert json.loads(rows[2][2])["id"] == rows[3][3]  # tool msg links the call id
    assert rows[3][4] == "search_concepts"
    assert rows[4][1] == "Answer with tools."


def test_reason_llamaindex_session_not_found(tmp_path, monkeypatch):
    _scripted_llm(monkeypatch, [{"text": "x", "tool_calls": None, "usage": None}])
    db = Database(str(tmp_path / "t.db"))
    try:
        result = reason_llamaindex(_cfg(tmp_path), db, "q", max_turns=5, session_id="sess-missing")
    finally:
        db.close()
    assert "Session not found" in result["answer"]
    assert result["tool_calls"] == []


# ── T9 session read recovery ─────────────────────────────────────────────────


def test_load_session_history_reads_back_written_messages(tmp_path):
    """write → read → restored message count matches the non-system rows."""
    from drbrain.rag.agent import _persist_reason_session, load_session_history

    db = Database(str(tmp_path / "t.db"))
    sid = None
    try:
        sid = _persist_reason_session(
            _cfg(tmp_path),
            db,
            "new",
            question="What is perovskite?",
            system_prompt="sys",
            answer="Perovskite solar cells dominate.",
            tool_calls=[
                {
                    "name": "search_concepts",
                    "args": {"query": "perovskite"},
                    "result_summary": "[ok]",
                }
            ],
            models=MODELS,
        )
        assert sid and sid.startswith("sess-")

        history = load_session_history(db, sid)
        # rows written: user, assistant(tool_calls), tool, assistant → 4
        assert len(history) == 4
        roles = [m.role.value for m in history]
        assert roles == ["user", "assistant", "tool", "assistant"]
        assert history[0].content == "What is perovskite?"
        # assistant tool-call message carries the OpenAI-format tool_calls
        tc = history[1].additional_kwargs["tool_calls"]
        assert isinstance(tc, list)
        assert tc[0]["function"]["name"] == "search_concepts"
        # tool message links the call id and carries the tool name
        assert history[2].additional_kwargs["tool_call_id"] == tc[0]["id"]
        assert history[2].additional_kwargs["name"] == "search_concepts"
        assert history[3].content == "Perovskite solar cells dominate."
    finally:
        db.close()


def test_load_session_history_skips_system_row(tmp_path):
    """The stored system message is not re-injected (build_agent re-injects)."""
    from drbrain.rag.agent import _persist_reason_session, load_session_history

    db = Database(str(tmp_path / "t.db"))
    try:
        sid = _persist_reason_session(
            _cfg(tmp_path),
            db,
            "new",
            question="q",
            system_prompt="sys",
            answer="a",
            tool_calls=[],
            models=MODELS,
        )
        history = load_session_history(db, sid)
        assert [m.role.value for m in history] == ["user", "assistant"]
        assert all(m.role.value != "system" for m in history)
    finally:
        db.close()


def test_load_session_history_empty_for_unknown_session(tmp_path):
    from drbrain.rag.agent import load_session_history

    db = Database(str(tmp_path / "t.db"))
    try:
        assert load_session_history(db, "sess-nope") == []
    finally:
        db.close()


def test_load_session_history_compresses_long_history(tmp_path):
    """Long histories collapse the middle into a [Context summary] message."""
    from drbrain.rag.agent import _persist_reason_session, load_session_history

    db = Database(str(tmp_path / "t.db"))
    try:
        sid = _persist_reason_session(
            _cfg(tmp_path),
            db,
            "new",
            question="q0",
            system_prompt="sys",
            answer="a0",
            tool_calls=[],
            models=MODELS,
        )
        # append 12 more user/assistant turns of bulk content
        seq = db.conn.execute(
            "SELECT COALESCE(MAX(seq), -1) + 1 FROM agent_messages WHERE session_id = ?",
            (sid,),
        ).fetchone()[0]
        for i in range(12):
            db.insert_agent_message(sid, seq, "user", content="filler " * 2000)
            seq += 1
            db.insert_agent_message(sid, seq, "assistant", content="filler " * 2000)
            seq += 1
        db.commit()

        history = load_session_history(db, sid, token_budget=2000)
        assert history, "history must be restored"
        # compressed: leading [Context summary] system message + recent tail
        assert history[0].role.value == "system"
        assert history[0].content.startswith("[Context summary]")
        # tail kept verbatim: last assistant turn present
        assert history[-1].content.startswith("filler")
    finally:
        db.close()


def test_reason_llamaindex_existing_session_injects_history(monkeypatch, tmp_path):
    """A resumed session passes the restored history as chat_history."""
    import drbrain.rag.agent as ra
    from drbrain.rag.agent import _persist_reason_session

    captured: dict = {}

    # build_agent is fully mocked at the FunctionAgent level via acall_with_messages;
    # to observe chat_history we wrap agent.run's call through a patched run.
    db = Database(str(tmp_path / "t.db"))
    sid = None
    try:
        sid = _persist_reason_session(
            _cfg(tmp_path),
            db,
            "new",
            question="old q",
            system_prompt="sys",
            answer="old a",
            tool_calls=[],
            models=MODELS,
        )

        # Patch agent.run to record chat_history, then delegate to a real loop
        # stub: the LLM is scripted so the loop terminates immediately.
        _scripted_llm(monkeypatch, [{"text": "new a", "tool_calls": None, "usage": None}])

        real_build = ra.build_agent

        class _Recorder:
            def __init__(self, inner):
                self._inner = inner

            async def run(self, *args, **kwargs):
                captured["chat_history"] = kwargs.get("chat_history")
                captured["user_msg"] = kwargs.get("user_msg")
                captured["max_iterations"] = kwargs.get("max_iterations")
                return await self._inner.run(*args, **kwargs)

        def _wrapped(*a, **kw):
            inner = real_build(*a, **kw)
            if inner is None:
                return None
            recorder = _Recorder(inner)
            return recorder

        monkeypatch.setattr(ra, "build_agent", _wrapped)

        result = reason_llamaindex(_cfg(tmp_path), db, "new q", max_turns=3, session_id=sid)
        assert result["answer"] == "new a"
        assert captured["user_msg"] == "new q"
        assert captured["max_iterations"] == 3
        history = captured["chat_history"] or []
        # restored: old user + old assistant (system row skipped)
        assert [m.role.value for m in history] == ["user", "assistant"]
        assert history[0].content == "old q"
        assert history[1].content == "old a"
    finally:
        db.close()


# ── CLI: reason --engine routing ────────────────────────────────────────────


def _cli_cfg(tmp_path: Path, enabled: bool) -> dict:
    return {
        "db": {"path": str(tmp_path / "t.db")},
        "llm": {"models": [{"provider": "openai", "model": "gpt-4o", "api_key": "x"}]},
        "dirs": {"papers": str(tmp_path), "cache": str(tmp_path), "reports": str(tmp_path)},
        "api": {"cache_ttl": 0},
        "llamaindex": {"enabled": enabled},
    }


def _cli_ctx(cfg: dict):
    from unittest import mock

    ctx = mock.MagicMock(spec=__import__("typer").Context)
    ctx.obj = {"config": cfg}
    return ctx


def _seed_cli_db(tmp_path: Path) -> None:
    db = Database(str(tmp_path / "t.db"))
    db.insert_paper("p1", "Test Paper", 2024, "uploaded")
    db.insert_concept("p1", "Problem", "overfitting", 0.9, year=2024)
    db.insert_concept("p1", "Method", "regularization", 0.85, year=2024)
    db.insert_edge("overfitting", "regularization", "challenges", "p1", 1.0)
    db.commit()
    db.close()


def test_reason_cmd_routes_to_reason_llamaindex(monkeypatch, tmp_path):
    from drbrain.cli.analysis_commands import reason_cmd

    _seed_cli_db(tmp_path)
    captured: dict = {}

    def fake_reason(cfg_, db_, question, max_turns=5, session_id=None, **kw):
        captured["question"] = question
        captured["max_turns"] = max_turns
        captured["session_id"] = session_id
        return {
            "answer": "rag answer",
            "tool_calls": [
                {"name": "search_concepts", "args": {"query": "x"}, "result_summary": "ok"}
            ],
            "turns": 2,
            "engine": "llamaindex",
        }

    monkeypatch.setattr("drbrain.rag.agent.reason_llamaindex", fake_reason)
    reason_cmd(_cli_ctx(_cli_cfg(tmp_path, enabled=True)), "some question")

    assert captured["question"] == "some question"
    assert captured["max_turns"] == 5
    assert captured["session_id"] is None


def test_reason_cmd_json_output(monkeypatch, tmp_path, capsys):
    from drbrain.cli.analysis_commands import reason_cmd

    _seed_cli_db(tmp_path)
    monkeypatch.setattr(
        "drbrain.rag.agent.reason_llamaindex",
        lambda *a, **k: {"answer": "a", "tool_calls": [], "turns": 1, "engine": "llamaindex"},
    )
    reason_cmd(_cli_ctx(_cli_cfg(tmp_path, enabled=True)), "q", json_output=True)
    out = capsys.readouterr().out
    payload = json.loads(out[out.index("{") :])  # JSON follows the header line
    assert payload["answer"] == "a"
    assert payload["engine"] == "llamaindex"


def test_reason_cmd_disabled_exits_with_warning(monkeypatch, tmp_path):
    """T9: llamaindex disabled → warning + exit 1 (no legacy fallback)."""
    import typer

    from drbrain.cli.analysis_commands import reason_cmd

    _seed_cli_db(tmp_path)
    with pytest.raises(typer.Exit) as exc:
        reason_cmd(_cli_ctx(_cli_cfg(tmp_path, enabled=False)), "q")
    assert exc.value.exit_code == 1


def test_reason_cmd_help_has_no_engine_option():
    import re

    import typer

    from drbrain.cli.analysis_commands import reason_cmd

    app = typer.Typer()

    @app.callback()
    def _cb(ctx: typer.Context):
        ctx.obj = {"config": {"db": {"path": ":memory:"}}}

    app.command("test")(reason_cmd)
    r = runner.invoke(app, ["test", "--help"])
    assert r.exit_code == 0
    # rich may emit ANSI color codes (e.g. FORCE_COLOR in CI) that split option
    # names across style spans; strip them before substring assertions.
    output = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", r.output)
    assert "--engine" not in output  # T9: legacy engine switch removed
    assert "--json" in output


# ── integration: real LLM over the test-run corpus ─────────────────────────


@pytest.mark.integration
def test_integration_reason_llamaindex_real_llm(tmp_path):
    """Real opencode.ai key + test-run corpus: answer and tool calls non-empty.

    Uses the test-run config's ``deepseek-v4-flash`` key (never hardcoded).
    Skipped when that config is absent.
    """
    from drbrain.graph.engine import GraphEngine

    test_run = Path(__file__).resolve().parents[1] / "test-run"
    cfg_path = test_run / "config.yaml"
    if not cfg_path.exists():
        pytest.skip("test-run/config.yaml (opencode test key) not present")
    cfg = Config.from_yaml(str(cfg_path), local_path=cfg_path.parent / "config.local.yaml")
    assert cfg.llm.models, "test-run config must define llm.models"
    assert "opencode" in (cfg.llm.models[0].get("base_url") or ""), (
        "expected opencode.ai test key as models[0]"
    )
    cfg.llamaindex.enabled = True
    cfg.dirs.cache = str(tmp_path)
    cfg.dirs.papers = str(test_run / "papers")

    db = Database(str(test_run / "db" / "drbrain.db"))
    graph = GraphEngine()
    graph.load_from_db(db)
    try:
        result = reason_llamaindex(
            cfg,
            db,
            "Use the search_concepts tool to find concepts related to perovskite "
            "or solar cells, then summarize what the knowledge graph says about them.",
            max_turns=5,
            graph=graph,
        )
    finally:
        db.close()

    assert result["engine"] == "llamaindex"
    assert result["answer"].strip(), "live agent returned an empty answer"
    assert result["tool_calls"], "live agent made no tool calls"
    assert result["turns"] >= 1
