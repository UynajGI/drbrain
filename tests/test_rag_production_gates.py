"""Production gates for grounded answers, QA generation, and trust boundaries."""

from __future__ import annotations

import builtins
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from drbrain.config import Config, LlamaIndexConfig, load_config
from drbrain.rag.mcp_tools import MCPTrustError, call_mcp_tool, discover_mcp_tools, load_mcp_tools
from drbrain.storage.database import Database

_HAS_LLAMA_INDEX = importlib.util.find_spec("llama_index") is not None
_HAS_MCP = importlib.util.find_spec("mcp") is not None
_ECHO_SERVER = Path(__file__).parent / "fixtures" / "mcp_echo_server.py"


def _cfg() -> Config:
    cfg = Config(llamaindex=LlamaIndexConfig(enabled=True))
    cfg.llm.models = [{"provider": "openai", "model": "test", "api_key": "test"}]
    cfg.llamaindex.storage_dir = "/nonexistent-rag-index"
    return cfg


def _trusted_echo_server(**overrides: object) -> dict[str, object]:
    server: dict[str, object] = {
        "id": "test-echo",
        "trusted": True,
        "command": sys.executable,
        "args": [str(_ECHO_SERVER)],
        "allowed_tools": ["echo"],
        "timeout_seconds": 5,
    }
    server.update(overrides)
    return server


@pytest.mark.skipif(not _HAS_LLAMA_INDEX, reason="llama_index not installed")
def test_query_engine_refuses_sources_without_stable_evidence_id(monkeypatch):
    from drbrain.rag import engine as engine_module

    class _Engine:
        def query(self, question):
            node = SimpleNamespace(metadata={}, node_id="")
            return SimpleNamespace(
                response="unsupported answer", source_nodes=[SimpleNamespace(node=node, score=1)]
            )

    monkeypatch.setattr(engine_module, "build_query_engine", lambda *args, **kwargs: _Engine())

    result = engine_module.ask_llamaindex(_cfg(), None, "What happened?", streaming=False)

    assert result["status"] == "insufficient_evidence"
    assert result["evidence_ids"] == []
    assert result["sources"] == []


@pytest.mark.skipif(not _HAS_LLAMA_INDEX, reason="llama_index not installed")
def test_query_engine_returns_retrieved_evidence_ids(monkeypatch):
    from drbrain.rag import engine as engine_module

    class _Engine:
        def query(self, question):
            node = SimpleNamespace(
                metadata={"paper_id": "paper-1", "node_id": "section-2", "title": "Methods"},
                node_id="paper-1:section-2",
            )
            return SimpleNamespace(
                response="grounded answer", source_nodes=[SimpleNamespace(node=node, score=1)]
            )

    monkeypatch.setattr(engine_module, "build_query_engine", lambda *args, **kwargs: _Engine())

    result = engine_module.ask_llamaindex(_cfg(), None, "What happened?", streaming=False)

    assert result["answer"] == "grounded answer"
    assert result["evidence_ids"] == ["paper-1:section-2"]


@pytest.mark.skipif(not _HAS_LLAMA_INDEX, reason="llama_index not installed")
def test_agent_refuses_final_answer_without_allowed_retrieval_evidence(monkeypatch):
    from drbrain.rag import agent as agent_module

    class _Store:
        async def get(self, key, default=None):
            return 1

    class _Handler:
        ctx = SimpleNamespace(store=_Store())

        def __await__(self):
            async def _result():
                return SimpleNamespace(
                    response=SimpleNamespace(content="unguarded answer"), tool_calls=[]
                )

            return _result().__await__()

    class _Agent:
        def run(self, **kwargs):
            return _Handler()

    monkeypatch.setattr(agent_module, "build_agent", lambda *args, **kwargs: _Agent())

    result = agent_module.reason_llamaindex(_cfg(), None, "Answer directly")

    assert result["status"] == "insufficient_evidence"
    assert result["evidence_ids"] == []
    assert result["tool_calls"] == []


@pytest.mark.skipif(not _HAS_LLAMA_INDEX, reason="llama_index not installed")
def test_agent_returns_answer_when_allowed_retrieval_tool_supplies_evidence(monkeypatch):
    from drbrain.rag import agent as agent_module

    class _Store:
        async def get(self, key, default=None):
            return 1

    class _Handler:
        ctx = SimpleNamespace(store=_Store())

        def __await__(self):
            async def _result():
                tool_output = SimpleNamespace(
                    content=json.dumps([{"paper_id": "paper-1", "node_id": "node-3"}]),
                    is_error=False,
                )
                tool_call = SimpleNamespace(
                    tool_name="search_documents",
                    tool_kwargs={"query": "evidence"},
                    tool_output=tool_output,
                )
                return SimpleNamespace(
                    response=SimpleNamespace(content="grounded answer"), tool_calls=[tool_call]
                )

            return _result().__await__()

    class _Agent:
        def run(self, **kwargs):
            return _Handler()

    monkeypatch.setattr(agent_module, "build_agent", lambda *args, **kwargs: _Agent())

    result = agent_module.reason_llamaindex(_cfg(), None, "Answer with evidence")

    assert result["answer"] == "grounded answer"
    assert result["evidence_ids"] == ["paper-1:node-3"]


@pytest.mark.skipif(not _HAS_LLAMA_INDEX, reason="llama_index not installed")
def test_session_principal_is_persisted_and_required_for_resume(monkeypatch, tmp_path):
    from drbrain.rag import agent as agent_module

    class _Store:
        async def get(self, key, default=None):
            return 1

    class _Handler:
        ctx = SimpleNamespace(store=_Store())

        def __await__(self):
            async def _result():
                return SimpleNamespace(response=SimpleNamespace(content="answer"), tool_calls=[])

            return _result().__await__()

    class _Agent:
        def run(self, **kwargs):
            return _Handler()

    monkeypatch.setattr(agent_module, "build_agent", lambda *args, **kwargs: _Agent())
    db = Database(str(tmp_path / "db.sqlite"))
    try:
        created = agent_module.reason_llamaindex(
            _cfg(), db, "first", session_id="new", principal="alice"
        )
        session_id = created["session_id"]
        owner = db.conn.execute(
            "SELECT owner_principal FROM agent_sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        assert owner == ("alice",)

        denied = agent_module.reason_llamaindex(
            _cfg(), db, "second", session_id=session_id, principal="bob"
        )
        assert denied["status"] == "permission_denied"
        with pytest.raises(PermissionError):
            agent_module.load_session_history(db, session_id, principal="bob")
    finally:
        db.close()


@pytest.mark.skipif(not _HAS_LLAMA_INDEX, reason="llama_index not installed")
def test_session_rejects_an_empty_authenticated_principal():
    from drbrain.rag import agent as agent_module

    result = agent_module.reason_llamaindex(_cfg(), None, "question", principal=" ")

    assert result["status"] == "permission_denied"
    assert result["evidence_ids"] == []


def test_config_loader_opens_base_and_overlay_as_utf8(monkeypatch, tmp_path):
    base = tmp_path / "config.yaml"
    local = tmp_path / "config.local.yaml"
    base.write_text("dirs:\n  reports: 报告\n", encoding="utf-8")
    local.write_text("api:\n  crossref_email: 用户@example.test\n", encoding="utf-8")

    seen_encodings: list[str | None] = []
    real_open = builtins.open

    def capture_open(*args, **kwargs):
        seen_encodings.append(kwargs.get("encoding"))
        return real_open(*args, **kwargs)

    monkeypatch.setattr(builtins, "open", capture_open)
    config = load_config(base, local)

    assert config.dirs.reports == "报告"
    assert seen_encodings == ["utf-8", "utf-8"]


def test_qagen_openai_integration_is_declared_and_importable():
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    assert '"llama-index-llms-openai' in pyproject.read_text(encoding="utf-8")
    assert importlib.util.find_spec("llama_index.llms.openai") is not None


@pytest.mark.skipif(not _HAS_LLAMA_INDEX, reason="llama_index not installed")
def test_qagen_reports_a_stable_missing_openai_dependency(monkeypatch, tmp_path):
    from drbrain.rag import indexer as indexer_module
    from drbrain.rag.eval import run_qagen

    fake_index = SimpleNamespace(
        docstore=SimpleNamespace(docs={"node-1": SimpleNamespace(node_id="node-1")})
    )
    monkeypatch.setattr(indexer_module, "load_index", lambda cfg: (fake_index, None))
    real_import = builtins.__import__

    def deny_openai(name, *args, **kwargs):
        if name == "llama_index.llms.openai":
            raise ImportError("simulated missing qagen integration")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", deny_openai)
    cfg = _cfg()
    cfg.llamaindex.eval.golden_set = str(tmp_path / "golden.jsonl")

    result = run_qagen(cfg, n_nodes=1)

    assert result == {
        "status": "unavailable",
        "reason": "qagen dependency missing: install llama-index-llms-openai",
    }


@pytest.mark.skipif(not _HAS_MCP, reason="mcp SDK not installed")
def test_agent_mcp_bridge_requires_trusted_server_and_tool_allowlist():
    untrusted = {"command": sys.executable, "args": [str(_ECHO_SERVER)]}
    assert load_mcp_tools([untrusted], require_trusted=True) == []

    trusted = _trusted_echo_server()
    assert [tool["name"] for tool in discover_mcp_tools(trusted, require_trusted=True)] == ["echo"]
    assert call_mcp_tool(trusted, "echo", {"text": "safe"}, require_trusted=True) == "echo: safe"

    with pytest.raises(MCPTrustError):
        call_mcp_tool(trusted, "other", {}, require_trusted=True)


@pytest.mark.skipif(not _HAS_MCP, reason="mcp SDK not installed")
def test_mcp_trust_policy_rejects_unbounded_or_invalid_timeout():
    with pytest.raises(MCPTrustError):
        discover_mcp_tools(_trusted_echo_server(timeout_seconds=0), require_trusted=True)
