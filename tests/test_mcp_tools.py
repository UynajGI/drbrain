"""Generic MCP bridge tests — connect/discover/call any stdio MCP server."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, get_args

import pytest

from drbrain.rag.mcp_tools import call_mcp_tool, discover_mcp_tools

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("mcp") is None,
    reason="mcp SDK not installed",
)

_ECHO_SERVER = Path(__file__).parent / "fixtures" / "mcp_echo_server.py"


def _server() -> dict:
    return {"command": sys.executable, "args": [str(_ECHO_SERVER)]}


def test_discover_mcp_tools():
    tools = discover_mcp_tools(_server())
    assert [t["name"] for t in tools] == ["echo"]
    assert tools[0]["description"] == "Echo back the text argument"
    assert tools[0]["inputSchema"]["required"] == ["text"]


def test_call_mcp_tool():
    result = call_mcp_tool(_server(), "echo", {"text": "hello"})
    assert result == "echo: hello"


def test_load_mcp_tools_bridges_and_calls():
    importlib.util.find_spec("llama_index") or pytest.skip("llama_index not installed")
    from drbrain.rag.mcp_tools import load_mcp_tools

    tools = load_mcp_tools([_server()])
    assert [t.metadata.name for t in tools] == ["echo"]
    # the bridged tool fn calls through to the MCP server (sync call, no async loop)
    assert tools[0].fn(text="hi") == "echo: hi"


def test_build_agent_loads_mcp_tools():
    importlib.util.find_spec("llama_index") or pytest.skip("llama_index not installed")
    from drbrain.config import Config, LlamaIndexConfig
    from drbrain.rag.agent import build_agent

    cfg = Config(llamaindex=LlamaIndexConfig(enabled=True))
    cfg.llm.models = [{"provider": "openai", "model": "gpt-4o", "api_key": "k", "base_url": None}]
    cfg.api.cache_ttl = 0
    cfg.llamaindex.storage_dir = "/nonexistent-rag-index"

    agent = build_agent(cfg, db=None, graph=None, mcp_servers=[_server()])
    assert agent is not None
    names = {t.metadata.name for t in agent.tools}
    assert "echo" in names  # MCP tool joined the agent's tool surface


def test_load_mcp_tools_error_paths():
    """Graceful degradation: no servers → []; one bad server skipped, rest bridged."""
    importlib.util.find_spec("llama_index") or pytest.skip("llama_index not installed")
    from drbrain.rag.mcp_tools import load_mcp_tools

    assert load_mcp_tools(None) == []
    assert load_mcp_tools([]) == []
    # a server whose command does not exist fails discovery → skipped without raising
    bad = {"command": "/nonexistent/bin/mcp-server", "args": []}
    tools = load_mcp_tools([bad, _server()])
    assert [t.metadata.name for t in tools] == ["echo"]  # bad server skipped, good bridged


def test_schema_to_model_edge_cases():
    """JSON Schema → pydantic model: required/optional fields, non-object, unknown types."""
    from drbrain.rag.mcp_tools import _schema_to_model

    m = _schema_to_model(
        "t",
        {
            "type": "object",
            "properties": {"name": {"type": "string"}, "count": {"type": "integer"}},
            "required": ["name"],
        },
    )
    assert m is not None
    fields = m.model_fields
    assert fields["name"].is_required()
    assert not fields["count"].is_required()
    assert type(None) in get_args(fields["count"].annotation)  # optional → None allowed
    # non-object schema / missing properties → None
    assert _schema_to_model("t", {"type": "string"}) is None
    assert _schema_to_model("t", {"type": "object"}) is None
    assert _schema_to_model("t", {}) is None
    # unknown type falls back to Any
    m2 = _schema_to_model("t", {"type": "object", "properties": {"x": {"type": "weird"}}})
    assert m2 is not None
    assert Any in get_args(m2.model_fields["x"].annotation)  # unknown type → Any
