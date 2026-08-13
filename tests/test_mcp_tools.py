"""Generic MCP bridge tests — connect/discover/call any stdio MCP server."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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
