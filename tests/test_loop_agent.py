"""Agent-backed node tests: the loop builds + runs a FunctionAgent.

Proves the loop layer's "independent agent" machinery — ``build_node_agent``
assembles the full tool surface (graph tools + external plugins) and
``run_agent`` runs it to a text answer. The LLM is stubbed via
``llm_client.acall_with_messages`` (offline, mirrors test_rag_agent.py).
"""

from __future__ import annotations

import asyncio
import importlib.util

import pytest

from drbrain.config import Config, LlamaIndexConfig
from drbrain.loop import ResearchLoopWorkflow

_HAS_LLAMA_INDEX = importlib.util.find_spec("llama_index") is not None
pytestmark = pytest.mark.skipif(not _HAS_LLAMA_INDEX, reason="llama_index not installed")

MODELS = [{"provider": "openai", "model": "gpt-4o", "api_key": "k", "base_url": None}]


def _cfg() -> Config:
    c = Config(llamaindex=LlamaIndexConfig(enabled=True))
    c.llm.models = list(MODELS)
    c.api.cache_ttl = 0
    c.llamaindex.storage_dir = "/nonexistent-rag-index"
    return c


def _scripted_llm(monkeypatch, script):
    calls = [0]

    async def fake(messages, models, tools=None, **kw):
        step = script[min(calls[0], len(script) - 1)]
        calls[0] += 1
        return dict(step)

    monkeypatch.setattr("drbrain.extractor.llm_client.acall_with_messages", fake)
    return fake


def test_build_node_agent_assembles_full_tool_surface(tmp_path):
    (tmp_path / "foo_plugin.py").write_text(
        "from drbrain.plugins import Plugin\n"
        "def register(registry):\n"
        "    registry.register(Plugin(name='foo', description='d', input_schema={}), lambda a: {})\n",
        encoding="utf-8",
    )
    wf = ResearchLoopWorkflow(cfg=_cfg(), plugins_dir=str(tmp_path))
    agent = wf.build_node_agent()
    assert agent is not None
    names = {t.metadata.name for t in agent.tools}
    assert "search_concepts" in names  # built-in graph tool
    assert "foo" in names  # external plugin tool


def test_build_node_agent_none_without_cfg():
    wf = ResearchLoopWorkflow()
    assert wf.build_node_agent() is None


def test_run_agent_returns_answer(monkeypatch):
    _scripted_llm(monkeypatch, [{"text": "Hello from agent.", "tool_calls": None, "usage": None}])
    wf = ResearchLoopWorkflow(cfg=_cfg())
    agent = wf.build_node_agent()
    answer = asyncio.run(wf.run_agent(agent, "hi"))
    assert answer == "Hello from agent."


def test_run_agent_none_returns_none():
    wf = ResearchLoopWorkflow()
    answer = asyncio.run(wf.run_agent(None, "hi"))
    assert answer is None


def test_retrieve_node_uses_agent(monkeypatch):
    """The retrieve node runs the agent and its answer becomes candidates."""
    _scripted_llm(monkeypatch, [{"text": "Paper A\nPaper B", "tool_calls": None, "usage": None}])
    wf = ResearchLoopWorkflow(cfg=_cfg())

    async def _go() -> str:
        handler = wf.run(task="flat band")
        return await handler

    result = asyncio.run(_go())
    assert "candidates=2" in result


def test_identify_gaps_node_proposes_hypotheses(monkeypatch):
    """identify_gaps runs the agent, parsing structured JSON into gaps + hypotheses."""
    _scripted_llm(
        monkeypatch,
        [
            {"text": "Paper A\nPaper B", "tool_calls": None, "usage": None},
            {
                "text": (
                    '{"gaps": ["gap1", "gap2"], '
                    '"hypotheses": [{"statement": "h1", "conditions": {}}, '
                    '{"statement": "h2", "conditions": {}}]}'
                ),
                "tool_calls": None,
                "usage": None,
            },
        ],
    )
    wf = ResearchLoopWorkflow(cfg=_cfg())

    async def _go() -> str:
        handler = wf.run(task="flat band")
        return await handler

    result = asyncio.run(_go())
    assert "gaps=2" in result
    assert "hypotheses=2" in result


def test_parse_json_lenient():
    from drbrain.loop.workflow import _parse_json_lenient

    assert _parse_json_lenient('```json\n{"a": 1}\n```') == {"a": 1}
    assert _parse_json_lenient('Here is the result: {"a": 1}') == {"a": 1}
    assert _parse_json_lenient("no json here") is None
