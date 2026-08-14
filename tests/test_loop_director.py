"""ResearchDirector tests — the continuous research loop (AutoScientists-style).

The director runs the 12-node workflow repeatedly until stagnation, keeping a
checkpointed champion/rejected/results state and a running report. LLM stubbed
via ``llm_client.acall_with_messages`` (offline, mirrors test_loop_agent.py).
"""

from __future__ import annotations

import asyncio
import importlib.util

import pytest

from drbrain.config import Config, LlamaIndexConfig
from drbrain.loop import ResearchDirector
from drbrain.loop.director import _default_state

_HAS_LLAMA_INDEX = importlib.util.find_spec("llama_index") is not None
pytestmark = pytest.mark.skipif(not _HAS_LLAMA_INDEX, reason="llama_index not installed")

MODELS = [{"provider": "openai", "model": "gpt-4o", "api_key": "k", "base_url": None}]


def _cfg() -> Config:
    c = Config(llamaindex=LlamaIndexConfig(enabled=True))
    c.llm.models = list(MODELS)
    c.api.cache_ttl = 0
    c.llamaindex.storage_dir = "/nonexistent-rag-index"
    return c


def _cyclic_llm(monkeypatch, script):
    """Scripted LLM that cycles through ``script`` (one entry per agent call)."""
    calls = [0]

    async def fake(messages, models, tools=None, **kw):
        step = script[calls[0] % len(script)]
        calls[0] += 1
        return dict(step)

    monkeypatch.setattr("drbrain.extractor.llm_client.acall_with_messages", fake)
    return fake


_CYCLE_SCRIPT = [
    {"text": '{"query": "flat band"}', "tool_calls": None, "usage": None},       # retrieve distill
    {"text": '{"entities": ["flat band"]}', "tool_calls": None, "usage": None},  # extract
    {"text": '{"gaps": ["gap1"], "hypotheses": [{"statement": "h1", "conditions": {}}]}',
     "tool_calls": None, "usage": None},                                          # identify_gaps
    {"text": '{"hypotheses": [{"statement": "h1", "score": 0.9}]}',
     "tool_calls": None, "usage": None},                                          # critique
    {"text": '{"verified": ["h1"], "predictions": ["p1"]}',
     "tool_calls": None, "usage": None},                                          # verify
    {"text": "cycle report", "tool_calls": None, "usage": None},                  # report
]


def _write_search_plugin(tmp_path) -> str:
    (tmp_path / "search_papers.py").write_text(
        "from drbrain.plugins import Plugin\n"
        "def register(registry):\n"
        "    registry.register(Plugin(name='search_papers', description='d',\n"
        "        input_schema={'type':'object','properties':{'query':{'type':'string'},\n"
        "        'limit':{'type':'integer'}}}),\n"
        "        lambda a: {'papers': [{'title': 'Paper A'}, {'title': 'Paper B'}]})\n",
        encoding="utf-8",
    )
    return str(tmp_path)


# ── unit: state classification ────────────────────────────────────────────────


def test_absorb_classifies_champion_and_rejected():
    from drbrain.loop.events import ResearchState

    d = ResearchDirector(cfg=_cfg())
    state = _default_state("t")
    rs = ResearchState(
        verified=["h1"], falsified=["h2"], predictions=["p1"],
        hypotheses=[_h("h1"), _h("h2"), _h("h3 unresolved")],
    )
    d._absorb(state, "report", rs)
    assert [c["statement"] for c in state["champion"]] == ["h1"]
    assert "h2" in state["rejected"]  # h2 was falsified (DISCARD)
    assert "h1" not in state["rejected"]  # h1 was verified (KEEP)
    assert "h3 unresolved" not in state["rejected"]  # not falsified → not a dead end
    assert state["consecutive_no_gain"] == 0


def test_absorb_no_gain_increments_stagnation():
    from drbrain.loop.events import ResearchState

    d = ResearchDirector(cfg=_cfg())
    state = _default_state("t")
    state["champion"].append({"statement": "h1 confirmed", "cycle": 1, "confidence": 1.0})
    # same verified conclusion again → no new champion → stagnation++
    rs = ResearchState(verified=["h1 confirmed"], hypotheses=[_h("h1 confirmed")])
    d._absorb(state, "report", rs)
    assert state["consecutive_no_gain"] == 1
    assert len(state["champion"]) == 1


def _h(statement: str):
    from drbrain.loop.events import Hypothesis

    return Hypothesis(statement=statement)


def test_build_prior_context_lists_champion_and_rejected():
    state = _default_state("t")
    state["champion"] = [{"statement": "A 是 B", "cycle": 1, "confidence": 1.0}]
    state["rejected"] = ["X 是 Y"]
    prior = ResearchDirector._build_prior_context(state)
    assert "A 是 B" in prior
    assert "X 是 Y" in prior


def test_checkpoint_roundtrip(tmp_path):
    d = ResearchDirector(cfg=_cfg(), run_dir=str(tmp_path))
    state = _default_state("topological flat band")
    state["cycles"] = 2
    state["champion"].append({"statement": "h1", "cycle": 1, "confidence": 1.0})
    state["rejected"].append("h0 dead end")
    d._save_state("topological flat band", state)
    loaded = d._load_state("topological flat band")
    assert loaded["cycles"] == 2
    assert loaded["champion"][0]["statement"] == "h1"
    assert loaded["rejected"] == ["h0 dead end"]
    # canonical files exist (AutoScientists-style workspace, not a JSON blob)
    td = tmp_path / "topological-flat-band"
    assert (td / "champion.md").exists()
    assert (td / "dead_ends.md").exists()
    assert (td / "knowledge" / "patterns.md").exists()
    assert (td / "run.json").exists()


# ── integration: continuous loop to stagnation ────────────────────────────────


def test_director_runs_cycles_to_stagnation(monkeypatch, tmp_path):
    _cyclic_llm(monkeypatch, _CYCLE_SCRIPT)
    d = ResearchDirector(
        cfg=_cfg(), plugins_dir=_write_search_plugin(tmp_path), run_dir=str(tmp_path / "runs")
    )

    async def _go():
        return await d.run(
            "topological flat band", max_cycles=10, stagnation_cycles=2, max_adaptations=1
        )

    state = asyncio.run(_go())

    # h1 verified every cycle → new champion in cycle 1, then no-gain → stagnation
    # → Phase 4 adapt #1 (pivot) → max_adaptations=1 → stop at cycle 3.
    assert state["cycles"] == 3
    assert [c["statement"] for c in state["champion"]] == ["h1"]
    assert state["adaptations"] == 1
    assert state["consecutive_no_gain"] == 0  # reset after pivot
    assert len(state["results"]) == 3
    # per-cycle evidence + canonical files + logs exist on disk
    run_dir = tmp_path / "runs" / "topological-flat-band"
    assert (run_dir / "champion.md").exists()
    assert (run_dir / "results" / "cycle-001.md").exists()
    assert (run_dir / "results" / "cycle-003.md").exists()
    assert (run_dir / "run.json").exists()
    # AutoScientists-style canonical logs: experiments.jsonl + sessions.jsonl
    assert (run_dir / "logs" / "experiments.jsonl").exists()
    assert (run_dir / "logs" / "sessions.jsonl").exists()
    import json

    exp_lines = [json.loads(ln) for ln in (run_dir / "logs" / "experiments.jsonl").read_text().splitlines()]
    assert len(exp_lines) == 3
    assert {e["outcome"] for e in exp_lines} == {"KEEP", "NO_GAIN"}
