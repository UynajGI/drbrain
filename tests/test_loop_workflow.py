"""P0 smoke tests for the research loop skeleton."""

from __future__ import annotations

import asyncio

from drbrain.loop import ResearchLoopWorkflow, ResearchState
from drbrain.loop.events import Evidence, Hypothesis


def _run(task: str) -> str:
    """Run the workflow to completion and return the report string."""

    async def _go() -> str:
        wf = ResearchLoopWorkflow(timeout=30)
        handler = wf.run(task=task)
        return await handler

    return asyncio.run(_go())


def test_loop_runs_end_to_end():
    result = _run("2D flat-band materials")
    assert isinstance(result, str)
    assert "task='2D flat-band materials'" in result
    assert "candidates=0" in result


def test_loop_terminates_on_empty_candidates():
    # Empty candidates drive the bounded retrieve-again loop; the pipeline must
    # still complete (no hang) and produce a report.
    result = _run("unretrievable topic")
    assert result.endswith("verified=0")


def test_state_schema_defaults():
    state = ResearchState(task="t")
    assert state.candidates == []
    assert state.hypotheses == []
    assert state.report == ""


def test_evidence_and_hypothesis_schema():
    ev = Evidence(paper_id="p1", page=3, value=0.9, unit="eV")
    assert ev.authority == ""
    assert ev.conditions == {}

    h = Hypothesis(statement="s", conditions={"T": 300})
    assert h.status == "proposed"
    assert h.conditions == {"T": 300}


def test_load_plugins_discovers_external(tmp_path):
    (tmp_path / "foo_plugin.py").write_text(
        "from drbrain.plugins import Plugin\n"
        "def register(registry):\n"
        "    registry.register(\n"
        "        Plugin(name='foo', description='d', input_schema={}),\n"
        "        lambda args: {},\n"
        "    )\n",
        encoding="utf-8",
    )
    wf = ResearchLoopWorkflow(plugins_dir=str(tmp_path))
    registry = wf.load_plugins()
    assert [p.name for p in registry.list_plugins()] == ["foo"]


def test_load_plugins_graceful_when_no_dir():
    wf = ResearchLoopWorkflow()
    registry = wf.load_plugins()
    assert registry.list_plugins() == []
