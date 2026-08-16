"""P0 smoke tests for the research loop skeleton."""

from __future__ import annotations

import asyncio
import json

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
    # still complete (no hang) and produce a report with the machine-readable
    # summary line.
    result = _run("unretrievable topic")
    assert "verified=0" in result


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


# ── T4: 实算证据门单元测试 ────────────────────────────────────────────────────


def test_job_log_has_number_gate(tmp_path):
    from drbrain.loop.workflow import _job_log_has_number

    jobs = tmp_path / "jobs"
    jobs.mkdir()
    run_dir = str(jobs)

    # empty run_dir / job_id → no evidence
    assert _job_log_has_number(None, "j1") is False
    assert _job_log_has_number(run_dir, "") is False
    # meta json missing
    assert _job_log_has_number(run_dir, "j1") is False
    # meta present but log file missing
    (jobs / "j1.json").write_text('{"job_id": "j1", "pid": 1}', encoding="utf-8")
    assert _job_log_has_number(run_dir, "j1") is False
    # log present but no parseable number
    (jobs / "j1.log").write_text("all done, nothing numeric", encoding="utf-8")
    assert _job_log_has_number(run_dir, "j1") is False
    # number appears in the log → evidence
    (jobs / "j1.log").write_text("result = 1.0", encoding="utf-8")
    assert _job_log_has_number(run_dir, "j1") is True
    # log_path from meta is honored (not just <job_id>.log)
    elsewhere = tmp_path / "elsewhere.log"
    elsewhere.write_text("computed 42", encoding="utf-8")
    (jobs / "j2.json").write_text(
        json.dumps({"job_id": "j2", "pid": 2, "log_path": str(elsewhere)}), encoding="utf-8"
    )
    assert _job_log_has_number(run_dir, "j2") is True
    # malformed meta json → no evidence
    (jobs / "j3.json").write_text("{broken", encoding="utf-8")
    assert _job_log_has_number(run_dir, "j3") is False


def test_classify_verification_t4_job_gate(tmp_path):
    from drbrain.loop.events import Verification
    from drbrain.loop.workflow import _classify_verification

    jobs = tmp_path / "jobs"
    jobs.mkdir()
    run_dir = str(jobs)

    # 无计算工具：computed 为空也 verified（保持现状路径）
    assert (
        _classify_verification(Verification(statement="h1", supports=1, refutes=0), 0.9, False)
        == "verified"
    )
    # 计算工具在场但无 job 证据：computed/value 都填了数值也降级 prediction（防编造核心）
    ver = Verification(statement="h1", supports=1, refutes=0, computed="1.0", value=1.0)
    assert _classify_verification(ver, 0.9, True, run_dir) == "prediction"
    # job_id 指向真实作业文件且日志含数值 → verified
    (jobs / "j1.log").write_text("value 2.5", encoding="utf-8")
    (jobs / "j1.json").write_text(
        json.dumps({"job_id": "j1", "pid": 1, "log_path": str(jobs / "j1.log")}),
        encoding="utf-8",
    )
    ver = Verification(
        statement="h1", supports=1, refutes=0, computed="2.5", value=2.5, job_id="j1"
    )
    assert _classify_verification(ver, 0.9, True, run_dir) == "verified"
    # falsified 不受计算门影响
    assert (
        _classify_verification(
            Verification(statement="h2", supports=0, refutes=2), 0.9, True, run_dir
        )
        == "falsified"
    )
