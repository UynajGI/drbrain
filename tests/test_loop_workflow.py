"""P0 smoke tests for the research loop skeleton."""

from __future__ import annotations

import asyncio
import json

from drbrain.loop import ResearchLoopWorkflow, ResearchState
from drbrain.loop.events import Evidence, GapsIdentified, Hypothesis
from drbrain.loop.front_half import DurableFrontHalf
from drbrain.loop.store import RunLedger
from drbrain.loop.transitions import TransitionService


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


def test_critique_rebuilds_the_compute_gate_from_durable_reviews(tmp_path, monkeypatch):
    class _Store:
        def __init__(self):
            self.values = {"research_state": ResearchState(task="durable critique")}

        async def get(self, key, default=None):
            return self.values.get(key, default)

        async def set(self, key, value):
            self.values[key] = value

    class _Context:
        def __init__(self):
            self.store = _Store()

    async def exercise():
        ledger = RunLedger(tmp_path / "ledger.sqlite3")
        run = ledger.get_or_create_run("durable critique")
        front_half = DurableFrontHalf(TransitionService(ledger), run.run_id)
        front_half.ensure_node_contracts()
        event = GapsIdentified(
            hypotheses=[
                Hypothesis(
                    claim_id="cl-durable-critique",
                    statement="durable review survives a crash",
                    prediction="a stored critic review enables compute",
                    falsification="a stored critic review is ignored",
                )
            ]
        )
        context = _Context()
        workflow = ResearchLoopWorkflow(durable_front_half=front_half)
        monkeypatch.setattr(workflow, "build_node_agent", lambda **_kwargs: None)

        pending = await workflow.critique(context, event)
        assert pending.hypotheses[0].status == "proposed"
        assert workflow._queue.claim("compute") is None  # noqa: SLF001

        proposal = front_half.snapshot()["proposals"][0]
        front_half.record_review(
            proposal["proposal_id"],
            reviewer="critic-1",
            score=0.9,
            verdict="KEEP",
            content="durably reviewed",
        )
        replayed_in_place = await workflow.critique(context, event)
        assert replayed_in_place.hypotheses[0].status == "critiqued"
        assert workflow._queue.claim("compute") is not None  # noqa: SLF001

        restored = ResearchLoopWorkflow(durable_front_half=front_half)
        monkeypatch.setattr(restored, "build_node_agent", lambda **_kwargs: None)

        accepted = await restored.critique(context, event)
        assert accepted.hypotheses[0].status == "critiqued"
        assert restored._queue.claim("compute") is not None  # noqa: SLF001
        event_types = [entry.event_type for entry in ledger.events(run.run_id)]
        assert {
            "front_half_contracts_registered",
            "proposal_recorded",
            "critic_review_recorded",
            "proposal_critiqued",
            "queue_item_recorded",
        } <= set(event_types)

        recovered_view = ResearchLoopWorkflow(durable_front_half=front_half)
        monkeypatch.setattr(recovered_view, "build_node_agent", lambda **_kwargs: None)
        await recovered_view.critique(context, GapsIdentified())
        durable_post = recovered_view._board.get_post(proposal["proposal_id"])  # noqa: SLF001
        assert durable_post is not None
        assert [comment.content for comment in durable_post.comments] == ["durably reviewed"]

    asyncio.run(exercise())


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
    dead_pid = 99999999  # beyond pid_max: /proc/<pid> never exists → job finished

    # empty run_dir / job_id → no evidence
    assert _job_log_has_number(None, "j1") is False
    assert _job_log_has_number(run_dir, "") is False
    # meta json missing
    assert _job_log_has_number(run_dir, "j1") is False
    # meta present but log file missing
    (jobs / "j1.json").write_text(json.dumps({"job_id": "j1", "pid": dead_pid}), encoding="utf-8")
    assert _job_log_has_number(run_dir, "j1") is False
    # log present but no parseable number
    (jobs / "j1.log").write_text("all done, nothing numeric", encoding="utf-8")
    assert _job_log_has_number(run_dir, "j1") is False
    # a parseable result JSON (non-empty min_bandwidth_ev) + job finished → evidence
    (jobs / "j1.log").write_text(
        '{"min_bandwidth_ev": 0.02, "flat_band_likely": true}', encoding="utf-8"
    )
    assert _job_log_has_number(run_dir, "j1") is True
    # still-running job (alive pid) with a numeric log → NOT final evidence
    (jobs / "j4.json").write_text('{"job_id": "j4", "pid": 1}', encoding="utf-8")
    (jobs / "j4.log").write_text("progress 0.5", encoding="utf-8")
    assert _job_log_has_number(run_dir, "j4") is False
    # log_path from meta is honored (not just <job_id>.log)
    elsewhere = tmp_path / "elsewhere.log"
    elsewhere.write_text('{"min_bandwidth_ev": 42.0}', encoding="utf-8")
    (jobs / "j2.json").write_text(
        json.dumps({"job_id": "j2", "pid": dead_pid, "log_path": str(elsewhere)}),
        encoding="utf-8",
    )
    assert _job_log_has_number(run_dir, "j2") is True
    # malformed meta json → no evidence
    (jobs / "j3.json").write_text("{broken", encoding="utf-8")
    assert _job_log_has_number(run_dir, "j3") is False
    # marker substring present but no parseable result JSON → not evidence
    (jobs / "j6.json").write_text(json.dumps({"job_id": "j6", "pid": dead_pid}), encoding="utf-8")
    (jobs / "j6.log").write_text("waiting on min_bandwidth_ev ... crashed", encoding="utf-8")
    assert _job_log_has_number(run_dir, "j6") is False
    # 毒化防护:meta 记了 status=failed 的作业,即使日志含完整结果 JSON 也不许过门
    (jobs / "j5.log").write_text(
        '{"min_bandwidth_ev": 0.02, "flat_band_likely": true}', encoding="utf-8"
    )
    (jobs / "j5.json").write_text(
        json.dumps(
            {
                "job_id": "j5",
                "pid": dead_pid,
                "status": "failed",
                "error": "NameError: name 'x' is not defined",
                "log_path": str(jobs / "j5.log"),
            }
        ),
        encoding="utf-8",
    )
    assert _job_log_has_number(run_dir, "j5") is False


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
    # job_id 指向真实作业文件、日志含数值、作业已结束 → verified
    (jobs / "j1.log").write_text('{"min_bandwidth_ev": 2.5}', encoding="utf-8")
    (jobs / "j1.json").write_text(
        json.dumps(
            {
                "job_id": "j1",
                "pid": 99999999,  # dead pid → job finished
                "log_path": str(jobs / "j1.log"),
            }
        ),
        encoding="utf-8",
    )
    ver = Verification(
        statement="h1", supports=1, refutes=0, computed="2.5", value=2.5, job_id="j1"
    )
    assert _classify_verification(ver, 0.9, True, run_dir) == "verified"
    # meta status=failed 的崩溃作业:即使日志含合法结果 JSON 也降级 prediction(不翻转 verified)
    (jobs / "j2.log").write_text('{"min_bandwidth_ev": 1.0}', encoding="utf-8")
    (jobs / "j2.json").write_text(
        json.dumps(
            {
                "job_id": "j2",
                "pid": 99999999,
                "status": "failed",
                "error": "NameError: name 'x' is not defined",
                "log_path": str(jobs / "j2.log"),
            }
        ),
        encoding="utf-8",
    )
    ver = Verification(
        statement="h2", supports=1, refutes=0, computed="1.0", value=1.0, job_id="j2"
    )
    assert _classify_verification(ver, 0.9, True, run_dir) == "prediction"
    # falsified 不受计算门影响
    assert (
        _classify_verification(
            Verification(statement="h2", supports=0, refutes=2), 0.9, True, run_dir
        )
        == "falsified"
    )
