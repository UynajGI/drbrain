"""Durability contracts for the experiment-to-settlement half of the loop."""

from __future__ import annotations

import json

import pytest

from drbrain.loop.durable_execution import ChampionVersionConflictError, DurableExecution
from drbrain.loop.events import Hypothesis, ResearchState, Verification, Verified
from drbrain.loop.store import RunLedger
from drbrain.loop.transitions import TransitionService
from drbrain.loop.workflow import ResearchLoopWorkflow


def _execution(tmp_path, *, noise_band: float = 0.0) -> DurableExecution:
    ledger = RunLedger(tmp_path / "ledger.sqlite3")
    run = ledger.get_or_create_run("durable execution")
    execution = DurableExecution(
        TransitionService(ledger), run.run_id, noise_band=noise_band, required_repeats=2
    )
    execution.ensure_node_contracts()
    return execution


def _hypothesis(claim_id: str = "cl-durable-execution") -> dict[str, object]:
    return {
        "claim_id": claim_id,
        "proposal_id": "prp-durable-execution",
        "statement": "A durable experiment needs a traceable numeric result.",
        "prediction": "The controlled metric increases by a measurable amount.",
        "falsification": "The metric remains unchanged after the intervention.",
        "conditions": {"seed": 7, "dataset": "fixture"},
    }


def _job(tmp_path, job_id: str, *, output: str = "metric=0.8") -> None:
    log = tmp_path / f"{job_id}.log"
    log.write_text(output, encoding="utf-8")
    (tmp_path / f"{job_id}.json").write_text(
        json.dumps({"pid": None, "log_path": str(log)}), encoding="utf-8"
    )


def test_experiment_records_replay_safe_artifact_lineage(tmp_path):
    execution = _execution(tmp_path)
    experiment = execution.record_experiment(
        _hypothesis(),
        environment={"python": "3.12"},
        config={"runner": "fixture"},
    )
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    _job(jobs, "job-1")

    first = execution.record_compute_output(
        experiment["experiment_id"],
        job_id="job-1",
        jobs_dir=jobs,
        tool_call_id="tool-1",
        code={"code": "print(0.8)"},
    )
    replay = execution.record_compute_output(
        experiment["experiment_id"],
        job_id="job-1",
        jobs_dir=jobs,
        tool_call_id="tool-1",
        code={"code": "print(0.8)"},
    )
    snapshot = execution.snapshot()

    assert first["numeric"] is True
    assert replay == first
    assert snapshot["experiments"][0]["claim_id"] == "cl-durable-execution"
    assert {artifact["kind"] for artifact in snapshot["artifacts"]} >= {
        "plan",
        "config",
        "environment",
        "seed",
        "input",
        "code",
        "output",
    }
    assert snapshot["artifacts"][-1]["tool_call_id"] == "tool-1"
    code = next(artifact for artifact in snapshot["artifacts"] if artifact["kind"] == "code")
    assert code["metadata"]["inline_payload"]["arguments"]["code"] == "print(0.8)"


def test_verifier_cannot_write_experiment_artifacts(tmp_path):
    execution = _execution(tmp_path)
    experiment = execution.record_experiment(_hypothesis(), environment={}, config={})

    with pytest.raises(PermissionError, match="compute"):
        execution.record_artifact(
            experiment["experiment_id"],
            actor="verifier",
            kind="output",
            uri="inline://invalid",
            payload={"value": 1},
        )


def test_settlement_blocks_promotion_without_numeric_artifact_or_evidence(tmp_path):
    execution = _execution(tmp_path)
    experiment = execution.record_experiment(_hypothesis(), environment={}, config={})

    settlement = execution.settle_verification(
        experiment["experiment_id"],
        verification={
            "claim_id": "cl-durable-execution",
            "statement": _hypothesis()["statement"],
            "status": "verified",
            "supports": 1,
            "refutes": 0,
            "evidence_ids": [],
            "value": 0.8,
        },
        expected_champion_version=0,
    )

    assert settlement["verdict"] == "insufficient"
    assert settlement["champion_version"] is None
    assert execution.snapshot()["champion_version"] == 0


def test_settlement_uses_cas_and_configured_noise_repeat_gate(tmp_path):
    execution = _execution(tmp_path, noise_band=0.05)
    jobs = tmp_path / "jobs"
    jobs.mkdir()

    first = execution.record_experiment(_hypothesis("cl-first"), environment={}, config={})
    _job(jobs, "job-first")
    execution.record_compute_output(first["experiment_id"], job_id="job-first", jobs_dir=jobs)
    noisy = execution.settle_verification(
        first["experiment_id"],
        verification={
            "claim_id": "cl-first",
            "statement": _hypothesis("cl-first")["statement"],
            "status": "verified",
            "supports": 1,
            "refutes": 0,
            "evidence_ids": ["evidence-1"],
            "value": 0.02,
        },
        expected_champion_version=0,
    )
    assert noisy["verdict"] == "insufficient"
    assert noisy["reason"] == "near_noise_requires_repeat"

    second = execution.record_experiment(_hypothesis("cl-second"), environment={}, config={})
    _job(jobs, "job-second")
    execution.record_compute_output(second["experiment_id"], job_id="job-second", jobs_dir=jobs)
    kept = execution.settle_verification(
        second["experiment_id"],
        verification={
            "claim_id": "cl-second",
            "statement": _hypothesis("cl-second")["statement"],
            "status": "verified",
            "supports": 1,
            "refutes": 0,
            "evidence_ids": ["evidence-2"],
            "value": 0.8,
        },
        expected_champion_version=0,
    )
    assert kept["verdict"] == "keep"
    assert kept["champion_version"] == 1

    third = execution.record_experiment(_hypothesis("cl-third"), environment={}, config={})
    _job(jobs, "job-third")
    execution.record_compute_output(third["experiment_id"], job_id="job-third", jobs_dir=jobs)
    with pytest.raises(ChampionVersionConflictError):
        execution.settle_verification(
            third["experiment_id"],
            verification={
                "claim_id": "cl-third",
                "statement": _hypothesis("cl-third")["statement"],
                "status": "verified",
                "supports": 1,
                "refutes": 0,
                "evidence_ids": ["evidence-3"],
                "value": 0.9,
            },
            expected_champion_version=0,
        )


def test_workflow_settle_and_report_consume_only_durable_settlements(tmp_path):
    class _Store:
        def __init__(self, state):
            self.values = {"research_state": state}

        async def get(self, key, default=None):
            return self.values.get(key, default)

        async def set(self, key, value):
            self.values[key] = value

    class _Context:
        def __init__(self, state):
            self.store = _Store(state)

    async def exercise():
        execution = _execution(tmp_path)
        hypothesis = _hypothesis()
        experiment = execution.record_experiment(hypothesis, environment={}, config={})
        state = ResearchState(
            task="durable report",
            hypotheses=[Hypothesis(**hypothesis, status="critiqued")],
        )
        context = _Context(state)
        workflow = ResearchLoopWorkflow(durable_execution=execution)
        # A durable report must not invoke an LLM to replace the canonical text.
        workflow.build_node_agent = lambda **_kwargs: object()  # type: ignore[method-assign]
        event = Verified(
            verified=["agent-injected claim"],
            verifications=[
                Verification(
                    claim_id=hypothesis["claim_id"],
                    statement=hypothesis["statement"],
                    status="verified",
                    supports=1,
                    evidence_ids=["evidence-1"],
                    value=0.9,
                )
            ],
            experiment_ids={hypothesis["statement"]: experiment["experiment_id"]},
        )

        settled = await workflow.settle(context, event)
        report = await workflow.report(context, settled)

        assert settled.verified == []
        assert "agent-injected claim" not in report.result
        assert "durable report" in report.result

    import asyncio

    asyncio.run(exercise())
