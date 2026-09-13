from __future__ import annotations

import pytest

from drbrain.loop.benchmark import run_frontier_benchmark
from drbrain.loop.frontier import (
    BranchOutcome,
    BranchSpec,
    BranchStatus,
    EvidenceRef,
    ResearchObjective,
)
from drbrain.loop.research_events import InMemoryEventLog
from drbrain.loop.supervisor import (
    ResearchSupervisor,
    SupervisorConfig,
    SupervisorError,
    WorkerContractError,
)


@pytest.mark.asyncio
async def test_director_adaptive_entrypoint_uses_supervisor(monkeypatch, tmp_path) -> None:
    from drbrain.loop import director as director_module
    from drbrain.loop.director import ResearchDirector
    from drbrain.loop.events import Evidence, ResearchState

    class Handler:
        ctx = type("Ctx", (), {"store": None})()

        def __init__(self) -> None:
            self.ctx.store = self

        def __await__(self):
            async def done():
                return "adaptive report"

            return done().__await__()

        async def get(self, key, default=None):
            if key == "research_state":
                return ResearchState(verified=["claim"], evidence=[Evidence(evidence_id="ev-1")])
            return default

    class FakeWorkflow:
        def __init__(self, **_kwargs) -> None:
            pass

        def run(self, **_kwargs):
            return Handler()

    monkeypatch.setattr(director_module, "ResearchLoopWorkflow", FakeWorkflow)
    director = ResearchDirector(cfg=object(), run_dir=tmp_path)
    result = await director.run_adaptive("question", max_evaluations=1)
    assert result.completed is True
    assert result.frontier.completed_evaluations == 1


@pytest.mark.asyncio
async def test_supervisor_runs_best_first_and_commits_events() -> None:
    seen: list[str] = []

    async def worker(branch, _objective, _done):
        seen.append(branch.branch_id)
        return BranchOutcome(
            branch_id=branch.branch_id,
            status=BranchStatus.RETAINED,
            claims=[branch.hypothesis],
            evidence=[EvidenceRef(evidence_id=f"ev-{branch.branch_id}")],
        )

    log = InMemoryEventLog()
    supervisor = ResearchSupervisor(
        objective=ResearchObjective(objective_id="o1", question="test"),
        worker=worker,
        event_log=log,
        done_contract=None,
        config=SupervisorConfig(max_parallel_branches=1, max_evaluations=2),
    )
    result = await supervisor.run(
        [
            BranchSpec(branch_id="low", hypothesis="low", feasibility=0.1),
            BranchSpec(branch_id="high", hypothesis="high", feasibility=0.9),
        ]
    )

    assert seen == ["high"]
    assert result.completed is True
    assert result.reason == "done_contract"
    assert [event.event_type for event in log.read(result.run_id)] == [
        "run_created",
        "branch_started",
        "branch_completed",
        "run_completed",
    ]


@pytest.mark.asyncio
async def test_expensive_branch_waits_for_approval_then_runs() -> None:
    calls: list[str] = []

    async def worker(branch, _objective, _done):
        calls.append(branch.branch_id)
        return BranchOutcome(branch_id=branch.branch_id, status=BranchStatus.PRUNED)

    branch = BranchSpec(branch_id="expensive", metadata={"expensive": True})
    supervisor = ResearchSupervisor(
        objective=ResearchObjective(question="test"),
        worker=worker,
        config=SupervisorConfig(max_evaluations=1, human_mode="approve_expensive"),
    )
    waiting = await supervisor.run([branch])
    assert waiting.reason == "frontier_exhausted"
    assert calls == []
    supervisor.approve_branch("expensive")
    finished = await supervisor.run()
    assert finished.frontier.branches["expensive"].status is BranchStatus.PRUNED
    assert calls == ["expensive"]


@pytest.mark.asyncio
async def test_supervisor_rejects_invalid_worker_outcome() -> None:
    async def worker(_branch, _objective, _done):
        return {"status": "retained"}

    supervisor = ResearchSupervisor(
        objective=ResearchObjective(question="test"),
        worker=worker,
        config=SupervisorConfig(max_evaluations=1),
    )
    with pytest.raises(WorkerContractError):
        await supervisor.run([BranchSpec(branch_id="b1", hypothesis="h")])


def test_supervisor_requires_worker_before_run() -> None:
    supervisor = ResearchSupervisor(objective=ResearchObjective(question="test"))
    with pytest.raises(SupervisorError):
        import asyncio

        asyncio.run(supervisor.run([BranchSpec(branch_id="b1")]))


@pytest.mark.asyncio
async def test_frontier_benchmark_reports_scheduler_metrics() -> None:
    async def worker(branch, _objective, _done):
        return BranchOutcome(branch_id=branch.branch_id, status=BranchStatus.PRUNED)

    metrics = await run_frontier_benchmark([BranchSpec(branch_id="b1")], worker)
    assert metrics["evaluations"] == 1
    assert metrics["pruned"] == 1
    assert metrics["event_count"] >= 3


def test_supervisor_restores_frontier_from_event_log() -> None:
    log = InMemoryEventLog()
    run_id = "run-replay"
    objective = ResearchObjective(objective_id="o1", question="recover")
    log.append(
        run_id=run_id,
        event_type="run_created",
        actor="supervisor",
        payload={"objective": objective.to_dict(), "done_contract": {}},
        idempotency_key=f"run:{run_id}:created",
    )
    branch = BranchSpec(branch_id="b1", objective_id="o1", hypothesis="h")
    log.append(
        run_id=run_id,
        event_type="branch_started",
        actor="supervisor",
        payload={"branch": {**branch.to_dict(), "status": "running"}},
        idempotency_key="branch:b1:started",
    )
    log.append(
        run_id=run_id,
        event_type="branch_completed",
        actor="supervisor",
        payload={
            "branch_id": "b1",
            "outcome": BranchOutcome(
                branch_id="b1",
                status=BranchStatus.RETAINED,
                evidence=[EvidenceRef(evidence_id="ev-1")],
            ).to_dict(),
            "admitted_children": [],
        },
        idempotency_key="branch:b1:completed",
    )

    supervisor = ResearchSupervisor(
        objective=ResearchObjective(question="placeholder"),
        worker=lambda *_: BranchOutcome(branch_id="b1", status=BranchStatus.RETAINED),
        event_log=log,
        run_id=run_id,
    )

    assert supervisor.objective.question == "recover"
    assert supervisor.frontier.completed_evaluations == 1
    assert supervisor.frontier.branches["b1"].status is BranchStatus.RETAINED
    assert supervisor.frontier.branches["b1"].metadata["supported_evidence"] == ["ev-1"]


def test_supervisor_restore_prefers_snapshot_then_replays_tail() -> None:
    from drbrain.loop.research_events import EventSourcedState, InMemorySnapshotStore
    from drbrain.loop.supervisor import _default_reducer

    log = InMemoryEventLog()
    snapshots = InMemorySnapshotStore()
    run_id = "run-snapshot"
    log.append(
        run_id=run_id,
        event_type="run_created",
        actor="supervisor",
        payload={"objective": {"question": "snapshot"}, "done_contract": {}},
        idempotency_key="run:snapshot:created",
    )
    branch = BranchSpec(branch_id="b1", hypothesis="h")
    log.append(
        run_id=run_id,
        event_type="branch_started",
        actor="supervisor",
        payload={"branch": {**branch.to_dict(), "status": "running"}},
        idempotency_key="branch:snapshot:started",
    )
    state = EventSourcedState(log, reducer=_default_reducer, snapshots=snapshots)
    state.snapshot(run_id)
    log.append(
        run_id=run_id,
        event_type="branch_completed",
        actor="supervisor",
        payload={"outcome": BranchOutcome(branch_id="b1", status=BranchStatus.RETAINED).to_dict()},
        idempotency_key="branch:snapshot:completed",
    )

    supervisor = ResearchSupervisor(
        objective=ResearchObjective(question="placeholder"),
        worker=lambda *_: BranchOutcome(branch_id="b1", status=BranchStatus.RETAINED),
        event_log=log,
        state=EventSourcedState(log, reducer=_default_reducer, snapshots=snapshots),
        run_id=run_id,
    )

    assert supervisor.frontier.completed_evaluations == 1
    assert supervisor.frontier.branches["b1"].status is BranchStatus.RETAINED
