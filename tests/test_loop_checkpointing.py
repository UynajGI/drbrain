"""Focused tests for step-level autoresearch checkpoint recovery."""

from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step

from drbrain.loop.checkpointing import (
    CheckpointCompatibilityError,
    CheckpointManifest,
    WorkflowCheckpointService,
)
from drbrain.loop.discussion import POST_PROPOSAL, MessageBoard, QueueItem, ResearchQueue
from drbrain.loop.events import Hypothesis, ResearchState
from drbrain.loop.store import RunLedger
from drbrain.loop.transitions import LeaseUnavailableError, TransitionService
from drbrain.loop.workflow import ResearchLoopWorkflow


class _FakeContext:
    """Minimal JSON-serializable Context stand-in for ledger boundary tests."""

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.serializer = None

    def to_dict(self, *, serializer):  # noqa: ANN001 - mirrors LlamaIndex Context
        self.serializer = serializer
        return self.payload


class _FakeWorkflow:
    def checkpoint_state(self) -> dict[str, object]:
        return {"board": {"posts": []}, "queue": {"pending": []}}


class _CheckpointWorkflow(Workflow):
    """Small real Workflow held at a safe boundary for Context serialization."""

    def __init__(self, entered: asyncio.Event, release: asyncio.Event) -> None:
        super().__init__(timeout=5)
        self._entered = entered
        self._release = release
        self.restored_state: dict[str, object] | None = None

    @step
    async def wait_for_checkpoint(self, ctx: Context, _ev: StartEvent) -> StopEvent:
        await ctx.store.set("research_state", ResearchState(task="checkpoint topic"))
        self._entered.set()
        await self._release.wait()
        return StopEvent(result="done")

    def checkpoint_state(self) -> dict[str, object]:
        return {"board": {"posts": []}, "queue": {"pending": []}}

    def restore_checkpoint_state(self, state: dict[str, object]) -> None:
        self.restored_state = state


def _started_cycle(tmp_path):
    ledger = RunLedger(tmp_path / "ledger.sqlite3")
    run = ledger.get_or_create_run("checkpoint topic")
    transitions = TransitionService(ledger)
    transitions.start_run(run.run_id)
    step_id = transitions.begin_cycle(
        run.run_id,
        cycle=1,
        worker_id="worker-a",
        lease_seconds=60,
    )
    attempt_id = ledger.active_attempt_id(step_id)
    assert attempt_id is not None
    return ledger, run.run_id, transitions, step_id, attempt_id


def _started_cycle_for_run(
    ledger: RunLedger, transitions: TransitionService, topic: str, worker_id: str
) -> tuple[str, str, str]:
    run = ledger.get_or_create_run(topic)
    transitions.start_run(run.run_id)
    step_id = transitions.begin_cycle(
        run.run_id,
        cycle=1,
        worker_id=worker_id,
        lease_seconds=60,
    )
    attempt_id = ledger.active_attempt_id(step_id)
    assert attempt_id is not None
    return run.run_id, step_id, attempt_id


def _manifest(*, model: str = "model-a") -> CheckpointManifest:
    return CheckpointManifest(
        workflow_version="research-loop-v1",
        model_manifest={"model": model},
        tool_manifest={"plugins": []},
        rag_generation=None,
    )


@pytest.mark.parametrize("operation", ["complete", "fail", "manual_review"])
def test_transition_rejects_step_owned_by_another_run(tmp_path, operation: str):
    ledger = RunLedger(tmp_path / "ledger.sqlite3")
    transitions = TransitionService(ledger)
    run_a, step_a, _attempt_a = _started_cycle_for_run(ledger, transitions, "run A", "worker-a")
    run_b, _step_b, _attempt_b = _started_cycle_for_run(ledger, transitions, "run B", "worker-b")

    with pytest.raises(KeyError, match="unknown step"):
        if operation == "complete":
            transitions.complete_cycle(
                run_b,
                step_id=step_a,
                cycle_result={},
                state_snapshot={},
                research_state=None,
                worker_id="worker-a",
            )
        elif operation == "fail":
            transitions.fail_cycle(
                run_b,
                step_id=step_a,
                error=RuntimeError("cross-run failure"),
                worker_id="worker-a",
            )
        else:
            transitions.mark_manual_review(
                run_b,
                step_id=step_a,
                reason="cross-run manual review",
            )

    with sqlite3.connect(ledger.path) as conn:
        status = conn.execute(
            "SELECT status FROM research_steps WHERE step_id = ?", (step_a,)
        ).fetchone()[0]
    assert status == "running"
    assert not any(event.payload.get("step_id") == step_a for event in ledger.events(run_b))
    assert ledger.get_run("run A").status == "running"


def test_resume_and_restore_reject_a_checkpoint_from_another_run_or_step(tmp_path):
    ledger = RunLedger(tmp_path / "ledger.sqlite3")
    transitions = TransitionService(ledger)
    run_a, step_a, attempt_a = _started_cycle_for_run(ledger, transitions, "run A", "worker-a")
    checkpoint = WorkflowCheckpointService(
        ledger=ledger,
        run_id=run_a,
        step_id=step_a,
        attempt_id=attempt_a,
        worker_id="worker-a",
        manifest=_manifest(),
        lease_seconds=60,
    ).capture(
        ctx=_FakeContext({"globals": {"cursor": "from-run-a"}}),
        workflow=_FakeWorkflow(),
        step_name="retrieve",
    )
    run_b, step_b, attempt_b = _started_cycle_for_run(ledger, transitions, "run B", "worker-b")
    with ledger.transaction() as conn:
        conn.execute("UPDATE research_steps SET lease_expires_at = 0 WHERE step_id = ?", (step_b,))
    transitions.reconcile_incomplete_cycles(run_b)

    with pytest.raises(KeyError, match="checkpoint"):
        transitions.resume_cycle(
            run_b,
            step_id=step_b,
            checkpoint_id=checkpoint.checkpoint_id,
            worker_id="worker-c",
            lease_seconds=60,
        )
    assert ledger.active_attempt_id(step_b) is None

    foreign_checkpoint = WorkflowCheckpointService(
        ledger=ledger,
        run_id=run_b,
        step_id=step_b,
        attempt_id=attempt_b,
        worker_id="worker-c",
        manifest=_manifest(),
        lease_seconds=60,
        checkpoint=checkpoint,
    )
    with pytest.raises(CheckpointCompatibilityError, match="run/step"):
        foreign_checkpoint.validate_checkpoint()


def test_checkpoint_is_json_and_renews_the_worker_lease(tmp_path):
    ledger, run_id, _transitions, step_id, attempt_id = _started_cycle(tmp_path)
    context = _FakeContext({"globals": {"research_state": {"task": "checkpoint topic"}}})
    service = WorkflowCheckpointService(
        ledger=ledger,
        run_id=run_id,
        step_id=step_id,
        attempt_id=attempt_id,
        worker_id="worker-a",
        manifest=_manifest(),
        lease_seconds=60,
    )

    checkpoint = service.capture(ctx=context, workflow=_FakeWorkflow(), step_name="retrieve")

    assert context.serializer is not None
    assert json.loads(json.dumps(checkpoint.context_payload)) == context.payload
    assert checkpoint.step_name == "retrieve"
    assert ledger.latest_checkpoint_for_step(step_id) == checkpoint
    assert ledger.active_attempt_id(step_id) == attempt_id
    assert ledger.inflight_workflow_step(step_id) is None
    assert ledger.events(run_id)[-1].event_type == "workflow_checkpointed"
    with sqlite3.connect(ledger.path) as conn:
        owner, expires_at = conn.execute(
            "SELECT lease_owner, lease_expires_at FROM research_steps WHERE step_id = ?",
            (step_id,),
        ).fetchone()
    assert owner == "worker-a"
    assert expires_at is not None


def test_real_llamaindex_context_json_round_trips(tmp_path):
    ledger, run_id, _transitions, step_id, attempt_id = _started_cycle(tmp_path)

    async def scenario() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()
        workflow = _CheckpointWorkflow(entered, release)
        handler = workflow.run()
        await asyncio.wait_for(entered.wait(), timeout=5)
        service = WorkflowCheckpointService(
            ledger=ledger,
            run_id=run_id,
            step_id=step_id,
            attempt_id=attempt_id,
            worker_id="worker-a",
            manifest=_manifest(),
            lease_seconds=60,
        )
        try:
            checkpoint = service.capture(
                ctx=handler.ctx,
                workflow=workflow,
                step_name="retrieve",
            )
        finally:
            release.set()
        await handler

        restored_workflow = _CheckpointWorkflow(asyncio.Event(), asyncio.Event())
        restored = WorkflowCheckpointService(
            ledger=ledger,
            run_id=run_id,
            step_id=step_id,
            attempt_id=attempt_id,
            worker_id="worker-a",
            manifest=_manifest(),
            lease_seconds=60,
            checkpoint=checkpoint,
        ).restore_context(restored_workflow)
        state = await restored.store.get("research_state")
        assert isinstance(state, ResearchState)
        assert state.task == "checkpoint topic"

    asyncio.run(scenario())


def test_expired_checkpointed_cycle_reclaims_once_and_resumes_new_attempt(tmp_path):
    ledger, run_id, transitions, step_id, attempt_id = _started_cycle(tmp_path)
    service = WorkflowCheckpointService(
        ledger=ledger,
        run_id=run_id,
        step_id=step_id,
        attempt_id=attempt_id,
        worker_id="worker-a",
        manifest=_manifest(),
        lease_seconds=60,
    )
    checkpoint = service.capture(
        ctx=_FakeContext({"globals": {"cursor": 1}}),
        workflow=_FakeWorkflow(),
        step_name="retrieve",
    )
    with ledger.transaction() as conn:
        conn.execute("UPDATE research_steps SET lease_expires_at = 0 WHERE step_id = ?", (step_id,))

    transitions.reconcile_incomplete_cycles(run_id)
    assert ledger.recoverable_step_ids(run_id) == [step_id]
    resumed_attempt = transitions.resume_cycle(
        run_id,
        step_id=step_id,
        checkpoint_id=checkpoint.checkpoint_id,
        worker_id="worker-b",
        lease_seconds=60,
    )

    assert resumed_attempt != attempt_id
    assert ledger.active_attempt_id(step_id) == resumed_attempt
    with sqlite3.connect(ledger.path) as conn:
        attempts = conn.execute(
            "SELECT attempt_no, status, checkpoint_ref FROM research_attempts "
            "WHERE step_id = ? ORDER BY attempt_no",
            (step_id,),
        ).fetchall()
        status, owner = conn.execute(
            "SELECT status, lease_owner FROM research_steps WHERE step_id = ?", (step_id,)
        ).fetchone()
    assert attempts == [
        (1, "unknown", checkpoint.checkpoint_id),
        (2, "running", checkpoint.checkpoint_id),
    ]
    assert (status, owner) == ("reconciling", "worker-b")


def test_live_lease_rejects_a_second_worker(tmp_path):
    _ledger, run_id, transitions, _step_id, _attempt_id = _started_cycle(tmp_path)

    with pytest.raises(LeaseUnavailableError, match="active lease"):
        transitions.begin_cycle(
            run_id,
            cycle=2,
            worker_id="worker-b",
            lease_seconds=60,
        )


def test_interrupted_external_node_is_visible_for_manual_recovery(tmp_path):
    ledger, run_id, transitions, step_id, attempt_id = _started_cycle(tmp_path)
    ledger.record_workflow_step_started(
        run_id=run_id,
        step_id=step_id,
        attempt_id=attempt_id,
        worker_id="worker-a",
        lease_seconds=60,
        node_name="compute",
    )

    assert ledger.inflight_workflow_step(step_id) == "compute"
    assert WorkflowCheckpointService.requires_manual_recovery("compute")
    assert not WorkflowCheckpointService.requires_manual_recovery("retrieve")
    with ledger.transaction() as conn:
        conn.execute("UPDATE research_steps SET lease_expires_at = 0 WHERE step_id = ?", (step_id,))
    transitions.reconcile_incomplete_cycles(run_id)
    transitions.mark_manual_review(
        run_id,
        step_id=step_id,
        reason="interrupted external-side-effect node",
    )

    with sqlite3.connect(ledger.path) as conn:
        status = conn.execute(
            "SELECT status FROM research_steps WHERE step_id = ?", (step_id,)
        ).fetchone()[0]
    assert status == "manual_review"


def test_incompatible_checkpoint_moves_the_cycle_to_manual_review(tmp_path):
    ledger, run_id, transitions, step_id, attempt_id = _started_cycle(tmp_path)
    checkpoint = WorkflowCheckpointService(
        ledger=ledger,
        run_id=run_id,
        step_id=step_id,
        attempt_id=attempt_id,
        worker_id="worker-a",
        manifest=_manifest(model="model-a"),
        lease_seconds=60,
    ).capture(
        ctx=_FakeContext({"globals": {"cursor": 1}}),
        workflow=_FakeWorkflow(),
        step_name="retrieve",
    )
    with ledger.transaction() as conn:
        conn.execute("UPDATE research_steps SET lease_expires_at = 0 WHERE step_id = ?", (step_id,))
    transitions.reconcile_incomplete_cycles(run_id)

    incompatible = WorkflowCheckpointService(
        ledger=ledger,
        run_id=run_id,
        step_id=step_id,
        attempt_id=attempt_id,
        worker_id="worker-b",
        manifest=_manifest(model="model-b"),
        lease_seconds=60,
        checkpoint=checkpoint,
    )
    with pytest.raises(CheckpointCompatibilityError, match="manifest"):
        incompatible.validate_checkpoint()
    transitions.mark_manual_review(
        run_id,
        step_id=step_id,
        reason="checkpoint manifest is incompatible",
    )

    with sqlite3.connect(ledger.path) as conn:
        status = conn.execute(
            "SELECT status FROM research_steps WHERE step_id = ?", (step_id,)
        ).fetchone()[0]
    assert status == "manual_review"
    assert ledger.events(run_id)[-1].event_type == "cycle_manual_review"


def test_workflow_checkpoint_state_round_trips_discussion_and_queue():
    workflow = object.__new__(ResearchLoopWorkflow)
    workflow._board = MessageBoard()  # noqa: SLF001 - exercise the checkpoint boundary
    workflow._queue = ResearchQueue()  # noqa: SLF001 - exercise the checkpoint boundary
    post_id = workflow._board.post(  # noqa: SLF001
        POST_PROPOSAL, "analyst", "checkpointed hypothesis"
    )
    workflow._board.comment(post_id, "critic", "needs evidence", score=0.4)  # noqa: SLF001
    workflow._queue.add(  # noqa: SLF001
        QueueItem(
            id="q-1",
            statement="checkpointed hypothesis",
            proposed_by="analyst",
            hypothesis=Hypothesis(
                statement="checkpointed hypothesis",
                prediction="checkpoint preserves the work",
                status="critiqued",
            ),
        )
    )

    snapshot = workflow.checkpoint_state()
    restored = object.__new__(ResearchLoopWorkflow)
    restored.restore_checkpoint_state(snapshot)

    assert restored._board.to_dict() == workflow._board.to_dict()  # noqa: SLF001
    assert restored._queue.to_dict() == workflow._queue.to_dict()  # noqa: SLF001
    assert restored._queue.list_pending()[0].hypothesis.status == "critiqued"  # noqa: SLF001
