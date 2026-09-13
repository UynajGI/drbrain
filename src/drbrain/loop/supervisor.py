"""Adaptive autoresearch orchestration around the existing workflow.

The supervisor owns branch selection and authoritative event commits.  A
worker is deliberately a small callable so the current LlamaIndex workflow,
an in-process fake, or a remote runner can all be used without changing the
frontier protocol.
"""

from __future__ import annotations

import asyncio
import inspect
import uuid
from collections.abc import Awaitable, Callable, Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from drbrain.loop.frontier import (
    BranchOutcome,
    BranchSpec,
    BranchStatus,
    DoneContract,
    ResearchFrontier,
    ResearchObjective,
)
from drbrain.loop.governance import RunGovernance
from drbrain.loop.research_events import (
    EventEnvelope,
    EventLog,
    EventLogSnapshotStore,
    EventSourcedState,
    RunLedgerEventLog,
    SnapshotStore,
)
from drbrain.loop.store import RunExecutionBlockedError
from drbrain.loop.transitions import LeaseUnavailableError, TransitionService


class BranchWorker(Protocol):
    """Execute one branch without mutating supervisor state."""

    def __call__(
        self,
        branch: BranchSpec,
        objective: ResearchObjective,
        done_contract: DoneContract,
    ) -> BranchOutcome | Awaitable[BranchOutcome]: ...


class SupervisorError(RuntimeError):
    """Base error raised by the adaptive orchestration layer."""


class WorkerContractError(SupervisorError):
    """Raised when a worker returns an invalid outcome."""


class WorkflowBranchWorker:
    """Adapt a configured :class:`ResearchLoopWorkflow` factory to BranchWorker.

    The factory receives the branch and objective so hosts can inject the same
    tool-space, checkpoint and RAG-generation policy used by the legacy
    director without making this module depend on LlamaIndex internals.
    """

    def __init__(self, workflow_factory: Callable[..., Any]) -> None:
        self.workflow_factory = workflow_factory

    async def __call__(
        self,
        branch: BranchSpec,
        objective: ResearchObjective,
        done_contract: DoneContract,
    ) -> BranchOutcome:
        workflow = self.workflow_factory(branch, objective, done_contract)
        handler = workflow.run(
            task=branch.hypothesis or objective.question,
            prior_context=branch.rationale,
            prior_champion=[],
            prior_rejected=[],
        )
        report = await handler
        state = await handler.ctx.store.get("research_state", default=None)
        verified = list(getattr(state, "verified", []) or [])
        falsified = list(getattr(state, "falsified", []) or [])
        evidence = []
        for item in list(getattr(state, "evidence", []) or []):
            evidence_id = str(getattr(item, "evidence_id", "") or "")
            if evidence_id:
                from drbrain.loop.frontier import EvidenceRef

                evidence.append(
                    EvidenceRef(
                        evidence_id=evidence_id,
                        source=str(getattr(item, "source", "") or ""),
                        locator=dict(getattr(item, "document_locator", {}) or {}),
                        checksum=str(getattr(item, "content_checksum", "") or ""),
                        relation="supports",
                    )
                )
        status = (
            BranchStatus.RETAINED
            if verified
            else BranchStatus.PRUNED
            if falsified
            else BranchStatus.VERIFYING
        )
        return BranchOutcome(
            branch_id=branch.branch_id,
            status=status,
            claims=verified or falsified,
            evidence=evidence,
            verification={"verified": verified, "falsified": falsified},
            summary=str(report or ""),
        )


@dataclass(frozen=True)
class SupervisorConfig:
    """Operational limits for one supervisor run."""

    max_parallel_branches: int = 2
    max_evaluations: int = 10
    exploration_budget_fraction: float = 0.6
    snapshot_every: int = 1
    max_retries: int = 1
    human_mode: str = "approve_expensive"

    def __post_init__(self) -> None:
        if self.max_parallel_branches < 1:
            raise ValueError("max_parallel_branches must be positive")
        if self.max_evaluations < 1:
            raise ValueError("max_evaluations must be positive")
        if not 0.0 < self.exploration_budget_fraction < 1.0:
            raise ValueError("exploration_budget_fraction must be between 0 and 1")
        if self.snapshot_every < 1:
            raise ValueError("snapshot_every must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if self.human_mode not in {"autonomous", "approve_expensive", "guided", "review_only"}:
            raise ValueError(f"unsupported human_mode: {self.human_mode!r}")


@dataclass
class SupervisorResult:
    """Materialized result of a run; the event stream remains authoritative."""

    run_id: str
    objective: ResearchObjective
    done_contract: DoneContract
    frontier: ResearchFrontier
    events: list[EventEnvelope] = field(default_factory=list)
    completed: bool = False
    reason: str = ""


def _default_reducer(state: MutableMapping[str, Any], event: EventEnvelope) -> Mapping[str, Any]:
    """Build a replayable supervisor projection, including the frontier.

    EventSourcedState uses this projection's serialized frontier in snapshots,
    then replays only events appended after the snapshot.
    """
    state = dict(state)
    state["last_event"] = event.event_type
    state["last_seq"] = event.seq
    state.setdefault("event_types", []).append(event.event_type)
    frontier = ResearchFrontier.from_dict(state.get("frontier", {}))
    if event.event_type == "run_created":
        objective = event.payload.get("objective")
        done_contract = event.payload.get("done_contract")
        if isinstance(objective, Mapping):
            state["objective"] = dict(objective)
        if isinstance(done_contract, Mapping):
            state["done_contract"] = dict(done_contract)
    elif event.event_type == "branch_started":
        raw_branch = event.payload.get("branch")
        if isinstance(raw_branch, Mapping):
            branch = BranchSpec.from_dict(raw_branch)
            branch.status = BranchStatus.RUNNING
            existing = frontier.branches.get(branch.branch_id)
            if existing is None:
                frontier.add(branch)
            elif existing.to_dict() != branch.to_dict():
                raise SupervisorError(
                    f"branch_started conflicts with existing branch {branch.branch_id!r}"
                )
    elif event.event_type == "branch_completed":
        raw_outcome = event.payload.get("outcome")
        if isinstance(raw_outcome, Mapping):
            outcome = BranchOutcome.from_dict(raw_outcome)
            completed_branch = frontier.branches.get(outcome.branch_id)
            if completed_branch is None:
                raise SupervisorError(
                    f"branch_completed references unknown branch {outcome.branch_id!r}"
                )
            completed_branch.metadata["supported_evidence"] = [
                item.evidence_id
                for item in outcome.evidence
                if item.relation == "supports" and item.evidence_id
            ]
            frontier.apply_outcome(outcome)
    elif event.event_type == "run_completed":
        state["completed"] = bool(event.payload.get("completed", False))
        state["reason"] = str(event.payload.get("reason", ""))
    state["frontier"] = frontier.to_dict()
    return state


class ResearchSupervisor:
    """Schedule bounded branch workers and commit their outcomes in order."""

    def __init__(
        self,
        *,
        objective: ResearchObjective,
        done_contract: DoneContract | None = None,
        worker: BranchWorker | Callable[..., Any] | None = None,
        frontier: ResearchFrontier | None = None,
        event_log: EventLog | None = None,
        state: EventSourcedState | None = None,
        snapshots: SnapshotStore | None = None,
        config: SupervisorConfig | None = None,
        run_id: str | None = None,
        governance: RunGovernance | None = None,
        worker_id: str | None = None,
        lease_seconds: float = 900.0,
    ) -> None:
        self.objective = objective
        self.done_contract = done_contract or DoneContract()
        self.worker = worker
        self.frontier = frontier or ResearchFrontier()
        self.event_log = event_log
        self.config = config or SupervisorConfig()
        self.run_id = run_id or str(uuid.uuid4())
        self.governance = governance
        self.worker_id = worker_id or uuid.uuid4().hex
        self.lease_seconds = max(1.0, float(lease_seconds))
        self.transitions = TransitionService(governance._ledger) if governance is not None else None
        # A ledger-backed event log gets durable snapshots by default.  Pure
        # in-memory logs keep the previous opt-in behaviour so synthetic runs
        # do not acquire persistence marker events unexpectedly.
        if snapshots is None and isinstance(event_log, RunLedgerEventLog):
            snapshots = EventLogSnapshotStore(event_log)
        self.state = state or (
            EventSourcedState(event_log, reducer=_default_reducer, snapshots=snapshots)
            if event_log is not None
            else None
        )
        self._events: list[EventEnvelope] = []
        if self.event_log is not None and self.event_log.latest_seq(self.run_id) > 0:
            self.restore()

    def _ledger_attempt_id(self, step_id: str) -> str | None:
        """Return the active attempt for a newly leased branch step."""
        ledger = getattr(self.governance, "_ledger", None)
        if ledger is None:
            return None
        return ledger.active_attempt_id(step_id)

    @classmethod
    def from_event_log(
        cls,
        event_log: EventLog,
        *,
        run_id: str,
        objective: ResearchObjective | None = None,
        done_contract: DoneContract | None = None,
        worker: BranchWorker | Callable[..., Any] | None = None,
        snapshots: SnapshotStore | None = None,
        config: SupervisorConfig | None = None,
    ) -> ResearchSupervisor:
        """Construct a supervisor whose state is restored from ``event_log``."""
        return cls(
            objective=objective or ResearchObjective(),
            done_contract=done_contract,
            worker=worker,
            event_log=event_log,
            snapshots=snapshots,
            config=config,
            run_id=run_id,
        )

    def restore(self) -> ResearchFrontier:
        """Restore the canonical frontier from snapshots and event tail."""
        if self.event_log is None:
            return self.frontier
        restored: Mapping[str, Any] = {}
        if self.state is not None:
            restored = self.state.replay(self.run_id)
        raw_frontier = restored.get("frontier") if isinstance(restored, Mapping) else None
        if not isinstance(raw_frontier, Mapping):
            # Custom reducers may not project frontier state; use the canonical
            # reducer over the complete stream as a compatibility fallback.
            state: MutableMapping[str, Any] = {}
            for event in self.event_log.read(self.run_id):
                state = dict(_default_reducer(state, event))
            restored = state
            raw_frontier = state.get("frontier")
        if isinstance(raw_frontier, Mapping):
            self.frontier = ResearchFrontier.from_dict(raw_frontier)
        objective = restored.get("objective") if isinstance(restored, Mapping) else None
        if isinstance(objective, Mapping):
            self.objective = ResearchObjective.from_dict(objective)
        done_contract = restored.get("done_contract") if isinstance(restored, Mapping) else None
        if isinstance(done_contract, Mapping):
            self.done_contract = DoneContract.from_dict(done_contract)
        self._events = list(self.event_log.read(self.run_id))
        return self.frontier

    restore_frontier = restore
    replay = restore

    def seed(self, branches: list[BranchSpec]) -> None:
        """Add initial branches before a run starts."""
        for branch in branches:
            if branch.branch_id in self.frontier.branches:
                # Resuming a run may pass the original seed list again.  The
                # event-replayed branch is authoritative, including its status.
                continue
            self.frontier.add(branch)

    def approve_branch(self, branch_id: str, *, actor: str = "operator", reason: str = "") -> None:
        """Release one expensive branch from the human approval gate."""
        branch = self.frontier.branches.get(branch_id)
        if branch is None:
            raise KeyError(f"unknown branch {branch_id!r}")
        self.frontier.transition(branch_id, BranchStatus.SCREENED)
        branch.metadata["approval"] = {"actor": actor, "reason": reason}
        branch.metadata["human_decision"] = {
            "actor": actor,
            "decision": "approved",
            "reason": reason,
        }
        self._append(
            "human_decision",
            {"branch_id": branch_id, "decision": "approved", "actor": actor, "reason": reason},
            key=f"branch:{branch_id}:approval",
        )

    def _append(
        self, event_type: str, payload: Mapping[str, Any], *, key: str
    ) -> EventEnvelope | None:
        if self.event_log is None:
            return None
        event = self.event_log.append(
            run_id=self.run_id,
            event_type=event_type,
            actor="supervisor",
            payload=payload,
            idempotency_key=key,
        )
        if not any(item.event_id == event.event_id for item in self._events):
            self._events.append(event)
        return event

    async def _execute(self, branch: BranchSpec) -> BranchOutcome:
        if self.worker is None:
            raise SupervisorError("a BranchWorker is required")
        result = self.worker(branch, self.objective, self.done_contract)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, BranchOutcome):
            raise WorkerContractError(
                f"worker for {branch.branch_id!r} returned {type(result).__name__}, "
                "expected BranchOutcome"
            )
        if result.branch_id != branch.branch_id:
            raise WorkerContractError(
                f"worker returned branch {result.branch_id!r} for {branch.branch_id!r}"
            )
        return result

    async def _execute_with_retry(self, branch: BranchSpec) -> BranchOutcome:
        """Retry a failed worker without duplicating the authoritative outcome."""
        for attempt in range(self.config.max_retries + 1):
            try:
                return await self._execute(branch)
            except SupervisorError:
                if attempt >= self.config.max_retries:
                    raise
                self._append(
                    "branch_retry",
                    {"branch_id": branch.branch_id, "attempt": attempt + 2},
                    key=f"branch:{branch.branch_id}:retry:{attempt + 1}",
                )
        raise AssertionError("unreachable")

    async def run(self, branches: list[BranchSpec] | None = None) -> SupervisorResult:
        """Run until the done contract, frontier exhaustion, or budget limit."""
        if self.worker is None:
            raise SupervisorError("a BranchWorker is required")
        if branches:
            self.seed(branches)
        self._append(
            "run_created",
            {
                "objective": self.objective.to_dict(),
                "done_contract": self.done_contract.to_dict(),
                "config": {
                    "max_parallel_branches": self.config.max_parallel_branches,
                    "max_evaluations": self.config.max_evaluations,
                },
            },
            key=f"run:{self.run_id}:created",
        )
        reason = "frontier_exhausted"
        completed = False
        while (
            self.frontier.pending()
            and self.frontier.completed_evaluations < self.config.max_evaluations
        ):
            if self.governance is not None:
                run_status = str(self.governance.status(self.run_id).get("status", ""))
                if run_status in {"cancelled", "paused", "failed"}:
                    reason = f"run_{run_status}"
                    break
            # The legacy transition service protects one active cycle lease per
            # run. Until branch-scoped leases land there, governance-backed
            # runs intentionally serialize while the in-memory adapter may
            # still use bounded parallelism.
            selection_limit = (
                1 if self.governance is not None else self.config.max_parallel_branches
            )
            selected = self.frontier.select_next(limit=selection_limit)
            if not selected:
                reason = "no_available_slots"
                break
            step_ids: dict[str, str] = {}
            transitions = self.transitions
            runnable: list[BranchSpec] = []
            for branch in selected:
                if (
                    self.config.human_mode == "approve_expensive"
                    and branch.metadata.get("expensive")
                    and not branch.metadata.get("approval")
                ):
                    self.frontier.transition(branch.branch_id, BranchStatus.NEEDS_HUMAN)
                    self._append(
                        "branch_needs_human",
                        {
                            "branch_id": branch.branch_id,
                            "reason": "expensive tool approval required",
                        },
                        key=f"branch:{branch.branch_id}:needs-human",
                    )
                    continue
                if self.governance is not None:
                    try:
                        self.governance.reserve(self.run_id, branch.budget or {"attempts": 1})
                        if transitions is None:
                            raise SupervisorError("governance requires a transition service")
                        step_ids[branch.branch_id] = transitions.begin_cycle(
                            self.run_id,
                            cycle=self.frontier.completed_evaluations + 1,
                            worker_id=self.worker_id,
                            lease_seconds=self.lease_seconds,
                        )
                        # Make the durable attempt address available to the
                        # branch adapter.  The metadata is part of the
                        # branch-start event, so a resumed worker can recreate
                        # its broker/checkpoint without hidden process state.
                        branch.metadata["_step_id"] = step_ids[branch.branch_id]
                        attempt_id = self._ledger_attempt_id(step_ids[branch.branch_id])
                        if attempt_id:
                            branch.metadata["_attempt_id"] = attempt_id
                    except (RunExecutionBlockedError, LeaseUnavailableError) as exc:
                        self.frontier.transition(branch.branch_id, BranchStatus.NEEDS_HUMAN)
                        self._append(
                            "branch_blocked",
                            {"branch_id": branch.branch_id, "reason": str(exc)},
                            key=f"branch:{branch.branch_id}:blocked",
                        )
                        continue
                runnable.append(branch)
                self._append(
                    "branch_started",
                    {"branch": branch.to_dict()},
                    key=f"branch:{branch.branch_id}:started",
                )
            if not runnable:
                continue
            outcomes = await asyncio.gather(
                *(self._execute_with_retry(branch) for branch in runnable), return_exceptions=True
            )
            for branch, outcome in zip(runnable, outcomes, strict=True):
                if isinstance(outcome, BaseException):
                    if isinstance(outcome, WorkerContractError):
                        raise outcome
                    outcome = BranchOutcome(
                        branch_id=branch.branch_id,
                        status=BranchStatus.NEEDS_HUMAN,
                        warnings=[f"worker failed: {outcome!r}"],
                    )
                lease_failed = False
                if self.governance is not None and transitions is not None:
                    # The existing transition service permits one active cycle
                    # per run. Governance-backed supervisors therefore run
                    # serially and close the lease before the next branch.
                    step_id = step_ids.get(branch.branch_id)
                    if step_id:
                        try:
                            transitions.complete_cycle(
                                self.run_id,
                                step_id=step_id,
                                cycle_result=outcome.to_dict(),
                                state_snapshot=self.frontier.to_dict(),
                                research_state=None,
                                worker_id=self.worker_id,
                            )
                        except Exception as exc:  # noqa: BLE001 - preserve worker outcome
                            lease_failed = True
                            self._append(
                                "lease_close_failed",
                                {"branch_id": branch.branch_id, "error": str(exc)},
                                key=f"branch:{branch.branch_id}:lease-close",
                            )
                    current_status = str(self.governance.status(self.run_id).get("status", ""))
                    if current_status in {"cancelled", "paused", "failed"}:
                        lease_failed = True
                if lease_failed:
                    outcome = BranchOutcome(
                        branch_id=branch.branch_id,
                        status=BranchStatus.NEEDS_HUMAN,
                        warnings=[
                            *outcome.warnings,
                            "governance lease was revoked before completion",
                        ],
                    )
                branch.metadata["supported_evidence"] = [
                    item.evidence_id
                    for item in outcome.evidence
                    if item.relation == "supports" and item.evidence_id
                ]
                branch.metadata["verification"] = dict(outcome.verification)
                branch.metadata["claims"] = list(outcome.claims)
                admitted = self.frontier.apply_outcome(outcome)
                self._append(
                    "branch_completed",
                    {
                        "branch_id": branch.branch_id,
                        "outcome": outcome.to_dict(),
                        "admitted_children": [child.to_dict() for child in admitted],
                    },
                    key=f"branch:{branch.branch_id}:completed",
                )
            if (
                self.state is not None
                and self.frontier.completed_evaluations % self.config.snapshot_every == 0
            ):
                self.state.snapshot(self.run_id)
            if self._objective_complete():
                completed = True
                reason = "done_contract"
                break
        if not completed and self.frontier.completed_evaluations >= self.config.max_evaluations:
            reason = "evaluation_budget_exhausted"
        self._append(
            "run_completed",
            {"completed": completed, "reason": reason},
            key=f"run:{self.run_id}:completed",
        )
        return SupervisorResult(
            run_id=self.run_id,
            objective=self.objective,
            done_contract=self.done_contract,
            frontier=self.frontier,
            events=list(self._events),
            completed=completed,
            reason=reason,
        )

    def _objective_complete(self) -> bool:
        supported = sum(
            len(branch.metadata.get("supported_evidence", []))
            for branch in self.frontier.branches.values()
            if branch.status == BranchStatus.RETAINED
        )
        retained = [b for b in self.frontier.branches.values() if b.status == BranchStatus.RETAINED]
        experiments = sum(1 for b in retained if b.experiment is not None)
        if not (
            supported >= self.done_contract.min_supported_evidence
            and experiments >= self.done_contract.required_experiments
        ):
            return False
        for branch in retained:
            verification = branch.metadata.get("verification", {})
            if not isinstance(verification, Mapping):
                verification = {}
            if self.done_contract.require_replication:
                repetitions = branch.experiment.repetitions if branch.experiment else 2
                job_ids = verification.get("job_ids", ())
                replicated = bool(verification.get("replicated"))
                if isinstance(job_ids, str):
                    job_ids = [job_ids]
                if not replicated and len(set(job_ids or ())) < repetitions:
                    return False
            if (
                self.done_contract.require_counter_evidence_search
                and (
                    branch.experiment is not None
                    or any(
                        key in verification
                        for key in (
                            "counter_evidence_searched",
                            "job_ids",
                            "replicated",
                            "confidence",
                            "stop_conditions",
                        )
                    )
                )
                and not bool(verification.get("counter_evidence_searched", False))
            ):
                return False
            if (
                float(verification.get("confidence", branch.metadata.get("confidence", 0.0)) or 0.0)
                < self.done_contract.min_confidence
            ):
                return False
            if self.done_contract.human_review and not branch.metadata.get("human_decision"):
                return False
            required_stops = self.done_contract.stop_conditions
            if required_stops:
                observed = set(verification.get("stop_conditions", ()) or ())
                observed.update(branch.metadata.get("stop_conditions", ()) or ())
                if not set(required_stops).issubset(observed):
                    return False
        return True


__all__ = [
    "BranchWorker",
    "WorkflowBranchWorker",
    "ResearchSupervisor",
    "SupervisorConfig",
    "SupervisorError",
    "SupervisorResult",
    "WorkerContractError",
]
