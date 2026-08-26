"""The sole state-writing service for durable autoresearch runs."""

from __future__ import annotations

import json
import math
import time
import uuid
from collections.abc import Mapping
from typing import Any

from drbrain.loop.state import (
    RUN_CANCELLED,
    RUN_FAILED,
    RUN_PAUSED,
    RUN_RUNNING,
    STEP_CLAIMED,
    STEP_FAILED,
    STEP_MANUAL_REVIEW,
    STEP_PENDING,
    STEP_READY,
    STEP_RECONCILING,
    STEP_RUNNING,
    STEP_SUCCEEDED,
    STEP_UNKNOWN,
    validate_run_transition,
    validate_step_transition,
)
from drbrain.loop.store import LedgerEvent, RunExecutionBlockedError, RunLedger


class LeaseUnavailableError(RuntimeError):
    """Raised when another worker still owns an in-flight cycle lease."""


class ChampionVersionConflictError(RuntimeError):
    """Raised when a champion promotion loses its expected-version CAS race."""


class TransitionService:
    """Apply validated lifecycle changes and append their audit events atomically."""

    def __init__(self, ledger: RunLedger) -> None:
        self._ledger = ledger

    def start_run(self, run_id: str) -> None:
        """Start a newly created or paused run; a running run is resumable."""
        with self._ledger.transaction() as conn:
            current = self._run_status(conn, run_id)
            if current == RUN_RUNNING:
                return
            validate_run_transition(current, RUN_RUNNING)
            self._set_run_status(conn, run_id, RUN_RUNNING)
            self._ledger.append_event(
                conn, run_id, actor="director", event_type="run_started", payload={"from": current}
            )

    def pause_run(self, run_id: str, *, reason: str) -> None:
        """Persist a bounded director session as a resumable run."""
        with self._ledger.transaction() as conn:
            current = self._run_status(conn, run_id)
            if current == RUN_PAUSED:
                return
            validate_run_transition(current, RUN_PAUSED)
            self._set_run_status(conn, run_id, RUN_PAUSED)
            self._ledger.append_event(
                conn, run_id, actor="director", event_type="run_paused", payload={"reason": reason}
            )

    def cancel_run(self, run_id: str, *, reason: str, actor: str = "operator") -> None:
        """Terminally stop a run and revoke every active cycle lease.

        Cancellation cannot pre-empt arbitrary Python code already executing in
        a worker. Revoking its lease prevents that worker from recording any
        further intent, observation, checkpoint, or cycle completion.
        """
        with self._ledger.transaction() as conn:
            current = self._run_status(conn, run_id)
            if current == RUN_CANCELLED:
                return
            validate_run_transition(current, RUN_CANCELLED)
            rows = conn.execute(
                """
                SELECT step_id, status FROM research_steps
                WHERE run_id = ? AND status IN (?, ?)
                """,
                (run_id, STEP_RUNNING, STEP_RECONCILING),
            ).fetchall()
            now = time.time()
            cancelled_steps: list[str] = []
            for row in rows:
                step_id = str(row["step_id"])
                status = str(row["status"])
                self._transition_step(conn, step_id, STEP_UNKNOWN)
                conn.execute(
                    """
                    UPDATE research_attempts
                    SET status = ?, completed_at = ?
                    WHERE step_id = ? AND status = ?
                    """,
                    (STEP_UNKNOWN, now, step_id, STEP_RUNNING),
                )
                conn.execute(
                    """
                    UPDATE research_steps
                    SET lease_owner = NULL, lease_expires_at = NULL, updated_at = ?
                    WHERE step_id = ?
                    """,
                    (now, step_id),
                )
                self._ledger.append_event(
                    conn,
                    run_id,
                    actor=actor,
                    event_type="cycle_cancelled",
                    payload={"step_id": step_id, "from": status, "reason": reason},
                )
                cancelled_steps.append(step_id)
            self._set_run_status(conn, run_id, RUN_CANCELLED, completed=True)
            self._ledger.append_event(
                conn,
                run_id,
                actor=actor,
                event_type="run_cancelled",
                payload={"reason": reason, "cancelled_step_ids": cancelled_steps},
            )

    def fail_run(self, run_id: str, *, error: BaseException) -> None:
        """Record a workflow failure without swallowing the original exception."""
        with self._ledger.transaction() as conn:
            current = self._run_status(conn, run_id)
            if current == RUN_FAILED:
                return
            validate_run_transition(current, RUN_FAILED)
            self._set_run_status(conn, run_id, RUN_FAILED, completed=True)
            self._ledger.append_event(
                conn,
                run_id,
                actor="director",
                event_type="run_failed",
                payload={"error_type": type(error).__name__, "message": str(error)},
            )

    def reconcile_incomplete_cycles(self, run_id: str) -> None:
        """Reclaim only expired in-flight leases for checkpoint recovery.

        A live worker is never pre-empted.  An absent lease is treated as a
        PR1-compatible interrupted cycle, while an expired lease leaves a clear
        audit trail and lets exactly one later worker create a resumed attempt.
        """
        with self._ledger.transaction() as conn:
            now = time.time()
            rows = conn.execute(
                """
                SELECT step_id, status FROM research_steps
                WHERE run_id = ?
                  AND status IN (?, ?)
                  AND (lease_owner IS NULL OR lease_expires_at IS NULL OR lease_expires_at <= ?)
                """,
                (run_id, STEP_RUNNING, STEP_RECONCILING, now),
            ).fetchall()
            for row in rows:
                step_id = str(row["step_id"])
                status = str(row["status"])
                if status == STEP_RUNNING:
                    validate_step_transition(status, STEP_UNKNOWN)
                    self._set_step_status(conn, step_id, STEP_UNKNOWN)
                conn.execute(
                    """
                    UPDATE research_attempts
                    SET status = ?, completed_at = ?
                    WHERE step_id = ? AND status = ?
                    """,
                    (STEP_UNKNOWN, now, step_id, STEP_RUNNING),
                )
                conn.execute(
                    """
                    UPDATE research_steps
                    SET lease_owner = NULL, lease_expires_at = NULL, updated_at = ?
                    WHERE step_id = ?
                    """,
                    (now, step_id),
                )
                self._ledger.append_event(
                    conn,
                    run_id,
                    actor="recovery",
                    event_type=(
                        "cycle_interrupted"
                        if status == STEP_RUNNING
                        else "cycle_resume_interrupted"
                    ),
                    payload={"step_id": step_id},
                )

    def begin_cycle(
        self,
        run_id: str,
        *,
        cycle: int,
        worker_id: str | None = None,
        lease_seconds: float | None = None,
    ) -> str:
        """Create and start one durable cycle step with its first attempt."""
        with self._ledger.transaction() as conn:
            if self._run_status(conn, run_id) != RUN_RUNNING:
                raise RuntimeError("cannot begin a cycle while the run is not running")
            now = time.time()
            if worker_id is not None:
                active = conn.execute(
                    """
                    SELECT step_id FROM research_steps
                    WHERE run_id = ?
                      AND status IN (?, ?)
                      AND lease_owner IS NOT NULL
                      AND lease_expires_at > ?
                    LIMIT 1
                    """,
                    (run_id, STEP_RUNNING, STEP_RECONCILING, now),
                ).fetchone()
                if active is not None:
                    raise LeaseUnavailableError("a cycle already holds an active lease")
            step_id = uuid.uuid4().hex
            lease_expires_at = (
                now + max(1.0, float(lease_seconds or 0.0)) if worker_id is not None else None
            )
            conn.execute(
                """
                INSERT INTO research_steps(
                    step_id, run_id, step_name, status, lease_owner, lease_expires_at,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    step_id,
                    run_id,
                    f"cycle:{cycle}",
                    STEP_PENDING,
                    worker_id,
                    lease_expires_at,
                    now,
                    now,
                ),
            )
            self._transition_step(conn, step_id, STEP_READY)
            self._transition_step(conn, step_id, STEP_CLAIMED)
            self._transition_step(conn, step_id, STEP_RUNNING)
            attempt_id = uuid.uuid4().hex
            conn.execute(
                """
                INSERT INTO research_attempts(
                    attempt_id, step_id, attempt_no, status, started_at
                ) VALUES (?, ?, 1, ?, ?)
                """,
                (attempt_id, step_id, STEP_RUNNING, now),
            )
            self._ledger.append_event(
                conn,
                run_id,
                actor="director",
                event_type="cycle_started",
                payload={
                    "cycle": cycle,
                    "step_id": step_id,
                    "attempt_id": attempt_id,
                    "worker_id": worker_id,
                },
            )
            return step_id

    def complete_cycle(
        self,
        run_id: str,
        *,
        step_id: str,
        cycle_result: Mapping[str, Any],
        state_snapshot: Mapping[str, Any],
        research_state: Mapping[str, Any] | None,
        checkpoint_id: str | None = None,
        worker_id: str | None = None,
    ) -> LedgerEvent:
        """Commit a completed cycle and its replayable compatibility snapshot."""
        with self._ledger.transaction() as conn:
            status = self._run_status(conn, run_id)
            if status not in {RUN_RUNNING, RUN_PAUSED}:
                raise RunExecutionBlockedError(
                    f"cannot complete cycle while run {run_id} is {status!r}"
                )
            self._require_lease(conn, run_id=run_id, step_id=step_id, worker_id=worker_id)
            self._transition_step(conn, step_id, STEP_SUCCEEDED)
            conn.execute(
                """
                UPDATE research_attempts
                SET status = ?, completed_at = ?
                WHERE step_id = ? AND status = ?
                """,
                (STEP_SUCCEEDED, time.time(), step_id, STEP_RUNNING),
            )
            conn.execute(
                """
                UPDATE research_steps
                SET lease_owner = NULL, lease_expires_at = NULL, updated_at = ?
                WHERE step_id = ?
                """,
                (time.time(), step_id),
            )
            return self._ledger.append_event(
                conn,
                run_id,
                actor="director",
                event_type="cycle_completed",
                payload={
                    "step_id": step_id,
                    "cycle_result": dict(cycle_result),
                    "state": dict(state_snapshot),
                    "research_state": dict(research_state) if research_state else None,
                    "checkpoint_id": checkpoint_id,
                },
            )

    def record_state_snapshot(
        self, run_id: str, *, reason: str, state_snapshot: Mapping[str, Any]
    ) -> LedgerEvent:
        """Commit a non-cycle state change before its legacy-file projection."""
        with self._ledger.transaction() as conn:
            return self._ledger.append_event(
                conn,
                run_id,
                actor="director",
                event_type="state_snapshot",
                payload={"reason": reason, "state": dict(state_snapshot)},
            )

    def fail_cycle(
        self,
        run_id: str,
        *,
        step_id: str,
        error: BaseException,
        worker_id: str | None = None,
    ) -> None:
        """Commit a failed workflow cycle before the exception propagates."""
        with self._ledger.transaction() as conn:
            self._require_lease(conn, run_id=run_id, step_id=step_id, worker_id=worker_id)
            self._transition_step(conn, step_id, STEP_FAILED)
            conn.execute(
                """
                UPDATE research_attempts
                SET status = ?, failure_category = ?, completed_at = ?
                WHERE step_id = ? AND status = ?
                """,
                (STEP_FAILED, type(error).__name__, time.time(), step_id, STEP_RUNNING),
            )
            conn.execute(
                """
                UPDATE research_steps
                SET lease_owner = NULL, lease_expires_at = NULL, updated_at = ?
                WHERE step_id = ?
                """,
                (time.time(), step_id),
            )
            self._ledger.append_event(
                conn,
                run_id,
                actor="director",
                event_type="cycle_failed",
                payload={
                    "step_id": step_id,
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            )

    def resume_cycle(
        self,
        run_id: str,
        *,
        step_id: str,
        checkpoint_id: str,
        worker_id: str,
        lease_seconds: float,
    ) -> str:
        """Claim an interrupted cycle and attach a fresh attempt to its checkpoint."""
        with self._ledger.transaction() as conn:
            if self._run_status(conn, run_id) != RUN_RUNNING:
                raise RuntimeError("cannot resume a cycle while the run is not running")
            row = conn.execute(
                "SELECT run_id, status FROM research_steps WHERE step_id = ?", (step_id,)
            ).fetchone()
            if row is None or str(row["run_id"]) != run_id:
                raise KeyError(f"unknown step {step_id!r} for run {run_id!r}")
            checkpoint = conn.execute(
                """
                SELECT checkpoint_id FROM research_checkpoints
                WHERE checkpoint_id = ? AND run_id = ? AND step_id = ?
                """,
                (checkpoint_id, run_id, step_id),
            ).fetchone()
            if checkpoint is None:
                raise KeyError(f"unknown checkpoint {checkpoint_id!r} for step {step_id!r}")
            status = str(row["status"])
            if status == STEP_UNKNOWN:
                self._transition_step(conn, step_id, STEP_RECONCILING)
            elif status != STEP_RECONCILING:
                raise RuntimeError(f"cannot resume cycle from step state {status!r}")

            now = time.time()
            active = conn.execute(
                """
                SELECT attempt_id FROM research_attempts
                WHERE step_id = ? AND status = ?
                LIMIT 1
                """,
                (step_id, STEP_RUNNING),
            ).fetchone()
            if active is not None:
                raise LeaseUnavailableError("a cycle already holds an active lease")
            attempt_no = int(
                conn.execute(
                    "SELECT COALESCE(MAX(attempt_no), 0) + 1 AS next_no "
                    "FROM research_attempts WHERE step_id = ?",
                    (step_id,),
                ).fetchone()["next_no"]
            )
            attempt_id = uuid.uuid4().hex
            conn.execute(
                """
                INSERT INTO research_attempts(
                    attempt_id, step_id, attempt_no, status, checkpoint_ref, started_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (attempt_id, step_id, attempt_no, STEP_RUNNING, checkpoint_id, now),
            )
            conn.execute(
                """
                UPDATE research_steps
                SET lease_owner = ?, lease_expires_at = ?, updated_at = ?
                WHERE step_id = ?
                """,
                (worker_id, now + max(1.0, float(lease_seconds)), now, step_id),
            )
            self._ledger.append_event(
                conn,
                run_id,
                actor="recovery",
                event_type="cycle_resumed",
                payload={
                    "step_id": step_id,
                    "attempt_id": attempt_id,
                    "checkpoint_id": checkpoint_id,
                    "worker_id": worker_id,
                },
            )
            return attempt_id

    def mark_manual_review(self, run_id: str, *, step_id: str, reason: str) -> None:
        """Stop an unsafe recovery without silently starting a fresh cycle."""
        with self._ledger.transaction() as conn:
            row = conn.execute(
                "SELECT run_id, status FROM research_steps WHERE step_id = ?", (step_id,)
            ).fetchone()
            if row is None or str(row["run_id"]) != run_id:
                raise KeyError(f"unknown step {step_id!r} for run {run_id!r}")
            status = str(row["status"])
            if status == STEP_RUNNING:
                self._transition_step(conn, step_id, STEP_UNKNOWN)
                status = STEP_UNKNOWN
            if status == STEP_UNKNOWN:
                self._transition_step(conn, step_id, STEP_RECONCILING)
                status = STEP_RECONCILING
            if status != STEP_RECONCILING:
                raise RuntimeError(f"cannot mark step {step_id!r} manual from {status!r}")
            self._transition_step(conn, step_id, STEP_MANUAL_REVIEW)
            now = time.time()
            conn.execute(
                """
                UPDATE research_attempts
                SET status = ?, completed_at = ?
                WHERE step_id = ? AND status = ?
                """,
                (STEP_UNKNOWN, now, step_id, STEP_RUNNING),
            )
            conn.execute(
                """
                UPDATE research_steps
                SET lease_owner = NULL, lease_expires_at = NULL, updated_at = ?
                WHERE step_id = ?
                """,
                (now, step_id),
            )
            self._ledger.append_event(
                conn,
                run_id,
                actor="recovery",
                event_type="cycle_manual_review",
                payload={"step_id": step_id, "reason": reason},
            )

    def register_front_half_node_contracts(
        self, run_id: str, node_specs: Mapping[str, Mapping[str, Any]]
    ) -> None:
        """Persist immutable node contracts for a run's durable front half."""
        with self._ledger.transaction() as conn:
            self._run_status(conn, run_id)
            inserted_nodes: list[str] = []
            for node_name, raw_spec in node_specs.items():
                spec = dict(raw_spec)
                max_attempts = int(spec.get("max_attempts", 0))
                retry_class = str(spec.get("retry_class", "")).strip()
                if not node_name or max_attempts < 1 or not retry_class:
                    raise ValueError("front-half node contract is incomplete")
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO research_front_half_node_specs(
                        run_id, node_name, input_schema_json, output_schema_json,
                        allowed_tools_json, max_attempts, retry_class, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        node_name,
                        json.dumps(spec.get("input_schema", {}), ensure_ascii=False),
                        json.dumps(spec.get("output_schema", {}), ensure_ascii=False),
                        json.dumps(list(spec.get("allowed_tools", [])), ensure_ascii=False),
                        max_attempts,
                        retry_class,
                        time.time(),
                    ),
                )
                if cursor.rowcount == 0:
                    row = conn.execute(
                        """
                        SELECT * FROM research_front_half_node_specs
                        WHERE run_id = ? AND node_name = ?
                        """,
                        (run_id, node_name),
                    ).fetchone()
                    if row is None or self._node_contract(
                        dict(spec)
                    ) != self._node_contract_from_row(row):
                        raise ValueError(
                            "front-half node contract conflicts with its existing record"
                        )
                if cursor.rowcount:
                    inserted_nodes.append(node_name)
            if inserted_nodes:
                self._ledger.append_event(
                    conn,
                    run_id,
                    actor="workflow",
                    event_type="front_half_contracts_registered",
                    payload={"nodes": sorted(inserted_nodes)},
                )

    def record_front_half_proposal(
        self,
        run_id: str,
        *,
        proposal_id: str,
        claim_id: str,
        author: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Record a proposal once; replay returns its canonical durable record."""
        if not proposal_id or not claim_id or not author:
            raise ValueError("proposal_id, claim_id and author are required")
        with self._ledger.transaction() as conn:
            self._run_status(conn, run_id)
            row = self._proposal_row(conn, proposal_id)
            if row is None:
                claim_row = conn.execute(
                    "SELECT proposal_id FROM research_proposals WHERE run_id = ? AND claim_id = ?",
                    (run_id, claim_id),
                ).fetchone()
                if claim_row is not None and str(claim_row["proposal_id"]) != proposal_id:
                    raise ValueError("claim_id already belongs to another durable proposal")
                now = time.time()
                conn.execute(
                    """
                    INSERT INTO research_proposals(
                        proposal_id, run_id, claim_id, author, payload_json, status,
                        review_score, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 'proposed', NULL, ?, ?)
                    """,
                    (
                        proposal_id,
                        run_id,
                        claim_id,
                        author,
                        json.dumps(dict(payload), ensure_ascii=False),
                        now,
                        now,
                    ),
                )
                self._ledger.append_event(
                    conn,
                    run_id,
                    actor=author,
                    event_type="proposal_recorded",
                    payload={"proposal_id": proposal_id, "claim_id": claim_id},
                )
                row = self._proposal_row(conn, proposal_id)
            elif str(row["author"]) != author or self._json_roundtrip(
                self._proposal_contract(dict(payload))
            ) != self._proposal_contract(self._json(row["payload_json"], {})):
                raise ValueError("durable proposal replay conflicts with its existing contract")
            if row is None or str(row["run_id"]) != run_id:
                raise KeyError(f"unknown proposal {proposal_id}")
            return self._proposal_dict(row)

    def record_front_half_review(
        self,
        run_id: str,
        *,
        proposal_id: str,
        review_id: str,
        reviewer: str,
        score: float,
        verdict: str,
        content: str,
    ) -> dict[str, Any]:
        """Record one non-author critic review idempotently."""
        with self._ledger.transaction() as conn:
            proposal = self._proposal_row(conn, proposal_id)
            if proposal is None or str(proposal["run_id"]) != run_id:
                raise KeyError(f"unknown proposal {proposal_id}")
            if reviewer == str(proposal["author"]):
                raise ValueError("a durable review must be from a non-author")
            row = conn.execute(
                "SELECT * FROM research_critic_reviews WHERE proposal_id = ? AND reviewer = ?",
                (proposal_id, reviewer),
            ).fetchone()
            if row is not None and str(row["review_id"]) != review_id:
                raise ValueError(
                    "reviewer already reviewed this proposal under a different review_id"
                )
            if row is not None and (
                not math.isclose(float(row["score"]), float(score), rel_tol=0.0, abs_tol=1e-12)
                or str(row["verdict"]) != verdict
                or str(row["content"]) != content
            ):
                self._ledger.append_event(
                    conn,
                    run_id,
                    actor=reviewer,
                    event_type="critic_review_replay_ignored",
                    payload={
                        "proposal_id": proposal_id,
                        "review_id": review_id,
                        "stored_score": float(row["score"]),
                        "replayed_score": float(score),
                    },
                )
            if row is None:
                now = time.time()
                conn.execute(
                    """
                    INSERT INTO research_critic_reviews(
                        review_id, proposal_id, reviewer, score, verdict, content, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (review_id, proposal_id, reviewer, float(score), verdict, content, now),
                )
                self._ledger.append_event(
                    conn,
                    run_id,
                    actor=reviewer,
                    event_type="critic_review_recorded",
                    payload={
                        "proposal_id": proposal_id,
                        "review_id": review_id,
                        "verdict": verdict,
                    },
                )
                row = conn.execute(
                    "SELECT * FROM research_critic_reviews WHERE review_id = ?", (review_id,)
                ).fetchone()
            if row is None:
                raise RuntimeError("durable critic review was not persisted")
            return self._review_dict(row)

    def settle_front_half_proposal(
        self,
        run_id: str,
        *,
        proposal_id: str,
        queue_item_id: str,
        discard_score: float,
    ) -> dict[str, Any]:
        """Atomically enforce the non-author gate and materialize one queue item."""
        with self._ledger.transaction() as conn:
            proposal = self._proposal_row(conn, proposal_id)
            if proposal is None or str(proposal["run_id"]) != run_id:
                raise KeyError(f"unknown proposal {proposal_id}")
            reviews = conn.execute(
                "SELECT * FROM research_critic_reviews WHERE proposal_id = ? ORDER BY created_at, review_id",
                (proposal_id,),
            ).fetchall()
            scores = [float(review["score"]) for review in reviews]
            if str(proposal["status"]) == "discarded":
                status, queue_status = "discarded", "discarded"
                score = float(proposal["review_score"] or 0.0)
            elif not reviews:
                status, queue_status, score = "discussion_pending", "pending_review", 0.0
            else:
                score = round(sum(scores) / len(scores), 4)
                all_discard = all(str(review["verdict"]) == "DISCARD" for review in reviews)
                if score < discard_score or all_discard:
                    status, queue_status = "discarded", "discarded"
                else:
                    status, queue_status = "critiqued", "ready"
            previous_status = str(proposal["status"])
            stored_score = proposal["review_score"]
            score_changed = stored_score is None or not math.isclose(
                float(stored_score), score, rel_tol=0.0, abs_tol=1e-12
            )
            if previous_status != status or score_changed:
                conn.execute(
                    """
                    UPDATE research_proposals
                    SET status = ?, review_score = ?, updated_at = ?
                    WHERE proposal_id = ?
                    """,
                    (status, score, time.time(), proposal_id),
                )
                self._ledger.append_event(
                    conn,
                    run_id,
                    actor="transition",
                    event_type=f"proposal_{status}",
                    payload={"proposal_id": proposal_id, "score": score},
                )
            queue = conn.execute(
                "SELECT * FROM research_queue_items WHERE proposal_id = ?", (proposal_id,)
            ).fetchone()
            if queue is None:
                now = time.time()
                conn.execute(
                    """
                    INSERT INTO research_queue_items(
                        queue_item_id, proposal_id, run_id, status, score, payload_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        queue_item_id,
                        proposal_id,
                        run_id,
                        queue_status,
                        score,
                        proposal["payload_json"],
                        now,
                        now,
                    ),
                )
                self._ledger.append_event(
                    conn,
                    run_id,
                    actor="transition",
                    event_type="queue_item_recorded",
                    payload={
                        "proposal_id": proposal_id,
                        "queue_item_id": queue_item_id,
                        "status": queue_status,
                    },
                )
                queue = conn.execute(
                    "SELECT * FROM research_queue_items WHERE queue_item_id = ?", (queue_item_id,)
                ).fetchone()
            elif str(queue["status"]) != queue_status or not math.isclose(
                float(queue["score"]), score, rel_tol=0.0, abs_tol=1e-12
            ):
                conn.execute(
                    """
                    UPDATE research_queue_items SET status = ?, score = ?, updated_at = ?
                    WHERE proposal_id = ?
                    """,
                    (queue_status, score, time.time(), proposal_id),
                )
                queue = conn.execute(
                    "SELECT * FROM research_queue_items WHERE proposal_id = ?", (proposal_id,)
                ).fetchone()
            if queue is None:
                raise RuntimeError("durable queue item was not persisted")
            return {"status": status, "score": score, "queue_item": self._queue_item_dict(queue)}

    def front_half_snapshot(self, run_id: str) -> dict[str, Any]:
        """Return canonical front-half facts so a crashed workflow can rebuild its view."""
        with self._ledger.transaction() as conn:
            self._run_status(conn, run_id)
            specs = conn.execute(
                "SELECT * FROM research_front_half_node_specs WHERE run_id = ? ORDER BY node_name",
                (run_id,),
            ).fetchall()
            proposals = conn.execute(
                "SELECT * FROM research_proposals WHERE run_id = ? ORDER BY created_at, proposal_id",
                (run_id,),
            ).fetchall()
            queue_items = conn.execute(
                "SELECT * FROM research_queue_items WHERE run_id = ? ORDER BY created_at, queue_item_id",
                (run_id,),
            ).fetchall()
            review_rows = conn.execute(
                """
                SELECT review.*
                FROM research_critic_reviews AS review
                JOIN research_proposals AS proposal ON proposal.proposal_id = review.proposal_id
                WHERE proposal.run_id = ?
                ORDER BY review.proposal_id, review.created_at, review.review_id
                """,
                (run_id,),
            ).fetchall()
            reviews_by_proposal: dict[str, list[dict[str, Any]]] = {}
            for review in review_rows:
                reviews_by_proposal.setdefault(str(review["proposal_id"]), []).append(
                    self._review_dict(review)
                )
            proposal_values = []
            for proposal in proposals:
                value = self._proposal_dict(proposal)
                value["reviews"] = reviews_by_proposal.get(str(proposal["proposal_id"]), [])
                proposal_values.append(value)
            return {
                "node_specs": {str(row["node_name"]): self._node_spec_dict(row) for row in specs},
                "proposals": proposal_values,
                "queue_items": [self._queue_item_dict(item) for item in queue_items],
            }

    def register_execution_node_contracts(
        self, run_id: str, node_specs: Mapping[str, Mapping[str, Any]]
    ) -> None:
        """Persist immutable contracts for compute, verify, settle and report."""
        with self._ledger.transaction() as conn:
            self._run_status(conn, run_id)
            inserted_nodes: list[str] = []
            for node_name, raw_spec in node_specs.items():
                spec = dict(raw_spec)
                max_attempts = int(spec.get("max_attempts", 0))
                retry_class = str(spec.get("retry_class", "")).strip()
                if not node_name or max_attempts < 1 or not retry_class:
                    raise ValueError("execution node contract is incomplete")
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO research_execution_node_specs(
                        run_id, node_name, input_schema_json, output_schema_json,
                        allowed_tools_json, max_attempts, retry_class, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        node_name,
                        json.dumps(spec.get("input_schema", {}), ensure_ascii=False),
                        json.dumps(spec.get("output_schema", {}), ensure_ascii=False),
                        json.dumps(list(spec.get("allowed_tools", [])), ensure_ascii=False),
                        max_attempts,
                        retry_class,
                        time.time(),
                    ),
                )
                if cursor.rowcount == 0:
                    row = conn.execute(
                        """
                        SELECT * FROM research_execution_node_specs
                        WHERE run_id = ? AND node_name = ?
                        """,
                        (run_id, node_name),
                    ).fetchone()
                    if row is None or self._node_contract(
                        dict(spec)
                    ) != self._node_contract_from_row(row):
                        raise ValueError(
                            "execution node contract conflicts with its existing record"
                        )
                else:
                    inserted_nodes.append(node_name)
            if inserted_nodes:
                self._ledger.append_event(
                    conn,
                    run_id,
                    actor="workflow",
                    event_type="execution_contracts_registered",
                    payload={"nodes": sorted(inserted_nodes)},
                )

    def record_experiment(
        self,
        run_id: str,
        *,
        experiment_id: str,
        proposal_id: str,
        claim_id: str,
        plan: Mapping[str, Any],
        environment: Mapping[str, Any],
        config: Mapping[str, Any],
        seed: int | None,
        producer_attempt_id: str | None = None,
        step_id: str | None = None,
        worker_id: str | None = None,
    ) -> dict[str, Any]:
        """Write a replay-safe experiment plan before compute is allowed to run."""
        if not experiment_id or not claim_id:
            raise ValueError("experiment_id and claim_id are required")
        with self._ledger.transaction() as conn:
            self._run_status(conn, run_id)
            if step_id:
                self._require_lease(conn, run_id=run_id, step_id=step_id, worker_id=worker_id)
            row = conn.execute(
                "SELECT * FROM research_experiments WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()
            contract = {
                "proposal_id": proposal_id,
                "claim_id": claim_id,
                "plan": self._json_roundtrip(plan),
                "environment": self._json_roundtrip(environment),
                "config": self._json_roundtrip(config),
                "seed": seed,
            }
            if row is None:
                existing = conn.execute(
                    "SELECT experiment_id FROM research_experiments WHERE run_id = ? AND claim_id = ?",
                    (run_id, claim_id),
                ).fetchone()
                if existing is not None and str(existing["experiment_id"]) != experiment_id:
                    raise ValueError("claim_id already belongs to another durable experiment")
                now = time.time()
                conn.execute(
                    """
                    INSERT INTO research_experiments(
                        experiment_id, run_id, proposal_id, claim_id, producer_attempt_id,
                        plan_json, environment_json, config_json, seed, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'planned', ?, ?)
                    """,
                    (
                        experiment_id,
                        run_id,
                        proposal_id,
                        claim_id,
                        producer_attempt_id,
                        json.dumps(contract["plan"], ensure_ascii=False),
                        json.dumps(contract["environment"], ensure_ascii=False),
                        json.dumps(contract["config"], ensure_ascii=False),
                        seed,
                        now,
                        now,
                    ),
                )
                self._ledger.append_event(
                    conn,
                    run_id,
                    actor="compute",
                    event_type="experiment_planned",
                    payload={"experiment_id": experiment_id, "claim_id": claim_id},
                )
                row = conn.execute(
                    "SELECT * FROM research_experiments WHERE experiment_id = ?", (experiment_id,)
                ).fetchone()
            elif str(row["run_id"]) != run_id or contract != self._experiment_contract(row):
                raise ValueError("durable experiment replay conflicts with its existing contract")
            if row is None:
                raise RuntimeError("durable experiment was not persisted")
            return self._experiment_dict(row)

    def record_execution_artifact(
        self,
        run_id: str,
        *,
        artifact_id: str,
        experiment_id: str,
        actor: str,
        kind: str,
        media_type: str,
        uri: str,
        sha256: str,
        byte_size: int,
        metadata: Mapping[str, Any],
        tool_call_id: str = "",
        producer_attempt_id: str | None = None,
        step_id: str | None = None,
        worker_id: str | None = None,
    ) -> dict[str, Any]:
        """Record immutable compute artifacts; verifier writes are always forbidden."""
        if actor != "compute":
            raise PermissionError("only compute may write experiment artifacts")
        if not artifact_id or not kind or not uri or not sha256:
            raise ValueError("artifact identity, kind, uri and sha256 are required")
        with self._ledger.transaction() as conn:
            experiment = conn.execute(
                "SELECT * FROM research_experiments WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()
            if experiment is None or str(experiment["run_id"]) != run_id:
                raise KeyError(f"unknown experiment {experiment_id!r}")
            if step_id:
                self._require_lease(conn, run_id=run_id, step_id=step_id, worker_id=worker_id)
            row = conn.execute(
                "SELECT * FROM research_artifacts WHERE artifact_id = ?", (artifact_id,)
            ).fetchone()
            contract = {
                "experiment_id": experiment_id,
                "kind": kind,
                "media_type": media_type,
                "uri": uri,
                "sha256": sha256,
                "byte_size": int(byte_size),
                "metadata": self._json_roundtrip(metadata),
                "tool_call_id": tool_call_id,
            }
            if row is None:
                duplicate = conn.execute(
                    """
                    SELECT * FROM research_artifacts
                    WHERE experiment_id = ? AND kind = ? AND sha256 = ? AND uri = ?
                    """,
                    (experiment_id, kind, sha256, uri),
                ).fetchone()
                if duplicate is not None:
                    row = duplicate
                else:
                    conn.execute(
                        """
                        INSERT INTO research_artifacts(
                            artifact_id, run_id, experiment_id, producer_attempt_id, tool_call_id,
                            kind, media_type, uri, sha256, byte_size, metadata_json, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            artifact_id,
                            run_id,
                            experiment_id,
                            producer_attempt_id,
                            tool_call_id or None,
                            kind,
                            media_type,
                            uri,
                            sha256,
                            int(byte_size),
                            json.dumps(contract["metadata"], ensure_ascii=False),
                            time.time(),
                        ),
                    )
                    self._ledger.append_event(
                        conn,
                        run_id,
                        actor="compute",
                        event_type="experiment_artifact_recorded",
                        payload={
                            "experiment_id": experiment_id,
                            "artifact_id": artifact_id,
                            "kind": kind,
                            "tool_call_id": tool_call_id or None,
                        },
                    )
                    row = conn.execute(
                        "SELECT * FROM research_artifacts WHERE artifact_id = ?", (artifact_id,)
                    ).fetchone()
            if row is None:
                raise RuntimeError("durable artifact was not persisted")
            if self._artifact_contract(row) != contract:
                raise ValueError("durable artifact replay conflicts with its existing contract")
            if kind == "output" and str(experiment["status"]) == "planned":
                conn.execute(
                    "UPDATE research_experiments SET status = 'computed', updated_at = ? WHERE experiment_id = ?",
                    (time.time(), experiment_id),
                )
            return self._artifact_dict(row)

    def settle_execution_claim(
        self,
        run_id: str,
        *,
        settlement_id: str,
        experiment_id: str,
        verification: Mapping[str, Any],
        evidence_ids: list[str],
        has_numeric_artifact: bool,
        expected_champion_version: int | None,
        noise_band: float,
        required_repeats: int,
    ) -> dict[str, Any]:
        """Derive KEEP/DISCARD/INSUFFICIENT and CAS-promote only safe keeps."""
        with self._ledger.transaction() as conn:
            experiment = conn.execute(
                "SELECT * FROM research_experiments WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()
            if experiment is None or str(experiment["run_id"]) != run_id:
                raise KeyError(f"unknown experiment {experiment_id!r}")
            claim_id = str(experiment["claim_id"])
            existing = conn.execute(
                "SELECT * FROM research_claim_settlements WHERE run_id = ? AND claim_id = ?",
                (run_id, claim_id),
            ).fetchone()
            if existing is not None:
                return self._settlement_dict(existing)

            status = str(verification.get("status") or "prediction")
            value = verification.get("value")
            numeric_value = (
                float(value)
                if isinstance(value, int | float) and not isinstance(value, bool)
                else None
            )
            has_evidence = bool(evidence_ids)
            verdict = "insufficient"
            reason = "unresolved_verification"
            if status == "verified":
                if not has_numeric_artifact:
                    reason = "missing_numeric_artifact"
                elif not has_evidence:
                    reason = "missing_evidence"
                elif (
                    noise_band > 0
                    and numeric_value is not None
                    and abs(numeric_value) <= noise_band
                    and required_repeats > 1
                ):
                    reason = "near_noise_requires_repeat"
                else:
                    verdict, reason = "keep", "verified_with_numeric_artifact_and_evidence"
            elif status == "falsified":
                if has_evidence:
                    verdict, reason = "discard", "falsified_by_evidence"
                else:
                    reason = "missing_evidence"

            champion_version: int | None = None
            if verdict == "keep":
                if expected_champion_version is None:
                    raise ValueError("KEEP settlement requires an expected champion version")
                champion = conn.execute(
                    "SELECT version FROM research_champion_versions WHERE run_id = ?", (run_id,)
                ).fetchone()
                current_version = int(champion["version"]) if champion is not None else 0
                if current_version != int(expected_champion_version):
                    raise ChampionVersionConflictError(
                        f"champion version changed: expected {expected_champion_version}, got {current_version}"
                    )
                champion_version = current_version + 1
                if champion is None:
                    conn.execute(
                        "INSERT INTO research_champion_versions(run_id, version, updated_at) VALUES (?, ?, ?)",
                        (run_id, champion_version, time.time()),
                    )
                else:
                    conn.execute(
                        "UPDATE research_champion_versions SET version = ?, updated_at = ? WHERE run_id = ?",
                        (champion_version, time.time(), run_id),
                    )

            now = time.time()
            result = self._json_roundtrip(dict(verification))
            conn.execute(
                """
                INSERT INTO research_claim_settlements(
                    settlement_id, run_id, experiment_id, claim_id, verdict, reason,
                    evidence_ids_json, result_json, expected_champion_version, champion_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    settlement_id,
                    run_id,
                    experiment_id,
                    claim_id,
                    verdict,
                    reason,
                    json.dumps(list(dict.fromkeys(evidence_ids)), ensure_ascii=False),
                    json.dumps(result, ensure_ascii=False),
                    expected_champion_version,
                    champion_version,
                    now,
                ),
            )
            conn.execute(
                "UPDATE research_experiments SET status = 'settled', updated_at = ? WHERE experiment_id = ?",
                (now, experiment_id),
            )
            self._ledger.append_event(
                conn,
                run_id,
                actor="settle",
                event_type="claim_settled",
                payload={
                    "settlement_id": settlement_id,
                    "experiment_id": experiment_id,
                    "claim_id": claim_id,
                    "verdict": verdict,
                    "reason": reason,
                    "champion_version": champion_version,
                },
            )
            row = conn.execute(
                "SELECT * FROM research_claim_settlements WHERE settlement_id = ?", (settlement_id,)
            ).fetchone()
            if row is None:
                raise RuntimeError("durable settlement was not persisted")
            return self._settlement_dict(row)

    def execution_snapshot(self, run_id: str) -> dict[str, Any]:
        """Return the execution facts used by recovery and deterministic reporting."""
        with self._ledger.transaction() as conn:
            self._run_status(conn, run_id)
            specs = conn.execute(
                "SELECT * FROM research_execution_node_specs WHERE run_id = ? ORDER BY node_name",
                (run_id,),
            ).fetchall()
            experiments = conn.execute(
                "SELECT * FROM research_experiments WHERE run_id = ? ORDER BY created_at, experiment_id",
                (run_id,),
            ).fetchall()
            artifacts = conn.execute(
                "SELECT * FROM research_artifacts WHERE run_id = ? ORDER BY created_at, artifact_id",
                (run_id,),
            ).fetchall()
            settlements = conn.execute(
                "SELECT * FROM research_claim_settlements WHERE run_id = ? ORDER BY created_at, settlement_id",
                (run_id,),
            ).fetchall()
            champion = conn.execute(
                "SELECT version FROM research_champion_versions WHERE run_id = ?", (run_id,)
            ).fetchone()
            return {
                "node_specs": {str(row["node_name"]): self._node_spec_dict(row) for row in specs},
                "experiments": [self._experiment_dict(row) for row in experiments],
                "artifacts": [self._artifact_dict(row) for row in artifacts],
                "settlements": [self._settlement_dict(row) for row in settlements],
                "champion_version": int(champion["version"]) if champion is not None else 0,
            }

    @classmethod
    def _experiment_contract(cls, row: Any) -> dict[str, Any]:
        return {
            "proposal_id": str(row["proposal_id"] or ""),
            "claim_id": str(row["claim_id"]),
            "plan": cls._json(row["plan_json"], {}),
            "environment": cls._json(row["environment_json"], {}),
            "config": cls._json(row["config_json"], {}),
            "seed": row["seed"],
        }

    @classmethod
    def _experiment_dict(cls, row: Any) -> dict[str, Any]:
        return {
            "experiment_id": str(row["experiment_id"]),
            "proposal_id": str(row["proposal_id"] or ""),
            "claim_id": str(row["claim_id"]),
            "producer_attempt_id": str(row["producer_attempt_id"] or ""),
            "plan": cls._json(row["plan_json"], {}),
            "environment": cls._json(row["environment_json"], {}),
            "config": cls._json(row["config_json"], {}),
            "seed": row["seed"],
            "status": str(row["status"]),
        }

    @classmethod
    def _artifact_contract(cls, row: Any) -> dict[str, Any]:
        return {
            "experiment_id": str(row["experiment_id"]),
            "kind": str(row["kind"]),
            "media_type": str(row["media_type"]),
            "uri": str(row["uri"]),
            "sha256": str(row["sha256"]),
            "byte_size": int(row["byte_size"]),
            "metadata": cls._json(row["metadata_json"], {}),
            "tool_call_id": str(row["tool_call_id"] or ""),
        }

    @classmethod
    def _artifact_dict(cls, row: Any) -> dict[str, Any]:
        value = cls._artifact_contract(row)
        value["artifact_id"] = str(row["artifact_id"])
        value["producer_attempt_id"] = str(row["producer_attempt_id"] or "")
        return value

    @classmethod
    def _settlement_dict(cls, row: Any) -> dict[str, Any]:
        return {
            "settlement_id": str(row["settlement_id"]),
            "experiment_id": str(row["experiment_id"]),
            "claim_id": str(row["claim_id"]),
            "verdict": str(row["verdict"]),
            "reason": str(row["reason"]),
            "evidence_ids": cls._json(row["evidence_ids_json"], []),
            "result": cls._json(row["result_json"], {}),
            "expected_champion_version": row["expected_champion_version"],
            "champion_version": row["champion_version"],
        }

    @staticmethod
    def _proposal_row(conn: Any, proposal_id: str) -> Any:
        return conn.execute(
            "SELECT * FROM research_proposals WHERE proposal_id = ?", (proposal_id,)
        ).fetchone()

    @staticmethod
    def _json(value: Any, default: Any) -> Any:
        try:
            return json.loads(str(value)) if value else default
        except (TypeError, json.JSONDecodeError):
            return default

    @staticmethod
    def _json_roundtrip(value: Any) -> Any:
        return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))

    @classmethod
    def _node_contract(cls, spec: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "input_schema": cls._json_roundtrip(spec.get("input_schema", {})),
            "output_schema": cls._json_roundtrip(spec.get("output_schema", {})),
            "allowed_tools": cls._json_roundtrip(list(spec.get("allowed_tools", []))),
            "max_attempts": int(spec.get("max_attempts", 0)),
            "retry_class": str(spec.get("retry_class", "")),
        }

    @classmethod
    def _node_contract_from_row(cls, row: Any) -> dict[str, Any]:
        return cls._node_contract(
            {
                "input_schema": cls._json(row["input_schema_json"], {}),
                "output_schema": cls._json(row["output_schema_json"], {}),
                "allowed_tools": cls._json(row["allowed_tools_json"], []),
                "max_attempts": row["max_attempts"],
                "retry_class": row["retry_class"],
            }
        )

    @staticmethod
    def _proposal_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
        """Exclude mutable workflow fields when validating an idempotent replay."""
        return {
            key: payload.get(key)
            for key in ("claim_id", "statement", "conditions", "prediction", "falsification")
        }

    @classmethod
    def _node_spec_dict(cls, row: Any) -> dict[str, Any]:
        return {
            "input_schema": cls._json(row["input_schema_json"], {}),
            "output_schema": cls._json(row["output_schema_json"], {}),
            "allowed_tools": cls._json(row["allowed_tools_json"], []),
            "max_attempts": int(row["max_attempts"]),
            "retry_class": str(row["retry_class"]),
        }

    @classmethod
    def _proposal_dict(cls, row: Any) -> dict[str, Any]:
        return {
            "proposal_id": str(row["proposal_id"]),
            "claim_id": str(row["claim_id"]),
            "author": str(row["author"]),
            "payload": cls._json(row["payload_json"], {}),
            "status": str(row["status"]),
            "review_score": row["review_score"],
        }

    @staticmethod
    def _review_dict(row: Any) -> dict[str, Any]:
        return {
            "review_id": str(row["review_id"]),
            "reviewer": str(row["reviewer"]),
            "score": float(row["score"]),
            "verdict": str(row["verdict"]),
            "content": str(row["content"]),
        }

    @classmethod
    def _queue_item_dict(cls, row: Any) -> dict[str, Any]:
        return {
            "queue_item_id": str(row["queue_item_id"]),
            "proposal_id": str(row["proposal_id"]),
            "status": str(row["status"]),
            "score": float(row["score"]),
            "payload": cls._json(row["payload_json"], {}),
        }

    @staticmethod
    def _run_status(conn: Any, run_id: str) -> str:
        row = conn.execute(
            "SELECT status FROM research_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown research run {run_id}")
        return str(row["status"])

    @staticmethod
    def _set_run_status(conn: Any, run_id: str, status: str, *, completed: bool = False) -> None:
        conn.execute(
            """
            UPDATE research_runs
            SET status = ?, updated_at = ?, completed_at = CASE WHEN ? THEN ? ELSE completed_at END
            WHERE run_id = ?
            """,
            (status, time.time(), completed, time.time(), run_id),
        )

    def _transition_step(self, conn: Any, step_id: str, target: str) -> None:
        row = conn.execute(
            "SELECT status FROM research_steps WHERE step_id = ?", (step_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown research step {step_id}")
        validate_step_transition(str(row["status"]), target)
        self._set_step_status(conn, step_id, target)

    @staticmethod
    def _require_lease(conn: Any, *, run_id: str, step_id: str, worker_id: str | None) -> None:
        """Require a step to belong to its run and, for new workers, hold its lease."""
        row = conn.execute(
            "SELECT run_id, lease_owner, lease_expires_at FROM research_steps WHERE step_id = ?",
            (step_id,),
        ).fetchone()
        if row is None or str(row["run_id"]) != run_id:
            raise KeyError(f"unknown step {step_id!r} for run {run_id!r}")
        if worker_id is None:
            return
        if (
            row["lease_owner"] != worker_id
            or row["lease_expires_at"] is None
            or float(row["lease_expires_at"]) <= time.time()
        ):
            raise LeaseUnavailableError(f"worker {worker_id!r} does not own active lease")

    @staticmethod
    def _set_step_status(conn: Any, step_id: str, status: str) -> None:
        conn.execute(
            "UPDATE research_steps SET status = ?, updated_at = ? WHERE step_id = ?",
            (status, time.time(), step_id),
        )
