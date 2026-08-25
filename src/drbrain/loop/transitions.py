"""The sole state-writing service for durable autoresearch runs."""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from typing import Any

from drbrain.loop.state import (
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
from drbrain.loop.store import LedgerEvent, RunLedger


class LeaseUnavailableError(RuntimeError):
    """Raised when another worker still owns an in-flight cycle lease."""


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
            self._require_lease(conn, step_id=step_id, worker_id=worker_id)
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
            self._require_lease(conn, step_id=step_id, worker_id=worker_id)
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
                "SELECT status FROM research_steps WHERE step_id = ?", (step_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown research step {step_id}")
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
    def _require_lease(conn: Any, *, step_id: str, worker_id: str | None) -> None:
        """Keep legacy callers working while enforcing leases for new workers."""
        if worker_id is None:
            return
        row = conn.execute(
            "SELECT lease_owner, lease_expires_at FROM research_steps WHERE step_id = ?", (step_id,)
        ).fetchone()
        if (
            row is None
            or row["lease_owner"] != worker_id
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
