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
    STEP_PENDING,
    STEP_READY,
    STEP_RUNNING,
    STEP_SUCCEEDED,
    STEP_UNKNOWN,
    validate_run_transition,
    validate_step_transition,
)
from drbrain.loop.store import LedgerEvent, RunLedger


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
        """Turn abandoned in-flight cycles into auditable ``unknown`` records.

        PR 1 deliberately does not resume the LlamaIndex Context mid-node.  It
        instead leaves the interrupted cycle visible and lets a later cycle run
        from the last compatible projection; checkpoint resume belongs to PR 2.
        """
        with self._ledger.transaction() as conn:
            rows = conn.execute(
                "SELECT step_id, status FROM research_steps WHERE run_id = ? AND status = ?",
                (run_id, STEP_RUNNING),
            ).fetchall()
            for row in rows:
                step_id = str(row["step_id"])
                validate_step_transition(str(row["status"]), STEP_UNKNOWN)
                self._set_step_status(conn, step_id, STEP_UNKNOWN)
                conn.execute(
                    """
                    UPDATE research_attempts
                    SET status = ?, completed_at = ?
                    WHERE step_id = ? AND status = ?
                    """,
                    (STEP_UNKNOWN, time.time(), step_id, STEP_RUNNING),
                )
                self._ledger.append_event(
                    conn,
                    run_id,
                    actor="recovery",
                    event_type="cycle_interrupted",
                    payload={"step_id": step_id},
                )

    def begin_cycle(self, run_id: str, *, cycle: int) -> str:
        """Create and start one durable cycle step with its first attempt."""
        with self._ledger.transaction() as conn:
            if self._run_status(conn, run_id) != RUN_RUNNING:
                raise RuntimeError("cannot begin a cycle while the run is not running")
            now = time.time()
            step_id = uuid.uuid4().hex
            conn.execute(
                """
                INSERT INTO research_steps(
                    step_id, run_id, step_name, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (step_id, run_id, f"cycle:{cycle}", STEP_PENDING, now, now),
            )
            self._transition_step(conn, step_id, STEP_READY)
            self._transition_step(conn, step_id, STEP_CLAIMED)
            self._transition_step(conn, step_id, STEP_RUNNING)
            conn.execute(
                """
                INSERT INTO research_attempts(
                    attempt_id, step_id, attempt_no, status, started_at
                ) VALUES (?, ?, 1, ?, ?)
                """,
                (uuid.uuid4().hex, step_id, STEP_RUNNING, now),
            )
            self._ledger.append_event(
                conn,
                run_id,
                actor="director",
                event_type="cycle_started",
                payload={"cycle": cycle, "step_id": step_id},
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
    ) -> LedgerEvent:
        """Commit a completed cycle and its replayable compatibility snapshot."""
        with self._ledger.transaction() as conn:
            self._transition_step(conn, step_id, STEP_SUCCEEDED)
            conn.execute(
                """
                UPDATE research_attempts
                SET status = ?, completed_at = ?
                WHERE step_id = ? AND status = ?
                """,
                (STEP_SUCCEEDED, time.time(), step_id, STEP_RUNNING),
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

    def fail_cycle(self, run_id: str, *, step_id: str, error: BaseException) -> None:
        """Commit a failed workflow cycle before the exception propagates."""
        with self._ledger.transaction() as conn:
            self._transition_step(conn, step_id, STEP_FAILED)
            conn.execute(
                """
                UPDATE research_attempts
                SET status = ?, failure_category = ?, completed_at = ?
                WHERE step_id = ? AND status = ?
                """,
                (STEP_FAILED, type(error).__name__, time.time(), step_id, STEP_RUNNING),
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
    def _set_step_status(conn: Any, step_id: str, status: str) -> None:
        conn.execute(
            "UPDATE research_steps SET status = ?, updated_at = ? WHERE step_id = ?",
            (status, time.time(), step_id),
        )
