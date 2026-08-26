"""Operator-facing controls and read-only inspection for durable research runs."""

from __future__ import annotations

from collections import Counter
from typing import Any

from drbrain.loop.store import LedgerRun, RunLedger
from drbrain.loop.transitions import TransitionService


class RunGovernance:
    """Additive operational API over a run ledger.

    Read methods deliberately use the ledger's read-only accessors. Lifecycle
    controls delegate to :class:`TransitionService`; approval and budget
    controls persist their own audited ledger mutations. All preserve existing
    workspace files and evidence artifacts.
    """

    def __init__(self, ledger: RunLedger) -> None:
        self._ledger = ledger
        self._transitions = TransitionService(ledger)

    def status(self, identifier: str) -> dict[str, Any]:
        """Return a compact, read-only status snapshot by run ID or topic."""
        run = self._resolve(identifier)
        budget = self._ledger.budget_snapshot(run.run_id)
        events = self._ledger.events(run.run_id)
        return {
            "run_id": run.run_id,
            "topic": run.topic,
            "status": run.status,
            "config": run.config,
            "budget": budget,
            "last_projected_event": run.last_projected_event,
            "active_steps": self._ledger.active_leased_steps(run.run_id),
            "recoverable_steps": self._ledger.recoverable_step_ids(run.run_id),
            "manual_review_steps": self._ledger.manual_review_step_ids(run.run_id),
            "event_count": len(events),
            "last_event": self._event_dict(events[-1]) if events else None,
        }

    def trace(self, identifier: str) -> dict[str, Any]:
        """Return an ordered immutable event trace without allocating new IDs."""
        run = self._resolve(identifier)
        return {
            "run_id": run.run_id,
            "topic": run.topic,
            "events": [self._event_dict(event) for event in self._ledger.events(run.run_id)],
            "tool_calls": [
                {
                    "tool_call_id": call.tool_call_id,
                    "step_id": call.step_id,
                    "attempt_id": call.attempt_id,
                    "node_name": call.node_name,
                    "tool_name": call.tool_name,
                    "status": call.status,
                    "idempotency_key": call.idempotency_key,
                }
                for call in self._ledger.tool_calls(run.run_id)
            ],
        }

    def audit_summary(self, identifier: str) -> dict[str, Any]:
        """Return an operator-friendly terminal or in-flight audit summary."""
        run = self._resolve(identifier)
        events = self._ledger.events(run.run_id)
        calls = self._ledger.tool_calls(run.run_id)
        event_types = Counter(event.event_type for event in events)
        tool_statuses = Counter(call.status for call in calls)
        return {
            "run_id": run.run_id,
            "topic": run.topic,
            "status": run.status,
            "event_count": len(events),
            "event_types": dict(sorted(event_types.items())),
            "tool_call_count": len(calls),
            "tool_statuses": dict(sorted(tool_statuses.items())),
            "budget": self._ledger.budget_snapshot(run.run_id),
            "has_complete_audit": bool(events)
            and run.status in {"paused", "succeeded", "failed", "cancelled"},
        }

    def pause(self, identifier: str, *, reason: str = "operator_pause") -> dict[str, Any]:
        run = self._resolve(identifier)
        self._transitions.pause_run(run.run_id, reason=reason)
        return self.status(run.run_id)

    def resume(self, identifier: str) -> dict[str, Any]:
        run = self._resolve(identifier)
        self._transitions.start_run(run.run_id)
        return self.status(run.run_id)

    def cancel(self, identifier: str, *, reason: str = "operator_cancel") -> dict[str, Any]:
        run = self._resolve(identifier)
        self._transitions.cancel_run(run.run_id, reason=reason)
        return self.status(run.run_id)

    def resolve_manual_review(
        self, identifier: str, *, step_id: str, reason: str, actor: str = "operator"
    ) -> dict[str, Any]:
        """Record an explicit abandon decision; it never retries the old step."""
        run = self._resolve(identifier)
        self._transitions.resolve_manual_review(
            run.run_id, step_id=step_id, reason=reason, actor=actor
        )
        return self.status(run.run_id)

    def approve(
        self, tool_call_id: str, *, actor: str = "operator", reason: str = ""
    ) -> dict[str, Any]:
        """Authorize one future retry of a waiting idempotent tool proposal."""
        return self._ledger.record_approval_decision(
            tool_call_id=tool_call_id,
            decision="approved",
            actor=actor,
            reason=reason,
        )

    def reject(
        self, tool_call_id: str, *, actor: str = "operator", reason: str = ""
    ) -> dict[str, Any]:
        """Reject one future retry of a waiting idempotent tool proposal."""
        return self._ledger.record_approval_decision(
            tool_call_id=tool_call_id,
            decision="rejected",
            actor=actor,
            reason=reason,
        )

    def reserve(self, run_id: str, amounts: dict[str, int | float]) -> dict[str, Any]:
        """Expose the same pre-execution budget guard used by runtime boundaries."""
        return self._ledger.reserve_budget(run_id, amounts)

    def consume_observed(self, run_id: str, amounts: dict[str, int | float]) -> dict[str, Any]:
        """Record provider-reported usage after an external boundary completes."""
        return self._ledger.consume_observed_budget(run_id, amounts)

    def _resolve(self, identifier: str) -> LedgerRun:
        run = self._ledger.get_run_by_id(identifier) or self._ledger.get_run(identifier)
        if run is None:
            raise KeyError(f"unknown research run or topic: {identifier}")
        return run

    @staticmethod
    def _event_dict(event: Any) -> dict[str, Any]:
        return {
            "event_seq": event.event_seq,
            "actor": event.actor,
            "event_type": event.event_type,
            "payload": event.payload,
            "trace_id": event.trace_id,
            "created_at": event.created_at,
        }


__all__ = ["RunGovernance"]
