"""State-machine rules for the durable autoresearch ledger.

The workflow remains free to decide *what* to research.  These rules only
govern durable lifecycle transitions, so a partial or interrupted run has a
small, inspectable set of legal states.
"""

from __future__ import annotations

from collections.abc import Mapping


class InvalidTransitionError(ValueError):
    """Raised when a durable run or step tries to skip its lifecycle."""


RUN_CREATED = "created"
RUN_RUNNING = "running"
RUN_PAUSED = "paused"
RUN_SUCCEEDED = "succeeded"
RUN_FAILED = "failed"
RUN_CANCELLED = "cancelled"

STEP_PENDING = "pending"
STEP_READY = "ready"
STEP_CLAIMED = "claimed"
STEP_RUNNING = "running"
STEP_WAITING_APPROVAL = "waiting_approval"
STEP_SUCCEEDED = "succeeded"
STEP_FAILED = "failed"
STEP_TIMED_OUT = "timed_out"
STEP_UNKNOWN = "unknown"
STEP_RECONCILING = "reconciling"
STEP_MANUAL_REVIEW = "manual_review"


RUN_TRANSITIONS: Mapping[str, frozenset[str]] = {
    RUN_CREATED: frozenset({RUN_RUNNING}),
    RUN_RUNNING: frozenset({RUN_PAUSED, RUN_SUCCEEDED, RUN_FAILED, RUN_CANCELLED}),
    RUN_PAUSED: frozenset({RUN_RUNNING, RUN_CANCELLED}),
    RUN_SUCCEEDED: frozenset(),
    RUN_FAILED: frozenset(),
    RUN_CANCELLED: frozenset(),
}

STEP_TRANSITIONS: Mapping[str, frozenset[str]] = {
    STEP_PENDING: frozenset({STEP_READY}),
    STEP_READY: frozenset({STEP_CLAIMED}),
    STEP_CLAIMED: frozenset({STEP_RUNNING}),
    STEP_RUNNING: frozenset(
        {
            STEP_WAITING_APPROVAL,
            STEP_SUCCEEDED,
            STEP_FAILED,
            STEP_TIMED_OUT,
            STEP_UNKNOWN,
        }
    ),
    STEP_WAITING_APPROVAL: frozenset({STEP_RUNNING}),
    STEP_UNKNOWN: frozenset({STEP_RECONCILING}),
    STEP_RECONCILING: frozenset({STEP_SUCCEEDED, STEP_FAILED, STEP_MANUAL_REVIEW, STEP_UNKNOWN}),
    STEP_SUCCEEDED: frozenset(),
    STEP_FAILED: frozenset(),
    STEP_TIMED_OUT: frozenset(),
    STEP_MANUAL_REVIEW: frozenset(),
}


def validate_transition(
    transitions: Mapping[str, frozenset[str]],
    *,
    kind: str,
    current: str,
    target: str,
) -> None:
    """Reject a transition that is not explicitly part of a lifecycle."""
    if target not in transitions.get(current, frozenset()):
        raise InvalidTransitionError(f"invalid {kind} transition: {current!r} -> {target!r}")


def validate_run_transition(current: str, target: str) -> None:
    """Validate one research-run transition."""
    validate_transition(RUN_TRANSITIONS, kind="run", current=current, target=target)


def validate_step_transition(current: str, target: str) -> None:
    """Validate one research-step transition."""
    validate_transition(STEP_TRANSITIONS, kind="step", current=current, target=target)
