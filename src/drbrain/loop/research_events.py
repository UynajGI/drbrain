"""Typed event-sourcing primitives for autoresearch orchestration.

The loop ledger predates the supervisor and stores a small, append-only event
stream.  This module adds a protocol-level envelope, idempotency, snapshots and
replay without changing that legacy table or its callers.  ``RunLedgerEventLog``
encodes the envelope metadata in a reserved payload namespace when writing to
:class:`~drbrain.loop.store.RunLedger`.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from drbrain.loop.store import LedgerEvent, RunLedger

_EVENT_META_KEY = "_research_event"
_EVENT_META_VERSION = 1
_SNAPSHOT_EVENT_TYPE = "research_snapshot"


class EventError(ValueError):
    """Base class for invalid event streams."""


class SequenceConflictError(EventError):
    """Raised when an append uses a stale expected sequence."""


class IdempotencyConflictError(EventError):
    """Raised when an idempotency key is reused for another event."""


class EventValidationError(EventError):
    """Raised when an event envelope contains invalid data."""


@dataclass(frozen=True)
class EventEnvelope:
    """Versioned, immutable event envelope shared by all loop adapters."""

    event_id: str
    run_id: str
    seq: int
    event_type: str
    schema_version: int
    actor: str
    idempotency_key: str | None
    occurred_at: float
    trace_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id or not self.run_id or not self.event_type or not self.actor:
            raise EventValidationError("event_id, run_id, event_type and actor are required")
        if self.seq < 1:
            raise EventValidationError("event seq must be positive")
        if self.schema_version < 1:
            raise EventValidationError("schema_version must be positive")
        if self.idempotency_key is not None and not self.idempotency_key:
            raise EventValidationError("idempotency_key must be non-empty when supplied")
        if not isinstance(self.payload, dict):
            raise EventValidationError("event payload must be a JSON object")
        try:
            json.dumps(self.payload, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise EventValidationError("event payload must be JSON serializable") from exc

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        event_type: str,
        actor: str,
        payload: Mapping[str, Any] | None = None,
        seq: int,
        schema_version: int = 1,
        idempotency_key: str | None = None,
        event_id: str | None = None,
        occurred_at: float | None = None,
        trace_id: str | None = None,
    ) -> EventEnvelope:
        return cls(
            event_id=event_id or str(uuid.uuid4()),
            run_id=run_id,
            seq=seq,
            event_type=event_type,
            schema_version=schema_version,
            actor=actor,
            idempotency_key=idempotency_key,
            occurred_at=time.time() if occurred_at is None else float(occurred_at),
            trace_id=trace_id,
            payload=dict(payload or {}),
        )

    @property
    def payload_digest(self) -> str:
        """Stable digest used to detect conflicting idempotent retries."""
        body = json.dumps(self.payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    def equivalent_retry(self, other: EventEnvelope) -> bool:
        return (
            self.run_id == other.run_id
            and self.event_type == other.event_type
            and self.actor == other.actor
            and self.schema_version == other.schema_version
            and self.payload_digest == other.payload_digest
        )


class EventLog(Protocol):
    """Minimal append/read contract required by the supervisor."""

    def append(
        self,
        *,
        run_id: str,
        event_type: str,
        actor: str,
        payload: Mapping[str, Any] | None = None,
        schema_version: int = 1,
        idempotency_key: str | None = None,
        trace_id: str | None = None,
        expected_seq: int | None = None,
    ) -> EventEnvelope: ...

    def read(self, run_id: str, *, after_seq: int = 0) -> list[EventEnvelope]: ...

    def latest_seq(self, run_id: str) -> int: ...


class InMemoryEventLog:
    """Strict reference event log useful for tests and local orchestration."""

    def __init__(self) -> None:
        self._events: dict[str, list[EventEnvelope]] = {}
        self._idempotency: dict[tuple[str, str], EventEnvelope] = {}

    def append(
        self,
        *,
        run_id: str,
        event_type: str,
        actor: str,
        payload: Mapping[str, Any] | None = None,
        schema_version: int = 1,
        idempotency_key: str | None = None,
        trace_id: str | None = None,
        expected_seq: int | None = None,
    ) -> EventEnvelope:
        events = self._events.setdefault(run_id, [])
        next_seq = len(events) + 1
        if idempotency_key is not None:
            previous = self._idempotency.get((run_id, idempotency_key))
            if previous is not None:
                candidate = EventEnvelope.create(
                    run_id=run_id,
                    event_type=event_type,
                    actor=actor,
                    payload=payload,
                    schema_version=schema_version,
                    idempotency_key=idempotency_key,
                    seq=previous.seq,
                    trace_id=trace_id,
                )
                if not previous.equivalent_retry(candidate):
                    raise IdempotencyConflictError(
                        f"idempotency key {idempotency_key!r} already identifies another event"
                    )
                return previous
        if expected_seq is not None and expected_seq != next_seq:
            raise SequenceConflictError(f"expected seq {expected_seq}, next seq is {next_seq}")
        event = EventEnvelope.create(
            run_id=run_id,
            event_type=event_type,
            actor=actor,
            payload=payload,
            schema_version=schema_version,
            idempotency_key=idempotency_key,
            seq=next_seq,
            trace_id=trace_id,
        )
        events.append(event)
        if idempotency_key is not None:
            self._idempotency[(run_id, idempotency_key)] = event
        return event

    def read(self, run_id: str, *, after_seq: int = 0) -> list[EventEnvelope]:
        if after_seq < 0:
            raise SequenceConflictError("after_seq cannot be negative")
        return [event for event in self._events.get(run_id, ()) if event.seq > after_seq]

    def latest_seq(self, run_id: str) -> int:
        events = self._events.get(run_id, ())
        return events[-1].seq if events else 0


class RunLedgerEventLog:
    """EventLog adapter backed by the existing per-run ``RunLedger`` table."""

    def __init__(self, ledger: RunLedger) -> None:
        self.ledger = ledger

    @staticmethod
    def _decode(event: LedgerEvent) -> EventEnvelope:
        payload = dict(event.payload)
        metadata = payload.pop(_EVENT_META_KEY, {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        return EventEnvelope(
            event_id=str(metadata.get("i") or f"legacy:{event.run_id}:{event.event_seq}"),
            run_id=event.run_id,
            seq=event.event_seq,
            event_type=event.event_type,
            schema_version=int(metadata.get("s") or 1),
            actor=event.actor,
            idempotency_key=(str(metadata["k"]) if metadata.get("k") else None),
            occurred_at=event.created_at,
            trace_id=event.trace_id,
            payload=payload,
        )

    def append(
        self,
        *,
        run_id: str,
        event_type: str,
        actor: str,
        payload: Mapping[str, Any] | None = None,
        schema_version: int = 1,
        idempotency_key: str | None = None,
        trace_id: str | None = None,
        expected_seq: int | None = None,
    ) -> EventEnvelope:
        candidate_payload = dict(payload or {})
        with self.ledger.transaction() as conn:
            if (
                conn.execute("SELECT 1 FROM research_runs WHERE run_id = ?", (run_id,)).fetchone()
                is None
            ):
                raise KeyError(f"unknown research run: {run_id}")
            rows = conn.execute(
                "SELECT run_id, event_seq, actor, event_type, payload_json, trace_id, created_at "
                "FROM research_events WHERE run_id = ? ORDER BY event_seq",
                (run_id,),
            ).fetchall()
            existing = [self._decode(self.ledger._event_from_row(row)) for row in rows]
            next_seq = (existing[-1].seq + 1) if existing else 1
            if idempotency_key is not None:
                for prior in existing:
                    if prior.idempotency_key != idempotency_key:
                        continue
                    candidate = EventEnvelope.create(
                        run_id=run_id,
                        event_type=event_type,
                        actor=actor,
                        payload=candidate_payload,
                        schema_version=schema_version,
                        idempotency_key=idempotency_key,
                        seq=prior.seq,
                        trace_id=trace_id,
                    )
                    if not prior.equivalent_retry(candidate):
                        raise IdempotencyConflictError(
                            f"idempotency key {idempotency_key!r} already identifies another event"
                        )
                    return prior
            if expected_seq is not None and expected_seq != next_seq:
                raise SequenceConflictError(f"expected seq {expected_seq}, next seq is {next_seq}")
            event_id = str(uuid.uuid4())
            stored_payload = {
                **candidate_payload,
                _EVENT_META_KEY: {
                    # Short neutral keys avoid the ledger's credential-field
                    # redactor treating the protocol metadata as secrets.
                    "i": event_id,
                    "s": schema_version,
                    "k": idempotency_key,
                    "v": _EVENT_META_VERSION,
                },
            }
            written = self.ledger.append_event(
                conn,
                run_id,
                actor=actor,
                event_type=event_type,
                payload=stored_payload,
                trace_id=trace_id,
            )
            return EventEnvelope(
                event_id=event_id,
                run_id=run_id,
                seq=written.event_seq,
                event_type=event_type,
                schema_version=schema_version,
                actor=actor,
                idempotency_key=idempotency_key,
                occurred_at=written.created_at,
                trace_id=trace_id,
                payload=candidate_payload,
            )

    def read(self, run_id: str, *, after_seq: int = 0) -> list[EventEnvelope]:
        if after_seq < 0:
            raise SequenceConflictError("after_seq cannot be negative")
        return [
            self._decode(event)
            for event in self.ledger.events(run_id)
            if event.event_seq > after_seq
        ]

    def latest_seq(self, run_id: str) -> int:
        events = self.ledger.events(run_id)
        return events[-1].event_seq if events else 0

    # Expose the snapshot-store protocol directly as a convenience for hosts
    # that already hold a ``RunLedgerEventLog``.  The implementation remains
    # an adapter, so callers can pass ``snapshots=event_log`` to
    # ``EventSourcedState`` without knowing about the marker-event format.
    def save(self, snapshot: EventSnapshot) -> None:
        EventLogSnapshotStore(self).save(snapshot)

    def latest(self, run_id: str) -> EventSnapshot | None:
        return EventLogSnapshotStore(self).latest(run_id)


class EventLogSnapshotStore:
    """Persist snapshots as typed events in any :class:`EventLog`.

    A snapshot records the state *before* the snapshot event is appended.  On
    replay, the snapshot event is skipped and the event tail starts at
    ``snapshot.seq + 1``.  This keeps the ledger append-only and means a
    supervisor can be reconstructed after process restart without a separate
    database table or in-memory cache.
    """

    def __init__(self, event_log: EventLog) -> None:
        self.event_log = event_log

    def save(self, snapshot: EventSnapshot) -> None:
        previous = self.latest(snapshot.run_id)
        if previous is not None:
            if snapshot.seq < previous.seq:
                raise SequenceConflictError("snapshot sequence cannot move backwards")
            if snapshot.seq == previous.seq:
                if (
                    snapshot.reducer_version == previous.reducer_version
                    and snapshot.state == previous.state
                ):
                    return
                raise SequenceConflictError("snapshot at the same sequence differs")
        payload = {
            "run_id": snapshot.run_id,
            "seq": snapshot.seq,
            "state": snapshot.state,
            "schema_version": snapshot.schema_version,
            "reducer_version": snapshot.reducer_version,
            "created_at": snapshot.created_at,
        }
        self.event_log.append(
            run_id=snapshot.run_id,
            event_type=_SNAPSHOT_EVENT_TYPE,
            actor="event-sourcing",
            payload=payload,
            schema_version=snapshot.schema_version,
            idempotency_key=(
                f"snapshot:{snapshot.run_id}:{snapshot.seq}:{snapshot.reducer_version}"
            ),
        )

    def latest(self, run_id: str) -> EventSnapshot | None:
        snapshots = [
            event
            for event in self.event_log.read(run_id)
            if event.event_type == _SNAPSHOT_EVENT_TYPE
        ]
        if not snapshots:
            return None
        payload = snapshots[-1].payload
        try:
            payload_run_id = str(payload.get("run_id") or run_id)
            if payload_run_id != run_id:
                raise ValueError("snapshot run_id does not match requested stream")
            state = payload["state"]
            if not isinstance(state, dict):
                raise TypeError("state must be an object")
            return EventSnapshot(
                run_id=payload_run_id,
                seq=int(payload["seq"]),
                state=copy.deepcopy(state),
                schema_version=int(payload.get("schema_version", 1)),
                reducer_version=str(payload.get("reducer_version", "1")),
                created_at=float(payload.get("created_at", 0.0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EventValidationError(f"invalid persisted snapshot for run {run_id!r}") from exc


@dataclass(frozen=True)
class EventSnapshot:
    """Serializable state snapshot at a specific event sequence."""

    run_id: str
    seq: int
    state: dict[str, Any]
    schema_version: int = 1
    reducer_version: str = "1"
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.seq < 0:
            raise EventValidationError("snapshot seq cannot be negative")
        if not isinstance(self.state, dict):
            raise EventValidationError("snapshot state must be a JSON object")
        try:
            json.dumps(self.state, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise EventValidationError("snapshot state must be JSON serializable") from exc


class SnapshotStore(Protocol):
    def save(self, snapshot: EventSnapshot) -> None: ...

    def latest(self, run_id: str) -> EventSnapshot | None: ...


class InMemorySnapshotStore:
    def __init__(self) -> None:
        self._snapshots: dict[str, EventSnapshot] = {}

    def save(self, snapshot: EventSnapshot) -> None:
        prior = self._snapshots.get(snapshot.run_id)
        if prior is not None and snapshot.seq < prior.seq:
            raise SequenceConflictError("snapshot sequence cannot move backwards")
        self._snapshots[snapshot.run_id] = copy.deepcopy(snapshot)

    def latest(self, run_id: str) -> EventSnapshot | None:
        snapshot = self._snapshots.get(run_id)
        return copy.deepcopy(snapshot) if snapshot is not None else None


Reducer = Callable[[MutableMapping[str, Any], EventEnvelope], Mapping[str, Any] | None]
T = TypeVar("T")


class EventSourcedState:
    """Replay an event stream from a snapshot and optionally persist snapshots."""

    def __init__(
        self,
        log: EventLog,
        *,
        reducer: Reducer,
        snapshots: SnapshotStore | None = None,
        reducer_version: str = "1",
        initial_state: Mapping[str, Any] | None = None,
    ) -> None:
        self.log = log
        self.reducer = reducer
        self.snapshots = snapshots
        self.reducer_version = reducer_version
        self.initial_state = dict(initial_state or {})

    def replay(self, run_id: str) -> dict[str, Any]:
        snapshot = self.snapshots.latest(run_id) if self.snapshots is not None else None
        if snapshot is not None and snapshot.reducer_version != self.reducer_version:
            snapshot = None
        state: MutableMapping[str, Any] = copy.deepcopy(
            snapshot.state if snapshot is not None else self.initial_state
        )
        after_seq = snapshot.seq if snapshot is not None else 0
        if snapshot is not None and after_seq > self.log.latest_seq(run_id):
            raise SequenceConflictError(f"snapshot for {run_id!r} is ahead of its event stream")
        events = self.log.read(run_id, after_seq=after_seq)
        expected = after_seq + 1
        for event in events:
            if event.seq != expected:
                raise SequenceConflictError(
                    f"event stream for {run_id!r} has seq {event.seq}, expected {expected}"
                )
            # Snapshot records are persistence markers.  Their state was
            # already materialized into ``state`` above and must not be fed to
            # the domain reducer a second time.
            if event.event_type != _SNAPSHOT_EVENT_TYPE:
                result = self.reducer(state, event)
                if result is not None:
                    state = dict(result)
            expected += 1
        return dict(state)

    def snapshot(self, run_id: str) -> EventSnapshot:
        state = self.replay(run_id)
        seq = self.log.latest_seq(run_id)
        snapshot = EventSnapshot(
            run_id=run_id,
            seq=seq,
            state=state,
            reducer_version=self.reducer_version,
        )
        if self.snapshots is not None:
            self.snapshots.save(snapshot)
        return snapshot


__all__ = [
    "EventEnvelope",
    "EventError",
    "EventLog",
    "EventSnapshot",
    "EventSourcedState",
    "EventValidationError",
    "EventLogSnapshotStore",
    "IdempotencyConflictError",
    "InMemoryEventLog",
    "InMemorySnapshotStore",
    "RunLedgerEventLog",
    "SequenceConflictError",
    "SnapshotStore",
]
