from __future__ import annotations

import pytest

from drbrain.loop.research_events import (
    EventEnvelope,
    EventLogSnapshotStore,
    EventSourcedState,
    EventValidationError,
    IdempotencyConflictError,
    InMemoryEventLog,
    InMemorySnapshotStore,
    RunLedgerEventLog,
    SequenceConflictError,
)
from drbrain.loop.store import RunLedger


def test_memory_log_assigns_sequence_and_replays_idempotently() -> None:
    log = InMemoryEventLog()
    first = log.append(run_id="run", event_type="created", actor="director", idempotency_key="k")
    again = log.append(run_id="run", event_type="created", actor="director", idempotency_key="k")
    assert first == again
    assert first.seq == 1
    assert log.latest_seq("run") == 1


def test_memory_log_rejects_conflicting_retry_and_stale_sequence() -> None:
    log = InMemoryEventLog()
    log.append(
        run_id="run", event_type="created", actor="director", payload={"x": 1}, idempotency_key="k"
    )
    with pytest.raises(IdempotencyConflictError):
        log.append(
            run_id="run",
            event_type="created",
            actor="director",
            payload={"x": 2},
            idempotency_key="k",
        )
    with pytest.raises(SequenceConflictError):
        log.append(run_id="run", event_type="next", actor="director", expected_seq=1)


def test_replay_uses_snapshot_and_detects_gaps() -> None:
    log = InMemoryEventLog()
    snapshots = InMemorySnapshotStore()

    def reducer(state: dict[str, int], event: EventEnvelope) -> dict[str, int]:
        state["n"] = state.get("n", 0) + int(event.payload.get("delta", 0))
        return state

    log.append(run_id="run", event_type="delta", actor="worker", payload={"delta": 2})
    log.append(run_id="run", event_type="delta", actor="worker", payload={"delta": 3})
    state = EventSourcedState(log, reducer=reducer, snapshots=snapshots, initial_state={"n": 0})
    assert state.replay("run") == {"n": 5}
    assert state.snapshot("run").seq == 2
    log.append(run_id="run", event_type="delta", actor="worker", payload={"delta": 4})
    assert state.replay("run") == {"n": 9}


def test_envelope_rejects_non_json_payload() -> None:
    with pytest.raises(EventValidationError):
        EventEnvelope.create(
            run_id="run", event_type="x", actor="a", seq=1, payload={"bad": object()}
        )


def test_run_ledger_adapter_round_trip(tmp_path) -> None:
    ledger = RunLedger(tmp_path / "ledger.sqlite")
    run = ledger.get_or_create_run("topic")
    log = RunLedgerEventLog(ledger)
    event = log.append(
        run_id=run.run_id,
        event_type="created",
        actor="director",
        payload={"topic": "topic"},
        schema_version=2,
        idempotency_key="create-1",
    )
    assert log.read(run.run_id)[-1] == event
    assert (
        log.append(
            run_id=run.run_id,
            event_type="created",
            actor="director",
            payload={"topic": "topic"},
            schema_version=2,
            idempotency_key="create-1",
        )
        == event
    )


def test_run_ledger_adapter_reads_legacy_events_without_metadata(tmp_path) -> None:
    """Old ledger rows remain replayable after the typed envelope migration."""
    ledger = RunLedger(tmp_path / "legacy-ledger.sqlite")
    run = ledger.get_or_create_run("legacy topic")
    with ledger.transaction() as conn:
        legacy = ledger.append_event(
            conn,
            run.run_id,
            actor="director",
            event_type="cycle_completed",
            payload={"cycle": 1, "result": {"ok": True}},
        )

    event = RunLedgerEventLog(ledger).read(run.run_id)[-1]
    assert event.event_id == f"legacy:{run.run_id}:{legacy.event_seq}"
    assert event.seq == legacy.event_seq
    assert event.event_type == "cycle_completed"
    assert event.payload == legacy.payload


def test_ledger_event_log_persists_snapshot_and_replays_after_restart(tmp_path) -> None:
    db_path = tmp_path / "snapshot-ledger.sqlite"
    ledger = RunLedger(db_path)
    run = ledger.get_or_create_run("snapshot topic")
    log = RunLedgerEventLog(ledger)
    snapshots = EventLogSnapshotStore(log)

    def reducer(state: dict[str, int], event: EventEnvelope) -> dict[str, int]:
        state["n"] = state.get("n", 0) + int(event.payload.get("delta", 0))
        return state

    log.append(run_id=run.run_id, event_type="delta", actor="worker", payload={"delta": 2})
    state = EventSourcedState(log, reducer=reducer, snapshots=snapshots, initial_state={"n": 0})
    snapshot = state.snapshot(run.run_id)
    # ``RunLedger.get_or_create_run`` contributes its durable run_created row.
    assert snapshot.seq == 2
    assert snapshots.latest(run.run_id) == snapshot

    # The snapshot is an append-only ledger record; a fresh process can load
    # it and replay only the tail of the stream.
    log.append(run_id=run.run_id, event_type="delta", actor="worker", payload={"delta": 3})
    restarted_log = RunLedgerEventLog(RunLedger(db_path))
    restarted = EventSourcedState(
        restarted_log,
        reducer=reducer,
        snapshots=restarted_log,
        initial_state={"n": 0},
    )
    assert restarted.replay(run.run_id) == {"n": 5}
    assert restarted.snapshot(run.run_id).seq == 4
