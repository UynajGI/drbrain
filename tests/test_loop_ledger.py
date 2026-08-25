"""Focused durability tests for the autoresearch SQLite run ledger."""

from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest

from drbrain.loop.director import ResearchDirector, _default_state
from drbrain.loop.events import ResearchState
from drbrain.loop.state import InvalidTransitionError
from drbrain.loop.store import RunLedger
from drbrain.loop.transitions import TransitionService


def test_run_lifecycle_rejects_a_skipped_transition(tmp_path):
    ledger = RunLedger(tmp_path / "ledger.sqlite3")
    run = ledger.get_or_create_run("durable topic")
    transitions = TransitionService(ledger)

    with pytest.raises(InvalidTransitionError, match="created.*paused"):
        transitions.pause_run(run.run_id, reason="skipped")

    transitions.start_run(run.run_id)
    transitions.pause_run(run.run_id, reason="bounded_session")

    assert ledger.get_run("durable topic").status == "paused"


def test_interrupted_cycle_is_marked_unknown_for_a_later_resume(tmp_path):
    ledger = RunLedger(tmp_path / "ledger.sqlite3")
    run = ledger.get_or_create_run("interrupted topic")
    transitions = TransitionService(ledger)
    transitions.start_run(run.run_id)
    step_id = transitions.begin_cycle(run.run_id, cycle=1)

    transitions.reconcile_incomplete_cycles(run.run_id)

    with sqlite3.connect(ledger.path) as conn:
        step_status = conn.execute(
            "SELECT status FROM research_steps WHERE step_id = ?", (step_id,)
        ).fetchone()[0]
        attempt_status = conn.execute(
            "SELECT status FROM research_attempts WHERE step_id = ?", (step_id,)
        ).fetchone()[0]
    assert step_status == "unknown"
    assert attempt_status == "unknown"
    assert ledger.events(run.run_id)[-1].event_type == "cycle_interrupted"


def test_director_replays_a_committed_cycle_after_projection_failure(tmp_path, monkeypatch):
    run_root = tmp_path / "runs"
    topic = "durable topic"
    director = ResearchDirector(cfg=object(), run_dir=run_root)

    async def fake_run_cycle(*_args, **_kwargs):
        return "cycle report", ResearchState(task=topic, verified=["durable conclusion"])

    monkeypatch.setattr(director, "_run_cycle", fake_run_cycle)

    def fail_projection(*_args, **_kwargs):
        raise OSError("simulated projection interruption")

    monkeypatch.setattr(director, "_save_state", fail_projection)

    with pytest.raises(OSError, match="projection interruption"):
        asyncio.run(director.run(topic, max_cycles=1))

    ledger = RunLedger(run_root / "ledger.sqlite3")
    run = ledger.get_run(topic)
    assert run is not None
    assert run.last_projected_event == 0
    assert [event.event_type for event in ledger.pending_projection_events(run.run_id)] == [
        "cycle_completed"
    ]

    recovered = ResearchDirector(cfg=object(), run_dir=run_root)
    state = asyncio.run(recovered.run(topic, max_cycles=1))

    topic_dir = run_root / "durable-topic"
    assert state["cycles"] == 1
    assert (topic_dir / "run.json").exists()
    assert (topic_dir / "champion.md").exists()
    assert (topic_dir / "results" / "cycle-001.md").exists()
    assert ledger.pending_projection_events(run.run_id) == []
    assert ledger.get_run(topic).status == "paused"

    experiment_lines = [
        json.loads(line)
        for line in (topic_dir / "logs" / "experiments.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [entry["cycle"] for entry in experiment_lines] == [1]


def test_legacy_workspace_is_imported_once_without_rewriting_history(tmp_path):
    topic = "legacy topic"
    director = ResearchDirector(cfg=object(), run_dir=tmp_path)
    state = _default_state(topic)
    state["cycles"] = 2
    state["champion"].append({"statement": "existing conclusion", "cycle": 1, "confidence": 1.0})
    director._save_state(topic, state)

    asyncio.run(director.run(topic, max_cycles=0))

    ledger = RunLedger(tmp_path / "ledger.sqlite3")
    run = ledger.get_run(topic)
    assert run is not None
    event_types = [event.event_type for event in ledger.events(run.run_id)]
    assert event_types.count("legacy_snapshot_imported") == 1

    asyncio.run(director.run(topic, max_cycles=0))
    event_types = [event.event_type for event in ledger.events(run.run_id)]
    assert event_types.count("legacy_snapshot_imported") == 1
