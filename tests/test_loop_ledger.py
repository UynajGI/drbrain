"""Focused durability tests for the autoresearch SQLite run ledger."""

from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest

from drbrain.loop.director import JANITOR_STALE_SECONDS, ResearchDirector, _default_state
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


def test_resume_records_effective_parameters_in_the_audit_trail(tmp_path):
    topic = "resumed topic"
    director = ResearchDirector(cfg=object(), run_dir=tmp_path, n_critics=2)

    asyncio.run(director.run(topic, max_cycles=0, stagnation_cycles=3, max_adaptations=2))
    asyncio.run(director.run(topic, max_cycles=0, stagnation_cycles=7, max_adaptations=4))

    ledger = RunLedger(tmp_path / "ledger.sqlite3")
    run = ledger.get_run(topic)
    assert run is not None
    resumed = [event for event in ledger.events(run.run_id) if event.event_type == "run_resumed"]
    assert len(resumed) == 1
    assert resumed[0].payload["config"] == {"n_critics": 2}
    assert resumed[0].payload["budget"] == {
        "max_cycles": 0,
        "stagnation_cycles": 7,
        "max_adaptations": 4,
    }


def test_janitor_projection_replay_does_not_duplicate_audit_lines(tmp_path):
    topic = "janitor topic"
    director = ResearchDirector(cfg=object(), run_dir=tmp_path)
    state = _default_state(topic)
    state["cycles"] = 1
    state["results"] = [
        {
            "verifications": [
                {
                    "status": "verified",
                    "job_id": "stale-job",
                    "statement": "unsupported result",
                }
            ]
        }
    ]
    topic_dir = director._topic_dir(topic)
    jobs_dir = topic_dir / "jobs"
    jobs_dir.mkdir(parents=True)
    (topic_dir / "logs").mkdir()
    (jobs_dir / "stale-job.json").write_text(json.dumps({"started_at": 0}), encoding="utf-8")

    director._janitor_scan(topic, state)
    director._janitor_scan(topic, state)

    entries = [
        json.loads(line)
        for line in (topic_dir / "logs" / "janitor.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(entries) == 2
    assert {entry["event"] for entry in entries} == {
        "[janitor] job stale-job stale",
        "[janitor] stale job trusted",
    }
    assert entries[0]["started_at"] == 0
    assert entries[0]["stale_seconds"] > JANITOR_STALE_SECONDS
