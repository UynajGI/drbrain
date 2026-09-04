"""Focused durability tests for the autoresearch SQLite run ledger."""

from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest

from drbrain.loop.director import JANITOR_STALE_SECONDS, ResearchDirector, _default_state
from drbrain.loop.events import ResearchState
from drbrain.loop.state import InvalidTransitionError
from drbrain.loop.store import LEDGER_SCHEMA_VERSION, RunBudgetExceededError, RunLedger
from drbrain.loop.transitions import TransitionService


def test_ledger_adds_config_json_to_preexisting_run_table(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE research_runs (
                run_id TEXT PRIMARY KEY,
                topic TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                budget_json TEXT NOT NULL DEFAULT '{}',
                last_projected_event INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                completed_at REAL
            )
            """
        )

    RunLedger(path).get_run("missing")

    with sqlite3.connect(path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(research_runs)")}
    assert "config_json" in columns


def test_ledger_rejects_a_newer_schema_before_altering_research_runs(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE ledger_schema_versions (version INTEGER PRIMARY KEY, applied_at REAL)"
        )
        conn.execute(
            "INSERT INTO ledger_schema_versions(version, applied_at) VALUES (?, 0)",
            (LEDGER_SCHEMA_VERSION + 1,),
        )
        conn.execute(
            """
            CREATE TABLE research_runs (
                run_id TEXT PRIMARY KEY,
                topic TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                budget_json TEXT NOT NULL DEFAULT '{}',
                last_projected_event INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                completed_at REAL
            )
            """
        )

    with pytest.raises(RuntimeError, match="newer than supported"):
        RunLedger(path).get_run("missing")

    with sqlite3.connect(path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(research_runs)")}
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    assert "config_json" not in columns
    assert "research_proposals" not in tables


def test_ledger_migrates_v4_runs_to_the_durable_front_half_tables(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    ledger = RunLedger(path)
    ledger.get_or_create_run("v4 front half")

    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE research_queue_items")
        conn.execute("DROP TABLE research_critic_reviews")
        conn.execute("DROP TABLE research_proposals")
        conn.execute("DROP TABLE research_front_half_node_specs")
        conn.execute(
            "DELETE FROM ledger_schema_versions WHERE version = ?", (LEDGER_SCHEMA_VERSION,)
        )
        conn.execute("INSERT INTO ledger_schema_versions(version, applied_at) VALUES (4, 0)")

    RunLedger(path).get_run("v4 front half")

    with sqlite3.connect(path) as conn:
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        version = conn.execute("SELECT MAX(version) FROM ledger_schema_versions").fetchone()[0]
    assert {
        "research_front_half_node_specs",
        "research_proposals",
        "research_critic_reviews",
        "research_queue_items",
        "research_execution_node_specs",
        "research_experiments",
        "research_artifacts",
        "research_claim_settlements",
        "research_champion_versions",
    } <= tables
    assert version == LEDGER_SCHEMA_VERSION


def test_ledger_migrates_v5_runs_to_execution_artifact_tables(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    ledger = RunLedger(path)
    ledger.get_or_create_run("v5 execution")

    execution_tables = {
        "research_execution_node_specs",
        "research_experiments",
        "research_artifacts",
        "research_claim_settlements",
        "research_champion_versions",
    }
    with sqlite3.connect(path) as conn:
        for table in execution_tables:
            conn.execute(f"DROP TABLE {table}")
        conn.execute(
            "DELETE FROM ledger_schema_versions WHERE version = ?", (LEDGER_SCHEMA_VERSION,)
        )
        conn.execute("INSERT INTO ledger_schema_versions(version, applied_at) VALUES (5, 0)")

    RunLedger(path).get_run("v5 execution")

    with sqlite3.connect(path) as conn:
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        version = conn.execute("SELECT MAX(version) FROM ledger_schema_versions").fetchone()[0]
    assert execution_tables <= tables
    assert version == LEDGER_SCHEMA_VERSION


def test_ledger_migrates_v6_runs_to_governance_tables(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    ledger = RunLedger(path)
    ledger.get_or_create_run("v6 governance")

    governance_tables = {"research_approval_decisions", "research_budget_usage"}
    with sqlite3.connect(path) as conn:
        for table in governance_tables:
            conn.execute(f"DROP TABLE {table}")
        conn.execute(
            "DELETE FROM ledger_schema_versions WHERE version = ?", (LEDGER_SCHEMA_VERSION,)
        )
        conn.execute("INSERT INTO ledger_schema_versions(version, applied_at) VALUES (6, 0)")

    RunLedger(path).get_run("v6 governance")

    with sqlite3.connect(path) as conn:
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        version = conn.execute("SELECT MAX(version) FROM ledger_schema_versions").fetchone()[0]
    assert governance_tables <= tables
    assert version == LEDGER_SCHEMA_VERSION


def test_ledger_migrates_v7_approval_decisions_to_consumable_grants(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    ledger = RunLedger(path)
    ledger.get_or_create_run("v7 approvals")

    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE research_approval_decisions")
        conn.execute(
            """
            CREATE TABLE research_approval_decisions (
                run_id TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                tool_call_id TEXT NOT NULL,
                decision TEXT NOT NULL,
                actor TEXT NOT NULL,
                reason TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (run_id, idempotency_key)
            )
            """
        )
        conn.execute(
            "DELETE FROM ledger_schema_versions WHERE version = ?", (LEDGER_SCHEMA_VERSION,)
        )
        conn.execute("INSERT INTO ledger_schema_versions(version, applied_at) VALUES (7, 0)")

    RunLedger(path).get_run("v7 approvals")

    with sqlite3.connect(path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(research_approval_decisions)")}
        version = conn.execute("SELECT MAX(version) FROM ledger_schema_versions").fetchone()[0]
    assert {"consumed_at", "consumed_by_tool_call_id"} <= columns
    assert version == LEDGER_SCHEMA_VERSION


def test_run_lifecycle_rejects_a_skipped_transition(tmp_path):
    ledger = RunLedger(tmp_path / "ledger.sqlite3")
    run = ledger.get_or_create_run("durable topic")
    transitions = TransitionService(ledger)

    with pytest.raises(InvalidTransitionError, match="created.*paused"):
        transitions.pause_run(run.run_id, reason="skipped")

    transitions.start_run(run.run_id)
    transitions.pause_run(run.run_id, reason="bounded_session")

    assert ledger.get_run("durable topic").status == "paused"


def test_ledger_records_a_fail_closed_rag_evidence_downgrade(tmp_path):
    ledger = RunLedger(tmp_path / "ledger.sqlite3")
    run = ledger.get_or_create_run("RAG retention topic", config={"rag_generation": "g-1"})

    event = ledger.record_rag_evidence_disabled(run.run_id, generation="g-1")

    assert event.event_type == "rag_evidence_disabled"
    assert event.payload == {"generation": "g-1", "reason": "retention_unavailable"}


def test_checkpoint_generation_stays_disabled_when_retain_fails_again(tmp_path, monkeypatch):
    """A checkpoint must not revive evidence for a generation that is not retained."""
    from drbrain.rag import indexer

    ledger = RunLedger(tmp_path / "ledger.sqlite3")
    run = ledger.get_or_create_run("checkpoint RAG retention", config={"rag_generation": "g-1"})
    director = ResearchDirector(cfg=object(), run_dir=tmp_path)
    attempts: list[str] = []

    def unavailable(_cfg, generation, _run_id):
        attempts.append(generation)
        raise OSError("generation reference storage unavailable")

    monkeypatch.setattr(indexer, "retain_index_generation", unavailable)

    director._rag_generation = director._retain_rag_generation(ledger, run.run_id, "g-1")
    director._adopt_checkpoint_rag_generation(ledger, run.run_id, "g-1")

    assert attempts == ["g-1", "g-1"]
    assert director._rag_generation is None
    assert [event.event_type for event in ledger.events(run.run_id)].count(
        "rag_evidence_disabled"
    ) == 2


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
    assert resumed[0].payload["config"] == {
        "n_critics": 2,
        # rag_engine defaults to the SQL-native engine; with no drbrain_rag.db
        # on disk the captured generation is None (no legacy snapshot involved).
        "rag_generation": None,
        "require_rag_evidence": False,
        "require_compute_tools": False,
        "compute_tool_names": ["run_python", "check_job", "read_job"],
    }
    assert resumed[0].payload["budget"] == {
        "max_cycles": 0,
        "stagnation_cycles": 7,
        "max_adaptations": 4,
    }
    assert run.budget == resumed[0].payload["budget"]


def test_resume_replaces_the_enforced_budget_with_the_audited_limits(tmp_path):
    ledger = RunLedger(tmp_path / "ledger.sqlite3")
    run = ledger.get_or_create_run("budget resume", budget={"max_model_calls": 4})

    resumed = ledger.record_resume(
        run.run_id,
        config={"n_critics": 3},
        budget={"max_model_calls": 1},
    )

    assert resumed.payload["previous_budget"] == {"max_model_calls": 4}
    assert resumed.payload["budget"] == {"max_model_calls": 1}
    assert ledger.budget_snapshot(run.run_id)["limits"] == {"max_model_calls": 1}

    TransitionService(ledger).start_run(run.run_id)
    ledger.reserve_budget(run.run_id, {"model_calls": 1})
    with pytest.raises(RunBudgetExceededError, match="model_calls"):
        ledger.reserve_budget(run.run_id, {"model_calls": 1})


def test_resume_reuses_the_generation_pinned_when_the_run_was_created(tmp_path, monkeypatch):
    """A later active-pointer change must not alter an existing run's evidence plane."""
    from drbrain.rag import indexer

    generations = iter(["g-original", "g-new-active"])
    monkeypatch.setattr(indexer, "capture_index_generation", lambda _cfg: next(generations))
    monkeypatch.setattr(indexer, "retain_index_generation", lambda *_args: True)
    topic = "pinned topic"

    first = ResearchDirector(cfg=object(), run_dir=tmp_path, n_critics=2)
    asyncio.run(first.run(topic, max_cycles=0))
    resumed = ResearchDirector(cfg=object(), run_dir=tmp_path, n_critics=2)
    asyncio.run(resumed.run(topic, max_cycles=0))

    ledger = RunLedger(tmp_path / "ledger.sqlite3")
    run = ledger.get_run(topic)
    assert run is not None
    assert run.config["rag_generation"] == "g-original"
    assert resumed._rag_generation == "g-original"
    resume_event = [
        event for event in ledger.events(run.run_id) if event.event_type == "run_resumed"
    ]
    assert resume_event[0].payload["config"]["rag_generation"] == "g-original"


def test_resume_reuses_the_strict_evidence_mode_recorded_for_the_run(tmp_path):
    topic = "strict evidence topic"
    initial = ResearchDirector(cfg=object(), run_dir=tmp_path, require_rag_evidence=True)
    asyncio.run(initial.run(topic, max_cycles=0))

    resumed = ResearchDirector(cfg=object(), run_dir=tmp_path, require_rag_evidence=False)
    asyncio.run(resumed.run(topic, max_cycles=0))

    ledger = RunLedger(tmp_path / "ledger.sqlite3")
    run = ledger.get_run(topic)
    assert run is not None
    assert run.config["require_rag_evidence"] is True
    assert resumed._require_rag_evidence is True
    resume_event = [
        event for event in ledger.events(run.run_id) if event.event_type == "run_resumed"
    ]
    assert resume_event[0].payload["config"]["require_rag_evidence"] is True


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
