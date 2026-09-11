"""M0 scope tests: project identity, session binding and scoped runs.

All persistence uses real SQLite (temporary files); nothing is mocked.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import pytest

from drbrain.app import service
from drbrain.loop.store import AmbiguousRunError, RunLedger
from drbrain.projects import DEFAULT_PROJECT_ID
from drbrain.storage.database import Database


@pytest.fixture(autouse=True)
def _fresh_run_manager():
    """The process-wide run manager must not leak worker state between tests."""
    service._RUN_MANAGER = None
    yield
    service._RUN_MANAGER = None


def _cfg(*, autoresearch_enabled: bool = False) -> dict:
    return {
        "db": {"path": "data/test.db"},
        "llm": {"models": []},
        "bm25": {"k1": 1.5, "b": 0.75},
        "dirs": {"papers": "data/papers"},
        "autoresearch": {
            "enabled": autoresearch_enabled,
            "run_dir": "workspace/runs",
            "plugins_dir": "",
        },
    }


def _root(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "root"
    root.mkdir()
    (root / "data").mkdir()
    monkeypatch.setenv("DRBRAIN_ROOT", str(root))
    return root


# ── core database ────────────────────────────────────────────────────────────


def test_default_project_seeded_and_migration_is_idempotent(tmp_path):
    db = Database(tmp_path / "t.db")
    projects = db.list_projects()
    assert [p["project_id"] for p in projects] == [DEFAULT_PROJECT_ID]
    assert projects[0]["is_default"] is True

    # Re-running the migrations must not duplicate the seeded project row.
    db._migrate()
    assert [p["project_id"] for p in db.list_projects()] == [DEFAULT_PROJECT_ID]
    db.close()


def test_legacy_agent_sessions_backfill_to_default_project(tmp_path):
    path = tmp_path / "t.db"
    db = Database(path)
    db.insert_agent_session("sess-legacy", title="old session")
    db.commit()
    # Emulate a pre-v21 database: no scope column, no project table.
    db.conn.execute("DROP INDEX IF EXISTS idx_agent_sessions_project")
    db.conn.execute("ALTER TABLE agent_sessions DROP COLUMN project_id")
    db.conn.execute("DROP TABLE projects")
    db.conn.execute("DELETE FROM schema_versions WHERE version = 21")
    db.commit()
    db.close()

    migrated = Database(path)
    row = migrated.get_agent_session("sess-legacy")
    assert row is not None and row["project_id"] == DEFAULT_PROJECT_ID
    assert migrated.get_project(DEFAULT_PROJECT_ID)["is_default"] is True
    migrated.close()


def test_project_identity_survives_rename(tmp_path):
    db = Database(tmp_path / "t.db")
    db.upsert_project("prj-abc", "第一版名称", workspace_name="ws-1")
    db.upsert_project("prj-abc", "新名称", workspace_name="ws-1")
    row = db.get_project("prj-abc")
    assert row["name"] == "新名称"
    assert len(db.list_projects()) == 2  # default + renamed  # noqa: PLR2004
    db.close()


# ── ledger scope ─────────────────────────────────────────────────────────────


def test_two_projects_can_share_one_topic(tmp_path):
    ledger = RunLedger(tmp_path / "ledger.sqlite3")
    run_a = ledger.get_or_create_run("flat band", project_id="prj-a")
    run_b = ledger.get_or_create_run("flat band", project_id="prj-b")
    assert run_a.run_id != run_b.run_id
    assert ledger.get_run("flat band", project_id="prj-a").run_id == run_a.run_id
    assert ledger.get_run("flat band", project_id="prj-b").run_id == run_b.run_id
    # Repeating the same scoped create resumes, it does not duplicate.
    again = ledger.get_or_create_run("flat band", project_id="prj-a")
    assert again.run_id == run_a.run_id


def test_same_project_sessions_are_distinct_runs(tmp_path):
    ledger = RunLedger(tmp_path / "ledger.sqlite3")
    first = ledger.get_or_create_run("topic", project_id="prj-a", session_id="s1")
    second = ledger.get_or_create_run("topic", project_id="prj-a", session_id="s2")
    assert first.run_id != second.run_id
    assert ledger.get_run("topic", project_id="prj-a", session_id="s1").run_id == first.run_id
    with pytest.raises(AmbiguousRunError):
        ledger.get_run("topic", project_id="prj-a")


def test_client_request_id_is_the_idempotency_key(tmp_path):
    ledger = RunLedger(tmp_path / "ledger.sqlite3")
    first = ledger.get_or_create_run(
        "goal", project_id="prj-a", session_id="s1", client_request_id="req-1"
    )
    # The same key resolves to the same run even if the retried body differs.
    retry = ledger.get_or_create_run(
        "goal retried", project_id="prj-a", session_id="s1", client_request_id="req-1"
    )
    assert retry.run_id == first.run_id
    # A different topic/scope with a fresh key is a new run.
    other = ledger.get_or_create_run(
        "another goal", project_id="prj-a", session_id="s1", client_request_id="req-2"
    )
    assert other.run_id != first.run_id
    # …while the same topic in the same scope always resumes the existing run,
    # regardless of which key the caller now carries.
    repeat = ledger.get_or_create_run(
        "goal", project_id="prj-a", session_id="s1", client_request_id="req-3"
    )
    assert repeat.run_id == first.run_id


def test_ledger_v8_migrates_to_scoped_runs_without_losing_events(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    ledger = RunLedger(path)
    run = ledger.get_or_create_run("legacy topic", config={"n_critics": 1})
    with ledger.transaction() as conn:
        ledger.append_event(
            conn,
            run.run_id,
            actor="analyst",
            event_type="proposal_recorded",
            payload={"claim_id": "cl-1"},
        )

    # Rebuild the table to its v8 shape (global topic UNIQUE, no scope columns).
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("PRAGMA legacy_alter_table=ON")
        conn.executescript(
            """
            BEGIN;
            CREATE TABLE research_runs_v8 (
                run_id TEXT PRIMARY KEY,
                topic TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                config_json TEXT NOT NULL DEFAULT '{}',
                budget_json TEXT NOT NULL DEFAULT '{}',
                last_projected_event INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                completed_at REAL
            );
            INSERT INTO research_runs_v8(
                run_id, topic, status, schema_version, config_json, budget_json,
                last_projected_event, created_at, updated_at, completed_at
            )
            SELECT run_id, topic, status, schema_version, config_json, budget_json,
                   last_projected_event, created_at, updated_at, completed_at
            FROM research_runs;
            DROP TABLE research_runs;
            ALTER TABLE research_runs_v8 RENAME TO research_runs;
            DELETE FROM ledger_schema_versions;
            INSERT INTO ledger_schema_versions(version, applied_at) VALUES (8, 0);
            COMMIT;
            """
        )

    migrated = RunLedger(path)
    found = migrated.get_run("legacy topic")
    assert found is not None and found.run_id == run.run_id
    assert found.project_id == DEFAULT_PROJECT_ID
    assert [e.event_type for e in migrated.events(run.run_id)][-1] == "proposal_recorded"
    # The global UNIQUE(topic) is gone: another project may reuse the topic.
    second = migrated.get_or_create_run("legacy topic", project_id="prj-b")
    assert second.run_id != run.run_id


# ── service scope ────────────────────────────────────────────────────────────


def _seed_library(root: Path, papers: list[str]) -> None:
    db = Database(root / "data" / "test.db")
    for local_id in papers:
        db.insert_paper(local_id, f"Paper {local_id}", 2024, "extracted")
    db.commit()
    db.close()


def _make_workspace(root: Path, name: str, paper_ids: list[str]) -> None:
    refs = root / "workspace" / name / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (root / "workspace" / name / "workspace.yaml").write_text(
        f"schema_version: 1\nname: {name}\ndescription: ''\n", encoding="utf-8"
    )
    (refs / "papers.json").write_text(
        json.dumps([{"local_id": pid, "added_at": "2026-09-11"} for pid in paper_ids]),
        encoding="utf-8",
    )


def test_workspace_projects_scope_papers_and_runs(tmp_path, monkeypatch):
    root = _root(tmp_path, monkeypatch)
    cfg = _cfg()
    _seed_library(root, ["p-shared", "p-ws"])
    _make_workspace(root, "flatband", ["p-ws"])

    first = service.projects(cfg, with_counts=True)
    assert [p["project_id"] for p in first if p["is_default"]] == [DEFAULT_PROJECT_ID]
    workspace_rows = [p for p in first if p["workspace_name"] == "flatband"]
    assert len(workspace_rows) == 1
    project_id = workspace_rows[0]["project_id"]
    assert workspace_rows[0]["papers"] == 1

    # Re-listing must not duplicate the synced project (idempotent migration).
    again = service.projects(cfg)
    assert [p["project_id"] for p in again if p["workspace_name"] == "flatband"] == [project_id]

    # Default project = whole library; workspace project = its references.
    default_items = service.papers(cfg)["items"]
    assert {i["local_id"] for i in default_items} == {"p-shared", "p-ws"}
    scoped = service.papers(cfg, project_id)
    assert scoped["total"] == 1
    assert [i["local_id"] for i in scoped["items"]] == ["p-ws"]
    with pytest.raises(service.PaperNotInProjectError):
        service.paper_detail(cfg, "p-shared", project_id)
    detail = service.paper_detail(cfg, "p-ws", project_id)
    assert detail["paper"]["title"] == "Paper p-ws"

    # Runs belong to exactly one project.
    ledger = RunLedger(service.ledger_path(cfg))
    default_run = ledger.get_or_create_run("scoped goal", project_id=DEFAULT_PROJECT_ID)
    scoped_run = ledger.get_or_create_run("scoped goal", project_id=project_id, session_id="s1")
    assert [r["run_id"] for r in service.runs(cfg)] == [default_run.run_id]
    assert [r["run_id"] for r in service.runs(cfg, project_id)] == [scoped_run.run_id]
    assert service.runs(cfg, project_id, session_id="s1")[0]["run_id"] == scoped_run.run_id
    assert service.runs(cfg, project_id, session_id="other") == []
    with pytest.raises(service.RunNotFoundError):
        service.run_detail(cfg, scoped_run.run_id, DEFAULT_PROJECT_ID)
    assert service.run_detail(cfg, scoped_run.run_id, project_id)["run_id"] == scoped_run.run_id


def test_unbound_runs_are_labeled_and_dashboard_is_scoped(tmp_path, monkeypatch):
    root = _root(tmp_path, monkeypatch)
    cfg = _cfg()
    _seed_library(root, ["p-1"])
    ledger = RunLedger(service.ledger_path(cfg))
    run = ledger.get_or_create_run("topic", project_id=DEFAULT_PROJECT_ID)
    detail = service.run_detail(cfg, run.run_id)
    assert detail["session_id"] == ""
    assert detail["session_label"] == "未绑定会话"
    dashboard = service.dashboard(cfg)
    assert dashboard["papers"] == 1
    assert dashboard["recent_runs"][0]["run_id"] == run.run_id
    assert dashboard["project"]["project_id"] == DEFAULT_PROJECT_ID


def test_run_launch_is_idempotent_and_persisted_before_worker(tmp_path, monkeypatch):
    root = _root(tmp_path, monkeypatch)
    cfg = _cfg(autoresearch_enabled=True)
    Database(root / "data" / "test.db").close()
    release = threading.Event()
    seen: list[tuple[str, str]] = []

    def fake_execute(self, cfg_, settings, topic, max_cycles, project_id="", session_id=""):
        seen.append((topic, project_id))
        release.wait(2)

    monkeypatch.setattr(service.RunManager, "_execute", fake_execute)
    first = service.start_run(
        cfg, "same goal", project_id=DEFAULT_PROJECT_ID, client_request_id="req-1"
    )
    # The durable row exists before the worker is observed.
    assert first["run_id"]
    detail = service.run_detail(cfg, first["run_id"])
    assert detail["client_request_id"] == "req-1"
    # A live worker keeps the durable "created" state from looking interrupted.
    assert detail["display_status"] == "created"
    # Retried request with the same key returns the same persisted run.
    second = service.start_run(
        cfg, "same goal", project_id=DEFAULT_PROJECT_ID, client_request_id="req-1"
    )
    assert second["run_id"] == first["run_id"]
    assert second["started"] is False
    release.set()
    service.run_manager()._run_threads[first["run_id"]].join(2)
    assert seen == [("same goal", DEFAULT_PROJECT_ID)]
    # The worker finishing leaves a durable row whose status is read from the ledger.
    final = service.run_detail(cfg, first["run_id"])
    assert final["status"] == "created"
    assert final["display_status"] == "created"


def test_launch_rejects_session_from_another_project(tmp_path, monkeypatch):
    root = _root(tmp_path, monkeypatch)
    cfg = _cfg(autoresearch_enabled=True)
    db = Database(root / "data" / "test.db")
    db.insert_agent_session("sess-other", title="belongs elsewhere")
    db.set_session_project("sess-other", "prj-other")
    db.commit()
    db.close()
    with pytest.raises(service.SessionNotFoundError):
        service.start_run(cfg, "goal", project_id=DEFAULT_PROJECT_ID, session_id="sess-other")
