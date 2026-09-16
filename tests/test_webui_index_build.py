"""A1: the durable index-build job — single-flight, contract and page wiring.

The build stages themselves are covered by ``tests/tree/*``; these tests pin the
job semantics the WebUI depends on:

* one deterministic slot per (database, profile, tree root): a second start
  never creates a second run, a finished run can be restarted, and a crashed
  (lease-expired) run can be taken over;
* the CLI and the WebUI run the same job, so a terminal build blocks the button
  and vice versa;
* the UI reads ``GET /api/jobs/{job_id}`` and the ``/index`` page polls the job
  fragment until a terminal state (no SSE).

Everything runs against real temporary SQLite databases; the only stub is the
stage pipeline itself (``prepare_unified_index``), which needs an embedding
endpoint no test environment has.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from drbrain.app import auth, service
from drbrain.app.web import create_app
from drbrain.services import index_build
from drbrain.storage.database import Database
from drbrain.tree.prepare import LAST_BUILD_KEY, PrepareOutcome

KIND = index_build.KIND


@pytest.fixture(autouse=True)
def _reset_index_build_state():
    service._INDEX_BUILD_WORKERS.clear()
    yield
    service._INDEX_BUILD_WORKERS.clear()
    service._RUN_MANAGER = None


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    root = tmp_path / "root"
    (root / "data").mkdir(parents=True)
    monkeypatch.setenv("DRBRAIN_ROOT", str(root))
    config = {
        "db": {"path": "data/test.db"},
        "llm": {"models": []},
        "bm25": {"k1": 1.5, "b": 0.75},
        "dirs": {"papers": "data/papers"},
        # A minimal, resolvable embedding profile: the stage pipeline is stubbed
        # in these tests, so nothing ever calls the endpoint — but the job has to
        # build the profile it scopes its slot with.
        "embed": {"provider": "local", "model": "test-embed", "dim": 8},
        "autoresearch": {"enabled": False, "run_dir": "workspace/runs", "plugins_dir": ""},
    }
    Database(root / "data" / "test.db").close()
    return config


@pytest.fixture
def web(cfg):
    token = auth.ensure_token(cfg)
    app = create_app(cfg)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/auth/verify", json={"token": token})
        assert response.status_code == 200
        yield SimpleNamespace(
            client=client, cfg=cfg, token=token, csrf=response.json()["csrf_token"]
        )


def _stub_stages(monkeypatch, *, gate: threading.Event | None = None):
    """Replace the stage pipeline with a deterministic, offline one."""
    calls: list[str] = []

    def fake_prepare(db, *, on_stage=None, **kwargs):
        if gate is not None:
            assert gate.wait(timeout=10), "the test never released the build"
        stages = (
            ("fts", {"status": "rebuilt", "indexed": 0, "blocks": 0}),
            ("vectors", {"status": "ok", "nodes": 0, "embedded": 0, "pending": 0}),
            ("hierarchy", {"status": "complete", "created": 0, "frontier_remaining": 0}),
            ("publication", {"status": "skipped", "reason": "unchanged"}),
        )
        for name, payload in stages:
            calls.append(name)
            if on_stage is not None:
                on_stage(name, payload)
        outcome = PrepareOutcome()
        for name, payload in stages:
            setattr(outcome, name, payload)
        return outcome

    monkeypatch.setattr("drbrain.tree.prepare.prepare_unified_index", fake_prepare)
    return calls


# ── core job semantics ───────────────────────────────────────────────────────


def test_slot_is_deterministic_and_single_flight(cfg):
    db = Database(":memory:")
    scope = index_build.index_build_scope_key(cfg, db, tree_storage=Path("/tmp/tree"))
    assert index_build.index_build_job_id(scope) == index_build.index_build_job_id(scope)

    row, disposition = index_build.ensure_index_build_job(db, scope, owner="worker-a")
    assert disposition == "claimed" and row["state"] == "running"
    assert index_build.job_is_active(db, row)

    second, again = index_build.ensure_index_build_job(db, scope, owner="worker-b")
    assert again == "reused" and second["owner"] == "worker-a"

    # A finished run reopens the same slot (a second build is a new attempt,
    # never a second row).
    db.finish_tree_job(row["job_id"], "done")
    third, disposition = index_build.ensure_index_build_job(db, scope, owner="worker-c")
    assert disposition == "claimed" and third["owner"] == "worker-c"
    assert third["job_id"] == row["job_id"]
    assert len(db.list_tree_jobs()) == 1


def test_lease_expiry_allows_takeover(cfg):
    db = Database(":memory:")
    scope = index_build.index_build_scope_key(cfg, db, tree_storage=Path("/tmp/tree"))
    row, _ = index_build.ensure_index_build_job(db, scope, owner="crashed")
    db.conn.execute(
        "UPDATE tree_build_jobs SET claim_expires_at = NULL WHERE job_id = ?", (row["job_id"],)
    )
    db.conn.commit()

    assert not index_build.job_is_active(db, db.get_tree_job(row["job_id"]))
    taken, disposition = index_build.ensure_index_build_job(db, scope, owner="next")
    assert disposition == "taken_over" and taken["owner"] == "next"


def test_renew_and_reopen_only_touch_the_holder():
    db = Database(":memory:")
    job_id = "job-test-renew"
    db.insert_tree_job(job_id, "scope", kind=KIND)
    assert db.reopen_tree_job(job_id, "worker") is False  # a pending slot is claimed, not reopened
    assert db.claim_tree_job(job_id, "worker")
    assert db.renew_tree_job(job_id, "worker") is True
    assert db.renew_tree_job(job_id, "someone-else") is False
    db.finish_tree_job(job_id, "done")
    assert db.renew_tree_job(job_id, "worker") is False  # terminal: nothing to renew
    assert db.reopen_tree_job(job_id, "worker") is True
    assert db.get_tree_job(job_id)["state"] == "running"


# ── the build body under the job ─────────────────────────────────────────────


def test_run_index_build_checkpoints_every_stage(cfg, monkeypatch):
    calls = _stub_stages(monkeypatch)
    db = Database(cfg["db"]["path"])
    try:
        payload = index_build.run_index_build(cfg, db=db, job_id="", owner="worker-a")
        job_id = payload["job_id"]
        row = db.get_tree_job(job_id)
    finally:
        db.close()
    assert payload["ok"] is True
    assert calls == ["fts", "vectors", "hierarchy", "publication"]
    assert row["state"] == "done"
    checkpoint = json.loads(row["checkpoint_json"])
    assert checkpoint["stages_done"] == ["lexical", "fts", "vectors", "hierarchy", "publication"]
    assert checkpoint["stage"] == ""
    metrics = json.loads(row["metrics_json"])
    assert metrics["ok"] is True and metrics["failed_stages"] == []


def test_run_index_build_reports_stage_failure_on_the_job(cfg, monkeypatch):
    def failing_prepare(db, *, on_stage=None, **kwargs):
        outcome = PrepareOutcome()
        outcome.fts = {"status": "ok"}
        outcome.vectors = {"status": "failed", "error": "no embedder"}
        outcome.hierarchy = {"status": "skipped", "reason": "vectors-failed"}
        outcome.publication = {"status": "skipped", "reason": "stage-failed"}
        if on_stage is not None:
            for name in ("fts", "vectors", "hierarchy", "publication"):
                on_stage(name, getattr(outcome, name))
        return outcome

    monkeypatch.setattr("drbrain.tree.prepare.prepare_unified_index", failing_prepare)
    db = Database(cfg["db"]["path"])
    try:
        payload = index_build.run_index_build(cfg, db=db, owner="worker-a")
        row = db.get_tree_job(payload["job_id"])
        state = index_build.index_job_payload(db, row)
    finally:
        db.close()
    assert payload["ok"] is False and payload["failed_stages"] == ["vectors"]
    assert row["state"] == "failed" and "vectors" in row["reason"]
    # The failed stage is named as such; the others stay honest.
    assert state["stage_states"]["vectors"] == "failed"
    assert state["stage_states"]["lexical"] == "done"
    assert state["failed_stages"] == ["vectors"]


def test_run_index_build_is_busy_when_a_live_lease_holds_the_slot(cfg, monkeypatch):
    _stub_stages(monkeypatch)
    db = Database(cfg["db"]["path"])
    try:
        scope = index_build.index_build_scope_key(cfg, db, tree_storage=None)
        row, _ = index_build.ensure_index_build_job(db, scope, owner="terminal-cli")
        with pytest.raises(index_build.IndexBuildBusyError):
            index_build.run_index_build(cfg, db=db, owner="webui")
        assert db.get_tree_job(row["job_id"])["owner"] == "terminal-cli"
    finally:
        db.close()


# ── service facade ───────────────────────────────────────────────────────────


def _wait_for_stage(cfg, job_id: str, *, timeout: float = 10.0) -> dict:
    """Wait until the worker owns the slot and wrote its first checkpoint.

    ``start_index_build`` returns as soon as the worker thread is launched, so
    "which stage is it in" only becomes a well-defined question once the
    worker has claimed the slot — this makes that transition observable
    instead of racing it.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = service.index_job_state(cfg, job_id)
        if state.get("stage"):
            return state
        time.sleep(0.05)
    raise AssertionError("the build never reported a stage")


def test_start_index_build_reuses_the_live_job(cfg, monkeypatch):
    gate = threading.Event()
    _stub_stages(monkeypatch, gate=gate)
    try:
        first = service.start_index_build(cfg)
        assert first["already_running"] is False and first["job_id"].startswith("job-")
        running = _wait_for_stage(cfg, first["job_id"])
        assert running["state"] == "running" and running["stage"] == "fts"

        second = service.start_index_build(cfg)
        assert second["already_running"] is True
        assert second["job_id"] == first["job_id"]
        # The reused payload carries the progress the first worker published.
        assert second["stage"] == "fts"
    finally:
        gate.set()

    for thread in list(service._INDEX_BUILD_WORKERS.values()):
        thread.join(timeout=10)
    state = service.index_job_state(cfg, first["job_id"])
    assert state["state"] == "done"
    assert service.index_jobs(cfg)[0]["job_id"] == first["job_id"]
    assert service.current_index_job(cfg)["job_id"] == first["job_id"]


def test_job_state_rejects_unknown_job(cfg):
    with pytest.raises(service.JobNotFoundError):
        service.index_job_state(cfg, "job-does-not-exist")


# ── HTTP contract ────────────────────────────────────────────────────────────


def test_api_job_contract_and_single_flight(web, monkeypatch):
    gate = threading.Event()
    _stub_stages(monkeypatch, gate=gate)
    try:
        started = web.client.post("/api/index/build", json={}, headers={"X-CSRF-Token": web.csrf})
        assert started.status_code == 202
        body = started.json()
        assert body["already_running"] is False

        again = web.client.post("/api/index/build", json={}, headers={"X-CSRF-Token": web.csrf})
        assert again.status_code == 202 and again.json()["already_running"] is True
        assert again.json()["job_id"] == body["job_id"]

        state = web.client.get(f"/api/jobs/{body['job_id']}").json()
        for key in (
            "job_id",
            "kind",
            "state",
            "stage",
            "stages_done",
            "stage_states",
            "pending",
            "counts",
            "live",
            "last_build",
            "lease_expired",
            "stale",
        ):
            assert key in state, key
        assert state["kind"] == KIND
        assert set(state["stage_states"]) == {
            "lexical",
            "fts",
            "vectors",
            "hierarchy",
            "publication",
        }
        assert state["live"]["leaves_ready"] == 0

        listing = web.client.get("/api/jobs").json()
        assert [item["job_id"] for item in listing["items"]] == [body["job_id"]]
    finally:
        gate.set()

    unknown = web.client.get("/api/jobs/job-nope")
    assert unknown.status_code == 404 and unknown.json()["code"] == "job_not_found"


def test_api_index_build_requires_csrf(web):
    assert web.client.post("/api/index/build", json={}).status_code == 403


def test_failed_job_names_the_stage_in_the_panel(web):
    """A failed run shows *which* stage failed and *why* — copyable, not JSON."""
    db = Database(web.cfg["db"]["path"])
    try:
        scope = index_build.index_build_scope_key(db=db, cfg=web.cfg, tree_storage=None)
        row, _ = index_build.ensure_index_build_job(db, scope, owner="webui-test")
        db.checkpoint_tree_job(
            row["job_id"],
            checkpoint=json.dumps(
                {"stage": "", "stages_done": list(index_build.STAGES), "pending": {}, "counts": {}}
            ),
            metrics=json.dumps({"ok": False, "failed_stages": ["hierarchy"], "changed": False}),
            owner="webui-test",
        )
        db.finish_tree_job(row["job_id"], "failed", reason="failed stages: hierarchy")
        # The recorded last build carries each stage's own error text; the panel
        # shows it as the copyable reason (FR-I5 "失败能自助定位").
        db.set_vector_metadata(
            LAST_BUILD_KEY,
            json.dumps(
                {
                    "ok": False,
                    "failed_stages": ["hierarchy"],
                    "published": "",
                    "changed": False,
                    "hierarchy": {
                        "status": "failed",
                        "error": "model role index_model: no endpoint is configured",
                    },
                }
            ),
        )
        db.commit()
        job_id = row["job_id"]
    finally:
        db.close()

    fragment = web.client.get(
        f"/ui/fragments/index-job?project_id=prj-default&job_id={job_id}",
        headers={"HX-Request": "true"},
    )
    assert fragment.status_code == 200
    assert "失败" in fragment.text
    assert "失败阶段：层次结构（主题摘要）" in fragment.text
    assert "failed stages: hierarchy" in fragment.text  # the job's own reason
    assert "no endpoint is configured" in fragment.text  # copyable stage detail
    assert "hx-trigger" not in fragment.text  # terminal: the poll stops

    state = web.client.get(f"/api/jobs/{job_id}").json()
    assert state["stage_states"]["hierarchy"] == "failed"
    assert state["failed_stages"] == ["hierarchy"]
    assert state["last_build"]["stage_errors"] == {
        "hierarchy": "model role index_model: no endpoint is configured"
    }


def test_index_page_offers_the_build_and_polls_the_job(web, monkeypatch):
    gate = threading.Event()
    _stub_stages(monkeypatch, gate=gate)
    page = web.client.get("/index")
    assert page.status_code == 200
    assert "构建索引" in page.text and 'action="/index/build"' in page.text
    assert "构建任务" in page.text
    assert "hx-trigger" not in page.text  # nothing to poll before the first run

    started = web.client.post(
        "/index/build",
        data={"project_id": "prj-default", "csrf_token": web.csrf},
        follow_redirects=False,
    )
    assert started.status_code == 303
    assert started.headers["location"].startswith("/index")

    # While the job is live the page polls it (htmx, 2 s interval, no SSE).
    running = _wait_for_stage(web.cfg, service.current_index_job(web.cfg)["job_id"])
    assert running["state"] == "running" and running["stage"] == "fts"
    live_page = web.client.get("/index")
    assert "hx-trigger" in live_page.text and "delay:2000ms" in live_page.text
    assert "进行中" in live_page.text
    gate.set()
    for thread in list(service._INDEX_BUILD_WORKERS.values()):
        thread.join(timeout=10)

    fragment = web.client.get(
        "/ui/fragments/index-job?project_id=prj-default", headers={"HX-Request": "true"}
    )
    assert fragment.status_code == 200
    assert "<html" not in fragment.text
    assert "已完成" in fragment.text
    assert "hx-trigger" not in fragment.text  # terminal jobs stop polling

    refreshed = web.client.get("/index")
    assert "已完成" in refreshed.text
