"""WebUI M1 tests: authentication boundary, CSRF, scope isolation, SSR pages,
SSE streaming, report download, memory write-back and plugin conformance.

Everything runs against real temporary SQLite databases and real template
rendering through the FastAPI test client.
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
from drbrain.app.web.routes import auth_routes
from drbrain.loop.store import RunLedger
from drbrain.security import REDACTED
from drbrain.storage.database import Database

PLUGIN_SOURCE = '''\
"""Demo plugin used by the WebUI conformance test."""

from drbrain.plugins.protocol import Plugin


def handler(arguments):
    return {"echo": arguments}


def register(registry):
    registry.register(
        Plugin(
            name="demo",
            description="demo plugin",
            input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
            plugin_type="software",
            version="0.1",
            side_effect="pure",
        ),
        handler,
    )
'''


@pytest.fixture(autouse=True)
def _reset_login_throttle():
    auth_routes._FAILURES.clear()
    yield
    auth_routes._FAILURES.clear()
    service._RUN_MANAGER = None


@pytest.fixture
def web(tmp_path, monkeypatch):
    root = tmp_path / "root"
    (root / "data").mkdir(parents=True)
    monkeypatch.setenv("DRBRAIN_ROOT", str(root))
    cfg = {
        "db": {"path": "data/test.db"},
        "llm": {"models": []},
        "bm25": {"k1": 1.5, "b": 0.75},
        "dirs": {"papers": "data/papers"},
        "autoresearch": {
            "enabled": False,
            "run_dir": "workspace/runs",
            "plugins_dir": "",
        },
    }
    Database(root / "data" / "test.db").close()
    token = auth.ensure_token(cfg)
    app = create_app(cfg)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/auth/verify", json={"token": token})
        assert response.status_code == 200
        csrf = response.json()["csrf_token"]
        yield SimpleNamespace(client=client, cfg=cfg, root=root, token=token, csrf=csrf, app=app)


def _seed_paper(root: Path, local_id: str, title: str) -> None:
    db = Database(root / "data" / "test.db")
    db.insert_paper(local_id, title, 2024, "extracted", authors="A. Author")
    db.commit()
    db.close()


def _make_workspace(root: Path, name: str, papers: list[str]) -> None:
    refs = root / "workspace" / name / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (root / "workspace" / name / "workspace.yaml").write_text(
        f"schema_version: 1\nname: {name}\ndescription: ''\n", encoding="utf-8"
    )
    (refs / "papers.json").write_text(
        json.dumps([{"local_id": pid, "added_at": "2026-09-11"} for pid in papers]),
        encoding="utf-8",
    )


def _seed_run(
    cfg: dict, topic: str, *, project_id: str = "prj-default", status: str = "created"
) -> str:
    ledger = RunLedger(service.ledger_path(cfg))
    run = ledger.get_or_create_run(topic, project_id=project_id)
    with ledger.transaction() as conn:
        ledger.append_event(conn, run.run_id, actor="analyst", event_type="proposal_recorded")
        ledger.append_event(conn, run.run_id, actor="critic", event_type="critique_recorded")
        conn.execute("UPDATE research_runs SET status = ? WHERE run_id = ?", (status, run.run_id))
    return run.run_id


def _login_form(client: TestClient, token: str, next_url: str = "/") -> None:
    response = client.post(
        "/login", data={"token": token, "next": next_url}, follow_redirects=False
    )
    assert response.status_code == 303


# ── authentication ───────────────────────────────────────────────────────────


def test_login_cookies_are_hardened(tmp_path, monkeypatch):
    root = tmp_path / "root"
    (root / "data").mkdir(parents=True)
    monkeypatch.setenv("DRBRAIN_ROOT", str(root))
    cfg = {
        "db": {"path": "data/test.db"},
        "autoresearch": {"enabled": False, "run_dir": "workspace/runs", "plugins_dir": ""},
    }
    token = auth.ensure_token(cfg)
    app = create_app(cfg)
    with TestClient(app) as client:
        response = client.post("/login", data={"token": token, "next": "/"}, follow_redirects=False)
    cookies = response.headers.get_list("set-cookie")
    session_cookie = next(c for c in cookies if c.startswith("drbrain_webui="))
    csrf_cookie = next(c for c in cookies if c.startswith("drbrain_csrf="))
    assert "HttpOnly" in session_cookie and "SameSite=strict" in session_cookie
    assert "HttpOnly" not in csrf_cookie and "SameSite=strict" in csrf_cookie


def test_login_failures_are_throttled(tmp_path, monkeypatch):
    root = tmp_path / "root"
    (root / "data").mkdir(parents=True)
    monkeypatch.setenv("DRBRAIN_ROOT", str(root))
    cfg = {
        "db": {"path": "data/test.db"},
        "autoresearch": {"enabled": False, "run_dir": "workspace/runs", "plugins_dir": ""},
    }
    auth.ensure_token(cfg)
    app = create_app(cfg)
    with TestClient(app) as client:
        for _ in range(8):
            assert client.post("/api/auth/verify", json={"token": "bad"}).status_code == 401
        blocked = client.post("/api/auth/verify", json={"token": "bad"})
        assert blocked.status_code == 429 and blocked.json()["code"] == "throttled"


def test_logout_and_token_rotation_revoke_sessions(web):
    assert web.client.get("/api/dashboard").status_code == 200
    response = web.client.post("/api/auth/logout", headers={"X-CSRF-Token": web.csrf})
    assert response.status_code == 200
    assert web.client.get("/api/dashboard").status_code == 401

    # Re-login, then rotate: every session dies and the old token stops working.
    _login_form(web.client, web.token)
    csrf = web.client.cookies.get("drbrain_csrf")
    rotated = web.client.post("/api/auth/rotate", headers={"X-CSRF-Token": csrf})
    assert rotated.status_code == 200
    new_token = rotated.json()["token"]
    assert new_token != web.token
    assert auth.ensure_token(web.cfg) == new_token
    with TestClient(web.app) as fresh:
        assert fresh.post("/api/auth/verify", json={"token": web.token}).status_code == 401
        ok = fresh.post("/api/auth/verify", json={"token": new_token})
        assert ok.status_code == 200


def test_expired_login_session_is_rejected(web, monkeypatch):
    monkeypatch.setattr(auth, "SESSION_TTL_SECONDS", 0.01)
    fresh = TestClient(web.app)
    fresh.post("/api/auth/verify", json={"token": web.token})
    time.sleep(0.05)
    assert fresh.get("/api/dashboard").status_code == 401


def test_bearer_requests_skip_csrf_but_need_a_valid_token(web):
    # A valid Bearer credential works without a CSRF token…
    fresh = TestClient(web.app)
    ok = fresh.post(
        "/api/ask",
        json={"question": "hi"},
        headers={"Authorization": f"Bearer {web.token}"},
    )
    assert ok.status_code == 503
    # …while a wrong Bearer header is not saved by an unrelated cookie.
    cookie_less = TestClient(web.app)
    denied = cookie_less.post(
        "/api/ask",
        json={"question": "hi"},
        headers={"Authorization": "Bearer wrong"},
    )
    assert denied.status_code == 401


# ── scope isolation ──────────────────────────────────────────────────────────


def test_workspace_project_scope_filters_library_and_blocks_cross_project(web):
    _seed_paper(web.root, "p-shared", "Shared in default")
    _seed_paper(web.root, "p-ws", "Workspace only")
    _make_workspace(web.root, "flatband", ["p-ws"])
    projects = {
        p["project_id"]: p for p in web.client.get("/api/projects?counts=true").json()["items"]
    }
    ws_project = next(p for p in projects.values() if p["workspace_name"] == "flatband")
    assert ws_project["papers"] == 1

    default_page = web.client.get("/api/papers").json()
    assert {i["local_id"] for i in default_page["items"]} == {"p-shared", "p-ws"}
    scoped_page = web.client.get(f"/api/papers?project_id={ws_project['project_id']}").json()
    assert [i["local_id"] for i in scoped_page["items"]] == ["p-ws"]
    cross = web.client.get(f"/api/papers/p-shared?project_id={ws_project['project_id']}")
    assert cross.status_code == 404 and cross.json()["code"] == "paper_not_found"
    # The project list itself is stable across repeated calls (idempotent sync).
    second = web.client.get("/api/projects").json()["items"]
    assert [p["project_id"] for p in second] == [p["project_id"] for p in projects.values()]


def test_sessions_and_runs_are_project_scoped(web):
    _make_workspace(web.root, "flatband", [])
    ws_project = next(
        p
        for p in web.client.get("/api/projects").json()["items"]
        if p["workspace_name"] == "flatband"
    )["project_id"]

    created = web.client.post(
        f"/api/projects/{ws_project}/sessions",
        json={"title": "workspace session"},
        headers={"X-CSRF-Token": web.csrf},
    )
    assert created.status_code == 201
    sid = created.json()["session_id"]
    assert web.client.get(f"/api/sessions/{sid}").status_code == 200
    assert web.client.get(f"/api/sessions/{sid}?project_id=prj-default").status_code == 404
    assert web.client.get(f"/sessions/{sid}?project_id=prj-default").status_code == 404

    run_id = _seed_run(web.cfg, "scoped run", project_id=ws_project)
    assert [r["run_id"] for r in web.client.get("/api/runs").json()] == []
    assert [r["run_id"] for r in web.client.get(f"/api/runs?project_id={ws_project}").json()] == [
        run_id
    ]
    # Direct URLs resolve the entity's own scope; an explicit foreign scope 404s.
    assert web.client.get(f"/api/runs/{run_id}").json()["run_id"] == run_id
    assert web.client.get(f"/api/runs/{run_id}?project_id=prj-default").status_code == 404
    assert web.client.get(f"/api/runs/{run_id}/events?project_id=prj-default").status_code == 404
    assert web.client.get(f"/api/runs/{run_id}/claims?project_id=prj-default").status_code == 404
    report = web.client.get(f"/api/runs/{run_id}/report?project_id=prj-default")
    assert report.status_code == 404
    # The SSE route must reject a foreign scope before the stream starts.
    assert web.client.get(f"/api/runs/{run_id}/stream?project_id=prj-default").status_code == 404


def test_ambiguous_topic_status_is_explicit(web):
    _seed_run(web.cfg, "same topic")
    ledger = RunLedger(service.ledger_path(web.cfg))
    other = ledger.get_or_create_run("same topic", project_id="prj-default", session_id="s-2")
    assert other.run_id
    response = web.client.get("/api/run-status?topic=same%20topic")
    assert response.status_code == 409 and "run_id" in response.json()["error"]


# ── pages ────────────────────────────────────────────────────────────────────


def test_pages_render_with_scope_and_content(web):
    _seed_paper(web.root, "p-1", "Flat band magic")
    tree_dir = web.root / "data" / "papers" / "p-1"
    tree_dir.mkdir(parents=True)
    (tree_dir / "tree.json").write_text(
        json.dumps([{"node_id": "n-1", "title": "Introduction", "nodes": []}]),
        encoding="utf-8",
    )
    run_id = _seed_run(web.cfg, "topological flat band", status="succeeded")

    dashboard = web.client.get("/")
    assert dashboard.status_code == 200 and "概览" in dashboard.text
    assert "Flat band magic" not in dashboard.text  # dashboard shows counters, not a library dump

    papers = web.client.get("/papers?q=flat")
    assert papers.status_code == 200 and "Flat band magic" in papers.text
    assert "下一页" not in papers.text  # single page hides the cursor pager

    detail = web.client.get("/papers/p-1")
    assert detail.status_code == 200
    assert "Introduction" in detail.text and "n-1" in detail.text
    assert "带入新会话" in detail.text

    sessions = web.client.get("/sessions")
    assert sessions.status_code == 200 and "新建会话" in sessions.text

    runs = web.client.get("/runs")
    assert runs.status_code == 200 and "topological flat band" in runs.text
    assert "成功" in runs.text

    run_page = web.client.get(f"/runs/{run_id}")
    assert run_page.status_code == 200
    assert "事件流" in run_page.text and "proposal_recorded" in run_page.text
    assert "数据流" not in run_page.text

    plugins = web.client.get("/plugins")
    assert plugins.status_code == 200 and "没有发现插件" in plugins.text

    settings = web.client.get("/settings")
    assert settings.status_code == 200 and "有效配置" in settings.text

    # Unknown page entities render the explicit 404 page, not a 500.
    missing = web.client.get("/runs/nope")
    assert missing.status_code == 404 and "找不到研究运行" in missing.text


def test_fragments_render_rows_and_events(web):
    _seed_paper(web.root, "p-1", "Flat band magic")
    run_id = _seed_run(web.cfg, "fragment run")
    rows = web.client.get("/ui/fragments/paper-rows?q=magic")
    assert rows.status_code == 200 and "Flat band magic" in rows.text
    events = web.client.get(f"/ui/fragments/run-events?run_id={run_id}&limit=10")
    assert events.status_code == 200 and "proposal_recorded" in events.text
    assert web.client.get("/ui/fragments/run-events?run_id=nope").status_code == 404
    # Missing required parameters keep the JSON error contract for htmx callers.
    missing = web.client.get("/ui/fragments/run-events")
    assert missing.status_code == 422
    htmx_missing = web.client.get("/ui/fragments/run-events", headers={"HX-Request": "true"})
    assert htmx_missing.status_code == 422
    assert htmx_missing.json()["code"] == "validation_error"


def test_form_redirects_carry_codes_not_free_text(web):
    created = web.client.post(
        "/api/projects/prj-default/sessions",
        json={"title": "codes"},
        headers={"X-CSRF-Token": web.csrf},
    ).json()
    session_id = created["session_id"]
    response = web.client.post(
        f"/sessions/{session_id}/runs",
        data={"topic": "x", "max_cycles": "999999", "csrf_token": web.csrf},
        follow_redirects=False,
    )
    assert response.status_code == 303
    location = response.headers["location"]
    assert "error_code=invalid_max_cycles" in location and "999999" not in location
    page = web.client.get(location)
    assert "最大轮数需在 1–100 之间" in page.text
    # A crafted code link cannot render arbitrary text.
    spoofed = web.client.get(
        f"/sessions/{session_id}?error_code=%E6%82%A8%E5%B7%B2%E8%A2%AB%E9%AA%97"
    )
    assert "您已被骗" not in spoofed.text


def test_paper_rows_container_is_not_hx_boosted(web):
    _seed_paper(web.root, "p-1", "Flat band magic")
    page = web.client.get("/papers")
    assert page.status_code == 200
    start = page.text.find('id="paper-rows"')
    container = page.text[start : start + 120]
    assert "hx-boost" not in container  # detail links must navigate normally
    assert "hx-boost" in page.text  # the search form still boosts


def test_cursor_pagination_is_stable_and_rejects_garbage(web):
    for index in range(5):
        _seed_paper(web.root, f"p-{index}", f"Paper number {index}")
    first = web.client.get("/api/papers?limit=2").json()
    assert len(first["items"]) == 2 and first["total"] == 5 and first["next_cursor"]
    second = web.client.get(f"/api/papers?limit=2&cursor={first['next_cursor']}").json()
    third = web.client.get(f"/api/papers?limit=2&cursor={second['next_cursor']}").json()
    seen = [item["local_id"] for item in first["items"] + second["items"] + third["items"]]
    assert len(seen) == len(set(seen)) == 5
    assert third["next_cursor"] is None

    invalid = web.client.get("/api/papers?cursor=%40%40%40")
    assert invalid.status_code == 422 and invalid.json()["code"] == "invalid_cursor"
    # Sessions share the same list contract.
    created = web.client.post(
        "/api/projects/prj-default/sessions",
        json={"title": "paged"},
        headers={"X-CSRF-Token": web.csrf},
    )
    assert created.status_code == 201
    sessions = web.client.get("/api/projects/prj-default/sessions?limit=1").json()
    assert sessions["total"] == 1 and sessions["next_cursor"] is None


def test_interrupted_run_and_pending_approval_are_durable_states(web):
    run_id = _seed_run(web.cfg, "restart surfacing", status="running")
    detail = web.client.get(f"/api/runs/{run_id}").json()
    assert detail["display_status"] == "interrupted"
    page = web.client.get(f"/runs/{run_id}")
    assert page.status_code == 200 and "没有活跃工作进程" in page.text

    with RunLedger(service.ledger_path(web.cfg)).transaction() as conn:
        conn.execute(
            "INSERT INTO research_steps (step_id, run_id, step_name, status, "
            "created_at, updated_at) VALUES (?,?,?,?,?,?)",
            ("stp-1", run_id, "deploy", "waiting_approval", time.time(), time.time()),
        )
    detail = web.client.get(f"/api/runs/{run_id}").json()
    assert detail["pending_approvals"] == 1
    assert detail["pending_steps"][0]["step_name"] == "deploy"
    page = web.client.get(f"/runs/{run_id}")
    assert "等待人工审核" in page.text


# ── run launch over HTTP ─────────────────────────────────────────────────────


def test_http_run_launch_is_durable_and_idempotent(web, monkeypatch):
    web.cfg["autoresearch"]["enabled"] = True
    release = threading.Event()

    def fake_execute(self, cfg_, settings, topic, max_cycles, *scope):
        release.wait(3)

    monkeypatch.setattr(service.RunManager, "_execute", fake_execute)
    payload = {"topic": "http goal", "client_request_id": "req-1"}
    first = web.client.post("/api/runs", json=payload, headers={"X-CSRF-Token": web.csrf})
    assert first.status_code == 202
    second = web.client.post("/api/runs", json=payload, headers={"X-CSRF-Token": web.csrf})
    assert second.status_code == 202
    assert first.json()["run_id"] == second.json()["run_id"]
    assert first.json()["started"] is True and second.json()["started"] is False
    detail = web.client.get(f"/api/runs/{first.json()['run_id']}").json()
    assert detail["client_request_id"] == "req-1"
    assert detail["display_status"] == "created"
    release.set()
    service.run_manager()._run_threads[first.json()["run_id"]].join(3)
    assert first.json()["run_id"] in [r["run_id"] for r in web.client.get("/api/runs").json()]


def test_run_launch_binds_session_and_validates_ownership(web):
    web.cfg["autoresearch"]["enabled"] = True
    created = web.client.post(
        "/api/projects/prj-default/sessions",
        json={"title": "run session"},
        headers={"X-CSRF-Token": web.csrf},
    ).json()
    sid = created["session_id"]

    import drbrain.app.service as service_module

    captured = {}

    def fake_execute(self, cfg_, settings, topic, max_cycles, project_id, session_id):
        captured.update(project=project_id, session=session_id)

    original = service_module.RunManager._execute
    service_module.RunManager._execute = fake_execute
    try:
        response = web.client.post(
            "/api/runs",
            json={"topic": "session bound", "session_id": sid},
            headers={"X-CSRF-Token": web.csrf},
        )
    finally:
        service_module.RunManager._execute = original
    assert response.status_code == 202
    run_id = response.json()["run_id"]
    assert captured == {"project": "prj-default", "session": sid}
    detail = web.client.get(f"/api/runs/{run_id}").json()
    assert detail["session_id"] == sid
    watermark = detail["config"]["memory_watermark"]
    assert watermark["session_bound"] is True and watermark["captured_at"] > 0
    listed = web.client.get(f"/api/runs?session_id={sid}").json()
    assert [r["run_id"] for r in listed] == [run_id]

    foreign = web.client.post(
        "/api/runs",
        json={"topic": "foreign", "session_id": "missing-session"},
        headers={"X-CSRF-Token": web.csrf},
    )
    assert foreign.status_code == 422


# ── SSE / report / memory / conformance ─────────────────────────────────────


def test_run_stream_replays_history_and_terminates(web):
    run_id = _seed_run(web.cfg, "streamed run", status="succeeded")
    with web.client.stream("GET", f"/api/runs/{run_id}/stream") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())
    assert "event: message" in body
    assert "id: 2" in body
    assert "event: status" in body and "event: end" in body

    resumed = web.client.get(
        f"/api/runs/{run_id}/stream",
        headers={"Last-Event-ID": "1"},
    )
    assert resumed.status_code == 200
    assert "id: 2" in resumed.text and "id: 1\n" not in resumed.text

    missing = web.client.get("/api/runs/nope/stream")
    assert missing.status_code == 404


def test_report_download_and_evidence_locator(web):
    _seed_paper(web.root, "p-1", "Flat band magic")
    run_id = _seed_run(web.cfg, "report run")
    report = web.client.get(f"/api/runs/{run_id}/report?format=json")
    assert report.status_code == 200
    payload = json.loads(report.text)
    assert payload["run"]["run_id"] == run_id
    evidence = web.client.get(f"/api/runs/{run_id}/evidence/p-1:n-9")
    assert evidence.status_code == 200
    assert evidence.json()["paper"]["title"] == "Flat band magic"
    assert evidence.json()["node_id"] == "n-9"
    unknown = web.client.get(f"/api/runs/{run_id}/evidence/p-ghost:n-1")
    assert unknown.status_code == 404
    artifact = web.client.get(f"/api/runs/{run_id}/artifacts/nope")
    assert artifact.status_code == 404


def test_memory_writeback_and_promotion_are_idempotent(web):
    run_id = _seed_run(web.cfg, "memory run", status="succeeded")
    with RunLedger(service.ledger_path(web.cfg)).transaction() as conn:
        now = time.time()
        conn.execute(
            "INSERT INTO research_proposals VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "prp-9",
                run_id,
                "cl-9",
                "analyst",
                json.dumps({"statement": "memory claim"}),
                "critiqued",
                0.9,
                now,
                now,
            ),
        )
        conn.execute(
            "INSERT INTO research_experiments (experiment_id, run_id, proposal_id, claim_id, "
            "plan_json, environment_json, config_json, seed, status, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("exp-9", run_id, "prp-9", "cl-9", "{}", "{}", "{}", 7, "settled", now, now),
        )
        conn.execute(
            "INSERT INTO research_claim_settlements (settlement_id, run_id, experiment_id, claim_id, verdict, reason, evidence_ids_json, result_json, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "stl-9",
                run_id,
                "exp-9",
                "cl-9",
                "keep",
                "ok",
                json.dumps(["p-1:n-1"]),
                "{}",
                now,
            ),
        )
    # A run without a session has no memory layer to write to.
    assert service.record_run_memory(web.cfg, run_id) == 0

    session_id = f"mem-{run_id[:8]}"
    db = Database(web.root / "data" / "test.db")
    db.insert_agent_session(session_id, title="memory session")
    db.set_session_project(session_id, "prj-default")
    db.commit()
    db.close()
    with RunLedger(service.ledger_path(web.cfg)).transaction() as conn:
        conn.execute(
            "UPDATE research_runs SET session_id = ? WHERE run_id = ?", (session_id, run_id)
        )
    # Settlement write-back carries run_id + evidence and is replay-safe.
    assert service.record_run_memory(web.cfg, run_id) == 2
    assert service.record_run_memory(web.cfg, run_id) == 0
    memory = web.client.get(f"/api/sessions/{session_id}/memory").json()
    assert memory["runs"] and memory["runs"][0]["run_id"] == run_id
    entry = next(m for m in memory["runs"] if m["kind"] == "claim")["memory_id"]
    promoted = web.client.post(
        f"/api/sessions/{session_id}/memory/promote",
        json={"memory_id": entry},
        headers={"X-CSRF-Token": web.csrf},
    )
    assert promoted.status_code == 200 and promoted.json()["promoted"] is True
    again = web.client.post(
        f"/api/sessions/{session_id}/memory/promote",
        json={"memory_id": entry},
        headers={"X-CSRF-Token": web.csrf},
    )
    assert again.json()["promoted"] is False
    memory = web.client.get(f"/api/sessions/{session_id}/memory").json()
    assert any(m["layer"] == "project" for m in memory["project"])


def test_settings_view_and_pages_never_leak_secrets(tmp_path, monkeypatch):
    root = tmp_path / "root"
    (root / "data").mkdir(parents=True)
    monkeypatch.setenv("DRBRAIN_ROOT", str(root))
    secret = "sk-testsecret1234567890abcdef"
    cfg = {
        "db": {"path": "data/test.db"},
        "llm": {"models": [{"model": "gpt-test", "api_key": secret}]},
        "api": {"deepxiv_token": secret},
        "dirs": {"papers": "data/papers"},
        "autoresearch": {"enabled": False, "run_dir": "workspace/runs", "plugins_dir": ""},
    }
    Database(root / "data" / "test.db").close()
    token = auth.ensure_token(cfg)
    app = create_app(cfg)
    with TestClient(app, raise_server_exceptions=False) as client:
        client.post("/api/auth/verify", json={"token": token})
        view = client.get("/api/settings").json()
        payload = json.dumps(view, ensure_ascii=False)
        assert secret not in payload
        assert REDACTED in payload
        page = client.get("/settings")
        assert page.status_code == 200 and secret not in page.text
        dashboard = client.get("/")
        assert secret not in dashboard.text


def test_login_next_target_cannot_redirect_off_site(web):
    fresh = TestClient(web.app)
    response = fresh.post(
        "/login",
        data={"token": web.token, "next": "https://evil.example/steal"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/"
    # Protocol-relative targets are rejected too.
    fresh2 = TestClient(web.app)
    response = fresh2.post(
        "/login",
        data={"token": web.token, "next": "//evil.example/steal"},
        follow_redirects=False,
    )
    assert response.headers["location"] == "/"


def test_packaged_webui_assets_live_inside_the_package():
    """Templates/static/vendor files ship with the package (wheel-verified).

    `uv build --wheel` was checked to include these paths; this test fails fast
    if an asset moves outside the package directory.
    """
    from pathlib import Path

    from drbrain import app as app_package

    root = Path(app_package.__file__).parent
    for relative in (
        "web/templates/base.html",
        "web/templates/login.html",
        "web/templates/error.html",
        "web/static/app.css",
        "web/static/app.js",
        "web/static/vendor/htmx.min.js",
        "web/static/vendor/htmx.LICENSE",
    ):
        assert (root / relative).is_file(), relative
    license_text = (root / "web/static/vendor/htmx.LICENSE").read_text(encoding="utf-8")
    assert "permission to use" in license_text.lower()


def test_plugin_catalog_and_conformance_run(web, tmp_path):
    plugin_dir = web.root / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "demo.py").write_text(PLUGIN_SOURCE, encoding="utf-8")
    web.cfg["autoresearch"]["plugins_dir"] = "plugins"

    catalog = web.client.get("/api/plugins").json()
    assert [p["name"] for p in catalog] == ["demo"]
    assert catalog[0]["conformance_state"] == "not_tested"

    started = web.client.post("/api/plugins/demo/conformance", headers={"X-CSRF-Token": web.csrf})
    assert started.status_code == 202
    check_id = started.json()["check_id"]

    deadline = time.time() + 10
    report = None
    while time.time() < deadline:
        report = web.client.get(f"/api/plugins/demo/conformance/{check_id}").json()
        if report["status"] != "pending":
            break
        time.sleep(0.1)
    assert report and report["status"] == "passed"
    assert any(c["name"].startswith("demo.") and c["passed"] for c in report["checks"])
    catalog = web.client.get("/api/plugins").json()
    assert catalog[0]["conformance_state"] == "passed"
    assert catalog[0]["conformance"]["stale"] is False
    fragment = web.client.get(f"/ui/fragments/conformance?plugin_name=demo&check_id={check_id}")
    assert fragment.status_code == 200 and "通过" in fragment.text
    unknown = web.client.post("/api/plugins/nope/conformance", headers={"X-CSRF-Token": web.csrf})
    assert unknown.status_code == 404
