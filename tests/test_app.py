"""Tests for the WebUI service layer and the FastAPI HTTP router.

Migrated from the legacy stdlib-server suite (17 tests) to the FastAPI app:
the same behaviors are asserted under the new authentication boundary, plus
the error contract (401/404/422) and CSRF requirements.
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
from drbrain.loop.store import RunLedger
from drbrain.security import REDACTED
from drbrain.storage.database import Database


@pytest.fixture
def cfg(tmp_path: Path) -> dict:
    db_file = tmp_path / "test.db"
    Database(db_file).close()  # create schema
    return {
        "db": {"path": str(db_file)},
        "llm": {"models": []},
        "bm25": {"k1": 1.5, "b": 0.75},
        "dirs": {"papers": str(tmp_path / "papers")},
        "autoresearch": {"enabled": False, "run_dir": str(tmp_path / "ws"), "plugins_dir": ""},
    }


def _seed_ledger(cfg: dict) -> tuple[str, str]:
    """Create a ledger with one run, one proposal-ish event and a settlement."""
    run_dir = Path(cfg["autoresearch"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger = RunLedger(run_dir / "ledger.sqlite3")
    run = ledger.get_or_create_run("topological flat band", config={}, budget={})
    with ledger.transaction() as conn:
        ledger.append_event(
            conn,
            run.run_id,
            actor="analyst",
            event_type="proposal_recorded",
            payload={"claim_id": "cl-1"},
        )
        ledger.append_event(
            conn,
            run.run_id,
            actor="settle",
            event_type="claim_settled",
            payload={"verdict": "keep"},
        )
    now = time.time()
    with ledger.transaction() as conn:
        conn.execute(
            "INSERT INTO research_proposals VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "prp-1",
                run.run_id,
                "cl-1",
                "analyst",
                json.dumps({"statement": "CrF3 flat band"}),
                "critiqued",
                0.5,
                now,
                now,
            ),
        )
        conn.execute(
            "INSERT INTO research_experiments (experiment_id, run_id, proposal_id, claim_id, plan_json, environment_json, config_json, seed, status, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                "exp-1",
                run.run_id,
                "prp-1",
                "cl-1",
                json.dumps({"tool": "gpaw"}),
                "{}",
                "{}",
                42,
                "settled",
                now,
                now,
            ),
        )
        conn.execute(
            "INSERT INTO research_claim_settlements (settlement_id, run_id, experiment_id, claim_id, verdict, reason, evidence_ids_json, result_json, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "stl-1",
                run.run_id,
                "exp-1",
                "cl-1",
                "keep",
                "verified",
                json.dumps(["ev-1", "ev-2"]),
                json.dumps({"min_bandwidth_ev": 0.05}),
                now,
            ),
        )
    return run.run_id, run.topic


# ── service: empty state ──


def test_dashboard_empty_state(cfg):
    d = service.dashboard(cfg)
    assert d["papers"] == 0 and d["concepts"] == 0
    assert d["ledger"] == {"runs": 0, "settlements": 0, "verified": 0, "events": 0}
    assert d["plugins"] == 0 and d["recent_runs"] == []
    assert d["project"]["project_id"] == "prj-default"


def test_search_empty_db_and_blank_query(cfg):
    assert service.search(cfg, "   ") == []
    assert service.search(cfg, "flat band") == []


def test_ask_reports_unavailable_engine(cfg):
    out = service.ask(cfg, "what is a flat band?")
    assert out.get("unavailable") is True and "llamaindex" in out["error"]
    assert service.ask(cfg, "")["error"] == "empty question"


def test_unknown_run_readers_fail_closed(cfg):
    assert service.runs(cfg) == []
    with pytest.raises(service.RunNotFoundError):
        service.run_events(cfg, "nope")
    with pytest.raises(service.RunNotFoundError):
        service.run_claims(cfg, "nope")
    assert service.experiments(cfg) == []


def test_plugins_and_assets_without_plugin_dir(cfg):
    assert service.plugins(cfg) == []
    assert service.plugin_catalog(cfg) == []
    a = service.assets(cfg)
    assert a["plugins_dir"] is None and a["ledger"]["bytes"] is None
    assert a["database"]["bytes"] is not None
    assert {e["label"] for e in a["exports"]} >= {"BibTeX", "GraphML"}


# ── service: with a seeded ledger ──


def test_ledger_readers_with_seeded_run(cfg):
    run_id, topic = _seed_ledger(cfg)
    rs = service.runs(cfg)
    assert len(rs) == 1 and rs[0]["topic"] == topic and rs[0]["verified"] == 1
    evs = service.run_events(cfg, run_id)
    assert [e["type"] for e in evs][-2:] == ["proposal_recorded", "claim_settled"]
    assert service.run_events(cfg, run_id, after=evs[-1]["seq"]) == []
    claims = service.run_claims(cfg, run_id)
    assert claims[0]["statement"] == "CrF3 flat band" and claims[0]["verdict"] == "keep"
    assert claims[0]["evidence_ids"] == ["ev-1", "ev-2"]
    xs = service.experiments(cfg)
    assert xs[0]["experiment_id"] == "exp-1" and xs[0]["result"] == {"min_bandwidth_ev": 0.05}
    assert service.experiments(cfg, run_id="other") == []
    d = service.dashboard(cfg)
    assert d["ledger"]["verified"] == 1 and d["recent_runs"][0]["run_id"] == run_id
    detail = service.run_detail(cfg, run_id)
    assert detail["events"] == 3 and detail["claims"] == 1 and detail["verified"] == 1
    assert detail["experiments"] == 1 and detail["session_label"] == "未绑定会话"


def test_run_report_carries_verdicts_and_evidence(cfg):
    run_id, _ = _seed_ledger(cfg)
    body, media, filename = service.run_report(cfg, run_id)
    assert media.startswith("text/markdown") and filename.endswith(".md")
    assert run_id in body and "CrF3 flat band" in body and "ev-1" in body
    payload, media, _ = service.run_report(cfg, run_id, fmt="json")
    assert media.startswith("application/json")
    decoded = json.loads(payload)
    assert decoded["run"]["run_id"] == run_id and decoded["claims"][0]["verdict"] == "keep"


def test_run_manager_refuses_when_disabled(cfg):
    rm = service.RunManager()
    with pytest.raises(RuntimeError):
        rm.start(cfg, "some goal")
    with pytest.raises(ValueError):
        rm.start(cfg, "   ")
    assert rm.status("some goal") == {"topic": "some goal", "alive": False, "error": None}


def test_run_manager_runs_in_background_and_reports_errors(cfg, monkeypatch):
    cfg["autoresearch"]["enabled"] = True
    rm = service.RunManager()
    seen: list[str] = []
    started = threading.Event()

    def fake_run(self, cfg_, settings, topic, max_cycles, *scope):
        seen.append(topic)
        started.set()
        raise RuntimeError("boom")

    monkeypatch.setattr(service.RunManager, "_execute", fake_run)
    out = rm.start(cfg, "goal A", max_cycles=3)
    assert out["started"] is True and out["run_id"]
    assert started.wait(2)
    rm._threads["goal A"].join(2)
    assert seen == ["goal A"]
    assert rm.status("goal A")["error"] == "RuntimeError: boom"


def test_run_manager_redacts_error_details(cfg, monkeypatch):
    cfg["autoresearch"]["enabled"] = True
    rm = service.RunManager()
    finished = threading.Event()

    def fake_run(self, cfg_, settings, topic, max_cycles, *scope):
        finished.set()
        raise RuntimeError("Authorization: Bearer manager-secret")

    monkeypatch.setattr(service.RunManager, "_execute", fake_run)
    rm.start(cfg, "secret-free topic")
    assert finished.wait(2)
    rm._threads["secret-free topic"].join(2)
    error = rm.status("secret-free topic")["error"]
    assert "manager-secret" not in error
    assert REDACTED in error


def test_run_manager_redacts_unlabelled_configured_secret(cfg, monkeypatch):
    cfg["autoresearch"]["enabled"] = True
    secret = "opaque-manager-provider-secret"
    cfg["llm"]["models"] = [{"api_key": secret}]
    rm = service.RunManager()

    def fake_run(self, cfg_, settings, topic, max_cycles, *scope):
        raise RuntimeError(f"provider rejected {secret}")

    monkeypatch.setattr(service.RunManager, "_execute", fake_run)
    rm.start(cfg, "opaque-secret topic")
    rm._threads["opaque-secret topic"].join(2)

    error = rm.status("opaque-secret topic")["error"]
    assert secret not in error
    assert REDACTED in error


def test_run_manager_rejects_run_directory_outside_runtime_root(cfg, tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    root.mkdir()
    cfg["autoresearch"]["enabled"] = True
    cfg["autoresearch"]["run_dir"] = str(tmp_path / "outside")
    monkeypatch.setenv("DRBRAIN_ROOT", str(root))

    with pytest.raises(ValueError, match="escapes runtime root"):
        service.RunManager().start(cfg, "isolated goal")


def test_run_manager_snapshots_normalized_config_for_background_thread(tmp_path, monkeypatch):
    """A worker keeps absolute runtime paths after its caller returns."""
    root = tmp_path / "runtime"
    root.mkdir()
    cfg = {
        "db": {"path": "data/library.sqlite"},
        "autoresearch": {
            "enabled": True,
            "run_dir": "workspace/runs",
            "plugins_dir": "",
        },
    }
    monkeypatch.setenv("DRBRAIN_ROOT", str(root))
    captured: dict = {}
    finished = threading.Event()

    def fake_run(self, cfg_, settings, topic, max_cycles, *scope):
        captured["cfg"] = cfg_
        captured["settings"] = settings
        finished.set()

    monkeypatch.setattr(service.RunManager, "_execute", fake_run)
    rm = service.RunManager()
    rm.start(cfg, "isolated snapshot")
    assert finished.wait(2)
    rm._threads["isolated snapshot"].join(2)
    assert captured["cfg"]["db"]["path"] == str(root / "data" / "library.sqlite")
    assert captured["settings"].run_dir == str(root / "workspace" / "runs")


def test_service_readers_redact_legacy_ledger_payload(cfg):
    run_id, _ = _seed_ledger(cfg)
    ledger = RunLedger(Path(cfg["autoresearch"]["run_dir"]) / "ledger.sqlite3")
    with ledger.transaction() as conn:
        conn.execute(
            "INSERT INTO research_events(run_id, event_seq, actor, event_type, payload_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, 99, "legacy", "legacy", json.dumps({"api_key": "legacy-secret"}), time.time()),
        )
    payload = service.run_events(cfg, run_id)[-1]["payload"]
    assert payload["api_key"] == REDACTED


# ── HTTP: authenticated FastAPI surface ──


@pytest.fixture
def api(tmp_path, monkeypatch):
    root = tmp_path / "root"
    (root / "data").mkdir(parents=True)
    monkeypatch.setenv("DRBRAIN_ROOT", str(root))
    cfg = {
        "db": {"path": "data/test.db"},
        "llm": {"models": []},
        "bm25": {"k1": 1.5, "b": 0.75},
        "dirs": {"papers": "data/papers"},
        "autoresearch": {"enabled": False, "run_dir": "workspace/runs", "plugins_dir": ""},
    }
    Database(root / "data" / "test.db").close()
    token = auth.ensure_token(cfg)
    app = create_app(cfg)
    # The error boundary returns the JSON contract instead of raising, so the
    # redaction test can assert the response body.
    with TestClient(app, raise_server_exceptions=False) as client:
        yield SimpleNamespace(client=client, cfg=cfg, root=root, token=token, app=app)


def _login(api) -> str:
    response = api.client.post("/api/auth/verify", json={"token": api.token})
    assert response.status_code == 200
    return response.json()["csrf_token"]


def test_http_requires_authentication(api):
    fresh = TestClient(api.app)
    r = fresh.get("/", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith("/login")
    r = fresh.get("/api/dashboard")
    assert r.status_code == 401 and r.json()["code"] == "unauthorized"
    r = fresh.post("/api/runs", json={"topic": "x"})
    assert r.status_code == 401


def test_http_login_and_core_routes(api):
    csrf = _login(api)
    r = api.client.get("/")
    assert r.status_code == 200 and "概览" in r.text
    status, d = (
        api.client.get("/api/dashboard").status_code,
        api.client.get("/api/dashboard").json(),
    )
    assert status == 200 and d["papers"] == 0
    s = api.client.get("/api/search?q=flat&limit=5").json()
    assert s == {"query": "flat", "results": []}
    a = api.client.post("/api/ask", json={"question": "hi"}, headers={"X-CSRF-Token": csrf})
    assert a.status_code == 503 and a.json()["unavailable"] is True
    r = api.client.post("/api/runs", json={"topic": "goal"}, headers={"X-CSRF-Token": csrf})
    assert r.status_code == 400 and "disabled" in r.json()["error"]
    r = api.client.post("/api/runs", json={"topic": ""}, headers={"X-CSRF-Token": csrf})
    assert r.status_code == 422 and r.json()["code"] == "validation_error"
    assert api.client.get("/api/runs").json() == []
    assert api.client.get("/api/experiments").json() == []
    assert api.client.get("/api/plugins").json() == []
    assert api.client.get("/api/assets").json()["plugins"] == []
    assert api.client.get("/api/run-status?topic=x").json()["alive"] is False


def test_http_csrf_contract(api):
    _login(api)
    # Cookie-authenticated writes need the double-submit token.
    r = api.client.post("/api/ask", json={"question": "hi"})
    assert r.status_code == 403 and r.json()["code"] == "csrf"
    r = api.client.post("/api/ask", json={"question": "hi"}, headers={"X-CSRF-Token": "wrong"})
    assert r.status_code == 403
    # Bearer clients carry no ambient cookie and are exempt.
    r = api.client.post(
        "/api/ask",
        json={"question": "hi"},
        headers={"Authorization": f"Bearer {api.token}"},
    )
    assert r.status_code == 503


def test_http_run_routes_with_ledger(api):
    _login(api)
    run_id, _ = _seed_ledger(api.cfg)
    evs = api.client.get(f"/api/runs/{run_id}/events?after=0").json()
    assert evs[-1]["type"] == "claim_settled"
    claims = api.client.get(f"/api/runs/{run_id}/claims").json()
    assert claims[0]["verdict"] == "keep"
    detail = api.client.get(f"/api/runs/{run_id}").json()
    assert detail["run_id"] == run_id and detail["verified"] == 1
    report = api.client.get(f"/api/runs/{run_id}/report?format=markdown")
    assert report.status_code == 200
    assert "attachment" in report.headers["content-disposition"]
    assert run_id in report.text
    unknown = api.client.get("/api/runs/nope/events")
    assert unknown.status_code == 404 and unknown.json()["code"] == "run_not_found"
    # Unknown routes keep the JSON contract.
    missing = api.client.get("/api/nope")
    assert missing.status_code == 404


def test_http_error_response_redacts_exception_details(api, monkeypatch):
    _login(api)

    def explode(*_args, **_kwargs):
        raise RuntimeError("opaque-http-secret")

    monkeypatch.setattr(service, "dashboard", explode)
    r = api.client.get("/api/dashboard")
    assert r.status_code == 500
    assert r.json()["error"] == "internal server error"


def test_http_static_serving_and_not_found(api):
    _login(api)
    css = api.client.get("/static/app.css")
    assert css.status_code == 200 and "text/css" in css.headers["content-type"]
    htmx = api.client.get("/static/vendor/htmx.min.js")
    assert htmx.status_code == 200 and len(htmx.content) > 10_000
    assert api.client.get("/static/missing.js").status_code == 404
    assert api.client.get("/static/../pyproject.toml").status_code == 404
