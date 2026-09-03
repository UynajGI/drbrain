"""Tests for the WebUI service layer and HTTP router (`drbrain webui`)."""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from drbrain.app import service
from drbrain.app.server import WebUIServer
from drbrain.loop.store import RunLedger
from drbrain.storage.database import Database


@pytest.fixture
def cfg(tmp_path: Path) -> dict:
    db_file = tmp_path / "test.db"
    Database(db_file).close()  # create schema
    return {
        "db": {"path": str(db_file)},
        "llm": {"models": []},
        "bm25": {"k1": 1.5, "b": 0.75},
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


def test_search_empty_db_and_blank_query(cfg):
    assert service.search(cfg, "   ") == []
    assert service.search(cfg, "flat band") == []


def test_ask_reports_unavailable_engine(cfg):
    out = service.ask(cfg, "what is a flat band?")
    assert out.get("unavailable") is True and "llamaindex" in out["error"]
    assert service.ask(cfg, "")["error"] == "empty question"


def test_ledger_readers_without_ledger(cfg):
    assert service.runs(cfg) == []
    assert service.run_events(cfg, "nope") == []
    assert service.run_claims(cfg, "nope") == []
    assert service.experiments(cfg) == []


def test_plugins_and_assets_without_plugin_dir(cfg):
    assert service.plugins(cfg) == []
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

    def fake_run(self, cfg_, settings, topic, max_cycles):
        seen.append(topic)
        started.set()
        raise RuntimeError("boom")

    monkeypatch.setattr(service.RunManager, "_execute", fake_run)
    out = rm.start(cfg, "goal A", max_cycles=3)
    assert out["started"] is True
    assert started.wait(2)
    rm._threads["goal A"].join(2)
    assert seen == ["goal A"]
    assert rm.status("goal A")["error"] == "RuntimeError: boom"


# ── HTTP router ──


@pytest.fixture
def server(cfg):
    srv = WebUIServer(("127.0.0.1", 0), cfg)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _get(url: str):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, json.loads(r.read().decode())


def _post(url: str, body: dict):
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def test_http_index_and_api_routes(server, cfg):
    srv, base = server
    with urllib.request.urlopen(base + "/", timeout=5) as r:
        html = r.read().decode()
    assert "DrBrain" in html and "研究闭环" in html
    status, d = _get(base + "/api/dashboard")
    assert status == 200 and d["papers"] == 0
    status, s = _get(base + "/api/search?q=flat&limit=5")
    assert status == 200 and s == {"query": "flat", "results": []}
    status, a = _post(base + "/api/ask", {"question": "hi"})
    assert status == 503 and a["unavailable"] is True
    status, r = _post(base + "/api/runs", {"topic": "goal"})
    assert status == 400 and "disabled" in r["error"]
    status, r = _post(base + "/api/runs", {"topic": ""})
    assert status == 400
    assert _get(base + "/api/runs")[1] == []
    assert _get(base + "/api/experiments")[1] == []
    assert _get(base + "/api/plugins")[1] == []
    assert _get(base + "/api/assets")[1]["plugins"] == []
    assert _get(base + "/api/run-status?topic=x")[1]["alive"] is False


def test_http_run_routes_with_ledger(server, cfg):
    srv, base = server
    run_id, _ = _seed_ledger(cfg)
    status, evs = _get(base + f"/api/runs/{run_id}/events?after=0")
    assert status == 200 and evs[-1]["type"] == "claim_settled"
    status, claims = _get(base + f"/api/runs/{run_id}/claims")
    assert status == 200 and claims[0]["verdict"] == "keep"


def test_http_not_found_and_static_escape(server):
    srv, base = server
    for path in ("/api/nope", "/static/../server.py", "/static/missing.js"):
        try:
            urllib.request.urlopen(base + path, timeout=5)
        except urllib.error.HTTPError as e:
            assert e.code == 404
        else:  # pragma: no cover
            pytest.fail(f"{path} should 404")
