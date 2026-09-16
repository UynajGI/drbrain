"""WebUI M1 tests: authentication boundary, CSRF, scope isolation, SSR pages,
SSE streaming, report download, memory write-back and plugin conformance.

Everything runs against real temporary SQLite databases and real template
rendering through the FastAPI test client.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from drbrain.app import auth, service
from drbrain.app.web import create_app
from drbrain.app.web.labels import index_reason
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


# ── v2 shell (M8) ────────────────────────────────────────────────────────────


def test_shell_navigation_and_asset_fingerprint(web):
    """Seven-item navigation, the shared shell and cache-busted static URLs."""
    home = web.client.get("/")
    assert home.status_code == 200
    for label in ("概览", "索引与语料", "检索与证据", "文献库", "会话", "研究运行", "设置"):
        assert label in home.text, label
    assert home.text.count('class="nav-item') == 7
    assert 'class="topbar"' in home.text and 'class="crumb"' in home.text

    # /static/* is served with a long max-age, so every asset URL must carry a
    # content fingerprint (baseline P1).
    urls = re.findall(r'/static/[^"\']+', home.text)
    assert urls and all("?v=" in url for url in urls)
    for url in urls:
        assert web.client.get(url).status_code == 200


def test_index_page_shows_three_states_without_guessing(web):
    """FR-I1: ingested / indexed / retrievable are three labelled states."""
    page = web.client.get("/index")
    assert page.status_code == 200
    assert "索引与语料" in page.text and "已入库" in page.text
    assert "已建索引" in page.text and "可检索" in page.text
    assert "三条找法" in page.text and "当前索引版本" in page.text
    assert "还没处理完的量" in page.text


def test_index_page_reads_the_cli_report_and_self_check(web):
    """FR-I3/I4: the page carries the CLI's report and a read-only self-check."""
    page = web.client.get("/index")
    assert page.status_code == 200
    assert "现在能不能搜？" in page.text
    # The raw payload stays in the diagnostic fold, not in the body.
    assert "诊断详情" in page.text and '"tree_state"' in page.text

    frag = web.client.get(
        "/ui/fragments/index-verify?project_id=prj-default", headers={"HX-Request": "true"}
    )
    assert frag.status_code == 200
    assert "<html" not in frag.text  # a fragment, not a page
    assert "自检是只读的" in frag.text


def test_index_leg_view_speaks_user_language():
    """FR-I2/D15: route legs and reason codes are translated, never pasted raw."""
    from drbrain.app.web.routes.pages import (
        _folded_route_note,
        _index_legs_view,
        _index_states_view,
    )

    report = {
        "route": {
            "requested": ["bm25", "vector", "pageindex", "raptor"],
            "legs": ["bm25", "vector", "tree"],
        },
        "states": {
            "ingested": {"ready": True, "papers": 210},
            "indexed": {"ready": False, "legs": ["fts", "vector", "tree"]},
            "retrievable": {"ready": False, "reasons": ["state.tree: no_published_generation"]},
        },
        "legs": {
            "lexical": {"ready": True, "documents": 3},
            "fts": {"ready": True, "indexed": 9},
            "vector": {
                "ready": False,
                "reasons": ["no_ready_vectors"],
                "ready_count": 0,
                "pending": 5,
                "nodes": 5,
            },
            "tree": {"ready": False, "reasons": ["no_published_generation"]},
        },
    }
    rows = _index_legs_view(report)
    assert [row["leg"]["label"] for row in rows] == ["关键词匹配", "语义相似", "按结构导航"]
    assert rows[0]["ready"] is True
    assert rows[0]["facts"] == [("词法索引文档", "3"), ("正文块已索引", "9")]
    assert rows[1]["ready"] is False
    assert rows[1]["reasons"][0]["label"] == "还有片段没算向量"
    assert rows[2]["reasons"][0]["label"] == "还没有发布过索引版本"

    states = _index_states_view(report)
    assert [state["value"] for state in states] == ["210 篇", "1/3 条腿", "未就绪"]

    note = _folded_route_note(report)
    assert "已合并为「按结构导航」" in note and "pageindex" in note and "raptor" in note
    assert _folded_route_note({"route": {"requested": ["bm25"], "legs": ["bm25"]}}) == ""
    # A state-prefixed reason is translated by its inner code, not pasted raw.
    assert index_reason("state.vector: no_ready_vectors")["label"] == "还有片段没算向量"
    assert index_reason("documents_failed:2")["hint"].startswith("运行 drbrain index status")


def test_search_page_explains_both_paths(web):
    page = web.client.get("/search")
    assert page.status_code == 200
    assert "找证据" in page.text and "要答案" in page.text
    answer_mode = web.client.get("/search?mode=answer")
    assert answer_mode.status_code == 200 and 'name="question"' in answer_mode.text


def test_search_answer_reports_status_instead_of_500(web):
    """FR-S6/A3: a missing capability is a status, never a 500 (D4)."""
    response = web.client.post(
        "/search/answer",
        data={
            "question": "what is a flat band?",
            "project_id": "prj-default",
            "csrf_token": web.csrf,
        },
    )
    assert response.status_code == 200
    assert "要答案不可用" in response.text or "还没有索引" in response.text
    assert "去索引页" in response.text and "诊断详情" in response.text

    empty = web.client.post(
        "/search/answer",
        data={"question": "   ", "project_id": "prj-default", "csrf_token": web.csrf},
        follow_redirects=False,
    )
    assert empty.status_code == 303
    assert "error_code=empty_question" in empty.headers["location"]


def test_ask_reports_reason_and_hint_for_an_unprepared_index(web, monkeypatch):
    """A3: the service says *why* and *what to do*; the API only 503s on 'off'."""
    from drbrain.rag import engine as rag_engine

    # Pretend the engine is selected: the index is then the only thing missing.
    # (An isolated patcher: undoing the fixture's monkeypatch would drop the
    # runtime root and every session with it.)
    engine_patch = pytest.MonkeyPatch()
    engine_patch.setattr(rag_engine, "resolve_engine", lambda cfg, name: "llamaindex")
    try:
        out = service.ask(web.cfg, "what is a flat band?")
        assert out["status"] == "source_unavailable"
        assert out["unavailable"] is True and out["unavailable_reason"] == "index_not_prepared"
        assert out["hint"] and out["sources"] == [] and out["evidence_ids"] == []

        api = web.client.post(
            "/api/ask", json={"question": "hi"}, headers={"X-CSRF-Token": web.csrf}
        )
        assert api.status_code == 200  # an unprepared index is a state, not a 503
    finally:
        engine_patch.undo()

    # With the engine switched off (the real config) it is a 503 — the only 503.
    disabled = service.ask(web.cfg, "hi")
    assert disabled["unavailable_reason"] == "engine_disabled"
    assert disabled["status"] == "source_unavailable" and disabled["hint"]
    off = web.client.post("/api/ask", json={"question": "hi"}, headers={"X-CSRF-Token": web.csrf})
    assert off.status_code == 503


def test_paper_detail_outline_expands_and_locates(web):
    """FR-L2/L3: outline nodes carry an excerpt and a copyable position."""
    _seed_paper(web.root, "p-1", "Flat band magic")
    tree_dir = web.root / "data" / "papers" / "p-1"
    tree_dir.mkdir(parents=True)
    (tree_dir / "tree.json").write_text(
        json.dumps(
            [
                {
                    "node_id": "n-1",
                    "title": "Introduction",
                    "text": "Legacy body text about flat bands.",
                    "nodes": [],
                }
            ]
        ),
        encoding="utf-8",
    )

    page = web.client.get("/papers/p-1")
    assert page.status_code == 200
    assert "Legacy body text about flat bands" in page.text
    assert "p-1#n-1@legacy" in page.text  # the copyable locator
    assert "旧格式目录文件" in page.text  # provider is named, not implied

    focused = web.client.get("/papers/p-1?node=n-1")
    assert focused.status_code == 200
    assert "已定位到" in focused.text and "is-target" in focused.text

    # A locator that no longer exists is an explicit state, not an empty page.
    missing = web.client.get("/papers/p-1?node=gone")
    assert missing.status_code == 200 and "没找到这个片段" in missing.text


def test_session_memory_is_grouped_and_not_mixed_with_claims(web):
    """FR-C2/C4: memory layers are explicit and never dressed up as verdicts."""
    created = web.client.post(
        "/api/projects/prj-default/sessions",
        json={"title": "layers"},
        headers={"X-CSRF-Token": web.csrf},
    ).json()
    page = web.client.get(f"/sessions/{created['session_id']}")
    assert page.status_code == 200
    assert "会话记忆" in page.text
    assert "不等于经过验证的研究结论" in page.text
    assert "本会话的运行" in page.text


def test_interrupted_run_points_at_a_resumable_next_step(web):
    """FR-R3: 已中断（可恢复） is a state with a way forward, not a failure."""
    run_id = _seed_run(web.cfg, "resume me", status="running")
    page = web.client.get(f"/runs/{run_id}")
    assert page.status_code == 200
    assert "已中断（可恢复）" in page.text and "重新发起同一目标" in page.text
    assert "失败" not in page.text.split("已中断（可恢复）")[0][-40:]

    # The next step is pre-filled, so resuming is one click, not retyping.
    listing = web.client.get("/runs?topic=resume+me")
    assert 'value="resume me"' in listing.text
    # One status→label mapping: the list says the same thing as the detail page.
    assert "已中断（可恢复）" in web.client.get("/runs").text


def test_run_claims_group_verified_and_unverified(web):
    """FR-R4: settled verdicts are separated from predicted/unsettled claims."""
    run_id = _seed_run(web.cfg, "claims split", status="succeeded")
    with RunLedger(service.ledger_path(web.cfg)).transaction() as conn:
        now = time.time()
        conn.execute(
            "INSERT INTO research_proposals VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "prp-1",
                run_id,
                "cl-1",
                "analyst",
                json.dumps({"statement": "kept claim"}),
                "critiqued",
                0.9,
                now,
                now,
            ),
        )
        conn.execute(
            "INSERT INTO research_proposals VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "prp-2",
                run_id,
                "cl-2",
                "analyst",
                json.dumps({"statement": "unsettled claim"}),
                "proposed",
                0.4,
                now,
                now,
            ),
        )
        conn.execute(
            "INSERT INTO research_experiments (experiment_id, run_id, proposal_id, claim_id, "
            "plan_json, environment_json, config_json, seed, status, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("exp-1", run_id, "prp-1", "cl-1", "{}", "{}", "{}", 1, "settled", now, now),
        )
        conn.execute(
            "INSERT INTO research_claim_settlements (settlement_id, run_id, experiment_id, "
            "claim_id, verdict, reason, evidence_ids_json, result_json, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "stl-1",
                run_id,
                "exp-1",
                "cl-1",
                "keep",
                "holds",
                json.dumps(["p-1:n-1"]),
                json.dumps({"supports": 3, "refutes": 0, "orthogonal": 1}),
                now,
            ),
        )
    page = web.client.get(f"/runs/{run_id}")
    assert page.status_code == 200
    assert "研究结论" in page.text
    assert "通过验证" in page.text and "未通过 / 尚未验证" in page.text
    assert "支持 3" in page.text and "正交 1" in page.text
    # A paper:node locator is a page (literature + focused node), not raw JSON.
    assert "/papers/p-1?node=n-1" in page.text
    # The verdict wording comes from labels.py, so both pages agree.
    assert "保留" in page.text


def test_evidence_search_scope_and_display_cap(web, monkeypatch):
    """FR-S8/A2: the display cap is applied and reported, never a fake total."""
    from drbrain.services import evidence_search as core

    captured: dict = {}

    def fake_run(cfg, query, *, limit=20, paper_ids=None, source="local"):
        captured.update(limit=limit, paper_ids=paper_ids, source=source)
        return {
            "query": query,
            "status": "ok",
            "hint": "",
            "engine": "sql",
            "source": source,
            "route": {},
            "generations": {},
            "legs": [],
            "evidence": [],
        }

    monkeypatch.setattr(core, "run_evidence_search", fake_run)
    out = service.evidence_search(web.cfg, "kagome", limit=500, project_id="prj-default")
    assert captured["limit"] == 100  # capped at MAX_EVIDENCE_LIMIT
    assert out["display_cap"] == 100
    assert out["scope"]["best_effort"] is False
    # The API declares the same cap, so an over-large limit is a validation error.
    assert web.client.get("/api/search/evidence?q=x&limit=500").status_code == 422


def test_evidence_search_page_renders_the_four_elements(web, monkeypatch):
    """FR-S2/S4: passage, source with a locator, finder, index version."""
    from drbrain.services import evidence_search as core

    def fake_run(cfg, query, *, limit=20, paper_ids=None, source="local"):
        if query != "kagome":  # the "nothing matched" case must stay a state
            return {
                "query": query,
                "status": "empty",
                "hint": "",
                "engine": "sql",
                "source": source,
                "route": {"legs": ["tree"], "extras": [], "notes": []},
                "generations": {},
                "legs": [{"source": "tree", "status": "empty", "count": 0, "reason": ""}],
                "evidence": [],
            }
        return {
            "query": query,
            "status": "ok",
            "hint": "",
            "engine": "sql",
            "source": source,
            "route": {"legs": ["tree"], "extras": [], "notes": []},
            "generations": {"result": "gen-1", "tree": "gen-1", "sql": None},
            "legs": [
                {"source": "tree", "status": "ok", "count": 1, "duration_ms": 1.0, "reason": ""}
            ],
            "evidence": [
                {
                    "evidence_id": "ev-1",
                    "paper_id": "p-1",
                    "node_id": "n-1",
                    "title": "Flat band magic",
                    "text": "kagome flat band evidence",
                    "score": 0.9,
                    "source": "tree",
                    "char_start": 10,
                    "char_end": 40,
                    "generation": "gen-1",
                }
            ],
        }

    monkeypatch.setattr(core, "run_evidence_search", fake_run)
    _seed_paper(web.root, "p-1", "Flat band magic")
    page = web.client.get("/search?q=kagome&mode=evidence")
    assert page.status_code == 200
    assert "kagome flat band evidence" in page.text  # ① the passage
    assert "Flat band magic" in page.text and "/papers/p-1" in page.text
    assert "node=n-1" in page.text  # ② source, linking to the focused node
    assert "找法：tree" in page.text  # ③ which finder produced it
    assert "gen-1" in page.text  # ④ the index version
    assert "原文片段" in page.text
    assert "展示上限，不是命中总数" in page.text
    # An empty result is "not found", not a broken page.
    empty = web.client.get("/search?q=nothing&mode=evidence")
    assert empty.status_code == 200 and "没有找到证据" in empty.text


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


def test_empty_workspace_reports_zero_counts(web):
    _seed_paper(web.root, "p-1", "Library paper")
    _make_workspace(web.root, "empty-ws", [])
    ws_project = next(
        p
        for p in web.client.get("/api/projects").json()["items"]
        if p["workspace_name"] == "empty-ws"
    )["project_id"]
    dash = web.client.get(f"/api/dashboard?project_id={ws_project}").json()
    assert (dash["papers"], dash["concepts"], dash["edges"], dash["arguments"]) == (0, 0, 0, 0)
    assert web.client.get("/api/dashboard").json()["papers"] == 1


def test_core_evidence_respects_project_membership(web):
    _seed_paper(web.root, "p-in", "Member paper")
    _seed_paper(web.root, "p-out", "Outside paper")
    _make_workspace(web.root, "evidence-ws", ["p-in"])
    ws_project = next(
        p
        for p in web.client.get("/api/projects").json()["items"]
        if p["workspace_name"] == "evidence-ws"
    )["project_id"]
    db = Database(web.root / "data" / "test.db")
    outside_evidence = db.record_evidence("p-out", "n-1")
    db.close()
    run_id = _seed_run(web.cfg, "evidence scope", project_id=ws_project)
    blocked = web.client.get(
        f"/api/runs/{run_id}/evidence/{outside_evidence}?project_id={ws_project}"
    )
    assert blocked.status_code == 404
    member = web.client.get(f"/api/runs/{run_id}/evidence/p-in:n-1?project_id={ws_project}")
    assert member.status_code == 200


def test_run_event_history_uses_tail_and_before_cursor(web):
    run_id = _seed_run(web.cfg, "history run")
    ledger = RunLedger(service.ledger_path(web.cfg))
    with ledger.transaction() as conn:
        for index in range(120):
            ledger.append_event(
                conn, run_id, actor="analyst", event_type="tick", payload={"i": index}
            )
    tail = service.run_events_tail(web.cfg, run_id, limit=100)
    assert len(tail) == 100
    everything = service.run_events(web.cfg, run_id, limit=1000)
    assert tail[-1]["seq"] == everything[-1]["seq"]
    older = service.run_events(web.cfg, run_id, before=tail[0]["seq"], limit=100)
    assert older and [e["seq"] for e in older] == sorted(e["seq"] for e in older)
    assert all(e["seq"] < tail[0]["seq"] for e in older)
    assert older[-1]["seq"] == tail[0]["seq"] - 1
    # The page opens on the tail and offers the earlier-history cursor.
    page = web.client.get(f"/runs/{run_id}")
    assert "首屏为最近 100 条" in page.text
    assert f"before={tail[0]['seq']}" in page.text
    api = web.client.get(f"/api/runs/{run_id}/events?before={tail[0]['seq']}&limit=100").json()
    assert api and api[-1]["seq"] == tail[0]["seq"] - 1


def test_session_run_list_carries_display_status(web):
    created = web.client.post(
        "/api/projects/prj-default/sessions",
        json={"title": "interrupted session"},
        headers={"X-CSRF-Token": web.csrf},
    ).json()
    session_id = created["session_id"]
    run_id = _seed_run(web.cfg, "session interrupted", status="running")
    ledger = RunLedger(service.ledger_path(web.cfg))
    with ledger.transaction() as conn:
        conn.execute(
            "UPDATE research_runs SET session_id = ? WHERE run_id = ?",
            (session_id, run_id),
        )
    data = web.client.get(f"/api/sessions/{session_id}").json()
    assert data["runs"][0]["display_status"] == "interrupted"
    page = web.client.get(f"/sessions/{session_id}")
    assert "已中断" in page.text


def test_project_search_restricts_candidates_before_ranking(web):
    for index in range(40):
        _seed_paper(web.root, f"noise-{index}", f"common term {index}")
    # A very long member document ranks below the global top-5, so a
    # post-filter would silently lose it.
    _seed_paper(web.root, "target", "common term " + "filler " * 200)
    _make_workspace(web.root, "narrow-ws", ["target"])
    ws_project = next(
        p
        for p in web.client.get("/api/projects").json()["items"]
        if p["workspace_name"] == "narrow-ws"
    )["project_id"]
    scoped = service.search(web.cfg, "common", limit=5, project_id=ws_project)
    assert [row["local_id"] for row in scoped] == ["target"]
    unscoped = service.search(web.cfg, "common", limit=5)
    assert "target" not in {row["local_id"] for row in unscoped}


# ── review fixes: evidence/answer scope + readiness probe ────────────────────


def _workspace_project(web, name: str) -> str:
    return next(
        p for p in web.client.get("/api/projects").json()["items"] if p["workspace_name"] == name
    )["project_id"]


def _capture_evidence_core(monkeypatch) -> dict:
    """Keep ``run_evidence_search`` offline and record the scope it was given."""
    from drbrain.services import evidence_search as core

    captured: dict = {}

    def fake_run(cfg, query, *, limit=20, paper_ids=None, source="local"):
        captured.update(limit=limit, paper_ids=paper_ids, source=source)
        return {
            "query": query,
            "status": "ok",
            "hint": "",
            "engine": "sql",
            "source": source,
            "route": {},
            "generations": {},
            "legs": [],
            "evidence": [],
        }

    monkeypatch.setattr(core, "run_evidence_search", fake_run)
    return captured


def test_evidence_scope_intersects_explicit_paper_ids(web, monkeypatch):
    """P1(a): an explicit filter can never widen a project's membership."""
    _make_workspace(web.root, "rev-ws", ["p-in"])
    captured = _capture_evidence_core(monkeypatch)
    pid = _workspace_project(web, "rev-ws")

    out = service.evidence_search(web.cfg, "q", paper_ids=["p-in", "p-out"], project_id=pid)
    assert captured["paper_ids"] == ["p-in"]  # the out-of-scope id is dropped
    assert out["scope"]["paper_ids"] == ["p-in"]
    assert out["scope"]["reason"] == ""

    empty = service.evidence_search(web.cfg, "q", paper_ids=["p-out"], project_id=pid)
    # An explicit filter that matches nothing is an empty scope, not "no filter".
    assert captured["paper_ids"] == []
    assert empty["scope"]["reason"] == "out_of_scope_paper_ids"


def test_empty_workspace_evidence_scope_is_never_unrestricted(web, monkeypatch):
    """P1(b): an empty project must not coerce ``[]`` into an unfiltered read."""
    _seed_paper(web.root, "p-1", "Library paper")
    _make_workspace(web.root, "rev-empty-ws", [])
    captured = _capture_evidence_core(monkeypatch)
    pid = _workspace_project(web, "rev-empty-ws")

    out = service.evidence_search(web.cfg, "q", project_id=pid)

    assert captured["paper_ids"] == []  # never None (which reads the whole corpus)
    assert out["scope"]["paper_ids"] == []
    assert out["scope"]["reason"] == "empty_project"


def test_project_scoped_answer_drops_out_of_scope_citations(web, monkeypatch):
    """P1(c): the answer path cannot present another project's papers."""
    import drbrain.rag.engine as engine

    _seed_paper(web.root, "p-in", "Member paper")
    _seed_paper(web.root, "p-out", "Outside paper")
    _make_workspace(web.root, "rev-answer-ws", ["p-in"])
    pid = _workspace_project(web, "rev-answer-ws")
    monkeypatch.setattr(engine, "resolve_engine", lambda cfg, name: "llamaindex")

    def fake_ask(cfg, db, question, top_k=5, **kwargs):
        return {
            "question": question,
            "answer": "synthesized",
            "engine": "llamaindex",
            "sources": [
                {"paper_id": "p-out", "node_id": "n-1"},
                {"paper_id": "p-in", "node_id": "n-2"},
            ],
            "evidence_ids": ["p-out:n-1", "p-in:n-2"],
        }

    monkeypatch.setattr(engine, "ask_llamaindex", fake_ask)
    out = service.ask(web.cfg, "compare", project_id=pid)
    assert [s["paper_id"] for s in out["sources"]] == ["p-in"]
    assert out["evidence_ids"] == ["p-in:n-2"]
    assert out["scope"]["dropped_sources"] == 1

    # Nothing in scope → an explicit state, never an answer about unseen papers.
    monkeypatch.setattr(
        engine,
        "ask_llamaindex",
        lambda cfg, db, question, top_k=5, **kwargs: {
            "question": question,
            "answer": "synthesized",
            "engine": "llamaindex",
            "sources": [{"paper_id": "p-out", "node_id": "n-1"}],
            "evidence_ids": ["p-out:n-1"],
        },
    )
    blocked = service.ask(web.cfg, "compare", project_id=pid)
    assert blocked["status"] == "permission_denied" and blocked["sources"] == []
    from drbrain.app.web.labels import answer_status

    assert answer_status(blocked)["key"] == "permission_denied"


def test_search_readiness_follows_a_non_unified_route(web):
    """P2: a route without the unified tree is not reported unready for lacking it."""
    cfg = dict(web.cfg)
    cfg["llamaindex"] = {"enabled": True, "rag_engine": "sql", "retrievers": ["bm25", "vector"]}

    ready = service._search_readiness(cfg)

    assert ready["ready"] is True and ready["reasons"] == []
    assert ready["legs"] == ["bm25", "vector"]


def test_search_page_renders_the_redirect_error_code(web):
    """P2: an empty question comes back as a rendered message, not silence."""
    page = web.client.get("/search?project_id=prj-default&mode=answer&error_code=empty_question")
    assert page.status_code == 200
    assert "请输入问题" in page.text
