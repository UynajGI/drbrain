"""Server-rendered pages (Jinja2) and their form actions.

Pages read the same service models as the JSON API; form actions redirect back
to a page so refresh never resubmits.  Ownership is verified by the service
layer on every entity-bearing route.
"""

from __future__ import annotations

import secrets
from collections.abc import Iterable
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, Form, Query, Request
from fastapi.responses import RedirectResponse, Response

from drbrain.app import auth, service
from drbrain.app.web import deps, labels
from drbrain.projects import DEFAULT_PROJECT_ID

router = APIRouter(dependencies=[Depends(deps.authenticate), Depends(deps.require_csrf)])

MAX_FORM_CYCLES = 100


def _query(path: str, **params: str) -> str:
    pairs = "&".join(
        f"{quote(str(key))}={quote(str(value))}"
        for key, value in params.items()
        if value not in (None, "")
    )
    return f"{path}?{pairs}" if pairs else path


def _evidence_links(evidence_ids: Iterable[str], project_id: str) -> dict[str, dict[str, str]]:
    """Where each evidence reference can actually be read.

    A ``paper:node`` locator has a real page (the literature detail with that
    node focused and highlighted); anything else is only resolvable through
    the run-scoped JSON view, so the chip keeps pointing there rather than
    pretending a page exists for it.
    """
    links: dict[str, dict[str, str]] = {}
    for raw in evidence_ids:
        evidence_id = str(raw)
        paper_id, _, node_id = evidence_id.partition(":")
        if paper_id and node_id:
            links[evidence_id] = {
                "kind": "paper",
                "href": (
                    f"/papers/{quote(paper_id, safe='')}"
                    f"?node={quote(node_id, safe='')}&project_id={quote(project_id, safe='')}"
                ),
            }
        else:
            links[evidence_id] = {"kind": "core", "href": ""}
    return links


def _parse_cycles(raw: str) -> int | None:
    """Parse the form's max_cycles with the same bound as the JSON API."""
    text = raw.strip()
    if not text:
        return None
    if not text.isdigit() or not (1 <= int(text) <= MAX_FORM_CYCLES):
        raise ValueError("invalid max_cycles")
    return int(text)


# ── dashboard ────────────────────────────────────────────────────────────────


@router.get("/")
def dashboard_page(request: Request, project_id: str = Query(DEFAULT_PROJECT_ID)) -> Response:
    project = deps.resolve_project(request, project_id)
    data = service.dashboard(deps.get_cfg(request), project["project_id"])
    return deps.render(request, "dashboard.html", nav="dashboard", project=project, data=data)


# ── index & corpus ───────────────────────────────────────────────────────────

#: Route leg → the index-leg keys that actually serve it (``index status`` reports
#: index legs; the page talks about the three ways of finding something).
_ROUTE_LEG_DETAIL: dict[str, tuple[str, ...]] = {
    "bm25": ("lexical", "fts"),
    "vector": ("vector",),
    "tree": ("tree",),
}


def _leg_facts(route_leg: str, parts: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """The few numbers a reader needs for one finder (no raw JSON in the UI)."""
    if route_leg == "bm25":
        lexical = parts[0] if parts else {}
        fts = parts[1] if len(parts) > 1 else {}
        return [
            ("词法索引文档", str(lexical.get("documents", 0))),
            ("正文块已索引", str(fts.get("indexed", 0))),
        ]
    if route_leg == "vector":
        vector = parts[0] if parts else {}
        return [
            ("已就绪向量", str(vector.get("ready_count", 0))),
            ("待补", str(vector.get("pending", 0))),
            ("片段总数", str(vector.get("nodes", 0))),
        ]
    if route_leg == "tree":
        tree = parts[0] if parts else {}
        return [
            ("原文片段", str(tree.get("leaves", 0))),
            ("主题摘要", str(tree.get("regions", 0))),
            ("待归父主题", str(tree.get("leaves_missing_parent", 0))),
        ]
    return []


def _index_legs_view(report: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per user-facing finder: ready?, why not, and the key numbers."""
    legs = report.get("legs") or {}
    rows: list[dict[str, Any]] = []
    for route_leg in (report.get("route") or {}).get("legs") or []:
        keys = _ROUTE_LEG_DETAIL.get(str(route_leg), (str(route_leg),))
        parts = [legs.get(key) or {} for key in keys]
        reasons = [r for part in parts for r in (part.get("reasons") or [])]
        rows.append(
            {
                "leg": labels.index_leg(route_leg),
                "ready": bool(parts) and all(bool(part.get("ready")) for part in parts),
                "reasons": [labels.index_reason(r) for r in reasons],
                "facts": _leg_facts(str(route_leg), parts),
            }
        )
    return rows


def _index_states_view(report: dict[str, Any]) -> list[dict[str, str]]:
    """The three states as one honest number each (FR-I1/D4: never one boolean)."""
    states = report.get("states") or {}
    legs = report.get("legs") or {}
    indexed_keys = [str(key) for key in ((states.get("indexed") or {}).get("legs") or [])]
    ready = sum(1 for key in indexed_keys if (legs.get(key) or {}).get("ready"))
    ingested = states.get("ingested") or {}
    retrievable = states.get("retrievable") or {}
    return [
        {
            "key": "ingested",
            "value": f"{ingested.get('papers', 0)} 篇",
            "tone": "ok" if ingested.get("ready") else "muted",
        },
        {
            "key": "indexed",
            "value": f"{ready}/{len(indexed_keys)} 条腿" if indexed_keys else "0 条腿",
            "tone": "ok"
            if indexed_keys and ready == len(indexed_keys)
            else ("warn" if ready else "muted"),
        },
        {
            "key": "retrievable",
            "value": "就绪" if retrievable.get("ready") else "未就绪",
            "tone": "ok" if retrievable.get("ready") else "warn",
        },
    ]


def _folded_route_note(report: dict[str, Any]) -> str:
    """A sentence when the configured route was folded (pageindex/raptor → tree).

    The core records this in English ``notes``; the UI must say it in the
    user's words (§1.3) instead of pasting the raw note into the page.
    """
    requested = [str(item) for item in (report.get("route") or {}).get("requested") or []]
    folded = [name for name in requested if name in ("pageindex", "raptor")]
    if not folded:
        return ""
    return (
        "配置里的 " + "、".join(folded) + " 已合并为「按结构导航」（同一条腿，一票），"
        "不是缺失的找法。"
    )


@router.get("/index")
def index_page(
    request: Request,
    project_id: str = Query(DEFAULT_PROJECT_ID),
    error_code: str = Query(""),
) -> Response:
    """索引与语料：能不能搜、搜的是哪个版本、还差什么。

    Reads the same report as ``drbrain index status`` (04-arch A2) and says the
    numbers in user language; the raw payload stays in the diagnostic fold.
    ``job`` is the durable build job (04-arch A1) — the page renders it and a
    poller keeps it fresh until it reaches a terminal state.
    """
    cfg = deps.get_cfg(request)
    project = deps.resolve_project(request, project_id)
    report = service.index_status(cfg, project["project_id"])
    return deps.render(
        request,
        "index_corpus.html",
        nav="index",
        project=project,
        index_report=report,
        index_legs=_index_legs_view(report),
        index_states=_index_states_view(report),
        folded_note=_folded_route_note(report),
        job=service.current_index_job(cfg, project["project_id"]),
        error=labels.error_text(error_code),
        assets=service.assets(cfg),
    )


@router.post("/index/build")
def start_index_build_form(
    request: Request,
    project_id: str = Form(DEFAULT_PROJECT_ID),
) -> Response:
    """Start the corpus build (or rejoin the live one) and come back to /index.

    Re-submitting is harmless: the service returns the same job and never starts
    a second worker; the page then shows that job's progress.
    """
    cfg = deps.get_cfg(request)
    project = deps.resolve_project(request, project_id)
    try:
        service.start_index_build(cfg, project_id=project["project_id"])
    except Exception as exc:  # noqa: BLE001 - a refused start is a page status, not a crash
        from loguru import logger

        from drbrain.security import configured_secret_values, safe_error

        logger.error(
            "[webui] index build refused: {}",
            safe_error(exc, secrets=configured_secret_values(cfg)),
        )
        return RedirectResponse(
            _query("/index", project_id=project["project_id"], error_code="index_build_refused"),
            status_code=303,
        )
    return RedirectResponse(_query("/index", project_id=project["project_id"]), status_code=303)


# ── search & evidence ────────────────────────────────────────────────────────


@router.get("/search")
def search_page(
    request: Request,
    project_id: str = Query(DEFAULT_PROJECT_ID),
    mode: str = Query("evidence"),
    q: str = Query(""),
    paper: str = Query(""),
    source: str = Query("local"),
    limit: int = Query(0),
    error_code: str = Query(""),
) -> Response:
    """检索与证据：要证据（原文片段）或要答案（带出处）。

    Evidence mode reads the same payload as ``drbrain search`` (04-arch A2); the
    result is capped and the page says so, because the retrieval layer has no
    total to report (FR-S8 rewrite).  ``error_code`` renders a redirect outcome
    (an empty question, say) through the shared error-code table — never raw
    text from the URL.
    """
    cfg = deps.get_cfg(request)
    project = deps.resolve_project(request, project_id)
    evidence = None
    status = None
    if mode != "answer" and q.strip():
        paper_ids = [item.strip() for item in paper.split(",") if item.strip()]
        evidence = service.evidence_search(
            cfg,
            q,
            limit=(limit or 20),
            paper_ids=paper_ids or None,
            source=source,
            project_id=project["project_id"],
        )
        status = labels.evidence_status(evidence)
    return deps.render(
        request,
        "search.html",
        nav="search",
        project=project,
        mode=mode or "evidence",
        question="",
        answer=None,
        evidence=evidence,
        evidence_status=status,
        evidence_query=q,
        paper_filter=paper,
        source=source,
        error=labels.error_text(error_code),
    )


def _ask_failure_payload(exc: Exception) -> dict[str, Any]:
    """Last-resort shape when ``service.ask`` itself blew up.

    The service owns the capability classification now (04-arch A3): a missing
    index or a disabled engine comes back as a structured status, never as an
    exception.  This only covers a genuinely unexpected failure, so the page
    still renders a state instead of a 500.
    """
    return {"error": "retrieval failed", "status": "search_failed"}


@router.post("/search/answer")
def search_answer_form(
    request: Request,
    question: str = Form(""),
    project_id: str = Form(DEFAULT_PROJECT_ID),
) -> Response:
    """要答案：检索 + 综合，答案带出处（沿用现有 ``service.ask`` 契约）。

    A retrieval failure is a *status* the page can explain, never a 500: the
    service contract already turns "engine unavailable" into a structured
    result, and anything else is caught here and shown as a status.
    """
    cfg = deps.get_cfg(request)
    project = deps.resolve_project(request, project_id)
    question = question.strip()
    if not question:
        return RedirectResponse(
            _query(
                "/search",
                project_id=project["project_id"],
                mode="answer",
                error_code="empty_question",
            ),
            status_code=303,
        )
    try:
        answer = service.ask(cfg, question, project_id=project["project_id"])
    except Exception as exc:  # noqa: BLE001 - a failed retrieval is a status, not a crash
        from loguru import logger

        from drbrain.security import configured_secret_values, safe_error

        logger.error(
            "[webui] ask failed: {}", safe_error(exc, secrets=configured_secret_values(cfg))
        )
        answer = _ask_failure_payload(exc)
    return deps.render(
        request,
        "search.html",
        nav="search",
        project=project,
        mode="answer",
        question=question,
        answer=answer,
        answer_status=labels.answer_status(answer),
    )


# ── literature ───────────────────────────────────────────────────────────────


@router.get("/papers")
def papers_page(
    request: Request,
    project_id: str = Query(DEFAULT_PROJECT_ID),
    q: str = Query(""),
    status: str = Query(""),
    cursor: str = Query(""),
) -> Response:
    project = deps.resolve_project(request, project_id)
    data = service.papers(
        deps.get_cfg(request),
        project["project_id"],
        query=q,
        status=status or None,
        cursor=cursor or None,
    )
    return deps.render(
        request,
        "papers.html",
        nav="papers",
        project=project,
        data=data,
        q=q,
        status=status,
    )


@router.get("/papers/{local_id:path}")
def paper_detail_page(
    request: Request,
    local_id: str,
    project_id: str = Query(DEFAULT_PROJECT_ID),
    node: str = Query(""),
) -> Response:
    """One paper: metadata, outline with excerpts, concepts and arguments.

    ``node`` focuses one structural node (an outline anchor or an evidence
    locator from a run): the matching node is highlighted and open, and the
    page always shows the copyable position so the reference can be handed on.
    """
    project = deps.resolve_project(request, project_id)
    detail = service.paper_detail(deps.get_cfg(request), local_id, project["project_id"])
    return deps.render(
        request,
        "paper_detail.html",
        nav="papers",
        project=project,
        detail=detail,
        node_target=node,
    )


# ── sessions ─────────────────────────────────────────────────────────────────


@router.get("/sessions")
def sessions_page(
    request: Request,
    project_id: str = Query(DEFAULT_PROJECT_ID),
    cursor: str = Query(""),
    new: str = Query(""),
    paper: str = Query(""),
) -> Response:
    project = deps.resolve_project(request, project_id)
    data = service.sessions(deps.get_cfg(request), project["project_id"], cursor=cursor or None)
    prefill_title = ""
    prefill_paper = ""
    if paper and new:
        try:
            detail = service.paper_detail(deps.get_cfg(request), paper, project["project_id"])
            prefill_title = f"关于《{detail['paper']['title']}》"
            prefill_paper = paper
        except service.PaperNotInProjectError:
            prefill_paper = ""
    return deps.render(
        request,
        "sessions.html",
        nav="sessions",
        project=project,
        data=data,
        prefill_title=prefill_title,
        prefill_paper=prefill_paper,
        new_session="1" if new else "",
    )


@router.post("/sessions")
def create_session_form(
    request: Request,
    project_id: str = Form(DEFAULT_PROJECT_ID),
    title: str = Form(""),
    paper_id: str = Form(""),
) -> Response:
    cfg = deps.get_cfg(request)
    project = deps.resolve_project(request, project_id)
    system_prompt = ""
    if paper_id:
        try:
            detail = service.paper_detail(cfg, paper_id, project["project_id"])
            paper = detail["paper"]
            title = title.strip() or f"关于《{paper['title']}》"
            system_prompt = (
                f"本次会话围绕本地文献 {paper_id}（《{paper['title']}》）展开；"
                "引用时请使用其 local_id 与章节 node_id 作为证据定位。"
            )
        except service.PaperNotInProjectError:
            paper_id = ""
    session = service.create_session(
        cfg,
        project["project_id"],
        title=title,
        system_prompt=system_prompt,
    )
    return RedirectResponse(
        _query(f"/sessions/{session['session_id']}", project_id=project["project_id"]),
        status_code=303,
    )


@router.get("/sessions/{session_id}")
def session_detail_page(
    request: Request,
    session_id: str,
    project_id: str = Query(""),
    error_code: str = Query(""),
    synced: str = Query(""),
) -> Response:
    cfg = deps.get_cfg(request)
    pid = project_id or service.session_project(cfg, session_id)
    project = deps.resolve_project(request, pid)
    data = service.get_session(cfg, session_id, project["project_id"])
    return deps.render(
        request,
        "session_detail.html",
        nav="sessions",
        project=project,
        data=data,
        error=labels.error_text(error_code),
        synced=synced,
        client_request_id=secrets.token_hex(8),
    )


@router.post("/sessions/{session_id}/chat")
def session_chat_form(
    request: Request,
    session_id: str,
    question: str = Form(""),
    project_id: str = Form(""),
) -> Response:
    cfg = deps.get_cfg(request)
    pid = project_id or service.session_project(cfg, session_id)
    project = deps.resolve_project(request, pid)
    if deps.is_htmx(request):
        try:
            result = service.chat_in_session(cfg, session_id, question, project["project_id"])
        except ValueError:
            return deps.render(
                request,
                "fragments/session_messages.html",
                messages=[],
                unavailable=False,
                error=labels.error_text("empty_question"),
            )
        return deps.render(
            request,
            "fragments/session_messages.html",
            messages=result.get("messages", []),
            unavailable=result.get("unavailable", False),
            error="" if result.get("unavailable") else result.get("error", ""),
        )
    try:
        service.chat_in_session(cfg, session_id, question, project["project_id"])
    except ValueError:
        return RedirectResponse(
            _query(
                f"/sessions/{session_id}",
                project_id=project["project_id"],
                error_code="empty_question",
            ),
            status_code=303,
        )
    return RedirectResponse(
        _query(f"/sessions/{session_id}", project_id=project["project_id"]),
        status_code=303,
    )


@router.post("/sessions/{session_id}/runs")
def start_run_form(
    request: Request,
    session_id: str,
    topic: str = Form(""),
    max_cycles: str = Form(""),
    client_request_id: str = Form(""),
    project_id: str = Form(""),
) -> Response:
    cfg = deps.get_cfg(request)
    pid = project_id or service.session_project(cfg, session_id)
    project = deps.resolve_project(request, pid)
    try:
        cycles = _parse_cycles(max_cycles)
    except ValueError:
        return RedirectResponse(
            _query(
                f"/sessions/{session_id}",
                project_id=project["project_id"],
                error_code="invalid_max_cycles",
            ),
            status_code=303,
        )
    try:
        started = service.start_run(
            cfg,
            topic,
            max_cycles=cycles,
            project_id=project["project_id"],
            session_id=session_id,
            client_request_id=client_request_id.strip() or None,
        )
    except RuntimeError:
        code = "autoresearch_disabled"
    except ValueError:
        code = "empty_topic" if not topic.strip() else "run_start_failed"
    else:
        return RedirectResponse(
            _query(
                f"/runs/{started['run_id']}",
                project_id=project["project_id"],
            ),
            status_code=303,
        )
    return RedirectResponse(
        _query(
            f"/sessions/{session_id}",
            project_id=project["project_id"],
            error_code=code,
        ),
        status_code=303,
    )


@router.post("/sessions/{session_id}/memory/promote")
def promote_memory_form(
    request: Request,
    session_id: str,
    memory_id: str = Form(""),
    project_id: str = Form(""),
) -> Response:
    cfg = deps.get_cfg(request)
    pid = project_id or service.session_project(cfg, session_id)
    project = deps.resolve_project(request, pid)
    try:
        service.promote_memory(cfg, session_id, memory_id, project["project_id"])
    except ValueError:
        return RedirectResponse(
            _query(
                f"/sessions/{session_id}",
                project_id=project["project_id"],
                error_code="memory_promote_failed",
            ),
            status_code=303,
        )
    return RedirectResponse(
        _query(f"/sessions/{session_id}", project_id=project["project_id"], synced="promoted"),
        status_code=303,
    )


# ── runs ─────────────────────────────────────────────────────────────────────


@router.get("/runs")
def runs_page(
    request: Request,
    project_id: str = Query(DEFAULT_PROJECT_ID),
    error_code: str = Query(""),
    topic: str = Query(""),
    session_id: str = Query(""),
) -> Response:
    """Runs list plus the launch form.

    ``topic``/``session_id`` prefill the launch form so a page can link back
    to "start the same thing again" (this is how an interrupted run offers a
    resumable next step instead of a dead end).
    """
    cfg = deps.get_cfg(request)
    project = deps.resolve_project(request, project_id)
    items = service.runs(cfg, project["project_id"], limit=100)
    for row in items:
        row["display_status"] = service.display_run_status(cfg, row)
    sessions_list = service.sessions(cfg, project["project_id"], limit=100)["items"]
    return deps.render(
        request,
        "runs.html",
        nav="runs",
        project=project,
        items=items,
        sessions=sessions_list,
        error=labels.error_text(error_code),
        client_request_id=secrets.token_hex(8),
        prefill_topic=topic,
        prefill_session=session_id,
    )


@router.post("/runs")
def start_run_from_runs(
    request: Request,
    topic: str = Form(""),
    session_id: str = Form(""),
    max_cycles: str = Form(""),
    client_request_id: str = Form(""),
    project_id: str = Form(DEFAULT_PROJECT_ID),
) -> Response:
    cfg = deps.get_cfg(request)
    project = deps.resolve_project(request, project_id)
    try:
        cycles = _parse_cycles(max_cycles)
    except ValueError:
        return RedirectResponse(
            _query("/runs", project_id=project["project_id"], error_code="invalid_max_cycles"),
            status_code=303,
        )
    try:
        started = service.start_run(
            cfg,
            topic,
            max_cycles=cycles,
            project_id=project["project_id"],
            session_id=session_id.strip(),
            client_request_id=client_request_id.strip() or None,
        )
    except RuntimeError:
        code = "autoresearch_disabled"
    except ValueError:
        code = "empty_topic" if not topic.strip() else "run_start_failed"
    else:
        return RedirectResponse(
            _query(f"/runs/{started['run_id']}", project_id=project["project_id"]),
            status_code=303,
        )
    return RedirectResponse(
        _query("/runs", project_id=project["project_id"], error_code=code),
        status_code=303,
    )


@router.get("/runs/{run_id}")
def run_detail_page(
    request: Request,
    run_id: str,
    project_id: str = Query(""),
    synced: str = Query(""),
) -> Response:
    cfg = deps.get_cfg(request)
    pid = project_id or service.run_project(cfg, run_id)
    project = deps.resolve_project(request, pid)
    detail = service.run_detail(cfg, run_id, project["project_id"])
    claims = service.run_claims(cfg, run_id, project_id=project["project_id"])
    experiments_list = service.experiments(cfg, run_id=run_id, project_id=project["project_id"])
    # Open on the newest events: the SSE stream can only append forward, so the
    # page must not start from the oldest slice (it would show a gap).
    events = service.run_events_tail(cfg, run_id, limit=100, project_id=project["project_id"])
    evidence_links = _evidence_links(
        [eid for claim in claims for eid in (claim.get("evidence_ids") or [])],
        project["project_id"],
    )
    return deps.render(
        request,
        "run_detail.html",
        nav="runs",
        project=project,
        detail=detail,
        claims=claims,
        experiments=experiments_list,
        events=events,
        evidence_links=evidence_links,
        synced=synced,
    )


@router.post("/runs/{run_id}/memory/sync")
def sync_run_memory_form(
    request: Request,
    run_id: str,
    project_id: str = Form(""),
) -> Response:
    cfg = deps.get_cfg(request)
    pid = project_id or service.run_project(cfg, run_id)
    project = deps.resolve_project(request, pid)
    written = service.record_run_memory(cfg, run_id, project["project_id"])
    return RedirectResponse(
        _query(f"/runs/{run_id}", project_id=project["project_id"], synced=str(written)),
        status_code=303,
    )


# ── plugins / settings ───────────────────────────────────────────────────────


@router.get("/plugins")
def plugins_page(request: Request) -> Response:
    cfg = deps.get_cfg(request)
    catalog = service.plugin_catalog(cfg)
    return deps.render(request, "plugins.html", nav="plugins", catalog=catalog)


@router.post("/plugins/{plugin_name}/conformance")
def start_conformance_form(request: Request, plugin_name: str) -> Response:
    cfg = deps.get_cfg(request)
    try:
        service.start_conformance(cfg, plugin_name)
    except ValueError as exc:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail=str(exc))
    return RedirectResponse("/plugins", status_code=303)


@router.get("/settings")
def settings_page(request: Request, token: str = Query("")) -> Response:
    cfg = deps.get_cfg(request)
    view = service.settings_view(cfg)
    return deps.render(request, "settings.html", nav="settings", view=view, new_token=token)


@router.post("/settings/token/rotate")
async def rotate_token_form(request: Request) -> Response:
    cfg = deps.get_cfg(request)
    form = await request.form()
    if str(form.get("confirm") or "") != "yes":
        return RedirectResponse("/settings", status_code=303)
    new_token = auth.rotate_bootstrap_token(cfg)
    view = service.settings_view(cfg)
    response = deps.render(request, "settings.html", nav="settings", view=view, new_token=new_token)
    response.delete_cookie(auth.SESSION_COOKIE, path="/")
    response.delete_cookie(auth.CSRF_COOKIE, path="/")
    return response
