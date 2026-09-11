"""Server-rendered pages (Jinja2) and their form actions.

Pages read the same service models as the JSON API; form actions redirect back
to a page so refresh never resubmits.  Ownership is verified by the service
layer on every entity-bearing route.
"""

from __future__ import annotations

import secrets
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
) -> Response:
    project = deps.resolve_project(request, project_id)
    detail = service.paper_detail(deps.get_cfg(request), local_id, project["project_id"])
    return deps.render(request, "paper_detail.html", nav="papers", project=project, detail=detail)


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
) -> Response:
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
    return deps.render(
        request,
        "run_detail.html",
        nav="runs",
        project=project,
        detail=detail,
        claims=claims,
        experiments=experiments_list,
        events=events,
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
