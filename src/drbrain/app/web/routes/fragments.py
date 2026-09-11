"""htmx fragments — partial HTML for in-page updates.

Fragments consume the same service models as full pages; they never call the
JSON API over HTTP, so there is one data path per view.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import Response

from drbrain.app import service
from drbrain.app.web import deps
from drbrain.projects import DEFAULT_PROJECT_ID

router = APIRouter(prefix="/ui/fragments", dependencies=[Depends(deps.authenticate)])


@router.get("/paper-rows")
def paper_rows(
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
        "fragments/paper_rows.html",
        data=data,
        project=project,
        q=q,
        status=status,
    )


@router.get("/run-events")
def run_events(
    request: Request,
    run_id: str = Query(""),
    after: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    project_id: str = Query(""),
) -> Response:
    cfg = deps.get_cfg(request)
    if not run_id:
        return Response(status_code=422)
    try:
        pid = project_id or service.run_project(cfg, run_id)
    except service.RunNotFoundError:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="unknown research run") from None
    project = deps.resolve_project(request, pid)
    events = service.run_events(
        cfg, run_id, after=after, limit=limit, project_id=project["project_id"]
    )
    return deps.render(
        request,
        "fragments/run_events.html",
        events=events,
        run_id=run_id,
        project=project,
        cursor=events[-1]["seq"] if events else after,
    )


@router.get("/session-messages")
def session_messages(
    request: Request,
    session_id: str = Query(""),
    after: int = Query(0, ge=0),
    project_id: str = Query(""),
) -> Response:
    cfg = deps.get_cfg(request)
    if not session_id:
        return Response(status_code=422)
    try:
        pid = project_id or service.session_project(cfg, session_id)
    except service.SessionNotFoundError:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="unknown session") from None
    project = deps.resolve_project(request, pid)
    messages = service.session_messages(cfg, session_id, project["project_id"], after_seq=after)
    return deps.render(
        request,
        "fragments/session_messages.html",
        messages=messages,
        unavailable=False,
        error="",
    )


@router.get("/conformance")
def conformance(
    request: Request,
    plugin_name: str = Query(""),
    check_id: str = Query(""),
) -> Response:
    cfg = deps.get_cfg(request)
    report = None
    if plugin_name and check_id:
        try:
            report = service.conformance_report(cfg, plugin_name, check_id)
        except ValueError:
            report = None
    return deps.render(
        request,
        "fragments/conformance.html",
        report=report,
        plugin_name=plugin_name,
        check_id=check_id,
    )
