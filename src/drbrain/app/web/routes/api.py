"""JSON API routes.

Contract notes (docs/webui-design.md §6):

* legacy routes keep their response shapes; scope parameters are additive and
  default to the implicit project when an old client does not send one;
* new list responses use ``items``/paging fields and live under
  ``/api/projects/{pid}/...``;
* errors are ``{"error": ..., "code": ...}`` with honest status codes
  (401/403/404/409/422/503) — no empty lists pretending to be "not found".
  Domain failures (``service.*Error``) map to 404 through the app-level
  handlers, so the code stays specific (``run_not_found``, …).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from drbrain.app import service
from drbrain.app.web import deps
from drbrain.loop.store import AmbiguousRunError, RunLedger
from drbrain.projects import DEFAULT_PROJECT_ID

router = APIRouter(prefix="/api", dependencies=[Depends(deps.authenticate)])


class RunStartRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=500)
    max_cycles: int | None = Field(default=None, ge=1, le=100)
    project_id: str | None = None
    session_id: str | None = None
    client_request_id: str | None = Field(default=None, max_length=120)


class SessionCreateRequest(BaseModel):
    title: str = Field(default="", max_length=200)
    system_prompt: str = Field(default="", max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    project_id: str | None = None


class PromoteRequest(BaseModel):
    memory_id: str = Field(min_length=1, max_length=120)
    project_id: str | None = None


# ── projects ─────────────────────────────────────────────────────────────────


@router.get("/projects")
def list_projects(request: Request, counts: bool = Query(False)) -> dict[str, Any]:
    return {"items": service.projects(deps.get_cfg(request), with_counts=counts)}


# ── dashboard / search / ask (legacy parity) ─────────────────────────────────


@router.get("/dashboard")
def dashboard(request: Request, project_id: str = Query(DEFAULT_PROJECT_ID)) -> dict[str, Any]:
    project = deps.resolve_project(request, project_id)
    return service.dashboard(deps.get_cfg(request), project["project_id"])


@router.get("/search")
def search(
    request: Request,
    q: str = Query(""),
    limit: int = Query(10, ge=1, le=100),
    type: str = Query(""),
    project_id: str = Query(DEFAULT_PROJECT_ID),
) -> dict[str, Any]:
    project = deps.resolve_project(request, project_id)
    results = service.search(
        deps.get_cfg(request),
        q,
        limit=limit,
        type_filter=type or None,
        project_id=project["project_id"],
    )
    return {"query": q, "results": results}


@router.post("/ask")
def ask(request: Request, body: dict[str, Any] | None = None) -> Response:
    payload = body or {}
    result = service.ask(
        deps.get_cfg(request),
        str(payload.get("question", "")),
        top_k=int(payload.get("top_k", 5) or 5),
    )
    status = 503 if result.get("unavailable") else 200
    return JSONResponse(result, status_code=status)


# ── literature ───────────────────────────────────────────────────────────────


@router.get("/papers")
def papers(
    request: Request,
    project_id: str = Query(DEFAULT_PROJECT_ID),
    q: str = Query(""),
    status: str = Query(""),
    cursor: str = Query(""),
    limit: int = Query(service.DEFAULT_PAGE_SIZE, ge=1, le=service.MAX_PAGE_SIZE),
) -> dict[str, Any]:
    project = deps.resolve_project(request, project_id)
    return service.papers(
        deps.get_cfg(request),
        project["project_id"],
        query=q,
        status=status or None,
        limit=limit,
        cursor=cursor or None,
    )


@router.get("/papers/{local_id:path}")
def paper_detail(
    request: Request,
    local_id: str,
    project_id: str = Query(DEFAULT_PROJECT_ID),
) -> dict[str, Any]:
    project = deps.resolve_project(request, project_id)
    return service.paper_detail(deps.get_cfg(request), local_id, project["project_id"])


# ── sessions ─────────────────────────────────────────────────────────────────


def _session_scope(request: Request, session_id: str, project_id: str | None) -> dict[str, Any]:
    cfg = deps.get_cfg(request)
    pid = project_id or service.session_project(cfg, session_id)
    return deps.resolve_project(request, pid)


@router.get("/projects/{project_id}/sessions")
def list_sessions(
    request: Request,
    project_id: str,
    cursor: str = Query(""),
    limit: int = Query(service.DEFAULT_PAGE_SIZE, ge=1, le=service.MAX_PAGE_SIZE),
) -> dict[str, Any]:
    project = deps.resolve_project(request, project_id)
    return service.sessions(
        deps.get_cfg(request),
        project["project_id"],
        limit=limit,
        cursor=cursor or None,
    )


@router.post("/projects/{project_id}/sessions", status_code=201)
def create_session(request: Request, project_id: str, body: SessionCreateRequest) -> dict[str, Any]:
    project = deps.resolve_project(request, project_id)
    return service.create_session(
        deps.get_cfg(request),
        project["project_id"],
        title=body.title,
        system_prompt=body.system_prompt,
    )


@router.get("/sessions/{session_id}")
def session_detail(
    request: Request, session_id: str, project_id: str = Query("")
) -> dict[str, Any]:
    project = _session_scope(request, session_id, project_id or None)
    return service.get_session(deps.get_cfg(request), session_id, project["project_id"])


@router.get("/sessions/{session_id}/memory")
def session_memory(
    request: Request, session_id: str, project_id: str = Query("")
) -> dict[str, Any]:
    project = _session_scope(request, session_id, project_id or None)
    return service.session_memory(
        deps.get_cfg(request), session_id, project_id=project["project_id"]
    )


@router.post("/sessions/{session_id}/chat")
def session_chat(request: Request, session_id: str, body: ChatRequest) -> Response:
    project = _session_scope(request, session_id, body.project_id)
    try:
        result = service.chat_in_session(
            deps.get_cfg(request), session_id, body.question, project["project_id"]
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if result.get("unavailable"):
        return JSONResponse(result, status_code=503)
    return JSONResponse(result)


@router.post("/sessions/{session_id}/memory/promote")
def promote_memory(request: Request, session_id: str, body: PromoteRequest) -> dict[str, Any]:
    project = _session_scope(request, session_id, body.project_id)
    try:
        return service.promote_memory(
            deps.get_cfg(request), session_id, body.memory_id, project["project_id"]
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ── runs ─────────────────────────────────────────────────────────────────────


@router.get("/runs")
def list_runs(
    request: Request,
    project_id: str = Query(DEFAULT_PROJECT_ID),
    session_id: str = Query(""),
    limit: int = Query(200, ge=1, le=1000),
) -> list[dict[str, Any]]:
    project = deps.resolve_project(request, project_id)
    return service.runs(
        deps.get_cfg(request),
        project["project_id"],
        session_id=session_id or None,
        limit=limit,
    )


@router.post("/runs", status_code=202)
def start_run(request: Request, body: RunStartRequest) -> dict[str, Any]:
    cfg = deps.get_cfg(request)
    project = deps.resolve_project(request, body.project_id or DEFAULT_PROJECT_ID)
    try:
        return service.start_run(
            cfg,
            body.topic,
            max_cycles=body.max_cycles,
            project_id=project["project_id"],
            session_id=body.session_id or "",
            client_request_id=body.client_request_id,
        )
    except service.SessionNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        from drbrain.security import safe_error

        raise HTTPException(status_code=400, detail=safe_error(exc, limit=200)) from exc


@router.get("/runs/{run_id}")
def run_detail(request: Request, run_id: str, project_id: str = Query("")) -> dict[str, Any]:
    cfg = deps.get_cfg(request)
    pid = project_id or service.run_project(cfg, run_id)
    project = deps.resolve_project(request, pid)
    return service.run_detail(cfg, run_id, project["project_id"])


@router.get("/run-status")
def run_status(
    request: Request,
    topic: str = Query(""),
    project_id: str = Query(DEFAULT_PROJECT_ID),
) -> dict[str, Any]:
    """Legacy topic-based status; ambiguous topics get an explicit 409."""
    cfg = deps.get_cfg(request)
    project = deps.resolve_project(request, project_id)
    topic = topic.strip()
    if not topic:
        return {"topic": "", "alive": False, "error": "missing topic"}
    ledger = RunLedger(service.ledger_path(cfg))
    try:
        run = ledger.get_run(topic, project_id=project["project_id"])
    except AmbiguousRunError:
        raise HTTPException(
            status_code=409,
            detail="topic matches several runs in this project; query by run_id",
        ) from None
    if run is None:
        status = service.run_manager().status(topic)
        return {"topic": topic, "alive": status["alive"], "error": status["error"]}
    status = service.run_manager().status_by_run(run.run_id)
    return {
        "topic": topic,
        "run_id": run.run_id,
        "alive": status["alive"],
        "error": status["error"],
    }


@router.get("/runs/{run_id}/events")
def run_events(
    request: Request,
    run_id: str,
    after: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=1000),
    project_id: str = Query(""),
) -> list[dict[str, Any]]:
    cfg = deps.get_cfg(request)
    pid = project_id or service.run_project(cfg, run_id)
    project = deps.resolve_project(request, pid)
    return service.run_events(
        cfg, run_id, after=after, limit=limit, project_id=project["project_id"]
    )


@router.get("/runs/{run_id}/claims")
def run_claims(request: Request, run_id: str, project_id: str = Query("")) -> list[dict[str, Any]]:
    cfg = deps.get_cfg(request)
    pid = project_id or service.run_project(cfg, run_id)
    project = deps.resolve_project(request, pid)
    return service.run_claims(cfg, run_id, project_id=project["project_id"])


@router.get("/experiments")
def experiments(
    request: Request,
    run_id: str = Query(""),
    project_id: str = Query(DEFAULT_PROJECT_ID),
) -> list[dict[str, Any]]:
    project = deps.resolve_project(request, project_id)
    return service.experiments(
        deps.get_cfg(request), run_id=run_id or None, project_id=project["project_id"]
    )


@router.get("/runs/{run_id}/report")
def run_report(
    request: Request,
    run_id: str,
    format: str = Query("markdown", pattern="^(markdown|json)$"),
    project_id: str = Query(""),
) -> Response:
    cfg = deps.get_cfg(request)
    pid = project_id or service.run_project(cfg, run_id)
    project = deps.resolve_project(request, pid)
    body, media_type, filename = service.run_report(cfg, run_id, project["project_id"], fmt=format)
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=body, media_type=media_type, headers=headers)


@router.get("/runs/{run_id}/evidence/{evidence_id:path}")
def evidence_detail(
    request: Request,
    run_id: str,
    evidence_id: str,
    project_id: str = Query(""),
) -> dict[str, Any]:
    cfg = deps.get_cfg(request)
    pid = project_id or service.run_project(cfg, run_id)
    project = deps.resolve_project(request, pid)
    return service.evidence_detail(cfg, run_id, evidence_id, project_id=project["project_id"])


@router.get("/runs/{run_id}/artifacts/{artifact_id}")
def artifact_detail(
    request: Request,
    run_id: str,
    artifact_id: str,
    project_id: str = Query(""),
) -> dict[str, Any]:
    cfg = deps.get_cfg(request)
    pid = project_id or service.run_project(cfg, run_id)
    project = deps.resolve_project(request, pid)
    return service.artifact_detail(cfg, run_id, artifact_id, project["project_id"])


# ── plugins / settings ───────────────────────────────────────────────────────


@router.get("/plugins")
def list_plugins(request: Request) -> list[dict[str, Any]]:
    return service.plugin_catalog(deps.get_cfg(request))


@router.post("/plugins/{plugin_name}/conformance", status_code=202)
def start_conformance(request: Request, plugin_name: str) -> dict[str, Any]:
    try:
        return service.start_conformance(deps.get_cfg(request), plugin_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/plugins/{plugin_name}/conformance/{check_id}")
def conformance_report(request: Request, plugin_name: str, check_id: str) -> dict[str, Any]:
    try:
        return service.conformance_report(deps.get_cfg(request), plugin_name, check_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/settings")
def settings(request: Request) -> dict[str, Any]:
    return service.settings_view(deps.get_cfg(request))


@router.get("/assets")
def assets(request: Request) -> dict[str, Any]:
    return service.assets(deps.get_cfg(request))
