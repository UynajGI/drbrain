"""Shared FastAPI dependencies for the WebUI.

Authentication, request scope resolution and template context live here so
route modules stay thin: they translate HTTP into service calls, never touch
SQL or the loop internals themselves.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote, urlparse

from fastapi import HTTPException, Request
from fastapi.responses import Response

from drbrain.app import auth, service
from drbrain.projects import DEFAULT_PROJECT_ID, UNBOUND_SESSION_LABEL, normalize_project_id


def get_cfg(request: Request) -> Any:
    return request.app.state.cfg


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()[:64]
    return (request.client.host if request.client else "")[:64]


def authenticate(request: Request) -> auth.AuthContext:
    """Resolve the request identity or raise 401.

    Accepts either the browser login cookie or ``Authorization: Bearer`` with
    the bootstrap token.  The returned context never carries credentials.
    """
    cfg = get_cfg(request)
    cookie_value = request.cookies.get(auth.SESSION_COOKIE)
    if cookie_value:
        row = auth.resolve_session(cfg, cookie_value)
        if row is not None:
            return auth.AuthContext(
                principal=auth.LOCAL_PRINCIPAL,
                webui_session_id=row["session_id"],
                csrf_token=request.cookies.get(auth.CSRF_COOKIE, ""),
                via="cookie",
            )
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        token = header.split(" ", 1)[1].strip()
        if auth.verify_bootstrap_token(cfg, token):
            return auth.AuthContext(
                principal=auth.LOCAL_PRINCIPAL,
                webui_session_id="bearer",
                csrf_token="",
                via="bearer",
            )
    raise HTTPException(status_code=401, detail="authentication required")


def resolve_project(request: Request, project_id: str | None = None) -> dict[str, Any]:
    """Resolve and validate the request's project scope."""
    try:
        pid = normalize_project_id(project_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        return service.resolve_project(get_cfg(request), pid)
    except service.ProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def resolve_session(
    request: Request, session_id: str, project_id: str | None = None
) -> dict[str, Any]:
    """Load a conversation session and verify it belongs to the scope."""
    try:
        return service.get_session(get_cfg(request), session_id, project_id)
    except service.SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def safe_next(target: str | None) -> str:
    """Return a safe same-site redirect target (open-redirect guard)."""
    if not target:
        return "/"
    parsed = urlparse(target)
    if parsed.scheme or parsed.netloc or not target.startswith("/") or target.startswith("//"):
        return "/"
    return target


def projects_for_switcher(request: Request) -> list[dict[str, Any]]:
    cfg = get_cfg(request)
    try:
        return service.projects(cfg)
    except Exception:  # noqa: BLE001 - the switcher must never break a page
        return []


def render(request: Request, template: str, *, status_code: int = 200, **context: Any) -> Response:
    """Render a Jinja2 template with the shared workbench context."""
    templates = request.app.state.templates
    ctx: dict[str, Any] = {
        "request": request,
        "projects": projects_for_switcher(request),
        "default_project_id": DEFAULT_PROJECT_ID,
        "csrf_token": request.cookies.get(auth.CSRF_COOKIE, ""),
        "unbound_session_label": UNBOUND_SESSION_LABEL,
    }
    ctx.update(context)
    return templates.TemplateResponse(request, template, ctx, status_code=status_code)


def is_htmx(request: Request) -> bool:
    return request.headers.get("hx-request", "").lower() == "true"


def login_redirect_target(request: Request) -> str:
    return "/login?next=" + quote(
        request.url.path + ("?" + request.url.query if request.url.query else "")
    )


__all__ = [
    "authenticate",
    "client_ip",
    "get_cfg",
    "is_htmx",
    "login_redirect_target",
    "projects_for_switcher",
    "render",
    "resolve_project",
    "resolve_session",
    "safe_next",
]
