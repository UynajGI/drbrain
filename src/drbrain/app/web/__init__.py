"""FastAPI application factory for the DrBrain WebUI.

``drbrain webui`` starts this app with uvicorn.  The layering rule from the
design contract applies here: ``app/web/`` only translates HTTP to service
calls — no SQL, no loop internals.  ``app/service.py`` remains the single
business facade, and ``auth.py`` the single authentication boundary.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from drbrain.app import auth
from drbrain.app.web import labels
from drbrain.app.web.deps import login_redirect_target

WEB_DIR = Path(__file__).parent
STATIC_DIR = WEB_DIR / "static"
TEMPLATE_DIR = WEB_DIR / "templates"


def _fmt_ts(value: Any) -> str:
    if value in (None, ""):
        return "—"
    try:
        if isinstance(value, (int, float)) or (
            isinstance(value, str) and value.replace(".", "", 1).isdigit()
        ):
            return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(value)))
    except (OverflowError, OSError, ValueError):
        return str(value)
    return str(value)[:19]


def _short(value: Any, limit: int = 12) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def build_templates() -> Jinja2Templates:
    templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
    env = templates.env
    env.filters["ts"] = _fmt_ts
    env.filters["status"] = labels.status_of
    env.filters["short"] = _short
    env.globals["status_label"] = labels.status_of
    env.globals["role_labels"] = labels.ROLE_LABELS
    env.globals["layer_labels"] = labels.LAYER_LABELS
    env.globals["error_text"] = labels.error_text
    return templates


class _SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach conservative headers; HTML/API responses are never cached."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        path = request.url.path
        if path.startswith("/static/"):
            response.headers.setdefault("Cache-Control", "public, max-age=86400")
        else:
            response.headers.setdefault("Cache-Control", "no-store")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; "
            "form-action 'self'",
        )
        return response


def _error_payload(detail: Any, code: str = "error") -> dict[str, Any]:
    if isinstance(detail, dict):
        payload = dict(detail)
        payload.setdefault("error", str(payload.get("detail") or code))
    else:
        payload = {"error": str(detail)}
    payload.setdefault("code", code)
    return payload


def _render_error(request: Request, message: str, code: str, status_code: int) -> Response:
    """JSON for API/htmx callers, the styled error page for browser navigation."""
    if (
        request.url.path.startswith("/api/")
        or request.headers.get("hx-request", "").lower() == "true"
    ):
        return JSONResponse({"error": message, "code": code}, status_code=status_code)
    try:
        from drbrain.app.web import deps

        return deps.render(
            request,
            "error.html",
            status_code=status_code,
            message=message,
            code=code,
            title=labels.error_title(code, status_code),
        )
    except Exception:  # noqa: BLE001 - never let the error page itself crash the response
        return JSONResponse({"error": message, "code": code}, status_code=status_code)


def create_app(cfg: Any, *, app_title: str = "DrBrain WebUI") -> FastAPI:
    """Build the WebUI application around one configuration snapshot."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            auth.ensure_token(cfg)
        except Exception:  # noqa: BLE001 - a broken token file must not kill startup
            from loguru import logger

            logger.warning("[webui] could not prepare the bootstrap token")
        yield

    app = FastAPI(
        title=app_title, docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan
    )
    app.state.cfg = cfg
    app.state.templates = build_templates()
    app.add_middleware(_SecurityHeadersMiddleware)

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> Response:
        detail = exc.detail
        if exc.status_code == 401:
            if request.headers.get("hx-request", "").lower() == "true":
                return Response(
                    status_code=401, headers={"HX-Redirect": login_redirect_target(request)}
                )
            if not request.url.path.startswith("/api/"):
                return RedirectResponse(login_redirect_target(request), status_code=303)
            return JSONResponse(_error_payload(detail, "unauthorized"), status_code=401)
        if isinstance(detail, dict):
            # Dependencies raise structured errors (csrf/origin); keep the code.
            return _render_error(
                request,
                str(detail.get("error") or ""),
                str(detail.get("code") or "http_error"),
                exc.status_code,
            )
        if exc.status_code == 404:
            code = "not_found"
        elif exc.status_code == 422:
            code = "validation_error"
        else:
            code = "http_error"
        return _render_error(request, str(detail), code, exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> Response:
        return JSONResponse(
            {
                "error": "request validation failed",
                "code": "validation_error",
                "detail": exc.errors(),
            },
            status_code=422,
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> Response:
        # The server's error boundary: log a redacted summary, return an
        # opaque message so internals/credentials never reach the client.
        from loguru import logger

        from drbrain.security import configured_secret_values, safe_error

        message = safe_error(exc, secrets=configured_secret_values(cfg))
        logger.error("[webui] unhandled error on {}: {}", request.url.path, message)
        return JSONResponse({"error": "internal server error", "code": "internal"}, status_code=500)

    def _register_not_found(exc_type: type[Exception], code: str) -> None:
        @app.exception_handler(exc_type)
        async def _handler(request: Request, exc: Exception) -> Response:
            return _render_error(request, str(exc), code, 404)

    from drbrain.app import service

    _register_not_found(service.ProjectNotFoundError, "project_not_found")
    _register_not_found(service.SessionNotFoundError, "session_not_found")
    _register_not_found(service.RunNotFoundError, "run_not_found")
    _register_not_found(service.PaperNotInProjectError, "paper_not_found")
    _register_not_found(service.EvidenceNotFoundError, "evidence_not_found")
    _register_not_found(service.ArtifactNotFoundError, "artifact_not_found")

    @app.exception_handler(service.CursorError)
    async def _cursor_error(request: Request, exc: service.CursorError) -> Response:
        return _render_error(request, str(exc), "invalid_cursor", 422)

    from drbrain.app.web.routes import api, auth_routes, fragments, pages, stream

    app.include_router(auth_routes.router)
    app.include_router(pages.router)
    app.include_router(fragments.router)
    app.include_router(api.router)
    app.include_router(stream.router)
    return app


__all__ = ["create_app"]
