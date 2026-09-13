"""Login, logout and token rotation routes."""

from __future__ import annotations

import threading
import time
from typing import Any

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response

from drbrain.app import auth, service
from drbrain.app.web import deps

router = APIRouter(dependencies=[Depends(deps.require_csrf)])

#: Small in-process throttle for failed bootstrap-token attempts.
_FAILURES: dict[str, list[float]] = {}
_FAILURE_WINDOW = 300.0
_FAILURE_LIMIT = 8
_FAILURE_LOCK = threading.Lock()


def _throttled(remote: str) -> bool:
    cutoff = time.time() - _FAILURE_WINDOW
    with _FAILURE_LOCK:
        attempts = [t for t in _FAILURES.get(remote, []) if t >= cutoff]
        if attempts:
            _FAILURES[remote] = attempts
        else:
            # Drop empty buckets: the key is attacker-influenced (X-Forwarded-For)
            # and would otherwise grow without bound.
            _FAILURES.pop(remote, None)
        return len(attempts) >= _FAILURE_LIMIT


def _record_failure(remote: str) -> None:
    with _FAILURE_LOCK:
        _FAILURES.setdefault(remote, []).append(time.time())


def _cookie_secure(request: Request) -> bool:
    return (
        request.url.scheme == "https"
        or request.headers.get("x-forwarded-proto", "").lower() == "https"
    )


def _set_login_cookies(response: Response, request: Request, session: dict[str, Any]) -> None:
    secure = _cookie_secure(request)
    response.set_cookie(
        auth.SESSION_COOKIE,
        session["cookie_value"],
        max_age=auth.SESSION_TTL_SECONDS,
        httponly=True,
        samesite="strict",
        secure=secure,
        path="/",
    )
    response.set_cookie(
        auth.CSRF_COOKIE,
        session["csrf_token"],
        max_age=auth.SESSION_TTL_SECONDS,
        httponly=False,
        samesite="strict",
        secure=secure,
        path="/",
    )


def _clear_cookies(response: Response) -> None:
    response.delete_cookie(auth.SESSION_COOKIE, path="/")
    response.delete_cookie(auth.CSRF_COOKIE, path="/")


@router.get("/login")
def login_page(request: Request, next: str = "/", error: str = "") -> Response:
    if request.cookies.get(auth.SESSION_COOKIE) and auth.resolve_session(
        deps.get_cfg(request), request.cookies.get(auth.SESSION_COOKIE)
    ):
        return RedirectResponse(deps.safe_next(next), status_code=303)
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "login.html",
        {
            "request": request,
            "next": deps.safe_next(next),
            "error": error,
            "host": request.headers.get("host", ""),
        },
    )


@router.post("/login")
def login_submit(
    request: Request,
    token: str = Form(""),
    next: str = Form("/"),
) -> Response:
    cfg = deps.get_cfg(request)
    remote = deps.client_ip(request)
    if _throttled(remote):
        return JSONResponse(
            {"error": "too many failed attempts; wait a few minutes", "code": "throttled"},
            status_code=429,
        )
    if not auth.verify_bootstrap_token(cfg, token.strip()):
        _record_failure(remote)
        with service.open_db(cfg) as db:
            db.record_webui_audit("login_failed", detail="bad token", remote_addr=remote)
        templates = request.app.state.templates
        return templates.TemplateResponse(
            request,
            "login.html",
            {
                "request": request,
                "next": deps.safe_next(next),
                "error": "令牌无效，请核对终端打印的 token。",
                "host": request.headers.get("host", ""),
            },
            status_code=401,
        )
    session = auth.create_login_session(
        cfg,
        remote_addr=remote,
        user_agent=request.headers.get("user-agent", "")[:200],
    )
    response = RedirectResponse(deps.safe_next(next), status_code=303)
    _set_login_cookies(response, request, session)
    return response


@router.post("/logout")
def logout_form(request: Request) -> Response:
    cfg = deps.get_cfg(request)
    auth.revoke_cookie_session(cfg, request.cookies.get(auth.SESSION_COOKIE))
    response = RedirectResponse("/login", status_code=303)
    _clear_cookies(response)
    return response


@router.post("/api/auth/verify")
async def api_verify(request: Request) -> Response:
    cfg = deps.get_cfg(request)
    remote = deps.client_ip(request)
    if _throttled(remote):
        return JSONResponse(
            {"error": "too many failed attempts; wait a few minutes", "code": "throttled"},
            status_code=429,
        )
    try:
        body = await request.json()
    except ValueError:
        body = {}
    token = str((body or {}).get("token") or "")
    if not auth.verify_bootstrap_token(cfg, token.strip()):
        _record_failure(remote)
        with service.open_db(cfg) as db:
            db.record_webui_audit("login_failed", detail="bad token", remote_addr=remote)
        return JSONResponse({"error": "invalid token", "code": "unauthorized"}, status_code=401)
    session = auth.create_login_session(
        cfg,
        remote_addr=remote,
        user_agent=request.headers.get("user-agent", "")[:200],
    )
    response = JSONResponse(
        {
            "ok": True,
            "csrf_token": session["csrf_token"],
            "expires_at": session["expires_at"],
        }
    )
    _set_login_cookies(response, request, session)
    return response


@router.post("/api/auth/logout")
def api_logout(request: Request) -> Response:
    cfg = deps.get_cfg(request)
    auth.revoke_cookie_session(cfg, request.cookies.get(auth.SESSION_COOKIE))
    response = JSONResponse({"ok": True})
    _clear_cookies(response)
    return response


@router.post("/api/auth/rotate")
def api_rotate(request: Request) -> Response:
    """Rotate the bootstrap token; every live login session (and stream) dies."""
    deps.authenticate(request)
    cfg = deps.get_cfg(request)
    new_token = auth.rotate_bootstrap_token(cfg)
    response = JSONResponse(
        {
            "ok": True,
            "token": new_token,
            "note": "旧登录态已全部失效；请用新 token 重新登录。",
        }
    )
    _clear_cookies(response)
    return response
