"""Local token authentication for the WebUI (v1 single user).

Model (docs/webui-design.md §4):

* the bootstrap token is generated on first start, printed to the terminal and
  stored at ``<root>/config/webui_token`` with mode 0600;
* verifying it once issues an independent *login session* stored in
  ``webui_sessions`` (hash only); the browser then carries an HttpOnly
  ``SameSite=Strict`` cookie;
* non-browser API clients may send ``Authorization: Bearer <bootstrap token>``
  instead of a cookie (no CSRF token needed — no ambient credentials);
* rotating the token revokes every login session, so old cookies and open SSE
  streams fail closed.

The module is deliberately framework-agnostic: the FastAPI dependencies wrap
these functions, tests call them directly.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drbrain.runtime import RuntimeContext

#: Cookie carrying the opaque login-session secret.
SESSION_COOKIE = "drbrain_webui"
#: Double-submit CSRF cookie (readable by page JS, compared server-side).
CSRF_COOKIE = "drbrain_csrf"
#: Absolute login-session lifetime (seconds).
SESSION_TTL_SECONDS = 12 * 60 * 60
#: Touch the stored last-seen timestamp at most this often (seconds).
TOUCH_INTERVAL_SECONDS = 60.0
#: Bootstrap token file, relative to the runtime root.
TOKEN_RELATIVE_PATH = Path("config") / "webui_token"
#: Principal recorded for single-user local access.
LOCAL_PRINCIPAL = "local"


class AuthError(Exception):
    """Authentication or login-session failure (maps to HTTP 401)."""


@dataclass(frozen=True)
class AuthContext:
    """Verified identity of one request."""

    principal: str
    webui_session_id: str
    csrf_token: str = ""
    via: str = "cookie"  # "cookie" | "bearer"


def webui_root(cfg: Any = None) -> Path:
    """Return the root that owns ``config/webui_token``.

    Mirrors the runtime boundary used by the rest of the service layer: an
    explicitly selected runtime wins; otherwise the process working directory
    (the CLI always starts in the data root).
    """
    if "DRBRAIN_ROOT" in os.environ or "DRBRAIN_RUNTIME_ROOT" in os.environ:
        return RuntimeContext.create().root
    return Path.cwd()


def token_path(cfg: Any = None) -> Path:
    return webui_root(cfg) / TOKEN_RELATIVE_PATH


def ensure_token(cfg: Any = None) -> str:
    """Load the bootstrap token, generating it on first start.

    The file is written atomically with mode 0600; an existing file with wider
    permissions is tightened instead of silently accepted.
    """
    path = token_path(cfg)
    if path.is_file():
        token = path.read_text(encoding="utf-8").strip()
        if not token:
            raise AuthError(f"webui token file is empty: {path}")
        _tighten_permissions(path)
        return token
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.parent.chmod(0o700)
    except OSError:  # pragma: no cover - best effort on exotic filesystems
        pass
    token = secrets.token_urlsafe(32)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(token + "\n")
        tmp.replace(path)
    finally:
        if tmp.exists():  # pragma: no cover - only on failure paths
            tmp.unlink(missing_ok=True)
    return token


def rotate_token(cfg: Any = None) -> str:
    """Replace the bootstrap token; callers must revoke live login sessions."""
    path = token_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(token + "\n")
        tmp.replace(path)
    finally:
        if tmp.exists():  # pragma: no cover
            tmp.unlink(missing_ok=True)
    return token


def _tighten_permissions(path: Path) -> None:
    try:
        mode = path.stat().st_mode & 0o777
        if mode != 0o600:
            path.chmod(0o600)
    except OSError:  # pragma: no cover - permissions are best effort
        pass


def verify_bootstrap_token(cfg: Any, candidate: str | None) -> bool:
    """Constant-time comparison against the bootstrap token."""
    if not candidate:
        return False
    expected = ensure_token(cfg)
    return hmac.compare_digest(candidate, expected)


def hash_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def create_login_session(
    cfg: Any,
    *,
    remote_addr: str = "",
    user_agent: str = "",
    now: float | None = None,
) -> dict[str, Any]:
    """Issue a login session; returns the cookie values exactly once."""
    from drbrain.app import service

    issued = time.time() if now is None else float(now)
    secret = secrets.token_urlsafe(32)
    csrf = secrets.token_urlsafe(24)
    session_id = f"web-{secrets.token_hex(12)}"
    with service.open_db(cfg) as db:
        db.insert_webui_session(
            session_id,
            hash_secret(secret),
            created_at=issued,
            expires_at=issued + SESSION_TTL_SECONDS,
            remote_addr=remote_addr,
            user_agent=user_agent,
        )
        db.record_webui_audit("login", detail="ok", remote_addr=remote_addr)
    return {
        "session_id": session_id,
        "cookie_value": secret,
        "csrf_token": csrf,
        "expires_at": issued + SESSION_TTL_SECONDS,
    }


def resolve_session(cfg: Any, cookie_value: str | None) -> dict[str, Any] | None:
    """Return the stored login session for a cookie value, or ``None``."""
    if not cookie_value:
        return None
    from drbrain.app import service

    with service.open_db(cfg) as db:
        row = db.find_webui_session_by_hash(hash_secret(cookie_value))
        if row is None:
            return None
        now = time.time()
        if row["revoked_at"] is not None or now >= row["expires_at"]:
            return None
        if now - row["last_seen_at"] >= TOUCH_INTERVAL_SECONDS:
            db.touch_webui_session(row["session_id"], last_seen_at=now)
        return row


def revoke_cookie_session(cfg: Any, cookie_value: str | None) -> None:
    if not cookie_value:
        return
    from drbrain.app import service

    with service.open_db(cfg) as db:
        row = db.find_webui_session_by_hash(hash_secret(cookie_value))
        if row is not None:
            db.revoke_webui_session(row["session_id"], revoked_at=time.time())
            db.record_webui_audit("logout", detail=row["session_id"])


def rotate_bootstrap_token(cfg: Any) -> str:
    """Rotate the token and revoke every live login session (fail closed)."""
    from drbrain.app import service

    new_token = rotate_token(cfg)
    with service.open_db(cfg) as db:
        revoked = db.revoke_all_webui_sessions(revoked_at=time.time())
        db.record_webui_audit("token_rotated", detail=f"revoked={revoked}")
    return new_token


__all__ = [
    "AuthContext",
    "AuthError",
    "CSRF_COOKIE",
    "LOCAL_PRINCIPAL",
    "SESSION_COOKIE",
    "SESSION_TTL_SECONDS",
    "create_login_session",
    "ensure_token",
    "hash_secret",
    "resolve_session",
    "revoke_cookie_session",
    "rotate_bootstrap_token",
    "rotate_token",
    "token_path",
    "verify_bootstrap_token",
    "webui_root",
]
