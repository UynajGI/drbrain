"""Shared SQLite connection factory with WAL pragmas."""

from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path

_URI_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def _resolve_db_path(db_path: str | Path) -> str | Path:
    """Resolve a connection target against the active runtime, when present.

    ``connect_wal`` is used by lower-level read and write services that do not
    necessarily receive a :class:`~drbrain.runtime.RuntimeContext` object.  An
    environment-selected root is therefore the last common isolation boundary:
    relative and absolute paths must both stay inside it.  With no selected
    root, retain the historical behavior and let SQLite resolve a relative
    path against the caller's working directory.
    """
    try:
        raw = os.fspath(db_path)
    except (TypeError, ValueError) as exc:
        raise ValueError("database path must be a string or filesystem path") from exc
    if not isinstance(raw, str):
        raise ValueError("database path must be a string or filesystem path")
    if raw == ":memory:":
        return raw
    if not raw:
        raise ValueError("database path must not be empty")
    # ``sqlite3.connect(..., uri=False)`` would normally treat this as a
    # literal filename, but accepting URI-looking input here makes behavior
    # dependent on the caller and invites accidental ``file:...`` targets.
    if _URI_SCHEME_RE.match(raw):
        raise ValueError("database path must be a local filesystem path, not a URI")

    if "DRBRAIN_ROOT" in os.environ or "DRBRAIN_RUNTIME_ROOT" in os.environ:
        from drbrain.runtime import RuntimeContext

        configured_root = str(RuntimeContext.create().root)
    else:
        configured_root = None
    path = Path(raw).expanduser()
    if not configured_root:
        return path

    # Construct a context with an explicit in-root scratch path.  The
    # connection factory must not inherit an unrelated DRBRAIN_TEMP_ROOT just
    # because a caller selected a database root; the context still validates
    # the root itself and rejects lexical symlink aliases.
    from drbrain.runtime import RuntimeContext

    runtime = RuntimeContext.create(configured_root, temp_root="data/.runtime/connect")
    return runtime.assert_within_root(path, label="database path")


def connect_wal(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection with WAL-mode pragmas applied.

    ``:memory:`` remains a valid SQLite sentinel.  When ``DRBRAIN_ROOT`` (or
    its legacy alias) is set, the target is resolved and checked before SQLite
    opens it so a direct service call cannot silently write into another
    worktree.
    """
    resolved = _resolve_db_path(db_path)
    conn = sqlite3.connect(str(resolved))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=60000")
    except Exception:
        # Do not leak a live handle if a pragma fails (for example on a
        # read-only or malformed SQLite target).
        conn.close()
        raise
    return conn
