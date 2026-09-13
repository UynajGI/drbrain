"""Metrics tracking via SQLite — LLM calls, generic events, WAL, thread-safe."""

from __future__ import annotations

import functools
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from loguru import logger

from drbrain.runtime import RuntimeContext
from drbrain.security import redact_sensitive_text

DB_PATH = Path("data/metrics.db")


def _active_runtime() -> RuntimeContext | None:
    if "DRBRAIN_ROOT" not in os.environ and "DRBRAIN_RUNTIME_ROOT" not in os.environ:
        return None
    return RuntimeContext.create()


def _default_db_path() -> Path:
    """Return the metrics DB under the active runtime root.

    Metrics are emitted from deep LLM call paths, where passing a config
    object through every caller is impractical.  Resolve the default lazily
    so an embedded CLI invocation can establish ``DRBRAIN_ROOT`` before the
    first event is recorded.  Explicit ``MetricsStore(path)`` values remain
    explicit, but an active runtime still constrains them to its root.
    """
    runtime = _active_runtime()
    if runtime is not None:
        return runtime.assert_within_root(DB_PATH, label="metrics database")
    return DB_PATH


def _safe_text(value: object) -> str:
    return redact_sensitive_text(str(value)) or ""


SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT NOT NULL,
    provider TEXT DEFAULT '',
    tokens_in INTEGER DEFAULT 0,
    tokens_out INTEGER DEFAULT 0,
    duration_ms INTEGER DEFAULT 0,
    source TEXT DEFAULT '',
    session_id TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT DEFAULT '',
    category TEXT NOT NULL,
    name TEXT DEFAULT '',
    duration_ms INTEGER DEFAULT 0,
    tokens_in INTEGER DEFAULT 0,
    tokens_out INTEGER DEFAULT 0,
    model TEXT DEFAULT '',
    status TEXT DEFAULT 'ok',
    detail TEXT DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

MIGRATIONS = [
    "ALTER TABLE llm_calls ADD COLUMN session_id TEXT DEFAULT ''",
]


class MetricsStore:
    """Thread-safe SQLite wrapper for recording LLM and generic usage events."""

    def __init__(self, db_path: str | Path | None = None):
        self._runtime = _active_runtime()
        raw_path = Path(db_path).expanduser() if db_path is not None else _default_db_path()
        if self._runtime is not None:
            self.path = self._runtime.assert_within_root(raw_path, label="metrics database")
        else:
            if raw_path.is_symlink():
                raise ValueError(f"metrics database must not be a symlink: {raw_path}")
            self.path = raw_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._validate_path()
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    def _validate_path(self) -> None:
        if self._runtime is not None:
            self._runtime.assert_within_root(self.path, label="metrics database")
        elif self.path.is_symlink():
            raise ValueError(f"metrics database must not be a symlink: {self.path}")

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._validate_path()
            self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(SCHEMA)
            self._run_migrations()
            self._conn.commit()
        return self._conn

    def _run_migrations(self) -> None:
        if self._conn is None:
            return
        for migration in MIGRATIONS:
            try:
                self._conn.execute(migration)
            except sqlite3.OperationalError:
                pass  # column already exists

    def record_llm(
        self,
        model: str,
        provider: str = "",
        tokens_in: int = 0,
        tokens_out: int = 0,
        duration_ms: int = 0,
        source: str = "",
    ) -> None:
        try:
            with self._lock:
                conn = self._ensure_conn()
                session_id = _get_session_id()
                conn.execute(
                    "INSERT INTO llm_calls "
                    "(model, provider, tokens_in, tokens_out, duration_ms, source, session_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        _safe_text(model),
                        _safe_text(provider),
                        tokens_in,
                        tokens_out,
                        duration_ms,
                        _safe_text(source),
                        session_id,
                    ),
                )
                conn.commit()
        except Exception:
            logger.warning("Failed to record LLM metrics")

    def _record_event(
        self,
        category: str,
        name: str = "",
        duration_ms: float = 0.0,
        status: str = "ok",
        *,
        tokens_in: int = 0,
        tokens_out: int = 0,
        model: str = "",
        detail: str = "{}",
    ) -> None:
        """Write a generic event to the events table."""
        try:
            with self._lock:
                conn = self._ensure_conn()
                session_id = _get_session_id()
                conn.execute(
                    "INSERT INTO events "
                    "(session_id, category, name, duration_ms, tokens_in, tokens_out, model, status, detail) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        session_id,
                        _safe_text(category),
                        _safe_text(name),
                        int(duration_ms),
                        tokens_in,
                        tokens_out,
                        _safe_text(model),
                        _safe_text(status),
                        _safe_text(detail),
                    ),
                )
                conn.commit()
        except Exception:
            logger.warning("Failed to record event metrics")

    @contextmanager
    def timer(self, category: str, name: str = ""):
        """Context manager that records duration on exit.

        Usage:
            with metrics.timer("llm", "gpt-4-call"):
                result = call_llm(...)
        """
        start = time.monotonic()
        status = "ok"
        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            duration_ms = (time.monotonic() - start) * 1000
            self._record_event(category, name, duration_ms, status)

    def timed(self, category: str, name: str = ""):
        """Decorator that records timing for the wrapped function.

        Usage:
            @metrics.timed("llm")
            def call_model(prompt):
                ...

            @metrics.timed("api", "fetch-papers")
            def fetch_from_arxiv(query):
                ...
        """

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.timer(category, name or func.__name__):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def close(self) -> None:
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None


# Module-level singleton
_store: MetricsStore | None = None


def get_metrics() -> MetricsStore:
    global _store
    desired = _default_db_path().expanduser().resolve()
    # A process embedding Typer (for example a WebUI or CliRunner test) can
    # execute commands for different roots sequentially.  Never let the
    # singleton retain the previous root's writable connection.
    if _store is None or _store.path.expanduser().resolve() != desired:
        if _store is not None:
            _store.close()
        _store = MetricsStore(desired)
    return _store


def _get_session_id() -> str:
    """Lazy import get_session_id from drbrain.log to avoid circular deps."""
    try:
        from drbrain.log import get_session_id

        return get_session_id()
    except Exception:
        logger.warning("metrics recording failed: could not get session ID")
        return ""
