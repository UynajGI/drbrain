"""Metrics tracking via SQLite — LLM calls, generic events, WAL, thread-safe."""

from __future__ import annotations

import functools
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from loguru import logger

DB_PATH = Path("data/metrics.db")

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

    def __init__(self, db_path: str | Path = str(DB_PATH)):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(SCHEMA)
            self._run_migrations()
            self._conn.commit()
        return self._conn

    def _run_migrations(self) -> None:
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
                    (model, provider, tokens_in, tokens_out, duration_ms, source, session_id),
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
                        category,
                        name,
                        int(duration_ms),
                        tokens_in,
                        tokens_out,
                        model,
                        status,
                        detail,
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
    if _store is None:
        _store = MetricsStore()
    return _store


def _get_session_id() -> str:
    """Lazy import get_session_id from drbrain.log to avoid circular deps."""
    try:
        from drbrain.log import get_session_id

        return get_session_id()
    except Exception:
        logger.warning("metrics recording failed: could not get session ID")
        return ""
