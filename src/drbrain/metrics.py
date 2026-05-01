"""LLM token usage tracking via SQLite."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from loguru import logger

DB_DIR = Path("data/metrics")
DB_PATH = DB_DIR / "metrics.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT NOT NULL,
    provider TEXT DEFAULT '',
    tokens_in INTEGER DEFAULT 0,
    tokens_out INTEGER DEFAULT 0,
    duration_ms INTEGER DEFAULT 0,
    source TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS api_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    api TEXT NOT NULL,
    endpoint TEXT DEFAULT '',
    duration_ms INTEGER DEFAULT 0,
    status TEXT DEFAULT 'ok',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


class MetricsStore:
    """Thin SQLite wrapper for recording LLM and API usage."""

    def __init__(self, db_path: str | Path = str(DB_PATH)):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.path))
            self._conn.executescript(SCHEMA)
            self._conn.commit()
        return self._conn

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
            conn = self._ensure_conn()
            conn.execute(
                "INSERT INTO llm_calls (model, provider, tokens_in, tokens_out, duration_ms, source) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (model, provider, tokens_in, tokens_out, duration_ms, source),
            )
            conn.commit()
        except Exception:
            logger.warning("Failed to record LLM metrics")

    def record_api(
        self, api: str, endpoint: str = "", duration_ms: int = 0, status: str = "ok"
    ) -> None:
        try:
            conn = self._ensure_conn()
            conn.execute(
                "INSERT INTO api_calls (api, endpoint, duration_ms, status) VALUES (?, ?, ?, ?)",
                (api, endpoint, duration_ms, status),
            )
            conn.commit()
        except Exception:
            logger.warning("Failed to record API metrics")

    def get_llm_stats(self) -> dict:
        try:
            conn = self._ensure_conn()
            row = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(tokens_in), 0), COALESCE(SUM(tokens_out), 0), "
                "COALESCE(SUM(duration_ms), 0) FROM llm_calls"
            ).fetchone()
            return {
                "total_calls": row[0],
                "total_tokens_in": row[1],
                "total_tokens_out": row[2],
                "total_duration_ms": row[3],
            }
        except Exception:
            return {
                "total_calls": 0,
                "total_tokens_in": 0,
                "total_tokens_out": 0,
                "total_duration_ms": 0,
            }

    def close(self) -> None:
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


def timed_llm(model: str, provider: str = "", source: str = ""):
    """Decorator/context-manager style helper for timing and recording LLM calls.

    Usage as wrapper:
        metrics = get_metrics()
        start = time.monotonic()
        result = call_llm(...)
        metrics.record_llm(model, provider, tokens_in=..., duration_ms=...)
    """
    # We provide a simple context manager pattern below
    pass


class LLMTimer:
    """Context manager for timing an LLM call and recording metrics."""

    def __init__(self, model: str, provider: str = "", source: str = ""):
        self.model = model
        self.provider = provider
        self.source = source
        self._start = 0.0

    def __enter__(self):
        self._start = time.monotonic()
        return self

    def __exit__(self, *args):
        pass

    def record(self, tokens_in: int = 0, tokens_out: int = 0) -> None:
        duration_ms = int((time.monotonic() - self._start) * 1000)
        get_metrics().record_llm(
            model=self.model,
            provider=self.provider,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            duration_ms=duration_ms,
            source=self.source,
        )
