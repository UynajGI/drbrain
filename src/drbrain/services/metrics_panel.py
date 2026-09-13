"""User behavior metrics — search keywords, most-read papers, weekly trends."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from drbrain.storage.connection import connect_wal


@contextmanager
def _connection(db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = connect_wal(db_path)
    try:
        yield conn
    finally:
        conn.close()


def _ensure_metrics_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connection(db_path) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS search_events (id INTEGER PRIMARY KEY AUTOINCREMENT, keyword TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT (datetime('now')));
        CREATE TABLE IF NOT EXISTS read_events (id INTEGER PRIMARY KEY AUTOINCREMENT, local_id TEXT NOT NULL, title TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT (datetime('now')));
        CREATE INDEX IF NOT EXISTS idx_search_keyword ON search_events(keyword);
        CREATE INDEX IF NOT EXISTS idx_read_local_id ON read_events(local_id);
        """)
        conn.commit()


def record_search(db_path: Path, keyword: str) -> None:
    normalized = " ".join(keyword.strip().lower().split())
    if not normalized:
        return
    _ensure_metrics_db(db_path)
    with _connection(db_path) as conn:
        conn.execute("INSERT INTO search_events (keyword) VALUES (?)", (normalized,))
        conn.commit()


def record_read(db_path: Path, local_id: str, title: str) -> None:
    _ensure_metrics_db(db_path)
    with _connection(db_path) as conn:
        conn.execute("INSERT INTO read_events (local_id, title) VALUES (?, ?)", (local_id, title))
        conn.commit()


def get_top_keywords(db_path: Path, limit: int = 10) -> list[dict]:
    _ensure_metrics_db(db_path)
    with _connection(db_path) as conn:
        rows = conn.execute(
            "SELECT keyword, COUNT(*) as cnt FROM search_events GROUP BY keyword ORDER BY cnt DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [{"keyword": r[0], "count": r[1]} for r in rows]


def get_most_read_papers(db_path: Path, limit: int = 10) -> list[dict]:
    _ensure_metrics_db(db_path)
    with _connection(db_path) as conn:
        rows = conn.execute(
            "SELECT local_id, title, COUNT(*) as cnt FROM read_events GROUP BY local_id, title ORDER BY cnt DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [{"local_id": r[0], "title": r[1], "count": r[2]} for r in rows]


def get_weekly_trend(db_path: Path) -> dict:
    _ensure_metrics_db(db_path)
    week_ago = (datetime.now(UTC) - timedelta(days=7)).strftime("%Y-%m-%d")
    with _connection(db_path) as conn:
        total_searches = conn.execute(
            "SELECT COUNT(*) FROM search_events WHERE created_at >= ?", (week_ago,)
        ).fetchone()[0]
        unique_keywords = conn.execute(
            "SELECT COUNT(DISTINCT keyword) FROM search_events WHERE created_at >= ?", (week_ago,)
        ).fetchone()[0]
        total_reads = conn.execute(
            "SELECT COUNT(*) FROM read_events WHERE created_at >= ?", (week_ago,)
        ).fetchone()[0]
        unique_papers = conn.execute(
            "SELECT COUNT(DISTINCT local_id) FROM read_events WHERE created_at >= ?", (week_ago,)
        ).fetchone()[0]
    return {
        "total_searches": total_searches,
        "total_reads": total_reads,
        "unique_keywords": unique_keywords,
        "unique_papers_read": unique_papers,
    }
