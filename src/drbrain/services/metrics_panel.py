"""User behavior metrics — search keywords, most-read papers, weekly trends.

Lightweight SQLite-backed analytics (separate from main DB).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from drbrain.storage.connection import connect_wal


def _ensure_metrics_db(db_path: Path) -> None:
    """Create metrics tables if they don't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect_wal(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS search_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            keyword TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS read_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            local_id TEXT NOT NULL,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_search_keyword ON search_events(keyword)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_read_local_id ON read_events(local_id)
    """)
    conn.commit()
    conn.close()


def record_search(db_path: Path, keyword: str) -> None:
    """Record a search event."""
    normalized = " ".join(keyword.strip().lower().split())
    if not normalized:
        return
    _ensure_metrics_db(db_path)
    conn = connect_wal(db_path)
    conn.execute(
        "INSERT INTO search_events (keyword) VALUES (?)",
        (normalized,),
    )
    conn.commit()
    conn.close()


def record_read(db_path: Path, local_id: str, title: str) -> None:
    """Record a paper read/view event."""
    _ensure_metrics_db(db_path)
    conn = connect_wal(db_path)
    conn.execute(
        "INSERT INTO read_events (local_id, title) VALUES (?, ?)",
        (local_id, title),
    )
    conn.commit()
    conn.close()


def get_top_keywords(db_path: Path, limit: int = 10) -> list[dict]:
    """Return most-used search keywords with counts."""
    _ensure_metrics_db(db_path)
    conn = connect_wal(db_path)
    rows = conn.execute(
        """SELECT keyword, COUNT(*) as cnt
           FROM search_events
           GROUP BY keyword
           ORDER BY cnt DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()
    conn.close()
    return [{"keyword": r[0], "count": r[1]} for r in rows]


def get_most_read_papers(db_path: Path, limit: int = 10) -> list[dict]:
    """Return most-viewed papers with read counts."""
    _ensure_metrics_db(db_path)
    conn = connect_wal(db_path)
    rows = conn.execute(
        """SELECT local_id, title, COUNT(*) as cnt
           FROM read_events
           GROUP BY local_id, title
           ORDER BY cnt DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()
    conn.close()
    return [{"local_id": r[0], "title": r[1], "count": r[2]} for r in rows]


def get_weekly_trend(db_path: Path) -> dict:
    """Return search/read counts for the past 7 days.

    Returns:
        Dict with ``total_searches``, ``total_reads``, ``unique_keywords``,
        ``unique_papers_read``.
    """
    _ensure_metrics_db(db_path)
    conn = connect_wal(db_path)
    week_ago = (datetime.now(UTC) - timedelta(days=7)).strftime("%Y-%m-%d")

    total_searches = conn.execute(
        "SELECT COUNT(*) FROM search_events WHERE created_at >= ?",
        (week_ago,),
    ).fetchone()[0]

    unique_keywords = conn.execute(
        "SELECT COUNT(DISTINCT keyword) FROM search_events WHERE created_at >= ?",
        (week_ago,),
    ).fetchone()[0]

    total_reads = conn.execute(
        "SELECT COUNT(*) FROM read_events WHERE created_at >= ?",
        (week_ago,),
    ).fetchone()[0]

    unique_papers = conn.execute(
        "SELECT COUNT(DISTINCT local_id) FROM read_events WHERE created_at >= ?",
        (week_ago,),
    ).fetchone()[0]

    conn.close()
    return {
        "total_searches": total_searches,
        "total_reads": total_reads,
        "unique_keywords": unique_keywords,
        "unique_papers_read": unique_papers,
    }
