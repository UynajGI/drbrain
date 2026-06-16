"""Shared SQLite connection factory with WAL pragmas."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def connect_wal(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection with WAL-mode pragmas applied."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn
