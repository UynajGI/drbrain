"""SQLite backend with schema management."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS papers (
    local_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    year INTEGER,
    status TEXT NOT NULL DEFAULT 'placeholder' CHECK(status IN ('uploaded', 'placeholder', 'merged')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS paper_ids (
    local_id TEXT NOT NULL REFERENCES papers(local_id) ON DELETE CASCADE,
    doi TEXT UNIQUE,
    arxiv TEXT UNIQUE,
    s2_id TEXT UNIQUE,
    openalex_id TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS concepts (
    concept_id INTEGER PRIMARY KEY AUTOINCREMENT,
    local_id TEXT NOT NULL REFERENCES papers(local_id),
    type TEXT NOT NULL CHECK(type IN ('Problem', 'Method', 'Conclusion', 'Debate', 'Gap', 'Actor')),
    label TEXT NOT NULL,
    confidence REAL DEFAULT 1.0
);

CREATE TABLE IF NOT EXISTS edges (
    src_id TEXT NOT NULL,
    dst_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    source_paper TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    PRIMARY KEY (src_id, dst_id, relation, source_paper)
);

CREATE TABLE IF NOT EXISTS aliases (
    variant TEXT PRIMARY KEY,
    canonical_id TEXT NOT NULL REFERENCES concepts(concept_id)
);

CREATE INDEX IF NOT EXISTS idx_concepts_type ON concepts(type);
CREATE INDEX IF NOT EXISTS idx_concepts_label ON concepts(label);
CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges(relation);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_id);
"""


class Database:
    """Thin SQLite wrapper with schema auto-init."""

    def __init__(self, db_path: str | Path = "data/drbrain.db"):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def executemany(self, sql: str, seq: list[tuple]) -> sqlite3.Cursor:
        return self.conn.executemany(sql, seq)

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # -- Paper queries --

    def get_paper_by_external_id(self, id_type: str, value: str) -> str | None:
        """Look up local_id by external identifier."""
        col = {"doi": "doi", "arxiv": "arxiv", "s2_id": "s2_id", "openalex_id": "openalex_id"}[id_type]
        row = self.conn.execute(
            f"SELECT local_id FROM paper_ids WHERE {col} = ?", (value,)
        ).fetchone()
        return row[0] if row else None

    def fuzzy_match_title_year(self, title: str, year: int) -> str | None:
        """Simple exact title+year match. Upgrade to SimHash later."""
        row = self.conn.execute(
            "SELECT local_id FROM papers WHERE title = ? AND year = ?",
            (title, year),
        ).fetchone()
        return row[0] if row else None

    def insert_paper(self, local_id: str, title: str, year: int | None, status: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO papers (local_id, title, year, status) VALUES (?, ?, ?, ?)",
            (local_id, title, year, status),
        )

    def insert_paper_ids(self, local_id: str, doi=None, arxiv=None, s2_id=None, openalex_id=None) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO paper_ids (local_id, doi, arxiv, s2_id, openalex_id) VALUES (?, ?, ?, ?, ?)",
            (local_id, doi, arxiv, s2_id, openalex_id),
        )

    def upgrade_placeholder(self, local_id: str) -> None:
        self.conn.execute(
            "UPDATE papers SET status = 'uploaded' WHERE local_id = ? AND status = 'placeholder'",
            (local_id,),
        )
