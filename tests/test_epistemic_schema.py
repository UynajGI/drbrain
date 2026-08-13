"""Tests for the epistemic-layer schema (v11-v13).

Covers the Entity → Claim/Evidence upgrade of `concepts` (provenance /
authority / validity window), plus the new `knowledge_snapshots` and
`answer_records` tables, for both fresh databases and pre-existing v10
databases that migrate in place.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

from drbrain.storage.database import Database

EPISTEMIC_CONCEPT_COLS = ["provenance", "authority", "valid_from", "valid_to"]


def _concept_cols(db: Database) -> list[str]:
    return [r[1] for r in db.conn.execute("PRAGMA table_info(concepts)").fetchall()]


def _table_names(db: Database) -> list[str]:
    return [
        r[0]
        for r in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    ]


def test_fresh_db_has_epistemic_concept_columns(tmp_db):
    """Fresh databases get provenance/authority/valid_from/valid_to on concepts."""
    cols = _concept_cols(tmp_db)
    for c in EPISTEMIC_CONCEPT_COLS:
        assert c in cols, f"missing column {c}"


def test_fresh_db_has_epistemic_tables(tmp_db):
    """Fresh databases get knowledge_snapshots and answer_records tables."""
    tables = _table_names(tmp_db)
    assert "knowledge_snapshots" in tables
    assert "answer_records" in tables


def test_fresh_db_concept_defaults(tmp_db):
    """New epistemic columns have sensible defaults ('' / NULL)."""
    tmp_db.insert_paper("p1", "Test", 2024, "uploaded")
    tmp_db.insert_concept("p1", "Method", "X", 0.9, year=2024)
    tmp_db.commit()

    row = tmp_db.conn.execute(
        "SELECT provenance, authority, valid_from, valid_to FROM concepts"
    ).fetchone()
    assert row[0] == ""  # provenance default
    assert row[1] == ""  # authority default
    assert row[2] is None  # valid_from: no known start
    assert row[3] is None  # valid_to: still valid


def test_answer_records_round_trip(tmp_db):
    """answer_records can store question/answer bound to evidence."""
    tmp_db.conn.execute(
        "INSERT INTO answer_records (question, answer, evidence_ids, snapshot_id) "
        "VALUES (?, ?, ?, ?)",
        (
            "What method solves X?",
            "Method Y (grounded in evidence).",
            '["c1", "e2"]',
            "snap-001",
        ),
    )
    tmp_db.commit()

    row = tmp_db.conn.execute("SELECT * FROM answer_records").fetchone()
    assert row is not None
    assert row[2] == "What method solves X?"  # question
    assert row[3] == "Method Y (grounded in evidence)."  # answer
    assert row[4] == '["c1", "e2"]'  # evidence_ids
    assert row[7] == "snap-001"  # snapshot_id


def test_knowledge_snapshots_round_trip(tmp_db):
    """knowledge_snapshots stores versioned snapshots keyed by snapshot_id."""
    tmp_db.conn.execute(
        "INSERT INTO knowledge_snapshots (snapshot_id, revision_id, description) VALUES (?, ?, ?)",
        ("2026-08-13T10:42:00", "rev-42", "baseline graph"),
    )
    tmp_db.commit()

    row = tmp_db.conn.execute(
        "SELECT snapshot_id, revision_id, description FROM knowledge_snapshots"
    ).fetchone()
    assert row == ("2026-08-13T10:42:00", "rev-42", "baseline graph")


def _make_v10_db(path: Path) -> None:
    """Build a minimal pre-v11 database (old concepts schema, versions 1-10)."""
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE papers (
            local_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            abstract TEXT DEFAULT '',
            year INTEGER,
            paper_type TEXT NOT NULL DEFAULT 'paper',
            status TEXT NOT NULL DEFAULT 'placeholder',
            journal TEXT DEFAULT '',
            publisher TEXT DEFAULT '',
            citation_count INTEGER DEFAULT 0,
            volume TEXT DEFAULT '',
            pages TEXT DEFAULT '',
            authors TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE concepts (
            concept_id INTEGER PRIMARY KEY AUTOINCREMENT,
            local_id TEXT NOT NULL REFERENCES papers(local_id),
            type TEXT NOT NULL,
            label TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,
            section TEXT DEFAULT '',
            node_id TEXT DEFAULT '',
            first_seen INTEGER,
            last_seen INTEGER,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE schema_versions (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.executemany(
        "INSERT INTO schema_versions (version) VALUES (?)",
        [(i,) for i in range(1, 11)],
    )
    conn.commit()
    conn.close()


def test_migrate_v10_adds_epistemic_schema():
    """Opening a v10 database migrates it to v13 in place, preserving data."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "old.db"
        _make_v10_db(db_path)

        # Seed one concept in the old schema.
        conn = sqlite3.connect(str(db_path))
        conn.execute("INSERT INTO papers (local_id, title) VALUES ('p1', 'Old paper')")
        conn.execute(
            "INSERT INTO concepts (local_id, type, label, confidence) "
            "VALUES ('p1', 'Method', 'legacy concept', 0.8)"
        )
        conn.commit()
        conn.close()

        db = Database(db_path)

        versions = [
            r[0]
            for r in db.conn.execute(
                "SELECT version FROM schema_versions ORDER BY version"
            ).fetchall()
        ]
        assert versions == list(range(1, 14))

        cols = _concept_cols(db)
        for c in EPISTEMIC_CONCEPT_COLS:
            assert c in cols, f"migration missing column {c}"

        tables = _table_names(db)
        assert "knowledge_snapshots" in tables
        assert "answer_records" in tables

        # Existing data survived the migration untouched.
        row = db.conn.execute(
            "SELECT label, confidence, provenance FROM concepts WHERE label='legacy concept'"
        ).fetchone()
        assert row == ("legacy concept", 0.8, "")

        db.close()


def test_migration_is_idempotent():
    """Re-opening an already-migrated database applies no new versions."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "mig.db"
        _make_v10_db(db_path)

        db = Database(db_path)
        db.close()

        db2 = Database(db_path)
        versions = [
            r[0]
            for r in db2.conn.execute(
                "SELECT version FROM schema_versions ORDER BY version"
            ).fetchall()
        ]
        assert versions == list(range(1, 14))
        assert "provenance" in _concept_cols(db2)
        db2.close()
