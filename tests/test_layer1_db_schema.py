"""Tests for Layer 1: DB schema + provenance (node_id, tree_vectors, tree_summaries)."""

import tempfile
from pathlib import Path

from drbrain.storage.database import Database


def test_concepts_has_node_id_column():
    """concepts table has node_id TEXT DEFAULT '' column."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        cols = {row[1] for row in db.conn.execute("PRAGMA table_info(concepts)")}
        assert "node_id" in cols


def test_arguments_has_node_id_column():
    """arguments table has node_id TEXT DEFAULT '' column."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        cols = {row[1] for row in db.conn.execute("PRAGMA table_info(arguments)")}
        assert "node_id" in cols


def test_tree_vectors_table_exists():
    """tree_vectors table exists with expected columns."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        cols = {row[1] for row in db.conn.execute("PRAGMA table_info(tree_vectors)")}
        assert "node_id" in cols
        assert "paper_id" in cols
        assert "embedding" in cols
        assert "content_hash" in cols
        assert "tree_layer" in cols


def test_tree_summaries_table_exists():
    """tree_summaries table exists with expected columns."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        cols = {row[1] for row in db.conn.execute("PRAGMA table_info(tree_summaries)")}
        assert "node_id" in cols
        assert "paper_id" in cols
        assert "summary_text" in cols
        assert "source_node_ids" in cols
        assert "tree_layer" in cols


def test_vector_metadata_table_exists():
    """vector_metadata table exists for signature tracking."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        cols = {row[1] for row in db.conn.execute("PRAGMA table_info(vector_metadata)")}
        assert "key" in cols
        assert "value" in cols


def test_node_id_defaults_to_empty():
    """New concepts and arguments have empty node_id by default."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        db = Database(db_path)

        # Insert a paper first (FK constraint)
        db.conn.execute(
            "INSERT INTO papers (local_id, title) VALUES (?, ?)",
            ("test-paper-1", "Test Paper"),
        )

        db.conn.execute(
            "INSERT INTO concepts (local_id, type, label) VALUES (?, ?, ?)",
            ("test-paper-1", "Method", "Test Concept"),
        )
        row = db.conn.execute(
            "SELECT node_id FROM concepts WHERE label = 'Test Concept'"
        ).fetchone()
        assert row is not None
        assert row[0] == ""


def test_node_id_read_write_roundtrip():
    """node_id can be written and read back."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        db = Database(db_path)

        db.conn.execute(
            "INSERT INTO papers (local_id, title) VALUES (?, ?)",
            ("test-paper-2", "Test Paper 2"),
        )

        db.conn.execute(
            "INSERT INTO concepts (local_id, type, label, node_id) VALUES (?, ?, ?, ?)",
            ("test-paper-2", "Method", "Roundtrip Concept", "node-abc-123"),
        )
        row = db.conn.execute(
            "SELECT node_id FROM concepts WHERE label = 'Roundtrip Concept'"
        ).fetchone()
        assert row is not None
        assert row[0] == "node-abc-123"


def test_migration_adds_node_id_to_existing_db():
    """Migration v4 adds node_id to an existing DB created before this change.

    Simulates an older schema without node_id columns, then creates a new
    Database connection which triggers auto-migration.
    """
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"

        # Create a DB with the old schema (no node_id)
        import sqlite3

        old_conn = sqlite3.connect(str(db_path))
        old_conn.executescript("""
            CREATE TABLE IF NOT EXISTS schema_versions (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS papers (
                local_id TEXT PRIMARY KEY,
                title TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS concepts (
                concept_id INTEGER PRIMARY KEY AUTOINCREMENT,
                local_id TEXT NOT NULL REFERENCES papers(local_id),
                type TEXT NOT NULL,
                label TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                section TEXT DEFAULT '',
                first_seen INTEGER,
                last_seen INTEGER
            );
            CREATE TABLE IF NOT EXISTS arguments (
                arg_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_paper TEXT NOT NULL,
                claim TEXT NOT NULL,
                claim_type TEXT NOT NULL,
                target_label TEXT NOT NULL,
                target_type TEXT NOT NULL,
                evidence_type TEXT,
                evidence_detail TEXT,
                mechanism TEXT DEFAULT '',
                section TEXT DEFAULT '',
                confidence REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            INSERT INTO schema_versions (version) VALUES (1);
            INSERT INTO schema_versions (version) VALUES (2);
            INSERT INTO schema_versions (version) VALUES (3);
        """)
        old_conn.commit()
        old_conn.close()

        # Now open with Database class — migration v4 should fire
        db = Database(db_path)

        cols = {row[1] for row in db.conn.execute("PRAGMA table_info(concepts)")}
        assert "node_id" in cols

        args_cols = {row[1] for row in db.conn.execute("PRAGMA table_info(arguments)")}
        assert "node_id" in args_cols

        # Existing data should have empty node_id
        db.conn.execute(
            "INSERT INTO papers (local_id, title) VALUES (?, ?)",
            ("old-paper", "Old Paper"),
        )
        db.conn.execute(
            "INSERT INTO concepts (local_id, type, label) VALUES (?, ?, ?)",
            ("old-paper", "Problem", "Old Concept"),
        )
        row = db.conn.execute("SELECT node_id FROM concepts WHERE label = 'Old Concept'").fetchone()
        assert row is not None
        assert row[0] == ""

        db.conn.close()


def test_performance_indexes_exist():
    """Phase 2: critical performance indexes exist after DB creation."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        indexes = {
            r[0] for r in db.conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
        }

        # These indexes were added in Phase 2.1 to eliminate slow table scans
        assert "idx_concepts_local_id" in indexes
        assert "idx_edges_dst" in indexes
        assert "idx_edges_source_paper" in indexes
        assert "idx_tree_vectors_paper" in indexes
        assert "idx_tree_summaries_paper" in indexes
        assert "idx_citation_cache_target" in indexes


def test_pragma_wal_mode_and_busy_timeout():
    """Phase 2: DB opens with WAL mode and busy_timeout for concurrency."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        journal_mode = db.conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert journal_mode == "wal"

        busy = db.conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert busy >= 5000
