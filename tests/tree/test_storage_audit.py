"""T49: read-only storage audit (no migration, no writes, no network)."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from drbrain.services.storage_audit import audit_storage, open_readonly, summarize
from drbrain.storage.database import Database
from drbrain.tree.blocks import build_content_blocks
from drbrain.tree.contracts import LeafRef, NodeRecord, leaf_node_id


def _legacy_db(path: Path) -> None:
    """A pre-unified database (schema v22 without content tables)."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE schema_versions (version INTEGER PRIMARY KEY, applied_at TIMESTAMP);
        INSERT INTO schema_versions (version) VALUES (22);
        CREATE TABLE papers (
            local_id TEXT PRIMARY KEY, title TEXT NOT NULL, year INTEGER,
            status TEXT NOT NULL DEFAULT 'placeholder', journal TEXT DEFAULT '',
            publisher TEXT DEFAULT '', citation_count INTEGER DEFAULT 0
        );
        INSERT INTO papers (local_id, title, year) VALUES
            ('p0000abcdef01', 'Legacy A', 2020),
            ('p0000abcdef02', 'Legacy B', 2021);
        CREATE TABLE evidence (evidence_id INTEGER PRIMARY KEY, local_id TEXT);
        """
    )
    conn.commit()
    conn.close()


#: A realistic current-generation id, so the legacy short-id audit stays quiet.
LONG_ID = "p0123456789abcdef01234567"


def _seeded_db(path: Path) -> Database:
    db = Database(path)
    text = "# Sec\n\nbody text for the audit\n"
    db.insert_paper(LONG_ID, "T", 2024, "uploaded")
    blocks = build_content_blocks(text, local_id=LONG_ID, revision=1, media_type="md")
    db.upsert_document_revision(
        LONG_ID,
        1,
        source_hash="s",
        canonical_hash=hashlib.sha256(text.encode()).hexdigest(),
        media_type="md",
    )
    db.insert_content_blocks(blocks)
    ref = LeafRef(
        local_id=LONG_ID,
        revision=1,
        block_id=blocks[1].block_id,
        char_start=0,
        char_end=len(blocks[1].text),
    )
    leaf = NodeRecord(
        node_id=leaf_node_id(ref),
        revision=1,
        kind="leaf",
        state="ready",
        layer=0,
        content_hash=blocks[1].text_hash,
        leaf=ref,
    )
    db.insert_tree_node(leaf, publish=True)
    # ``insert_paper`` deliberately leaves its row uncommitted until artifacts
    # exist (test_new_paper_is_not_committed_before_artifacts), and the ingest
    # flow commits explicitly; a helper that closes without committing would
    # silently discard every write.
    db.commit()
    return db


class TestReadOnlyGuarantees:
    def test_open_readonly_refuses_writes(self, tmp_path):
        path = tmp_path / "db.sqlite"
        _seeded_db(path).close()
        conn = open_readonly(path)
        try:
            with pytest.raises(sqlite3.OperationalError):
                conn.execute("INSERT INTO schema_versions (version) VALUES (999)")
        finally:
            conn.close()

    def test_audit_does_not_upgrade_a_legacy_database(self, tmp_path):
        path = tmp_path / "legacy.sqlite"
        _legacy_db(path)
        before = path.read_bytes()
        report = audit_storage(db_path=path)
        assert path.read_bytes() == before
        assert not report.unified_store
        assert report.schema_version == 22
        assert any(f.category == "legacy_store" for f in report.findings)

    def test_audit_is_deterministic(self, tmp_path):
        path = tmp_path / "db.sqlite"
        _seeded_db(path).close()
        first = audit_storage(db_path=path).to_json()
        second = audit_storage(db_path=path).to_json()
        assert first["categories"] == second["categories"]
        assert first["counts"] == second["counts"]

    def test_missing_database_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            audit_storage(db_path=tmp_path / "nope.sqlite")


class TestUnifiedFindings:
    def test_healthy_store_has_no_errors(self, tmp_path):
        path = tmp_path / "db.sqlite"
        _seeded_db(path).close()
        report = audit_storage(db_path=path)
        assert report.unified_store and report.ok(), [f.to_json() for f in report.findings]
        assert report.counts["blocks"] > 0
        assert report.counts["fts_verified"] == report.counts["fts_sampled"] > 0

    def test_content_hash_conflict_is_detected(self, tmp_path):
        path = tmp_path / "db.sqlite"
        db = _seeded_db(path)
        db.conn.execute("UPDATE content_blocks SET text = 'tampered' WHERE ordinal = 0")
        db.conn.commit()
        db.close()
        report = audit_storage(db_path=path)
        assert any(f.category == "content_hash_conflict" for f in report.findings)
        assert not report.ok()

    def test_content_gap_is_detected(self, tmp_path):
        path = tmp_path / "db.sqlite"
        db = _seeded_db(path)
        # Move a block so it no longer starts where the previous one ended.
        db.conn.execute("UPDATE content_blocks SET char_start = char_start + 5 WHERE ordinal = 1")
        db.conn.commit()
        db.close()
        report = audit_storage(db_path=path)
        assert any(f.category == "content_gap" for f in report.findings)

    def test_span_mismatch_is_detected(self, tmp_path):
        path = tmp_path / "db.sqlite"
        db = _seeded_db(path)
        db.conn.execute("UPDATE content_blocks SET char_end = char_end + 5 WHERE ordinal = 1")
        db.conn.commit()
        db.close()
        report = audit_storage(db_path=path)
        assert any(f.category == "content_span_mismatch" for f in report.findings)

    def test_missing_body_is_detected(self, tmp_path):
        path = tmp_path / "db.sqlite"
        db = _seeded_db(path)
        db.conn.execute("DELETE FROM content_blocks")
        db.conn.commit()
        db.close()
        report = audit_storage(db_path=path)
        assert any(f.category == "missing_body" for f in report.findings)

    def test_fts_drift_is_detected(self, tmp_path):
        path = tmp_path / "db.sqlite"
        db = _seeded_db(path)
        # Drop the derived index out from under the blocks (triggers bypassed).
        db.conn.execute("INSERT INTO content_fts(content_fts) VALUES('delete-all')")
        db.conn.commit()
        db.close()
        report = audit_storage(db_path=path)
        assert any(f.category == "fts_drift" for f in report.findings)

    def test_short_id_risk_is_reported(self, tmp_path):
        path = tmp_path / "db.sqlite"
        db = _seeded_db(path)
        db.close()
        conn = sqlite3.connect(path)
        # Legacy generator ids are ``p`` + 6 hex (7 chars).
        conn.execute("INSERT INTO papers (local_id, title) VALUES ('p0ab12c', 'short')")
        conn.commit()
        conn.close()
        report = audit_storage(db_path=path)
        assert any(f.category == "short_id_risk" for f in report.findings)
        assert report.by_severity()["info"] >= 1

    def test_summary_line_mentions_severities(self, tmp_path):
        path = tmp_path / "db.sqlite"
        _seeded_db(path).close()
        text = summarize(audit_storage(db_path=path))
        assert "schema v" in text and "errors" in text


class TestLegacyArtifacts:
    def test_missing_source_and_broken_tree(self, tmp_path):
        papers = tmp_path / "papers"
        directory = papers / "p1"
        directory.mkdir(parents=True)
        (directory / "raw.md").write_text("# body\n", encoding="utf-8")
        (directory / "tree.json").write_text("{not json", encoding="utf-8")
        db_path = tmp_path / "db.sqlite"
        _seeded_db(db_path).close()
        report = audit_storage(db_path=db_path, papers_root=papers)
        categories = report.by_category()
        assert categories.get("missing_source") == 1
        assert categories.get("broken_tree") == 1
        assert report.counts["paper_dirs"] == 1

    def test_legacy_vector_without_blob(self, tmp_path):
        path = tmp_path / "db.sqlite"
        db = _seeded_db(path)
        columns = {row[1] for row in db.conn.execute("PRAGMA table_info(tree_vectors)").fetchall()}
        base = {"node_id": "n1", "paper_id": "p1", "tree_layer": "pageindex"}
        values = {key: value for key, value in base.items() if key in columns}
        if "embedding" in columns:
            values["embedding"] = b""  # empty blob: unusable vector
        if "content_hash" in columns:
            values["content_hash"] = "h"
        names = ", ".join(values)
        placeholders = ", ".join("?" * len(values))
        db.conn.execute(
            f"INSERT INTO tree_vectors ({names}) VALUES ({placeholders})",
            tuple(values[key] for key in values),
        )
        if "embedding" not in columns:
            t.skip("tree_vectors has no embedding column in this schema")
        db.conn.commit()
        db.close()
        report = audit_storage(db_path=path)
        assert any(f.category == "legacy_vector_unusable" for f in report.findings)


class TestGenerations:
    def test_missing_generation_directory_is_reported(self, tmp_path):
        path = tmp_path / "db.sqlite"
        _seeded_db(path).close()
        report = audit_storage(db_path=path, storage_dir=tmp_path / "storage")
        assert any(f.category == "no_generation" for f in report.findings)

    def test_staging_leftover_is_reported(self, tmp_path):
        from drbrain.tree.publish import GENERATIONS_DIR_NAME, STAGING_PREFIX

        path = tmp_path / "db.sqlite"
        _seeded_db(path).close()
        storage = tmp_path / "storage"
        (storage / GENERATIONS_DIR_NAME / f"{STAGING_PREFIX}gen-x").mkdir(parents=True)
        report = audit_storage(db_path=path, storage_dir=storage)
        assert any(f.category == "staging_generation" for f in report.findings)
        assert any(f.category == "no_active_generation" for f in report.findings)

    def test_published_generation_verifies_clean(self, tmp_path):
        from drbrain.tree.publish import publish_tree_generation

        path = tmp_path / "db.sqlite"
        db = _seeded_db(path)
        storage = tmp_path / "storage"
        publish_tree_generation(db, storage, profile_id="emb-test")
        db.close()
        report = audit_storage(db_path=path, storage_dir=storage)
        assert not any(f.category == "generation_inconsistent" for f in report.findings)
        assert report.counts.get("active_generation") == 1
