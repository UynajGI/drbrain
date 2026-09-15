"""T08: canonical content + document revision storage."""

from __future__ import annotations

import hashlib
import sqlite3

import pytest

from drbrain.storage.database import Database
from drbrain.tree.contracts import ContentBlock, content_block_id


def _block(local_id: str, revision: int, ordinal: int, text: str, start: int, **kw) -> ContentBlock:
    return ContentBlock(
        block_id=content_block_id(local_id, revision, ordinal),
        local_id=local_id,
        revision=revision,
        ordinal=ordinal,
        text=text,
        text_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        char_start=start,
        char_end=start + len(text),
        **kw,
    )


def _revision(db: Database, local_id: str = "p1", revision: int = 1, canonical: str = "alpha beta"):
    db.insert_paper(local_id, "T", 2024, "uploaded")
    db.upsert_document_revision(
        local_id,
        revision,
        source_hash="src-hash",
        canonical_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        backend="test",
        media_type="md",
    )
    return canonical


def _two_blocks(local_id: str = "p1", revision: int = 1):
    return [
        _block(local_id, revision, 0, "alpha ", 0, heading_path=("Intro",), kind="title"),
        _block(local_id, revision, 1, "beta", 6, heading_path=("Intro", "Body")),
    ]


class TestSchema:
    def test_fresh_db_has_content_store(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        versions = {int(row[0]) for row in db.conn.execute("SELECT version FROM schema_versions")}
        assert 23 in versions and 24 in versions
        names = {
            row[0] for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {"document_revisions", "content_blocks"} <= names

    def test_old_database_upgrades_in_place(self, tmp_path):
        path = tmp_path / "old.sqlite"
        conn = sqlite3.connect(path)
        conn.executescript(
            """
            CREATE TABLE schema_versions (version INTEGER PRIMARY KEY, applied_at TIMESTAMP);
            INSERT INTO schema_versions (version) VALUES (20);
            CREATE TABLE papers (
                local_id TEXT PRIMARY KEY, title TEXT NOT NULL, abstract TEXT DEFAULT '',
                year INTEGER, paper_type TEXT NOT NULL DEFAULT 'paper',
                status TEXT NOT NULL DEFAULT 'placeholder', journal TEXT DEFAULT '',
                publisher TEXT DEFAULT '', citation_count INTEGER DEFAULT 0,
                volume TEXT DEFAULT '', pages TEXT DEFAULT '', authors TEXT DEFAULT '',
                categories TEXT DEFAULT '', created_at TIMESTAMP, updated_at TIMESTAMP,
                embedding_revision INTEGER DEFAULT 0
            );
            """
        )
        conn.commit()
        conn.close()
        db = Database(path)
        versions = {int(row[0]) for row in db.conn.execute("SELECT version FROM schema_versions")}
        assert 21 in versions and 22 in versions and 23 in versions and 24 in versions


class TestRevisionWrites:
    def test_roundtrip_and_latest_selection(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        _revision(db, "p1", 1, "alpha beta")
        assert db.get_document_revision("p1")["revision"] == 1
        assert db.next_document_revision("p1") == 2
        db.upsert_document_revision(
            "p1",
            2,
            source_hash="src-hash-2",
            canonical_hash=hashlib.sha256(b"gamma").hexdigest(),
            media_type="md",
        )
        assert db.get_document_revision("p1")["revision"] == 2
        assert [row["revision"] for row in db.list_document_revisions("p1")] == [1, 2]

    def test_same_revision_with_new_content_is_rejected(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        _revision(db, "p1", 1, "alpha beta")
        with pytest.raises(ValueError, match="different hashes"):
            db.upsert_document_revision(
                "p1",
                1,
                source_hash="other",
                canonical_hash=hashlib.sha256(b"other").hexdigest(),
                media_type="md",
            )

    def test_state_lifecycle(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        _revision(db, "p1", 1)
        db.set_document_revision_state("p1", 1, "stale")
        assert db.get_document_revision("p1")["state"] == "stale"
        with pytest.raises(ValueError, match="unknown document revision"):
            db.set_document_revision_state("p1", 9, "ready")


class TestBlockWrites:
    def test_blocks_roundtrip_reconstructs_text(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        canonical = _revision(db, "p1", 1, "alpha beta")
        written = db.insert_content_blocks(_two_blocks())
        assert written == 2
        rows = db.get_content_blocks("p1")
        assert "".join(row["text"] for row in rows) == canonical
        assert rows[0]["kind"] == "title" and rows[1]["heading_path"] == '["Intro", "Body"]'
        assert db.count_content_blocks("p1") == 2

    def test_repeat_insert_is_idempotent(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        _revision(db, "p1", 1, "alpha beta")
        db.insert_content_blocks(_two_blocks())
        assert db.insert_content_blocks(_two_blocks()) == 0
        assert db.count_content_blocks("p1") == 2

    def test_conflicting_block_id_is_rejected(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        _revision(db, "p1", 1, "alpha beta")
        db.insert_content_blocks(_two_blocks())
        conflicting = _two_blocks()
        bad = ContentBlock(
            block_id=conflicting[0].block_id,
            local_id="p1",
            revision=1,
            ordinal=0,
            text="alpha!",  # same id, different text
            text_hash=hashlib.sha256(b"alpha!").hexdigest(),
            char_start=0,
            char_end=6,
        )
        with pytest.raises(ValueError):
            db.insert_content_blocks([bad, conflicting[1]])

    def test_noncontiguous_blocks_fail_without_writes(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        _revision(db, "p1", 1, "alpha beta")
        gap = [
            _block("p1", 1, 0, "alpha ", 0),
            _block("p1", 1, 1, "beta", 7),  # gap: 6 != 7
        ]
        with pytest.raises(ValueError, match="contiguous"):
            db.insert_content_blocks(gap)
        assert db.count_content_blocks("p1") == 0

    def test_canonical_hash_mismatch_is_rejected(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        _revision(db, "p1", 1, "alpha beta")
        wrong = [_block("p1", 1, 0, "not the body", 0)]
        with pytest.raises(ValueError, match="canonical text hash"):
            db.insert_content_blocks(wrong)
        assert db.count_content_blocks("p1") == 0

    def test_unknown_revision_is_rejected(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        db.insert_paper("p1", "T", 2024, "uploaded")
        with pytest.raises(ValueError, match="does not exist"):
            db.insert_content_blocks([_block("p1", 1, 0, "x", 0)])

    def test_same_text_different_provenance_stays_distinct(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        _revision(db, "p1", 1, "same text")
        _revision(db, "p2", 1, "same text")
        db.insert_content_blocks([_block("p1", 1, 0, "same text", 0)])
        db.insert_content_blocks([_block("p2", 1, 0, "same text", 0)])
        assert db.count_content_blocks() == 2
        rows_p1 = db.get_content_blocks("p1")
        rows_p2 = db.get_content_blocks("p2")
        assert rows_p1[0]["block_id"] != rows_p2[0]["block_id"]

    def test_blocks_of_older_revision_remain_readable(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        _revision(db, "p1", 1, "alpha beta")
        db.insert_content_blocks(_two_blocks("p1", 1))
        db.upsert_document_revision(
            "p1",
            2,
            source_hash="src-hash-2",
            canonical_hash=hashlib.sha256(b"gamma").hexdigest(),
            media_type="md",
        )
        db.insert_content_blocks([_block("p1", 2, 0, "gamma", 0)])
        assert "".join(row["text"] for row in db.get_content_blocks("p1", 1)) == "alpha beta"
        assert "".join(row["text"] for row in db.get_content_blocks("p1")) == "gamma"
