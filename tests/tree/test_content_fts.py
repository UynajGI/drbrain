"""T12: BM25 over canonical blocks via the external-content FTS index."""

from __future__ import annotations

import hashlib

import pytest

from drbrain.storage.database import Database
from drbrain.tree.blocks import build_content_blocks

DOC_A = "# Neural Retrieval\n\nSparse retrieval with BM25 over tokenized passages.\n\n## Methods\n\nWe fuse sparse and dense retrieval signals.\n"
DOC_B = "# Crop Yields\n\nIrrigation scheduling improves maize yields under drought.\n"


def _seed(db: Database, local_id: str, text: str, revision: int = 1):
    db.insert_paper(local_id, "T", 2024, "uploaded")
    blocks = build_content_blocks(
        text, local_id=local_id, revision=revision, media_type="md", parser="test"
    )
    db.upsert_document_revision(
        local_id,
        revision,
        source_hash=f"src-{local_id}-{revision}",
        canonical_hash=hashlib.sha256(text.encode()).hexdigest(),
        backend="test",
        media_type="md",
    )
    db.insert_content_blocks(blocks)
    return blocks


class TestIndexConsistency:
    def test_index_matches_blocks_after_insert(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        _seed(db, "p1", DOC_A)
        status = db.content_fts_status()
        assert status["consistent"] and status["indexed"] > 0

    def test_match_is_immediate_and_updates_consistently(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        _seed(db, "p1", DOC_A)
        hits = db.search_content("tokenized")
        assert hits and all(hit["local_id"] == "p1" for hit in hits)
        # Update a block in place: index must follow immediately.
        row = db.conn.execute(
            "SELECT block_id, text FROM content_blocks WHERE text LIKE '%tokenized%'"
        ).fetchone()
        new_text = row[1].replace("tokenized", "chunked")
        db.conn.execute(
            "UPDATE content_blocks SET text = ?, text_hash = ? WHERE block_id = ?",
            (new_text, hashlib.sha256(new_text.encode()).hexdigest(), row[0]),
        )
        db.conn.commit()
        assert db.search_content("tokenized") == []
        assert db.search_content("chunked")

    def test_rollback_leaves_no_index_rows(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        _seed(db, "p1", DOC_A)
        before = db.content_fts_status()
        with pytest.raises(ValueError):
            # A bad block set fails before writing; nothing may be indexed.
            bad = build_content_blocks(DOC_B, local_id="p1", revision=1, media_type="md")
            bad = [
                type(bad[0])(
                    block_id="cb-rollback",
                    local_id="p1",
                    revision=1,
                    ordinal=len(bad),
                    text="phantom text",
                    text_hash=hashlib.sha256(b"phantom text").hexdigest(),
                    char_start=bad[-1].char_end,
                    char_end=bad[-1].char_end + len("phantom text"),
                )
            ]
            db.insert_content_blocks(bad)
        assert db.content_fts_status() == before
        assert db.search_content("phantom") == []

    def test_rebuild_restores_index(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        _seed(db, "p1", DOC_A)
        db.conn.execute("INSERT INTO content_fts(content_fts) VALUES('delete-all')")
        db.conn.commit()
        assert db.search_content("bm25") == []
        rebuilt = db.rebuild_content_fts()
        assert rebuilt > 0
        assert db.search_content("bm25")


class TestSearch:
    def test_ranking_is_relevance_ordered(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        _seed(db, "p1", DOC_A)
        _seed(db, "p2", DOC_B)
        hits = db.search_content("retrieval")
        assert hits
        assert hits[0]["local_id"] == "p1"
        assert hits[0]["score"] <= hits[-1]["score"]

    def test_scope_filter_limits_documents(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        _seed(db, "p1", DOC_A)
        _seed(db, "p2", DOC_B)
        scoped = db.search_content("retrieval", local_id="p2")
        assert scoped == []
        assert db.search_content("retrieval", local_id="p1")

    def test_results_carry_exact_locators(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        blocks = _seed(db, "p1", DOC_A)
        hit = db.search_content("irrigation" if False else "tokenized")[0]
        block = next(item for item in blocks if item.block_id == hit["block_id"])
        assert hit["char_start"] == block.char_start and hit["char_end"] == block.char_end
        assert hit["text_hash"] == block.text_hash
        assert hit["snippet"]

    def test_empty_and_invalid_queries(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        _seed(db, "p1", DOC_A)
        with pytest.raises(ValueError, match="non-empty"):
            db.search_content("   ")
        with pytest.raises(ValueError, match="invalid FTS query"):
            db.search_content("unbalanced(")
