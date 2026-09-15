"""T10: canonical content read API."""

from __future__ import annotations

import hashlib

import pytest

from drbrain.storage.content import (
    ContentUnavailableError,
    block_at_offset,
    blocks_in_lines,
    blocks_in_pages,
    read_range,
    read_section,
    read_text,
    resolve_revision,
    sections,
    summarize_coverage,
    text_hash,
)
from drbrain.storage.database import Database
from drbrain.tree.blocks import build_content_blocks

FIXTURE_MD = "# Title\n\nIntro paragraph.\n\n## Methods\n\nStep one.\nStep two.\n\n## Results\n\nOutcome text.\n"


def _seed(db: Database, local_id: str = "p1", revision: int = 1, text: str = FIXTURE_MD):
    db.insert_paper(local_id, "T", 2024, "uploaded")
    blocks = build_content_blocks(
        text, local_id=local_id, revision=revision, media_type="md", parser="test"
    )
    db.upsert_document_revision(
        local_id,
        revision,
        source_hash="src",
        canonical_hash=hashlib.sha256(text.encode()).hexdigest(),
        backend="test",
        media_type="md",
    )
    db.insert_content_blocks(blocks)
    return blocks


class TestReads:
    def test_full_text_roundtrip(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        _seed(db)
        assert read_text(db, "p1") == FIXTURE_MD
        assert text_hash(read_text(db, "p1")) == hashlib.sha256(FIXTURE_MD.encode()).hexdigest()

    def test_range_reads_cross_blocks(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        _seed(db)
        whole = read_text(db, "p1")
        start = whole.index("Intro")
        end = whole.index("Step two") + len("Step two")
        assert read_range(db, "p1", start, end) == whole[start:end]

    def test_range_boundaries_are_exact(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        _seed(db)
        whole = read_text(db, "p1")
        assert read_range(db, "p1", 0, 1) == whole[0]
        assert read_range(db, "p1", 0, len(whole)) == whole
        assert read_range(db, "p1", len(whole) - 1, len(whole)) == whole[-1]

    def test_invalid_ranges_raise(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        _seed(db)
        with pytest.raises(ContentUnavailableError, match="outside"):
            read_range(db, "p1", -1, 5)
        with pytest.raises(ContentUnavailableError, match="outside"):
            read_range(db, "p1", 0, 99999)
        with pytest.raises(ContentUnavailableError, match="outside"):
            read_range(db, "p1", 5, 5)

    def test_unknown_document_raises(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        with pytest.raises(ContentUnavailableError, match="no canonical content"):
            read_text(db, "missing")

    def test_repeated_reads_hash_stable(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        _seed(db)
        assert text_hash(read_text(db, "p1")) == text_hash(read_text(db, "p1"))
        assert summarize_coverage(db, "p1")["canonical_hash"] == text_hash(FIXTURE_MD)


class TestRevisions:
    def test_old_revision_remains_readable(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        _seed(db, "p1", 1, "old body")
        _seed(db, "p1", 2, "new body")
        assert read_text(db, "p1", 1) == "old body"
        assert read_text(db, "p1") == "new body"
        assert resolve_revision(db, "p1") == 2
        with pytest.raises(ContentUnavailableError):
            read_text(db, "p1", 99)


class TestStructure:
    def test_sections_group_by_heading_path(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        _seed(db)
        found = sections(db, "p1")
        anchors = [section["anchor"] for section in found]
        assert "Methods" in anchors and "Results" in anchors
        methods = next(section for section in found if section["anchor"] == "Methods")
        assert "Step one" in methods["text"] and "Outcome" not in methods["text"]
        assert methods["char_hash"] == text_hash(methods["text"])

    def test_read_section_by_anchor(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        _seed(db)
        section = read_section(db, "p1", "Results")
        assert "Outcome text" in section["text"]
        with pytest.raises(ContentUnavailableError, match="no section"):
            read_section(db, "p1", "Nonexistent")

    def test_block_at_offset(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        _seed(db)
        whole = read_text(db, "p1")
        block = block_at_offset(db, "p1", whole.index("Step one"))
        assert block is not None and "Step one" in block.text
        assert block_at_offset(db, "p1", len(whole)) is None

    def test_page_and_line_filters(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        db.insert_paper("pdf1", "T", 2024, "uploaded")
        text = "page one\npage two\n"
        split = text.index("page two")
        blocks = build_content_blocks(
            text,
            local_id="pdf1",
            revision=1,
            media_type="pdf",
            page_marks=[(1, 0), (2, split)],
        )
        db.upsert_document_revision(
            "pdf1",
            1,
            source_hash="s",
            canonical_hash=hashlib.sha256(text.encode()).hexdigest(),
            media_type="pdf",
        )
        db.insert_content_blocks(blocks)
        page_two = blocks_in_pages(db, "pdf1", 2, 2)
        assert len(page_two) == 1 and "page two" in page_two[0].text
        assert blocks_in_lines(db, "pdf1", 1, 1) == []  # PDFs carry no lines
