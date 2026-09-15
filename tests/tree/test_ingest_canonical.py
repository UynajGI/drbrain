"""T22: ingest writes canonical content; page alignment stays honest."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from drbrain.storage.database import Database
from drbrain.storage.inbox import file_sha256
from drbrain.tree.align import align_page_marks, normalize_with_map, page_span_for_offset


def _seed_paper(db: Database, local_id: str = "p1") -> None:
    db.insert_paper(local_id, "T", 2024, "uploaded")


def _parsed(text: str, backend: str = "test") -> SimpleNamespace:
    return SimpleNamespace(raw_md=text, backend=backend, pdf_type="")


class TestCanonicalWrite:
    def test_markdown_write_creates_revision_blocks_and_leaves(self, tmp_path):
        from drbrain.cli._helpers.db_ingest import _write_canonical_content

        db = Database(tmp_path / "db.sqlite")
        _seed_paper(db)
        source = tmp_path / "paper.md"
        text = "# Title\n\nfirst paragraph with words\n\n# Second\n\nsecond paragraph\n"
        source.write_text(text, encoding="utf-8")
        result = _write_canonical_content(db, "p1", _parsed(text), source)
        assert result["ok"] and not result["reused"]
        revision = db.get_document_revision("p1")
        assert revision["revision"] == 1
        assert revision["media_type"] == "md"
        blocks = db.get_content_blocks("p1")
        assert "".join(block["text"] for block in blocks) == text
        leaves = db.list_tree_nodes(kind="leaf", state="ready")
        assert len(leaves) == len(blocks)
        assert db.leaves_missing_parent() != []  # roots until a parent exists

    def test_reingest_same_bytes_reuses_revision(self, tmp_path):
        from drbrain.cli._helpers.db_ingest import _write_canonical_content

        db = Database(tmp_path / "db.sqlite")
        _seed_paper(db)
        source = tmp_path / "paper.md"
        text = "# Title\n\nbody text here\n"
        source.write_text(text, encoding="utf-8")
        first = _write_canonical_content(db, "p1", _parsed(text), source)
        second = _write_canonical_content(db, "p1", _parsed(text), source)
        assert second["reused"] and second["revision"] == first["revision"]
        assert db.count_content_blocks("p1") == first["blocks"]
        assert len(db.list_tree_nodes(kind="leaf")) == first["blocks"]

    def test_changed_content_gets_a_new_revision_and_keeps_the_old(self, tmp_path):
        from drbrain.cli._helpers.db_ingest import _write_canonical_content

        db = Database(tmp_path / "db.sqlite")
        _seed_paper(db)
        source = tmp_path / "paper.md"
        source.write_text("# One\n\noriginal body\n", encoding="utf-8")
        _write_canonical_content(db, "p1", _parsed("# One\n\noriginal body\n"), source)
        source.write_text("# One\n\nupdated body\n", encoding="utf-8")
        result = _write_canonical_content(db, "p1", _parsed("# One\n\nupdated body\n"), source)
        assert result["revision"] == 2
        from drbrain.storage.content import read_text

        assert read_text(db, "p1", 1) == "# One\n\noriginal body\n"
        assert read_text(db, "p1") == "# One\n\nupdated body\n"

    def test_input_hash_is_unchanged_by_ingest(self, tmp_path):
        from drbrain.cli._helpers.db_ingest import _write_canonical_content

        db = Database(tmp_path / "db.sqlite")
        _seed_paper(db)
        source = tmp_path / "paper.tex"
        text = "\\section{Intro}\n\nbody\n"
        source.write_text(text, encoding="utf-8")
        before = file_sha256(source)
        _write_canonical_content(db, "p1", _parsed(text, "latex-native"), source)
        assert source.exists() and file_sha256(source) == before
        assert db.get_document_revision("p1")["media_type"] == "tex"

    def test_empty_text_is_reported_not_written(self, tmp_path):
        from drbrain.cli._helpers.db_ingest import _write_canonical_content

        db = Database(tmp_path / "db.sqlite")
        _seed_paper(db)
        source = tmp_path / "paper.md"
        source.write_text("", encoding="utf-8")
        result = _write_canonical_content(db, "p1", _parsed("   "), source)
        assert not result["ok"] and result["reason"] == "empty_canonical_text"
        assert db.get_document_revision("p1") is None


class TestIngestRegistersCanonicalOnly:
    """T22: a full ingest registers canonical content and writes no MD/tree files."""

    def _cfg(self, tmp_path: Path) -> dict:
        return {
            "db": {"path": str(tmp_path / "db.sqlite")},
            "dirs": {"papers": str(tmp_path / "papers")},
            "llm": {"models": [{"provider": "test", "model": "test"}]},
        }

    def _run_ingest(
        self,
        tmp_path: Path,
        monkeypatch,
        *,
        suffix: str = ".pdf",
        text: str = "# Title\n\nAbstract\n\nBody paragraph about the method.\n",
    ) -> tuple[Database, Path, dict]:
        """Run the real ingest helper offline against one input material."""
        from drbrain.cli._helpers.db_ingest import _ingest_single_paper
        from drbrain.dedup.resolver import DedupEngine
        from drbrain.parser.mineru_parser import ParsedPaper

        monkeypatch.setenv("DRBRAIN_OFFLINE", "1")
        db = Database(tmp_path / "db.sqlite")
        source = tmp_path / f"input{suffix}"
        if suffix == ".pdf":
            source.write_bytes(b"%PDF-1.4 test bytes")
        else:
            source.write_text(text, encoding="utf-8")
        parsed = ParsedPaper(title="Canonical ingest", year=2024, raw_md=text)
        cfg = self._cfg(tmp_path)
        with (
            mock.patch("drbrain.cli._helpers.db_ingest.extract_pdf", return_value=parsed),
            mock.patch("drbrain.cli._helpers.db_ingest.extract_material", return_value=parsed),
            mock.patch("drbrain.extractor.openalex.search_authors_by_work", return_value=[]),
        ):
            result = _ingest_single_paper(source, cfg, db, DedupEngine(db), json_mode=True)
        return db, source, result

    @pytest.mark.parametrize(
        ("suffix", "media_type"),
        [(".pdf", "pdf"), (".md", "md"), (".tex", "tex")],
    )
    def test_ingest_writes_no_per_paper_md_or_tree_files(
        self, tmp_path, monkeypatch, suffix, media_type
    ):
        db, _source, result = self._run_ingest(tmp_path, monkeypatch, suffix=suffix)
        try:
            assert result["ok"] is True, result
            pid = result["local_id"]
            paper_dir = Path(self._cfg(tmp_path)["dirs"]["papers"]) / pid
            assert not (paper_dir / "raw.md").exists()
            assert not (paper_dir / "tree.json").exists()
            assert (paper_dir / f"source{suffix}").exists()

            revision = db.get_document_revision(pid)
            assert revision["state"] == "ready" and revision["revision"] == 1
            assert revision["media_type"] == media_type
            blocks = db.count_content_blocks(pid, 1)
            assert blocks > 0
            assert len(db.list_tree_nodes(kind="leaf", state="ready")) == blocks

            raw = db.get_paper_artifact(pid, "raw")
            assert raw["status"] == "ready"
            metadata = json.loads(raw["metadata_json"])
            assert metadata["revision"] == 1 and metadata["blocks"] == blocks
            # Ingest must not claim a hierarchy it did not build.
            assert db.get_paper_artifact(pid, "tree")["status"] == "skipped"
            assert db.get_paper(pid)["status"] == "uploaded"
        finally:
            db.close()

    def test_canonical_failure_fails_the_paper(self, tmp_path, monkeypatch):
        """A canonical failure is a paper failure, not a warning to continue past."""
        from drbrain.cli._helpers.db_ingest import _ingest_single_paper
        from drbrain.dedup.resolver import DedupEngine

        monkeypatch.setenv("DRBRAIN_OFFLINE", "1")
        db = Database(tmp_path / "db.sqlite")
        source = tmp_path / "input.pdf"
        source.write_bytes(b"%PDF-1.4 test bytes")
        try:
            with (
                mock.patch(
                    "drbrain.cli._helpers.db_ingest.extract_pdf",
                    return_value=_parsed_for_retry(),
                ),
                mock.patch("drbrain.extractor.openalex.search_authors_by_work", return_value=[]),
                mock.patch(
                    "drbrain.cli._helpers.db_ingest._write_canonical_content",
                    return_value={"ok": False, "reason": "empty_canonical_text"},
                ),
            ):
                failed = _ingest_single_paper(
                    source, self._cfg(tmp_path), db, DedupEngine(db), json_mode=True
                )
            assert failed["ok"] is False
            assert "canonical" in str(failed["error"])
            # A new paper must not be published without its body.
            assert db.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 0
            assert db.conn.execute("SELECT COUNT(*) FROM document_revisions").fetchone()[0] == 0
            entry = db.get_spool_input(file_sha256(source))
            assert entry is not None and entry["status"] == "failed"
            assert "canonical" in str(entry["reason"])
        finally:
            db.close()

    def test_canonical_retry_after_failure_registers_the_body(self, tmp_path, monkeypatch):
        import drbrain.cli._helpers.db_ingest as db_ingest_mod
        from drbrain.dedup.resolver import DedupEngine

        monkeypatch.setenv("DRBRAIN_OFFLINE", "1")
        db = Database(tmp_path / "db.sqlite")
        source = tmp_path / "input.pdf"
        source.write_bytes(b"%PDF-1.4 test bytes")
        try:
            with (
                mock.patch(
                    "drbrain.cli._helpers.db_ingest.extract_pdf",
                    return_value=_parsed_for_retry(),
                ),
                mock.patch("drbrain.extractor.openalex.search_authors_by_work", return_value=[]),
                mock.patch(
                    "drbrain.cli._helpers.db_ingest._write_canonical_content",
                    side_effect=lambda *a, **k: {"ok": False, "reason": "transient"},
                ),
            ):
                failed = db_ingest_mod._ingest_single_paper(
                    source, self._cfg(tmp_path), db, DedupEngine(db), json_mode=True
                )
            assert failed["ok"] is False
            db.commit()
            assert db.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 0
        finally:
            db.close()

        db2, _source, second = self._run_ingest(tmp_path, monkeypatch)
        try:
            assert second["ok"] is True, second
            revision = db2.get_document_revision(second["local_id"])
            assert revision is not None and revision["state"] == "ready"
        finally:
            db2.close()


def _parsed_for_retry():
    from drbrain.parser.mineru_parser import ParsedPaper

    return ParsedPaper(
        title="Canonical ingest",
        year=2024,
        raw_md="# Title\n\nAbstract\n\nBody paragraph about the method.\n",
    )


class TestPageAlignment:
    def test_alignment_verifies_pages_in_order(self, tmp_path):
        fitz = pytest.importorskip("fitz")
        pdf_path = tmp_path / "paper.pdf"
        document = fitz.open()
        page_texts = []
        for index in range(3):
            page = document.new_page()
            text = f"Page {index + 1} heading\n\nbody of page {index + 1}\n"
            page.insert_text((72, 72), text)
            page_texts.append(text)
        document.save(str(pdf_path))
        document.close()
        canonical = "\n".join(page_texts)
        marks = align_page_marks(pdf_path, canonical)
        assert marks is not None
        assert [page for page, _offset in marks] == [1, 2, 3]
        assert marks[0][1] == 0
        offsets = [offset for _page, offset in marks]
        assert offsets == sorted(offsets)
        assert page_span_for_offset(marks, offsets[1]) == 2

    def test_unverifiable_pages_yield_no_marks(self, tmp_path):
        fitz = pytest.importorskip("fitz")
        pdf_path = tmp_path / "paper.pdf"
        document = fitz.open()
        page = document.new_page()
        page.insert_text((72, 72), "Real page text\n")
        document.save(str(pdf_path))
        document.close()
        # The canonical text does not contain the page's text at all.
        assert align_page_marks(pdf_path, "completely different body\n") is None

    def test_missing_pdf_yields_no_marks(self, tmp_path):
        assert align_page_marks(tmp_path / "missing.pdf", "some text\n") is None

    def test_normalize_map_roundtrip(self):
        normalized, index_map = normalize_with_map("a  b\n\nc")
        assert normalized == "a b c"
        for position, char in enumerate(normalized):
            source_index = index_map[position]
            assert source_index < len("a  b\n\nc")
            assert char in ("a", "b", "c", " ")

    def test_pdf_blocks_without_marks_have_no_page_fields(self, tmp_path):
        from drbrain.tree.blocks import build_content_blocks

        blocks = build_content_blocks(
            "body text without verified pages\n",
            local_id="p1",
            revision=1,
            media_type="pdf",
        )
        assert all(block.page_start is None and block.page_end is None for block in blocks)
        assert all(block.line_start is None for block in blocks)
