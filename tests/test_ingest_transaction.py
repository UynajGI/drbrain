"""Transactional guarantees for the CLI paper-ingest helper."""

from pathlib import Path
from unittest.mock import patch

import pytest

from drbrain.cli._helpers.db_ingest import _ingest_single_paper
from drbrain.dedup.resolver import DedupEngine
from drbrain.parser.mineru_parser import ParsedPaper
from drbrain.storage.database import Database


def test_new_paper_is_not_committed_before_artifacts(tmp_path: Path) -> None:
    """An artifact failure must not leave a durable paper row behind."""
    db = Database(tmp_path / "db.sqlite")
    source_pdf = tmp_path / "input.pdf"
    source_pdf.write_bytes(b"%PDF-1.4 test")
    cfg = {
        "dirs": {"papers": str(tmp_path / "papers")},
        "llm": {"models": [{"provider": "test", "model": "test"}]},
    }
    parsed = ParsedPaper(
        title="Transactional paper",
        year=2024,
        doi="10.1234/transactional",
        raw_md="# Transactional paper\n",
    )

    try:
        with (
            patch("drbrain.cli._helpers.db_ingest.extract_pdf", return_value=parsed),
            patch("drbrain.extractor.openalex.search_authors_by_work", return_value=[]),
            patch(
                "drbrain.cli._helpers.db_ingest._save_paper_artifacts",
                side_effect=OSError("artifact write failed"),
            ),
        ):
            with pytest.raises(OSError, match="artifact write failed"):
                _ingest_single_paper(
                    source_pdf,
                    cfg,
                    db,
                    DedupEngine(db),
                    json_mode=True,
                )

        # Explicitly commit after the failed call: if the helper had already
        # committed its new-paper row, this would make the orphan observable.
        db.commit()
        assert db.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 0
        assert db.conn.execute("SELECT COUNT(*) FROM paper_ids").fetchone()[0] == 0
    finally:
        db.close()


@pytest.mark.parametrize("failure", ["ids", "authors"])
def test_new_paper_rolls_back_identity_side_effects(tmp_path: Path, failure: str) -> None:
    """Failures before artifact publication must not leave paper/ID rows."""
    db = Database(tmp_path / f"{failure}.sqlite")
    source_pdf = tmp_path / f"{failure}.pdf"
    source_pdf.write_bytes(b"%PDF-1.4 test")
    cfg = {
        "dirs": {"papers": str(tmp_path / f"papers-{failure}")},
        "llm": {"models": [{"provider": "test", "model": "test"}]},
    }
    parsed = ParsedPaper(
        title=f"Transactional {failure}",
        year=2024,
        doi=f"10.1234/{failure}",
        raw_md="# Transactional paper\n",
    )

    try:
        id_patch = patch.object(
            db,
            "insert_paper_ids",
            side_effect=RuntimeError("identity write failed"),
        )
        author_patch = patch(
            "drbrain.extractor.openalex.search_authors_by_work",
            side_effect=RuntimeError("author lookup failed"),
        )
        with (
            patch("drbrain.cli._helpers.db_ingest.extract_pdf", return_value=parsed),
            id_patch
            if failure == "ids"
            else patch.object(db, "insert_paper_ids", wraps=db.insert_paper_ids),
            author_patch
            if failure == "authors"
            else patch("drbrain.extractor.openalex.search_authors_by_work", return_value=[]),
            patch("drbrain.cli._helpers.db_ingest._save_paper_artifacts", return_value=None),
        ):
            with pytest.raises(RuntimeError):
                _ingest_single_paper(source_pdf, cfg, db, DedupEngine(db), json_mode=True)

        db.commit()
        assert db.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 0
        assert db.conn.execute("SELECT COUNT(*) FROM paper_ids").fetchone()[0] == 0
    finally:
        db.close()
