"""Tests for batch ingest functionality."""
import tempfile
from pathlib import Path
from unittest import mock

from brbrain.extractor.concept import ExtractedConcepts
from brbrain.parser.mineru_parser import ParsedPaper

from brbrain.cli.commands import ingest_cmd


def _make_minimal_config(db_path: str, reports_dir: str) -> dict:
    """Return config dict for testing."""
    return {
        "db": {"path": db_path},
        "llm": {"models": [{"provider": "openai", "model": "gpt-4", "api_key": "x"}]},
        "mineru": {"token": "", "model": "vlm", "is_ocr": False, "enable_formula": True, "enable_table": True},
        "dirs": {"reports": reports_dir, "pdfs": "data/pdfs", "cache": "data/cache", "logs": "data/logs"},
        "api": {"s2_rate_limit": 100, "cache_ttl": 86400},
        "queue": {"weak_threshold": 0.7, "auto_accept": 0.9},
        "bm25": {"k1": 1.5, "b": 0.75},
        "openalex_token": None,
    }


def _make_parsed_paper(idx: int = 0) -> ParsedPaper:
    """Create a dummy ParsedPaper with unique metadata."""
    return ParsedPaper(
        title=f"Test Paper {idx}",
        year=2024 + idx,
        arxiv=f"2401.{10000 + idx}",
        text_blocks=[f"Introduction.—This is test paper #{idx} about ML."],
        raw_md=f"# Test Paper {idx}\n\nIntroduction.—This is test paper #{idx} about ML.",
    )


def _make_concepts(idx: int = 0) -> ExtractedConcepts:
    """Create dummy ExtractedConcepts with unique labels."""
    data = {
        "problems": [{"label": f"Test Problem {idx}", "confidence": 0.9}],
        "methods": [{"label": f"Test Method {idx}", "confidence": 0.9}],
        "conclusions": [], "debates": [], "gaps": [], "actors": [],
        "relations": [{"head": f"Test Method {idx}", "rel": "addresses", "tail": f"Test Problem {idx}"}],
        "arguments": [],
    }
    return ExtractedConcepts(data)


def _common_mocks(db_path: str, reports_dir: str):
    """Return a context manager that mocks load_config."""
    cfg = _make_minimal_config(db_path, reports_dir)
    return mock.patch("brbrain.cli.commands.load_config", return_value=cfg)


def test_ingest_single_file():
    """ingest_cmd processes a single PDF file successfully."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        reports_dir = Path(td) / "reports"
        reports_dir.mkdir()

        pdf_path = Path(td) / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 dummy")

        with _common_mocks(str(db_path), str(reports_dir)), \
             mock.patch("brbrain.cli.commands.extract_pdf", return_value=_make_parsed_paper(0)), \
             mock.patch("brbrain.cli.commands.extract_concepts", return_value=_make_concepts(0)):
            ingest_cmd([str(pdf_path)])

        from brbrain.storage.database import Database
        db = Database(str(db_path))
        papers = [p for p in db.get_all_papers() if p["status"] == "uploaded"]
        assert len(papers) == 1
        assert papers[0]["title"] == "Test Paper 0"
        db.close()


def test_ingest_directory():
    """ingest_cmd processes all PDFs in a directory."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        reports_dir = Path(td) / "reports"
        reports_dir.mkdir()
        pdfs_dir = Path(td) / "pdfs"
        pdfs_dir.mkdir()

        (pdfs_dir / "paper1.pdf").write_bytes(b"%PDF-1.4 dummy1")
        (pdfs_dir / "paper2.pdf").write_bytes(b"%PDF-1.4 dummy2")

        call_idx = [0]

        def extract_side_effect(path, cfg):
            idx = call_idx[0]
            call_idx[0] += 1
            return _make_parsed_paper(idx)

        def concepts_side_effect(text, models):
            idx = call_idx[0] - 1  # Use the last assigned index
            return _make_concepts(idx)

        with _common_mocks(str(db_path), str(reports_dir)), \
             mock.patch("brbrain.cli.commands.extract_pdf", side_effect=extract_side_effect), \
             mock.patch("brbrain.cli.commands.extract_concepts", side_effect=concepts_side_effect):
            ingest_cmd([str(pdfs_dir)])

        from brbrain.storage.database import Database
        db = Database(str(db_path))
        papers = [p for p in db.get_all_papers() if p["status"] == "uploaded"]
        assert len(papers) == 2
        titles = {p["title"] for p in papers}
        assert titles == {"Test Paper 0", "Test Paper 1"}
        db.close()


def test_ingest_skips_failed_papers():
    """ingest_cmd continues processing when one paper fails."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        reports_dir = Path(td) / "reports"
        reports_dir.mkdir()
        pdfs_dir = Path(td) / "pdfs"
        pdfs_dir.mkdir()

        (pdfs_dir / "ok1.pdf").write_bytes(b"%PDF-1.4 dummy")
        (pdfs_dir / "fail.pdf").write_bytes(b"%PDF-1.4 fail")
        (pdfs_dir / "ok2.pdf").write_bytes(b"%PDF-1.4 dummy")

        call_count = 0
        paper_idx = [0]

        def side_effect_extract(path, cfg):
            nonlocal call_count
            call_count += 1
            if "fail" in str(path):
                raise Exception("simulated parse failure")
            idx = paper_idx[0]
            paper_idx[0] += 1
            return _make_parsed_paper(idx)

        def concepts_side_effect(text, models):
            idx = paper_idx[0] - 1
            return _make_concepts(idx)

        with _common_mocks(str(db_path), str(reports_dir)), \
             mock.patch("brbrain.cli.commands.extract_pdf", side_effect=side_effect_extract), \
             mock.patch("brbrain.cli.commands.extract_concepts", side_effect=concepts_side_effect):
            ingest_cmd([str(pdfs_dir)])

        # Should have attempted all 3 files
        assert call_count == 3

        from brbrain.storage.database import Database
        db = Database(str(db_path))
        papers = [p for p in db.get_all_papers() if p["status"] == "uploaded"]
        # Only 2 should succeed
        assert len(papers) == 2
        db.close()


def test_ingest_multiple_files():
    """ingest_cmd handles multiple individual file paths."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        reports_dir = Path(td) / "reports"
        reports_dir.mkdir()

        pdf1 = Path(td) / "a.pdf"
        pdf2 = Path(td) / "b.pdf"
        pdf1.write_bytes(b"%PDF-1.4 dummy")
        pdf2.write_bytes(b"%PDF-1.4 dummy")

        call_idx = [0]

        def extract_side_effect(path, cfg):
            idx = call_idx[0]
            call_idx[0] += 1
            return _make_parsed_paper(idx)

        def concepts_side_effect(text, models):
            idx = call_idx[0] - 1
            return _make_concepts(idx)

        with _common_mocks(str(db_path), str(reports_dir)), \
             mock.patch("brbrain.cli.commands.extract_pdf", side_effect=extract_side_effect), \
             mock.patch("brbrain.cli.commands.extract_concepts", side_effect=concepts_side_effect):
            ingest_cmd([str(pdf1), str(pdf2)])

        from brbrain.storage.database import Database
        db = Database(str(db_path))
        papers = [p for p in db.get_all_papers() if p["status"] == "uploaded"]
        assert len(papers) == 2
        db.close()


def test_ingest_exits_when_no_pdfs():
    """ingest_cmd raises Exit when no PDF files found."""
    import click
    with tempfile.TemporaryDirectory() as td:
        empty_dir = Path(td) / "empty"
        empty_dir.mkdir()

        try:
            ingest_cmd([str(empty_dir)])
            assert False, "Should have raised Exit"
        except click.exceptions.Exit as e:
            assert e.exit_code == 1
