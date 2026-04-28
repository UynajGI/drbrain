"""Integration tests using real PDFs from data/pdfs/.

These tests are slow (MinerU API + LLM calls) and are marked with the
``integration`` pytest marker.  Run them explicitly when you want to
validate the full pipeline end-to-end::

    pytest tests/test_integration.py -v -m integration
"""
import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest
from typer.testing import CliRunner

from drbrain.cli.main import app

runner = CliRunner()

# All PDFs available in the repo
PDF_DIR = Path("data/pdfs")
TEST_PDFS = sorted(str(p) for p in PDF_DIR.glob("*.pdf"))


def _minimal_cfg(db_path: str, reports_dir: str) -> dict:
    """Config that mocks load_config but leaves MinerU/LLM calls untouched."""
    return {
        "db": {"path": str(db_path)},
        "llm": {
            "models": [
                {
                    "provider": "openai",
                    "model": "qwen3.6-plus",
                    "api_key": "sk-sp-03c27dd523c94bd2b8c63228ffff9c23",
                    "base_url": "https://coding.dashscope.aliyuncs.com/v1",
                },
            ],
        },
        "mineru": {
            "token": "eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI2NjcwMDgxMCJ9.test",
            "model": "vlm",
            "is_ocr": False,
            "enable_formula": True,
            "enable_table": True,
        },
        "dirs": {
            "reports": str(reports_dir),
            "pdfs": "data/pdfs",
            "cache": "data/cache",
            "logs": "data/logs",
        },
        "api": {"s2_rate_limit": 100, "cache_ttl": 86400},
        "queue": {"weak_threshold": 0.7, "auto_accept": 0.9},
        "bm25": {"k1": 1.5, "b": 0.75},
    }


@pytest.mark.integration
@pytest.mark.parametrize("pdf_path", TEST_PDFS)
def test_ingest_real_pdf(pdf_path: str):
    """Ingest a real PDF end-to-end: parse → extract → store → report."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        reports_dir = Path(td) / "reports"
        reports_dir.mkdir()
        papers_dir = Path(td) / "papers"

        cfg = _minimal_cfg(str(db_path), str(reports_dir))

        with mock.patch("drbrain.cli.commands.load_config", return_value=cfg):
            # Override papers_dir to our temp location
            with mock.patch(
                "drbrain.cli.commands.save_raw_md",
                wraps=_make_temp_save_raw_md(papers_dir),
            ):
                result = runner.invoke(app, ["ingest", pdf_path])

        assert result.exit_code == 0, f"Ingest failed for {pdf_path}: {result.output}"

        # Verify report was generated
        report_files = list(reports_dir.glob("*.json"))
        assert len(report_files) == 1, "Expected exactly one report file"

        with open(report_files[0]) as f:
            report = json.load(f)

        assert report["paper"]["status"] == "uploaded"
        assert len(report["concepts"].get("problems", [])) > 0 or \
               len(report["concepts"].get("methods", [])) > 0, \
               "Expected at least some concepts"

        # Verify raw MD was saved
        md_files = list(papers_dir.glob("*.md"))
        assert len(md_files) == 1, "Expected saved markdown file"


def _make_temp_save_raw_md(papers_dir: Path):
    """Return a save_raw_md wrapper that uses a temp papers_dir."""
    from drbrain.cli.commands import save_raw_md as _original

    def wrapper(raw_md, local_id, _papers_dir=None, images_src=None):
        return _original(raw_md, local_id, papers_dir, images_src)

    return wrapper
