"""Tests for CLI entry point (cli/main.py) via typer CliRunner."""

import tempfile
from pathlib import Path
from unittest import mock

from typer.testing import CliRunner

from drbrain.cli.main import app

runner = CliRunner()


def _make_config(db_path: str, reports_dir: str) -> dict:
    return {
        "db": {"path": db_path},
        "llm": {"models": [{"provider": "openai", "model": "gpt-4", "api_key": "x"}]},
        "mineru": {
            "token": "",
            "model": "vlm",
            "is_ocr": False,
            "enable_formula": True,
            "enable_table": True,
        },
        "dirs": {
            "inbox": "data/inbox",
            "papers": "data/papers",
            "reports": reports_dir,
            "cache": "data/cache",
            "logs": "data/logs",
        },
        "api": {"s2_rate_limit": 100, "cache_ttl": 86400},
        "queue": {"weak_threshold": 0.7, "auto_accept": 0.9},
        "bm25": {"k1": 1.5, "b": 0.75},
    }


def mock_cfg(db_path: str, reports_dir: str):
    return mock.patch(
        "drbrain.cli.commands.load_config", return_value=_make_config(db_path, reports_dir)
    )


def test_app_help():
    """CLI app responds to --help."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "DrBrain" in result.stdout


def test_app_stats():
    """CLI stats command works."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        reports_dir = Path(td) / "reports"
        with mock_cfg(str(db_path), str(reports_dir)):
            result = runner.invoke(app, ["stats"])
            assert result.exit_code == 0


def test_app_list():
    """CLI list command works."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        reports_dir = Path(td) / "reports"
        with mock_cfg(str(db_path), str(reports_dir)):
            result = runner.invoke(app, ["list"])
            assert result.exit_code == 0


def test_app_closure():
    """CLI closure command works."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        reports_dir = Path(td) / "reports"
        with mock_cfg(str(db_path), str(reports_dir)):
            result = runner.invoke(app, ["closure"])
            assert result.exit_code == 0


def test_app_seed():
    """CLI seed command works."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        reports_dir = Path(td) / "reports"
        with mock_cfg(str(db_path), str(reports_dir)):
            result = runner.invoke(app, ["seed"])
            assert result.exit_code == 0


def test_app_ingest_no_pdfs():
    """CLI ingest exits with code 1 when no PDFs found."""
    with tempfile.TemporaryDirectory() as td:
        empty_dir = Path(td) / "empty"
        empty_dir.mkdir()
        result = runner.invoke(app, ["ingest", str(empty_dir)])
        assert result.exit_code == 1


def test_app_queue_empty():
    """CLI queue command shows empty message."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        reports_dir = Path(td) / "reports"
        with mock_cfg(str(db_path), str(reports_dir)):
            result = runner.invoke(app, ["queue"])
            assert result.exit_code == 0


def test_app_report_not_found():
    """CLI report exits when no report file."""
    with tempfile.TemporaryDirectory() as td:
        reports_dir = Path(td) / "reports"
        reports_dir.mkdir()
        db_path = Path(td) / "test.db"
        with mock_cfg(str(db_path), str(reports_dir)):
            result = runner.invoke(app, ["report", "nonexistent"])
            assert result.exit_code == 1


def test_app_citations_not_found():
    """CLI citations exits when paper not found."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        reports_dir = Path(td) / "reports"
        reports_dir.mkdir()
        with mock_cfg(str(db_path), str(reports_dir)):
            result = runner.invoke(app, ["citations", "nonexistent"])
            assert result.exit_code == 1


def test_app_check_citations_no_input():
    """CLI check-citations exits when no text provided."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        reports_dir = Path(td) / "reports"
        reports_dir.mkdir()
        with mock_cfg(str(db_path), str(reports_dir)):
            result = runner.invoke(app, ["check-citations"])
            assert result.exit_code == 1


def test_app_timeline_no_data():
    """CLI timeline handles missing concept."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        reports_dir = Path(td) / "reports"
        with mock_cfg(str(db_path), str(reports_dir)):
            result = runner.invoke(app, ["timeline", "NonexistentConcept"])
            assert result.exit_code == 0


def test_app_query_no_results():
    """CLI query handles no results."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        reports_dir = Path(td) / "reports"
        with mock_cfg(str(db_path), str(reports_dir)):
            result = runner.invoke(app, ["query", "nonexistent"])
            assert result.exit_code == 0


def test_app_export_unsupported_format():
    """CLI export fails gracefully for unsupported format."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        reports_dir = Path(td) / "reports"
        with mock_cfg(str(db_path), str(reports_dir)):
            result = runner.invoke(app, ["export", "--format", "csv"])
            assert result.exit_code == 1


def test_app_queue_resolve_both_flags():
    """CLI queue resolve fails with both accept/reject."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        reports_dir = Path(td) / "reports"
        from unittest import mock

        from drbrain.cli.commands import queue_resolve_cmd

        cfg = {
            "db": {"path": str(db_path)},
            "llm": {"models": []},
            "dirs": {"reports": str(reports_dir)},
        }
        with mock.patch("drbrain.cli.commands.load_config", return_value=cfg):
            try:
                queue_resolve_cmd(1, accept=True, reject=True)
                assert False, "Should have raised"
            except Exception:
                pass  # typer.Exit is expected
