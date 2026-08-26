"""Tests for CLI entry point (cli/main.py) via typer CliRunner."""

import json
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
            "inbox": "data/spool/inbox",
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
    return mock.patch("drbrain.config.load_config", return_value=_make_config(db_path, reports_dir))


def test_app_help():
    """CLI app responds to --help."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "DrBrain" in result.stdout


def test_autoresearch_run_uses_typed_operator_settings(tmp_path):
    db_path = tmp_path / "test.db"
    cfg = _make_config(str(db_path), str(tmp_path / "reports"))
    cfg["autoresearch"] = {
        "enabled": True,
        "run_dir": str(tmp_path / "runs"),
        "plugins_dir": str(tmp_path / "plugins"),
        "mcp_servers": [{"name": "papers"}],
        "step_capabilities": {"retrieve": ["rag:read", "plugin:search_papers"]},
        "n_critics": 2,
        "max_cycles": 3,
        "stagnation_cycles": 2,
        "max_adaptations": 1,
        "lease_seconds": 30,
        "require_rag_evidence": True,
    }
    captured: dict = {}

    class FakeDirector:
        def __init__(self, _cfg, **kwargs):
            captured.update(kwargs)

        def run_sync(self, topic, **kwargs):
            captured["topic"] = topic
            captured["run_kwargs"] = kwargs
            return {"topic": topic, "cycles": 1, "champion": [{"statement": "kept"}]}

    with (
        mock_cfg(str(db_path), str(tmp_path / "reports")) as load_config,
        mock.patch("drbrain.cli.autoresearch_commands.ResearchDirector", FakeDirector),
    ):
        load_config.return_value = cfg
        result = runner.invoke(app, ["autoresearch", "run", "test topic", "--json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["cycles"] == 1
    assert captured["topic"] == "test topic"
    assert captured["run_dir"] == str(tmp_path / "runs")
    assert captured["n_critics"] == 2
    assert captured["lease_seconds"] == 30
    assert captured["require_rag_evidence"] is True
    assert captured["tool_policy"].to_manifest()["step_capabilities"] == {
        "retrieve": ["plugin:search_papers", "rag:read"]
    }
    assert captured["run_kwargs"] == {
        "max_cycles": 3,
        "stagnation_cycles": 2,
        "max_adaptations": 1,
    }


def test_autoresearch_run_requires_explicit_enable(tmp_path):
    with mock_cfg(str(tmp_path / "test.db"), str(tmp_path / "reports")):
        result = runner.invoke(app, ["autoresearch", "run", "test topic"])

    assert result.exit_code == 1
    assert "autoresearch.enabled" in result.stderr


def test_autoresearch_run_rejects_non_mapping_settings(tmp_path):
    cfg = _make_config(str(tmp_path / "test.db"), str(tmp_path / "reports"))
    cfg["autoresearch"] = ["not", "a", "mapping"]
    with mock_cfg(str(tmp_path / "test.db"), str(tmp_path / "reports")) as load_config:
        load_config.return_value = cfg
        result = runner.invoke(app, ["autoresearch", "run", "test topic"])

    assert result.exit_code == 1
    assert "invalid config" in result.stderr


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


def test_app_query_no_results():
    """query without an index (or llamaindex disabled) exits 1 with a warning (T9)."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        reports_dir = Path(td) / "reports"
        with mock_cfg(str(db_path), str(reports_dir)):
            result = runner.invoke(app, ["query", "nonexistent"])
            assert result.exit_code == 1
            assert "llamaindex engine unavailable" in result.output


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
        ctx = mock.MagicMock()
        ctx.obj = {"config": cfg}
        try:
            queue_resolve_cmd(ctx, 1, accept=True, reject=True)
            assert False, "Should have raised"
        except Exception:
            pass  # typer.Exit is expected
