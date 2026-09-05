"""Regression tests for explicit runtime/config selector semantics."""

from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from drbrain.cli.export_commands import _runtime_data_path as export_runtime_data_path
from drbrain.cli.ingest_commands import _pipeline_runtime_paths
from drbrain.cli.ingest_commands import _runtime_data_path as ingest_runtime_data_path
from drbrain.cli.main import app


def _bare_context() -> SimpleNamespace:
    return SimpleNamespace(obj={})


@pytest.mark.parametrize("resolver", [ingest_runtime_data_path, export_runtime_data_path])
def test_data_path_helpers_reject_empty_primary_root(resolver, tmp_path, monkeypatch):
    """Direct ingest/export callers must not fall through an empty selector."""
    monkeypatch.setenv("DRBRAIN_ROOT", "")
    monkeypatch.setenv("DRBRAIN_RUNTIME_ROOT", str(tmp_path))

    with pytest.raises(ValueError, match="DRBRAIN_ROOT.*empty"):
        resolver(_bare_context(), tmp_path / "data", label="test path")


def test_pipeline_runtime_paths_preserve_empty_primary_selectors(monkeypatch, tmp_path):
    """Pipeline child selection must keep an empty primary value visible."""
    monkeypatch.setenv("DRBRAIN_ROOT", "")
    monkeypatch.setenv("DRBRAIN_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setenv("DRBRAIN_CONFIG", "")
    monkeypatch.setenv("DRBRAIN_CONFIG_PATH", "config.alt.yaml")

    config_path, root = _pipeline_runtime_paths(_bare_context())

    assert config_path == ""
    assert root == ""


def test_cli_rejects_empty_primary_config_selector(tmp_path, monkeypatch):
    """An empty DRBRAIN_CONFIG must not fall through CONFIG_PATH."""
    (tmp_path / "config.yaml").write_text("db:\n  path: data/base.db\n", encoding="utf-8")
    (tmp_path / "config.alt.yaml").write_text("db:\n  path: data/overlay.db\n", encoding="utf-8")
    monkeypatch.setenv("DRBRAIN_CONFIG", "")
    monkeypatch.setenv("DRBRAIN_CONFIG_PATH", "config.alt.yaml")

    result = CliRunner().invoke(app, ["--root", str(tmp_path), "stats"])

    assert result.exit_code == 2
    assert "Config overlay" in result.stderr or "must not be empty" in result.stderr
    assert not (tmp_path / "data" / "base.db").exists()
    assert not (tmp_path / "data" / "overlay.db").exists()


def test_cli_rejects_explicit_empty_config_option(tmp_path):
    """An explicitly empty --config value is not the same as omitting it."""
    (tmp_path / "config.yaml").write_text("db:\n  path: data/base.db\n", encoding="utf-8")

    result = CliRunner().invoke(app, ["--root", str(tmp_path), "--config", "", "stats"])

    assert result.exit_code == 2
    assert "Config overlay" in result.stderr or "must not be empty" in result.stderr
    assert not (tmp_path / "data" / "base.db").exists()


def test_cli_rejects_explicit_empty_root_option(tmp_path):
    """An explicitly empty --root value must not become the checkout CWD."""
    (tmp_path / "config.yaml").write_text("db:\n  path: data/base.db\n", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        ["--root", "", "--config", str(tmp_path / "config.yaml"), "stats"],
    )

    assert result.exit_code == 2
    assert "Runtime root" in result.stderr or "must not be empty" in result.stderr
