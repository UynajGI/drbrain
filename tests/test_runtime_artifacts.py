"""Fail-closed runtime selectors for non-CLI artifact helpers."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    ("module_name", "helper_name"),
    [
        ("drbrain.report.generator", "_default_reports_dir"),
        ("drbrain.storage.workspace", "_default_workspace_root"),
        ("drbrain.storage.backup", "_default_backup_dir"),
    ],
)
def test_default_artifact_roots_reject_empty_runtime_selector(
    monkeypatch, module_name: str, helper_name: str
):
    """An explicit empty root must never silently select the process CWD."""
    module = __import__(module_name, fromlist=[helper_name])
    monkeypatch.setenv("DRBRAIN_ROOT", "")
    with pytest.raises(ValueError, match="DRBRAIN_ROOT.*empty"):
        getattr(module, helper_name)()


def test_fetch_rejects_empty_runtime_selector(monkeypatch, tmp_path):
    """Fetch must fail before choosing a papers directory when root is blank."""
    from unittest.mock import patch

    from drbrain.services.fetch import fetch_paper

    monkeypatch.setenv("DRBRAIN_ROOT", "")
    with (
        patch("drbrain.services.fetch.resolve_pdf_url", return_value="https://x/p.pdf"),
        patch(
            "drbrain.services.fetch._resolve_metadata",
            return_value={"local_id": "p1", "title": "T", "year": 2020},
        ),
    ):
        with pytest.raises(ValueError, match="(DRBRAIN_ROOT.*empty|Runtime root.*empty)"):
            fetch_paper(doi="10.1/x", fetch_config={"papers_root": str(tmp_path)})


def test_fetch_rejects_external_root_before_network_io(monkeypatch, tmp_path):
    """An invalid destination must fail before provider or metadata probes."""
    from unittest.mock import patch

    from drbrain.services.fetch import fetch_paper

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    outside = tmp_path / "outside"
    monkeypatch.setenv("DRBRAIN_ROOT", str(runtime))
    with (
        patch("drbrain.services.fetch.resolve_pdf_url", return_value="https://x/p.pdf") as resolve,
        patch("drbrain.services.fetch._resolve_metadata") as metadata,
    ):
        with pytest.raises(ValueError, match="fetch papers_root.*escapes runtime root"):
            fetch_paper(doi="10.1/x", fetch_config={"papers_root": str(outside)})
    resolve.assert_not_called()
    metadata.assert_not_called()


def test_download_rejects_external_directory_before_creating_it(monkeypatch, tmp_path):
    """The low-level downloader must not create an out-of-runtime directory."""
    from unittest.mock import patch

    from drbrain.services.fetch import download_pdf

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    outside = tmp_path / "outside" / "paper"
    monkeypatch.setenv("DRBRAIN_ROOT", str(runtime))
    with patch("drbrain.services.fetch.requests.get") as request:
        with pytest.raises(ValueError, match="download paper directory.*escapes runtime root"):
            download_pdf("https://x/p.pdf", outside)
    request.assert_not_called()
    assert not outside.exists()


def test_remaining_runtime_helpers_reject_empty_selector(monkeypatch, tmp_path):
    """Secondary services must share RuntimeContext's empty-selector policy."""
    from drbrain.app.service import _runtime_path as app_runtime_path
    from drbrain.cli.check_commands import _runtime_path as check_runtime_path
    from drbrain.cli.setup import _active_runtime
    from drbrain.extractor.llm_client import _llm_trace_path
    from drbrain.rag.sql_retrie import _default_rag_db
    from drbrain.services.citation_styles import default_styles_dir
    from drbrain.storage.proceedings import default_path

    monkeypatch.setenv("DRBRAIN_ROOT", "")
    checks = (
        lambda: _default_rag_db({"db": {"path": str(tmp_path / "library.db")}}),
        default_path,
        lambda: app_runtime_path("data/run", label="run directory"),
        _llm_trace_path,
        default_styles_dir,
        lambda: check_runtime_path(None, "data/run", label="run directory"),
        _active_runtime,
    )
    for check in checks:
        with pytest.raises(ValueError, match="(DRBRAIN_ROOT.*empty|Runtime root.*empty)"):
            check()
