"""Tests for batch-fetch command."""

import tempfile
from pathlib import Path
from unittest import mock

import typer

from drbrain.cli.ingest_commands import batch_fetch_cmd


def _make_minimal_config(db_path: str) -> dict:
    """Return config dict for testing."""
    return {
        "db": {"path": db_path},
        "dirs": {
            "inbox": "data/spool/inbox",
            "papers": "data/papers",
            "reports": "/tmp/test_reports",
            "cache": "data/cache",
            "logs": "data/logs",
        },
        "fetch": {},
    }


def _make_ctx(cfg: dict):
    """Create a minimal typer.Context mock with config pre-loaded."""
    ctx = mock.MagicMock(spec=typer.Context)
    ctx.obj = {"config": cfg}
    return ctx


def _make_mock_db(return_value_for_get=None):
    """Create a mock db whose __enter__ returns a properly configured mock."""
    db_mock = mock.MagicMock()
    db_mock.get_paper_by_external_id.return_value = return_value_for_get
    # Make the context manager return our db_mock directly
    cm_mock = mock.MagicMock()
    cm_mock.__enter__ = mock.MagicMock(return_value=db_mock)
    cm_mock.__exit__ = mock.MagicMock(return_value=False)
    return cm_mock, db_mock


class TestFileParsing:
    """Test input file parsing: skip comments, blanks."""

    def test_skip_comments_and_blanks(self):
        """Comments (#) and blank lines are skipped."""
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "test.db")
            output_dir = Path(td) / "inbox"
            output_dir.mkdir()

            input_file = Path(td) / "dois.txt"
            input_file.write_text(
                "# This is a comment\n\n10.1234/one\n  \n10.1234/two\n# Another comment\n\n"
            )

            cfg = _make_minimal_config(db_path)
            ctx = _make_ctx(cfg)

            resolve_mock = mock.MagicMock(return_value="https://example.com/paper.pdf")
            download_mock = mock.MagicMock(return_value=output_dir / "10.1234_one" / "source.pdf")
            cm_mock, db_mock = _make_mock_db(return_value_for_get=None)

            with (
                mock.patch("drbrain.cli.ingest_commands.open_db", return_value=cm_mock),
                mock.patch("drbrain.services.fetch.resolve_pdf_url", resolve_mock),
                mock.patch("drbrain.services.fetch.download_pdf", download_mock),
            ):
                batch_fetch_cmd(ctx, str(input_file), str(output_dir), delay=0.0)

            # Should have resolved exactly 2 DOIs (comments/blanks skipped)
            assert resolve_mock.call_count == 2
            assert resolve_mock.call_args_list[0].kwargs["doi"] == "10.1234/one"
            assert resolve_mock.call_args_list[1].kwargs["doi"] == "10.1234/two"

    def test_empty_file_exits(self):
        """Empty file (only comments/blanks) raises typer.Exit."""
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "test.db")
            input_file = Path(td) / "dois.txt"
            input_file.write_text("# only comments\n\n  \n")

            cfg = _make_minimal_config(db_path)
            ctx = _make_ctx(cfg)

            try:
                batch_fetch_cmd(
                    ctx,
                    str(input_file),
                    str(Path(td) / "inbox"),
                )
                assert False, "Should have raised Exit"
            except typer.Exit as e:
                assert e.exit_code == 1

    def test_missing_file_exits(self):
        """Non-existent input file raises typer.Exit."""
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "test.db")
            cfg = _make_minimal_config(db_path)
            ctx = _make_ctx(cfg)

            try:
                batch_fetch_cmd(
                    ctx,
                    str(Path(td) / "nonexistent.txt"),
                    str(Path(td) / "inbox"),
                )
                assert False, "Should have raised Exit"
            except typer.Exit as e:
                assert e.exit_code == 1


class TestSkipExisting:
    """Test skip-existing logic with mocked DB query."""

    def test_skips_existing_doi(self):
        """DOIs already in DB are skipped when --skip-existing is True."""
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "test.db")
            output_dir = Path(td) / "inbox"
            output_dir.mkdir()

            input_file = Path(td) / "dois.txt"
            input_file.write_text("10.1234/existing\n10.1234/new\n")

            cfg = _make_minimal_config(db_path)
            ctx = _make_ctx(cfg)

            resolve_mock = mock.MagicMock(return_value="https://example.com/paper.pdf")
            download_mock = mock.MagicMock(return_value=output_dir / "source.pdf")

            cm_mock, db_mock = _make_mock_db(return_value_for_get=None)
            # First DOI exists, second does not
            db_mock.get_paper_by_external_id.side_effect = ["paper_001", None]

            with (
                mock.patch("drbrain.cli.ingest_commands.open_db", return_value=cm_mock),
                mock.patch("drbrain.services.fetch.resolve_pdf_url", resolve_mock),
                mock.patch("drbrain.services.fetch.download_pdf", download_mock),
            ):
                batch_fetch_cmd(ctx, str(input_file), str(output_dir), delay=0.0)

            # Only the new DOI should have been resolved/downloaded
            assert resolve_mock.call_count == 1
            assert resolve_mock.call_args.kwargs["doi"] == "10.1234/new"
            assert download_mock.call_count == 1
            # Verify DB was queried for both DOIs
            assert db_mock.get_paper_by_external_id.call_count == 2

    def test_skip_existing_false_fetches_all(self):
        """With skip_existing=False, all DOIs are fetched regardless."""
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "test.db")
            output_dir = Path(td) / "inbox"
            output_dir.mkdir()

            input_file = Path(td) / "dois.txt"
            input_file.write_text("10.1234/one\n10.1234/two\n")

            cfg = _make_minimal_config(db_path)
            ctx = _make_ctx(cfg)

            resolve_mock = mock.MagicMock(return_value="https://example.com/paper.pdf")
            download_mock = mock.MagicMock(return_value=output_dir / "source.pdf")
            cm_mock, db_mock = _make_mock_db(return_value_for_get=None)

            with (
                mock.patch("drbrain.cli.ingest_commands.open_db", return_value=cm_mock),
                mock.patch("drbrain.services.fetch.resolve_pdf_url", resolve_mock),
                mock.patch("drbrain.services.fetch.download_pdf", download_mock),
            ):
                batch_fetch_cmd(
                    ctx,
                    str(input_file),
                    str(output_dir),
                    delay=0.0,
                    skip_existing=False,
                )

            # Both DOIs should be fetched, DB not queried for existence
            assert resolve_mock.call_count == 2
            db_mock.get_paper_by_external_id.assert_not_called()


class TestSummaryOutput:
    """Test that summary output includes correct counts."""

    def test_summary_format_fetched_and_failed(self):
        """Summary shows correct fetched/skipped/failed counts."""
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "test.db")
            output_dir = Path(td) / "inbox"
            output_dir.mkdir()

            input_file = Path(td) / "dois.txt"
            input_file.write_text("10.1234/good\n10.1234/bad\n")

            cfg = _make_minimal_config(db_path)
            ctx = _make_ctx(cfg)

            cm_mock, db_mock = _make_mock_db(return_value_for_get=None)

            # First DOI resolves and downloads; second fails to resolve
            def resolve_side_effect(doi=None, fetch_config=None):
                if "good" in doi:
                    return "https://example.com/good.pdf"
                return None

            resolve_mock = mock.MagicMock(side_effect=resolve_side_effect)
            download_mock = mock.MagicMock(return_value=output_dir / "source.pdf")

            with (
                mock.patch("drbrain.cli.ingest_commands.open_db", return_value=cm_mock),
                mock.patch("drbrain.services.fetch.resolve_pdf_url", resolve_mock),
                mock.patch("drbrain.services.fetch.download_pdf", download_mock),
            ):
                batch_fetch_cmd(ctx, str(input_file), str(output_dir), delay=0.0)

            # Verify resolve was called for both
            assert resolve_mock.call_count == 2
            # Download only called for the one that resolved
            assert download_mock.call_count == 1

    def test_graceful_error_on_bad_doi(self):
        """A bad DOI that raises exception in resolve is handled gracefully."""
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "test.db")
            output_dir = Path(td) / "inbox"
            output_dir.mkdir()

            input_file = Path(td) / "dois.txt"
            input_file.write_text("10.1234/good\n10.1234/throws\n10.1234/also_good\n")

            cfg = _make_minimal_config(db_path)
            ctx = _make_ctx(cfg)

            cm_mock, db_mock = _make_mock_db(return_value_for_get=None)

            def resolve_side_effect(doi=None, fetch_config=None):
                if "throws" in doi:
                    raise ValueError("bad DOI format")
                return "https://example.com/paper.pdf"

            resolve_mock = mock.MagicMock(side_effect=resolve_side_effect)
            download_mock = mock.MagicMock(return_value=output_dir / "source.pdf")

            with (
                mock.patch("drbrain.cli.ingest_commands.open_db", return_value=cm_mock),
                mock.patch("drbrain.services.fetch.resolve_pdf_url", resolve_mock),
                mock.patch("drbrain.services.fetch.download_pdf", download_mock),
            ):
                # Should not raise - error is caught and logged
                batch_fetch_cmd(ctx, str(input_file), str(output_dir), delay=0.0)

            # All 3 resolved, 2 downloaded (the throwing one returns None after catch)
            assert resolve_mock.call_count == 3
            assert download_mock.call_count == 2


class TestDirectPdfUrl:
    """Test handling of direct PDF URL entries."""

    def test_direct_pdf_url_fallback(self):
        """Direct PDF URLs are used when resolve_pdf_url returns None."""
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "test.db")
            output_dir = Path(td) / "inbox"
            output_dir.mkdir()

            input_file = Path(td) / "dois.txt"
            input_file.write_text("https://example.org/paper.pdf\n")

            cfg = _make_minimal_config(db_path)
            ctx = _make_ctx(cfg)

            cm_mock, db_mock = _make_mock_db(return_value_for_get=None)
            resolve_mock = mock.MagicMock(return_value=None)
            download_mock = mock.MagicMock(return_value=output_dir / "source.pdf")

            with (
                mock.patch("drbrain.cli.ingest_commands.open_db", return_value=cm_mock),
                mock.patch("drbrain.services.fetch.resolve_pdf_url", resolve_mock),
                mock.patch("drbrain.services.fetch.download_pdf", download_mock),
            ):
                batch_fetch_cmd(ctx, str(input_file), str(output_dir), delay=0.0)

            # Should still download via the direct URL fallback
            assert download_mock.call_count == 1
            # The URL passed to download should be the original PDF URL
            call_url = download_mock.call_args[0][0]
            assert call_url == "https://example.org/paper.pdf"
