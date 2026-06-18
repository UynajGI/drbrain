"""CLI tests for ingest/fetch/batch-fetch/citations commands."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import typer
from typer.testing import CliRunner


def _cfg(**ov):
    c = {
        "db": {"path": "/tmp/fake.db"},
        "dirs": {"papers": "/tmp/p", "inbox": "/tmp/inbox", "cache": "/tmp/cache"},
        "llm": {"models": []},
        "fetch": {},
        "api": {},
    }
    c.update(ov)
    return c


def _make_app(cmd_fn, config):
    app = typer.Typer()

    @app.callback()
    def cb(ctx: typer.Context):
        ctx.obj = {"config": config}

    app.command("test")(cmd_fn)
    return app


runner = CliRunner()


class TestIngestCmd:
    def test_nonexistent_file(self):
        from drbrain.cli.ingest_commands import ingest_cmd

        app = _make_app(ingest_cmd, _cfg())
        r = runner.invoke(app, ["test", "/nonexistent.pdf"])
        assert r.exit_code == 1

    def test_json_empty_inbox(self):
        from drbrain.cli.ingest_commands import ingest_cmd

        with tempfile.TemporaryDirectory() as td:
            app = _make_app(ingest_cmd, _cfg(dirs={"inbox": td, "papers": td}))
            r = runner.invoke(app, ["test", "--json"])
            assert r.exit_code == 1

    def test_empty_inbox_text(self):
        from drbrain.cli.ingest_commands import ingest_cmd

        with tempfile.TemporaryDirectory() as td:
            app = _make_app(ingest_cmd, _cfg(dirs={"inbox": td, "papers": td}))
            r = runner.invoke(app, ["test"])
            assert r.exit_code == 1


class TestFetchCmd:
    def test_fetch_no_result(self):
        from drbrain.cli.ingest_commands import fetch_cmd

        app = _make_app(fetch_cmd, _cfg())
        with patch("drbrain.cli.ingest_commands.fetch_paper", return_value=None):
            r = runner.invoke(app, ["test", "10.1/xx"])
            assert r.exit_code == 1
            assert "Could not find" in r.output

    def test_fetch_arxiv_success(self):
        from drbrain.cli.ingest_commands import fetch_cmd

        app = _make_app(fetch_cmd, _cfg(llm={"models": [{"provider": "x", "model": "y"}]}))
        mock_result = {"pdf_path": "/tmp/test.pdf", "title": "T", "year": 2024, "local_id": "p1"}
        mock_ingest = {"ok": True, "local_id": "p1", "error": None}
        with (
            patch("drbrain.cli.ingest_commands.fetch_paper", return_value=mock_result),
            patch("drbrain.cli.ingest_commands._ingest_single_paper", return_value=mock_ingest),
        ):
            r = runner.invoke(app, ["test", "2401.00001", "--arxiv"])
            assert r.exit_code == 0
            assert "Ingested" in r.output

    def test_fetch_ingest_fail(self):
        from drbrain.cli.ingest_commands import fetch_cmd

        app = _make_app(fetch_cmd, _cfg())
        mock_result = {"pdf_path": "/tmp/test.pdf", "title": "T", "year": 2024}
        mock_ingest = {"ok": False, "error": "parse failed", "local_id": None}
        with (
            patch("drbrain.cli.ingest_commands.fetch_paper", return_value=mock_result),
            patch("drbrain.cli.ingest_commands._ingest_single_paper", return_value=mock_ingest),
        ):
            r = runner.invoke(app, ["test", "10.1/xx"])
            assert r.exit_code == 1


class TestBatchFetch:
    def test_input_not_found(self):
        from drbrain.cli.ingest_commands import batch_fetch_cmd

        app = _make_app(batch_fetch_cmd, _cfg())
        r = runner.invoke(app, ["test", "/no/list.txt"])
        assert r.exit_code == 1

    def test_no_entries(self):
        from drbrain.cli.ingest_commands import batch_fetch_cmd

        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "list.txt"
            f.write_text("# comment\n\n")
            app = _make_app(batch_fetch_cmd, _cfg())
            r = runner.invoke(app, ["test", str(f)])
            assert r.exit_code == 1

    def test_skips_comments_and_processes_entries(self):
        from drbrain.cli.ingest_commands import batch_fetch_cmd

        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "list.txt"
            f.write_text("# comment\n\n10.1/x\n")
            app = _make_app(batch_fetch_cmd, _cfg())
            with patch("drbrain.cli.ingest_commands.resolve_pdf_url", return_value=None):
                r = runner.invoke(app, ["test", str(f), "--delay", "0"])
                assert r.exit_code == 0
                assert "Batch fetch" in r.output


class TestCitationsCmd:
    def test_invalid_type(self):
        from drbrain.cli.ingest_commands import citations_cmd

        app = _make_app(citations_cmd, _cfg())
        r = runner.invoke(app, ["test", "p1", "--type", "bogus"])
        assert r.exit_code == 1
