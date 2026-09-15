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
    def test_accepts_pdf_markdown_and_latex_inputs(self):
        from drbrain.cli.ingest_commands import ingest_cmd

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = [root / "paper.pdf", root / "notes.md", root / "source.tex"]
            for path in paths:
                path.write_text("# sample", encoding="utf-8")
            app = _make_app(ingest_cmd, _cfg(dirs={"inbox": td, "papers": td}))
            fake_db = type("DB", (), {"conn": type("C", (), {"rollback": lambda self: None})()})()
            with (
                patch("drbrain.cli.ingest_commands.open_db") as db_patch,
                patch("drbrain.cli.ingest_commands.DedupEngine"),
                patch(
                    "drbrain.cli.ingest_commands._ingest_single_paper",
                    side_effect=lambda p, *a, **k: {"ok": True, "local_id": p.stem, "report": {}},
                ) as ingest,
            ):
                db_patch.return_value.__enter__.return_value = fake_db
                db_patch.return_value.__exit__.return_value = False
                r = runner.invoke(app, ["test", *map(str, paths), "--json"])
            assert r.exit_code == 0
            assert ingest.call_count == 3

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


class TestIngestLinkCanonical:
    """T22: URL ingest registers canonical content and writes no raw.md."""

    _URL = "https://example.org/paper"
    _EXTRACTED = {
        "url": _URL,
        "title": "Web Paper",
        "text": "Intro paragraph.\n\n## Method\n\nDetails from the page.",
        "error": "",
    }

    def _config(self, tmp_path: Path) -> dict:
        return {
            "db": {"path": str(tmp_path / "db.sqlite")},
            "dirs": {"papers": str(tmp_path / "papers")},
        }

    def _invoke(self, tmp_path: Path, *, canonical=None):
        import contextlib

        from drbrain.cli.ingest_commands import ingest_link_cmd

        app = _make_app(ingest_link_cmd, self._config(tmp_path))
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                patch("drbrain.providers.webtools.check_webextract_service", return_value=True)
            )
            stack.enter_context(
                patch("drbrain.providers.webtools.extract_web", return_value=dict(self._EXTRACTED))
            )
            if canonical is not None:
                stack.enter_context(
                    patch("drbrain.services.canonical_content.write_canonical_content", **canonical)
                )
            return runner.invoke(app, ["test", self._URL, "--json"])

    def test_url_ingest_registers_canonical_without_raw_md(self, tmp_path):
        import json

        from drbrain.storage.content import read_text
        from drbrain.storage.database import Database

        r = self._invoke(tmp_path)
        assert r.exit_code == 0, r.output
        payload = json.loads(r.output)
        assert len(payload) == 1 and payload[0]["status"] == "ok", payload
        local_id = payload[0]["local_id"]

        # No per-paper file is materialized for a URL ingest.
        papers_root = tmp_path / "papers"
        if papers_root.exists():
            assert not list(papers_root.rglob("*")), "URL ingest must not write per-paper files"

        db = Database(str(tmp_path / "db.sqlite"))
        try:
            revision = db.get_document_revision(local_id)
            assert revision is not None and revision["state"] == "ready"
            assert revision["media_type"] == "md"
            body = read_text(db, local_id)
            assert "# Web Paper" in body and "Details from the page." in body
            raw = db.get_paper_artifact(local_id, "raw")
            assert raw["status"] == "ready"
            metadata = json.loads(raw["metadata_json"])
            assert metadata["source"] == "url" and metadata["revision"] == 1
            assert db.get_paper_artifact(local_id, "tree")["status"] == "skipped"
            assert db.get_paper(local_id)["status"] == "uploaded"
        finally:
            db.close()

    def test_canonical_failure_fails_the_url_and_leaves_no_paper(self, tmp_path):
        import json

        from drbrain.storage.database import Database

        r = self._invoke(
            tmp_path,
            canonical={"side_effect": RuntimeError("canonical boom")},
        )
        assert r.exit_code == 1
        # The JSON report follows the stderr warning; parse from the first
        # JSON bracket.
        output = r.output
        payload = json.loads(output[output.index("[") :])
        assert payload[0]["status"] == "error"
        assert "canonical" in payload[0]["error"]

        db = Database(str(tmp_path / "db.sqlite"))
        try:
            assert db.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 0
            assert db.conn.execute("SELECT COUNT(*) FROM document_revisions").fetchone()[0] == 0
        finally:
            db.close()
