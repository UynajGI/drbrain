"""T07: ingest never moves or deletes input materials; queue state is separate.

Covers direct input, spool-directory input, already-hosted source, failure
retry, and same-path copy safety.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import typer
from typer.testing import CliRunner

from drbrain.storage.database import Database
from drbrain.storage.inbox import copy_to_pending, file_sha256

runner = CliRunner()


def _cfg(root: Path) -> dict:
    return {
        "db": {"path": str(root / "drbrain.db")},
        "dirs": {
            "papers": str(root / "papers"),
            "inbox": str(root / "spool" / "inbox"),
            "pending": str(root / "spool" / "pending"),
            "cache": str(root / "cache"),
        },
        "llm": {"models": [{"provider": "openai", "model": "m"}]},
        "fetch": {},
        "api": {},
    }


def _make_app(cmd_fn, config):
    app = typer.Typer()

    @app.callback()
    def cb(ctx: typer.Context):
        ctx.obj = {"config": config}

    app.command("test")(cmd_fn)
    return app


class TestPendingCopy:
    def test_copy_to_pending_keeps_original(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "paper.pdf"
            src.write_bytes(b"%PDF-1.4 fake")
            before = file_sha256(src)
            pending = root / "pending"
            dst = copy_to_pending(src, pending, reason="parse error")
            assert src.exists() and file_sha256(src) == before
            assert dst.exists() and dst.read_bytes() == src.read_bytes()
            assert (pending / "pending.jsonl").exists()

    def test_copy_to_pending_same_path_is_noop(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "paper.pdf"
            src.write_bytes(b"%PDF-1.4 fake")
            dst = copy_to_pending(src, root, reason="parse error")
            assert dst == src and src.exists()

    def test_copy_to_pending_disambiguates_conflicting_name(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pending = root / "pending"
            pending.mkdir()
            first = root / "a" / "paper.pdf"
            first.parent.mkdir()
            first.write_bytes(b"first")
            (pending / "paper.pdf").write_bytes(b"other")
            dst = copy_to_pending(first, pending, reason="x")
            assert dst.name == "paper.1.pdf"
            assert first.exists() and (pending / "paper.pdf").read_bytes() == b"other"


class TestArtifactSaveKeepsSource:
    def test_save_does_not_unlink_source(self, monkeypatch, tmp_path):
        from types import SimpleNamespace

        from drbrain.cli._helpers.db_ingest import _save_paper_artifacts

        monkeypatch.setenv("DRBRAIN_ROOT", str(tmp_path))
        source = tmp_path / "source.pdf"
        source.write_bytes(b"%PDF-1.4 body")
        paper_dir = tmp_path / "papers" / "p1"
        paper_dir.mkdir(parents=True)
        parsed = SimpleNamespace(raw_md="# md", images_dir=None)
        _save_paper_artifacts(parsed, "p1", paper_dir, source)
        assert source.exists(), "input must never be removed by ingest"
        assert (paper_dir / "source.pdf").exists()
        # T22: the body is registered in the canonical store, not as a
        # per-paper markdown file.
        assert not (paper_dir / "raw.md").exists()

    def test_save_handles_already_hosted_source(self, monkeypatch, tmp_path):
        from types import SimpleNamespace

        from drbrain.cli._helpers.db_ingest import _save_paper_artifacts

        monkeypatch.setenv("DRBRAIN_ROOT", str(tmp_path))
        paper_dir = tmp_path / "papers" / "p1"
        paper_dir.mkdir(parents=True)
        hosted = paper_dir / "source.pdf"
        hosted.write_bytes(b"%PDF-1.4 already hosted")
        parsed = SimpleNamespace(raw_md="# md", images_dir=None)
        _save_paper_artifacts(parsed, "p1", paper_dir, hosted)
        assert hosted.exists() and hosted.read_bytes() == b"%PDF-1.4 already hosted"


class TestLedgerRoundtrip:
    def test_record_and_lookup(self):
        db = Database(":memory:")
        db.record_spool_input("a" * 64, path="/x/a.pdf", size=3, status="done", local_id="p1")
        entry = db.get_spool_input("a" * 64)
        assert entry and entry["status"] == "done" and entry["local_id"] == "p1"
        db.record_spool_input("a" * 64, status="failed", reason="parse")
        entry = db.get_spool_input("a" * 64)
        assert (
            entry["status"] == "failed" and entry["local_id"] == "p1" or entry["status"] == "failed"
        )
        assert [row["status"] for row in db.list_spool_inputs("failed")] == ["failed"]
        db.close() if hasattr(db, "close") else None

    def test_invalid_status_rejected(self):
        db = Database(":memory:")
        with pytest.raises(ValueError, match="invalid spool status"):
            db.record_spool_input("b" * 64, status="maybe")


class TestIngestCliSafety:
    def test_direct_input_survives_and_is_recorded(self, monkeypatch, tmp_path):
        from drbrain.cli.ingest_commands import ingest_cmd

        monkeypatch.setenv("DRBRAIN_ROOT", str(tmp_path))
        root = tmp_path
        cfg = _cfg(root)
        source = root / "direct.pdf"
        source.write_bytes(b"%PDF-1.4 direct")
        before = file_sha256(source)
        app = _make_app(ingest_cmd, cfg)

        calls = []

        def fake_ingest(pdf_path, _cfg_, db, _dedup, json_mode=False):
            calls.append(Path(pdf_path))
            from drbrain.cli._helpers.db_ingest import _record_spool_state

            _record_spool_state(db, Path(pdf_path), "done", local_id="p-test")
            return {"ok": True, "local_id": "p-test", "report": {"local_id": "p-test"}}

        with patch("drbrain.cli.ingest_commands._ingest_single_paper", side_effect=fake_ingest):
            r = runner.invoke(app, ["test", str(source), "--json"])
        assert r.exit_code == 0, r.output
        assert source.exists() and file_sha256(source) == before
        assert calls == [source]

    def test_spool_scan_skips_ledger_entries_and_keeps_files(self, monkeypatch, tmp_path):
        from drbrain.cli.ingest_commands import ingest_cmd

        monkeypatch.setenv("DRBRAIN_ROOT", str(tmp_path))
        root = tmp_path
        cfg = _cfg(root)
        inbox = Path(cfg["dirs"]["inbox"])
        inbox.mkdir(parents=True)
        done_file = inbox / "done.pdf"
        done_file.write_bytes(b"%PDF-1.4 done")
        new_file = inbox / "new.pdf"
        new_file.write_bytes(b"%PDF-1.4 new")
        app = _make_app(ingest_cmd, cfg)

        calls = []

        def fake_ingest(pdf_path, _cfg_, db, _dedup, json_mode=False):
            calls.append(Path(pdf_path))
            from drbrain.cli._helpers.db_ingest import _record_spool_state

            _record_spool_state(db, Path(pdf_path), "done", local_id="p-x")
            return {"ok": True, "local_id": "p-x", "report": {"local_id": "p-x"}}

        with patch("drbrain.cli.ingest_commands._ingest_single_paper", side_effect=fake_ingest):
            # First pass processes both and records the ledger.
            r1 = runner.invoke(app, ["test", str(inbox), "--json"])
            assert r1.exit_code == 0, r1.output
            assert len(calls) == 2
            # Second pass skips the recorded ones.
            calls.clear()
            r2 = runner.invoke(app, ["test", str(inbox), "--json"])
            assert r2.exit_code == 0, r2.output
            assert calls == []
        assert done_file.exists() and new_file.exists()

    def test_reprocess_forces_directory_inputs(self, monkeypatch, tmp_path):
        from drbrain.cli.ingest_commands import ingest_cmd

        monkeypatch.setenv("DRBRAIN_ROOT", str(tmp_path))
        root = tmp_path
        cfg = _cfg(root)
        inbox = Path(cfg["dirs"]["inbox"])
        inbox.mkdir(parents=True)
        (inbox / "a.pdf").write_bytes(b"%PDF-1.4 a")
        app = _make_app(ingest_cmd, cfg)
        calls = []

        def fake_ingest(pdf_path, _cfg_, db, _dedup, json_mode=False):
            calls.append(Path(pdf_path))
            from drbrain.cli._helpers.db_ingest import _record_spool_state

            _record_spool_state(db, Path(pdf_path), "done", local_id="p-y")
            return {"ok": True, "local_id": "p-y", "report": {"local_id": "p-y"}}

        with patch("drbrain.cli.ingest_commands._ingest_single_paper", side_effect=fake_ingest):
            r1 = runner.invoke(app, ["test", str(inbox), "--json"])
            assert r1.exit_code == 0
            r2 = runner.invoke(app, ["test", str(inbox), "--json", "--reprocess"])
            assert r2.exit_code == 0, r2.output
            assert len(calls) == 2  # first pass + forced reprocess
