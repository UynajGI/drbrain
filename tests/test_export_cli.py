"""Tests for drbrain.cli.export_commands — export/queue/delete/backup/restore/style/lineage."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

import typer

from drbrain.storage.database import Database


def _make_minimal_config(db_path: str, reports_dir: str = "reports") -> dict:
    return {
        "db": {"path": db_path},
        "llm": {"models": [{"provider": "openai", "model": "gpt-4", "api_key": "x"}]},
        "mineru": {},
        "dirs": {
            "inbox": "data/spool/inbox",
            "papers": "data/papers",
            "reports": reports_dir,
            "cache": "data/cache",
            "logs": "data/logs",
        },
        "api": {},
        "queue": {"weak_threshold": 0.5, "auto_accept": False},
        "bm25": {"k1": 1.5, "b": 0.75},
    }


def _make_ctx(cfg: dict):
    ctx = mock.MagicMock(spec=typer.Context)
    ctx.obj = {"config": cfg}
    return ctx


def _capture_factory(buf: list[str]):
    def _capture(msg="", *args, **kwargs):
        buf.append(str(msg))

    return _capture


def _seed_paper(db_path, local_id="p1", title="Test Paper"):
    db = Database(str(db_path))
    db.insert_paper(local_id, title, 2024, "uploaded")
    db.commit()
    db.close()


# ── export_cmd ────────────────────────────────────────────────────────────


def test_export_bibtex_single_paper():
    """export_cmd outputs BibTeX for a single paper."""
    from drbrain.cli.export_commands import export_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        _seed_paper(db_path)
        ctx = _make_ctx(cfg)

        captured: list[str] = []
        with mock.patch("typer.echo", side_effect=_capture_factory(captured)):
            export_cmd(
                ctx,
                local_id="p1",
                format="bib",
                all=False,
                output=None,
                style="apa",
                json_output=False,
            )

        output = "\n".join(captured)
        assert "@article{" in output
        assert "Test Paper" in output


def test_export_ris_single_paper():
    """export_cmd --format ris outputs RIS."""
    from drbrain.cli.export_commands import export_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        _seed_paper(db_path, title="RIS Paper")
        ctx = _make_ctx(cfg)

        captured: list[str] = []
        with mock.patch("typer.echo", side_effect=_capture_factory(captured)):
            export_cmd(
                ctx,
                local_id="p1",
                format="ris",
                all=False,
                output=None,
                style="apa",
                json_output=False,
            )

        output = "\n".join(captured)
        assert output.startswith("TY  - JOUR")
        assert "RIS Paper" in output


def test_export_markdown_single_paper():
    """export_cmd --format md outputs Markdown."""
    from drbrain.cli.export_commands import export_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        _seed_paper(db_path, title="Markdown Paper")
        ctx = _make_ctx(cfg)

        captured: list[str] = []
        with mock.patch("typer.echo", side_effect=_capture_factory(captured)):
            export_cmd(
                ctx,
                local_id="p1",
                format="md",
                all=False,
                output=None,
                style="apa",
                json_output=False,
            )

        output = "\n".join(captured)
        assert "Markdown Paper" in output


def test_export_json_output_format():
    """export_cmd --json emits a JSON document with format and result keys."""
    from drbrain.cli.export_commands import export_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        _seed_paper(db_path, title="JSON Paper")
        ctx = _make_ctx(cfg)

        captured: list[str] = []
        with mock.patch("typer.echo", side_effect=_capture_factory(captured)):
            export_cmd(
                ctx,
                local_id="p1",
                format="bib",
                all=False,
                output=None,
                style="apa",
                json_output=True,
            )

        payload = json.loads(captured[0])
        assert payload["format"] == "bib"
        assert "JSON Paper" in payload["result"]


def test_export_unknown_format_exits_1():
    """export_cmd rejects an unsupported format with Exit(1)."""
    from drbrain.cli.export_commands import export_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        ctx = _make_ctx(cfg)

        try:
            export_cmd(
                ctx,
                local_id="p1",
                format="csv",
                all=False,
                output=None,
                style="apa",
                json_output=False,
            )
            assert False, "Should have raised Exit"
        except typer.Exit as e:
            assert e.exit_code == 1


def test_export_missing_paper_exits_1():
    """export_cmd exits with code 1 when local_id not in DB."""
    from drbrain.cli.export_commands import export_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        ctx = _make_ctx(cfg)

        try:
            export_cmd(
                ctx,
                local_id="missing",
                format="bib",
                all=False,
                output=None,
                style="apa",
                json_output=False,
            )
            assert False, "Should have raised Exit"
        except typer.Exit as e:
            assert e.exit_code == 1


def test_export_requires_local_id_or_all():
    """export_cmd without local_id and without --all exits with code 1."""
    from drbrain.cli.export_commands import export_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        ctx = _make_ctx(cfg)

        try:
            export_cmd(
                ctx,
                local_id=None,
                format="bib",
                all=False,
                output=None,
                style="apa",
                json_output=False,
            )
            assert False, "Should have raised Exit"
        except typer.Exit as e:
            assert e.exit_code == 1


def test_export_writes_output_file():
    """export_cmd -o writes content to the named file."""
    from drbrain.cli.export_commands import export_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        _seed_paper(db_path, title="ToFile")
        ctx = _make_ctx(cfg)

        out_file = Path(td) / "out.bib"
        with mock.patch("typer.echo") as mock_echo:
            export_cmd(
                ctx,
                local_id="p1",
                format="bib",
                all=False,
                output=str(out_file),
                style="apa",
                json_output=False,
            )
            mock_echo.assert_any_call(f"Exported to {out_file}")

        written = out_file.read_text()
        assert "ToFile" in written


def test_export_all_papers():
    """export_cmd --all batch-exports every paper in the DB."""
    from drbrain.cli.export_commands import export_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))

        db = Database(str(db_path))
        db.insert_paper("p1", "Paper One", 2024, "uploaded")
        db.insert_paper("p2", "Paper Two", 2023, "uploaded")
        db.commit()
        db.close()

        ctx = _make_ctx(cfg)
        captured: list[str] = []
        with mock.patch("typer.echo", side_effect=_capture_factory(captured)):
            export_cmd(
                ctx,
                local_id=None,
                format="bib",
                all=True,
                output=None,
                style="apa",
                json_output=False,
            )

        output = "\n".join(captured)
        assert "Paper One" in output
        assert "Paper Two" in output


# ── queue_cmd ─────────────────────────────────────────────────────────────


def test_queue_cmd_empty_message():
    """queue_cmd prints 'Queue is empty.' when there are no pending items."""
    from drbrain.cli.export_commands import queue_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        ctx = _make_ctx(cfg)

        with mock.patch("typer.echo") as mock_echo:
            queue_cmd(ctx, json_output=False)
            mock_echo.assert_any_call("Queue is empty.")


def test_queue_cmd_json_empty_outputs_list():
    """queue_cmd --json with empty queue emits an empty JSON array."""
    from drbrain.cli.export_commands import queue_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        ctx = _make_ctx(cfg)

        captured: list[str] = []
        with mock.patch("typer.echo", side_effect=_capture_factory(captured)):
            queue_cmd(ctx, json_output=True)

        assert json.loads(captured[0]) == []


# ── queue_resolve_cmd ─────────────────────────────────────────────────────


def test_queue_resolve_both_flags_exits_1():
    """queue_resolve_cmd rejects simultaneous accept and reject flags."""
    from drbrain.cli.export_commands import queue_resolve_cmd

    cfg = _make_minimal_config(":memory:")
    ctx = _make_ctx(cfg)

    try:
        queue_resolve_cmd(ctx, 1, accept=True, reject=True, json_output=False)
        assert False, "Should have raised Exit"
    except typer.Exit as e:
        assert e.exit_code == 1


def test_queue_resolve_neither_flag_exits_1():
    """queue_resolve_cmd rejects when neither accept nor reject is set."""
    from drbrain.cli.export_commands import queue_resolve_cmd

    cfg = _make_minimal_config(":memory:")
    ctx = _make_ctx(cfg)

    try:
        queue_resolve_cmd(ctx, 1, accept=False, reject=False, json_output=False)
        assert False, "Should have raised Exit"
    except typer.Exit as e:
        assert e.exit_code == 1


def test_queue_resolve_accept_invokes_resolver():
    """queue_resolve_cmd --accept calls resolve_accept with the queue id."""
    from drbrain.cli.export_commands import queue_resolve_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        ctx = _make_ctx(cfg)

        with (
            mock.patch("drbrain.extractor.queue.resolve_accept") as mock_accept,
            mock.patch("typer.echo"),
        ):
            queue_resolve_cmd(ctx, 7, accept=True, reject=False, json_output=False)
            mock_accept.assert_called_once()

        assert mock_accept.call_args.args[1] == 7


def test_queue_resolve_all_neither_flag_exits_1():
    """queue_resolve_all_cmd rejects when neither flag is set."""
    from drbrain.cli.export_commands import queue_resolve_all_cmd

    cfg = _make_minimal_config(":memory:")
    ctx = _make_ctx(cfg)

    try:
        queue_resolve_all_cmd(
            ctx,
            accept=False,
            reject=False,
            type_filter=None,
            max_conf=None,
            json_output=False,
        )
        assert False, "Should have raised Exit"
    except typer.Exit as e:
        assert e.exit_code == 1


def test_queue_resolve_all_accept_json_output():
    """queue_resolve_all_cmd --accept --json emits count JSON."""
    from drbrain.cli.export_commands import queue_resolve_all_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        ctx = _make_ctx(cfg)

        captured: list[str] = []
        with (
            mock.patch("drbrain.extractor.queue.resolve_all", return_value={"count": 0}),
            mock.patch("typer.echo", side_effect=_capture_factory(captured)),
        ):
            queue_resolve_all_cmd(
                ctx,
                accept=True,
                reject=False,
                type_filter=None,
                max_conf=None,
                json_output=True,
            )

        payload = json.loads(captured[0])
        assert payload["action"] == "accept"
        assert payload["count"] == 0


# ── delete_cmd ────────────────────────────────────────────────────────────


def test_delete_cmd_unknown_paper_exits_1():
    """delete_cmd exits with code 1 for a paper not in the DB."""
    from drbrain.cli.export_commands import delete_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        ctx = _make_ctx(cfg)

        try:
            delete_cmd(ctx, "missing", force=True, rm_files=False, json_output=False)
            assert False, "Should have raised Exit"
        except typer.Exit as e:
            assert e.exit_code == 1


def test_delete_cmd_json_missing_paper():
    """delete_cmd --json on a missing paper still raises Exit(1)."""
    from drbrain.cli.export_commands import delete_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        ctx = _make_ctx(cfg)

        try:
            delete_cmd(ctx, "missing", force=True, rm_files=False, json_output=True)
            assert False, "Should have raised Exit"
        except typer.Exit as e:
            assert e.exit_code == 1


# ── backup_cmd ────────────────────────────────────────────────────────────


def test_backup_list_empty_message():
    """backup_cmd --list echoes 'No backups found.' when there are none."""
    from drbrain.cli.export_commands import backup_cmd

    cfg = _make_minimal_config(":memory:")
    ctx = _make_ctx(cfg)

    with (
        mock.patch("drbrain.storage.backup.list_backups", return_value=[]),
        mock.patch("typer.echo") as mock_echo,
    ):
        backup_cmd(
            ctx,
            output=None,
            list_only=True,
            target=None,
            dry_run=False,
            json_output=False,
        )
        mock_echo.assert_any_call("No backups found.")


def test_backup_list_json_empty():
    """backup_cmd --list --json emits {"backups": []} when none exist."""
    from drbrain.cli.export_commands import backup_cmd

    cfg = _make_minimal_config(":memory:")
    ctx = _make_ctx(cfg)

    captured: list[str] = []
    with (
        mock.patch("drbrain.storage.backup.list_backups", return_value=[]),
        mock.patch("typer.echo", side_effect=_capture_factory(captured)),
    ):
        backup_cmd(
            ctx,
            output=None,
            list_only=True,
            target=None,
            dry_run=False,
            json_output=True,
        )

    payload = json.loads(captured[0])
    assert payload == {"backups": []}


def test_backup_list_shows_existing():
    """backup_cmd --list lists existing backups with their size."""
    from drbrain.cli.export_commands import backup_cmd

    cfg = _make_minimal_config(":memory:")
    ctx = _make_ctx(cfg)

    with tempfile.TemporaryDirectory() as td:
        fake_file = Path(td) / "drbrain-20240101-000000.tar.gz"
        fake_file.write_bytes(b"x" * 2048)

        with (
            mock.patch("drbrain.storage.backup.list_backups", return_value=[fake_file]),
            mock.patch("typer.echo") as mock_echo,
        ):
            backup_cmd(
                ctx,
                output=None,
                list_only=True,
                target=None,
                dry_run=False,
                json_output=False,
            )
            calls = [str(c) for c in mock_echo.call_args_list]
            assert any("drbrain-20240101-000000.tar.gz" in c for c in calls)
            assert any("MB" in c for c in calls)


def test_backup_creates_tarball(tmp_path):
    """backup_cmd without --list/--target creates a tar.gz archive."""
    from drbrain.cli.export_commands import backup_cmd

    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    (papers_dir / "paper.txt").write_text("data")

    db_path = tmp_path / "test.db"
    db_path.write_bytes(b"sqlite")  # non-empty so create_backup includes it
    reports_dir = tmp_path / "reports"
    backup_dir = tmp_path / "backups"

    cfg = {
        "db": {"path": str(db_path)},
        "llm": {"models": []},
        "dirs": {
            "papers": str(papers_dir),
            "reports": str(reports_dir),
            "backups": str(backup_dir),
        },
    }
    ctx = _make_ctx(cfg)

    captured: list[str] = []
    with mock.patch("typer.echo", side_effect=_capture_factory(captured)):
        backup_cmd(
            ctx,
            output=None,
            list_only=False,
            target=None,
            dry_run=False,
            json_output=False,
        )

    assert captured[0].startswith("Backup created:")
    assert "MB" in captured[0]
    assert any(backup_dir.glob("drbrain-*.tar.gz"))


def test_backup_custom_output_path(tmp_path):
    """backup_cmd -o writes to the requested custom output path."""
    from drbrain.cli.export_commands import backup_cmd

    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    db_path = tmp_path / "test.db"
    db_path.write_bytes(b"sqlite")

    cfg = {
        "db": {"path": str(db_path)},
        "llm": {"models": []},
        "dirs": {"papers": str(papers_dir), "reports": "reports", "backups": "backups"},
    }
    ctx = _make_ctx(cfg)

    out_file = tmp_path / "custom.tar.gz"
    captured: list[str] = []
    with mock.patch("typer.echo", side_effect=_capture_factory(captured)):
        backup_cmd(
            ctx,
            output=str(out_file),
            list_only=False,
            target=None,
            dry_run=False,
            json_output=False,
        )

    assert out_file.exists()
    assert captured[0].startswith("Backup created:")


# ── restore_cmd ───────────────────────────────────────────────────────────


def test_restore_missing_file_exits_1():
    """restore_cmd exits with code 1 when the backup path doesn't exist."""
    from drbrain.cli.export_commands import restore_cmd

    cfg = _make_minimal_config(":memory:")
    ctx = _make_ctx(cfg)

    try:
        restore_cmd(
            ctx, "/nonexistent/drbrain-xxx.tar.gz", target=None, force=False, json_output=False
        )
        assert False, "Should have raised Exit"
    except typer.Exit as e:
        assert e.exit_code == 1


def test_restore_unsupported_format_exits_1(tmp_path):
    """restore_cmd exits with code 1 for a non-archive, non-directory path."""
    from drbrain.cli.export_commands import restore_cmd

    weird = tmp_path / "not-a-backup.weird"
    weird.write_bytes(b"junk")

    cfg = _make_minimal_config(":memory:")
    ctx = _make_ctx(cfg)

    try:
        restore_cmd(ctx, str(weird), target=None, force=False, json_output=False)
        assert False, "Should have raised Exit"
    except typer.Exit as e:
        assert e.exit_code == 1


def test_restore_directory_backup(tmp_path):
    """restore_cmd copies a directory backup into the target."""
    from drbrain.cli.export_commands import restore_cmd

    src = tmp_path / "src"
    src.mkdir()
    (src / "paper.txt").write_text("hello")
    dest = tmp_path / "dest"

    cfg = _make_minimal_config(":memory:")
    ctx = _make_ctx(cfg)

    captured: list[str] = []
    with mock.patch("typer.echo", side_effect=_capture_factory(captured)):
        restore_cmd(ctx, str(src), target=str(dest), force=False, json_output=False)

    assert any("Restored" in m for m in captured)
    assert (dest / "paper.txt").read_text() == "hello"


def test_restore_json_output(tmp_path):
    """restore_cmd --json emits a JSON document with restored entries."""
    from drbrain.cli.export_commands import restore_cmd

    src = tmp_path / "src"
    src.mkdir()
    (src / "note.md").write_text("data")
    dest = tmp_path / "dest"

    cfg = _make_minimal_config(":memory:")
    ctx = _make_ctx(cfg)

    captured: list[str] = []
    with mock.patch("typer.echo", side_effect=_capture_factory(captured)):
        restore_cmd(ctx, str(src), target=str(dest), force=False, json_output=True)

    payload = json.loads(captured[0])
    assert "restored" in payload
    assert payload["source"] == str(src)


# ── style_cmd ─────────────────────────────────────────────────────────────


def test_style_list_outputs_builtins():
    """style_cmd --list lists built-in citation styles."""
    from drbrain.cli.export_commands import style_cmd

    cfg = _make_minimal_config(":memory:")
    ctx = _make_ctx(cfg)

    fake_styles = [
        {"name": "apa", "description": "APA style", "source": "built-in"},
        {"name": "vancouver", "description": "Vancouver style", "source": "built-in"},
    ]

    with (
        mock.patch("drbrain.services.citation_styles.list_styles", return_value=fake_styles),
        mock.patch("typer.echo") as mock_echo,
    ):
        style_cmd(
            ctx,
            list_styles_flag=True,
            show=None,
            json_output=False,
        )
        calls = [str(c) for c in mock_echo.call_args_list]
        assert any("apa" in c for c in calls)
        assert any("vancouver" in c for c in calls)


def test_style_list_json_output():
    """style_cmd --list --json emits the styles as a JSON array."""
    from drbrain.cli.export_commands import style_cmd

    cfg = _make_minimal_config(":memory:")
    ctx = _make_ctx(cfg)

    fake_styles = [{"name": "apa", "description": "APA", "source": "built-in"}]

    captured: list[str] = []
    with (
        mock.patch("drbrain.services.citation_styles.list_styles", return_value=fake_styles),
        mock.patch("typer.echo", side_effect=_capture_factory(captured)),
    ):
        style_cmd(
            ctx,
            list_styles_flag=True,
            show=None,
            json_output=True,
        )

    payload = json.loads(captured[0])
    assert payload == fake_styles


# ── lineage_cmd ───────────────────────────────────────────────────────────


def test_lineage_no_args_exits_1():
    """lineage_cmd without args exits with code 1."""
    from drbrain.cli.export_commands import lineage_cmd

    cfg = _make_minimal_config(":memory:")
    ctx = _make_ctx(cfg)

    try:
        lineage_cmd(ctx, author_id=None, list_all=False, name=None, json_output=False)
        assert False, "Should have raised Exit"
    except typer.Exit as e:
        assert e.exit_code == 1


def test_lineage_list_all_empty_message():
    """lineage_cmd --list shows 'No actors found.' when DB has none."""
    from drbrain.cli.export_commands import lineage_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path), str(Path(td) / "reports"))
        ctx = _make_ctx(cfg)

        with mock.patch("typer.echo") as mock_echo:
            lineage_cmd(ctx, author_id=None, list_all=True, name=None, json_output=False)
            mock_echo.assert_any_call("No actors found.")


# ── document_cmd ──────────────────────────────────────────────────────────


def test_document_missing_file_exits_1():
    """document_cmd exits with code 1 when the file does not exist."""
    from drbrain.cli.export_commands import document_cmd

    cfg = _make_minimal_config(":memory:")
    ctx = _make_ctx(cfg)

    try:
        document_cmd(ctx, "/nonexistent.docx", fmt=None)
        assert False, "Should have raised Exit"
    except typer.Exit as e:
        assert e.exit_code == 1


def test_document_calls_inspect(tmp_path):
    """document_cmd delegates to services.document.inspect on a real file."""
    from drbrain.cli.export_commands import document_cmd

    f = tmp_path / "doc.docx"
    f.write_bytes(b"PK\x03\x04")

    cfg = _make_minimal_config(":memory:")
    ctx = _make_ctx(cfg)

    with (
        mock.patch(
            "drbrain.services.document.inspect", return_value="INSPECTED OUTPUT"
        ) as mock_inspect,
        mock.patch("typer.echo") as mock_echo,
    ):
        document_cmd(ctx, str(f), fmt=None)
        mock_inspect.assert_called_once()
        mock_echo.assert_any_call("INSPECTED OUTPUT")


# ── metrics_cmd ───────────────────────────────────────────────────────────


def test_metrics_json_output_empty(tmp_path, monkeypatch):
    """metrics_cmd --json emits weekly_trend/top_keywords/most_read on an empty DB."""
    from drbrain.cli.export_commands import metrics_cmd

    cfg = _make_minimal_config(":memory:")
    ctx = _make_ctx(cfg)

    captured: list[str] = []
    monkeypatch.setattr("drbrain.services.metrics_panel._ensure_metrics_db", lambda p: None)
    monkeypatch.setattr(
        "drbrain.services.metrics_panel.get_weekly_trend",
        lambda p: {
            "total_searches": 0,
            "total_reads": 0,
            "unique_keywords": 0,
            "unique_papers_read": 0,
        },
    )
    monkeypatch.setattr("drbrain.services.metrics_panel.get_top_keywords", lambda p, limit=5: [])
    monkeypatch.setattr(
        "drbrain.services.metrics_panel.get_most_read_papers", lambda p, limit=5: []
    )

    with mock.patch("typer.echo", side_effect=_capture_factory(captured)):
        metrics_cmd(ctx, json_output=True)

    payload = json.loads(captured[0])
    assert "weekly_trend" in payload
    assert "top_keywords" in payload
    assert "most_read" in payload


def test_metrics_text_output_no_data(tmp_path, monkeypatch):
    """metrics_cmd (text mode) shows a no-metrics hint when DB is empty."""
    from drbrain.cli.export_commands import metrics_cmd

    cfg = _make_minimal_config(":memory:")
    ctx = _make_ctx(cfg)

    monkeypatch.setattr("drbrain.services.metrics_panel._ensure_metrics_db", lambda p: None)
    monkeypatch.setattr(
        "drbrain.services.metrics_panel.get_weekly_trend",
        lambda p: {
            "total_searches": 0,
            "total_reads": 0,
            "unique_keywords": 0,
            "unique_papers_read": 0,
        },
    )
    monkeypatch.setattr("drbrain.services.metrics_panel.get_top_keywords", lambda p, limit=5: [])
    monkeypatch.setattr(
        "drbrain.services.metrics_panel.get_most_read_papers", lambda p, limit=5: []
    )

    with mock.patch("typer.echo") as mock_echo:
        metrics_cmd(ctx, json_output=False)
        mock_echo.assert_any_call("\nNo metrics recorded yet. Search and read papers to populate.")
