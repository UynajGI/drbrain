"""Tests for drbrain.cli.session_commands — session new/list/ask/delete/export."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

import typer

from drbrain.storage.database import Database


def _make_minimal_config(db_path: str) -> dict:
    return {
        "db": {"path": db_path},
        "llm": {"models": [{"provider": "openai", "model": "gpt-4", "api_key": "x"}]},
        "dirs": {
            "inbox": "data/spool/inbox",
            "papers": "data/papers",
            "reports": "data/reports",
            "cache": "data/cache",
            "logs": "data/logs",
        },
        "api": {},
        "mineru": {},
        "bm25": {"k1": 1.5, "b": 0.75},
        "queue": {"weak_threshold": 0.5, "auto_accept": False},
    }


def _make_ctx(cfg: dict):
    """Create a minimal typer.Context mock with config pre-loaded."""
    ctx = mock.MagicMock(spec=typer.Context)
    ctx.obj = {"config": cfg}
    return ctx


def _seed_session(db_path: str, title: str = "Seeded", *, models=None) -> str:
    """Insert a session row directly, return its session_id."""
    db = Database(str(db_path))
    models = models or [{"provider": "openai", "model": "gpt-4", "api_key": "x"}]
    agent_mod = __import__("drbrain.extractor.session_agent", fromlist=["SessionAgent"])
    agent = agent_mod.SessionAgent()
    sid = agent.create_session(db, title=title, models=models)
    db.close()
    return sid


# ── session new ───────────────────────────────────────────────────────────


def test_session_new_creates_session():
    """session_new_cmd creates a session row and echoes the new id."""
    from drbrain.cli.session_commands import session_new_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path))
        ctx = _make_ctx(cfg)

        with mock.patch("typer.echo") as mock_echo:
            session_new_cmd(ctx, title="My Session")

        # Output mentions the session creation
        calls = [str(c) for c in mock_echo.call_args_list]
        assert any("Session created:" in c for c in calls)
        assert any("My Session" in c for c in calls)

        # DB now has one session row
        db = Database(str(db_path))
        rows = db.conn.execute("SELECT session_id FROM agent_sessions").fetchall()
        db.close()
        assert len(rows) == 1


def test_session_new_no_models_exits_1():
    """session_new_cmd exits with code 1 when no models configured."""
    from drbrain.cli.session_commands import session_new_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path))
        cfg["llm"]["models"] = []
        ctx = _make_ctx(cfg)

        try:
            session_new_cmd(ctx, title="No Models")
            assert False, "Should have raised Exit"
        except typer.Exit as e:
            assert e.exit_code == 1

        # No session row was created
        db = Database(str(db_path))
        rows = db.conn.execute("SELECT COUNT(*) FROM agent_sessions").fetchone()
        db.close()
        assert rows[0] == 0


def test_session_new_without_title():
    """session_new_cmd works with an empty title (default)."""
    from drbrain.cli.session_commands import session_new_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path))
        ctx = _make_ctx(cfg)

        session_new_cmd(ctx, title="")

        db = Database(str(db_path))
        rows = db.conn.execute("SELECT title FROM agent_sessions").fetchall()
        db.close()
        assert len(rows) == 1
        assert rows[0][0] == ""


# ── session list ──────────────────────────────────────────────────────────


def test_session_list_empty_shows_message():
    """session_list_cmd with no sessions echoes 'No sessions found.'."""
    from drbrain.cli.session_commands import session_list_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path))
        ctx = _make_ctx(cfg)

        with mock.patch("typer.echo") as mock_echo:
            session_list_cmd(ctx, show_all=False)
            mock_echo.assert_any_call("No sessions found. Create one with: drbrain session new")


def test_session_list_shows_active_session():
    """session_list_cmd renders a table when an active session exists."""
    from drbrain.cli.session_commands import session_list_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path))
        ctx = _make_ctx(cfg)

        _seed_session(db_path, title="Visible Session")

        # console.print is what renders the table; patch it to ensure no crash.
        with mock.patch("drbrain.cli.session_commands.console.print") as mock_print:
            session_list_cmd(ctx, show_all=False)
            assert mock_print.called


def test_session_list_all_includes_deleted():
    """session_list_cmd --all also returns deleted/archived sessions."""
    from drbrain.cli.session_commands import session_list_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path))
        ctx = _make_ctx(cfg)

        sid = _seed_session(db_path, title="To Delete")
        db = Database(str(db_path))
        db.conn.execute("UPDATE agent_sessions SET status='deleted' WHERE session_id=?", (sid,))
        db.commit()
        db.close()

        # Without --all: deleted sessions are filtered out
        with mock.patch("typer.echo") as mock_echo_active:
            session_list_cmd(ctx, show_all=False)
            mock_echo_active.assert_any_call(
                "No sessions found. Create one with: drbrain session new"
            )

        # With --all: deleted session shows up via the table
        with mock.patch("drbrain.cli.session_commands.console.print") as mock_print:
            session_list_cmd(ctx, show_all=True)
            assert mock_print.called


# ── session ask ───────────────────────────────────────────────────────────


def test_session_ask_unknown_session_exits_1():
    """session_ask_cmd exits with code 1 when session not found."""
    from drbrain.cli.session_commands import session_ask_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path))
        ctx = _make_ctx(cfg)

        try:
            session_ask_cmd(ctx, "sess-does-not-exist", "hello")
            assert False, "Should have raised Exit"
        except typer.Exit as e:
            assert e.exit_code == 1


def test_session_ask_returns_answer_with_mock_llm():
    """session_ask_cmd invokes agent.ask and prints the answer."""
    from drbrain.cli.session_commands import session_ask_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path))
        ctx = _make_ctx(cfg)
        sid = _seed_session(db_path, title="Ask Me")

        with mock.patch("drbrain.extractor.session_agent.SessionAgent") as mock_agent_cls:
            instance = mock_agent_cls.return_value
            instance.load_session.return_value = True
            instance.ask = mock.AsyncMock(return_value="42 is the answer")
            instance.messages = [{"role": "system"}, {"role": "user"}, {"role": "assistant"}]

            with mock.patch("typer.echo") as mock_echo:
                session_ask_cmd(ctx, sid, "what is the answer?")

            instance.load_session.assert_called_once()
            assert instance.ask.await_args is not None
            assert instance.ask.await_args.args == ("what is the answer?",)

            calls = [str(c) for c in mock_echo.call_args_list]
            assert any("42 is the answer" in c for c in calls)


def test_session_ask_json_output():
    """session_ask_cmd --json emits a JSON document with session_id/question/answer."""
    from drbrain.cli.session_commands import session_ask_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path))
        ctx = _make_ctx(cfg)
        sid = _seed_session(db_path, title="JSON Ask")

        with mock.patch("drbrain.extractor.session_agent.SessionAgent") as mock_agent_cls:
            instance = mock_agent_cls.return_value
            instance.load_session.return_value = True
            instance.ask = mock.AsyncMock(return_value="json answer")
            instance.messages = [{"role": "system"}]

            captured: list[str] = []

            def _capture(msg="", *args, **kwargs):
                captured.append(str(msg))

            with mock.patch("typer.echo", side_effect=_capture):
                session_ask_cmd(ctx, sid, "q?", json_output=True)

            payload = json.loads(captured[0])
            assert payload["session_id"] == sid
            assert payload["question"] == "q?"
            assert payload["answer"] == "json answer"


# ── session delete ────────────────────────────────────────────────────────


def test_session_delete_unknown_session_exits_1():
    """session_delete_cmd exits with code 1 when session_id does not exist."""
    from drbrain.cli.session_commands import session_delete_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path))
        ctx = _make_ctx(cfg)

        try:
            session_delete_cmd(ctx, "sess-missing", force=True)
            assert False, "Should have raised Exit"
        except typer.Exit as e:
            assert e.exit_code == 1


def test_session_delete_force_marks_deleted():
    """session_delete_cmd --force soft-deletes an existing session."""
    from drbrain.cli.session_commands import session_delete_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path))
        ctx = _make_ctx(cfg)
        sid = _seed_session(db_path, title="Delete Me")

        with mock.patch("typer.echo") as mock_echo:
            session_delete_cmd(ctx, sid, force=True)
            mock_echo.assert_any_call(f"Session deleted: {sid}")

        db = Database(str(db_path))
        status = db.conn.execute(
            "SELECT status FROM agent_sessions WHERE session_id=?", (sid,)
        ).fetchone()
        db.close()
        assert status is not None
        assert status[0] == "deleted"


def test_session_delete_prompt_cancel():
    """session_delete_cmd without --force prompts; cancellation skips deletion."""
    from drbrain.cli.session_commands import session_delete_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path))
        ctx = _make_ctx(cfg)
        sid = _seed_session(db_path, title="Prompt Me")

        with mock.patch("typer.confirm", return_value=False):
            with mock.patch("typer.echo") as mock_echo:
                session_delete_cmd(ctx, sid, force=False)
                mock_echo.assert_any_call("Cancelled.")

        # Status remains active
        db = Database(str(db_path))
        status = db.conn.execute(
            "SELECT status FROM agent_sessions WHERE session_id=?", (sid,)
        ).fetchone()
        db.close()
        assert status[0] == "active"


# ── session export ────────────────────────────────────────────────────────


def test_session_export_unknown_session_exits_1():
    """session_export_cmd exits with code 1 for a missing session."""
    from drbrain.cli.session_commands import session_export_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path))
        ctx = _make_ctx(cfg)

        try:
            session_export_cmd(ctx, "sess-missing", output="", fmt="json")
            assert False, "Should have raised Exit"
        except typer.Exit as e:
            assert e.exit_code == 1


def test_session_export_json_stdout():
    """session_export_cmd emits valid JSON to stdout for an existing session."""
    from drbrain.cli.session_commands import session_export_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path))
        ctx = _make_ctx(cfg)
        sid = _seed_session(db_path, title="Exportable")

        captured: list[str] = []

        def _capture(msg="", *args, **kwargs):
            captured.append(str(msg))

        with mock.patch("typer.echo", side_effect=_capture):
            session_export_cmd(ctx, sid, output="", fmt="json")

        payload = json.loads(captured[0])
        assert payload["session_id"] == sid
        assert payload["title"] == "Exportable"
        assert isinstance(payload["messages"], list)


def test_session_export_markdown_stdout():
    """session_export_cmd with fmt=markdown produces markdown output."""
    from drbrain.cli.session_commands import session_export_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path))
        ctx = _make_ctx(cfg)
        sid = _seed_session(db_path, title="MD Export")

        captured: list[str] = []

        def _capture(msg="", *args, **kwargs):
            captured.append(str(msg))

        with mock.patch("typer.echo", side_effect=_capture):
            session_export_cmd(ctx, sid, output="", fmt="markdown")

        text = captured[0]
        assert text.startswith("# Session:")
        assert sid in text


def test_session_export_invalid_format_exits_1():
    """session_export_cmd exits with code 1 for an unsupported format."""
    from drbrain.cli.session_commands import session_export_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path))
        ctx = _make_ctx(cfg)
        sid = _seed_session(db_path, title="Bad Fmt")

        try:
            session_export_cmd(ctx, sid, output="", fmt="xml")
            assert False, "Should have raised Exit"
        except typer.Exit as e:
            assert e.exit_code == 1


def test_session_export_writes_file():
    """session_export_cmd -o writes the JSON document to disk."""
    from drbrain.cli.session_commands import session_export_cmd

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        cfg = _make_minimal_config(str(db_path))
        ctx = _make_ctx(cfg)
        sid = _seed_session(db_path, title="ToFile")

        out_file = Path(td) / "session.json"
        with mock.patch("typer.echo") as mock_echo:
            session_export_cmd(ctx, sid, output=str(out_file), fmt="json")
            mock_echo.assert_any_call(f"Exported to: {out_file}")

        written = json.loads(out_file.read_text())
        assert written["session_id"] == sid
        assert written["title"] == "ToFile"
