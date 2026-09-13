"""CLI commands for persistent session-based reasoning."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from drbrain.runtime import RuntimeContext
from drbrain.security import redact_sensitive, redact_sensitive_text
from drbrain.storage.database import Database
from drbrain.storage.paths import writable_artifact_path

session_app = typer.Typer(name="session", help="Manage continuous reasoning sessions.")
console = Console()


def _session_export_runtime(ctx: typer.Context) -> RuntimeContext | None:
    """Return the invocation runtime, rejecting an empty inherited selector."""
    obj = getattr(ctx, "obj", None)
    if isinstance(obj, dict) and obj.get("runtime") is not None:
        return obj["runtime"]
    if "DRBRAIN_ROOT" in os.environ or "DRBRAIN_RUNTIME_ROOT" in os.environ:
        return RuntimeContext.create()
    return None


def _ensure_export_parent(path: Path) -> Path:
    """Create an export parent without following a lexical symlink."""
    parent = path.parent
    for ancestor in (parent, *parent.parents):
        if ancestor.is_symlink():
            raise ValueError(f"session export directory contains a symlink: {ancestor}")
    current = parent
    missing: list[Path] = []
    while not current.exists():
        missing.append(current)
        next_parent = current.parent
        if next_parent == current:
            break
        current = next_parent
    if current.is_symlink() or not current.is_dir():
        raise ValueError(f"session export directory is not a real directory: {current}")
    for directory in reversed(missing):
        directory.mkdir(exist_ok=True)
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"session export directory is not a real directory: {directory}")
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(f"session export directory is not a real directory: {parent}")
    return parent


def _session_export_target(ctx: typer.Context, output: str) -> tuple[Path, Path] | None:
    """Resolve an optional output path and return (destination, parent)."""
    if not output:
        return None
    if "\x00" in output:
        raise ValueError("session export output must not contain NUL bytes")
    path = Path(output).expanduser()
    runtime = _session_export_runtime(ctx)
    if runtime is not None:
        path = runtime.assert_within_root(path, label="session export output")
    if not path.name or path.name in {".", ".."}:
        raise ValueError("session export output must be a regular file path")
    if path.exists() and not path.is_file():
        raise ValueError(f"session export output is not a regular file: {path}")
    parent = _ensure_export_parent(path)
    # Validate the final target after parent creation; this rejects a stale
    # symlink without following it during the eventual atomic replacement.
    destination = writable_artifact_path(parent, path.name)
    return destination, parent


def _write_session_export(target: tuple[Path, Path], text: str) -> Path:
    """Publish an export atomically beside its destination."""
    destination, parent = target
    if destination.is_symlink():
        raise ValueError(f"session export output is a symlink: {destination}")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=str(parent), text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
    return destination


def _load_models(ctx: typer.Context) -> list[dict]:
    """Extract LLM model configs from CLI context."""
    cfg = ctx.obj["config"]
    return cfg.get("llm", {}).get("models", [])


def _load_graph(db: Database):
    """Load GraphEngine from database."""
    from drbrain.graph.engine import GraphEngine

    graph = GraphEngine()
    graph.load_from_db(db)
    return graph


@session_app.command("new")
def session_new_cmd(
    ctx: typer.Context,
    title: str = typer.Option("", "--title", "-t", help="Session title"),
):
    """Create a new reasoning session."""
    from drbrain.extractor.session_agent import SessionAgent

    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])
    models = _load_models(ctx)

    if not models:
        db.close()
        typer.echo("No LLM models configured. Run: drbrain setup", err=True)
        raise typer.Exit(1)

    agent = SessionAgent()
    sid = agent.create_session(db, title=title, models=models)
    typer.echo(f"Session created: {sid}")
    if title:
        typer.echo(f"  Title: {title}")
    typer.echo(f"  Models: {len(models)}")
    typer.echo(f'\nUse: drbrain session ask {sid} "your question"')
    typer.echo(f"Or:  drbrain session chat {sid}")
    db.close()


@session_app.command("ask")
def session_ask_cmd(
    ctx: typer.Context,
    session_id: str = typer.Argument(..., help="Session ID"),
    question: str = typer.Argument(..., help="Question to ask"),
    max_turns: int = typer.Option(8, "--max-turns", "-m", help="Max tool-calling rounds"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Ask a question within an existing session (context-aware)."""
    from drbrain.extractor.session_agent import SessionAgent

    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])
    graph = _load_graph(db)
    models = _load_models(ctx)

    agent = SessionAgent()
    if not agent.load_session(db, session_id, graph=graph, models=models):
        typer.echo(f"Session not found: {session_id}", err=True)
        db.close()
        raise typer.Exit(1)

    answer = asyncio.run(agent.ask(question, max_turns=max_turns))

    if json_output:
        typer.echo(
            json.dumps(
                {"session_id": session_id, "question": question, "answer": answer},
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        typer.echo(f"\nQ: {question}\n")
        typer.echo(f"A: {answer}")
        typer.echo(f"\n(session: {session_id}, {len(agent.messages)} messages)")

    db.close()


@session_app.command("chat")
def session_chat_cmd(
    ctx: typer.Context,
    session_id: str = typer.Argument(..., help="Session ID"),
    max_turns: int = typer.Option(
        8, "--max-turns", "-m", help="Max tool-calling rounds per question"
    ),
):
    """Enter interactive chat mode for a session."""
    from drbrain.extractor.session_agent import SessionAgent

    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])
    graph = _load_graph(db)
    models = _load_models(ctx)

    agent = SessionAgent()
    if not agent.load_session(db, session_id, graph=graph, models=models):
        typer.echo(f"Session not found: {session_id}", err=True)
        db.close()
        raise typer.Exit(1)

    typer.echo(f"Loaded session: {session_id} ({len(agent.messages)} messages in context)\n")
    asyncio.run(agent.chat(max_turns_per_question=max_turns))
    db.close()


@session_app.command("list")
def session_list_cmd(
    ctx: typer.Context,
    show_all: bool = typer.Option(False, "--all", "-a", help="Include deleted/archived sessions"),
):
    """List all active sessions."""
    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])

    if show_all:
        rows = db.conn.execute(
            "SELECT session_id, title, status, created_at, updated_at "
            "FROM agent_sessions ORDER BY created_at DESC"
        ).fetchall()
    else:
        rows = db.conn.execute(
            "SELECT session_id, title, status, created_at, updated_at "
            "FROM agent_sessions WHERE status = 'active' ORDER BY updated_at DESC"
        ).fetchall()

    if not rows:
        typer.echo("No sessions found. Create one with: drbrain session new")
        db.close()
        return

    table = Table(title="Sessions")
    table.add_column("Session ID", style="cyan", no_wrap=True)
    table.add_column("Title", style="white")
    table.add_column("Status", style="green")
    table.add_column("Messages", justify="right")
    table.add_column("Created", style="dim")
    table.add_column("Updated", style="dim")

    for row in rows:
        sid, title, status, created_at, updated_at = row
        msg_count = db.conn.execute(
            "SELECT COUNT(*) FROM agent_messages WHERE session_id = ?", (sid,)
        ).fetchone()[0]
        table.add_row(
            sid,
            title or "(untitled)",
            status,
            str(msg_count),
            created_at[:19] if created_at else "",
            updated_at[:19] if updated_at else "",
        )

    console.print(table)
    db.close()


@session_app.command("delete")
def session_delete_cmd(
    ctx: typer.Context,
    session_id: str = typer.Argument(..., help="Session ID to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Delete a session (soft delete)."""
    from drbrain.extractor.session_agent import SessionAgent

    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])

    # Verify exists
    row = db.conn.execute(
        "SELECT title FROM agent_sessions WHERE session_id = ? AND status != 'deleted'",
        (session_id,),
    ).fetchone()
    if not row:
        typer.echo(f"Session not found: {session_id}", err=True)
        db.close()
        raise typer.Exit(1)

    if not force:
        confirm = typer.confirm(f"Delete session '{session_id}' ({row[0] or 'untitled'})?")
        if not confirm:
            typer.echo("Cancelled.")
            db.close()
            return

    agent = SessionAgent()
    agent.delete_session(db, session_id)
    typer.echo(f"Session deleted: {session_id}")
    db.close()


@session_app.command("export")
def session_export_cmd(
    ctx: typer.Context,
    session_id: str = typer.Argument(..., help="Session ID to export"),
    output: str = typer.Option("", "--output", "-o", help="Output file path (default: stdout)"),
    fmt: str = typer.Option("json", "--format", "-F", help="Format: json or markdown"),
):
    """Export session history as JSON or Markdown."""
    output_target = _session_export_target(ctx, output)
    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])

    # Load metadata
    meta = db.conn.execute(
        "SELECT session_id, title, system_prompt, status, model_config, created_at "
        "FROM agent_sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if not meta:
        typer.echo(f"Session not found: {session_id}", err=True)
        db.close()
        raise typer.Exit(1)

    # Load messages
    rows = db.conn.execute(
        "SELECT role, content, tool_calls_json, tool_call_id, tool_name, created_at "
        "FROM agent_messages WHERE session_id = ? ORDER BY seq",
        (session_id,),
    ).fetchall()

    db.close()

    if fmt == "json":
        data = {
            "session_id": meta[0],
            "title": redact_sensitive_text(meta[1]) if meta[1] else "",
            "system_prompt": redact_sensitive(meta[2]) or "",
            "status": meta[3],
            "created_at": meta[5],
            "messages": [
                {
                    "role": r[0],
                    "content": redact_sensitive(r[1]) or "",
                    "tool_calls": redact_sensitive(json.loads(r[2])) if r[2] else None,
                    "tool_call_id": redact_sensitive_text(r[3]) if r[3] else None,
                    "tool_name": redact_sensitive_text(r[4]) if r[4] else None,
                    "created_at": r[5],
                }
                for r in rows
            ],
        }
        text = json.dumps(data, indent=2, ensure_ascii=False)
    elif fmt == "markdown":
        title = redact_sensitive_text(meta[1]) if meta[1] else ""
        lines = [f"# Session: {meta[0]}", f"**Title**: {title or '(untitled)'}", ""]
        for r in rows:
            role, content = r[0], redact_sensitive(r[1]) or ""
            lines.append(f"## [{role}]")
            if r[2]:
                tc = json.loads(r[2])
                lines.append(
                    f"*Tool calls: {[t.get('function', {}).get('name', '?') for t in tc]}*"
                )
            if content:
                lines.append(content)
            lines.append("")
        text = "\n".join(lines)
    else:
        typer.echo(f"Unknown format: {fmt}", err=True)
        raise typer.Exit(1)

    if output_target is not None:
        _write_session_export(output_target, text)
        typer.echo(f"Exported to: {output}")
    else:
        typer.echo(text)
