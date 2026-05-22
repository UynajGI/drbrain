"""CLI commands for persistent session-based reasoning."""

from __future__ import annotations

import asyncio
import json

import typer
from rich.console import Console
from rich.table import Table

from drbrain.storage.database import Database

session_app = typer.Typer(name="session", help="Manage continuous reasoning sessions.")
console = Console()


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
            "title": meta[1],
            "system_prompt": meta[2],
            "status": meta[3],
            "created_at": meta[5],
            "messages": [
                {
                    "role": r[0],
                    "content": r[1],
                    "tool_calls": json.loads(r[2]) if r[2] else None,
                    "tool_call_id": r[3] or None,
                    "tool_name": r[4] or None,
                    "created_at": r[5],
                }
                for r in rows
            ],
        }
        text = json.dumps(data, indent=2, ensure_ascii=False)
    elif fmt == "markdown":
        lines = [f"# Session: {meta[0]}", f"**Title**: {meta[1] or '(untitled)'}", ""]
        for r in rows:
            role, content = r[0], r[1] or ""
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

    if output:
        from pathlib import Path

        Path(output).write_text(text, encoding="utf-8")
        typer.echo(f"Exported to: {output}")
    else:
        typer.echo(text)
