"""DrBrain CLI — ingest, query, expand, and more."""

from __future__ import annotations

import typer

from drbrain.cli.commands import (
    backup_cmd,
    check_citations_cmd,
    check_cmd,
    citations_cmd,
    clean_cmd,
    closure_cmd,
    delete_cmd,
    export_cmd,
    ingest_cmd,
    lineage_cmd,
    list_cmd,
    query_cmd,
    queue_cmd,
    queue_resolve_all_cmd,
    queue_resolve_cmd,
    report_cmd,
    seed_cmd,
    serve_cmd,
    stats_cmd,
    timeline_cmd,
    ws_add_cmd,
    ws_create_cmd,
    ws_delete_cmd,
    ws_list_cmd,
    ws_remove_cmd,
    ws_show_cmd,
)
from drbrain.cli.setup import setup_cmd

app = typer.Typer(help="DrBrain — Academic Knowledge Graph System")

app.command("setup")(setup_cmd)
app.command("ingest")(ingest_cmd)
app.command("citations")(citations_cmd)
app.command("check-citations")(check_citations_cmd)
app.command("report")(report_cmd)
app.command("closure")(closure_cmd)
app.command("seed")(seed_cmd)
app.command("list")(list_cmd)
app.command("stats")(stats_cmd)
app.command("query")(query_cmd)
app.command("export")(export_cmd)
app.command("queue")(queue_cmd)
app.command("queue resolve")(queue_resolve_cmd)
app.command("queue resolve-all")(queue_resolve_all_cmd)
app.command("timeline")(timeline_cmd)
app.command("delete")(delete_cmd)
app.command("serve")(serve_cmd)
app.command("lineage")(lineage_cmd)
app.command("check")(check_cmd)
app.command("clean")(clean_cmd)
app.command("backup")(backup_cmd)

# Workspace subcommands
ws_app = typer.Typer(help="Manage paper workspaces")
ws_app.command("create")(ws_create_cmd)
ws_app.command("add")(ws_add_cmd)
ws_app.command("remove")(ws_remove_cmd)
ws_app.command("list")(ws_list_cmd)
ws_app.command("show")(ws_show_cmd)
ws_app.command("delete")(ws_delete_cmd)
app.add_typer(ws_app, name="ws")

if __name__ == "__main__":
    app()
