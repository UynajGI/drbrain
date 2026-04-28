"""DrBrain CLI — ingest, query, expand, and more."""

from __future__ import annotations

import typer

from drbrain.cli.commands import (
    check_cmd,
    closure_cmd,
    delete_cmd,
    expand_cmd,
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
)
from drbrain.cli.setup import setup_cmd

app = typer.Typer(help="DrBrain — Academic Knowledge Graph System")

app.command("setup")(setup_cmd)
app.command("ingest")(ingest_cmd)
app.command("expand")(expand_cmd)
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

if __name__ == "__main__":
    app()
