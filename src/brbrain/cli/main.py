"""DrBrain CLI — ingest, query, expand, and more."""
from __future__ import annotations

import typer

from brbrain.cli.commands import (
    ingest_cmd, expand_cmd, report_cmd, closure_cmd, seed_cmd,
    list_cmd, stats_cmd, query_cmd, export_cmd,
    queue_cmd, queue_resolve_cmd, timeline_cmd,
)
from brbrain.cli.setup import setup_cmd

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
app.command("timeline")(timeline_cmd)

if __name__ == "__main__":
    app()
