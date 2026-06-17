"""DrBrain CLI — ingest, query, expand, and more."""

from __future__ import annotations

import sys

import typer
from loguru import logger

from drbrain.cli.analysis_commands import (
    ask_cmd,
    descendants_cmd,
    difficulty_cmd,
    evolve_cmd,
    frontier_cmd,
    isomorphism_cmd,
    landscape_cmd,
    paradigm_cmd,
    reason_cmd,
    transfers_cmd,
)
from drbrain.cli.build_commands import (
    build_cmd,
    embed_cmd,
    translate_cmd,
)
from drbrain.cli.check_commands import (
    analyze_cmd,
    check_cmd,
    clean_cmd,
)
from drbrain.cli.export_commands import (
    backup_cmd,
    delete_cmd,
    document_cmd,
    export_cmd,
    lineage_cmd,
    metrics_cmd,
    queue_cmd,
    queue_resolve_all_cmd,
    queue_resolve_cmd,
    restore_cmd,
    style_cmd,
)
from drbrain.cli.graph_commands import graph_app
from drbrain.cli.ingest_commands import (
    batch_fetch_cmd,
    check_citations_cmd,
    citations_cmd,
    closure_cmd,
    explore_cmd,
    fetch_cmd,
    ingest_cmd,
    ingest_link_cmd,
    patent_search_cmd,
    pipeline_cmd,
    proceedings_cmd,
    report_cmd,
)
from drbrain.cli.query_commands import (
    fsearch_cmd,
    index_cmd,
    list_cmd,
    query_cmd,
    search_cmd,
    seed_cmd,
    show_cmd,
    stats_cmd,
)
from drbrain.cli.repair_commands import (
    enrich_cmd,
    import_cmd,
    repair_cmd,
)
from drbrain.cli.session_commands import session_app
from drbrain.cli.setup import setup_cmd
from drbrain.cli.ws_commands import ws_app
from drbrain.log import setup_logging
from drbrain.services.audit import audit_cmd

app = typer.Typer(help="DrBrain — Academic Knowledge Graph System")


@app.callback()
def _main_callback(ctx: typer.Context) -> None:
    """Called before every command. Sets up logging and loads config."""
    setup_logging()
    from drbrain.config import load_config
    from drbrain.log import get_session_id

    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config()

    cmd = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "(no args)"
    logger.info(f"CLI invoked [{get_session_id()}]: {cmd}")


app.command("setup")(setup_cmd)
app.command("ingest")(ingest_cmd)
app.command("ingest-link")(ingest_link_cmd)
app.command("patent-search")(patent_search_cmd)
app.command("pipeline")(pipeline_cmd)
app.command("proceedings")(proceedings_cmd)
app.command("explore")(explore_cmd)
app.command("batch-fetch")(batch_fetch_cmd)
app.command("fetch")(fetch_cmd)
app.command("citations")(citations_cmd)
app.command("check-citations")(check_citations_cmd)
app.command("report")(report_cmd)
app.command("closure")(closure_cmd)
app.command("seed")(seed_cmd)
app.command("list")(list_cmd)
app.command("stats")(stats_cmd)
app.command("show")(show_cmd)
app.command("index")(index_cmd)
app.command("query")(query_cmd)
app.command("fsearch")(fsearch_cmd)
app.command("search")(search_cmd)
app.command("export")(export_cmd)
app.command("queue")(queue_cmd)
app.command("queue resolve")(queue_resolve_cmd)
app.command("queue resolve-all")(queue_resolve_all_cmd)
app.command("delete")(delete_cmd)
app.command("lineage")(lineage_cmd)
app.command("ask")(ask_cmd)
app.command("check")(check_cmd)
app.command("audit")(audit_cmd)
app.command("style")(style_cmd)
app.command("document")(document_cmd)
app.command("metrics")(metrics_cmd)
app.command("clean")(clean_cmd)
app.command("backup")(backup_cmd)
app.command("restore")(restore_cmd)
app.command("analyze")(analyze_cmd)
app.command("repair")(repair_cmd)
app.command("enrich")(enrich_cmd)
app.command("import")(import_cmd)
app.command("translate")(translate_cmd)
app.command("build")(build_cmd)
app.command("embed")(embed_cmd)
app.command("evolve")(evolve_cmd)
app.command("descendants")(descendants_cmd)
app.command("landscape")(landscape_cmd)
app.command("paradigm")(paradigm_cmd)
app.command("transfers")(transfers_cmd)
app.command("isomorphism")(isomorphism_cmd)
app.command("difficulty")(difficulty_cmd)
app.command("frontier")(frontier_cmd)
app.command("reason")(reason_cmd)

# Sub-apps
app.add_typer(session_app, name="session")
app.add_typer(graph_app, name="graph")
app.add_typer(ws_app, name="ws")

if __name__ == "__main__":
    app()
