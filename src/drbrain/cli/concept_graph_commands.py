"""CLI commands for the concept graph (co-occurrence) layer.

Exposed as the ``drbrain cg`` sub-app. Additive to the existing CLI; does not
touch the full-text parser or RAG commands.
"""

from __future__ import annotations

import json

import typer
from loguru import logger

from drbrain.cli._common import open_db

cg_app = typer.Typer(help="Concept co-occurrence graph layer (build / embed / predict / recommend)")


@cg_app.command("ingest")
def cg_ingest_cmd(
    ctx: typer.Context,
    query: str = typer.Argument(None, help="Free-text search query (optional with filters)"),
    source: str = typer.Option(
        "sciverse", "--source", "-s", help="Corpus source: sciverse | openalex"
    ),
    year_from: int = typer.Option(None, "--year-from", help="Minimum publication year"),
    year_to: int = typer.Option(None, "--year-to", help="Maximum publication year"),
    venue: list[str] = typer.Option(None, "--venue", help="Venue filter (repeatable)"),
    limit: int = typer.Option(100, "--limit", "-n", help="Max records to fetch"),
    with_citations: bool = typer.Option(
        False, "--with-citations", help="Also harvest citation edges"
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON stats"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Count matches without writing"),
) -> None:
    """Ingest paper metadata from an external academic API into the library."""
    from drbrain.concept_graph.ingest import ingest_citations, ingest_corpus
    from drbrain.concept_graph.sources.registry import get_source

    cfg = ctx.obj["config"]
    try:
        src = get_source(source)
    except KeyError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    if dry_run:
        count = sum(
            1
            for _ in src.search(
                query, year_from=year_from, year_to=year_to, venues=venue or None, limit=limit
            )
        )
        payload = {"dry_run": True, "source": source, "would_fetch": count}
        typer.echo(json.dumps(payload) if json_output else f"[dry-run] {source}: {count} matches")
        return

    with open_db(cfg) as db:
        stats = ingest_corpus(
            db, src, query, year_from=year_from, year_to=year_to, venues=venue or None, limit=limit
        )
        if with_citations:
            cit = ingest_citations(db, src)
            stats.citations = cit.citations

    payload = {
        "source": source,
        "fetched": stats.fetched,
        "inserted": stats.inserted,
        "skipped": stats.skipped,
        "citations": stats.citations,
    }
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False))
    else:
        typer.echo(
            f"[cg.ingest] {source}: fetched={stats.fetched} inserted={stats.inserted} "
            f"skipped={stats.skipped} citations={stats.citations}"
        )
    logger.info("[cg.ingest] done: {}", payload)


@cg_app.command("build")
def cg_build_cmd(
    ctx: typer.Context,
    source: str = typer.Option(
        "terms", "--source", "-s", help="Concept source: terms | concepts | abstract"
    ),
    min_freq: int = typer.Option(
        3, "--min-freq", help="Minimum document frequency to keep a concept"
    ),
    min_words: int = typer.Option(2, "--min-words", help="Minimum words in a concept label"),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON stats"),
) -> None:
    """Build the concept co-occurrence graph (cliques + frequency filtering)."""
    from drbrain.concept_graph.builder import apply_filter, build_cliques

    cfg = ctx.obj["config"]
    with open_db(cfg) as db:
        edges = build_cliques(db, source=source)
        stats = apply_filter(db, min_freq=min_freq, min_words=min_words)

    payload = {"edges": edges, "total_concepts": stats["total_concepts"], "kept": stats["kept"]}
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False))
    else:
        typer.echo(
            f"[cg.build] edges={edges} concepts={stats['total_concepts']} kept={stats['kept']}"
        )
