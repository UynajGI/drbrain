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


@cg_app.command("embed")
def cg_embed_cmd(
    ctx: typer.Context,
    context: bool = typer.Option(
        False, "--context", help="Average label with containing-paper titles"
    ),
    model_name: str = typer.Option("", "--model", help="Model label recorded in the DB"),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON stats"),
) -> None:
    """Compute semantic embeddings for concept nodes."""
    from drbrain.concept_graph.embeddings import compute_concept_embeddings

    cfg = ctx.obj["config"]
    with open_db(cfg) as db:
        count = compute_concept_embeddings(db, cfg, context=context, model_name=model_name)
    if json_output:
        typer.echo(json.dumps({"embedded": count}))
    else:
        typer.echo(f"[cg.embed] embedded {count} concepts")


@cg_app.command("neighbors")
def cg_neighbors_cmd(
    ctx: typer.Context,
    concept: str = typer.Argument(..., help="Query concept label"),
    top: int = typer.Option(10, "--top", "-k", help="Number of neighbours"),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON"),
) -> None:
    """Show the nearest concepts to a query concept by embedding similarity."""
    from drbrain.concept_graph.embeddings import nearest_neighbors

    cfg = ctx.obj["config"]
    with open_db(cfg) as db:
        results = nearest_neighbors(db, concept, k=top)
    if json_output:
        typer.echo(
            json.dumps([{"label": l, "score": round(s, 4)} for l, s in results], ensure_ascii=False)
        )
        return
    if not results:
        typer.echo(f"No embedding found for '{concept}'.")
        return
    for label, score in results:
        typer.echo(f"  {score:.4f}  {label}")


@cg_app.command("map")
def cg_map_cmd(
    ctx: typer.Context,
    output: str = typer.Option("concept_map.html", "--output", "-o", help="Output HTML path"),
) -> None:
    """Export an interactive UMAP concept map to HTML."""
    from drbrain.concept_graph.map import export_html

    cfg = ctx.obj["config"]
    with open_db(cfg) as db:
        path = export_html(db, output)
    typer.echo(f"[cg.map] wrote {path}")
