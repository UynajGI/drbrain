"""Core query commands."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

from drbrain.cli._common import (
    _resolve_workspace_papers,
    open_db,
)
from drbrain.graph.engine import GraphEngine

console = Console()


def seed_cmd(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
    workspace: str = typer.Option(None, "--workspace", "-w", help="Limit to workspace"),
):
    """Detect research seeds from graph patterns."""
    cfg = ctx.obj["config"]
    with open_db(cfg) as db:
        graph = GraphEngine()
        paper_ids = _resolve_workspace_papers(workspace)
        graph.load_from_db(db, paper_ids=paper_ids)

        seeds = graph.detect_research_seeds(db)

    if json_output:
        typer.echo(json.dumps(seeds, indent=2, ensure_ascii=False, default=str))
        return

    typer.echo(f"Research seeds found: {len(seeds)}")

    for seed in seeds:
        typer.echo(f"  [{seed['type']}] {seed['concept']}: {seed['description']}")


def list_cmd(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
):
    """List all papers in database."""
    cfg = ctx.obj["config"]
    with open_db(cfg) as db:
        papers = db.get_all_papers()

    if json_output:
        typer.echo(json.dumps(papers, indent=2, ensure_ascii=False, default=str))
        return

    if not papers:
        typer.echo("No papers in database. Run: drbrain ingest <paper.pdf>")
        return

    table = Table(title="Papers")
    table.add_column("ID", style="cyan")
    table.add_column("Title", style="green")
    table.add_column("Year", justify="right")
    table.add_column("Status")
    for p in papers:
        table.add_row(p["local_id"], p["title"], str(p["year"] or ""), p["status"])
    console.print(table)


def stats_cmd(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
    workspace: str = typer.Option(None, "--workspace", "-w", help="Limit to workspace"),
):
    """Database statistics."""
    cfg = ctx.obj["config"]
    with open_db(cfg) as db:
        paper_ids_filter = None
        if workspace:
            paper_ids_filter = _resolve_workspace_papers(workspace)
        s = db.get_stats(paper_ids=paper_ids_filter)  # type: ignore[arg-type]  # pre-existing: see mypy debt

    papers = s["papers"]
    uploaded = s["uploaded"]
    placeholders = s["placeholders"]
    concepts = s["concepts"]
    edges = s["edges"]
    aliases = s["aliases"]
    seeds = s["research_seeds"]
    arguments = s["arguments"]
    queue_pending = s["queue_pending"]

    data = {
        "papers": papers,
        "uploaded": uploaded,
        "placeholders": placeholders,
        "concepts": concepts,
        "edges": edges,
        "aliases": aliases,
        "research_seeds": seeds,
        "arguments": arguments,
        "queue_pending": queue_pending,
    }

    if json_output:
        typer.echo(json.dumps(data, indent=2, default=str))
        return

    table = Table(title="DrBrain Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right", style="green")
    table.add_row("Total papers", str(papers))
    table.add_row("Uploaded", str(uploaded))
    table.add_row("Placeholders", str(placeholders))
    table.add_row("Concepts", str(concepts))
    table.add_row("Arguments", str(arguments))
    table.add_row("Edges", str(edges))
    table.add_row("Aliases", str(aliases))
    table.add_row("Research seeds", str(seeds))
    table.add_row("Queue pending", str(queue_pending))
    console.print(table)


def show_cmd(
    ctx: typer.Context,
    local_id: str = typer.Argument(..., help="Paper local_id"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Show detailed view of a single paper."""
    cfg = ctx.obj["config"]
    with open_db(cfg) as db:
        paper = db.get_paper(local_id)
        if not paper:
            typer.echo(f"Paper not found: {local_id}", err=True)
            raise typer.Exit(1)

        concepts = db.get_concepts_by_paper(local_id)
        arguments = db.get_arguments_by_paper(local_id)
        edges_out = db.conn.execute(
            "SELECT relation, dst_id FROM edges WHERE src_id = ?", (local_id,)
        ).fetchall()
        edges_in = db.conn.execute(
            "SELECT src_id, relation FROM edges WHERE dst_id = ?", (local_id,)
        ).fetchall()

    if json_output:
        typer.echo(
            json.dumps(
                {
                    "paper": paper,
                    "concepts": concepts,
                    "arguments": arguments,
                    "edges": {
                        "outgoing": [{"relation": r[0], "target": r[1]} for r in edges_out],
                        "incoming": [{"source": r[0], "relation": r[1]} for r in edges_in],
                    },
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )
        return

    typer.echo(f"\n[bold]{paper['title']}[/bold]")
    typer.echo(
        f"  ID: {paper['local_id']}  |  Year: {paper.get('year', '?')}  "
        f"|  Type: {paper.get('paper_type', '?')}  |  Status: {paper.get('status', '?')}"
    )
    if paper.get("journal"):
        typer.echo(f"  Journal: {paper['journal']}")
    if paper.get("doi"):
        typer.echo(f"  DOI: {paper['doi']}")
    if paper.get("abstract"):
        typer.echo(f"\n  Abstract: {paper['abstract'][:500]}")
    if paper.get("citation_count"):
        typer.echo(f"  Citations: {paper['citation_count']}")

    if concepts:
        typer.echo(f"\n[bold]Concepts ({len(concepts)})[/bold]")
        by_type: dict[str, list] = {}
        for c in concepts:
            by_type.setdefault(c["type"], []).append(c["label"])
        for ct, labels in by_type.items():
            typer.echo(f"  {ct}: {', '.join(labels[:10])}")

    if arguments:
        typer.echo(f"\n[bold]Arguments ({len(arguments)})[/bold]")
        for a in arguments[:10]:
            typer.echo(f"  [{a['claim_type']}] {a['claim'][:120]} -> {a['target_label']}")

    if edges_out:
        typer.echo(f"\n[bold]Outgoing edges ({len(edges_out)})[/bold]")
        for r in edges_out[:15]:
            typer.echo(f"  --{r[0]}--> {r[1]}")
    if edges_in:
        typer.echo(f"\n[bold]Incoming edges ({len(edges_in)})[/bold]")
        for r in edges_in[:15]:
            typer.echo(f"  {r[0]} --{r[1]}--> {paper['local_id']}")

    typer.echo()


def index_cmd(
    ctx: typer.Context,
    rebuild: bool = typer.Option(False, "--rebuild", help="Force full rebuild"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Rebuild the BM25 search index.

    By default incremental: skips rebuild if no paper changed since the last
    successful index run. Use --rebuild to force a full rebuild.
    """
    cfg = ctx.obj["config"]
    with open_db(cfg) as db:
        from drbrain.query.bm25 import build_bm25_index

        # Incremental check: skip if nothing changed since last index run
        if not rebuild:
            last_run = db.get_last_run("index")
            max_ts = db.get_max_paper_timestamp()
            if last_run is not None and (max_ts is None or max_ts <= last_run):
                count = db.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
                if json_output:
                    typer.echo(
                        json.dumps({"documents": count, "indexed": False, "up_to_date": True})
                    )
                else:
                    typer.echo(f"Index up to date ({count} documents, no changes since last run)")
                return

        typer.echo("Building BM25 index...")
        index = build_bm25_index(db)
        doc_count = len(index._documents)
        db.set_last_run("index")
        db.commit()

    if json_output:
        typer.echo(json.dumps({"documents": doc_count, "indexed": True}))
    else:
        typer.echo(f"Indexed {doc_count} documents")


def query_cmd(
    ctx: typer.Context,
    text: str,
    limit: int = typer.Option(20, "--limit", help="Maximum results"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON array to stdout"),
    jsonl: bool = typer.Option(False, "--jsonl", help="Output JSONL stream to stdout"),
):
    """Query papers/sections via the LlamaIndex fusion retriever (T9: sole engine).

    Section-level hits with paper/node/source back-links. Requires
    ``llamaindex.enabled: true`` and a built index (``drbrain rag index``).
    The legacy BM25 + graph-traversal / PageIndex ``--paper`` paths were
    removed in T9 (终态清理, design §1 替换清单).
    """
    cfg = ctx.obj["config"]

    from drbrain.rag.engine import resolve_engine

    if resolve_engine(cfg, "llamaindex") != "llamaindex":
        typer.echo(
            "[query] llamaindex engine unavailable: set `llamaindex.enabled: true` "
            "in config.yaml and run `drbrain rag index` to build the index",
            err=True,
        )
        raise typer.Exit(1)

    _limit = int(limit.default) if isinstance(limit, typer.models.OptionInfo) else limit
    _json_output = (
        json_output.default if isinstance(json_output, typer.models.OptionInfo) else json_output
    )
    _jsonl = jsonl.default if isinstance(jsonl, typer.models.OptionInfo) else jsonl
    try:
        _query_llamaindex_cli(cfg, text, _limit, _json_output, _jsonl)
    except Exception as exc:
        typer.echo(f"[query] llamaindex query failed: {exc}", err=True)
        raise typer.Exit(1)


def fsearch_cmd(
    ctx: typer.Context,
    query: list[str] = typer.Argument(..., help="Search query terms"),
    arxiv: bool = typer.Option(False, "--arxiv", help="Also search arXiv"),
    arxiv_only: bool = typer.Option(False, "--arxiv-only", help="Search arXiv only"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results per source"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
):
    """Federated search — local library + arXiv with ingested annotation."""
    from drbrain.services.fsearch import (
        _merge_with_local_status,
        search_arxiv,
        search_local,
    )

    query_str = " ".join(query)
    output: dict = {"query": query_str, "local": [], "arxiv": []}

    # Local search
    if not arxiv_only:
        cfg = ctx.obj["config"]
        db_path = cfg.get("db", {}).get("path", "data/drbrain.db")
        results = search_local(db_path, query_str, limit=limit)
        output["local"] = results
        if not json_output:
            typer.echo(f'── Local library: "{query_str}" ──')
            if not results:
                typer.echo("  No results.")
            for i, r in enumerate(results, 1):
                authors = r.get("authors", "?")
                year = f" ({r['year']})" if r.get("year") else ""
                typer.echo(f"  [{i}] {r['title']}{year} — {authors}")
            typer.echo()

    # arXiv search
    if arxiv or arxiv_only:
        if not json_output:
            typer.echo(f'── arXiv: "{query_str}" ──')
        try:
            arxiv_results = search_arxiv(query_str, max_results=limit)
        except Exception:
            typer.echo("  arXiv search failed (network or parse error).", err=True)
            arxiv_results = []

        if not arxiv_only:
            # Cross-reference with local library
            cfg = ctx.obj["config"]
            db_path = cfg.get("db", {}).get("path", "data/drbrain.db")
            in_lib_dois: set[str] = set()
            in_lib_arxiv_ids: set[str] = set()
            try:
                # Simple cross-ref: check paper_ids table
                from pathlib import Path

                from drbrain.services.fsearch import _normalize_arxiv_ref
                from drbrain.storage.connection import connect_wal

                if Path(db_path).exists():
                    conn = connect_wal(db_path)
                    paper_ids_rows = conn.execute(
                        "SELECT doi, arxiv FROM paper_ids WHERE doi != '' OR arxiv != ''"
                    ).fetchall()
                    for row in paper_ids_rows:
                        if row[0]:
                            in_lib_dois.add(row[0].lower())
                        if row[1]:
                            in_lib_arxiv_ids.add(_normalize_arxiv_ref(row[1]))
                    conn.close()
            except Exception:
                pass

            arxiv_results = _merge_with_local_status(arxiv_results, in_lib_dois, in_lib_arxiv_ids)

        if not json_output and not arxiv_results:
            typer.echo("  No results.")

        output["arxiv"] = [
            {
                "title": r["title"],
                "authors": r.get("authors", []),
                "year": r.get("year"),
                "doi": r.get("doi", ""),
                "arxiv_id": r.get("arxiv_id", ""),
                "ingested": r.get("ingested", False),
            }
            for r in arxiv_results
        ]

        if not json_output:
            for i, r in enumerate(arxiv_results, 1):
                authors = r.get("authors", [])
                first = (authors[0] if authors else "?") + (" et al." if len(authors) > 1 else "")
                doi = r.get("doi", "")
                arxiv_id = r.get("arxiv_id", "")
                ingested = "[ingested]" if r.get("ingested") else ""
                typer.echo(f"  [{i}] [{r.get('year', '?')}] {r['title']} {ingested}")
                detail = f"       {first}"
                if arxiv_id:
                    detail += f" | arxiv:{arxiv_id}"
                if doi:
                    detail += f" | doi:{doi}"
                typer.echo(detail)

    if json_output:
        typer.echo(json.dumps(output, ensure_ascii=False, indent=2, default=str))


def search_cmd(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="Search query string"),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum results"),
    type: str = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter by document type (Problem, Method, Conclusion, Gap, Debate, Actor, Paper, Argument)",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
):
    """Quick local BM25 keyword search over papers, concepts, and arguments."""
    cfg = ctx.obj["config"]
    with open_db(cfg) as db:
        from drbrain.query.bm25 import build_bm25_index

        bm25 = build_bm25_index(db)
        results = bm25.search(
            query,
            type_filter=type,
            limit=limit,
        )

    if not results:
        if json_output:
            typer.echo(json.dumps({"query": query, "results": []}))
        else:
            typer.echo(f"No results for: {query}")
        return

    if json_output:
        typer.echo(json.dumps(results, indent=2, ensure_ascii=False, default=str))
        return

    typer.echo(f'Search: "{query}" — {len(results)} results')
    for i, r in enumerate(results, 1):
        extra = ""
        if r["type"] == "Argument":
            extra = f" [{r.get('arg_type', '')}]"
        year_str = f" ({r.get('year', '?')})" if r.get("year") else ""
        conf_str = f", conf: {r['confidence']:.2f}" if "confidence" in r else ""
        typer.echo(
            f"  {i}. [{r['type']}] {r['label']}{extra}"
            f" (score: {r['score']:.3f}, paper: {r['local_id']}{year_str}{conf_str})"
        )


def hybrid_cmd(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="Natural language query"),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum results"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
):
    """Hybrid retrieval (BM25 + vector fused via RRF) through LlamaIndex (T9: sole engine).

    Paper-level hits with per-leg ``sources``. Requires ``llamaindex.enabled:
    true`` and a built index; the legacy ``hybrid_search`` path was removed in
    T9 (终态清理, design §1 替换清单).
    """
    cfg = ctx.obj["config"]

    from drbrain.rag.engine import resolve_engine

    if resolve_engine(cfg, "llamaindex") != "llamaindex":
        typer.echo(
            "[hybrid] llamaindex engine unavailable: set `llamaindex.enabled: true` "
            "in config.yaml and run `drbrain rag index` to build the index",
            err=True,
        )
        raise typer.Exit(1)

    _limit = int(limit.default) if isinstance(limit, typer.models.OptionInfo) else limit
    _json_output = (
        json_output.default if isinstance(json_output, typer.models.OptionInfo) else json_output
    )
    try:
        _hybrid_llamaindex_cli(cfg, query, _limit, _json_output)
    except Exception as exc:
        typer.echo(f"[hybrid] llamaindex retrieval failed: {exc}", err=True)
        raise typer.Exit(1)


def _query_llamaindex_cli(
    cfg: dict, query: str, limit: int, json_output: bool, jsonl: bool
) -> None:
    """Run ``query --engine llamaindex``: fusion retrieval, section-level hits.

    Output shape is comparable to the legacy ``query`` command: JSON emits
    ``{"query", "engine", "results"}`` with ``{paper_id, node_id, title,
    score, sources}`` rows; plain mode prints the same rows as text.
    """
    from llama_index.core.schema import QueryBundle

    from drbrain.rag.engine import build_hybrid_retriever, extract_sources

    with open_db(cfg) as db:
        retriever = build_hybrid_retriever(cfg, db, top_k=limit)
        if retriever is None:
            raise RuntimeError("llamaindex fusion retriever unavailable (no index built yet?)")
        nodes = retriever.retrieve(QueryBundle(query_str=query))
    rows = extract_sources(nodes)[: int(limit)]

    if json_output:
        typer.echo(
            json.dumps(
                {"query": query, "engine": "llamaindex", "results": rows},
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if jsonl:
        for r in rows:
            typer.echo(json.dumps(r, ensure_ascii=False, default=str))
        return
    if not rows:
        typer.echo(f"No results for: {query}")
        return

    typer.echo(f"Query: {query}")
    typer.echo("  Engine: llamaindex")
    typer.echo(f"  Results: {len(rows)}")
    for i, r in enumerate(rows, 1):
        sources = ", ".join(r.get("sources") or [])
        title = r.get("title") or r.get("node_id") or ""
        score = r.get("score")
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
        typer.echo(
            f"  {i}. {title} (score: {score_str}, paper: {r.get('paper_id')}, sources: {sources})"
        )


def _hybrid_llamaindex_cli(cfg: dict, query: str, limit: int, json_output: bool) -> None:
    """Run ``hybrid --engine llamaindex``: fusion retriever, paper-level hits.

    Output shape is comparable to the legacy ``hybrid`` command: JSON emits
    ``{"query", "engine", "results"}`` with ``{paper_id, title, score,
    sections, sources, rank}`` rows (the legacy JSON carries ``{paper_id,
    score, rank, source, payload, metadata}`` — both are paper-level, so the
    two engines are directly comparable).
    """
    from llama_index.core.schema import QueryBundle

    from drbrain.rag.engine import build_hybrid_retriever, nodes_to_paper_results

    with open_db(cfg) as db:
        retriever = build_hybrid_retriever(cfg, db, top_k=limit)
        if retriever is None:
            raise RuntimeError("llamaindex fusion retriever unavailable (no index built yet?)")
        nodes = retriever.retrieve(QueryBundle(query_str=query))
    results = nodes_to_paper_results(nodes, top_k=limit)

    if json_output:
        typer.echo(
            json.dumps(
                {"query": query, "engine": "llamaindex", "results": results},
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if not results:
        typer.echo(f"No results for: {query}")
        return

    typer.echo(f'Hybrid search (llamaindex): "{query}" — {len(results)} papers')
    for hit in results:
        sources = ", ".join(hit.get("sources") or [])
        typer.echo(
            f"  {hit['rank']}. {hit['paper_id']}  (rrf: {hit['score']:.4f}, "
            f"sections: {hit['sections']}, sources: {sources})"
        )
