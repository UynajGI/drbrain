"""Core query commands."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from drbrain.cli._common import (
    _resolve_workspace_papers,
)
from drbrain.graph.engine import GraphEngine
from drbrain.query.tree_retrieval import query_by_structure_hybrid
from drbrain.storage.database import Database
from drbrain.storage.paths import tree_json_path

console = Console()


def seed_cmd(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
    workspace: str = typer.Option(None, "--workspace", "-w", help="Limit to workspace"),
):
    """Detect research seeds from graph patterns."""
    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])
    graph = GraphEngine()
    paper_ids = _resolve_workspace_papers(workspace)
    graph.load_from_db(db, paper_ids=paper_ids)

    seeds = graph.detect_research_seeds(db)
    db.close()

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
    db = Database(cfg["db"]["path"])
    papers = db.get_all_papers()
    db.close()

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
    db = Database(cfg["db"]["path"])
    ph_counts = 0
    if workspace:
        paper_ids = _resolve_workspace_papers(workspace)
        if paper_ids:
            ph = ",".join("?" for _ in paper_ids)
            params = tuple(paper_ids)
            papers = db.conn.execute(
                f"SELECT COUNT(*) FROM papers WHERE local_id IN ({ph})",
                params,
            ).fetchone()[0]
            uploaded = db.conn.execute(
                f"SELECT COUNT(*) FROM papers WHERE status='uploaded' AND local_id IN ({ph})",
                params,
            ).fetchone()[0]
            ph_counts = db.conn.execute(
                f"SELECT COUNT(*) FROM papers WHERE status='placeholder' AND local_id IN ({ph})",
                params,
            ).fetchone()[0]
            concepts = db.conn.execute(
                f"SELECT COUNT(*) FROM concepts WHERE local_id IN ({ph})",
                params,
            ).fetchone()[0]
            edges = db.conn.execute(
                f"SELECT COUNT(*) FROM edges WHERE source_paper IN ({ph})",
                params,
            ).fetchone()[0]
            arguments = db.conn.execute(
                f"SELECT COUNT(*) FROM arguments WHERE source_paper IN ({ph})",
                params,
            ).fetchone()[0]
            aliases = db.conn.execute("SELECT COUNT(*) FROM aliases").fetchone()[0]
            seeds = db.conn.execute("SELECT COUNT(*) FROM research_seeds").fetchone()[0]
            queue_pending = db.conn.execute(
                "SELECT COUNT(*) FROM confidence_queue WHERE status = 'pending'"
            ).fetchone()[0]
        else:
            papers = 0
            uploaded = 0
            ph_counts = 0
            concepts = 0
            edges = 0
            aliases = 0
            seeds = 0
            arguments = 0
            queue_pending = 0
        placeholders = ph_counts
    else:
        papers = db.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        uploaded = db.conn.execute(
            "SELECT COUNT(*) FROM papers WHERE status='uploaded'"
        ).fetchone()[0]
        placeholders = db.conn.execute(
            "SELECT COUNT(*) FROM papers WHERE status='placeholder'"
        ).fetchone()[0]
    concepts = db.conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
    edges = db.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    aliases = db.conn.execute("SELECT COUNT(*) FROM aliases").fetchone()[0]
    seeds = db.conn.execute("SELECT COUNT(*) FROM research_seeds").fetchone()[0]
    arguments = db.conn.execute("SELECT COUNT(*) FROM arguments").fetchone()[0]
    queue_pending = db.conn.execute(
        "SELECT COUNT(*) FROM confidence_queue WHERE status = 'pending'"
    ).fetchone()[0]
    db.close()

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
    db = Database(cfg["db"]["path"])

    paper = db.get_paper(local_id)
    if not paper:
        db.close()
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
    db.close()

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
    """Rebuild the BM25 search index."""
    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])
    from drbrain.query.bm25 import build_bm25_index

    typer.echo("Building BM25 index...")
    index, doc_ids = build_bm25_index(db, force=rebuild)
    db.close()

    if json_output:
        typer.echo(json.dumps({"documents": len(doc_ids), "indexed": True}))
    else:
        typer.echo(f"Indexed {len(doc_ids)} documents")


def query_cmd(
    ctx: typer.Context,
    text: str,
    type_filter: str = typer.Option(
        None, "--type-filter", help="Filter by concept type (Problem, Method, etc.)"
    ),
    arg_type: str = typer.Option(
        None, "--arg-type", help="Filter by argument claim type (supports, challenges, etc.)"
    ),
    year_start: int = typer.Option(None, "--year-start", help="Filter by minimum year"),
    year_end: int = typer.Option(None, "--year-end", help="Filter by maximum year"),
    min_confidence: float = typer.Option(
        None, "--min-confidence", help="Minimum confidence threshold"
    ),
    limit: int = typer.Option(20, "--limit", help="Maximum results"),
    neighbors: int = typer.Option(
        0, "--neighbors", "-n", help="Expand results by N hops of graph traversal"
    ),
    relation: str = typer.Option(
        None,
        "--relation",
        "-R",
        help="Comma-separated relation types to follow (e.g. addresses,extends,challenges)",
    ),
    direction: str = typer.Option(
        "both",
        "--direction",
        "-D",
        help="Traversal direction: forward, backward, or both",
    ),
    hybrid: bool = typer.Option(
        False, "--hybrid", help="Boost results by graph centrality (PageRank)"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output JSON array to stdout"),
    jsonl: bool = typer.Option(False, "--jsonl", help="Output JSONL stream to stdout"),
    paper: str = typer.Option(
        None,
        "--paper",
        help="Paper local_id for PageIndex tree retrieval (bypasses BM25 when set)",
    ),
    workspace: str = typer.Option(None, "--workspace", "-w", help="Limit to workspace"),
):
    """Query concepts and arguments with BM25 + filters, or use PageIndex tree retrieval."""
    cfg = ctx.obj["config"]

    # --- Tree retrieval path ---
    # Normalize: when called directly (not through typer CLI), OptionInfo is still the default
    _paper = paper if not isinstance(paper, typer.models.OptionInfo) else paper.default

    # Normalize typer defaults for direct-call compatibility
    _relation = relation if not isinstance(relation, typer.models.OptionInfo) else relation.default
    _direction = (
        direction if not isinstance(direction, typer.models.OptionInfo) else direction.default
    )
    _hybrid = hybrid if not isinstance(hybrid, typer.models.OptionInfo) else hybrid.default

    if _paper:
        papers_dir = Path(cfg["dirs"]["papers"])
        paper_dir = papers_dir / _paper
        if not paper_dir.exists():
            typer.echo(f"Paper not found: {_paper}", err=True)
            raise typer.Exit(1)
        if not tree_json_path(paper_dir).exists():
            typer.echo(f"tree.json not found for {_paper}. Run 'drbrain ingest' first.", err=True)
            raise typer.Exit(1)

        llm_models = cfg.get("llm", {}).get("models", [])
        db_path_val = cfg.get("db", {}).get("path", "data/drbrain.db")
        from drbrain.config import EmbedConfig

        embed_cfg_raw = cfg.get("embed", EmbedConfig())
        embed_cfg = (
            EmbedConfig(**embed_cfg_raw) if isinstance(embed_cfg_raw, dict) else embed_cfg_raw
        )
        sections = asyncio.run(
            query_by_structure_hybrid(text, paper_dir, Path(db_path_val), llm_models, embed_cfg)
        )

        if sections is None:
            if json_output:
                typer.echo(
                    json.dumps(
                        {"query": text, "paper": _paper, "mode": "hybrid", "sections": []},
                        ensure_ascii=False,
                    )
                )
            else:
                typer.echo(f"No relevant sections found for: {text}")
            return

        if json_output:
            typer.echo(
                json.dumps(
                    {"query": text, "paper": _paper, "mode": "hybrid", "sections": sections},
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return

        # Rich text output
        typer.echo(f"Query: {text}")
        typer.echo(f"  Paper: {_paper}")
        typer.echo("  Mode: hybrid (LLM + vector)")
        typer.echo(f"  Sections found: {len(sections)}")
        typer.echo()

        for i, sec in enumerate(sections):
            title_tag = (
                f" [{sec['node_id']}] {sec['title']}"
                if sec.get("title")
                else f" [{sec['node_id']}]"
            )
            typer.echo(f"  {title_tag}")
            content = sec["content"]
            typer.echo(content[:500] + ("..." if len(content) > 500 else ""))
            if i < len(sections) - 1:
                typer.echo()
        return

    db = Database(cfg["db"]["path"])

    # Parse and validate graph traversal flags (only when expansion is active)
    _relations: set[str] | None = None
    if neighbors > 0:
        if _relation is not None:
            _relations = {r.strip() for r in _relation.split(",") if r.strip()}
            valid_relations = {
                "addresses",
                "leaves_open",
                "points_to",
                "proposes",
                "extends",
                "replaces",
                "solves",
                "supports",
                "challenges",
                "limits",
                "constrains",
                "affiliated_with",
            }
            invalid = _relations - valid_relations
            if invalid:
                typer.echo(f"Invalid relation(s): {', '.join(sorted(invalid))}", err=True)
                typer.echo(f"Valid relations: {', '.join(sorted(valid_relations))}", err=True)
                raise typer.Exit(1)

        if _direction not in ("forward", "backward", "both"):
            typer.echo(
                f"Invalid direction '{_direction}'. Must be: forward, backward, or both",
                err=True,
            )
            raise typer.Exit(1)

    from drbrain.query.bm25 import build_bm25_index

    bm25 = build_bm25_index(db)
    results = bm25.search(
        text,
        type_filter=type_filter,
        arg_type_filter=arg_type,
        limit=limit,
        min_confidence=min_confidence,
    )

    # Post-filter by year range
    if year_start is not None or year_end is not None:
        y_start = year_start or 0
        y_end = year_end or 9999
        results = [r for r in results if y_start <= r.get("year", y_end) <= y_end]

    # Post-filter by workspace
    if workspace:
        ws_paper_ids = _resolve_workspace_papers(workspace)
        if ws_paper_ids is not None:
            results = [r for r in results if r["local_id"] in ws_paper_ids]

    # Hybrid ranking: boost by graph centrality (PageRank)
    if _hybrid and results:
        graph = GraphEngine()
        graph.load_from_db(db)
        if graph.graph.number_of_nodes() > 0:
            # Minimal PageRank — avoids scipy dependency
            g = graph.graph
            n = g.number_of_nodes()
            damping = 0.85
            pr = {node: 1.0 / n for node in g.nodes()}
            for _ in range(100):
                new_pr: dict[str, float] = {}
                for node in g.nodes():
                    rank = (1 - damping) / n
                    for pred in g.predecessors(node):
                        out_deg = g.out_degree(pred)
                        if out_deg > 0:
                            rank += damping * pr[pred] / out_deg
                    new_pr[node] = rank
                # Check convergence
                diff = sum(abs(new_pr[node] - pr[node]) for node in g.nodes())
                pr = new_pr
                if diff < 1e-6:
                    break
            # Compute percentile rank for each node
            sorted_nodes = sorted(pr.items(), key=lambda x: x[1])
            n = len(sorted_nodes)
            percentiles: dict[str, float] = {}
            for rank, (node, _) in enumerate(sorted_nodes):
                percentiles[node] = rank / (n - 1) if n > 1 else 0.5
            # Apply multiplicative boost [1.0, 2.0]
            for r in results:
                node_id = r["local_id"]
                boost = 1.0 + percentiles.get(node_id, 0.0)
                r["score"] = round(r["score"] * boost, 4)
                r["_hybrid_boost"] = round(boost, 3)
            # Re-sort by boosted score
            results.sort(key=lambda r: r["score"], reverse=True)
        graph.graph = None

    # Map BM25 concept results to use labels as local_id for graph traversal
    concept_types = {"Problem", "Method", "Conclusion", "Debate", "Gap", "Actor"}
    for r in results:
        if r["type"] in concept_types:
            r["_paper_id"] = r["local_id"]
            r["local_id"] = r["label"]

    # Expand by graph traversal
    if neighbors > 0 and results:
        graph = GraphEngine()
        graph.load_from_db(db)
        # Seed from top-scoring BM25 result(s) only, so lower-scored hits
        # become discoverable via traverse() rather than being pre-seeded
        max_score = max(r["score"] for r in results)
        seed_ids = {r["local_id"] for r in results if r["score"] >= max_score}

        traverse_results = graph.traverse(
            start_nodes=seed_ids,
            hops=neighbors,
            relations=_relations,
            direction=_direction,
        )

        seen_ids = seed_ids.copy()
        for tr in traverse_results:
            if tr.target in seen_ids:
                continue
            seen_ids.add(tr.target)

            # Resolve node type from DB
            node_type = "Unknown"
            row = db.conn.execute(
                "SELECT type FROM concepts WHERE label = ? LIMIT 1", (tr.target,)
            ).fetchone()
            if row:
                node_type = row[0]
            else:
                paper = db.get_paper(tr.target)
                if paper:
                    node_type = "Paper"

            if node_type == "Paper":
                paper = db.get_paper(tr.target)
                label = paper["title"] if paper else tr.target
                text = paper.get("abstract", "") if paper else ""
                year = paper.get("year") if paper else None
            else:
                label = tr.target
                text = ""
                year = None

            results.append(
                {
                    "local_id": tr.target,
                    "type": node_type,
                    "label": label,
                    "text": text,
                    "year": year,
                    "score": 0.0,
                    "_via_graph": True,
                    "_source_seed": tr.source,
                    "_distance": tr.distance,
                    "_path": [
                        {
                            "src": s.src,
                            "relation": s.relation,
                            "dst": s.dst,
                            "hop": s.hop,
                        }
                        for s in tr.path
                    ],
                }
            )

        graph.graph = None  # Free memory
        db.close()
    else:
        db.close()

    # JSON output modes
    if json_output:
        typer.echo(json.dumps(results, indent=2, ensure_ascii=False, default=str))
        return

    if jsonl:
        for r in results:
            typer.echo(json.dumps(r, ensure_ascii=False, default=str))
        return

    if not results:
        typer.echo(f"No results for: {text}")
        return

    typer.echo(f"Query: {text}")
    filters = []
    if type_filter:
        filters.append(f"type={type_filter}")
    if arg_type:
        filters.append(f"arg_type={arg_type}")
    if year_start or year_end:
        filters.append(f"year={year_start or '...'}-{year_end or '...'}")
    if min_confidence is not None:
        filters.append(f"min_confidence={min_confidence}")
    if neighbors:
        if _relation:
            rel_str = ",".join(sorted(_relations))
            filters.append(f"neighbors={neighbors}, relation={rel_str}")
        else:
            filters.append(f"neighbors={neighbors}")
        if _direction and _direction != "both":
            filters.append(f"direction={_direction}")
    if filters:
        typer.echo(f"  Filters: {', '.join(filters)}")
    typer.echo(f"  Results: {len(results)}")
    for i, r in enumerate(results, 1):
        extra = ""
        if r["type"] == "Argument":
            extra = f" [{r.get('arg_type', '')}]"
        if r.get("_via_graph"):
            path_parts = [r["_source_seed"]]
            for step in r.get("_path", []):
                path_parts.append(step["relation"])
                path_parts.append(step["dst"])
            path_str = " -> ".join(path_parts)
            extra += f" [graph: {path_str}]"
        boost_str = f", boost: {r['_hybrid_boost']:.1f}x" if "_hybrid_boost" in r else ""
        year_str = f" ({r.get('year', '?')})" if r.get("year") else ""
        conf_str = f", confidence: {r['confidence']:.2f}" if "confidence" in r else ""
        typer.echo(
            f"  {i}. [{r['type']}] {r['label']}{extra} (score: {r['score']:.3f}{boost_str}, paper: {r['local_id']}{year_str}{conf_str})"
        )
