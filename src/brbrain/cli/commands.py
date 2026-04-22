"""Full CLI command implementations."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from brbrain.config import load_config
from brbrain.parser.mineru_parser import extract_pdf
from brbrain.extractor.concept import extract_concepts
from brbrain.extractor.canonical import AliasTable
from brbrain.dedup.resolver import DedupEngine, PaperIDs
from brbrain.storage.database import Database
from brbrain.graph.engine import GraphEngine
from brbrain.report.generator import PaperReport

console = Console()


def ingest_cmd(pdf_path: str, json_flag: bool = False):
    """Full ingest pipeline: parse -> identify -> extract -> validate -> queue -> align -> ingest -> expand -> report."""
    cfg = load_config()
    db = Database(cfg["db"]["path"])
    graph = GraphEngine()
    graph.load_from_db(db)
    dedup = DedupEngine(db)
    alias_table = AliasTable()

    queue_cfg = cfg.get("queue", {})
    weak_threshold = queue_cfg.get("weak_threshold", 0.7)
    auto_accept = queue_cfg.get("auto_accept", 0.9)

    # Stage 1: Parse
    typer.echo(f"Parsing: {pdf_path}")
    try:
        parsed = extract_pdf(pdf_path, cfg)
    except Exception as e:
        typer.echo(f"Error parsing PDF: {e}", err=True)
        raise typer.Exit(1)
    typer.echo(f"  Title: {parsed.title}")
    typer.echo(f"  Year: {parsed.year}")
    typer.echo(f"  Sections: {len(parsed.text_blocks)} high-signal blocks")

    # Stage 2: Identify
    ids = PaperIDs(doi=parsed.doi, arxiv=parsed.arxiv)
    local_id = dedup.resolve(ids, title=parsed.title, year=parsed.year)
    is_new = local_id is None
    if is_new:
        local_id = f"p{uuid.uuid4().hex[:6]}"
        db.insert_paper(local_id, parsed.title, parsed.year, "uploaded")
        db.insert_paper_ids(local_id, doi=ids.doi, arxiv=ids.arxiv)
        db.commit()
        typer.echo(f"  [new] {local_id}")
    else:
        db.upgrade_placeholder(local_id)
        db.commit()
        typer.echo(f"  [upgrade] {local_id}")

    # Stage 3: Extract
    typer.echo("  Extracting concepts + arguments...")
    llm_models = cfg.get("llm", {}).get("models", [])
    if not llm_models:
        typer.echo("Error: no LLM models configured. Run: drbrain setup", err=True)
        raise typer.Exit(1)

    import asyncio
    full_text = "\n\n".join(parsed.text_blocks)
    concepts = asyncio.run(extract_concepts(full_text, llm_models))
    if concepts is None:
        typer.echo("Error: LLM extraction failed. All models exhausted.", err=True)
        raise typer.Exit(1)

    # Stage 3.5: Validate
    from brbrain.validator.schema import validate_extraction
    typer.echo("  Validating extraction...")
    concept_data = {
        "problems": concepts.problems, "methods": concepts.methods,
        "conclusions": concepts.conclusions, "debates": concepts.debates,
        "gaps": concepts.gaps, "actors": concepts.actors,
    }
    validation = validate_extraction(concept_data, concepts.relations)
    typer.echo(f"  Valid items: {len(validation['valid'])}")
    if validation["rejected"]:
        typer.echo(f"  Rejected: {len(validation['rejected'])}")
        for r in validation["rejected"]:
            typer.echo(f"    [yellow]{r['reason']}[/yellow]")

    valid_relations = [r["detail"] for r in validation["valid"] if r["type"] == "relation"]

    # Stage 3.6: Queue low-confidence concepts
    from brbrain.extractor.queue import route_item
    typed_count = 0
    queued_count = 0
    weak_count = 0
    all_items = [
        ("Problem", concepts.problems), ("Method", concepts.methods),
        ("Conclusion", concepts.conclusions), ("Debate", concepts.debates),
        ("Gap", concepts.gaps), ("Actor", concepts.actors),
    ]
    for ctype, items in all_items:
        for item in items:
            label = item.get("label", "")
            conf = item.get("confidence", 1.0)
            routing = route_item(db, local_id, "concept",
                               {"label": label, "type": ctype}, conf,
                               weak_threshold, auto_accept)
            if routing["action"] == "queued":
                queued_count += 1
            elif routing["action"] == "weak":
                weak_count += 1
            typed_count += 1

    # Stage 4: Align + Stage 5: Ingest
    for ctype, items in all_items:
        for item in items:
            label = item.get("label", "")
            conf = item.get("confidence", 1.0)
            if conf >= weak_threshold:
                alias_table.get_or_create(label)
                db.insert_concept(local_id, ctype, label, conf, year=parsed.year)

    for rel in valid_relations:
        db.insert_edge(rel["head"], rel["tail"], rel["rel"], local_id)

    # Ingest arguments
    from brbrain.extractor.argument import validate_arguments
    valid_args, rejected_args = validate_arguments(concepts.arguments)
    for arg in valid_args:
        db.insert_argument(
            local_id, arg.claim, arg.claim_type, arg.target, arg.target_type,
            arg.evidence_type, arg.evidence_detail, arg.confidence,
        )

    db.commit()
    typer.echo(f"  Concepts inserted: {typed_count}")
    typer.echo(f"  Arguments inserted: {len(valid_args)}")
    if queued_count:
        typer.echo(f"  Queued for review: {queued_count}")
    if weak_count:
        typer.echo(f"  Weak (ingested with marker): {weak_count}")
    if rejected_args:
        typer.echo(f"  Arguments rejected: {len(rejected_args)}")

    # Stage 6: Expand
    typer.echo("  Expanding citations...")
    from brbrain.extractor.citation import expand_citations
    refs, cits = expand_citations(db, local_id, cfg)
    refs_in = sum(1 for r in refs if r.in_graph)
    cits_in = sum(1 for c in cits if c.in_graph)
    typer.echo(f"  References: {len(refs)} ({refs_in} in graph)")
    typer.echo(f"  Citations: {len(cits)} ({cits_in} in graph)")

    # Stage 8: Closure
    typer.echo("  Running rule closure...")
    graph.load_from_db(db)
    inferred = graph.closure()
    for edge in inferred:
        db.insert_edge(edge["src"], edge["dst"], edge["relation"], local_id)
    db.commit()
    typer.echo(f"  Inferred edges: {len(inferred)}")

    # Stage 9: Report
    report = PaperReport(
        local_id=local_id, title=parsed.title, year=parsed.year,
        ids={"doi": parsed.doi, "arxiv": parsed.arxiv},
        status="uploaded",
        concepts=concepts.to_dict(),
        arguments=[a.to_dict() for a in valid_args],
        references=refs,
        citations=cits,
        validation={
            "items_rejected": len(validation["rejected"]),
            "items_queued": queued_count,
            "tbox_violations": [r["reason"] for r in validation["rejected"]],
            "rbox_violations": [],
        },
    )
    report_dir = Path(cfg["dirs"]["reports"])
    report_path = report.save(report_dir)
    typer.echo(f"  Report saved: {report_path}")

    summary = report.summary
    if summary["graph_coverage"] < 0.3:
        typer.echo(f"\n  [bold yellow]Warning: Low coverage ({summary['graph_coverage']:.1%}). Consider ingesting missing references.[/bold yellow]")

    db.close()
    typer.echo(f"\nDone: {local_id}")


def expand_cmd(local_id: str, depth: int = 2):
    """Expand a paper's citation neighborhood."""
    cfg = load_config()
    db = Database(cfg["db"]["path"])
    paper = db.get_paper(local_id)
    if not paper:
        typer.echo(f"Paper not found: {local_id}", err=True)
        raise typer.Exit(1)

    from brbrain.extractor.citation import expand_citations
    refs, cits = expand_citations(db, local_id, cfg)
    db.close()

    typer.echo(f"Expanded: {paper['title']}")
    typer.echo(f"  References: {len(refs)} ({sum(1 for r in refs if r.in_graph)} in graph)")
    typer.echo(f"  Citations: {len(cits)} ({sum(1 for c in cits if c.in_graph)} in graph)")


def report_cmd(local_id: str):
    """Display single-paper report."""
    cfg = load_config()
    report_dir = Path(cfg["dirs"]["reports"])
    report_path = report_dir / f"{local_id}.json"
    if not report_path.exists():
        typer.echo(f"No report found for {local_id}. Run: drbrain ingest or drbrain expand", err=True)
        raise typer.Exit(1)

    data = json.loads(report_path.read_text())
    s = data["summary"]
    typer.echo(f"\nReport: {data['paper']['title']}")
    typer.echo(f"  Status: {data['paper']['status']}")
    typer.echo(f"  Coverage: {s['graph_coverage']:.1%}")
    typer.echo(f"  References in graph: {s['refs_in_graph']}/{s['total_refs']}")
    typer.echo(f"  Citations in graph: {s['cits_in_graph']}/{s['total_cits']}")

    concepts = data["concepts"]
    for ctype in ["problems", "methods", "conclusions", "debates", "gaps", "actors"]:
        if concepts.get(ctype):
            typer.echo(f"  {ctype}: {len(concepts[ctype])}")

    if data["boundary_alert"].get("low_coverage"):
        typer.echo("  [bold yellow]Alert: Low coverage - consider expanding citation network[/bold yellow]")


def closure_cmd():
    """Run rule-based closure on the full graph."""
    cfg = load_config()
    db = Database(cfg["db"]["path"])
    graph = GraphEngine()
    graph.load_from_db(db)

    inferred = graph.closure()
    typer.echo(f"Inferred edges: {len(inferred)}")
    for edge in inferred:
        typer.echo(f"  {edge['src']} --[{edge['relation']}]--> {edge['dst']} (via {edge.get('via', 'unknown')})")

    # Persist new edges
    for edge in inferred:
        db.insert_edge(edge["src"], edge["dst"], edge["relation"], "closure")
    db.commit()
    db.close()


def seed_cmd():
    """Detect research seeds from graph patterns."""
    cfg = load_config()
    db = Database(cfg["db"]["path"])
    graph = GraphEngine()
    graph.load_from_db(db)

    seeds = graph.detect_research_seeds()
    typer.echo(f"Research seeds found: {len(seeds)}")

    for seed in seeds:
        typer.echo(f"  [{seed['type']}] {seed['node']}: {seed['signal']}")

    db.close()


def list_cmd():
    """List all papers in database."""
    cfg = load_config()
    db = Database(cfg["db"]["path"])
    papers = db.get_all_papers()
    db.close()

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


def stats_cmd():
    """Database statistics."""
    cfg = load_config()
    db = Database(cfg["db"]["path"])
    papers = db.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    uploaded = db.conn.execute("SELECT COUNT(*) FROM papers WHERE status='uploaded'").fetchone()[0]
    placeholders = db.conn.execute("SELECT COUNT(*) FROM papers WHERE status='placeholder'").fetchone()[0]
    concepts = db.conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
    edges = db.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    aliases = db.conn.execute("SELECT COUNT(*) FROM aliases").fetchone()[0]
    seeds = db.conn.execute("SELECT COUNT(*) FROM research_seeds").fetchone()[0]
    arguments = db.conn.execute("SELECT COUNT(*) FROM arguments").fetchone()[0]
    queue_pending = db.conn.execute(
        "SELECT COUNT(*) FROM confidence_queue WHERE status = 'pending'"
    ).fetchone()[0]
    db.close()

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


def query_cmd(text: str, type_filter: str = None, limit: int = 20):
    """Query concepts with BM25 + type filter."""
    cfg = load_config()
    db = Database(cfg["db"]["path"])

    from brbrain.query.bm25 import build_bm25_index
    bm25 = build_bm25_index(db)
    results = bm25.search(text, type_filter=type_filter, limit=limit)
    db.close()

    if not results:
        typer.echo(f"No results for: {text}")
        return

    typer.echo(f"Query: {text}")
    if type_filter:
        typer.echo(f"  Type filter: {type_filter}")
    typer.echo(f"  Results: {len(results)}")
    for i, r in enumerate(results, 1):
        typer.echo(f"  {i}. [{r['type']}] {r['label']} (score: {r['score']:.3f}, paper: {r['local_id']})")


def export_cmd(format: str = "json"):
    """Export graph data."""
    cfg = load_config()
    db = Database(cfg["db"]["path"])
    graph = GraphEngine()
    graph.load_from_db(db)

    if format == "json":
        data = {
            "nodes": [
                {"id": n, **d} for n, d in graph.graph.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, **d}
                for u, v, d in graph.graph.edges(data=True)
            ],
        }
        typer.echo(json.dumps(data, indent=2, default=str))
    elif format == "graphml":
        import networkx as nx
        typer.echo(nx.generate_graphml(graph.graph))
    else:
        typer.echo(f"Unsupported format: {format}", err=True)
        raise typer.Exit(1)

    db.close()


def queue_cmd():
    """List all pending confidence queue items."""
    cfg = load_config()
    db = Database(cfg["db"]["path"])
    pending = db.get_queue_pending()
    db.close()

    if not pending:
        typer.echo("Queue is empty.")
        return

    table = Table(title="Confidence Queue")
    table.add_column("ID", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Data")
    table.add_column("Confidence", justify="right")
    table.add_column("Paper")
    for item in pending:
        data = json.loads(item["item_data"])
        label = data.get("label", "N/A")
        item_type = data.get("type", item["item_type"])
        table.add_row(
            str(item["queue_id"]),
            item["item_type"],
            f"{item_type}: {label}",
            f"{item['confidence']:.2f}",
            item["source_paper"],
        )
    console.print(table)


def queue_resolve_cmd(queue_id: int, accept: bool = False, reject: bool = False):
    """Resolve a queue item: accept or reject."""
    if accept and reject:
        typer.echo("Error: cannot both accept and reject", err=True)
        raise typer.Exit(1)
    if not accept and not reject:
        typer.echo("Error: specify --accept or --reject", err=True)
        raise typer.Exit(1)

    cfg = load_config()
    db = Database(cfg["db"]["path"])

    from brbrain.extractor.queue import resolve_accept, resolve_reject
    if accept:
        resolve_accept(db, queue_id)
        typer.echo(f"Queue item {queue_id} accepted.")
    else:
        resolve_reject(db, queue_id)
        typer.echo(f"Queue item {queue_id} rejected.")

    db.close()


def timeline_cmd(concept: str):
    """Show concept evolution over time."""
    cfg = load_config()
    db = Database(cfg["db"]["path"])
    evolution = db.get_concept_evolution(concept)

    if not evolution:
        typer.echo(f"No data for concept: {concept}")
        db.close()
        return

    row = db.conn.execute(
        "SELECT type FROM concepts WHERE label = ? LIMIT 1", (concept,)
    ).fetchone()
    ctype = row[0] if row else "unknown"

    typer.echo(f"\nConcept: {concept} ({ctype})")

    from datetime import datetime
    current_year = datetime.now().year

    for entry in evolution:
        year = entry["year"]
        count = entry["count"]
        avg_conf = entry["avg_conf"]

        if year == evolution[0]["year"]:
            signal = "first appeared"
        elif year >= current_year - 2:
            signal = "recent"
        else:
            signal = ""

        line = f"  {year}: {count} paper{'s' if count > 1 else ''} (avg confidence {avg_conf:.2f})"
        if signal:
            line += f" — {signal}"
        typer.echo(line)

    signals = db.detect_evolution_signals()
    matching = [s for s in signals if s["label"] == concept]
    if matching:
        typer.echo(f"Status: {matching[0]['signal'].upper()}")

    db.close()
