"""Full CLI command implementations."""
from __future__ import annotations

import json
import re
import shutil
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


def ingest_cmd(
    paths: list[str] = typer.Argument(None, help="PDF file(s) or directory. Defaults to data/pdfs/."),
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON to stdout"),
):
    """Full ingest pipeline: parse -> identify -> extract -> validate -> queue -> align -> ingest -> expand -> report.

    Accepts single file, multiple files, or a directory of PDFs.
    Defaults to data/pdfs/ when no paths provided.
    """
    if not paths:
        paths = ["data/pdfs/"]

    pdf_files: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            pdf_files.extend(sorted(path.glob("*.pdf")))
        elif path.is_file():
            pdf_files.append(path)
        else:
            if not json_output:
                typer.echo(f"File not found: {p}", err=True)

    if not pdf_files:
        if json_output:
            typer.echo(json.dumps({"error": "No PDF files found"}))
        else:
            typer.echo("No PDF files found.", err=True)
        raise typer.Exit(1)

    cfg = load_config()
    db = Database(cfg["db"]["path"])
    graph = GraphEngine()
    graph.load_from_db(db)
    dedup = DedupEngine(db)
    alias_table = AliasTable()

    queue_cfg = cfg.get("queue", {})
    weak_threshold = queue_cfg.get("weak_threshold", 0.7)
    auto_accept = queue_cfg.get("auto_accept", 0.9)

    results = []
    for i, pdf_path in enumerate(pdf_files, 1):
        if not json_output and len(pdf_files) > 1:
            typer.echo(f"\n{'='*60}")
            typer.echo(f"[{i}/{len(pdf_files)}] {pdf_path}")
            typer.echo(f"{'='*60}")

        result = _ingest_single_paper(
            pdf_path, cfg, db, graph, dedup, alias_table,
            weak_threshold, auto_accept,
            json_mode=json_output,
        )
        results.append(result)

    if json_output:
        output = {
            "ingested": len(results),
            "successful": sum(1 for r in results if r.get("ok")),
            "failed": sum(1 for r in results if not r.get("ok")),
            "papers": [r.get("report", {}) for r in results if r.get("ok")],
            "errors": [r.get("error", str(pdf_files[i])) for i, r in enumerate(results) if not r.get("ok")],
        }
        typer.echo(json.dumps(output, indent=2, ensure_ascii=False, default=str))
    else:
        if len(pdf_files) > 1:
            typer.echo(f"\n{'='*60}")
            typer.echo(f"Batch complete: {len(results)} papers ingested")
            success = sum(1 for r in results if r.get("ok"))
            typer.echo(f"  Successful: {success}, Failed: {len(results) - success}")

    db.close()


def _ingest_single_paper(
    pdf_path: Path,
    cfg: dict,
    db: "Database",
    graph: "GraphEngine",
    dedup: "DedupEngine",
    alias_table: "AliasTable",
    weak_threshold: float,
    auto_accept: float,
    json_mode: bool = False,
) -> dict:
    """Ingest a single paper. Returns {"ok": bool, "local_id": str|None, "report": dict|None, "error": str|None}."""
    def echo(msg: str):
        if not json_mode:
            typer.echo(msg)

    # Stage 1: Parse
    echo(f"Parsing: {pdf_path}")
    try:
        parsed = extract_pdf(pdf_path, cfg)
    except Exception as e:
        echo(f"Error parsing PDF: {e}")
        return {"ok": False, "local_id": None, "error": str(e)}
    echo(f"  Title: {parsed.title}")
    echo(f"  Year: {parsed.year}")
    echo(f"  arXiv: {parsed.arxiv}")
    echo(f"  Sections: {len(parsed.text_blocks)} high-signal blocks")

    # Stage 2: Identify
    ids = PaperIDs(doi=parsed.doi, arxiv=parsed.arxiv)
    local_id = dedup.resolve(ids, title=parsed.title, year=parsed.year)
    is_new = local_id is None

    if is_new:
        # Check for duplicate placeholders sharing the same external ID
        merged_id = _check_and_merge_duplicates(db, ids, parsed.title, parsed.year)
        if merged_id is not None:
            local_id = merged_id
            echo(f"  [merged] {local_id}")
            db.upgrade_placeholder(local_id)
            db.commit()
        else:
            local_id = f"p{uuid.uuid4().hex[:6]}"
            db.insert_paper(local_id, parsed.title, parsed.year, "uploaded")
            db.insert_paper_ids(local_id, doi=ids.doi, arxiv=ids.arxiv)
            db.commit()
            echo(f"  [new] {local_id}")
    else:
        db.upgrade_placeholder(local_id)
        db.commit()
        echo(f"  [upgrade] {local_id}")

    # Save parsed markdown for inspection
    papers_dir = Path(cfg.get("dirs", {}).get("pdfs", "data/pdfs")).parent / "papers"
    save_raw_md(parsed.raw_md, local_id, papers_dir, parsed.images_dir)

    # Stage 3: Extract
    echo("  Extracting concepts + arguments...")
    llm_models = cfg.get("llm", {}).get("models", [])
    if not llm_models:
        echo("Error: no LLM models configured. Run: drbrain setup")
        _log_error(cfg, "No LLM models configured")
        raise typer.Exit(1)

    import asyncio
    full_text = "\n\n".join(parsed.text_blocks)
    concepts = asyncio.run(extract_concepts(full_text, llm_models))
    if concepts is None:
        echo("Error: LLM extraction failed. All models exhausted.")
        _log_error(cfg, f"LLM extraction failed for {local_id}")
        raise typer.Exit(1)

    # Stage 3.5: Validate
    from brbrain.validator.schema import validate_extraction
    echo("  Validating extraction...")
    concept_data = {
        "problems": concepts.problems, "methods": concepts.methods,
        "conclusions": concepts.conclusions, "debates": concepts.debates,
        "gaps": concepts.gaps, "actors": concepts.actors,
    }
    validation = validate_extraction(concept_data, concepts.relations)
    echo(f"  Valid items: {len(validation['valid'])}")
    if validation["rejected"]:
        echo(f"  Rejected: {len(validation['rejected'])}")
        for r in validation["rejected"]:
            echo(f"    [yellow]{r['reason']}[/yellow]")

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
    echo(f"  Concepts inserted: {typed_count}")
    echo(f"  Arguments inserted: {len(valid_args)}")
    if queued_count:
        echo(f"  Queued for review: {queued_count}")
    if weak_count:
        echo(f"  Weak (ingested with marker): {weak_count}")
    if rejected_args:
        echo(f"  Arguments rejected: {len(rejected_args)}")

    # Stage 6: Expand
    echo("  Expanding citations...")
    from brbrain.extractor.citation import expand_citations
    refs, cits = expand_citations(db, local_id, cfg)
    refs_in = sum(1 for r in refs if r.in_graph)
    cits_in = sum(1 for c in cits if c.in_graph)
    echo(f"  References: {len(refs)} ({refs_in} in graph)")
    echo(f"  Citations: {len(cits)} ({cits_in} in graph)")

    # Stage 7: DOI enrichment — multi-source fallback chain
    current_doi = db.get_paper(local_id).get("doi")
    if not current_doi and parsed.title:
        crossref_email = cfg.get("api", {}).get("crossref_email")
        openalex_token = cfg.get("api", {}).get("openalex_token")

        # Try S2-provided DOI first (already backfilled in expand_citations)
        # Then try: CrossRef title → CrossRef arXiv → CrossRef direct DOI → OpenAlex title → OpenAlex arXiv
        doi_info = None
        sources = [
            ("CrossRef title", lambda: _enrich_doi_from_crossref(parsed.title, crossref_email)),
            ("CrossRef arXiv", lambda: _enrich_doi_from_crossref_arxiv(parsed.arxiv, crossref_email) if parsed.arxiv else None),
            ("CrossRef DOI", lambda: _enrich_doi_from_crossref_doi(parsed.doi, crossref_email) if parsed.doi else None),
            ("OpenAlex title", lambda: _enrich_doi_from_openalex(parsed.title, parsed.arxiv, openalex_token)),
        ]
        for name, fn in sources:
            if doi_info and doi_info.get("doi"):
                break
            echo(f"  Trying {name}...")
            doi_info = fn()

        if doi_info and doi_info.get("doi"):
            db.conn.execute(
                "UPDATE paper_ids SET doi = ? WHERE local_id = ?",
                (doi_info["doi"], local_id),
            )
            db.commit()
            echo(f"  Found DOI: {doi_info['doi']}")
        else:
            echo("  No DOI found in any source")

    # Stage 8: Closure
    echo("  Running rule closure...")
    graph.load_from_db(db)
    inferred = graph.closure()
    for edge in inferred:
        db.insert_edge(edge["src"], edge["dst"], edge["relation"], local_id)
    db.commit()
    echo(f"  Inferred edges: {len(inferred)}")

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
    echo(f"  Report saved: {report_path}")

    summary = report.summary
    if summary["graph_coverage"] < 0.3:
        echo(f"\n  [bold yellow]Warning: Low coverage ({summary['graph_coverage']:.1%}). Consider ingesting missing references.[/bold yellow]")

    # Log validation failures
    tbox_violations = [r["reason"] for r in validation["rejected"]]
    for reason in tbox_violations:
        _log_error(cfg, f"[{local_id}] TBox violation: {reason}")

    echo(f"\nDone: {local_id}")
    return {"ok": True, "local_id": local_id, "report": report.to_dict()}


def _check_and_merge_duplicates(db: "Database", ids: "PaperIDs", title: str, year: int | None) -> str | None:
    """Find existing placeholders sharing the same external ID and merge. Returns merged local_id or None."""
    target_id = None

    for key, val in [("doi", ids.doi), ("arxiv", ids.arxiv)]:
        if not val:
            continue
        existing = db.get_paper_by_external_id(key, val)
        if existing and existing != target_id:
            if target_id is None:
                target_id = existing
            else:
                # Two different placeholders share the same ID — merge them
                _merge_papers(db, keep_id=target_id, merge_id=existing)

    return target_id


def _merge_papers(db: "Database", keep_id: str, merge_id: str) -> None:
    """Merge merge_id into keep_id: move concepts, edges, update references."""
    # Move concepts
    db.conn.execute(
        "UPDATE concepts SET local_id = ? WHERE local_id = ?",
        (keep_id, merge_id),
    )
    # Move arguments
    db.conn.execute(
        "UPDATE arguments SET source_paper = ? WHERE source_paper = ?",
        (keep_id, merge_id),
    )
    # Redirect edges pointing to merge_id
    db.conn.execute(
        "UPDATE edges SET src_id = ? WHERE src_id = ?",
        (keep_id, merge_id),
    )
    db.conn.execute(
        "UPDATE edges SET dst_id = ? WHERE dst_id = ?",
        (keep_id, merge_id),
    )
    # Update source_paper in edges
    db.conn.execute(
        "UPDATE edges SET source_paper = ? WHERE source_paper = ?",
        (keep_id, merge_id),
    )
    # Delete the merged paper (cascades to paper_ids)
    db.conn.execute("DELETE FROM papers WHERE local_id = ?", (merge_id,))
    db.commit()


def _log_error(cfg: dict, message: str) -> None:
    """Append error message to data/logs/validation.log."""
    logs_dir = Path(cfg.get("dirs", {}).get("logs", "data/logs"))
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "validation.log"
    import datetime
    timestamp = datetime.datetime.now().isoformat()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def save_raw_md(raw_md: str, local_id: str, papers_dir: Path | None = None,
                images_src: Path | None = None) -> bool:
    """Save parsed markdown to data/papers/<local_id>.md and images to data/papers/images/<local_id>/."""
    if not raw_md or not local_id:
        return False
    if papers_dir is None:
        papers_dir = Path("data/papers")
    papers_dir.mkdir(parents=True, exist_ok=True)

    # Copy images to data/papers/images/<local_id>/
    if images_src and images_src.exists():
        img_dst = papers_dir / "images" / local_id
        shutil.copytree(images_src, img_dst, dirs_exist_ok=True)
        # Rewrite image refs in MD to point to local copies
        raw_md = re.sub(r"!\[(.*?)\]\((images/[^)]+)\)", rf"![\1](images/{local_id}/\2)", raw_md)

    md_path = papers_dir / f"{local_id}.md"
    md_path.write_text(raw_md, encoding="utf-8")
    return True


def _enrich_doi_from_crossref(title: str, email: str | None = None) -> dict | None:
    """Try to find DOI for a paper title via CrossRef API."""
    try:
        from brbrain.extractor.crossref import fetch_doi_by_title
        return fetch_doi_by_title(title, email=email)
    except Exception:
        return None


def _enrich_doi_from_crossref_arxiv(arxiv_id: str, email: str | None = None) -> dict | None:
    """Fallback: find DOI via arXiv ID in CrossRef."""
    try:
        from brbrain.extractor.crossref import fetch_doi_by_arxiv
        return fetch_doi_by_arxiv(arxiv_id, email=email)
    except Exception:
        return None


def _enrich_doi_from_crossref_doi(doi: str, email: str | None = None) -> dict | None:
    """Fallback: resolve DOI directly via CrossRef."""
    try:
        from brbrain.extractor.crossref import fetch_doi_by_doi
        return fetch_doi_by_doi(doi, email=email)
    except Exception:
        return None


def _enrich_doi_from_openalex(title: str, arxiv: str | None = None,
                              token: str | None = None) -> dict | None:
    """Try OpenAlex title search, then arXiv fallback."""
    try:
        from brbrain.extractor.openalex import search_work_by_title, search_work_by_arxiv
        result = search_work_by_title(title, token=token)
        if result and result.get("doi"):
            return result
        if arxiv:
            result = search_work_by_arxiv(arxiv, token=token)
            if result and result.get("doi"):
                return result
        return None
    except Exception:
        return None


def expand_cmd(
    local_id: str,
    depth: int = typer.Option(2, "--depth", help="Expansion depth"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
):
    """Expand a paper's citation neighborhood."""
    cfg = load_config()
    db = Database(cfg["db"]["path"])
    paper = db.get_paper(local_id)
    if not paper:
        msg = {"error": f"Paper not found: {local_id}"}
        if json_output:
            typer.echo(json.dumps(msg))
        else:
            typer.echo(f"Paper not found: {local_id}", err=True)
        raise typer.Exit(1)

    from brbrain.extractor.citation import expand_citations
    refs, cits = expand_citations(db, local_id, cfg)
    db.close()

    result = {
        "paper": {"local_id": local_id, "title": paper["title"]},
        "references": {
            "total": len(refs),
            "in_graph": sum(1 for r in refs if r.in_graph),
            "papers": [{"title": r.title, "year": r.year, "in_graph": r.in_graph} for r in refs],
        },
        "citations": {
            "total": len(cits),
            "in_graph": sum(1 for c in cits if c.in_graph),
            "papers": [{"title": c.title, "year": c.year, "in_graph": c.in_graph} for c in cits],
        },
    }

    if json_output:
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return

    typer.echo(f"Expanded: {paper['title']}")
    typer.echo(f"  References: {len(refs)} ({sum(1 for r in refs if r.in_graph)} in graph)")
    typer.echo(f"  Citations: {len(cits)} ({sum(1 for c in cits if c.in_graph)} in graph)")


def report_cmd(
    local_id: str,
    json_output: bool = typer.Option(False, "--json", help="Output full report JSON to stdout"),
):
    """Display single-paper report."""
    cfg = load_config()
    report_dir = Path(cfg["dirs"]["reports"])
    report_path = report_dir / f"{local_id}.json"
    if not report_path.exists():
        msg = {"error": f"No report found for {local_id}"}
        if json_output:
            typer.echo(json.dumps(msg))
        else:
            typer.echo(f"No report found for {local_id}. Run: drbrain ingest or drbrain expand", err=True)
        raise typer.Exit(1)

    data = json.loads(report_path.read_text())

    if json_output:
        typer.echo(json.dumps(data, indent=2, ensure_ascii=False, default=str))
        return

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


def closure_cmd(
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
):
    """Run rule-based closure on the full graph."""
    cfg = load_config()
    db = Database(cfg["db"]["path"])
    graph = GraphEngine()
    graph.load_from_db(db)

    inferred = graph.closure()

    # Persist new edges
    for edge in inferred:
        db.insert_edge(edge["src"], edge["dst"], edge["relation"], "closure")
    db.commit()
    db.close()

    if json_output:
        typer.echo(json.dumps({"inferred": inferred, "count": len(inferred)}, indent=2, ensure_ascii=False, default=str))
        return

    typer.echo(f"Inferred edges: {len(inferred)}")
    for edge in inferred:
        typer.echo(f"  {edge['src']} --[{edge['relation']}]--> {edge['dst']} (via {edge.get('via', 'unknown')})")


def seed_cmd(
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
):
    """Detect research seeds from graph patterns."""
    cfg = load_config()
    db = Database(cfg["db"]["path"])
    graph = GraphEngine()
    graph.load_from_db(db)

    seeds = graph.detect_research_seeds(db)
    db.close()

    if json_output:
        typer.echo(json.dumps(seeds, indent=2, ensure_ascii=False, default=str))
        return

    typer.echo(f"Research seeds found: {len(seeds)}")

    for seed in seeds:
        typer.echo(f"  [{seed['type']}] {seed['node']}: {seed['signal']}")


def list_cmd(
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
):
    """List all papers in database."""
    cfg = load_config()
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
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
):
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

    data = {
        "papers": papers, "uploaded": uploaded, "placeholders": placeholders,
        "concepts": concepts, "edges": edges, "aliases": aliases,
        "research_seeds": seeds, "arguments": arguments, "queue_pending": queue_pending,
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


def query_cmd(
    text: str,
    type_filter: str = typer.Option(None, "--type-filter", help="Filter by concept type (Problem, Method, etc.)"),
    arg_type: str = typer.Option(None, "--arg-type", help="Filter by argument claim type (supports, challenges, etc.)"),
    year_start: int = typer.Option(None, "--year-start", help="Filter by minimum year"),
    year_end: int = typer.Option(None, "--year-end", help="Filter by maximum year"),
    limit: int = typer.Option(20, "--limit", help="Maximum results"),
    neighbors: int = typer.Option(0, "--neighbors", "-n", help="Expand results by N hops of graph traversal"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON array to stdout"),
    jsonl: bool = typer.Option(False, "--jsonl", help="Output JSONL stream to stdout"),
):
    """Query concepts and arguments with BM25 + filters."""
    cfg = load_config()
    db = Database(cfg["db"]["path"])

    from brbrain.query.bm25 import build_bm25_index
    bm25 = build_bm25_index(db)
    results = bm25.search(text, type_filter=type_filter, arg_type_filter=arg_type, limit=limit)

    # Post-filter by year range
    if year_start is not None or year_end is not None:
        y_start = year_start or 0
        y_end = year_end or 9999
        results = [r for r in results if y_start <= r.get("year", y_end) <= y_end]

    # Expand by graph traversal
    if neighbors > 0 and results:
        graph = GraphEngine()
        graph.load_from_db(db)
        seed_ids = {r["local_id"] for r in results}
        expanded_ids = set()
        for sid in seed_ids:
            expanded_ids.update(graph.get_neighbors(sid, hops=neighbors))
        # Add neighbor papers as Paper-type results
        for nid in expanded_ids - seed_ids:
            paper = db.get_paper(nid)
            if paper:
                results.append({
                    "local_id": nid,
                    "type": "Paper",
                    "label": paper["title"],
                    "text": paper.get("abstract", ""),
                    "year": paper.get("year"),
                    "score": 0.0,
                    "_via_neighbors": True,
                })
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
    if neighbors:
        filters.append(f"neighbors={neighbors}")
    if filters:
        typer.echo(f"  Filters: {', '.join(filters)}")
    typer.echo(f"  Results: {len(results)}")
    for i, r in enumerate(results, 1):
        extra = ""
        if r["type"] == "Argument":
            extra = f" [{r.get('arg_type', '')}]"
        if r.get("_via_neighbors"):
            extra += " [neighbor]"
        year_str = f" ({r.get('year', '?')})" if r.get("year") else ""
        typer.echo(f"  {i}. [{r['type']}] {r['label']}{extra} (score: {r['score']:.3f}, paper: {r['local_id']}{year_str})")


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


def queue_cmd(
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
):
    """List all pending confidence queue items."""
    cfg = load_config()
    db = Database(cfg["db"]["path"])
    pending = db.get_queue_pending()
    db.close()

    if json_output:
        items = []
        for item in pending:
            data = json.loads(item["item_data"])
            items.append({**item, "item_data_parsed": data})
        typer.echo(json.dumps(items, indent=2, ensure_ascii=False, default=str))
        return

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


def queue_resolve_cmd(
    queue_id: int,
    accept: bool = typer.Option(False, "--accept", help="Accept the queue item"),
    reject: bool = typer.Option(False, "--reject", help="Reject the queue item"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
):
    """Resolve a queue item: accept or reject."""
    if accept and reject:
        msg = {"error": "cannot both accept and reject"}
        if json_output:
            typer.echo(json.dumps(msg))
        else:
            typer.echo("Error: cannot both accept and reject", err=True)
        raise typer.Exit(1)
    if not accept and not reject:
        msg = {"error": "specify --accept or --reject"}
        if json_output:
            typer.echo(json.dumps(msg))
        else:
            typer.echo("Error: specify --accept or --reject", err=True)
        raise typer.Exit(1)

    cfg = load_config()
    db = Database(cfg["db"]["path"])

    from brbrain.extractor.queue import resolve_accept, resolve_reject
    if accept:
        resolve_accept(db, queue_id)
        action = "accepted"
    else:
        resolve_reject(db, queue_id)
        action = "rejected"

    db.close()

    if json_output:
        typer.echo(json.dumps({"queue_id": queue_id, "action": action}, indent=2))
        return

    typer.echo(f"Queue item {queue_id} {action}.")


def timeline_cmd(
    concept: str,
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
):
    """Show concept evolution over time."""
    cfg = load_config()
    db = Database(cfg["db"]["path"])
    evolution = db.get_concept_evolution(concept)

    if not evolution:
        db.close()
        if json_output:
            typer.echo(json.dumps({"label": concept, "evolution": [], "signal": None}))
        else:
            typer.echo(f"No data for concept: {concept}")
        return

    row = db.conn.execute(
        "SELECT type FROM concepts WHERE label = ? LIMIT 1", (concept,)
    ).fetchone()
    ctype = row[0] if row else "unknown"

    signal_info = db.get_concept_signal(concept)

    db.close()

    if json_output:
        data = {
            "label": concept,
            "type": ctype,
            "evolution": evolution,
            "signal": signal_info["signal"] if signal_info else None,
        }
        typer.echo(json.dumps(data, indent=2, ensure_ascii=False, default=str))
        return

    typer.echo(f"\nConcept: {concept} ({ctype})")

    trend_labels = {
        "first_appeared": "— first appeared",
        "growing": "— rapid adoption",
        "declining": "— declining",
        "stable": "",
    }

    for entry in evolution:
        year = entry["year"]
        count = entry["count"]
        avg_conf = entry["avg_conf"]
        trend = entry.get("trend", "stable")
        label = trend_labels.get(trend, "")

        line = f"  {year}: {count} paper{'s' if count > 1 else ''} (avg confidence {avg_conf:.2f})"
        if label:
            line += f" {label}"
        typer.echo(line)

    if signal_info:
        typer.echo(f"Status: {signal_info['signal'].upper()}")
