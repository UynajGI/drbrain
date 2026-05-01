"""Full CLI command implementations."""

from __future__ import annotations

import asyncio
import importlib
import json
import re
import shutil
import uuid
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from drbrain.config import load_config
from drbrain.dedup.resolver import DedupEngine, PaperIDs
from drbrain.extractor.canonical import SmartAligner
from drbrain.extractor.concept import extract_concepts, extract_concepts_from_tree
from drbrain.graph.engine import GraphEngine
from drbrain.parser.mineru_parser import extract_pdf
from drbrain.query.tree_retrieval import query_by_structure
from drbrain.report.generator import PaperReport
from drbrain.storage.database import Database

console = Console()


def ingest_cmd(
    paths: list[str] = typer.Argument(
        None, help="PDF file(s) or directory. Defaults to data/inbox/."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output machine-readable JSON to stdout"
    ),
):
    """Full ingest pipeline: parse -> identify -> extract -> validate -> queue -> align -> ingest -> expand -> report.

    Accepts single file, multiple files, or a directory of PDFs.
    Defaults to data/inbox/ when no paths provided.
    """
    if not paths:
        cfg = load_config()
        inbox_path = cfg.get("dirs", {}).get("inbox", "data/spool/inbox")
        paths = [inbox_path]

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
    llm_models = cfg.get("llm", {}).get("models", [])

    aligner = SmartAligner(db, models=llm_models)

    queue_cfg = cfg.get("queue", {})
    weak_threshold = queue_cfg.get("weak_threshold", 0.7)
    auto_accept = queue_cfg.get("auto_accept", 0.9)

    results = []
    for i, pdf_path in enumerate(pdf_files, 1):
        if not json_output and len(pdf_files) > 1:
            typer.echo(f"\n{'=' * 60}")
            typer.echo(f"[{i}/{len(pdf_files)}] {pdf_path}")
            typer.echo(f"{'=' * 60}")

        result = _ingest_single_paper(
            pdf_path,
            cfg,
            db,
            graph,
            dedup,
            aligner,
            weak_threshold,
            auto_accept,
            json_mode=json_output,
        )
        results.append(result)

    if json_output:
        output = {
            "ingested": len(results),
            "successful": sum(1 for r in results if r.get("ok")),
            "failed": sum(1 for r in results if not r.get("ok")),
            "papers": [r.get("report", {}) for r in results if r.get("ok")],
            "errors": [
                r.get("error", str(pdf_files[i])) for i, r in enumerate(results) if not r.get("ok")
            ],
        }
        typer.echo(json.dumps(output, indent=2, ensure_ascii=False, default=str))
    else:
        if len(pdf_files) > 1:
            typer.echo(f"\n{'=' * 60}")
            typer.echo(f"Batch complete: {len(results)} papers ingested")
            success = sum(1 for r in results if r.get("ok"))
            typer.echo(f"  Successful: {success}, Failed: {len(results) - success}")

    db.close()


def _ingest_single_paper(
    pdf_path: Path,
    cfg: dict,
    db: Database,
    graph: GraphEngine,
    dedup: DedupEngine,
    aligner: SmartAligner,
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
        _move_to_pending(pdf_path, cfg, f"PDF parse error: {e}")
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

    # Insert OpenAlex-derived Actor concepts (deduplicated author IDs)
    from drbrain.extractor.openalex import search_authors_by_work

    oa_authors = search_authors_by_work(doi=ids.doi, title=parsed.title)
    if oa_authors:
        echo(f"  Authors: {len(oa_authors)} via OpenAlex")
        for author in oa_authors:
            actor_label = author["author_id"]
            db.insert_concept(local_id, "Actor", actor_label, 1.0, year=parsed.year)
            db.insert_alias(author["display_name"], actor_label)
            db.insert_edge(local_id, actor_label, "affiliated_with", local_id)
        db.commit()

    # Save parsed markdown and source PDF into per-paper directory
    papers_base = Path(cfg.get("dirs", {}).get("papers", "data/papers"))
    paper_dir = papers_base / local_id
    paper_dir.mkdir(parents=True, exist_ok=True)
    _save_paper_artifacts(parsed, local_id, paper_dir, pdf_path)

    llm_models = cfg.get("llm", {}).get("models", [])
    if not llm_models:
        echo("Error: no LLM models configured. Run: drbrain setup")
        _log_error(cfg, "No LLM models configured")
        raise typer.Exit(1)

    # Stage 2.1: Detect paper type
    from drbrain.extractor.detection import detect_paper_type_async

    echo("  Detecting paper type...")
    first_page = parsed.text_blocks[0] if parsed.text_blocks else getattr(parsed, "abstract", None)
    paper_type = asyncio.run(
        detect_paper_type_async(
            title=parsed.title,
            abstract=getattr(parsed, "abstract", None),
            first_page=first_page,
            models=llm_models,
        )
    )
    echo(f"  Paper type: {paper_type}")
    # Update paper_type in DB (already inserted as 'paper' default)
    db.conn.execute(
        "UPDATE papers SET paper_type = ? WHERE local_id = ?",
        (paper_type, local_id),
    )
    db.commit()

    # Stage 2.5: Structure markdown into tree (PageIndex)
    md_path = paper_dir / "raw.md"
    tree_json_path = paper_dir / "tree.json"
    echo("  Structuring document tree...")
    try:
        from drbrain.parser.pageindex_parser import TreeConfig, md_to_tree

        pageindex_cfg = TreeConfig(
            if_thinning=True,
            min_token_threshold=5000,
            if_add_node_summary=True,
            if_add_doc_description=True,
            if_add_node_text=False,  # Content loaded on demand, not embedded
            if_add_node_id=True,
            max_node_tokens=10000,
        )
        doc_tree = asyncio.run(md_to_tree(md_path, config=pageindex_cfg, models=llm_models))
        tree_json_path.write_text(doc_tree.to_json(), encoding="utf-8")
        echo(f"  Document tree: {len(doc_tree.structure)} sections → {tree_json_path.name}")
    except Exception as e:
        echo(f"  [yellow]Warning: tree structuring failed: {e}[/yellow]")
        _log_error(cfg, f"Tree structuring failed for {local_id}: {e}")

    # Stage 3: Extract
    echo("  Extracting concepts + arguments...")
    if tree_json_path.exists():
        tree_data = json.loads(tree_json_path.read_text(encoding="utf-8"))
        concepts = asyncio.run(
            extract_concepts_from_tree(md_path, tree_data["structure"], llm_models)
        )
    else:
        concepts = None

    if concepts is None:
        # Fallback: flat text extraction (no tree or tree extraction failed)
        full_text = "\n\n".join(parsed.text_blocks)
        concepts = asyncio.run(extract_concepts(full_text, llm_models))
    if concepts is None:
        echo("Error: LLM extraction failed. All models exhausted.")
        _log_error(cfg, f"LLM extraction failed for {local_id}")
        _move_to_pending(pdf_path, cfg, "LLM extraction failed")
        raise typer.Exit(1)

    # Stage 3.5: Validate
    from drbrain.validator.schema import validate_extraction

    echo("  Validating extraction...")
    concept_data = {
        "problems": concepts.problems,
        "methods": concepts.methods,
        "conclusions": concepts.conclusions,
        "debates": concepts.debates,
        "gaps": concepts.gaps,
        "actors": concepts.actors,
    }
    validation = validate_extraction(concept_data, concepts.relations)
    echo(f"  Valid items: {len(validation['valid'])}")
    if validation["rejected"]:
        echo(f"  Rejected: {len(validation['rejected'])}")
        for r in validation["rejected"]:
            echo(f"    [yellow]{r['reason']}[/yellow]")

    valid_relations = [r["detail"] for r in validation["valid"] if r["type"] == "relation"]

    # Stage 3.6: Queue low-confidence concepts
    from drbrain.extractor.queue import route_item

    typed_count = 0
    queued_count = 0
    weak_count = 0
    all_items = [
        ("Problem", concepts.problems),
        ("Method", concepts.methods),
        ("Conclusion", concepts.conclusions),
        ("Debate", concepts.debates),
        ("Gap", concepts.gaps),
    ]
    for ctype, items in all_items:
        for item in items:
            label = item.get("label", "")
            conf = item.get("confidence", 1.0)
            routing = route_item(
                db,
                local_id,
                "concept",
                {"label": label, "type": ctype},
                conf,
                weak_threshold,
                auto_accept,
            )
            if routing["action"] == "queued":
                queued_count += 1
            elif routing["action"] == "weak":
                weak_count += 1
            typed_count += 1

    # Stage 4: Align + Stage 5: Ingest
    # Stage 6: Expand
    # Stage 7: DOI enrichment
    # Stage 8: Closure
    # All wrapped in try/except for transaction rollback (Spec §20)
    try:
        # Stage 4: Align + Stage 5: Ingest
        for ctype, items in all_items:
            for item in items:
                label = item.get("label", "")
                conf = item.get("confidence", 1.0)
                if conf >= weak_threshold:
                    canonical_id = aligner.align(label, ctype)
                    db.insert_concept(
                        local_id,
                        ctype,
                        label,
                        conf,
                        year=parsed.year,
                        section=item.get("section", ""),
                    )
                    db.insert_alias(label, canonical_id)

        for rel in valid_relations:
            db.insert_edge(rel["head"], rel["tail"], rel["rel"], local_id)

        # Gap → Problem/Method (points_to) — infer from label overlap
        if concepts.gaps and (concepts.problems or concepts.methods):
            gap_labels = {g["label"] for g in concepts.gaps}
            prob_labels = {p["label"] for p in concepts.problems}
            method_labels = {m["label"] for m in concepts.methods}
            for gap in gap_labels:
                # Simple keyword overlap heuristic
                gap_words = set(gap.lower().split())
                for prob in prob_labels:
                    prob_words = set(prob.lower().split())
                    if gap_words & prob_words:
                        db.insert_edge(gap, prob, "points_to", local_id)
                for method in method_labels:
                    method_words = set(method.lower().split())
                    if gap_words & method_words:
                        db.insert_edge(gap, method, "points_to", local_id)

        # Ingest arguments
        from drbrain.extractor.argument import validate_arguments

        valid_args, rejected_args = validate_arguments(concepts.arguments)
        for arg in valid_args:
            db.insert_argument(
                local_id,
                arg.claim,
                arg.claim_type,
                arg.target,
                arg.target_type,
                arg.evidence_type,
                arg.evidence_detail,
                arg.mechanism,
                arg.confidence,
                section=arg.section,
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
        from drbrain.extractor.citation import expand_citations

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

            doi_info = None
            sources = [
                ("CrossRef title", lambda: _enrich_doi_from_crossref(parsed.title, crossref_email)),
                (
                    "CrossRef arXiv",
                    lambda: (
                        _enrich_doi_from_crossref_arxiv(parsed.arxiv, crossref_email)
                        if parsed.arxiv
                        else None
                    ),
                ),
                (
                    "CrossRef DOI",
                    lambda: (
                        _enrich_doi_from_crossref_doi(parsed.doi, crossref_email)
                        if parsed.doi
                        else None
                    ),
                ),
                (
                    "OpenAlex title",
                    lambda: _enrich_doi_from_openalex(parsed.title, parsed.arxiv, openalex_token),
                ),
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

        # Stage 8: Closure (full + incremental for new paper)
        echo("  Running rule closure...")
        graph.load_from_db(db)
        inferred = graph.closure()
        # Incremental closure from this paper's nodes
        affected_nodes = {local_id}
        for ctype, items in all_items:
            for item in items:
                affected_nodes.add(item.get("label", ""))
        incr_inferred = graph.closure_incremental(affected_nodes)
        inferred.extend(incr_inferred)
        for edge in inferred:
            db.insert_edge(edge["src"], edge["dst"], edge["relation"], local_id)
        db.commit()
        echo(f"  Inferred edges: {len(inferred)}")

    except Exception as e:
        db.conn.rollback()
        echo(f"[red]Error during ingestion, rolled back: {e}[/red]")
        _log_error(cfg, f"Ingestion rollback for {local_id}: {e}")
        return {"ok": False, "local_id": local_id, "error": str(e)}

    # Stage 9: Report
    report = PaperReport(
        local_id=local_id,
        title=parsed.title,
        year=parsed.year,
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
        echo(
            f"\n  [bold yellow]Warning: Low coverage ({summary['graph_coverage']:.1%}). Consider ingesting missing references.[/bold yellow]"
        )

    # Log validation failures
    tbox_violations = [r["reason"] for r in validation["rejected"]]
    for reason in tbox_violations:
        _log_error(cfg, f"[{local_id}] TBox violation: {reason}")

    # Flush pending LLM alignments
    aligner.flush_pending()

    echo(f"\nDone: {local_id}")
    return {"ok": True, "local_id": local_id, "report": report.to_dict()}


def _check_and_merge_duplicates(
    db: Database, ids: PaperIDs, title: str, year: int | None
) -> str | None:
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


def _merge_papers(db: Database, keep_id: str, merge_id: str) -> None:
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


def _move_to_pending(pdf_path: Path, cfg: dict, reason: str) -> None:
    """Move a failed PDF to the pending directory."""
    from drbrain.storage.inbox import move_to_pending

    pending_dir = Path(cfg.get("dirs", {}).get("pending", "data/spool/pending"))
    try:
        move_to_pending(pdf_path, pending_dir, reason=reason)
    except Exception:
        pass  # Best-effort; don't block the error path


def _save_paper_artifacts(parsed, local_id: str, paper_dir: Path, source_pdf: Path) -> None:
    """Save all paper artifacts into a per-paper directory.

    Layout:
        data/papers/<local_id>/
            source.pdf   — original PDF (copied from inbox)
            raw.md       — MinerU markdown output
            images/      — extracted images
    """
    # Copy source PDF
    dst_pdf = paper_dir / "source.pdf"
    if not dst_pdf.exists():
        shutil.copy2(source_pdf, dst_pdf)

    # Copy images and rewrite refs
    raw_md = parsed.raw_md
    if parsed.images_dir and parsed.images_dir.exists():
        img_dst = paper_dir / "images"
        shutil.copytree(parsed.images_dir, img_dst, dirs_exist_ok=True)
        # MinerU outputs "images/<hash>/file.jpg", rewrite to "images/<hash>/file.jpg"
        # (no local_id prefix needed — images/ is already inside paper_dir)
        raw_md = re.sub(
            r"!\[(.*?)\]\(images/([^)]+)\)",
            r"![\1](images/\2)",
            raw_md,
        )

    md_path = paper_dir / "raw.md"
    md_path.write_text(raw_md, encoding="utf-8")


def _enrich_doi_from_crossref(title: str, email: str | None = None) -> dict | None:
    """Try to find DOI for a paper title via CrossRef API."""
    try:
        from drbrain.extractor.crossref import fetch_doi_by_title

        return fetch_doi_by_title(title, email=email)
    except Exception:
        return None


def _enrich_doi_from_crossref_arxiv(arxiv_id: str, email: str | None = None) -> dict | None:
    """Fallback: find DOI via arXiv ID in CrossRef."""
    try:
        from drbrain.extractor.crossref import fetch_doi_by_arxiv

        return fetch_doi_by_arxiv(arxiv_id, email=email)
    except Exception:
        return None


def _enrich_doi_from_crossref_doi(doi: str, email: str | None = None) -> dict | None:
    """Fallback: resolve DOI directly via CrossRef."""
    try:
        from drbrain.extractor.crossref import fetch_doi_by_doi

        return fetch_doi_by_doi(doi, email=email)
    except Exception:
        return None


def _enrich_doi_from_openalex(
    title: str, arxiv: str | None = None, token: str | None = None
) -> dict | None:
    """Try OpenAlex title search, then arXiv fallback."""
    try:
        from drbrain.extractor.openalex import search_work_by_arxiv, search_work_by_title

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

    from drbrain.extractor.citation import expand_citations

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
            typer.echo(
                f"No report found for {local_id}. Run: drbrain ingest or drbrain expand", err=True
            )
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
        typer.echo(
            "  [bold yellow]Alert: Low coverage - consider expanding citation network[/bold yellow]"
        )


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
        typer.echo(
            json.dumps(
                {"inferred": inferred, "count": len(inferred)},
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )
        return

    typer.echo(f"Inferred edges: {len(inferred)}")
    for edge in inferred:
        typer.echo(
            f"  {edge['src']} --[{edge['relation']}]--> {edge['dst']} (via {edge.get('via', 'unknown')})"
        )


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


def query_cmd(
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
    json_output: bool = typer.Option(False, "--json", help="Output JSON array to stdout"),
    jsonl: bool = typer.Option(False, "--jsonl", help="Output JSONL stream to stdout"),
    paper: str = typer.Option(
        None,
        "--paper",
        help="Paper local_id for PageIndex tree retrieval (bypasses BM25 when set)",
    ),
):
    """Query concepts and arguments with BM25 + filters, or use PageIndex tree retrieval."""
    cfg = load_config()

    # --- Tree retrieval path ---
    # Normalize: when called directly (not through typer CLI), OptionInfo is still the default
    _paper = paper if not isinstance(paper, typer.models.OptionInfo) else paper.default

    if _paper:
        papers_dir = Path(cfg["dirs"]["papers"])
        paper_dir = papers_dir / _paper
        if not paper_dir.exists():
            typer.echo(f"Paper not found: {_paper}", err=True)
            raise typer.Exit(1)
        if not (paper_dir / "tree.json").exists():
            typer.echo(f"tree.json not found for {_paper}. Run 'drbrain ingest' first.", err=True)
            raise typer.Exit(1)

        llm_models = cfg.get("llm", {}).get("models", [])
        sections = asyncio.run(query_by_structure(text, paper_dir, llm_models))

        if sections is None:
            if json_output:
                typer.echo(
                    json.dumps(
                        {"query": text, "paper": _paper, "mode": "pageindex", "sections": []},
                        ensure_ascii=False,
                    )
                )
            else:
                typer.echo(f"No relevant sections found for: {text}")
            return

        if json_output:
            typer.echo(
                json.dumps(
                    {"query": text, "paper": _paper, "mode": "pageindex", "sections": sections},
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return

        # Rich text output
        typer.echo(f"Query: {text}")
        typer.echo(f"  Paper: {_paper}")
        typer.echo("  Mode: pageindex")
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
                results.append(
                    {
                        "local_id": nid,
                        "type": "Paper",
                        "label": paper["title"],
                        "text": paper.get("abstract", ""),
                        "year": paper.get("year"),
                        "score": 0.0,
                        "_via_neighbors": True,
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
        conf_str = f", confidence: {r['confidence']:.2f}" if "confidence" in r else ""
        typer.echo(
            f"  {i}. [{r['type']}] {r['label']}{extra} (score: {r['score']:.3f}, paper: {r['local_id']}{year_str}{conf_str})"
        )


def export_cmd(
    local_id: str = typer.Argument(None, help="Paper local_id"),
    format: str = typer.Option("bib", "--format", "-f", help="Export format: bib, ris, md"),
    all: bool = typer.Option(False, "--all", help="Export all papers"),
    output: str = typer.Option(None, "--output", "-o", help="Output file path"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
):
    """Export paper metadata to BibTeX, RIS, or Markdown."""
    from drbrain.storage.export import (
        batch_export,
        meta_to_bibtex,
        meta_to_markdown,
        meta_to_ris,
    )

    if format not in ("bib", "ris", "md"):
        typer.echo(f"Unknown format: {format}. Use bib, ris, or md.", err=True)
        raise typer.Exit(1)

    cfg = load_config()
    db = Database(cfg["db"]["path"])

    if all:
        papers = db.get_all_papers()
        metas = [_export_paper_to_meta(db, p["local_id"]) for p in papers]
        result = batch_export(metas, format)
    elif local_id:
        paper = db.get_paper(local_id)
        if not paper:
            db.close()
            typer.echo(f"Paper not found: {local_id}", err=True)
            raise typer.Exit(1)
        meta = _export_paper_to_meta(db, local_id)
        formatters = {"bib": meta_to_bibtex, "ris": meta_to_ris, "md": meta_to_markdown}
        result = formatters[format](meta)
    else:
        db.close()
        typer.echo("Specify a paper local_id or use --all", err=True)
        raise typer.Exit(1)

    db.close()

    if json_output:
        typer.echo(json.dumps({"format": format, "result": result}, ensure_ascii=False))
        return

    if output:
        Path(output).write_text(result + "\n", encoding="utf-8")
        typer.echo(f"Exported to {output}")
    else:
        typer.echo(result)


def _export_paper_to_meta(db: Database, local_id: str) -> dict:
    """Build export-ready metadata dict from DB."""
    paper = db.get_paper(local_id)
    if not paper:
        return {}

    authors = db.conn.execute(
        "SELECT GROUP_CONCAT(a.variant, ' and ') "
        "FROM concepts c JOIN aliases a ON a.canonical_id = c.label "
        "WHERE c.local_id = ? AND c.type = 'Actor'",
        (local_id,),
    ).fetchone()

    author_list = authors[0] if authors and authors[0] else ""
    first_author = author_list.split(" and ")[0].strip() if author_list else ""
    lastname = first_author.split()[-1] if first_author else ""

    return {
        "local_id": local_id,
        "title": paper.get("title", ""),
        "year": paper.get("year"),
        "doi": paper.get("doi", ""),
        "arxiv": paper.get("arxiv", ""),
        "authors": author_list,
        "first_author_lastname": lastname,
        "paper_type": paper.get("paper_type", "paper"),
    }


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

    from drbrain.extractor.queue import resolve_accept, resolve_reject

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


def queue_resolve_all_cmd(
    accept: bool = typer.Option(False, "--accept", help="Accept all pending items"),
    reject: bool = typer.Option(False, "--reject", help="Reject all pending items"),
    type_filter: str = typer.Option(
        None, "--type", help="Filter by item type (concept, alias, relation)"
    ),
    max_conf: float = typer.Option(
        None, "--max-conf", help="Only process items with confidence <= this value"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
):
    """Batch resolve all pending queue items."""
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

    from drbrain.extractor.queue import resolve_all

    action = "accept" if accept else "reject"
    result = resolve_all(db, action, type_filter=type_filter, max_conf=max_conf)

    db.close()

    if json_output:
        typer.echo(
            json.dumps(
                {
                    "action": action,
                    "count": result["count"],
                    "filters": {
                        "type": type_filter,
                        "max_conf": max_conf,
                    },
                },
                indent=2,
            )
        )
        return

    if result["count"] == 0:
        typer.echo("No matching pending items.")
        return

    typer.echo(f"{result['count']} item(s) {action}ed.")


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


def delete_cmd(
    local_id: str,
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt"),
    rm_files: bool = typer.Option(False, "--rm-files", help="Also delete paper directory"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
):
    """Delete a paper and all its associated data from the graph."""
    import shutil as _shutil

    cfg = load_config()
    db = Database(cfg["db"]["path"])

    paper = db.get_paper(local_id)
    if paper is None:
        db.close()
        if json_output:
            typer.echo(json.dumps({"error": f"paper {local_id} not found"}))
        else:
            typer.echo(f"Paper {local_id} not found.", err=True)
        raise typer.Exit(1)

    counts = db.delete_paper(local_id)
    db.close()

    file_deleted = False
    if rm_files:
        papers_dir = Path(cfg.get("dirs", {}).get("papers", "data/papers"))
        paper_dir = papers_dir / local_id
        if paper_dir.exists():
            _shutil.rmtree(paper_dir)
            file_deleted = True

    if json_output:
        typer.echo(
            json.dumps(
                {
                    "deleted": local_id,
                    "title": paper["title"],
                    "files_deleted": file_deleted,
                    **counts,
                },
                indent=2,
            )
        )
        return

    typer.echo(f"Deleted paper: {paper['title']} ({local_id})")
    typer.echo(
        f"  concepts: {counts['concepts']}, arguments: {counts['arguments']}, "
        f"edges: {counts['edges']}, queue items: {counts['queue_items']}"
    )
    if file_deleted:
        typer.echo(f"  files: removed data/papers/{local_id}/")


def serve_cmd(
    host: str = typer.Option("127.0.0.1", "--host", help="Server host"),
    port: int = typer.Option(8501, "--port", help="Server port"),
):
    """Launch the Streamlit UI for graph visualization."""
    import subprocess
    import sys

    cfg = load_config()
    app_path = Path(__file__).parent.parent / "api" / "app.py"

    if not app_path.exists():
        typer.echo(f"Streamlit app not found at {app_path}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Starting DrBrain UI at http://{host}:{port}")
    typer.echo(f"DB: {cfg['db']['path']}")
    typer.echo("Press Ctrl+C to stop.")

    env = {**__import__("os").environ, "DRBRAIN_DB_PATH": cfg["db"]["path"]}
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.address",
            host,
            "--server.port",
            str(port),
            "--server.headless",
            "true",
        ],
        env=env,
    )


def lineage_cmd(
    author_id: str = typer.Argument(None, help="OpenAlex author ID (e.g., A5023806754)"),
    list_all: bool = typer.Option(False, "--list", help="List all actors with paper counts"),
    name: str = typer.Option(None, "--name", "-n", help="Search actors by display name"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Explore author/research lineage via OpenAlex deduplicated IDs."""
    cfg = load_config()
    db = Database(cfg["db"]["path"])

    if list_all:
        rows = db.conn.execute(
            "SELECT c.label, COUNT(DISTINCT c.local_id) as paper_count, "
            "GROUP_CONCAT(DISTINCT a.variant) as aliases "
            "FROM concepts c "
            "LEFT JOIN aliases a ON a.canonical_id = c.label "
            "WHERE c.type = 'Actor' "
            "GROUP BY c.label "
            "ORDER BY paper_count DESC, c.label"
        ).fetchall()
        db.close()

        if not rows:
            typer.echo("No actors found.")
            return

        if json_output:
            data = [
                {"author_id": r[0], "paper_count": r[1], "aliases": r[2].split(",") if r[2] else []}
                for r in rows
            ]
            typer.echo(json.dumps(data, indent=2, ensure_ascii=False))
            return

        typer.echo(f"Authors ({len(rows)} total):")
        table = Table(show_header=True, header_style="bold")
        table.add_column("Author ID", style="cyan")
        table.add_column("Display Name", style="green")
        table.add_column("Papers", justify="right")
        for r in rows:
            display = r[2].split(",")[0] if r[2] else r[0]
            table.add_row(r[0], display, str(r[1]))
        console.print(table)

    elif name:
        # Search by display name → resolve to author_id(s)
        rows = db.conn.execute(
            "SELECT DISTINCT canonical_id FROM aliases WHERE variant LIKE ? COLLATE NOCASE",
            (f"%{name}%",),
        ).fetchall()
        if not rows:
            db.close()
            typer.echo(f"No actors matching '{name}'.")
            return
        db.close()
        # Show each matching actor
        for (matched_id,) in rows:
            _show_actor(cfg, matched_id)

    elif author_id:
        _show_actor(cfg, author_id)
        db.close()

    else:
        typer.echo(
            "Usage: drbrain lineage <author_id>\n"
            "       drbrain lineage --list\n"
            "       drbrain lineage --name <display_name>",
            err=True,
        )
        raise typer.Exit(1)


def _show_actor(cfg: dict, author_id: str) -> None:
    """Show detailed info for a single actor."""
    db = Database(cfg["db"]["path"])

    # Get display names from aliases
    aliases = db.conn.execute(
        "SELECT variant FROM aliases WHERE canonical_id = ?", (author_id,)
    ).fetchall()
    display_names = [a[0] for a in aliases]

    # Get papers
    papers = db.conn.execute(
        "SELECT DISTINCT c.local_id, p.title, p.year "
        "FROM concepts c JOIN papers p ON c.local_id = p.local_id "
        "WHERE c.type = 'Actor' AND c.label = ? ORDER BY p.year",
        (author_id,),
    ).fetchall()
    paper_ids = [p[0] for p in papers]

    # Get shared_actor connections (edges between papers of this author and other papers)
    connected_papers: list[str] = []
    if paper_ids:
        placeholders = ",".join("?" for _ in paper_ids)
        rows = db.conn.execute(
            f"SELECT DISTINCT e.dst_id FROM edges e "
            f"WHERE e.relation = 'shared_actor' AND e.src_id IN ({placeholders})",
            paper_ids,
        ).fetchall()
        connected_papers = [r[0] for r in rows if r[0] not in paper_ids]

    db.close()

    if not papers:
        typer.echo(f"Actor '{author_id}' has no associated papers.")
        return

    typer.echo(f"\nAuthor: {author_id}")
    if display_names:
        typer.echo(f"Display: {', '.join(display_names)}")
    typer.echo(f"Papers: {len(papers)}")
    for title, year in [(p[1], p[2]) for p in papers]:
        year_str = f" ({year})" if year else ""
        typer.echo(f"  - {title}{year_str}")

    if connected_papers:
        typer.echo(f"\nShared actor connections ({len(connected_papers)}):")
        # Resolve connected papers to their actors
        cfg2 = load_config()
        db2 = Database(cfg2["db"]["path"])
        for pid in connected_papers:
            paper = db2.get_paper(pid)
            title = paper["title"][:80] if paper else pid
            connected_actors = db2.conn.execute(
                "SELECT DISTINCT c.label FROM concepts c WHERE c.type = 'Actor' AND c.local_id = ?",
                (pid,),
            ).fetchall()
            actor_names = []
            for (aid,) in connected_actors:
                alias_row = db2.conn.execute(
                    "SELECT variant FROM aliases WHERE canonical_id = ? LIMIT 1", (aid,)
                ).fetchone()
                actor_names.append(alias_row[0] if alias_row else aid)
            typer.echo(f"  - [{', '.join(actor_names)}] {title}")
        db2.close()


def check_cmd():
    """Check dependencies, configuration, and environment variables."""
    console = Console()
    warnings = []
    errors = []

    console.print("\n[bold]DrBrain — Dependency & Configuration Check[/bold]\n")

    # -- Python packages --
    console.print("[bold]Python Packages[/bold]")
    table = Table(show_header=False, box=None, padding=(0, 2))
    required_packages = [
        ("pypdfium2", "pypdfium2"),
        ("litellm", "litellm"),
        ("typer", "typer"),
        ("rich", "rich"),
        ("pyyaml", "yaml"),
        ("pydantic", "pydantic"),
        ("streamlit", "streamlit"),
    ]
    for pkg_name, import_name in required_packages:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "installed")
            table.add_row(f"  {pkg_name}", f"[green]OK[/green] ({version})")
        except ImportError:
            table.add_row(f"  {pkg_name}", "[red]MISSING[/red]")
            errors.append(f"Missing Python package: {pkg_name}")
    console.print(table)

    # -- External CLI tools --
    console.print("\n[bold]External Tools[/bold]")
    table2 = Table(show_header=False, box=None, padding=(0, 2))
    cli_tools = {
        "mineru-open-api": "MinerU PDF parser CLI",
    }
    for tool, desc in cli_tools.items():
        found = shutil.which(tool)
        if found:
            table2.add_row(f"  {tool}", f"[green]OK[/green] ({found}) — {desc}")
        else:
            table2.add_row(
                f"  {tool}",
                "[yellow]NOT FOUND[/yellow]",
                f"{desc} (optional, fallback to pypdfium2)",
            )
            warnings.append(f"{tool} not found — PDF parsing will use pypdfium2 fallback only")
    console.print(table2)

    # -- Config files --
    console.print("\n[bold]Configuration[/bold]")
    table3 = Table(show_header=False, box=None, padding=(0, 2))

    base_config = Path("config.yaml")
    local_config = Path("config.local.yaml")

    if base_config.exists():
        table3.add_row("  config.yaml", "[green]Found[/green]")
    else:
        table3.add_row("  config.yaml", "[red]Missing[/red]")
        errors.append("config.yaml not found")

    if local_config.exists():
        table3.add_row("  config.local.yaml", "[green]Found (overrides base)[/green]")
    else:
        table3.add_row("  config.local.yaml", "[yellow]Not found[/yellow]")
        warnings.append(
            "No config.local.yaml — using base config values (env var placeholders unresolved)"
        )

    # Load config and check key values
    try:
        cfg = load_config()

        console.print("\n[bold]API Keys & Tokens[/bold]")
        table4 = Table(show_header=False, box=None, padding=(0, 2))

        # LLM models
        models = cfg.get("llm", {}).get("models", [])
        for i, m in enumerate(models):
            label = f"  LLM [{i}] {m.get('provider', '?')}/{m.get('model', '?')}"
            api_key = m.get("api_key", "")
            if api_key and not api_key.startswith("${"):
                masked = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "***"
                table4.add_row(label, f"[green]Set[/green] ({masked})")
            else:
                table4.add_row(label, "[yellow]Not configured[/yellow]")
                warnings.append(f"LLM model {m.get('provider')}/{m.get('model')} has no API key")

        # MinerU token
        mineru_token = cfg.get("mineru", {}).get("token", "")
        if mineru_token and not mineru_token.startswith("${"):
            table4.add_row("  MinerU token", "[green]Set[/green]")
        else:
            table4.add_row(
                "  MinerU token", "[yellow]Not set[/yellow]", "(flash mode will be used)"
            )
            warnings.append("MinerU token not configured — flash extraction mode only")

        # CrossRef email
        crossref_email = cfg.get("api", {}).get("crossref_email", "")
        if crossref_email and not crossref_email.startswith("${"):
            table4.add_row("  CrossRef email", f"[green]Set[/green] ({crossref_email})")
        else:
            table4.add_row(
                "  CrossRef email", "[yellow]Not set[/yellow]", "(optional, polite pool)"
            )
            warnings.append("CrossRef email not set — will use anonymous polite pool access")

        # OpenAlex token
        oa_token = cfg.get("api", {}).get("openalex_token", "")
        if oa_token and not oa_token.startswith("${"):
            table4.add_row("  OpenAlex token", "[green]Set[/green]")
        else:
            table4.add_row(
                "  OpenAlex token",
                "[yellow]Not set[/yellow]",
                "(anonymous access, lower rate limit)",
            )
            warnings.append("OpenAlex token not set — using anonymous access with lower rate limit")

        console.print(table4)

    except FileNotFoundError as e:
        console.print(f"  [red]{e}[/red]")

    # -- Directories --
    console.print("\n[bold]Directories[/bold]")
    table5 = Table(show_header=False, box=None, padding=(0, 2))
    try:
        cfg = load_config()
        dirs_config = cfg.get("dirs", {})
        dir_paths = (
            list(dirs_config.values())
            if dirs_config
            else ["data/inbox", "data/papers", "data/reports", "data/cache", "data/logs"]
        )
        for dir_path in dir_paths:
            p = Path(dir_path)
            if p.exists():
                table5.add_row(f"  {dir_path}", "[green]Exists[/green]")
            else:
                table5.add_row(f"  {dir_path}", "[yellow]Missing (will be created on use)[/yellow]")
    except Exception:
        for d in ["data/inbox", "data/papers"]:
            p = Path(d)
            if p.exists():
                table5.add_row(f"  {d}", "[green]Exists[/green]")
            else:
                table5.add_row(f"  {d}", "[yellow]Missing[/yellow]")
    console.print(table5)

    # -- Database --
    console.print("\n[bold]Database[/bold]")
    table6 = Table(show_header=False, box=None, padding=(0, 2))
    try:
        cfg = load_config()
        db_path = cfg.get("db", {}).get("path", "data/db/drbrain.db")
        p = Path(db_path)
        if p.exists():
            table6.add_row(f"  {db_path}", "[green]Exists[/green]")
        else:
            table6.add_row(
                f"  {db_path}",
                "[yellow]Not yet created[/yellow]",
                "(run `drbrain ingest` to initialize)",
            )
    except Exception:
        table6.add_row("  (config unavailable)", "[yellow]Unknown[/yellow]")
    console.print(table6)

    # -- Summary --
    console.print("\n[bold]Summary[/bold]")
    if errors:
        console.print(f"\n[bold red]Errors ({len(errors)}):[/bold red]")
        for e in errors:
            console.print(f"  [red]✗[/red] {e}")
    if warnings:
        console.print(f"\n[bold yellow]Warnings ({len(warnings)}):[/bold yellow]")
        for w in warnings:
            console.print(f"  [yellow]![/yellow] {w}")
    if not errors and not warnings:
        console.print("\n[bold green]All checks passed![/bold green]")
    elif not errors:
        console.print(
            f"\n[bold green]Ready to use[/bold green] ({len(warnings)} optional warnings)"
        )
    else:
        console.print(f"\n[bold red]Not ready[/bold red] — fix {len(errors)} error(s) above")

    if errors:
        raise typer.Exit(1)


def clean_cmd(
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt"),
    config_path: str = typer.Option("config.yaml", "--config", "-c", help="Config file path"),
) -> None:
    """Clear data directories (db, cache, logs, papers, reports). Keeps inbox (PDFs) intact."""
    cfg = load_config(config_path)
    dirs = cfg.get("dirs", {})

    targets = [
        dirs.get("db", "data/db"),
        dirs.get("cache", "data/cache"),
        dirs.get("logs", "data/logs"),
        dirs.get("papers", "data/papers"),
        dirs.get("reports", "data/reports"),
    ]

    existing = [t for t in targets if Path(t).exists()]
    if not existing:
        typer.echo("Nothing to clean — data directories are already empty.")
        return

    if not force:
        typer.echo("Will clear these directories:")
        for d in existing:
            count = sum(1 for _ in Path(d).rglob("*") if _.is_file())
            typer.echo(f"  {d}/ ({count} files)")
        confirm = typer.confirm("Proceed?", default=False)
        if not confirm:
            typer.echo("Cancelled.")
            return

    for d in existing:
        p = Path(d)
        for item in p.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        typer.echo(f"  Cleared {d}/")

    # Ensure directories still exist
    for t in targets:
        Path(t).mkdir(parents=True, exist_ok=True)

    typer.echo("Done. Inbox (PDFs) untouched.")
