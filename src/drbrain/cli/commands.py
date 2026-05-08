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
from drbrain.graph.engine import GraphEngine
from drbrain.parser.mineru_parser import extract_pdf
from drbrain.query.tree_retrieval import query_by_structure_hybrid
from drbrain.services.fetch import _resolve_identifier, fetch_paper
from drbrain.storage.database import Database
from drbrain.storage.paths import (
    images_dir,
    raw_md_path,
    source_pdf_path,
    tree_json_path,
)

console = Console()


def ingest_cmd(
    ctx: typer.Context,
    paths: list[str] = typer.Argument(
        None, help="PDF file(s) or directory. Defaults to data/spool/inbox/."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output machine-readable JSON to stdout"
    ),
):
    """Ingest pipeline: parse -> identify -> tree -> paper record.

    Accepts single file, multiple files, or a directory of PDFs.
    Defaults to data/spool/inbox/ when no paths provided.
    """
    cfg = ctx.obj["config"]
    if not paths:
        # Expose API tokens to libraries that read them from environment
        import os as _os

        _dx_token = cfg.get("api", {}).get("deepxiv_token", "")
        if _dx_token and "DEEPXIV_TOKEN" not in _os.environ:
            _os.environ["DEEPXIV_TOKEN"] = _dx_token
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

    db = Database(cfg["db"]["path"])
    dedup = DedupEngine(db)

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
            dedup,
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
    dedup: DedupEngine,
    json_mode: bool = False,
) -> dict:
    """Ingest a single paper: parse -> identify -> tree. Returns {"ok": bool, "local_id": str|None, "error": str|None}."""

    def echo(msg: str):
        if not json_mode:
            typer.echo(msg)

    from loguru import logger as _ingest_log

    # Stage 1: Parse
    echo(f"Parsing: {pdf_path}")
    _ingest_log.info(f"Stage 1: Parse {pdf_path}")
    try:
        parsed = extract_pdf(pdf_path, cfg)
    except Exception as e:
        _ingest_log.error(f"Parse failed for {pdf_path}: {e}")
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
            db.update_paper_venue(
                local_id,
                title=parsed.title,
                year=parsed.year,
                journal=parsed.journal,
                publisher=parsed.publisher,
                citation_count=parsed.citation_count,
            )
            db.commit()
        else:
            local_id = f"p{uuid.uuid4().hex[:6]}"
            db.insert_paper(
                local_id,
                parsed.title,
                parsed.year,
                "uploaded",
                journal=parsed.journal,
                publisher=parsed.publisher,
                citation_count=parsed.citation_count,
            )
            db.insert_paper_ids(
                local_id,
                doi=ids.doi,
                arxiv=ids.arxiv,
                s2_id=parsed.s2_id,
                openalex_id=parsed.openalex_id,
            )
            db.commit()
            echo(f"  [new] {local_id}")
    else:
        db.upgrade_placeholder(local_id)
        db.update_paper_venue(
            local_id,
            title=parsed.title,
            year=parsed.year,
            journal=parsed.journal,
            publisher=parsed.publisher,
            citation_count=parsed.citation_count,
        )
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
    md_path = raw_md_path(paper_dir)
    tree_path = tree_json_path(paper_dir)
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
        tree_path.write_text(doc_tree.to_json(), encoding="utf-8")
        echo(f"  Document tree: {len(doc_tree.structure)} sections → {tree_path.name}")
        # Extract abstract from tree structure
        for node in doc_tree.structure:
            title = node.get("title", "") if isinstance(node, dict) else getattr(node, "title", "")
            if "abstract" in title.lower():
                abstract = ""
                if isinstance(node, dict):
                    abstract = node.get("summary", "") or node.get("content", "")
                else:
                    abstract = getattr(node, "summary", "") or getattr(node, "content", "")
                if abstract.strip():
                    db.set_paper_abstract(local_id, abstract[:2000])
                break
    except Exception as e:
        echo(f"  [yellow]Warning: tree structuring failed: {e}[/yellow]")
        _log_error(cfg, f"Tree structuring failed for {local_id}: {e}")

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
            # Year consistency check: reject if parsed year is >5 years from DOI source
            doi_year = doi_info.get("year")
            if parsed.year and doi_year and abs(parsed.year - doi_year) > 5:
                echo(f"  DOI rejected (year mismatch: paper={parsed.year}, doi={doi_year})")
            else:
                db.conn.execute(
                    "UPDATE paper_ids SET doi = ? WHERE local_id = ?",
                    (doi_info["doi"], local_id),
                )
                db.commit()
                echo(f"  Found DOI: {doi_info['doi']}")
        else:
            echo("  No DOI found in any source")

    # ── Quality Gates (non-blocking) ──────────────────────────────────

    # Gate 1: raw.md size > 200 bytes
    md_path_check = raw_md_path(paper_dir)
    if md_path_check.exists():
        md_size = md_path_check.stat().st_size
        if md_size <= 200:
            echo(
                f"  [yellow]Quality Gate 1: raw.md is only {md_size} bytes (expected > 200)[/yellow]"
            )
            _ingest_log.warning(f"Quality Gate 1 failed for {local_id}: raw.md is {md_size} bytes")
    else:
        echo(f"  [yellow]Quality Gate 1: raw.md not found for {local_id}[/yellow]")
        _ingest_log.warning(f"Quality Gate 1 failed for {local_id}: raw.md missing")

    # Gate 2: title non-empty + year 1900-2030 + has external ID
    gate2_issues = []
    if not parsed.title or not parsed.title.strip():
        gate2_issues.append("empty title")
    if parsed.year is None or not (1900 <= parsed.year <= 2030):
        gate2_issues.append(f"year out of range ({parsed.year})")
    if not ids.doi and not ids.arxiv:
        gate2_issues.append("no external ID (DOI/arXiv)")
    if gate2_issues:
        echo(f"  [yellow]Quality Gate 2: {', '.join(gate2_issues)}[/yellow]")
        _ingest_log.warning(f"Quality Gate 2 failed for {local_id}: {', '.join(gate2_issues)}")

    # Gate 3: post-build — concepts >= 1 and edges >= 1
    concept_count = len(db.get_concepts_by_paper(local_id))
    edge_count = db.conn.execute(
        "SELECT COUNT(*) FROM edges WHERE source_paper = ?", (local_id,)
    ).fetchone()[0]
    if concept_count < 1 or edge_count < 1:
        echo(
            f"  [dim]Quality Gate 3: concepts={concept_count}, edges={edge_count} "
            f"(post-build check — run 'drbrain build {local_id}' to populate)[/dim]"
        )
        _ingest_log.info(
            f"Quality Gate 3 for {local_id}: concepts={concept_count}, edges={edge_count}"
        )

    echo(f"  Ingested: {local_id}")
    return {"ok": True, "local_id": local_id, "report": {"local_id": local_id}}


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
    """Log error via loguru."""
    from loguru import logger

    logger.error(message)


def _resolve_workspace_papers(workspace: str | None) -> set[str] | None:
    """Resolve --workspace flag to a set of paper IDs, or None."""
    # Normalize: when called directly (not via typer CLI), OptionInfo is the default
    if isinstance(workspace, typer.models.OptionInfo):
        workspace = workspace.default
    if not workspace:
        return None
    from drbrain.storage.workspace import load_workspace_papers

    ids = load_workspace_papers(workspace)
    return set(ids) if ids else None


def _resolve_node_type(db: Database, node_id: str) -> tuple[str, dict | None]:
    """Determine node type (Problem/Method/Gap/.../Paper) and optional paper data.

    Returns (type_str, paper_dict_or_None).
    """
    row = db.conn.execute(
        "SELECT type FROM concepts WHERE label = ? LIMIT 1", (node_id,)
    ).fetchone()
    if row:
        return row[0], None
    paper = db.get_paper(node_id)
    if paper:
        return "Paper", paper
    return "Unknown", None


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
    # Move source PDF from inbox to paper directory
    dst_pdf = source_pdf_path(paper_dir)
    if not dst_pdf.exists():
        shutil.copy2(source_pdf, dst_pdf)
        try:
            source_pdf.unlink()
        except OSError:
            pass

    # Copy images and rewrite refs
    raw_md = parsed.raw_md
    if parsed.images_dir and parsed.images_dir.exists():
        img_dst = images_dir(paper_dir)
        shutil.copytree(parsed.images_dir, img_dst, dirs_exist_ok=True)
        # MinerU outputs "images/<hash>/file.jpg", rewrite to "images/<hash>/file.jpg"
        # (no local_id prefix needed — images/ is already inside paper_dir)
        raw_md = re.sub(
            r"!\[(.*?)\]\(images/([^)]+)\)",
            r"![\1](images/\2)",
            raw_md,
        )

    md_path = raw_md_path(paper_dir)
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


def fetch_cmd(
    ctx: typer.Context,
    identifier: str = typer.Argument(..., help="DOI, title, or arXiv ID to fetch"),
    arxiv: bool = typer.Option(False, "--arxiv", help="Treat identifier as arXiv ID"),
):
    """Fetch a paper: find PDF from open access sources -> download -> ingest."""
    # Normalize typer params when called directly (not through CLI)
    if isinstance(arxiv, typer.models.OptionInfo):
        arxiv = arxiv.default

    cfg = ctx.obj["config"]

    doi, title, arxiv_id = _resolve_identifier(identifier, is_arxiv=arxiv)

    fetch_cfg = cfg.get("fetch", {})

    typer.echo(f"Fetching: {identifier}")
    result = fetch_paper(doi=doi, title=title, arxiv_id=arxiv_id, fetch_config=fetch_cfg)

    if not result:
        typer.echo("Could not find a downloadable PDF from any source.", err=True)
        raise typer.Exit(1)

    typer.echo(f"  Downloaded: {result['pdf_path']}")

    # Ingest the downloaded paper
    pdf_path = Path(result["pdf_path"])
    db = Database(cfg["db"]["path"])
    dedup = DedupEngine(db)
    ingest_result = _ingest_single_paper(pdf_path, cfg, db, dedup, json_mode=False)
    db.close()

    if ingest_result.get("ok"):
        typer.echo(f"  Ingested: {ingest_result.get('local_id')}")
        typer.echo(f"  Next: drbrain build {ingest_result.get('local_id')}")
    else:
        typer.echo(f"  Ingest failed: {ingest_result.get('error', 'unknown error')}", err=True)
        raise typer.Exit(1)


def citations_cmd(
    ctx: typer.Context,
    local_id: str = typer.Argument(..., help="Paper local_id"),
    ctype: str = typer.Option(
        "all", "--type", "-t", help="Query type: refs, citing, shared-refs, all"
    ),
    limit: int = typer.Option(200, "--limit", "-l", help="Max results per type"),
    sort: str = typer.Option(
        "cited_by_count:desc",
        "--sort",
        "-s",
        help="Sort: cited_by_count:desc, publication_date:desc, relevance_score:desc",
    ),
    workspace: str = typer.Option(None, "--workspace", "-w", help="Limit to workspace"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
    fetch_interested: bool = typer.Option(
        False, "--fetch-interested", help="Interactively select and fetch placeholder papers"
    ),
):
    """Query citation graph for a paper: refs, citing, shared-refs."""
    # Normalize typer params when called directly (not through CLI)
    if isinstance(ctype, typer.models.OptionInfo):
        ctype = ctype.default
    if isinstance(workspace, typer.models.OptionInfo):
        workspace = workspace.default
    if isinstance(json_output, typer.models.OptionInfo):
        json_output = json_output.default
    if isinstance(fetch_interested, typer.models.OptionInfo):
        fetch_interested = fetch_interested.default

    if ctype not in ("refs", "citing", "shared-refs", "all"):
        typer.echo("Type must be: refs, citing, shared-refs, all", err=True)
        raise typer.Exit(1)

    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])

    paper = db.get_paper(local_id)
    if not paper:
        db.close()
        typer.echo(f"Paper not found: {local_id}", err=True)
        raise typer.Exit(1)

    from drbrain.storage.citation_graph import query_citation_graph

    # Auto-expand citations if none stored yet
    existing = db.conn.execute(
        "SELECT COUNT(*) FROM citation_cache WHERE source_paper = ?", (local_id,)
    ).fetchone()[0]
    if existing == 0:
        typer.echo("  Expanding citations (OpenAlex + S2 + CrossRef)...")
        from drbrain.extractor.citation import expand_citations_multi

        refs_added, citing_added = expand_citations_multi(db, local_id, limit=limit, sort=sort)
        typer.echo(f"  Found {refs_added} references, {citing_added} citing")

    result = query_citation_graph(local_id, db.conn, ctype=ctype)

    if workspace:
        paper_ids = _resolve_workspace_papers(workspace)
        if paper_ids and result.get("refs"):
            result["refs"] = [r for r in result["refs"] if r.get("local_id", "") in paper_ids]

    db.close()

    if json_output:
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return

    p = result["paper"]
    c = result.get("counts", {})
    typer.echo(f"\nCitation Graph: {p['title']} ({p['year']})")
    typer.echo(f"  References: {c.get('references', 0)} | Cited by: {c.get('citing', 0)}")

    if result.get("refs"):
        typer.echo("\nReferences:")
        for r in result["refs"]:
            year_str = f" ({r['year']})" if r.get("year") else ""
            typer.echo(f"  - {r['title']}{year_str}")

    if result.get("citing"):
        typer.echo("\nCited by:")
        for cit in result["citing"]:
            year_str = f" ({cit['year']})" if cit.get("year") else ""
            typer.echo(f"  - {cit['title']}{year_str}")

    if result.get("shared_refs"):
        typer.echo("\nShared References:")
        for sr in result["shared_refs"]:
            tag = " [unlinked]" if sr["status"] == "unlinked" else ""
            typer.echo(f"  - {sr['shared_with_title']} ({sr['shared_count']} shared){tag}")

    # --fetch-interested: interactive selection and batch fetch
    if fetch_interested:
        _fetch_citations_interested(ctx, result)


def _fetch_citations_interested(ctx: typer.Context, result: dict) -> None:
    """Interactively let user select and fetch papers from citation results."""
    # Collect selectable papers: refs + citing, with DOI
    selectable: list[dict] = []
    seen_dois: set[str] = set()

    for source in ("refs", "citing"):
        for entry in result.get(source, []):
            doi = entry.get("doi", "")
            if not doi or not doi.startswith("10."):
                continue
            if doi in seen_dois:
                continue
            seen_dois.add(doi)

            # Skip papers that already have a PDF
            local_id = entry.get("local_id", "") or ""
            if local_id:
                from drbrain.storage.paths import paper_dir, source_pdf_path

                papers_root = Path(ctx.obj["config"].get("dirs", {}).get("papers", "data/papers"))
                pdir = paper_dir(Path(papers_root), local_id)
                if source_pdf_path(pdir).exists():
                    typer.echo(
                        f"  Skipping {local_id} ({entry.get('title', '')[:60]}) — already downloaded"
                    )
                    continue

            selectable.append(entry)

    if not selectable:
        typer.echo("No placeholder papers available to fetch (all have PDFs or no DOI).")
        return

    # Show selectable papers
    typer.echo(f"\n--- Selectable Papers ({len(selectable)}) ---")
    for i, entry in enumerate(selectable):
        title = (entry.get("title", "") or "")[:70]
        doi = entry.get("doi", "")
        year = entry.get("year", "")
        local_id = entry.get("local_id", "") or "?"
        typer.echo(f"  [{i + 1}] {title} ({year})  DOI: {doi}  local: {local_id}")

    # Get user selection
    choice = typer.prompt(
        "\nEnter numbers (comma-separated) to fetch, or 'a' for all, or Enter to skip"
    ).strip()

    if not choice:
        return

    if choice.lower() == "a":
        indices = list(range(len(selectable)))
    else:
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(",") if x.strip().isdigit()]
        except ValueError:
            typer.echo("Invalid selection.", err=True)
            return

    if not indices:
        return

    # Fetch selected papers concurrently (respect max_concurrent from config)
    cfg = ctx.obj["config"]
    fetch_cfg = cfg.get("fetch", {})
    max_concurrent = fetch_cfg.get("max_concurrent", 3)

    def _fetch_one(entry: dict) -> tuple[bool, str]:
        """Fetch and ingest a single paper. Returns (ok, local_id_or_error)."""
        doi = entry.get("doi")
        title = entry.get("title")
        typer.echo(f"\nFetching: {doi} — {title}")

        result_fetch = fetch_paper(doi=doi, title=title, fetch_config=fetch_cfg)
        if not result_fetch:
            return (False, "failed to fetch PDF URL")
        typer.echo(f"  Downloaded: {result_fetch['pdf_path']}")

        pdf_path = Path(result_fetch["pdf_path"])
        if not pdf_path.exists():
            return (False, f"PDF not found at {pdf_path}")

        db = Database(cfg["db"]["path"])
        dedup = DedupEngine(db)
        ingest_result = _ingest_single_paper(pdf_path, cfg, db, dedup, json_mode=False)
        db.close()
        if ingest_result.get("ok"):
            return (True, ingest_result.get("local_id", "unknown"))
        else:
            return (False, ingest_result.get("error", "unknown ingest error"))

    valid_entries = [selectable[idx] for idx in indices if 0 <= idx < len(selectable)]
    success = 0
    fail = 0

    from concurrent.futures import ThreadPoolExecutor, as_completed

    from loguru import logger

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        future_map = {executor.submit(_fetch_one, entry): entry for entry in valid_entries}
        for future in as_completed(future_map):
            try:
                ok, msg = future.result()
                if ok:
                    typer.echo(f"  Ingested: {msg}")
                    success += 1
                else:
                    typer.echo(f"  Failed: {msg}", err=True)
                    fail += 1
            except Exception as exc:
                logger.exception(f"Unexpected fetch error: {exc}")
                fail += 1

    typer.echo(f"\nFetch complete: {success} succeeded, {fail} failed")


def check_citations_cmd(
    ctx: typer.Context,
    text: str = typer.Argument(None, help="Text to check citations in"),
    file: str = typer.Option(None, "--file", "-f", help="Read text from file"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Verify in-text citations against local library."""
    # Normalize typer params when called directly (not through CLI)
    if isinstance(text, typer.models.ArgumentInfo):
        text = text.default
    if isinstance(file, typer.models.OptionInfo):
        file = file.default
    if isinstance(json_output, typer.models.OptionInfo):
        json_output = json_output.default

    if file:
        text = Path(file).read_text(encoding="utf-8")

    if not text:
        typer.echo("Provide text or use --file", err=True)
        raise typer.Exit(1)

    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])

    from drbrain.extractor.citation_check import extract_citations, match_citations

    citations = extract_citations(text)
    citations = match_citations(citations, db)
    db.close()

    if json_output:
        result = [
            {
                "author": c.author,
                "year": c.year,
                "raw": c.raw,
                "found": c.found,
                "matched_id": c.matched_id,
                "matched_title": c.matched_title,
            }
            for c in citations
        ]
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if not citations:
        typer.echo("No citations found in text.")
        return

    typer.echo(f"Found {len(citations)} citations:")
    for c in citations:
        if c.found:
            typer.echo(f'  ✓ {c.author} ({c.year}) → {c.matched_id} "{c.matched_title}"')
        else:
            typer.echo(f"  ✗ {c.author} ({c.year}) → no match")


def report_cmd(
    ctx: typer.Context,
    local_id: str,
    json_output: bool = typer.Option(False, "--json", help="Output full report JSON to stdout"),
):
    """Display single-paper report."""
    cfg = ctx.obj["config"]
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
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Output inferred edges but do not persist to database"
    ),
    rule: list[str] = typer.Option(
        None, "--rule", help="Run only the named rule(s). Repeatable. Omit for all."
    ),
    workspace: str = typer.Option(None, "--workspace", "-w", help="Limit to workspace"),
    mode: str = typer.Option("symbolic", "--mode", help="Inference mode: symbolic or hybrid"),
    mine_rules: bool = typer.Option(
        False, "--mine-rules", help="Mine path rules from TransE embeddings"
    ),
    min_confidence: float = typer.Option(
        0.6, "--min-confidence", help="Minimum confidence for mined rules (0.0-1.0)"
    ),
    ground: bool = typer.Option(
        False, "--ground", help="Ground transitive rules as concrete triples (t-norm)"
    ),
):
    """Run rule-based closure on the full graph."""
    # Normalize typer OptionInfo objects when calling directly (not via CLI)
    if isinstance(rule, typer.models.OptionInfo):
        rule = rule.default
    if isinstance(dry_run, typer.models.OptionInfo):
        dry_run = dry_run.default
    if isinstance(json_output, typer.models.OptionInfo):
        json_output = json_output.default
    if isinstance(mode, typer.models.OptionInfo):
        mode = mode.default
    if isinstance(mine_rules, typer.models.OptionInfo):
        mine_rules = mine_rules.default
    if isinstance(min_confidence, typer.models.OptionInfo):
        min_confidence = min_confidence.default
    if isinstance(ground, typer.models.OptionInfo):
        ground = ground.default

    valid_rules = {
        "creates_debate",
        "gap_addressed",
        "indirect_evolution",
        "gap_to_debate",
        "shared_actor",
        "transitive_closure",
        "asymmetric_violations",
        "method_supersedes_problem",
        "challenge_chain",
        "gap_inheritance",
        "indirect_support",
    }
    if rule is not None:
        invalid = set(rule) - valid_rules
        if invalid:
            typer.echo(f"Invalid rule(s): {', '.join(sorted(invalid))}", err=True)
            typer.echo(f"Valid rules: {', '.join(sorted(valid_rules))}", err=True)
            raise typer.Exit(1)

    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])
    graph = GraphEngine()
    paper_ids = _resolve_workspace_papers(workspace)
    graph.load_from_db(db, paper_ids=paper_ids)

    inferred = graph.closure(mode=mode)

    # ── Embedding-driven rule mining ───────────────────────────────────
    if mine_rules:
        from drbrain.extractor.rule_miner import mine_path_rules

        mined_rules = mine_path_rules(graph, db, min_confidence=min_confidence, top_k=20)
        mined_edges = _apply_mined_rules(graph, mined_rules)
        inferred.extend(mined_edges)
        if not json_output:
            typer.echo(
                f"Mined {len(mined_rules)} path rules from embeddings -> {len(mined_edges)} inferred edges"
            )

    # ── Rule grounding (t-norm transitive closure) ──────────────────────
    if ground:
        grounded = graph.ground_rules(min_confidence=min_confidence)
        if grounded:
            inferred.extend(grounded)
            if not json_output:
                typer.echo(f"Grounded {len(grounded)} transitive rule instances (t-norm)")

    if rule is not None:
        rule_set = set(rule)
        inferred = [e for e in inferred if e["relation"] in rule_set]

    if not dry_run:
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


def _apply_mined_rules(graph, mined_rules: list[dict]) -> list[dict]:
    """Apply mined path rules to the graph, returning inferred edges.

    Each mined rule has `body_path` (list of relations) and `head` (inferred relation).
    Matches the path pattern in the graph and infers direct edges with the head relation.
    """
    if not mined_rules:
        return []

    inferred: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    for rule in mined_rules:
        body = rule["body_path"]
        head = rule["head"]
        confidence = rule.get("confidence", 0.5)

        if len(body) < 2:
            continue

        # Build pattern matches: for each 2-hop path matching body_path,
        # infer the head relation between source and target.
        # body_path: [r_i, r_j] means: src -[r_i]-> mid -[r_j]-> dst => src -[head]-> dst
        # Convert to (relation, direction) pattern for matching
        pattern = [(rel, "forward") for rel in body]

        matches = _match_pattern(graph, pattern)
        for src, dst in matches:
            edge_key = (src, dst, head)
            if edge_key not in seen:
                seen.add(edge_key)
                rule_name = f"mined:{head}"
                inferred.append(
                    {
                        "src": src,
                        "dst": dst,
                        "relation": head,
                        "via": rule_name,
                        "confidence": round(float(confidence), 4),
                    }
                )

    return inferred


def _match_pattern(graph, pattern: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Find all node pairs matching a relation path pattern.

    Pattern is a list of (relation, direction) steps where direction is
    always "forward" (src → dst along the relation edge).
    """
    from collections import defaultdict

    if len(pattern) < 2:
        return []

    # Build adjacency indices for each relation in the pattern
    rel_indices = []
    for rel, direction in pattern:
        idx: dict[str, set[str]] = defaultdict(set)
        for u, v, data in graph.graph.edges(data=True):
            if data["relation"] == rel:
                if direction == "forward":
                    idx[v].add(u)  # given v, find u where u→v
                else:
                    idx[u].add(v)
        rel_indices.append(idx)

    first_idx = rel_indices[0]
    results: list[tuple[str, str]] = []
    visited_edges: set[tuple[str, str, str]] = set()

    for middle_node, prev_nodes in first_idx.items():
        for prev in prev_nodes:
            end_nodes = _extend_chain(graph, rel_indices[1:], middle_node)
            for end in end_nodes:
                edge_key = (prev, end, pattern[0][0])
                if edge_key not in visited_edges:
                    visited_edges.add(edge_key)
                    results.append((prev, end))

    return results


def _extend_chain(graph, remaining_indices: list[dict[str, set[str]]], current: str) -> set[str]:
    """Recursively extend a chain through remaining relation indices."""
    if not remaining_indices:
        return {current}

    idx = remaining_indices[0]
    next_nodes = idx.get(current, set())
    if not remaining_indices[1:]:
        return next_nodes

    result: set[str] = set()
    for node in next_nodes:
        result |= _extend_chain(graph, remaining_indices[1:], node)
    return result


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


def export_cmd(
    ctx: typer.Context,
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

    cfg = ctx.obj["config"]
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
    from drbrain.storage.export import _extract_lastname

    lastname = _extract_lastname(first_author)

    return {
        "local_id": local_id,
        "title": paper.get("title", ""),
        "year": paper.get("year"),
        "doi": paper.get("doi", ""),
        "arxiv": paper.get("arxiv", ""),
        "authors": author_list,
        "first_author_lastname": lastname,
        "paper_type": paper.get("paper_type", "paper"),
        "abstract": paper.get("abstract", ""),
        "journal": paper.get("journal", ""),
        "publisher": paper.get("publisher", ""),
        "citation_count": paper.get("citation_count", 0),
        "volume": paper.get("volume", ""),
        "pages": paper.get("pages", ""),
    }


def queue_cmd(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
):
    """List all pending confidence queue items."""
    cfg = ctx.obj["config"]
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
    ctx: typer.Context,
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

    cfg = ctx.obj["config"]
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
    ctx: typer.Context,
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

    cfg = ctx.obj["config"]
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


def delete_cmd(
    ctx: typer.Context,
    local_id: str,
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt"),
    rm_files: bool = typer.Option(False, "--rm-files", help="Also delete paper directory"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
):
    """Delete a paper and all its associated data from the graph."""
    import shutil as _shutil

    cfg = ctx.obj["config"]
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


def backup_cmd(
    ctx: typer.Context,
    output: str = typer.Option(None, "--output", "-o", help="Custom output path"),
    list_only: bool = typer.Option(False, "--list", help="List existing backups"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
):
    """Create or list tar.gz backups of papers, DB, and workspace."""
    from drbrain.storage.backup import create_backup, list_backups

    if list_only:
        backups = list_backups()
        if json_output:
            typer.echo(
                json.dumps(
                    {"backups": [{"name": b.name, "path": str(b)} for b in backups]},
                    indent=2,
                )
            )
            return
        if not backups:
            typer.echo("No backups found.")
            return
        typer.echo(f"Backups ({len(backups)}):")
        for b in backups:
            size_mb = b.stat().st_size / (1024 * 1024)
            typer.echo(f"  {b.name} ({size_mb:.1f} MB)")
        return

    cfg = ctx.obj["config"]
    papers_dir = Path(cfg.get("dirs", {}).get("papers", "data/papers"))
    db_path = Path(cfg.get("db", {}).get("path", "data/drbrain.db"))
    backup_dir = Path(cfg.get("dirs", {}).get("backups", "data/backups"))
    workspace_dir = Path("workspace")
    reports_dir = Path(cfg.get("dirs", {}).get("reports", "data/reports"))

    if output:
        path = create_backup(
            papers_dir=papers_dir,
            db_path=db_path,
            backup_dir=Path(output).parent,
            workspace_dir=workspace_dir if workspace_dir.exists() else None,
            reports_dir=reports_dir if reports_dir.exists() else None,
        )
        dest = Path(output)
        path.rename(dest)
        path = dest
    else:
        path = create_backup(
            papers_dir=papers_dir,
            db_path=db_path,
            backup_dir=backup_dir,
            workspace_dir=workspace_dir if workspace_dir.exists() else None,
            reports_dir=reports_dir if reports_dir.exists() else None,
        )

    if json_output:
        typer.echo(json.dumps({"backup": str(path), "size_bytes": path.stat().st_size}))
        return

    size_mb = path.stat().st_size / (1024 * 1024)
    typer.echo(f"Backup created: {path} ({size_mb:.1f} MB)")


def lineage_cmd(
    ctx: typer.Context,
    author_id: str = typer.Argument(None, help="OpenAlex author ID (e.g., A5023806754)"),
    list_all: bool = typer.Option(False, "--list", help="List all actors with paper counts"),
    name: str = typer.Option(None, "--name", "-n", help="Search actors by display name"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Explore author/research lineage via OpenAlex deduplicated IDs."""
    cfg = ctx.obj["config"]
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


def _enrich_tree_with_sections(tree: dict, graph: GraphEngine, db: Database) -> None:
    """Recursively enrich a genealogy tree with section provenance."""
    labels: list[str] = []

    def _collect(node: dict) -> None:
        for key in ("concept", "label"):
            if key in node:
                labels.append(str(node[key]))
        for child in node.get("children", []):
            _collect(child)

    _collect(tree)
    if not labels:
        return

    section_map = graph.get_section_contexts_batch(db.conn, labels)

    def _enrich(node: dict) -> None:
        for key in ("concept", "label"):
            if key in node and node[key] in section_map:
                node["section"] = section_map[node[key]]["section"]
                node["node_id"] = section_map[node[key]]["node_id"]
        for child in node.get("children", []):
            _enrich(child)

    _enrich(tree)


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
        db2 = Database(cfg["db"]["path"])
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


def check_cmd(ctx: typer.Context):
    """Check dependencies, configuration, and environment variables."""
    console = Console()
    warnings = []
    errors = []

    cfg = ctx.obj["config"]

    console.print("\n[bold]DrBrain — Dependency & Configuration Check[/bold]\n")

    # -- Python packages --
    console.print("[bold]Python Packages[/bold]")
    table = Table(show_header=False, box=None, padding=(0, 2))
    required_packages = [
        ("pymupdf", "fitz"),
        ("litellm", "litellm"),
        ("typer", "typer"),
        ("rich", "rich"),
        ("pyyaml", "yaml"),
        ("pydantic", "pydantic"),
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
    mineru_found = shutil.which("mineru-open-api")
    pymupdf_found = False
    try:
        importlib.import_module("fitz")
        pymupdf_found = True
    except ImportError:
        pass

    cli_tools = {
        "mineru-open-api": ("MinerU PDF parser CLI", mineru_found),
        "PyMuPDF (fitz)": ("PDF fallback parser", pymupdf_found),
    }
    for tool, (desc, found) in cli_tools.items():
        if found:
            label = f"  {tool}"
            if isinstance(found, str):
                table2.add_row(label, f"[green]OK[/green] ({found}) — {desc}")
            else:
                table2.add_row(label, f"[green]OK[/green] — {desc}")
        else:
            table2.add_row(f"  {tool}", "[yellow]NOT FOUND[/yellow]", f"{desc} (optional)")
            if tool == "mineru-open-api":
                warnings.append("mineru-open-api not found — using PyMuPDF fallback")

    console.print(table2)
    # Parser path
    console.print("\n[bold]Parser Path[/bold]")
    if mineru_found:
        console.print("  MinerU CLI → [green]PyMuPDF[/green]")
    elif pymupdf_found:
        console.print("  [yellow]MinerU not found[/yellow], using: [green]PyMuPDF[/green]")
    else:
        console.print("  [red]No PDF parser available[/red] — install pymupdf")

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

    # Check key values in config
    try:
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
        dirs_config = cfg.get("dirs", {})
        dir_paths = (
            list(dirs_config.values())
            if dirs_config
            else [
                "data/spool/inbox",
                "data/spool/pending",
                "data/papers",
                "data/reports",
                "data/cache",
                "data/logs",
            ]
        )
        for dir_path in dir_paths:
            p = Path(dir_path)
            if p.exists():
                table5.add_row(f"  {dir_path}", "[green]Exists[/green]")
            else:
                p.mkdir(parents=True, exist_ok=True)
                table5.add_row(f"  {dir_path}", "[green]Created[/green]")
    except Exception:
        for d in ["data/spool/inbox", "data/spool/pending", "data/papers"]:
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
        db_path = cfg.get("db", {}).get("path", "data/drbrain.db")
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

    # -- Paper count --
    console.print("\n[bold]Library[/bold]")
    table_lib = Table(show_header=False, box=None, padding=(0, 2))
    try:
        db = Database(cfg["db"]["path"])
        paper_count = db.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        concept_count = db.conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
        table_lib.add_row("  Papers", f"[green]{paper_count}[/green]")
        table_lib.add_row("  Concepts", f"[green]{concept_count}[/green]")
        db.close()
    except Exception:
        table_lib.add_row("  (db unavailable)", "[yellow]Unknown[/yellow]")
    console.print(table_lib)

    # -- Disk space --
    console.print("\n[bold]Disk Space[/bold]")
    table_disk = Table(show_header=False, box=None, padding=(0, 2))
    data_path = Path("data")
    if data_path.exists():
        usage = shutil.disk_usage(data_path)
        free_gb = usage.free / (1024**3)
        total_gb = usage.total / (1024**3)
        if free_gb < 1:
            table_disk.add_row(
                "  data/ free space",
                f"[red]{free_gb:.1f} GB[/red]",
                f"(total {total_gb:.0f} GB) — critically low",
            )
            warnings.append(f"Low disk space: {free_gb:.1f} GB free on data/ partition")
        elif free_gb < 10:
            table_disk.add_row(
                "  data/ free space",
                f"[yellow]{free_gb:.1f} GB[/yellow]",
                f"(total {total_gb:.0f} GB)",
            )
        else:
            table_disk.add_row(
                "  data/ free space",
                f"[green]{free_gb:.1f} GB[/green]",
                f"(total {total_gb:.0f} GB)",
            )
    console.print(table_disk)

    # -- MinerU API connectivity --
    console.print("\n[bold]API Connectivity[/bold]")
    table_api = Table(show_header=False, box=None, padding=(0, 2))
    try:
        mineru_token = cfg.get("mineru", {}).get("token", "")
        if mineru_token and not mineru_token.startswith("${"):
            import urllib.request as _urllib

            try:
                req = _urllib.request.Request(
                    "https://api.mineru.com/api/v1/status",
                    headers={"Authorization": f"Bearer {mineru_token}"},
                )
                _urllib.request.urlopen(req, timeout=5)
                table_api.add_row("  MinerU API", "[green]Reachable[/green]")
            except Exception:
                table_api.add_row(
                    "  MinerU API", "[yellow]Unreachable[/yellow]", "(check token/network)"
                )
                warnings.append("MinerU API unreachable (token-tier may not work)")
        else:
            table_api.add_row(
                "  MinerU API", "[yellow]Not configured[/yellow]", "(flash mode will be used)"
            )
    except Exception:
        table_api.add_row("  MinerU API", "[yellow]Unknown[/yellow]")

    # -- MinerU CLI --
    mineru_cli_available = False
    try:
        import shutil as _shutil

        cli = _shutil.which("mineru-open-api")
        if cli:
            mineru_cli_available = True
            table_api.add_row("  MinerU CLI", f"[green]Found[/green] ({cli})")
        else:
            table_api.add_row(
                "  MinerU CLI", "[yellow]Not found[/yellow]", "(install: npm i -g mineru-open-api)"
            )
    except Exception:
        table_api.add_row("  MinerU CLI", "[yellow]Unknown[/yellow]")

    # Only warn about PyMuPDF fallback if no MinerU path works
    if not mineru_cli_available:
        warnings.append("MinerU unavailable — PDF parsing will use PyMuPDF fallback")

    # -- DeepXiv connectivity --
    try:
        dx_token = cfg.get("api", {}).get("deepxiv_token", "")
        if dx_token and not dx_token.startswith("${"):
            try:
                from deepxiv_sdk import Reader as _dxReader

                r = _dxReader(token=dx_token)
                r.brief("1706.03762")
                table_api.add_row("  DeepXiv", "[green]Reachable[/green]")
            except Exception:
                table_api.add_row(
                    "  DeepXiv", "[yellow]Unreachable[/yellow]", "(check token at data.rag.ac.cn)"
                )
        else:
            table_api.add_row(
                "  DeepXiv",
                "[yellow]Not configured[/yellow]",
                "(register at https://data.rag.ac.cn/register)",
            )
    except Exception:
        table_api.add_row("  DeepXiv", "[yellow]Unknown[/yellow]")

    # -- LLM API connectivity --
    try:
        llm_models = cfg.get("llm", {}).get("models", [])
        for i, m in enumerate(llm_models):
            label = f"  LLM [{i}] {m.get('provider', '?')}/{m.get('model', '?')}"
            api_key = m.get("api_key", "")
            if api_key and api_key.startswith("${"):
                table_api.add_row(label, "[yellow]Env var not set[/yellow]")
                continue
            try:
                import litellm as _llm

                name = f"{m['provider']}/{m['model']}"
                kwargs = {
                    "model": name,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 5,
                    "timeout": 10,
                }
                if m.get("api_key"):
                    kwargs["api_key"] = m["api_key"]
                if m.get("base_url"):
                    kwargs["api_base"] = m["base_url"]
                _llm.completion(**kwargs)
                table_api.add_row(label, "[green]Reachable[/green]")
            except Exception as e:
                err_msg = str(e)[:60]
                table_api.add_row(label, "[yellow]Unreachable[/yellow]", f"({err_msg})")
                warnings.append(f"LLM [{i}] {m.get('model', '?')} unreachable")
    except Exception:
        table_api.add_row("  LLM", "[yellow]Not configured[/yellow]", "(run `drbrain setup`)")

    console.print(table_api)

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


def analyze_cmd(
    ctx: typer.Context,
    local_id: str = typer.Argument(None, help="Paper local_id (single paper mode)"),
    papers: str = typer.Option(None, "--papers", help="Comma-separated paper IDs"),
    query: str = typer.Option(None, "--query", help="BM25 search query to select papers"),
    discover: str = typer.Option(None, "--discover", help="LLM graph discovery question"),
    workspace: str = typer.Option(None, "--workspace", "-w", help="Workspace boundary scan"),
    full: bool = typer.Option(False, "--full", "-f", help="Full analysis (slower)"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Analyze knowledge frontier: seeds, causal chains, hypotheses, and more.

    Paper selection (mutually exclusive, first match wins):
    - <local_id>: single paper
    - --papers p1,p2,...: specific papers
    - --query "text": BM25 search then analyze matches
    - --discover "question": LLM graph exploration to find relevant papers
    - --workspace myws: all papers in workspace
    - (none): error — specify one of the above
    """
    # Normalize typer OptionInfo objects when calling directly (not via CLI)
    if isinstance(papers, typer.models.OptionInfo):
        papers = papers.default
    if isinstance(query, typer.models.OptionInfo):
        query = query.default
    if isinstance(discover, typer.models.OptionInfo):
        discover = discover.default
    if isinstance(workspace, typer.models.OptionInfo):
        workspace = workspace.default
    if isinstance(full, typer.models.OptionInfo):
        full = full.default
    if isinstance(json_output, typer.models.OptionInfo):
        json_output = json_output.default

    from drbrain.report.analyzer import analyze_paper

    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])
    graph = GraphEngine()
    llm_models = cfg.get("llm", {}).get("models", [])

    # ── Paper selection ──
    selected: list[dict] = []

    if local_id:
        p = db.get_paper(local_id)
        if p:
            selected = [p]
        else:
            typer.echo(f"Paper not found: {local_id}", err=True)
            db.close()
            raise typer.Exit(1)
    elif papers:
        for pid in papers.split(","):
            pid = pid.strip()
            p = db.get_paper(pid)
            if p:
                selected.append(p)
            else:
                typer.echo(f"Paper not found: {pid}", err=True)
    elif query:
        from drbrain.query.bm25 import build_bm25_index

        bm25 = build_bm25_index(db)
        results = bm25.search(query, limit=20)
        seen = set()
        for r in results:
            pid = r["local_id"]
            if pid not in seen:
                seen.add(pid)
                p = db.get_paper(pid)
                if p:
                    selected.append(p)
        typer.echo(f"Query '{query}': {len(selected)} papers matched")
    elif discover:
        from drbrain.extractor.reasoner import ReasonerAgent

        agent = ReasonerAgent(db=db, graph_engine=graph, models=llm_models)
        typer.echo(f"Discovering papers for: {discover}")
        answer = asyncio.run(
            agent.reason(
                f"Find papers in the knowledge graph relevant to: {discover}. "
                "Search concepts and explore neighbors. Return ONLY a comma-separated "
                "list of the most relevant paper IDs (like pe211dc,p6a321e). Max 10."
            )
        )
        import re as _re

        ids = _re.findall(r"p[a-f0-9]{6}", answer)
        for pid in ids[:10]:
            p = db.get_paper(pid)
            if p:
                selected.append(p)
        typer.echo(f"Discovered: {len(selected)} papers")
    elif workspace:
        paper_ids = _resolve_workspace_papers(workspace)
        all_papers = db.get_all_papers()
        selected = [p for p in all_papers if paper_ids and p["local_id"] in paper_ids]
    else:
        typer.echo(
            "Specify a paper ID, --papers, --query, --discover, or --workspace.",
            err=True,
        )
        db.close()
        raise typer.Exit(1)

    if not selected:
        typer.echo("No papers to analyze.")
        db.close()
        raise typer.Exit(1)

    # Load full graph for seed detection and cross-paper insights
    graph.load_from_db(db)

    # ── Run analysis ──
    reports = [
        analyze_paper(db, graph, p["local_id"], full=full, models=llm_models) for p in selected
    ]

    # Add cross-paper insights for multi-paper analysis
    if len(reports) > 1:
        from drbrain.report.analyzer import add_cross_paper_insights

        reports = add_cross_paper_insights(reports, db=db)

    db.close()

    if json_output:
        typer.echo(json.dumps(reports, indent=2, ensure_ascii=False, default=str))
    elif len(reports) == 1:
        _print_analyze_report(reports[0])
    else:
        typer.echo(f"Analysis: {len(reports)} papers\n")
        for r in reports:
            _print_analyze_report(r)


def _print_analyze_report(report: dict) -> None:
    """Print a formatted analysis report."""
    if "error" in report:
        typer.echo(f"Error: {report['error']}", err=True)
        return

    if report.get("executive_summary"):
        typer.echo("\n[bold]Executive Summary[/bold]")
        typer.echo(f"  {report['executive_summary']}")

    p = report["paper"]
    s = report["summary"]
    typer.echo(f"\n[bold]Knowledge Frontier: {p['title']} ({p['year']})[/bold]")

    if report.get("cross_paper_insights"):
        insights = report["cross_paper_insights"]
        typer.echo(f"\n[bold]── Cross-paper Insights ({len(insights)})[/bold]")
        for ins in insights[:5]:
            typer.echo(f"  Method '{ins['method']}' ({ins['method_paper']})")
            typer.echo(f"    → could address Problem '{ins['problem']}' ({ins['problem_paper']})")
            typer.echo(f"    (similarity: {ins['similarity']})")

    typer.echo(f"\n[bold]── Research Seeds ({s['seeds']})[/bold]")
    for seed in report.get("seeds", []):
        typer.echo(
            f"  [{seed.get('type', '?')}] {seed.get('concept', '?')}: {seed.get('description', '?')}"
        )
        if seed.get("suggested_solutions"):
            typer.echo(f"    → {seed['suggested_solutions']}")

    typer.echo(f"\n[bold]── Causal Chains ({s['causal_chains']})[/bold]")
    for chain in report.get("causal_chains", []):
        typer.echo(f"  {chain['source']} → {chain['target']} (via: {chain['via']})")

    typer.echo(f"\n[bold]── Inferred Edges ({s['inferred_edges']})[/bold]")

    if report.get("critical_nodes"):
        typer.echo(f"\n[bold]── Critical Nodes ({s['critical_nodes']})[/bold]")
        for node in report["critical_nodes"]:
            typer.echo(f"  {node}")

    if report.get("hypotheses"):
        typer.echo(f"\n[bold]── Hypotheses ({s['hypotheses']})[/bold]")
        for hyp in report["hypotheses"]:
            typer.echo(f"  [{hyp['type']}] {hyp['description']} ({hyp['confidence']:.2f})")

    if report.get("isomorphisms"):
        typer.echo(f"\n[bold]── Isomorphisms ({s['isomorphisms']})[/bold]")
        for iso in report["isomorphisms"]:
            typer.echo(f"  {iso['pattern']} ({iso['similarity']:.2f})")

    typer.echo()


def clean_cmd(
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt"),
    config_path: str = typer.Option("config.yaml", "--config", "-c", help="Config file path"),
) -> None:
    """Clear data directories (db, cache, logs, papers, reports). Keeps inbox (PDFs) intact."""
    cfg = load_config(config_path)
    dirs = cfg.get("dirs", {})

    targets = [
        cfg.get("db", {}).get("path", "data/drbrain.db"),
        "data/metrics.db",
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

    if force:
        from drbrain.auth import has_password, verify_password

        if has_password(cfg):
            pw = typer.prompt("Admin password", hide_input=True)
            if not verify_password(pw, cfg["admin"]["password_hash"]):
                typer.echo("Wrong password.", err=True)
                raise typer.Exit(1)

    for d in existing:
        p = Path(d)
        if p.is_file():
            p.unlink()
            typer.echo(f"  Removed {d}")
        elif p.is_dir():
            for item in p.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            typer.echo(f"  Cleared {d}/")

    # Recreate directories (skip file paths)
    for t in targets:
        p = Path(t)
        if not p.suffix:  # directory path (no extension)
            p.mkdir(parents=True, exist_ok=True)

    typer.echo("Done. Inbox (PDFs) untouched.")


# -- Workspace commands --


def ws_create_cmd(
    name: str = typer.Argument(..., help="Workspace name"),
    description: str = typer.Option("", "--description", "-d", help="Description"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Create a new workspace."""
    from drbrain.storage.workspace import WorkspaceError, create_workspace

    try:
        create_workspace(name, description=description)
        if json_output:
            typer.echo(json.dumps({"created": name, "description": description}))
        else:
            typer.echo(f"Workspace created: {name}")
    except WorkspaceError as e:
        if json_output:
            typer.echo(json.dumps({"error": str(e)}))
        else:
            typer.echo(str(e), err=True)
        raise typer.Exit(1)


def ws_add_cmd(
    name: str = typer.Argument(..., help="Workspace name"),
    local_ids: list[str] = typer.Argument(..., help="Paper local_id(s) to add"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Add papers to a workspace."""
    from drbrain.storage.workspace import WorkspaceError, add_papers, get_workspace

    try:
        add_papers(name, local_ids)
        ws = get_workspace(name)
        if json_output:
            typer.echo(json.dumps(ws, indent=2))
        else:
            typer.echo(f"Added {len(local_ids)} paper(s) to '{name}' ({ws['paper_count']} total)")
    except WorkspaceError as e:
        if json_output:
            typer.echo(json.dumps({"error": str(e)}))
        else:
            typer.echo(str(e), err=True)
        raise typer.Exit(1)


def ws_remove_cmd(
    name: str = typer.Argument(..., help="Workspace name"),
    local_ids: list[str] = typer.Argument(..., help="Paper local_id(s) to remove"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Remove papers from a workspace."""
    from drbrain.storage.workspace import get_workspace, remove_papers

    remove_papers(name, local_ids)
    ws = get_workspace(name)
    if json_output:
        typer.echo(json.dumps(ws, indent=2))
    else:
        typer.echo(f"Removed {len(local_ids)} paper(s) from '{name}' ({ws['paper_count']} total)")


def ws_list_cmd(
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """List all workspaces."""
    from drbrain.storage.workspace import list_workspaces

    names = list_workspaces()
    if json_output:
        typer.echo(json.dumps({"workspaces": names}))
    elif not names:
        typer.echo("No workspaces. Create one with: drbrain ws create <name>")
    else:
        typer.echo(f"Workspaces ({len(names)}):")
        for n in names:
            typer.echo(f"  {n}")


def ws_show_cmd(
    name: str = typer.Argument(..., help="Workspace name"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Show workspace details and paper list."""
    from drbrain.storage.workspace import get_workspace

    ws = get_workspace(name)
    if ws is None:
        msg = f"Workspace not found: {name}"
        if json_output:
            typer.echo(json.dumps({"error": msg}))
        else:
            typer.echo(msg, err=True)
        raise typer.Exit(1)

    if json_output:
        typer.echo(json.dumps(ws, indent=2, default=str))
        return

    typer.echo(f"Workspace: {ws['name']}")
    typer.echo(f"  Description: {ws['description']}")
    typer.echo(f"  Created: {ws['created']}")
    typer.echo(f"  Papers: {ws['paper_count']}")
    for pid in ws["papers"]:
        typer.echo(f"    - {pid}")


def ws_delete_cmd(
    name: str = typer.Argument(..., help="Workspace name"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Delete a workspace."""
    from drbrain.storage.workspace import delete_workspace, get_workspace

    ws = get_workspace(name)
    if ws is None:
        msg = f"Workspace not found: {name}"
        if json_output:
            typer.echo(json.dumps({"error": msg}))
        else:
            typer.echo(msg, err=True)
        raise typer.Exit(1)

    delete_workspace(name)
    if json_output:
        typer.echo(json.dumps({"deleted": name}))
    else:
        typer.echo(f"Workspace deleted: {name}")


def ws_rename_cmd(
    old_name: str = typer.Argument(..., help="Current workspace name"),
    new_name: str = typer.Argument(..., help="New workspace name"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Rename a workspace."""
    from drbrain.storage.workspace import rename_workspace

    try:
        new_path = rename_workspace(old_name, new_name)
        if json_output:
            typer.echo(json.dumps({"renamed": old_name, "to": new_name, "path": str(new_path)}))
        else:
            typer.echo(f"Workspace renamed: {old_name} -> {new_name}")
    except (ValueError, FileNotFoundError, FileExistsError) as e:
        if json_output:
            typer.echo(json.dumps({"error": str(e)}))
        else:
            typer.echo(str(e), err=True)
        raise typer.Exit(1)


# -- repair + import commands --


def repair_cmd(
    ctx: typer.Context,
    local_id: str = typer.Argument(None, help="Paper local_id"),
    all: bool = typer.Option(False, "--all", help="Repair all papers"),
    workspace: str = typer.Option(None, "--workspace", "-w", help="Limit to workspace"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview only, no changes"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Repair paper metadata via CrossRef, arXiv, and OpenAlex."""
    from drbrain.services.repair import repair_paper

    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])

    if all or workspace:
        papers = db.get_all_papers()
        if workspace:
            ws_ids = _resolve_workspace_papers(workspace)
            papers = [p for p in papers if ws_ids and p["local_id"] in ws_ids]
    elif local_id:
        paper = db.get_paper(local_id)
        if not paper:
            db.close()
            typer.echo(f"Paper not found: {local_id}", err=True)
            raise typer.Exit(1)
        papers = [paper]
    else:
        db.close()
        typer.echo("Specify a paper, --all, or --workspace", err=True)
        raise typer.Exit(1)

    all_repairs = []
    for paper in papers:
        if not paper:
            continue
        repairs = repair_paper(db, paper["local_id"], dry_run=dry_run)
        if repairs:
            all_repairs.append(
                {"paper": paper["local_id"], "title": paper.get("title", ""), "repairs": repairs}
            )

    db.close()

    if json_output:
        typer.echo(json.dumps(all_repairs, indent=2, ensure_ascii=False, default=str))
        return

    total_fixed = sum(len(r["repairs"]) for r in all_repairs)
    if dry_run:
        typer.echo(
            f"[DRY RUN] Would repair {total_fixed} fields across {len(all_repairs)} papers:\n"
        )
    else:
        typer.echo(f"Repaired {total_fixed} fields across {len(all_repairs)} papers:\n")

    for r in all_repairs:
        typer.echo(f'  {r["paper"]} "{r["title"][:60]}"')
        for repair in r["repairs"]:
            old_str = str(repair.get("old", "")) if repair.get("old") is not None else "(empty)"
            new_str = str(repair.get("new", "")) if repair.get("new") is not None else "(empty)"
            typer.echo(f"    {repair['field']}: {old_str} → {new_str} ({repair.get('source')})")


def import_cmd(
    ctx: typer.Context,
    source: str = typer.Argument(..., help="Source type: zotero, bibtex, or endnote"),
    path: str = typer.Argument(..., help="Path to zotero.sqlite, .bib, .ris, or .xml file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview only"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
    list_collections: bool = typer.Option(
        False, "--list-collections", help="List collections and exit"
    ),
    collection: str = typer.Option(None, "--collection", help="Filter by collection key"),
    api_key: str = typer.Option(None, "--api-key", help="Zotero API key (for Web API mode)"),
    library_id: str = typer.Option(
        None, "--library-id", help="Zotero library ID (for Web API mode)"
    ),
    library_type: str = typer.Option(
        "user", "--library-type", help="Zotero library type: user or group"
    ),
    no_pdf: bool = typer.Option(False, "--no-pdf", help="Skip PDF detection/download"),
    import_collections: bool = typer.Option(
        False, "--import-collections", help="Create workspaces per collection after import"
    ),
):
    """Import papers from Zotero, BibTeX, or Endnote.

    Sources:
      zotero  - Zotero local SQLite database or Web API
      bibtex  - BibTeX .bib file
      endnote - Endnote .xml or .ris export
    """
    import uuid

    if source not in ("zotero", "bibtex", "endnote"):
        typer.echo("Source must be: zotero, bibtex, or endnote", err=True)
        raise typer.Exit(1)

    # --list-collections mode (zotero only)
    if list_collections:
        if source != "zotero":
            typer.echo("--list-collections only supported for zotero source", err=True)
            raise typer.Exit(1)

        if library_id and api_key:
            from drbrain.services.zotero_import import list_collections_api

            try:
                collections = list_collections_api(library_id, api_key, library_type=library_type)
            except ImportError as e:
                typer.echo(str(e), err=True)
                raise typer.Exit(1)
        else:
            p = Path(path)
            if not p.exists():
                typer.echo(f"File not found: {path}", err=True)
                raise typer.Exit(1)
            from drbrain.services.zotero_import import list_collections_local

            try:
                collections = list_collections_local(p)
            except Exception as e:
                typer.echo(f"Failed to list collections: {e}", err=True)
                raise typer.Exit(1)

        if json_output:
            typer.echo(json.dumps(collections, indent=2, ensure_ascii=False))
        else:
            if not collections:
                typer.echo("No collections found.")
            else:
                typer.echo(f"{'KEY':<10} {'ITEMS':>6}  NAME")
                typer.echo("-" * 50)
                for c in collections:
                    typer.echo(f"{c['key']:<10} {c['numItems']:>6}  {c['name']}")
        return

    # Parse papers based on source
    papers: list[dict] = []

    if source == "zotero":
        if library_id and api_key:
            # Web API mode
            from drbrain.services.zotero_import import fetch_zotero_api

            try:
                papers = fetch_zotero_api(
                    library_id,
                    api_key,
                    library_type=library_type,
                    collection_key=collection,
                )
            except ImportError as e:
                typer.echo(str(e), err=True)
                raise typer.Exit(1)
        else:
            # Local SQLite mode
            p = Path(path)
            if not p.exists():
                typer.echo(f"File not found: {path}", err=True)
                raise typer.Exit(1)

            import sqlite3

            from drbrain.services.zotero_import import import_zotero_db

            conn = sqlite3.connect(str(p))
            try:
                zotero_storage = None
                if not no_pdf:
                    zotero_storage = p.parent / "storage"
                    if not zotero_storage.exists():
                        zotero_storage = None

                papers = import_zotero_db(
                    conn,
                    collection_key=collection,
                    storage_dir=zotero_storage,
                )
            finally:
                conn.close()

    elif source == "bibtex":
        p = Path(path)
        if not p.exists():
            typer.echo(f"File not found: {path}", err=True)
            raise typer.Exit(1)

        from drbrain.services.zotero_import import import_bibtex_file

        papers = import_bibtex_file(p)

    elif source == "endnote":
        p = Path(path)
        if not p.exists():
            typer.echo(f"File not found: {path}", err=True)
            raise typer.Exit(1)

        suffix = p.suffix.lower()
        if suffix == ".ris":
            from drbrain.services.zotero_import import parse_endnote_ris

            papers = parse_endnote_ris(p)
        elif suffix == ".xml":
            from drbrain.services.zotero_import import parse_endnote_xml

            try:
                papers = parse_endnote_xml(p)
            except ImportError as e:
                typer.echo(str(e), err=True)
                raise typer.Exit(1)
        else:
            typer.echo("Endnote source requires .ris or .xml file", err=True)
            raise typer.Exit(1)

    # Handle empty results
    if not papers:
        typer.echo("No papers found to import.")
        return

    # Dry-run: preview only
    if dry_run:
        if json_output:
            typer.echo(
                json.dumps(
                    {"dry_run": True, "count": len(papers), "papers": papers},
                    indent=2,
                    ensure_ascii=False,
                )
            )
        else:
            typer.echo(f"[DRY RUN] Would import {len(papers)} papers:")
            for paper in papers:
                typer.echo(
                    f"  - {paper['title'][:80]} ({paper.get('year', '?')}) [{paper['paper_type']}]"
                )
        return

    # Insert papers into database
    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])

    # Dedup: check DOI before insert
    existing_dois: set[str] = set()
    try:
        all_papers = db.get_all_papers()
        for paper_row in all_papers:
            pid = paper_row.get("local_id", "")
            if pid:
                row = db.conn.execute(
                    "SELECT doi FROM paper_ids WHERE local_id = ?", (pid,)
                ).fetchone()
                if row and row[0]:
                    existing_dois.add(row[0].lower().strip())
    except Exception:
        pass

    imported = []
    skipped_dupes = 0
    for paper in papers:
        # DOI dedup check
        doi = (paper.get("doi") or "").lower().strip()
        if doi and doi in existing_dois:
            skipped_dupes += 1
            continue

        local_id = f"p{uuid.uuid4().hex[:6]}"
        db.insert_paper(
            local_id,
            paper["title"],
            paper.get("year"),
            "placeholder",
            paper_type=paper.get("paper_type", "paper"),
            journal=paper.get("journal", ""),
            publisher=paper.get("publisher", ""),
            citation_count=paper.get("citation_count", 0),
            volume=paper.get("volume", ""),
            pages=paper.get("pages", ""),
        )
        if paper.get("doi"):
            db.insert_paper_ids(local_id, doi=paper["doi"])
        if paper.get("authors"):
            for author in paper["authors"].split(" and "):
                author = author.strip()
                if author:
                    db.insert_concept(local_id, "Actor", author, 1.0, year=paper.get("year"))
                    db.insert_alias(author, author)

        # Copy PDF to paper directory if available
        pdf_src = paper.get("pdf_path", "")
        if pdf_src and not no_pdf:
            pdf_src_path = Path(pdf_src)
            if pdf_src_path.exists():
                import shutil as _shutil

                paper_dir = Path(cfg["db"]["path"]).parent / "papers" / local_id
                paper_dir.mkdir(parents=True, exist_ok=True)
                _shutil.copy2(str(pdf_src_path), str(paper_dir / pdf_src_path.name))

        imported.append({"local_id": local_id, "title": paper["title"]})
        if doi:
            existing_dois.add(doi)

    db.commit()
    db.close()

    if json_output:
        typer.echo(
            json.dumps(
                {
                    "imported": len(imported),
                    "skipped_duplicates": skipped_dupes,
                    "papers": imported,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    msg = f"Imported {len(imported)} papers as placeholders."
    if skipped_dupes:
        msg += f" Skipped {skipped_dupes} duplicates."
    typer.echo(msg)
    typer.echo(
        "Run 'drbrain ingest' to process them, or 'drbrain repair --all' to fix metadata first."
    )


def translate_cmd(
    ctx: typer.Context,
    local_id: str = typer.Argument(..., help="Paper local_id"),
    target_lang: str = typer.Option(
        "zh", "--lang", "-l", help="Target language code: zh, en, ja, etc."
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force re-translation even if output exists"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Translate a paper's markdown via LLM."""
    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])

    paper = db.get_paper(local_id)
    db.close()

    if not paper:
        typer.echo(f"Paper not found: {local_id}", err=True)
        raise typer.Exit(1)

    papers_dir = Path(cfg.get("dirs", {}).get("papers", "data/papers"))
    paper_dir = papers_dir / local_id

    if not raw_md_path(paper_dir).exists():
        typer.echo(f"No raw.md found for {local_id}. Run 'drbrain ingest' first.", err=True)
        raise typer.Exit(1)

    llm_models = cfg.get("llm", {}).get("models", [])
    if not llm_models:
        typer.echo("No LLM models configured.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Translating: {paper['title']} (→ {target_lang})")

    from drbrain.services.translate import translate_paper

    result = translate_paper(
        paper_dir,
        models=llm_models,
        target_lang=target_lang,
        force=force,
    )

    if not result.ok:
        if result.partial:
            msg = f"Partial translation ({result.completed_chunks}/{result.total_chunks} chunks) — re-run to resume"
        elif result.skip_reason:
            msg = f"Translation skipped: {result.skip_reason}"
        else:
            msg = "Translation failed."
        if json_output:
            typer.echo(
                json.dumps(
                    {"error": msg, "partial": result.partial, "skip_reason": result.skip_reason},
                    ensure_ascii=False,
                )
            )
        else:
            typer.echo(msg, err=True)
        if not result.partial:
            raise typer.Exit(1)
        return

    if json_output:
        typer.echo(
            json.dumps(
                {
                    "paper": local_id,
                    "output": str(result.path),
                    "completed_chunks": result.completed_chunks,
                    "total_chunks": result.total_chunks,
                },
                ensure_ascii=False,
            )
        )
    else:
        typer.echo(f"Translated: {result.path}")


def build_cmd(
    ctx: typer.Context,
    paper_id: list[str] = typer.Argument(
        None, help="Paper IDs to build graph for. Omit for all unprocessed."
    ),
    all_papers: bool = typer.Option(
        False, "--all", help="Build graph for all papers in the database"
    ),
    skip_refine: bool = typer.Option(
        False, "--skip-refine", help="Skip iterative refinement stage"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
):
    """Build knowledge graph from ingested papers using 5-stage LLM extraction."""
    from drbrain.extractor.concept import build_graph_from_tree

    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])

    # Select papers to process
    if all_papers:
        papers = db.get_all_papers()
    elif paper_id:
        papers = []
        for pid in paper_id:
            p = db.get_paper(pid)
            if p:
                papers.append(p)
            else:
                typer.echo(f"Paper not found: {pid}", err=True)
    else:
        all_papers = db.get_all_papers()
        papers = [p for p in all_papers if p.get("status") == "uploaded"]

    if not papers:
        typer.echo("No papers to build. Run: drbrain ingest first")
        db.close()
        return

    llm_models = cfg.get("llm", {}).get("models", [])
    if not llm_models:
        typer.echo("No LLM models configured. Run: drbrain setup", err=True)
        db.close()
        raise typer.Exit(1)

    papers_dir = Path(cfg.get("dirs", {}).get("papers", "data/papers"))
    all_results = []

    for paper in papers:
        pid = paper["local_id"]
        typer.echo(f"\n{pid}: {paper['title'][:80]}")

        tree_path = tree_json_path(papers_dir / pid)
        md_path = raw_md_path(papers_dir / pid)

        # Retry tree generation if raw.md exists but tree.json is missing
        if not tree_path.exists() and md_path.exists():
            typer.echo("  Tree missing, retrying...")
            try:
                from drbrain.parser.pageindex_parser import TreeConfig, md_to_tree

                pageindex_cfg = TreeConfig(
                    if_add_node_summary=True,
                    if_add_doc_description=True,
                    if_add_node_text=False,
                    if_add_node_id=True,
                    max_node_tokens=10000,
                    min_token_threshold=5000,
                )
                doc_tree = asyncio.run(
                    md_to_tree(str(md_path), config=pageindex_cfg, models=llm_models)
                )
                tree_path.write_text(doc_tree.to_json(), encoding="utf-8")
                typer.echo(f"  Tree regenerated: {len(doc_tree.structure)} sections")
            except Exception as e:
                typer.echo(f"  Tree regeneration failed: {e}")
                continue
        elif not md_path.exists():
            typer.echo("  No raw.md — ingest this paper first")
            continue

        import json as _json

        tree = _json.loads(tree_path.read_text(encoding="utf-8"))
        structure = tree.get("structure", [])
        if not structure:
            typer.echo("  Empty tree structure — skipping")
            continue

        # Run 5-stage pipeline
        typer.echo("  Stage 1: Ontology...")
        result = asyncio.run(
            build_graph_from_tree(md_path, structure, llm_models, skip_refine=skip_refine)
        )

        concepts = result.get("concepts", [])
        relations = result.get("relations", [])
        merges = result.get("merges", [])
        corrections = result.get("corrections", [])

        typer.echo(f"  Stage 2: Entities...   {len(concepts)} concepts")
        typer.echo(f"  Stage 3: Relations...  {len(relations)} edges")
        typer.echo(f"  Stage 4: Coreference... {len(merges)} merges")
        if not skip_refine:
            typer.echo(f"  Stage 5: Refine...     {len(corrections)} corrections")

        # Validate and insert concepts
        valid_types = {"Problem", "Method", "Conclusion", "Debate", "Gap", "Actor"}
        valid_count = 0
        rejected = 0
        for c in concepts:
            ctype = c.get("type", "")
            label = c.get("label", "")
            conf = c.get("confidence", 0.5)
            if ctype not in valid_types or not label:
                rejected += 1
                continue
            db.insert_concept(
                pid, ctype, label, conf, section=c.get("section", ""), node_id=c.get("node_id", "")
            )
            valid_count += 1

        # Insert relations
        for r in relations:
            head = r.get("head", "")
            rel = r.get("rel", "")
            tail = r.get("tail", "")
            if head and rel and tail:
                try:
                    db.insert_edge(
                        head,
                        tail,
                        rel,
                        pid,
                        node_id=r.get("node_id", ""),
                        section=r.get("section", ""),
                    )
                except Exception:
                    pass  # duplicate edge or invalid reference

        # Mark as extracted
        db.conn.execute("UPDATE papers SET status = 'extracted' WHERE local_id = ?", (pid,))
        db.commit()

        typer.echo(f"  Valid: {valid_count} | Rejected: {rejected}")
        all_results.append({"paper_id": pid, "concepts": valid_count, "relations": len(relations)})

    if json_output:
        typer.echo(json.dumps({"results": all_results}, indent=2, ensure_ascii=False))
    elif all_results:
        total_c = sum(r["concepts"] for r in all_results)
        total_r = sum(r["relations"] for r in all_results)
        # Cross-paper concept deduplication
        from drbrain.extractor.concept import dedup_concepts_by_label

        merged = dedup_concepts_by_label(db)
        if merged:
            typer.echo(f"  Dedup: {merged} duplicate concepts merged")

        typer.echo(
            f"\nBuild complete: {total_c} concepts, {total_r} relations across {len(all_results)} papers"
        )

    db.close()


def embed_cmd(
    ctx: typer.Context,
    dim: int = typer.Option(128, "--dim", help="Embedding dimension"),
    epochs: int = typer.Option(100, "--epochs", help="Training epochs"),
    retrain: bool = typer.Option(False, "--retrain", help="Force retrain"),
    tree: bool = typer.Option(
        False, "--tree", help="Generate tree node text embeddings (PageIndex + RAPTOR)"
    ),
):
    """Train TransE graph embeddings. Use --tree for text embeddings."""
    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])

    # --tree mode: text embeddings for tree nodes (Layer 2)
    if tree:
        import asyncio

        from drbrain.config import EmbedConfig

        embed_cfg = cfg.get("embed", EmbedConfig())
        if isinstance(embed_cfg, dict):
            embed_cfg = EmbedConfig(**embed_cfg)

        if getattr(embed_cfg, "provider", "local") == "none":
            typer.echo("embed.provider=none; tree vector generation is disabled")
            db.close()
            return

        papers_dir = Path(cfg["dirs"]["papers"])
        llm_models_raw = cfg.get("llm", {})
        llm_models = (
            llm_models_raw.get("models", [])
            if hasattr(llm_models_raw, "get")
            else getattr(llm_models_raw, "models", [])
        )
        bridge_mod = __import__("drbrain.services.embedding", fromlist=["build_paper_tree_vectors"])
        total = 0
        for paper_path in sorted(papers_dir.iterdir()):
            if not paper_path.is_dir():
                continue
            count = asyncio.run(
                bridge_mod.build_paper_tree_vectors(paper_path, db.path, embed_cfg, llm_models)
            )
            if count:
                typer.echo(f"  {paper_path.name}: {count} vectors+summaries")
            total += count

        typer.echo(f"Tree vectors+summaries: {total} total")
        db.close()
        return
    graph = GraphEngine()
    graph.load_from_db(db)

    if graph.graph.number_of_nodes() == 0:
        typer.echo("No graph data. Run: drbrain build first", err=True)
        db.close()
        raise typer.Exit(1)

    # Load existing embeddings for incremental training
    existing = db.load_embeddings()
    init_ents = existing if existing and not retrain else None
    init_rels = None  # relations are re-learned each time (fewer, changes matter)

    db.clear_embeddings()
    from drbrain.graph.embedding import TransE

    t = TransE(dim=dim, epochs=epochs)
    typer.echo(
        f"Training embeddings (dim={dim}, epochs={epochs}, "
        f"nodes={graph.graph.number_of_nodes()}"
        f"{', incremental' if init_ents else ', from scratch'})..."
    )
    t.train(graph.graph, init_entities=init_ents, init_relations=init_rels)

    for label, vec in t.entities.items():
        db.save_embedding(label, vec, dim)
    for label, vec in t.relations.items():
        db.save_embedding(f"__rel__{label}", vec, dim)
    db.commit()
    typer.echo(f"Trained {len(t.entities)} entities, {len(t.relations)} relations")
    db.close()


def reason_cmd(
    ctx: typer.Context,
    question: str = typer.Argument(..., help="Question to reason about using the knowledge graph"),
    bidirectional: bool = typer.Option(
        False,
        "--bidirectional",
        "-b",
        help="Use bidirectional LLM-KG iterative reasoning loop (validates hypotheses against graph constraints)",
    ),
    max_rounds: int = typer.Option(
        3,
        "--max-rounds",
        "-r",
        help="Maximum hypothesis-revision rounds for bidirectional mode",
    ),
):
    """LLM agent that reasons over the knowledge graph using tool-calling."""
    from drbrain.extractor.reasoner import ReasonerAgent

    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])
    graph = GraphEngine()
    graph.load_from_db(db)

    models = cfg.get("llm", {}).get("models", [])
    if not models:
        typer.echo("No LLM models configured. Run: drbrain setup", err=True)
        db.close()
        raise typer.Exit(1)

    agent = ReasonerAgent(db=db, graph_engine=graph, models=models)

    if bidirectional:
        typer.echo(f"Bidirectional reasoning: {question}\n")
        result = asyncio.run(agent.reason_bidirectional(question, max_rounds=max_rounds))
        if "error" in result:
            typer.echo(f"Error: {result['error']}", err=True)
        else:
            typer.echo(f"Answer (round {result['rounds']}): {result['answer']}\n")
            typer.echo(f"Hypotheses explored: {len(result['hypotheses'])}")
            for i, (h, v) in enumerate(zip(result["hypotheses"], result["kg_validations"]), 1):
                typer.echo(
                    f"  Round {i}: consistent={v['consistent']}, "
                    f"violations={len(v['violations'])}, patterns={len(v['patterns'])}"
                )
    else:
        typer.echo(f"Reasoning: {question}\n")
        answer = asyncio.run(agent.reason(question))
        typer.echo(answer)

    db.close()


def ask_cmd(
    ctx: typer.Context,
    question: list[str] = typer.Argument(
        ..., help="Natural language question about the knowledge graph"
    ),
    top_k: int = typer.Option(5, "--top", "-k", help="Number of graph concepts to retrieve"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Ask a question in natural language — searches the KG and returns an answer.

    Example: drbrain ask "Is attention better than CNN for NLP?"
    """
    import asyncio

    question_text = " ".join(question)
    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])

    from drbrain.query.bm25 import build_bm25_index

    idx = build_bm25_index(db)
    results = idx.search(question_text, limit=top_k)

    if not results:
        db.close()
        typer.echo("No relevant concepts found in the knowledge graph.")
        return

    graph = GraphEngine()
    graph.load_from_db(db)
    context_parts = []
    for r in results[:top_k]:
        label = r.get("label", "")
        ctype = r.get("type", "")
        paper_id = r.get("local_id", "")
        paper_info = db.get_paper(paper_id) if paper_id else {}
        paper_title = paper_info.get("title", paper_id) if paper_info else paper_id
        context_parts.append(f"- {label} ({ctype}) from {paper_title}")
        if label in graph.graph:
            neighbors = graph.traverse({label}, hops=1, direction="both")[:5]
            for n in neighbors:
                rel = n.path[0].relation.replace("_", " ") if n.path else "related to"
                context_parts.append(f"  --{rel}--> {n.target}")

    context = "\n".join(context_parts[:50])

    models = cfg.llm.models if hasattr(cfg, "llm") else cfg.get("llm", {}).get("models", [])
    if not models:
        db.close()
        typer.echo("No LLM models configured. Showing graph context only:\n\n" + context)
        return

    from drbrain.extractor.llm_client import acall_text_with_fallback

    prompt = (
        f"Answer this research question using the knowledge graph context below.\n\n"
        f"Question: {question_text}\n\n"
        f"Knowledge Graph Context:\n{context}\n\n"
        f"Answer concisely in 2-4 sentences. If the context doesn't contain "
        f"enough information, say so. Cite specific concepts and relations."
    )

    answer = asyncio.run(acall_text_with_fallback(prompt, models, max_tokens=300))
    db.close()

    if json_output:
        import json as _json

        typer.echo(
            _json.dumps(
                {"question": question_text, "answer": answer, "context": context},
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    typer.echo(f"\nQ: {question_text}\n")
    typer.echo(f"A: {answer}\n")
    typer.echo(f"(based on {len(results)} graph concepts)")


def evolve_cmd(
    ctx: typer.Context,
    concept: str = typer.Argument(..., help="Concept label to trace evolution of"),
    direction: str = typer.Option(
        "both",
        "--direction",
        "-d",
        help="Traversal direction: ancestors, descendants, both",
    ),
    max_depth: int = typer.Option(3, "--max-depth", "-n", help="Max traversal depth"),
    mermaid: bool = typer.Option(False, "--mermaid", help="Output as Mermaid diagram"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Show how a concept evolved — its ancestors and descendants in the knowledge graph."""
    if direction not in ("ancestors", "descendants", "both"):
        typer.echo("--direction must be: ancestors, descendants, both", err=True)
        raise typer.Exit(1)

    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])
    graph = GraphEngine()
    graph.load_from_db(db)

    from drbrain.graph.genealogy import evolve_concept, format_tree

    trees = evolve_concept(graph, db, concept, direction=direction, max_depth=max_depth)

    if not trees:
        db.close()
        typer.echo(f"No concept found matching: {concept}")
        raise typer.Exit(0)

    if json_output:
        typer.echo(json.dumps(trees, indent=2, ensure_ascii=False, default=str))
    elif mermaid:
        typer.echo(format_tree(trees, mermaid=True))
    else:
        typer.echo(f"\nEvolution of: {concept}\n")
        for root in trees:
            typer.echo(format_tree([root]))

    db.close()


def descendants_cmd(
    ctx: typer.Context,
    paper_id: str = typer.Argument(..., help="Paper local_id to trace descendants of"),
    generations: int = typer.Option(
        3, "--generations", "-g", help="Number of generations to trace"
    ),
    mermaid: bool = typer.Option(False, "--mermaid", help="Output as Mermaid diagram"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    sections: bool = typer.Option(
        False, "--sections", help="Show section provenance for each concept"
    ),
):
    """Trace a paper's academic offspring — who cited, extended, or refined it."""
    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])
    graph = GraphEngine()
    graph.load_from_db(db)

    from drbrain.graph.genealogy import format_tree, trace_descendants

    tree = trace_descendants(db, graph, paper_id, generations=generations)

    if tree is None:
        db.close()
        typer.echo(f"Paper not found: {paper_id}", err=True)
        raise typer.Exit(1)

    if sections and tree:
        _enrich_tree_with_sections(tree, graph, db)

    if json_output:
        typer.echo(json.dumps(tree, indent=2, ensure_ascii=False, default=str))
    elif mermaid:
        typer.echo(format_tree([tree], mermaid=True))
    else:
        typer.echo(f"\nDescendants of: {paper_id}\n")
        typer.echo(format_tree([tree]))

    db.close()


def landscape_cmd(
    ctx: typer.Context,
    workspace: str = typer.Argument(None, help="Workspace name or path"),
    top_n: int = typer.Option(5, "--top-n", help="Top concepts/items to show"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Show a domain landscape -- timeline, persistent gaps, and debates."""
    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])

    # Resolve workspace
    if isinstance(workspace, typer.models.OptionInfo):
        workspace = workspace.default
    paper_ids = None
    if workspace:
        from drbrain.storage.workspace import load_workspace_papers

        try:
            paper_ids = load_workspace_papers(workspace)
        except (FileNotFoundError, OSError):
            paper_ids = []

    from drbrain.graph.genealogy import landscape_workspace

    result = landscape_workspace(db, workspace_path=workspace, paper_ids=paper_ids)

    if json_output:
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    elif "error" in result:
        typer.echo(f"Error: {result['error']}", err=True)
    else:
        _render_landscape(result, top_n)

    db.close()


def _render_landscape(result: dict, top_n: int):
    """Render landscape as ASCII timeline."""
    timeline = result.get("timeline", [])
    if not timeline:
        typer.echo("No papers found.")
        return

    typer.echo("\nLandscape")
    typer.echo("=" * 60)

    current_year = None
    for entry in timeline:
        year = entry["year"]
        title = entry["title"]

        if year != current_year:
            current_year = year
            typer.echo(f"\n  {year}  ", nl=False)
        else:
            typer.echo("        ", nl=False)

        typer.echo(f"{title}")

        for concept in entry.get("key_concepts", [])[:top_n]:
            typer.echo(f"        +- {concept['label']} [{concept['type']}]")

    gaps = result.get("gaps", [])
    if gaps:
        typer.echo(f"\nPersistent gaps ({len(gaps)}):")
        for g in gaps[:top_n]:
            provenance = g.get("provenance", "")
            typer.echo(f"  * {g['description'][:120]} ({g.get('concept', '')})")
            if provenance:
                typer.echo(f"        {provenance}")

    debates = result.get("debates", [])
    if debates:
        typer.echo(f"\nDebates ({len(debates)}):")
        for d in debates[:top_n]:
            provenance = d.get("provenance", "")
            typer.echo(f"  * {d['description'][:120]} ({d.get('concept', '')})")
            if provenance:
                typer.echo(f"        {provenance}")


def paradigm_cmd(
    ctx: typer.Context,
    concept: str = typer.Argument(None, help="Concept to check for paradigm shifts"),
    workspace: str = typer.Option(None, "--workspace", "-w", help="Scan entire workspace"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Detect paradigm shifts -- replacement, explosion, or cross-domain invasion."""
    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])
    graph = GraphEngine()
    graph.load_from_db(db)

    # Resolve workspace
    if isinstance(workspace, typer.models.OptionInfo):
        workspace = workspace.default
    paper_ids = None
    if workspace:
        from drbrain.storage.workspace import load_workspace_papers

        try:
            paper_ids = load_workspace_papers(workspace)
        except (FileNotFoundError, OSError):
            paper_ids = []

    from drbrain.graph.genealogy import detect_paradigm_shifts

    results = detect_paradigm_shifts(graph, db, concept=concept, paper_ids=paper_ids)

    if json_output:
        typer.echo(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    else:
        if not results:
            typer.echo("No paradigm shifts detected.")
        else:
            type_labels = {
                "replacement": "Replacement",
                "explosion": "Explosion",
                "cross_domain": "Cross-Domain",
            }
            for r in results:
                typer.echo(f"\n[{type_labels.get(r['type'], r['type'])}] {r['description']}")
                prov = r.get("provenance") or r.get("source_provenance") or ""
                if prov:
                    typer.echo(f"        {prov}")
                old_prov = r.get("old_provenance", "")
                new_prov = r.get("new_provenance", "")
                if old_prov and new_prov:
                    typer.echo(f"        old: {old_prov}")
                    typer.echo(f"        new: {new_prov}")

    db.close()


def transfers_cmd(
    ctx: typer.Context,
    from_ws: str = typer.Option(None, "--from", help="Source workspace (methods)"),
    to_ws: str = typer.Option(None, "--to", help="Target workspace (problems)"),
    auto: bool = typer.Option(False, "--auto", help="Auto-detect domains"),
    min_confidence: float = typer.Option(
        0.3, "--min-confidence", help="Minimum transfer confidence"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    history: bool = typer.Option(False, "--history", help="Show historical transfer timeline"),
    sections: bool = typer.Option(
        False, "--sections", help="Show section provenance for transferred concepts"
    ),
):
    """Discover cross-domain method migration opportunities."""
    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])
    graph = GraphEngine()
    graph.load_from_db(db)

    if sections:
        # Collect concept labels for section enrichment
        labels: set[str] = set()
        if history:
            from drbrain.graph.genealogy import find_transfer_history

            results = find_transfer_history(db, graph)
        else:
            from drbrain.graph.genealogy import (
                find_transfer_opportunities,
                find_transfer_opportunities_auto,
            )

            if auto:
                results = find_transfer_opportunities_auto(db, graph, min_confidence=min_confidence)
            elif from_ws and to_ws:
                src_papers = _resolve_workspace_papers(from_ws)
                tgt_papers = _resolve_workspace_papers(to_ws)
                results = find_transfer_opportunities(
                    db,
                    graph,
                    source_paper_ids=list(src_papers) if src_papers else [],
                    target_paper_ids=list(tgt_papers) if tgt_papers else [],
                    min_confidence=min_confidence,
                )
            else:
                typer.echo(
                    "Use --from/--to for explicit workspaces, --auto for automatic detection, "
                    "or --history for historical transfers.",
                    err=True,
                )
                db.close()
                raise typer.Exit(1)

        for r in results or []:
            if "source_method" in r:
                labels.add(str(r["source_method"]))
            if "target_problem" in r:
                labels.add(str(r["target_problem"]))

        section_map = graph.get_section_contexts_batch(db.conn, list(labels))

        # Enrich results
        for r in results or []:
            if "source_method" in r and r["source_method"] in section_map:
                r["source_section"] = section_map[r["source_method"]]["section"]
            if "target_problem" in r and r["target_problem"] in section_map:
                r["target_section"] = section_map[r["target_problem"]]["section"]

        if json_output:
            typer.echo(json.dumps(results, indent=2, ensure_ascii=False, default=str))
        else:
            if not results:
                typer.echo("No transfers found.")
            else:
                for r in results:
                    parts = [
                        f"{r.get('source_concept', '?')}",
                    ]
                    if r.get("source_section"):
                        parts.append(f"[{r['source_section']}]")
                    parts.append("→")
                    parts.append(f"{r.get('target_concept', '?')}")
                    if r.get("target_section"):
                        parts.append(f"[{r['target_section']}]")
                    if r.get("confidence"):
                        parts.append(f"({r['confidence']:.2f})")
                    typer.echo(" ".join(parts))
        db.close()
        return

    if history:
        from drbrain.graph.genealogy import find_transfer_history

        results = find_transfer_history(db, graph)
        if json_output:
            typer.echo(json.dumps(results, indent=2, ensure_ascii=False, default=str))
        else:
            if not results:
                typer.echo("No historical transfers found.")
            else:
                typer.echo("\nCross-Domain Transfer History")
                typer.echo("═" * 40)
                current_year = None
                for r in results:
                    year = r.get("year", "?")
                    if year != current_year:
                        current_year = year
                        typer.echo(f"\n{year}  ", nl=False)
                    else:
                        typer.echo("      ", nl=False)
                    typer.echo(
                        f"{r['source_concept']} → {r['target_concept']} ({r['confidence']:.2f})"
                    )
        db.close()
        return

    from drbrain.graph.genealogy import (
        find_transfer_opportunities,
        find_transfer_opportunities_auto,
    )

    if auto:
        results = find_transfer_opportunities_auto(db, graph, min_confidence=min_confidence)
    elif from_ws and to_ws:
        src_papers = _resolve_workspace_papers(from_ws)
        tgt_papers = _resolve_workspace_papers(to_ws)
        results = find_transfer_opportunities(
            db,
            graph,
            source_paper_ids=list(src_papers) if src_papers else [],
            target_paper_ids=list(tgt_papers) if tgt_papers else [],
            min_confidence=min_confidence,
        )
    else:
        typer.echo(
            "Use --from/--to for explicit workspaces, or --auto for automatic detection.", err=True
        )
        db.close()
        raise typer.Exit(1)

    if json_output:
        typer.echo(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    elif not results:
        typer.echo("No transfer opportunities found.")
    else:
        for r in results:
            typer.echo(f"  {r['source_method']} -> {r['target_problem']} ({r['confidence']:.2f})")

    db.close()


def isomorphism_cmd(
    ctx: typer.Context,
    concept: str = typer.Argument(None, help="Concept to find isomorphic patterns for"),
    min_confidence: float = typer.Option(
        0.5, "--min-confidence", help="Minimum confidence threshold"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Find structurally isomorphic subgraphs — concepts with similar relation patterns."""
    from drbrain.extractor.isomorphism import (
        enrich_isomorphisms_with_raptor,
        find_isomorphic_patterns,
    )

    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])
    graph = GraphEngine()
    graph.load_from_db(db)

    mappings = find_isomorphic_patterns(graph)
    mappings = enrich_isomorphisms_with_raptor(mappings, db)

    if concept:
        mappings = [m for m in mappings if m.source_domain == concept or m.target_domain == concept]
    if min_confidence > 0:
        mappings = [m for m in mappings if m.confidence >= min_confidence]

    if json_output:
        result = [
            {
                "source": m.source_domain,
                "target": m.target_domain,
                "shared_structure": m.shared_structure,
                "confidence": m.confidence,
                "raptor_source_context": m.raptor_source_context,
                "raptor_target_context": m.raptor_target_context,
            }
            for m in mappings
        ]
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        if not mappings:
            typer.echo("No isomorphic patterns found.")
        else:
            typer.echo(f"\nIsomorphic patterns ({len(mappings)}):")
            for m in mappings:
                typer.echo(
                    f"  {m.source_domain} ↔ {m.target_domain} "
                    f"({m.confidence:.2f}) [{m.shared_structure}]"
                )

    db.close()


def difficulty_cmd(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Show a difficulty map — gaps classified by source section type."""
    from drbrain.graph.genealogy import analyze_difficulty

    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])
    result = analyze_difficulty(db)

    if json_output:
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        total = sum(len(v) for v in result.values())
        if total == 0:
            typer.echo("No gaps found. Run: drbrain build first.")
        else:
            typer.echo(f"\nDifficulty map ({total} gaps)")
            typer.echo("=" * 50)
            for cat, label in [
                ("limitation", "Limitation gaps"),
                ("future_work", "Future work gaps"),
                ("discussion", "Discussion gaps"),
                ("uncategorized", "Uncategorized gaps"),
            ]:
                items = result.get(cat, [])
                if items:
                    typer.echo(f"\n{label} ({len(items)}):")
                    for g in items:
                        typer.echo(f"  * {g['label']}")
                        typer.echo(f"        {g['provenance']}")

    db.close()


def frontier_cmd(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Show knowledge frontier — active gaps, debates, and paradigm shifts."""
    from drbrain.graph.genealogy import analyze_frontier

    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])
    result = analyze_frontier(db)

    if json_output:
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        typer.echo("\nKnowledge Frontier")
        typer.echo("=" * 50)
        typer.echo(f"\n{result['summary']}\n")

        active = result.get("active_gaps", [])
        if active:
            typer.echo(f"Active gaps ({len(active)}):")
            for g in active[:10]:
                typer.echo(f"  * {g['label']} ({g['year']})")
                typer.echo(f"        {g['provenance']}")

        debates = result.get("debates", [])
        if debates:
            typer.echo(f"\nActive debates ({len(debates)}):")
            for d in debates[:5]:
                typer.echo(f"  * {d['description'][:120]}")

        shifts = result.get("paradigm_shifts", [])
        if shifts:
            typer.echo(f"\nParadigm shifts ({len(shifts)}):")
            for s in shifts[:5]:
                typer.echo(f"  [{s['type']}] {s['description'][:120]}")

    db.close()
