"""Shared helper functions for CLI commands."""

from __future__ import annotations

import asyncio
import re
import shutil
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import typer

from drbrain.cli._helpers.enrich import (
    _enrich_doi_from_crossref,
    _enrich_doi_from_crossref_arxiv,
    _enrich_doi_from_crossref_doi,
    _enrich_doi_from_openalex,
)
from drbrain.dedup.resolver import DedupEngine, PaperIDs
from drbrain.parser.mineru_parser import extract_pdf
from drbrain.services.fetch import fetch_paper
from drbrain.storage.database import Database
from drbrain.storage.paths import (
    images_dir,
    raw_md_path,
    source_pdf_path,
    tree_json_path,
)


@contextmanager
def open_db(cfg: dict) -> Iterator[Database]:
    """Context manager that opens the Database from config and ensures cleanup.

    Usage::

        with open_db(cfg) as db:
            papers = db.get_all_papers()

    Replaces the manual ``db = Database(cfg["db"]["path"])`` / ``db.close()``
    pattern repeated ~50 times across CLI commands. Closes even on exceptions.
    """
    db = Database(cfg["db"]["path"])
    try:
        yield db
    finally:
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

    import time as _time

    from loguru import logger as _ingest_log

    _t0 = _time.monotonic()

    # Stage 1: Parse
    echo(f"Parsing: {pdf_path}")
    _ingest_log.info(f"[ingest] Stage 1/4 parse: {pdf_path.name}")
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
    _t1 = _time.monotonic()
    _ingest_log.info(
        f"[ingest] parse done in {_t1 - _t0:.1f}s — {len(parsed.text_blocks)} blocks, title={parsed.title[:80]}"
    )

    # Stage 2: Identify
    _ingest_log.info(f"[ingest] Stage 2/4 identify: doi={parsed.doi} arxiv={parsed.arxiv}")
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

    _t2 = _time.monotonic()
    _ingest_log.info(
        f"[ingest] identify done in {_t2 - _t1:.1f}s — local_id={local_id} status={'new' if is_new else 'upgraded'}"
    )

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

    # Stage 3: Structure markdown into tree (PageIndex)
    _t_pp = _time.monotonic()
    _ingest_log.info(f"[ingest] Stage 3/4 tree: type={paper_type}")
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
        _t3 = _time.monotonic()
        _ingest_log.info(
            f"[ingest] tree done in {_t3 - _t_pp:.1f}s — {len(doc_tree.structure)} sections"
        )
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

    _t_total = _time.monotonic() - _t0
    _ingest_log.info(
        f"[ingest] Stage 4/4 done — total {_t_total:.1f}s local_id={local_id} "
        f"title={parsed.title[:60]} year={parsed.year}"
    )
    echo(f"  Ingested: {local_id} ({_t_total:.1f}s)")
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
