"""Shared helper functions for CLI commands."""

from __future__ import annotations

import asyncio
import re
import shutil
import uuid
from pathlib import Path

import typer

from drbrain.dedup.resolver import DedupEngine, PaperIDs
from drbrain.graph.engine import GraphEngine
from drbrain.parser.mineru_parser import extract_pdf
from drbrain.services.fetch import fetch_paper
from drbrain.storage.database import Database
from drbrain.storage.paths import (
    images_dir,
    raw_md_path,
    source_pdf_path,
    tree_json_path,
)


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


def _build_closure_context(
    graph,
    seed_labels: list[str],
    top_k: int = 5,
) -> str:
    """Build a context string from closure-inferred edges for seed concept labels.

    Runs ``closure_incremental`` scoped to the given seed labels, sorts by
    confidence (descending), and returns lines in the format::

        --[inferred: <relation>]--> <dst> (confidence: X.XX, via: <via>)

    Args:
        graph: GraphEngine instance loaded from DB.
        seed_labels: Concept labels that were matched by BM25/search.
        top_k: Maximum number of inferred edges to include.

    Returns:
        Formatted multi-line string, or empty string if no edges inferred.
    """
    if not seed_labels or graph.graph.number_of_edges() == 0:
        return ""

    inferred = graph.closure_incremental(set(seed_labels))
    if not inferred:
        return ""

    # Sort by confidence descending (default 1.0 if missing)
    sorted_edges = sorted(
        inferred,
        key=lambda e: e.get("confidence", 1.0),
        reverse=True,
    )

    lines: list[str] = []
    for edge in sorted_edges[:top_k]:
        relation = edge["relation"].replace("_", " ")
        conf = edge.get("confidence", 1.0)
        via = edge.get("via", "")
        # Build annotation
        annotation_parts = [f"confidence: {conf:.2f}"]
        if via:
            annotation_parts.append(f"via: {via}")
        annotation = ", ".join(annotation_parts)
        lines.append(f"  --[inferred: {relation}]--> {edge['dst']} ({annotation})")

    return "\n".join(lines)
