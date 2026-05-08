"""Repair and import commands."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from drbrain.cli._common import _resolve_workspace_papers
from drbrain.storage.database import Database


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
