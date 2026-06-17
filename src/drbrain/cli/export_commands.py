"""Export and data operation commands."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from drbrain.cli._common import (
    _export_paper_to_meta,
    _show_actor,
    open_db,
)

console = Console()


def export_cmd(
    ctx: typer.Context,
    local_id: str = typer.Argument(None, help="Paper local_id"),
    format: str = typer.Option("bib", "--format", "-f", help="Export format: bib, ris, md"),
    all: bool = typer.Option(False, "--all", help="Export all papers"),
    output: str = typer.Option(None, "--output", "-o", help="Output file path"),
    style: str = typer.Option("apa", "--style", "-s", help="Citation style for md export"),
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
    with open_db(cfg) as db:
        if all:
            papers = db.get_all_papers()
            metas = [_export_paper_to_meta(db, p["local_id"]) for p in papers]
            result = batch_export(metas, format, style=style)
        elif local_id:
            paper = db.get_paper(local_id)
            if not paper:
                typer.echo(f"Paper not found: {local_id}", err=True)
                raise typer.Exit(1)
            meta = _export_paper_to_meta(db, local_id)
            formatters = {
                "bib": meta_to_bibtex,
                "ris": meta_to_ris,
                "md": lambda m: meta_to_markdown(m, style=style),
            }
            result = formatters[format](meta)
        else:
            typer.echo("Specify a paper local_id or use --all", err=True)
            raise typer.Exit(1)

    if json_output:
        typer.echo(json.dumps({"format": format, "result": result}, ensure_ascii=False))
        return

    if output:
        Path(output).write_text(result + "\n", encoding="utf-8")
        typer.echo(f"Exported to {output}")
    else:
        typer.echo(result)


def queue_cmd(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
):
    """List all pending confidence queue items."""
    cfg = ctx.obj["config"]
    with open_db(cfg) as db:
        pending = db.get_queue_pending()

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
    with open_db(cfg) as db:
        from drbrain.extractor.queue import resolve_accept, resolve_reject

        if accept:
            resolve_accept(db, queue_id)
            action = "accepted"
        else:
            resolve_reject(db, queue_id)
            action = "rejected"

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
    with open_db(cfg) as db:
        from drbrain.extractor.queue import resolve_all

        action = "accept" if accept else "reject"
        result = resolve_all(db, action, type_filter=type_filter, max_conf=max_conf)

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
    with open_db(cfg) as db:
        paper = db.get_paper(local_id)
        if paper is None:
            if json_output:
                typer.echo(json.dumps({"error": f"paper {local_id} not found"}))
            else:
                typer.echo(f"Paper {local_id} not found.", err=True)
            raise typer.Exit(1)

        counts = db.delete_paper(local_id)

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
    target: str = typer.Option(None, "--target", "-t", help="Rsync backup target name"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Rsync dry-run (no transfer)"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
):
    """Create tar.gz backups or sync to rsync remote targets."""
    from drbrain.storage.backup import (
        create_backup,
        list_backups,
        run_backup,
    )

    # cfg may be unavailable when invoked without a Typer context (e.g. direct
    # calls in tests). Lazy-load and guard the rsync-targets section.
    cfg = ctx.obj["config"] if ctx is not None and ctx.obj is not None else None

    if list_only:
        backups = list_backups()
        # Also show rsync targets if configured
        if cfg is not None and cfg.backup.targets:
            typer.echo("Rsync backup targets:\n")
            for name, t in sorted(cfg.backup.targets.items()):
                status = "enabled" if t.enabled else "disabled"
                remote = f"{t.user}@{t.host}" if t.user else t.host
                typer.echo(f"  [{name}] {status}")
                typer.echo(f"    Remote: {remote}:{t.path}")
                typer.echo(f"    Mode: {t.mode}  Compress: {'on' if t.compress else 'off'}")
                if t.exclude:
                    typer.echo(f"    Exclude: {', '.join(t.exclude)}")
            typer.echo()

        typer.echo("Local tar.gz backups:\n")
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

    # Rsync mode
    if target:
        import shlex as _shlex

        if not cfg.backup.targets:
            typer.echo("No rsync backup targets configured.", err=True)
            raise typer.Exit(1)

        # Default source: data/ directory
        source_dir = cfg.get("dirs", {}).get("papers", "data/papers")
        source_dir = str(Path(source_dir).parent)

        try:
            cmd_parts = ["rsync", "-a", "--stats", "--human-readable"]
            if dry_run:
                cmd_parts.append("--dry-run")
            typer.echo("About to run backup command: ")
            typer.echo("  " + _shlex.join(cmd_parts))
            typer.echo("  ... (see config for full SSH/rsync options)")

            result = run_backup(
                rsync_bin=cfg.backup.rsync_bin,
                ssh_bin=cfg.backup.ssh_bin,
                targets=cfg.backup.targets,
                source_dir=source_dir,
                target_name=target,
                dry_run=dry_run,
            )
        except Exception as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1)

        if json_output:
            typer.echo(
                json.dumps(
                    {
                        "target": target,
                        "returncode": result.returncode,
                        "dry_run": dry_run,
                    }
                )
            )
            return

        if result.stdout.strip():
            typer.echo()
            typer.echo(result.stdout.rstrip())
        if result.stderr.strip():
            typer.echo()
            typer.echo(result.stderr.rstrip())
        if result.returncode != 0:
            typer.echo(f"Backup failed, exit code: {result.returncode}", err=True)
            raise typer.Exit(result.returncode)
        if dry_run:
            typer.echo()
            typer.echo("Dry run complete: no files were transferred.")
        else:
            typer.echo()
            typer.echo("Backup completed.")
        return

    # Tar.gz mode (default)
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


def style_cmd(
    ctx: typer.Context,
    list_styles_flag: bool = typer.Option(
        False, "--list", "-l", help="List available citation styles"
    ),
    show: str = typer.Option(None, "--show", help="Show source of a specific style"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
):
    """Manage citation styles for Markdown export (APA, Vancouver, Chicago, MLA, custom)."""
    from pathlib import Path as _Path

    from drbrain.services.citation_styles import (
        DEFAULT_STYLES_DIR,
        list_styles,
        show_style,
    )

    cfg = ctx.obj["config"]
    styles_dir = _Path(cfg.get("dirs", {}).get("citation_styles", str(DEFAULT_STYLES_DIR)))

    if show:
        try:
            result = show_style(show, styles_dir)
        except (ValueError, FileNotFoundError) as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(1)
        typer.echo(result)
        return

    styles = list_styles(styles_dir)
    if json_output:
        typer.echo(json.dumps(styles, ensure_ascii=False, indent=2))
        return

    if not styles:
        typer.echo("No citation styles available.")
        return

    builtins = [s for s in styles if s["source"] == "built-in"]
    customs = [s for s in styles if s["source"] != "built-in"]

    typer.echo(f"Available citation styles ({len(styles)} total):\n")
    for s in builtins:
        typer.echo(f"  [{s['source']}] {s['name']} — {s['description']}")
    for s in customs:
        src_label = s.get("source", "custom")
        desc = s.get("description", "")
        desc_part = f" — {desc}" if desc else ""
        typer.echo(f"  [{src_label}] {s['name']}{desc_part}")


def lineage_cmd(
    ctx: typer.Context,
    author_id: str = typer.Argument(None, help="OpenAlex author ID (e.g., A5023806754)"),
    list_all: bool = typer.Option(False, "--list", help="List all actors with paper counts"),
    name: str = typer.Option(None, "--name", "-n", help="Search actors by display name"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Explore author/research lineage via OpenAlex deduplicated IDs."""
    cfg = ctx.obj["config"]
    with open_db(cfg) as db:
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

            if not rows:
                typer.echo("No actors found.")
                return

            if json_output:
                data = [
                    {
                        "author_id": r[0],
                        "paper_count": r[1],
                        "aliases": r[2].split(",") if r[2] else [],
                    }
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
                typer.echo(f"No actors matching '{name}'.")
                return
            # Show each matching actor
            for (matched_id,) in rows:
                _show_actor(cfg, matched_id)

        elif author_id:
            _show_actor(cfg, author_id)

        else:
            typer.echo(
                "Usage: drbrain lineage <author_id>\n"
                "       drbrain lineage --list\n"
                "       drbrain lineage --name <display_name>",
                err=True,
            )
            raise typer.Exit(1)


def document_cmd(
    ctx: typer.Context,
    file: str = typer.Argument(..., help="Path to Office file (.docx, .pptx, .xlsx)"),
    fmt: str = typer.Option(None, "--format", "-f", help="Override format detection"),
):
    """Inspect an Office document (DOCX, PPTX, XLSX) — structured text summary."""
    from drbrain.services.document import inspect

    path = Path(file)
    if not path.exists():
        typer.echo(f"File not found: {file}", err=True)
        raise typer.Exit(1)

    try:
        result = inspect(path, fmt=fmt)
    except (ValueError, ImportError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)

    typer.echo(result)


def metrics_cmd(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
):
    """Show user behavior analytics — top keywords, most-read papers, weekly trends."""
    from pathlib import Path as _Path

    from drbrain.services.metrics_panel import (
        _ensure_metrics_db,
        get_most_read_papers,
        get_top_keywords,
        get_weekly_trend,
    )

    db_path = _Path("data/metrics.db")
    _ensure_metrics_db(db_path)
    trend = get_weekly_trend(db_path)
    keywords = get_top_keywords(db_path, limit=5)
    papers = get_most_read_papers(db_path, limit=5)

    if json_output:
        typer.echo(
            json.dumps(
                {
                    "weekly_trend": trend,
                    "top_keywords": keywords,
                    "most_read": papers,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    typer.echo("── Weekly Trend (7 days) ──")
    typer.echo(f"  Searches: {trend['total_searches']}  |  Reads: {trend['total_reads']}")
    typer.echo(
        f"  Unique keywords: {trend['unique_keywords']}  |  Unique papers: {trend['unique_papers_read']}"
    )

    if keywords:
        typer.echo("\n── Top Search Keywords ──")
        for kw in keywords:
            typer.echo(f"  {kw['keyword']}: {kw['count']}")

    if papers:
        typer.echo("\n── Most-Read Papers ──")
        for p in papers:
            typer.echo(f"  [{p['local_id']}] {p['title'][:60]} — {p['count']} views")

    if not keywords and not papers:
        typer.echo("\nNo metrics recorded yet. Search and read papers to populate.")
