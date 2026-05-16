"""Data quality audit: 15-rule full-library scan."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from drbrain.storage.database import Database
from drbrain.storage.paths import raw_md_path, tree_json_path

SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}

console = Console()


def _normalize_title(title: str) -> str:
    """Normalize title for duplicate detection: lowercase, collapse whitespace, strip punctuation."""
    t = title.lower()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def audit_papers(
    db: Database,
    paper_dirs: Path,
    *,
    severity: str = "warning",
) -> list[dict]:
    """Scan all papers and return list of issues.

    Args:
        db: Database instance.
        paper_dirs: Root directory containing per-paper subdirectories.
        severity: Minimum severity level to include (error/warning/info).
                  error = only errors; warning = errors + warnings; info = all.

    Returns:
        List of issue dicts with keys: paper_id, title, rule, severity, message.
    """
    import time as _time

    from loguru import logger as _audit_log

    _t0 = _time.monotonic()
    min_severity = SEVERITY_ORDER.get(severity, 1)
    issues: list[dict] = []
    papers = db.get_all_papers()
    _audit_log.info("[audit] scanning %d papers (severity>=%s)", len(papers), severity)

    # Build title index for duplicate detection
    title_index: dict[str, list[str]] = {}
    for p in papers:
        tid = p.get("local_id", "")
        title = p.get("title") or ""
        if title.strip():
            norm = _normalize_title(title)
            title_index.setdefault(norm, []).append(tid)

    for p in papers:
        pid = p.get("local_id", "")
        title = p.get("title") or ""
        year = p.get("year")
        abstract = p.get("abstract") or ""
        journal = p.get("journal") or ""
        status = p.get("status", "")
        created_at_raw = p.get("created_at", "")
        doi = p.get("doi")
        arxiv = p.get("arxiv")
        s2_id = p.get("s2_id")

        paper_dir = paper_dirs / pid
        concepts = db.get_concepts_by_paper(pid)

        # --- error rules ---

        # missing_title
        if not title.strip():
            if SEVERITY_ORDER["error"] <= min_severity:
                issues.append(
                    {
                        "paper_id": pid,
                        "title": title,
                        "rule": "missing_title",
                        "severity": "error",
                        "message": f"Paper {pid} has no title or empty title",
                    }
                )

        # missing_md
        md_path = raw_md_path(paper_dir)
        if not md_path.exists():
            if SEVERITY_ORDER["error"] <= min_severity:
                issues.append(
                    {
                        "paper_id": pid,
                        "title": title,
                        "rule": "missing_md",
                        "severity": "error",
                        "message": f"No raw.md in {paper_dir}",
                    }
                )

        # --- warning rules ---

        # missing_doi
        if not doi and not arxiv and not s2_id:
            if SEVERITY_ORDER["warning"] <= min_severity:
                issues.append(
                    {
                        "paper_id": pid,
                        "title": title,
                        "rule": "missing_doi",
                        "severity": "warning",
                        "message": f"No DOI, arXiv, or S2 ID for {pid}",
                    }
                )

        # missing_abstract
        if not abstract.strip():
            if SEVERITY_ORDER["warning"] <= min_severity:
                issues.append(
                    {
                        "paper_id": pid,
                        "title": title,
                        "rule": "missing_abstract",
                        "severity": "warning",
                        "message": f"Abstract is empty for {pid} ({title})",
                    }
                )

        # missing_year
        if year is None:
            if SEVERITY_ORDER["warning"] <= min_severity:
                issues.append(
                    {
                        "paper_id": pid,
                        "title": title,
                        "rule": "missing_year",
                        "severity": "warning",
                        "message": f"Year is NULL for {pid} ({title})",
                    }
                )

        # missing_journal
        if not journal.strip():
            if SEVERITY_ORDER["warning"] <= min_severity:
                issues.append(
                    {
                        "paper_id": pid,
                        "title": title,
                        "rule": "missing_journal",
                        "severity": "warning",
                        "message": f"Journal is empty for {pid} ({title})",
                    }
                )

        # missing_authors
        has_actor = any(c.get("type") == "Actor" for c in concepts)
        if not has_actor:
            if SEVERITY_ORDER["warning"] <= min_severity:
                issues.append(
                    {
                        "paper_id": pid,
                        "title": title,
                        "rule": "missing_authors",
                        "severity": "warning",
                        "message": f"No Actor-type concepts for {pid} ({title})",
                    }
                )

        # short_md
        if md_path.exists():
            md_size = md_path.stat().st_size
            if md_size < 200:
                if SEVERITY_ORDER["warning"] <= min_severity:
                    issues.append(
                        {
                            "paper_id": pid,
                            "title": title,
                            "rule": "short_md",
                            "severity": "warning",
                            "message": f"raw.md exists but is only {md_size} chars for {pid}",
                        }
                    )

        # empty_tree
        tree_path = tree_json_path(paper_dir)
        if not tree_path.exists() or tree_path.stat().st_size == 0:
            if SEVERITY_ORDER["warning"] <= min_severity:
                issues.append(
                    {
                        "paper_id": pid,
                        "title": title,
                        "rule": "empty_tree",
                        "severity": "warning",
                        "message": f"tree.json missing or empty for {pid}",
                    }
                )

        # low_concept_count
        concept_count = len(concepts)
        if concept_count < 3:
            if SEVERITY_ORDER["warning"] <= min_severity:
                issues.append(
                    {
                        "paper_id": pid,
                        "title": title,
                        "rule": "low_concept_count",
                        "severity": "warning",
                        "message": f"Only {concept_count} concepts for {pid} (shallow extraction)",
                    }
                )

        # unresolved_env
        if "${" in title:
            if SEVERITY_ORDER["warning"] <= min_severity:
                issues.append(
                    {
                        "paper_id": pid,
                        "title": title,
                        "rule": "unresolved_env",
                        "severity": "warning",
                        "message": f"Title contains '${{}}' (env not resolved) for {pid}: {title}",
                    }
                )

        # --- info rules ---

        # no_edges
        if concept_count > 0:
            edge_count = db.conn.execute(
                "SELECT COUNT(*) FROM edges WHERE source_paper = ?", (pid,)
            ).fetchone()[0]
            if edge_count == 0:
                if SEVERITY_ORDER["info"] <= min_severity:
                    issues.append(
                        {
                            "paper_id": pid,
                            "title": title,
                            "rule": "no_edges",
                            "severity": "info",
                            "message": f"Paper {pid} has {concept_count} concepts but zero edges",
                        }
                    )

        # placeholder_status
        if status == "placeholder":
            if SEVERITY_ORDER["info"] <= min_severity:
                issues.append(
                    {
                        "paper_id": pid,
                        "title": title,
                        "rule": "placeholder_status",
                        "severity": "info",
                        "message": f"Paper {pid} ({title}) has status 'placeholder'",
                    }
                )

        # old_placeholder
        if status == "placeholder" and created_at_raw:
            try:
                created_date = datetime.strptime(created_at_raw[:10], "%Y-%m-%d")
                if datetime.now() - created_date > timedelta(days=30):
                    if SEVERITY_ORDER["info"] <= min_severity:
                        issues.append(
                            {
                                "paper_id": pid,
                                "title": title,
                                "rule": "old_placeholder",
                                "severity": "info",
                                "message": (
                                    f"Placeholder {pid} ({title}) older than 30 days "
                                    f"(created {created_at_raw[:10]})"
                                ),
                            }
                        )
            except (ValueError, IndexError):
                pass

        # duplicate_title
        if title.strip():
            norm = _normalize_title(title)
            dupes = title_index.get(norm, [])
            if len(dupes) > 1 and dupes[0] == pid:
                if SEVERITY_ORDER["info"] <= min_severity:
                    others = [d for d in dupes if d != pid]
                    issues.append(
                        {
                            "paper_id": pid,
                            "title": title,
                            "rule": "duplicate_title",
                            "severity": "info",
                            "message": f"Normalized title matches {others}",
                        }
                    )

    _t_done = _time.monotonic() - _t0
    _by_sev: dict[str, int] = {}
    for i in issues:
        _by_sev[i["severity"]] = _by_sev.get(i["severity"], 0) + 1
    _audit_log.info("[audit] done in %.1fs — %d issues: %s", _t_done, len(issues), dict(_by_sev))
    return issues


def audit_cmd(
    ctx: typer.Context,
    severity: str = typer.Option(
        "warning",
        "--severity",
        "-s",
        help="Minimum severity level: error, warning, info",
    ),
    workspace: str = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Limit audit to a workspace",
    ),
    json_mode: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON",
    ),
) -> None:
    """Scan the library for data quality issues (15 rules, 3 severity levels)."""
    if severity not in SEVERITY_ORDER:
        console.print(f"[red]Invalid severity '{severity}'. Use: error, warning, info[/red]")
        raise typer.Exit(1)

    cfg = ctx.obj["config"]
    db_path = cfg.get("dirs", {}).get("db", "data/drbrain.db")
    papers_root = Path(cfg.get("dirs", {}).get("papers", "data/papers"))
    db = Database(db_path)

    # Resolve workspace filter
    from drbrain.cli.commands import _resolve_workspace_papers

    ws_ids = _resolve_workspace_papers(workspace)

    issues = audit_papers(db, papers_root, severity=severity)

    # Filter by workspace if specified
    if ws_ids is not None:
        issues = [i for i in issues if i["paper_id"] in ws_ids]

    if json_mode:
        console.print(json.dumps(issues, indent=2, default=str))
        return

    # Rich table output
    severity_colors = {"error": "red", "warning": "yellow", "info": "dim"}
    if not issues:
        console.print(f"\n[green]No issues found at severity >= {severity}[/green]")
        return

    console.print(
        f"\n[bold]Data Quality Audit[/bold] — {len(issues)} issue(s) (severity >= {severity})\n"
    )

    table = Table(show_header=True)
    table.add_column("Paper ID")
    table.add_column("Title", max_width=40)
    table.add_column("Severity")
    table.add_column("Rule")
    table.add_column("Message", max_width=60)

    for i in issues:
        color = severity_colors.get(i["severity"], "")
        table.add_row(
            i["paper_id"],
            i["title"][:40],
            f"[{color}]{i['severity']}[/{color}]",
            i["rule"],
            i["message"],
        )

    console.print(table)

    # Summary by severity
    counts: dict[str, int] = {"error": 0, "warning": 0, "info": 0}
    for i in issues:
        counts[i["severity"]] = counts.get(i["severity"], 0) + 1
    console.print(
        f"\n[bold]Summary:[/bold] [red]{counts['error']} error(s)[/red], "
        f"[yellow]{counts['warning']} warning(s)[/yellow], "
        f"[dim]{counts['info']} info(s)[/dim]"
    )
