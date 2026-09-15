"""``drbrain storage`` — read-only storage inspection and (later) migration.

Commands here never mutate the store: ``audit`` opens the database read-only
(no migration, no network) and reports categorized findings; ``migrate``
prepares a deterministic plan first and only applies it on explicit request.
"""

from __future__ import annotations

import json as _json
from pathlib import Path

import typer

from drbrain.cli._common import runtime_data_path

storage_app = typer.Typer(help="Storage inspection and migration (read-only audit)")


def _runtime_config(ctx: typer.Context) -> dict:
    config = getattr(ctx, "obj", None) or {}
    return config.get("config", config) if isinstance(config, dict) else {}


def _db_path(ctx: typer.Context, cfg: dict) -> Path:
    raw = ""
    if isinstance(cfg, dict):
        db_cfg = cfg.get("db", {})
        raw = db_cfg.get("path", "") if isinstance(db_cfg, dict) else ""
    return runtime_data_path(ctx, raw or "data/drbrain.db", label="database path")


@storage_app.command("audit")
def storage_audit_cmd(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Machine-readable report"),
    sample: int = typer.Option(5, "--sample", help="Findings to keep per category"),
    papers_root: str = typer.Option(
        "", "--papers-root", help="Inspect legacy paper directories under this root"
    ),
    storage_dir: str = typer.Option(
        "", "--storage-dir", help="Verify the published generation under this directory"
    ),
):
    """Audit the unified store and legacy artifacts without changing anything."""
    from drbrain.services.storage_audit import audit_storage, summarize

    cfg = _runtime_config(ctx)
    db_path = _db_path(ctx, cfg)
    if papers_root:
        root = Path(runtime_data_path(ctx, papers_root, label="papers root"))
    else:
        default_root = "data/papers"
        if isinstance(cfg, dict):
            dirs_cfg = cfg.get("dirs", {})
            if isinstance(dirs_cfg, dict):
                default_root = dirs_cfg.get("papers", default_root)
        root = Path(runtime_data_path(ctx, default_root, label="papers root"))
    generation_dir = (
        Path(runtime_data_path(ctx, storage_dir, label="generation storage"))
        if storage_dir
        else None
    )
    report = audit_storage(
        db_path=db_path,
        papers_root=root if root.is_dir() else None,
        storage_dir=generation_dir,
        sample_limit=max(1, int(sample)),
    )
    if json_output:
        typer.echo(_json.dumps(report.to_json(), indent=2, ensure_ascii=False))
    else:
        typer.echo(f"Storage audit: {summarize(report)}")
        for category, count in sorted(report.by_category().items()):
            typer.echo(f"  {category}: {count}")
        for finding in report.findings[:20]:
            location = f" [{finding.local_id}]" if finding.local_id else ""
            typer.echo(f"  - ({finding.severity}) {finding.category}{location}: {finding.message}")
        if len(report.findings) > 20:
            typer.echo(f"  … {len(report.findings) - 20} more findings")
    if not report.ok():
        raise typer.Exit(code=1)


__all__ = ["storage_app", "storage_audit_cmd"]


@storage_app.command("migrate")
def storage_migrate_cmd(
    ctx: typer.Context,
    dry_run: bool = typer.Option(
        True, "--dry-run/--apply", help="Plan only (default) or execute the plan"
    ),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable plan"),
    papers_root: str = typer.Option(
        "", "--papers-root", help="Legacy paper directories to plan against"
    ),
    max_items: int = typer.Option(
        0, "--max-items", help="Apply at most N items, then checkpoint and pause"
    ),
):
    """Plan the legacy migration; --apply executes it resumably (T50/T51)."""
    from drbrain.services.storage_migration import apply_migration_plan, build_migration_plan

    cfg = _runtime_config(ctx)
    db_path = _db_path(ctx, cfg)
    if papers_root:
        root: Path | None = Path(runtime_data_path(ctx, papers_root, label="papers root"))
    else:
        default_root = "data/papers"
        if isinstance(cfg, dict):
            dirs_cfg = cfg.get("dirs", {})
            if isinstance(dirs_cfg, dict):
                default_root = dirs_cfg.get("papers", default_root)
        candidate = Path(runtime_data_path(ctx, default_root, label="papers root"))
        root = candidate if candidate.is_dir() else None
    plan = build_migration_plan(db_path=db_path, papers_root=root)
    payload = plan.to_json()
    if dry_run:
        if json_output:
            typer.echo(_json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            _print_plan_human(payload)
        return

    from drbrain.storage.database import Database

    database = Database(db_path)
    try:
        outcome = apply_migration_plan(
            database,
            plan,
            papers_root=root if root is not None else Path("."),
            max_items=max_items or None,
        )
    finally:
        database.close()
    result = outcome.to_json()
    if json_output:
        typer.echo(_json.dumps(result, indent=2, ensure_ascii=False))
    else:
        _print_plan_human(payload)
        counts = result["counts"]
        typer.echo(
            f"Migration {outcome.job_id}: applied={counts['applied']} reused={counts['reused']} "
            f"skipped={counts['skipped']} failed={counts['failed']} remaining={counts['remaining']}"
        )
        for entry in result["failed"][:10]:
            typer.echo(f"  - failed {entry['local_id']}: {entry['error']}", err=True)
    if outcome.failed:
        raise typer.Exit(code=1)


@storage_app.command("export")
def storage_export_cmd(
    ctx: typer.Context,
    paper: str = typer.Option(..., "--paper", help="Paper local_id to export"),
    format: str = typer.Option("json", "--format", help="json | md | tree"),
    output: str = typer.Option("", "--output", help="Write to this file instead of stdout"),
    papers_root: str = typer.Option(
        "", "--papers-root", help="Legacy paper directories for the fallback"
    ),
):
    """Explicitly export one paper (T53): canonical first, legacy fallback.

    Only this explicit command materializes an export; daily reads never write
    files, and nothing is copied into a second retrieval store.
    """
    from drbrain.storage.paper_view import export_paper_view

    fmt = str(format).strip().lower()
    if fmt not in ("json", "md", "tree"):
        raise typer.BadParameter("format must be json, md or tree", param_hint="--format")

    cfg = _runtime_config(ctx)
    db_path = _db_path(ctx, cfg)
    if papers_root:
        root: Path | None = Path(runtime_data_path(ctx, papers_root, label="papers root"))
    else:
        default_root = "data/papers"
        if isinstance(cfg, dict):
            dirs_cfg = cfg.get("dirs", {})
            if isinstance(dirs_cfg, dict):
                default_root = dirs_cfg.get("papers", default_root)
        candidate = Path(runtime_data_path(ctx, default_root, label="papers root"))
        root = candidate if candidate.is_dir() else None

    from drbrain.storage.database import Database

    database = Database(db_path)
    try:
        bundle = export_paper_view(database, paper, papers_root=root)
    finally:
        database.close()

    if not bundle.text and not bundle.structure:
        typer.echo(f"Nothing to export for {paper}", err=True)
        raise typer.Exit(code=1)
    if fmt == "md":
        payload_text = bundle.text
    elif fmt == "tree":
        payload_text = _json.dumps(
            {"structure": list(bundle.structure)}, indent=2, ensure_ascii=False
        )
    else:
        payload_text = _json.dumps(bundle.to_json(), indent=2, ensure_ascii=False)
    if output:
        target = Path(runtime_data_path(ctx, output, label="export output"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            payload_text if payload_text.endswith("\n") else payload_text + "\n",
            encoding="utf-8",
        )
        typer.echo(f"Exported {paper} ({fmt}, source={bundle.source or 'none'}) -> {target}")
    else:
        typer.echo(payload_text)


def _print_plan_human(payload: dict) -> None:
    summary = payload["summary"]
    typer.echo(f"Migration plan {payload['plan_id']} (schema v{payload['schema_version']})")
    typer.echo("  " + ", ".join(f"{action}={count}" for action, count in sorted(summary.items())))
    for item in payload["items"][:25]:
        typer.echo(f"  - {item['action']:<9} {item['local_id']}: {item['reason']}")
    if len(payload["items"]) > 25:
        typer.echo(f"  … {len(payload['items']) - 25} more items")
