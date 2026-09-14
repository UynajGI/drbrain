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
