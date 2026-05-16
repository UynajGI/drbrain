"""Backup creation via tar.gz archive."""

from __future__ import annotations

import tarfile
from datetime import datetime
from pathlib import Path

from loguru import logger

BACKUP_DIR = "data/backups"


def _backup_filename() -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"drbrain-{ts}.tar.gz"


def create_backup(
    papers_dir: Path,
    db_path: Path,
    backup_dir: Path | None = None,
    workspace_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> Path:
    """Create a tar.gz backup of papers, DB, workspace, and reports.

    Args:
        papers_dir: Path to data/papers/.
        db_path: Path to drbrain.db.
        backup_dir: Output directory for the archive (default: data/backups).
        workspace_dir: Optional path to workspace/.
        reports_dir: Optional path to data/reports/.

    Returns:
        Path to the created archive.
    """
    backup_dir = Path(backup_dir or BACKUP_DIR)
    backup_dir.mkdir(parents=True, exist_ok=True)

    out_path = backup_dir / _backup_filename()

    with tarfile.open(out_path, "w:gz") as tar:
        if papers_dir.exists():
            tar.add(str(papers_dir), arcname="papers")
        if db_path.exists():
            tar.add(str(db_path), arcname="db/drbrain.db")
        if workspace_dir and workspace_dir.exists():
            for item in workspace_dir.iterdir():
                tar.add(str(item), arcname=f"workspace/{item.name}")
        if reports_dir and reports_dir.exists():
            tar.add(str(reports_dir), arcname="reports")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info("[backup] created %s (%.1f MB)", out_path.name, size_mb)
    return out_path


def list_backups(backup_dir: Path | None = None) -> list[Path]:
    """Return sorted list of backup files (newest first)."""
    d = Path(backup_dir or BACKUP_DIR)
    if not d.exists():
        return []
    backups = sorted(
        d.glob("drbrain-*.tar.gz"),
        key=lambda p: p.name,
        reverse=True,
    )
    return backups
