"""Inbox scanning and pending file management."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

PENDING_LOG = "pending.jsonl"


def scan_inbox(inbox_dir: Path) -> list[Path]:
    """Return sorted list of PDF paths in the inbox directory."""
    if not inbox_dir.exists() or not inbox_dir.is_dir():
        return []
    return sorted(p for p in inbox_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")


def move_to_pending(pdf_path: Path, pending_dir: Path, reason: str) -> None:
    """Move a failed PDF to the pending directory and log the reason.

    Args:
        pdf_path: Source PDF path in inbox.
        pending_dir: Destination pending directory.
        reason: Human-readable failure reason.
    """
    pending_dir.mkdir(parents=True, exist_ok=True)
    dst = pending_dir / pdf_path.name
    pdf_path.rename(dst)
    _log_pending(dst, reason)


def _log_pending(pdf_path: Path, reason: str) -> None:
    """Append an entry to pending.jsonl in the pending directory."""
    log_path = pdf_path.parent / PENDING_LOG
    entry = {
        "filename": pdf_path.name,
        "reason": reason,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_pending_log(pending_dir: Path) -> list[dict]:
    """Read all entries from pending.jsonl."""
    log_path = pending_dir / PENDING_LOG
    if not log_path.exists():
        return []
    entries = []
    for line in log_path.read_text(encoding="utf-8").strip().splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries
