"""Inbox scanning and pending file management."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

PENDING_LOG = "pending.jsonl"


def first_symlink_component(path: Path) -> Path | None:
    """Return the first symlink traversed by *path*, without resolving it.

    ``Path.resolve()`` erases the evidence that an input was a symlink.  Inbox
    inputs are later copied and removed, so retain the lexical path and reject
    any symlink in the path (including an intermediate directory) first.
    Missing components are allowed; callers may be about to create them.
    """
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        # ``absolute()`` does not dereference symlinks, unlike ``resolve()``.
        candidate = Path.cwd() / candidate

    current = Path(candidate.anchor) if candidate.anchor else Path.cwd()
    for part in candidate.parts:
        if part == candidate.anchor:
            continue
        current /= part
        try:
            if current.is_symlink():
                return current
        except OSError:
            # A disappearing/unreadable component is not evidence of a safe
            # path; leave the normal filesystem operation to report the error.
            return None
    return None


def scan_inbox(inbox_dir: Path) -> list[Path]:
    """Return sorted list of PDF paths in the inbox directory."""
    inbox_dir = Path(inbox_dir)
    if first_symlink_component(inbox_dir) is not None:
        return []
    if not inbox_dir.exists() or not inbox_dir.is_dir():
        return []

    # ``Path.is_file()`` follows symlinks.  Check the lexical entry first so
    # a link to a PDF outside the inbox is never handed to the ingest pipeline.
    return sorted(
        p
        for p in inbox_dir.iterdir()
        if first_symlink_component(p) is None and p.is_file() and p.suffix.lower() == ".pdf"
    )


def scan_materials(
    inbox_dir: Path,
    extensions: tuple[str, ...] = (".pdf", ".md", ".markdown", ".txt", ".tex", ".text"),
) -> list[Path]:
    """Return supported source materials without following symlink aliases."""
    inbox_dir = Path(inbox_dir)
    if first_symlink_component(inbox_dir) is not None:
        return []
    if not inbox_dir.exists() or not inbox_dir.is_dir():
        return []
    allowed = {suffix.lower() for suffix in extensions}
    return sorted(
        p
        for p in inbox_dir.iterdir()
        if first_symlink_component(p) is None and p.is_file() and p.suffix.lower() in allowed
    )


def move_to_pending(pdf_path: Path, pending_dir: Path, reason: str) -> None:
    """Move a failed PDF to the pending directory and log the reason.

    Args:
        pdf_path: Source PDF path in inbox.
        pending_dir: Destination pending directory.
        reason: Human-readable failure reason.
    """
    pdf_path = Path(pdf_path)
    pending_dir = Path(pending_dir)
    source_link = first_symlink_component(pdf_path)
    if source_link is not None:
        raise ValueError(f"pending source must not contain a symlink: {source_link}")
    destination_link = first_symlink_component(pending_dir)
    if destination_link is not None:
        raise ValueError(f"pending directory must not contain a symlink: {destination_link}")

    pending_dir.mkdir(parents=True, exist_ok=True)
    # Re-check the final directory after mkdir in case an existing path was
    # swapped while the destination was being prepared.
    if pending_dir.is_symlink() or not pending_dir.is_dir():
        raise ValueError(f"pending directory is not a real directory: {pending_dir}")
    dst = pending_dir / pdf_path.name
    if dst.is_symlink():
        raise ValueError(f"pending destination must not be a symlink: {dst}")
    pdf_path.rename(dst)
    _log_pending(dst, reason)


def _log_pending(pdf_path: Path, reason: str) -> None:
    """Append an entry to pending.jsonl in the pending directory."""
    log_path = pdf_path.parent / PENDING_LOG
    if first_symlink_component(log_path) is not None:
        raise ValueError(f"pending log must not be a symlink: {log_path}")
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
    if first_symlink_component(log_path) is not None:
        return []
    if not log_path.exists():
        return []
    entries = []
    for line in log_path.read_text(encoding="utf-8").strip().splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries
