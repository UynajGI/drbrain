"""Tests for inbox scanning and pending management."""

import json
from pathlib import Path

import pytest

from drbrain.storage.inbox import (
    PENDING_LOG,
    move_to_pending,
    read_pending_log,
    scan_inbox,
)


def test_scan_inbox_finds_pdfs(tmp_path):
    """scan_inbox returns list of PDF paths in inbox directory."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "test.pdf").touch()
    (inbox / "not_a_pdf.txt").touch()

    pdfs = scan_inbox(inbox)
    assert len(pdfs) == 1
    assert pdfs[0].name == "test.pdf"


def test_scan_inbox_empty(tmp_path):
    """Empty inbox returns empty list."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    assert scan_inbox(inbox) == []


def test_scan_inbox_nonexistent():
    """Non-existent directory returns empty list."""
    assert scan_inbox(Path("/nonexistent/inbox")) == []


def test_scan_inbox_skips_symlinked_pdf(tmp_path):
    """A PDF symlink is not treated as an ingest source."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"external PDF")
    (inbox / "linked.pdf").symlink_to(outside)
    safe = inbox / "safe.pdf"
    safe.write_bytes(b"local PDF")

    assert scan_inbox(inbox) == [safe]
    assert outside.read_bytes() == b"external PDF"


def test_scan_inbox_rejects_symlinked_root(tmp_path):
    """A symlinked inbox root cannot expose another directory's files."""
    real_inbox = tmp_path / "real-inbox"
    real_inbox.mkdir()
    (real_inbox / "paper.pdf").write_bytes(b"external PDF")
    linked_inbox = tmp_path / "inbox"
    linked_inbox.symlink_to(real_inbox, target_is_directory=True)

    assert scan_inbox(linked_inbox) == []


def test_move_to_pending(tmp_path):
    """move_to_pending moves PDF and logs reason."""
    inbox = tmp_path / "inbox"
    pending = tmp_path / "pending"
    inbox.mkdir()
    pending.mkdir()
    pdf = inbox / "broken.pdf"
    pdf.write_text("fake pdf content")

    move_to_pending(pdf, pending, reason="LLM extraction failed")

    assert not pdf.exists()
    assert (pending / "broken.pdf").exists()
    log_path = pending / PENDING_LOG
    assert log_path.exists()
    entry = json.loads(log_path.read_text().splitlines()[0])
    assert entry["filename"] == "broken.pdf"
    assert entry["reason"] == "LLM extraction failed"


def test_move_to_pending_creates_pending_dir(tmp_path):
    """move_to_pending creates pending dir if it doesn't exist."""
    inbox = tmp_path / "inbox"
    pending = tmp_path / "pending"
    inbox.mkdir()
    pdf = inbox / "fail.pdf"
    pdf.write_text("content")

    move_to_pending(pdf, pending, reason="parse error")

    assert pending.exists()
    assert not pdf.exists()


def test_move_to_pending_rejects_symlink_source(tmp_path):
    """Moving a failed symlink must not move or delete its external target."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"external PDF")
    linked = inbox / "failed.pdf"
    linked.symlink_to(outside)
    pending = tmp_path / "pending"

    with pytest.raises(ValueError, match="symlink"):
        move_to_pending(linked, pending, reason="parse error")

    assert linked.is_symlink()
    assert outside.read_bytes() == b"external PDF"
    assert not pending.exists()


def test_read_pending_log_empty(tmp_path):
    """Reading non-existent pending log returns empty list."""
    pending = tmp_path / "pending"
    pending.mkdir()
    assert read_pending_log(pending) == []


def test_read_pending_log_with_entries(tmp_path):
    """read_pending_log returns list of pending entries."""
    pending = tmp_path / "pending"
    pending.mkdir()
    log_path = pending / PENDING_LOG
    log_path.write_text(
        json.dumps({"filename": "a.pdf", "reason": "fail", "timestamp": "2026-01-01T00:00:00"})
        + "\n"
        + json.dumps({"filename": "b.pdf", "reason": "error", "timestamp": "2026-01-02T00:00:00"})
        + "\n"
    )
    entries = read_pending_log(pending)
    assert len(entries) == 2
    assert entries[0]["filename"] == "a.pdf"
