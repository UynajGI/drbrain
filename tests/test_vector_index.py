"""Direct unit tests for drbrain.storage.vector_index watermark semantics."""

from __future__ import annotations

import sqlite3

from drbrain.storage import vector_index as vi


def _conn() -> sqlite3.Connection:
    return sqlite3.connect(":memory:")


def test_vec_synced_cold_store_is_false_and_bootstrap_needs_no_ddl_afterwards():
    """OCR r6: the per-query hot path must not run schema DDL.

    Cold store → False (table created once on the error path); afterwards
    the read is a plain SELECT, and the writer hooks flip the watermark.
    """
    conn = _conn()
    assert vi.vec_synced(conn) is False
    assert vi.vec_synced(conn) is False  # still False, no CREATE on the hot path
    vi.mark_vec_synced(conn)
    assert vi.vec_synced(conn) is True
    vi.mark_vec_dirty(conn)
    assert vi.vec_synced(conn) is False


def test_mark_vec_dirty_clears_a_completed_sync():
    conn = _conn()
    vi.mark_vec_synced(conn)
    assert vi.vec_synced(conn) is True
    vi.mark_vec_dirty(conn)
    assert vi.vec_synced(conn) is False
