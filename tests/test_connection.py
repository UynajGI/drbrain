"""Tests for connect_wal() — ensures WAL pragmas on secondary connections."""

import tempfile
from pathlib import Path

from drbrain.storage.connection import connect_wal


def test_connect_wal_sets_journal_mode():
    with tempfile.TemporaryDirectory() as td:
        conn = connect_wal(Path(td) / "test.db")
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        conn.close()


def test_connect_wal_sets_busy_timeout():
    with tempfile.TemporaryDirectory() as td:
        conn = connect_wal(Path(td) / "test.db")
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] >= 5000
        conn.close()


def test_connect_wal_sets_synchronous_normal():
    with tempfile.TemporaryDirectory() as td:
        conn = connect_wal(Path(td) / "test.db")
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1
        conn.close()


def test_connect_wal_accepts_string_path():
    with tempfile.TemporaryDirectory() as td:
        conn = connect_wal(str(Path(td) / "test.db"))
        assert conn.execute("SELECT 1").fetchone()[0] == 1
        conn.close()
