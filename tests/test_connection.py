"""Tests for connect_wal() — ensures WAL pragmas on secondary connections."""

import tempfile
from pathlib import Path

import pytest

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


def test_connect_wal_preserves_memory_sentinel_with_runtime_root(tmp_path, monkeypatch):
    monkeypatch.setenv("DRBRAIN_ROOT", str(tmp_path))

    conn = connect_wal(":memory:")
    try:
        assert conn.execute("SELECT 1").fetchone()[0] == 1
    finally:
        conn.close()


def test_connect_wal_resolves_relative_path_under_runtime_root(tmp_path, monkeypatch):
    monkeypatch.setenv("DRBRAIN_ROOT", str(tmp_path))
    (tmp_path / "data").mkdir()

    conn = connect_wal("data/secondary.db")
    try:
        assert (tmp_path / "data" / "secondary.db").is_file()
    finally:
        conn.close()


def test_connect_wal_rejects_absolute_escape_with_runtime_root(tmp_path, monkeypatch):
    outside = tmp_path.parent / "connection-outside.db"
    monkeypatch.setenv("DRBRAIN_ROOT", str(tmp_path))

    with pytest.raises(ValueError, match="database path escapes runtime root"):
        connect_wal(outside)


@pytest.mark.parametrize("value", ["file:/tmp/shared.db", "https://example.invalid/db"])
def test_connect_wal_rejects_uri_targets(value):
    with pytest.raises(ValueError, match="local filesystem path.*URI"):
        connect_wal(value)


def test_connect_wal_rejects_symlink_target_with_runtime_root(tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "link.db"
    link.symlink_to(outside / "real.db")
    monkeypatch.setenv("DRBRAIN_ROOT", str(tmp_path))

    with pytest.raises(ValueError, match="symlink"):
        connect_wal(link)


def test_connect_wal_keeps_relative_cwd_behavior_without_runtime_root(tmp_path, monkeypatch):
    monkeypatch.delenv("DRBRAIN_ROOT", raising=False)
    monkeypatch.delenv("DRBRAIN_RUNTIME_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)

    conn = connect_wal("legacy.db")
    try:
        assert (tmp_path / "legacy.db").is_file()
    finally:
        conn.close()
