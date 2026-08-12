"""Tests for the DuckDB analytics sidecar (optional dependency).

Whole module is skipped when duckdb is not installed
(``pytest.importorskip("duckdb")``) — the module itself must still import
cleanly without it.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

duckdb = pytest.importorskip("duckdb")

from drbrain.storage import analytics  # noqa: E402


def _make_sqlite(tmp_path):
    """Create a small SQLite database with two tables."""
    path = tmp_path / "src.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE papers (id INTEGER, title TEXT)")
    con.executemany("INSERT INTO papers VALUES (?, ?)", [(1, "one"), (2, "two")])
    con.execute("CREATE TABLE refs (pid INTEGER, ref TEXT)")
    con.execute("INSERT INTO refs VALUES (1, 'r1')")
    con.commit()
    con.close()
    return path


def test_snapshot_sqlite(tmp_path):
    src = _make_sqlite(tmp_path)
    con = duckdb.connect(tmp_path / "ana.duckdb")
    try:
        failed = analytics.snapshot_sqlite(con, str(src), ["papers", "refs", "missing_table"])
        assert failed == ["missing_table"]
        assert con.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 2
        assert con.execute("SELECT title FROM papers WHERE id = 2").fetchone()[0] == "two"
        assert con.execute("SELECT ref FROM refs WHERE pid = 1").fetchone()[0] == "r1"
        # Already-present table -> recorded as failure, no raise.
        assert analytics.snapshot_sqlite(con, str(src), ["papers"]) == ["papers"]
    finally:
        con.close()


def test_register_json_glob(tmp_path):
    for i in range(2):
        (tmp_path / f"shard{i}.jsonl").write_text(
            json.dumps({"doi": f"d{i}", "tags": [f"t{i}a"]})
            + "\n"
            + json.dumps({"doi": f"d{i}x", "extra": i})
            + "\n"
        )
    con = duckdb.connect()
    try:
        analytics.register_json_glob(con, str(tmp_path / "shard*.jsonl"), "shards")
        rows = con.execute("SELECT doi, tags, extra FROM shards ORDER BY doi").fetchall()
        assert len(rows) == 4
        # union_by_name=True -> shard-missing column becomes NULL.
        assert rows[0] == ("d0", ["t0a"], None)
        assert rows[1] == ("d0x", None, 0)
        # Re-registering replaces the table in place.
        analytics.register_json_glob(con, str(tmp_path / "shard*.jsonl"), "shards")
        assert con.execute("SELECT COUNT(*) FROM shards").fetchone()[0] == 4
    finally:
        con.close()


def test_register_view_unnest(tmp_path):
    con = duckdb.connect()
    try:
        con.execute("CREATE TABLE recipes_raw (doi VARCHAR, recipes STRUCT(m VARCHAR)[])")
        con.execute("INSERT INTO recipes_raw VALUES ('x', [{m: 'A'}, {m: 'B'}])")
        con.execute("INSERT INTO recipes_raw VALUES ('y', [])")
        analytics.register_view_unnest(con, "recipes_raw", "recipes", "recipes_view")
        rows = con.execute(
            "SELECT doi, unnest_col.m FROM recipes_view ORDER BY doi, unnest_col.m"
        ).fetchall()
        assert rows == [("x", "A"), ("x", "B")]
        # Source table is untouched; view also carries the source columns.
        assert con.execute("SELECT COUNT(*) FROM recipes_raw").fetchone()[0] == 2
        assert con.execute("SELECT COUNT(*) FROM recipes_view WHERE doi = 'x'").fetchone()[0] == 2
    finally:
        con.close()


def test_rebuild_table_atomic_swap(tmp_path):
    con = duckdb.connect()
    try:
        con.execute("CREATE TABLE nodes (label VARCHAR, df BIGINT)")
        con.execute("INSERT INTO nodes VALUES ('old', 1)")
        rows = [(f"c{i}", i) for i in range(3)]
        analytics.rebuild_table(
            con,
            "nodes",
            "CREATE TABLE nodes (label VARCHAR, df BIGINT)",
            rows,
        )
        got = con.execute("SELECT label, df FROM nodes ORDER BY df").fetchall()
        assert got == [("c0", 0), ("c1", 1), ("c2", 2)]
        # Temp table cleaned up; no leftover from the swap.
        leftovers = con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_name IN ('nodes_new', 'nodes_old')"
        ).fetchall()
        assert leftovers == []
    finally:
        con.close()


def test_rebuild_table_empty_rows(tmp_path):
    con = duckdb.connect()
    try:
        con.execute("CREATE TABLE t (a INTEGER)")
        con.execute("INSERT INTO t VALUES (1)")
        analytics.rebuild_table(con, "t", "CREATE TABLE t (a INTEGER)", [])
        assert con.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 0
    finally:
        con.close()


def test_rebuild_table_rejects_non_create_sql(tmp_path):
    con = duckdb.connect()
    try:
        with pytest.raises(ValueError, match="create_sql"):
            analytics.rebuild_table(con, "t", "SELECT 1", [])
    finally:
        con.close()


def test_duckdb_missing_raises_clear_error(monkeypatch):
    monkeypatch.setattr(analytics, "duckdb", None)
    with pytest.raises(ImportError, match=r"drbrain\[analytics\]"):
        analytics.snapshot_sqlite(None, "x.db", ["t"])
