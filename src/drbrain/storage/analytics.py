"""DuckDB analytics sidecar — read-only analytical layer over the SQLite write path.

The main project writes exclusively through SQLite (single-writer transaction
semantics). This module adds an *optional* DuckDB analysis layer: SQLite stays
the source of truth, and heavy re-aggregations are snapshotted / exported into
a DuckDB database (the unified ``cg.duckdb`` pattern from ``research/``) where
columnar, parallel execution makes them fast. Nothing here ever writes back to
the SQLite source.

``duckdb`` is an optional dependency. It is imported lazily so that importing
this module never fails; calling any function without duckdb installed raises a
clear ``ImportError`` pointing at ``pip install 'drbrain[analytics]'``.

Performance notes learned in ``research/scripts/`` — keep in mind when writing
downstream SQL:

- ``list_transform`` lambdas cannot contain subqueries. For element-wise enrich
  with a lookup table, use ``UNNEST`` + ``LEFT JOIN`` and re-aggregate with
  ``list(struct_pack(...))`` instead
  (``research/scripts/cg_recipe_conditions_merge.py``, 2026-08-11).
- DuckDB's unnest output column has no user-controllable alias: for struct
  arrays access fields as ``unnest.field``
  (``research/scripts/cg_duckdb_migrate.py`` ``recipes`` view) or name the
  column explicitly with ``UNNEST(arr) AS t(col)`` as done here.
- On multi-million-row tables (2M+), bulk ``COPY FROM jsonl`` beats
  row-by-row INSERT/executemany by a wide margin: the v2–v6 incremental
  INSERT/UPDATE/COPY rebuilds were all slow, while the v7 full rebuild
  (DROP + CREATE + COPY) finished in 2–4 min
  (``research/scripts/cg_concept_merge_v7.py``).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

try:  # pragma: no cover - exercised implicitly when the optional dep is absent
    import duckdb
except ImportError:  # pragma: no cover
    duckdb = None  # type: ignore[assignment]

__all__ = [
    "snapshot_sqlite",
    "register_json_glob",
    "register_view_unnest",
    "rebuild_table",
]

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _require_duckdb() -> None:
    """Raise a clear error when the optional duckdb dependency is missing."""
    if duckdb is None:
        raise ImportError(
            "duckdb is not installed: pip install 'drbrain[analytics]' "
            "(or: uv add --optional analytics duckdb)"
        )


def _check_ident(name: str) -> None:
    """Reject identifiers that would break out of SQL string interpolation."""
    if not _IDENT_RE.match(name):
        raise ValueError(f"not a plain SQL identifier: {name!r}")


def _quote_literal(s: str) -> str:
    """Single-quote a SQL string literal, escaping embedded quotes."""
    return "'" + s.replace("'", "''") + "'"


def snapshot_sqlite(
    duckdb_con: duckdb.DuckDBPyConnection,
    sqlite_path: str,
    tables: list[str],
) -> list[str]:
    """Snapshot tables from a SQLite file into the DuckDB connection.

    ATTACHes the SQLite database as ``src``, then runs ``CREATE TABLE t AS
    SELECT * FROM src.t`` per requested table. A table that already exists in
    the DuckDB database, is missing from the SQLite file, or fails for any
    other reason is recorded in the returned failure list instead of raising
    (mirroring the reference migration, which rebuilds the analysis DB from
    scratch and logs skips).

    Reference: ``research/scripts/cg_duckdb_migrate.py`` — ATTACH sqlite →
    CREATE TABLE AS snapshot; per-table try/except; DETACH after the loop.

    Args:
        duckdb_con: open DuckDB connection (the analysis database).
        sqlite_path: path to the source SQLite database file.
        tables: names of tables to copy from the SQLite file.

    Returns:
        List of table names that failed to snapshot (already present in
        DuckDB, absent in SQLite, schema errors, ...). Empty on full success.
    """
    _require_duckdb()
    for t in tables:
        _check_ident(t)
    duckdb_con.execute(f"ATTACH {_quote_literal(sqlite_path)} AS src (TYPE sqlite)")
    failed: list[str] = []
    try:
        for t in tables:
            try:
                duckdb_con.execute(f"CREATE TABLE {t} AS SELECT * FROM src.{t}")
            except Exception:  # per-table tolerance; failures are reported, not raised
                failed.append(t)
    finally:
        duckdb_con.execute("DETACH src")
    return failed


def register_json_glob(
    duckdb_con: duckdb.DuckDBPyConnection,
    path_glob: str,
    table: str,
    union_by_name: bool = True,
) -> None:
    """Aggregate JSON/JSONL shards matching a glob into a DuckDB table.

    Runs ``CREATE OR REPLACE TABLE <table> AS SELECT * FROM
    read_json_auto('<glob>', union_by_name=<bool>)`` — multi-shard exports (one
    file per shard, shards may have differing columns) merge into a single
    columnar table.

    Reference: ``research/scripts/cg_duckdb_migrate.py`` — the ``concepts`` /
    ``recipes_raw`` shard aggregation via ``read_json_auto`` with
    ``union_by_name=True``.

    Args:
        duckdb_con: open DuckDB connection (the analysis database).
        path_glob: file glob, e.g. ``data/cg_concepts/cg_concepts_s*.jsonl``.
        table: destination table name (replaced if it already exists).
        union_by_name: when True (default), columns from all shards are unioned
            and missing columns become NULL; when False, shards must agree on
            schema.
    """
    _require_duckdb()
    _check_ident(table)
    flag = "union_by_name=true" if union_by_name else "union_by_name=false"
    duckdb_con.execute(
        f"CREATE OR REPLACE TABLE {table} AS "
        f"SELECT * FROM read_json_auto({_quote_literal(path_glob)}, {flag})"
    )


def register_view_unnest(
    duckdb_con: duckdb.DuckDBPyConnection,
    table: str,
    array_col: str,
    alias: str,
) -> None:
    """Flatten a nested array column into a view, one row per array element.

    Creates ``CREATE OR REPLACE VIEW <alias> AS SELECT <table>.*, unnest_col
    FROM <table> CROSS JOIN UNNEST(<table>.<array_col>) AS t(unnest_col)``.

    The expansion column is explicitly named ``unnest_col`` (DuckDB's unnest
    output has no user alias otherwise): reference it as ``unnest_col`` for
    scalar arrays, or ``unnest_col.<field>`` for struct arrays — the same
    pattern as the reference ``recipes`` view
    (``SELECT doi, unnest.target_material ... FROM recipes_raw,
    unnest(recipes_raw.recipes)``).

    Reference: ``research/scripts/cg_duckdb_migrate.py`` (``recipes`` view) and
    ``research/scripts/cg_route_c_recipe_graph.py`` (scalar-array unnest).

    Args:
        duckdb_con: open DuckDB connection (the analysis database).
        table: source table name.
        array_col: name of the array column to flatten.
        alias: name of the view to create or replace.
    """
    _require_duckdb()
    _check_ident(table)
    _check_ident(array_col)
    _check_ident(alias)
    duckdb_con.execute(
        f"CREATE OR REPLACE VIEW {alias} AS "
        f"SELECT {table}.*, unnest_col "
        f"FROM {table} CROSS JOIN UNNEST({table}.{array_col}) AS t(unnest_col)"
    )


def rebuild_table(
    duckdb_con: duckdb.DuckDBPyConnection,
    table: str,
    create_sql: str,
    rows: Iterable[Sequence[Any]],
) -> None:
    """Atomically rebuild a table: build ``<table>_new``, then swap it in.

    Steps: DROP ``<table>_new`` if present → CREATE ``<table>_new`` from
    ``create_sql`` → batch INSERT ``rows`` → DROP ``<table>`` → ALTER TABLE
    ``<table>_new`` RENAME TO ``<table>``. The old table stays queryable until
    the final swap, so a failed build never leaves the analysis DB with a
    half-written table.

    ``create_sql`` must be a plain ``CREATE TABLE <table> ...`` statement
    written for the final table name; the first ``CREATE TABLE <table>`` token
    is rewritten to target ``<table>_new`` internally (``IF NOT EXISTS`` and
    ``OR REPLACE`` variants are accepted).

    Two rebuild modes, from ``research/scripts/``:
    - This function: new-table + batch INSERT + rename — the
      ``cg_drt_prep_duckdb.py`` pattern (``CREATE OR REPLACE TABLE
      concept_nodes_new AS ...`` → ``DROP TABLE IF EXISTS`` → ``ALTER TABLE
      ... RENAME TO ...``). Best when rows are already materialized in Python
      and the payload is moderate.
    - DROP + CREATE + COPY FROM jsonl — the ``cg_concept_merge_v7.py`` pattern.
      On a 2M-row table the incremental v2–v6 INSERT/UPDATE/COPY paths were all
      slow; the v7 full rebuild via COPY finished in 2–4 min. For
      multi-million-row rebuilds prefer writing rows to a temp JSONL and
      COPYing it in (or ``duckdb_con.register`` + ``COPY FROM`` for in-memory
      data) over executemany.

    Args:
        duckdb_con: open DuckDB connection (the analysis database).
        table: final table name; the temp table is ``<table>_new``.
        create_sql: CREATE TABLE statement for ``table`` (name rewritten).
        rows: row tuples to insert; materialized once, so a generator is fine.
    """
    _require_duckdb()
    _check_ident(table)
    tmp = f"{table}_new"
    pattern = re.compile(
        rf"(CREATE\s+(?:OR REPLACE\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?)"
        rf"{re.escape(table)}(?=\s|\()",
        re.IGNORECASE,
    )
    create_sql_new, n = pattern.subn(rf"\g<1>{tmp}", create_sql, count=1)
    if n == 0:
        raise ValueError(f"create_sql does not create table {table!r}: {create_sql!r}")

    rows = list(rows)
    duckdb_con.execute(f"DROP TABLE IF EXISTS {tmp}")
    duckdb_con.execute(create_sql_new)
    if rows:
        ncols = len(rows[0])
        placeholders = ", ".join("?" for _ in range(ncols))
        duckdb_con.executemany(f"INSERT INTO {tmp} VALUES ({placeholders})", rows)
    duckdb_con.execute(f"DROP TABLE IF EXISTS {table}")
    duckdb_con.execute(f"ALTER TABLE {tmp} RENAME TO {table}")
