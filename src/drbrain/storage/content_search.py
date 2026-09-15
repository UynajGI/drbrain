"""Canonical FTS reader shared by the live database and immutable snapshots."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence


def search_content(
    conn: sqlite3.Connection,
    query: str,
    *,
    local_id: str | None = None,
    local_ids: Sequence[str] | None = None,
    limit: int = 50,
    snippet_tokens: int = 24,
) -> list[dict]:
    query = str(query).strip()
    if not query:
        raise ValueError("search query must be non-empty")
    scope = None if local_ids is None else {str(value) for value in local_ids}
    if local_id is not None:
        scope = {local_id} if scope is None else scope & {local_id}
    if scope is not None:
        if not scope:
            return []
        if any(not value or value != value.strip() or "\x00" in value for value in scope):
            raise ValueError("invalid paper local_id")
    sql = (
        "SELECT b.block_id, b.local_id, b.revision, b.ordinal, b.char_start, "
        "b.char_end, b.page_start, b.page_end, b.line_start, b.line_end, "
        "b.heading_path, b.kind, b.text_hash, bm25(content_fts) AS score, "
        "snippet(content_fts, 0, '[', ']', '…', ?) AS snippet "
        "FROM content_fts JOIN content_blocks b ON b.rowid = content_fts.rowid "
        "WHERE content_fts MATCH ?"
    )
    params = [max(1, int(snippet_tokens)), query]
    if scope is not None:
        sql += " AND b.local_id IN (SELECT value FROM json_each(?))"
        params.append(json.dumps(sorted(scope)))
    sql += " ORDER BY score, b.local_id, b.ordinal LIMIT ?"
    params.append(max(1, int(limit)))
    try:
        cursor = conn.execute(sql, params)
    except sqlite3.OperationalError as exc:
        raise ValueError(f"invalid FTS query {query!r}: {exc}") from exc
    columns = [item[0] for item in cursor.description or ()]
    rows = [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]
    for row in rows:
        row["score"] = float(row["score"])
    return rows
