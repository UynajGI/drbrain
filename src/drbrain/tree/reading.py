"""Read-only access to a published unified tree snapshot (T37/T43).

A published generation is immutable: readers must never open (and therefore
migrate or write) the snapshot through the writable :class:`Database`.  This
module is the narrow read surface the navigation tools need — the same SELECT
shapes as ``Database``, opened ``mode=ro`` with ``query_only`` so an accidental
write fails instead of corrupting a published generation.

Only reads live here; every write still goes through ``storage.database``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

_NODE_COLUMNS = (
    "node_id",
    "revision",
    "kind",
    "state",
    "layer",
    "local_id",
    "doc_revision",
    "block_id",
    "char_start",
    "char_end",
    "title",
    "summary",
    "heading_path",
    "content_hash",
    "fingerprint",
    "contract_json",
    "origin",
    "provenance_json",
    "created_at",
    "updated_at",
)

_BLOCK_COLUMNS = (
    "block_id",
    "local_id",
    "revision",
    "ordinal",
    "text",
    "text_hash",
    "char_start",
    "char_end",
    "page_start",
    "page_end",
    "line_start",
    "line_end",
    "heading_path",
    "anchor",
    "kind",
    "parser",
    "provenance_json",
)


class TreeSnapshotError(RuntimeError):
    """The published snapshot could not be opened for reading."""


class ReadOnlyTreeStore:
    """The tool-facing read view of one published generation snapshot."""

    def __init__(self, path: str | Path, *, local_ids=None) -> None:
        target = Path(path)
        if not target.is_file():
            raise TreeSnapshotError(f"tree snapshot is missing: {target}")
        self.path = target
        self.conn = sqlite3.connect(target.resolve().as_uri() + "?mode=ro", uri=True)
        self.conn.execute("PRAGMA query_only = ON")
        self._scope: frozenset[str] | None = (
            None if local_ids is None else frozenset(str(value) for value in local_ids)
        )
        self._visibility: dict[str, bool] = {}

    def search_content(self, query: str, **kwargs) -> list[dict]:
        from drbrain.storage.content_search import search_content

        requested = kwargs.pop("local_ids", None)
        scope = self._scope
        if requested is not None:
            scope = frozenset(requested) if scope is None else scope.intersection(requested)
        return search_content(
            self.conn,
            query,
            local_ids=None if scope is None else sorted(scope),
            **kwargs,
        )

    def _allowed(self, row: dict) -> bool:
        if self._scope is None:
            return True
        if row["kind"] == "leaf":
            return str(row["local_id"]) in self._scope
        node_id = row["node_id"]
        if node_id not in self._visibility:
            # A mixed-scope summary also contains outside evidence. Do not expose
            # it to the planner, even if final leaf results would be filtered.
            origins = self.conn.execute(
                "WITH RECURSIVE members(node_id) AS (SELECT ? UNION "
                "SELECT c.child_id FROM tree_node_children c JOIN members m "
                "ON c.parent_id=m.node_id) SELECT DISTINCT n.local_id "
                "FROM members m JOIN tree_nodes n ON n.node_id=m.node_id WHERE n.kind='leaf'",
                (node_id,),
            ).fetchall()
            self._visibility[node_id] = bool(origins) and all(
                str(item[0]) in self._scope for item in origins
            )
        return self._visibility[node_id]

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> ReadOnlyTreeStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ── node surface (mirrors Database.read shapes) ─────────────────────
    def get_tree_node(self, node_id: str) -> dict | None:
        row = self.conn.execute(
            f"SELECT {', '.join(_NODE_COLUMNS)} FROM tree_nodes WHERE node_id = ?",
            (str(node_id),),
        ).fetchone()
        if row is None:
            return None
        result = dict(zip(_NODE_COLUMNS, row, strict=False))
        return result if self._allowed(result) else None

    def get_tree_children(self, parent_id: str) -> list[dict]:
        cursor = self.conn.execute(
            "SELECT c.child_id, c.ordinal, c.weight, c.origin, "
            "       n.revision AS child_revision, n.kind, n.state, n.layer "
            "FROM tree_node_children c JOIN tree_nodes n ON n.node_id = c.child_id "
            "WHERE c.parent_id = ? ORDER BY c.ordinal, c.child_id",
            (str(parent_id),),
        )
        columns = [item[0] for item in cursor.description or ()]
        return [
            dict(zip(columns, row, strict=False))
            for row in cursor.fetchall()
            if self.get_tree_node(str(row[0])) is not None
        ]

    def get_tree_parents(self, child_id: str) -> list[dict]:
        cursor = self.conn.execute(
            "SELECT c.parent_id, c.ordinal, c.weight, c.origin, "
            "       n.revision AS parent_revision, n.state, n.layer "
            "FROM tree_node_children c JOIN tree_nodes n ON n.node_id = c.parent_id "
            "WHERE c.child_id = ? ORDER BY c.parent_id",
            (str(child_id),),
        )
        columns = [item[0] for item in cursor.description or ()]
        return [
            dict(zip(columns, row, strict=False))
            for row in cursor.fetchall()
            if self.get_tree_node(str(row[0])) is not None
        ]

    def get_document_revision(self, local_id: str, revision: int | None = None) -> dict | None:
        if self._scope is not None and str(local_id) not in self._scope:
            return None
        if revision is None:
            row = self.conn.execute(
                "SELECT revision FROM document_revisions WHERE local_id = ? "
                "ORDER BY revision DESC LIMIT 1",
                (str(local_id),),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT revision FROM document_revisions WHERE local_id = ? AND revision = ?",
                (str(local_id), int(revision)),
            ).fetchone()
        if row is None:
            return None
        return {"local_id": str(local_id), "revision": int(row[0])}

    def get_content_blocks(self, local_id: str, revision: int | None = None) -> list[dict]:
        if self._scope is not None and str(local_id) not in self._scope:
            return []
        if revision is None:
            current = self.get_document_revision(local_id)
            if current is None:
                return []
            revision = int(current["revision"])
        cursor = self.conn.execute(
            f"SELECT {', '.join(_BLOCK_COLUMNS)} FROM content_blocks "
            "WHERE local_id = ? AND revision = ? ORDER BY ordinal",
            (str(local_id), int(revision)),
        )
        return [dict(zip(_BLOCK_COLUMNS, row, strict=False)) for row in cursor.fetchall()]

    def list_tree_nodes(
        self, *, kind: str | None = None, state: str | None = None, limit: int = 10_000
    ) -> list[dict]:
        sql = f"SELECT {', '.join(_NODE_COLUMNS)} FROM tree_nodes"
        clauses: list[str] = []
        params: list = []
        if kind is not None:
            clauses.append("kind = ?")
            params.append(str(kind))
        if state is not None:
            clauses.append("state = ?")
            params.append(str(state))
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY layer, node_id LIMIT ?"
        params.append(max(1, int(limit)))
        cursor = self.conn.execute(sql, tuple(params))
        rows = [dict(zip(_NODE_COLUMNS, row, strict=False)) for row in cursor.fetchall()]
        return [row for row in rows if self._allowed(row)]


__all__ = ["ReadOnlyTreeStore", "TreeSnapshotError"]
