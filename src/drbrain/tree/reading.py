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

    def __init__(self, path: str | Path) -> None:
        target = Path(path)
        if not target.is_file():
            raise TreeSnapshotError(f"tree snapshot is missing: {target}")
        self.path = target
        self.conn = sqlite3.connect(target.resolve().as_uri() + "?mode=ro", uri=True)
        self.conn.execute("PRAGMA query_only = ON")

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
        return dict(zip(_NODE_COLUMNS, row, strict=False))

    def get_tree_children(self, parent_id: str) -> list[dict]:
        cursor = self.conn.execute(
            "SELECT c.child_id, c.ordinal, c.weight, c.origin, "
            "       n.revision AS child_revision, n.kind, n.state, n.layer "
            "FROM tree_node_children c JOIN tree_nodes n ON n.node_id = c.child_id "
            "WHERE c.parent_id = ? ORDER BY c.ordinal, c.child_id",
            (str(parent_id),),
        )
        columns = [item[0] for item in cursor.description or ()]
        return [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]

    def get_tree_parents(self, child_id: str) -> list[dict]:
        cursor = self.conn.execute(
            "SELECT c.parent_id, c.ordinal, c.weight, c.origin, "
            "       n.revision AS parent_revision, n.state, n.layer "
            "FROM tree_node_children c JOIN tree_nodes n ON n.node_id = c.parent_id "
            "WHERE c.child_id = ? ORDER BY c.parent_id",
            (str(child_id),),
        )
        columns = [item[0] for item in cursor.description or ()]
        return [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]

    def get_document_revision(self, local_id: str, revision: int | None = None) -> dict | None:
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
        return [dict(zip(_NODE_COLUMNS, row, strict=False)) for row in cursor.fetchall()]


__all__ = ["ReadOnlyTreeStore", "TreeSnapshotError"]
