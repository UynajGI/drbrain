"""Canonical node-to-text projection (plan T15).

Every downstream text index consumes this projection: embedding and the RAG
layer read canonical tree nodes first (one record per block-backed leaf, plus
region summaries on request) and only fall back to the legacy PageIndex files
through this compatibility entry — consumers never re-slice the body
themselves, and the same block yields the same node id and text hash for
every caller.

``collect_canonical_node_records`` serves the canonical store;
``collect_tree_node_records`` is the legacy file adapter; ``collect_node_records``
is the entry consumers call (canonical first, legacy fallback).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from drbrain.storage.paths import raw_md_path, tree_json_path

#: Projection contract version recorded by preparation metadata.
NODE_PROJECTION_VERSION = "storage.node_projection.v2"

_CANONICAL_LEAF_SQL = """
SELECT n.node_id, n.revision, n.block_id, n.heading_path, n.title,
       b.text, b.text_hash, b.ordinal, b.line_start, b.line_end
FROM tree_nodes n
JOIN content_blocks b ON b.block_id = n.block_id
WHERE n.local_id = ? AND n.doc_revision = ? AND n.state = 'ready' AND n.kind = 'leaf'
ORDER BY b.ordinal, n.node_id
"""

# Region nodes carry no ``local_id`` of their own: membership edges scope them
# to the paper(s) whose leaves they cover, so walk the DAG upward from the
# revision's leaves.
_CANONICAL_REGION_SQL = """
WITH RECURSIVE paper_nodes(node_id) AS (
    SELECT node_id FROM tree_nodes
    WHERE local_id = ? AND doc_revision = ? AND state = 'ready' AND kind = 'leaf'
    UNION
    SELECT c.parent_id FROM tree_node_children c
    JOIN paper_nodes pn ON c.child_id = pn.node_id
)
SELECT n.node_id, n.revision, n.layer, n.heading_path, n.title, n.summary
FROM tree_nodes n
JOIN paper_nodes pn ON pn.node_id = n.node_id
WHERE n.kind = 'region' AND n.state = 'ready'
ORDER BY n.layer, n.node_id
"""


def _load_tree(tree_json: str | Path | dict[str, Any] | None, paper_dir: Path) -> dict[str, Any]:
    if isinstance(tree_json, dict):
        return tree_json
    path = Path(tree_json) if tree_json is not None else tree_json_path(paper_dir)
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def collect_tree_node_records(
    paper_dir: str | Path,
    tree_json: str | Path | dict[str, Any] | None = None,
    *,
    paper_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return one canonical record per PageIndex node in document order.

    ``text`` is always ``<title>\n<body>``.  Body resolution is deterministic:
    explicit ranges, then ``line_num`` ranges, then inline ``text``/``content``
    / summaries.  Missing or malformed trees return an empty list so callers
    can mark the artifact as degraded instead of crashing a whole batch.
    """
    paper_dir = Path(paper_dir)
    tree = _load_tree(tree_json, paper_dir)
    structure = tree.get("structure")
    if not isinstance(structure, list):
        return []

    flat: list[dict[str, Any]] = []

    def flatten(nodes: list[Any]) -> None:
        for node in nodes:
            if not isinstance(node, dict):
                continue
            flat.append(node)
            children = node.get("nodes")
            if isinstance(children, list):
                flatten(children)

    flatten(structure)
    if not flat:
        return []

    needs_raw = any(
        node.get("line_num") is not None
        or (node.get("line_start") is not None and node.get("line_end") is not None)
        for node in flat
    )
    raw_lines: list[str] | None = None
    if needs_raw:
        raw_path = raw_md_path(paper_dir)
        if raw_path.is_file():
            try:
                raw_lines = raw_path.read_text(encoding="utf-8").split("\n")
            except OSError:
                raw_lines = None

    resolved_paper_id = str(paper_id or paper_dir.name)
    records: list[dict[str, Any]] = []
    for index, node in enumerate(flat):
        node_id = str(node.get("node_id") or "").strip()
        if not node_id:
            continue
        title = str(node.get("title") or "").strip()
        body = ""
        line_start: int | None = None
        line_end: int | None = None

        start, end = node.get("line_start"), node.get("line_end")
        used_raw_range = False
        if start is not None and end is not None and raw_lines is not None:
            try:
                candidate_start, candidate_end = int(start), int(end)
            except (TypeError, ValueError):
                candidate_start, candidate_end = -1, -1
            if 0 <= candidate_start <= candidate_end:
                line_start, line_end = candidate_start, candidate_end
                body = "\n".join(raw_lines[line_start:line_end])
                used_raw_range = True
        if not used_raw_range and node.get("line_num") is not None and raw_lines is not None:
            try:
                header_line = int(str(node["line_num"]))
            except (TypeError, ValueError):
                header_line = 0
            if header_line > 0:
                line_start = header_line - 1
                line_end = len(raw_lines)
                for following in flat[index + 1 :]:
                    try:
                        following_line = int(str(following.get("line_num")))
                    except (TypeError, ValueError):
                        continue
                    if following_line > 0:
                        line_end = following_line - 1
                        break
                line_end = max(line_start, line_end)
                body = "\n".join(raw_lines[line_start:line_end])
                used_raw_range = True
        if not used_raw_range:
            for key in ("text", "content", "summary", "prefix_summary"):
                value = node.get(key)
                if value:
                    body = str(value)
                    break

        text = f"{title}\n{body}".strip()
        if not text:
            continue
        records.append(
            {
                "paper_id": resolved_paper_id,
                "node_id": node_id,
                "node_key": f"{resolved_paper_id}:{node_id}",
                "title": title,
                "text": text,
                "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "line_start": line_start,
                "line_end": line_end,
                "origin": "legacy",
            }
        )
    return records


def _heading_tuple(raw: Any) -> tuple[str, ...]:
    if not raw:
        return ()
    if isinstance(raw, (list, tuple)):
        return tuple(str(part) for part in raw)
    try:
        value = json.loads(str(raw))
    except (TypeError, ValueError):
        return ()
    return tuple(str(part) for part in value) if isinstance(value, list) else ()


def _record_title(title: Any, heading: tuple[str, ...]) -> str:
    return str(title or "").strip() or (heading[-1] if heading else "")


def collect_canonical_node_records(
    conn,
    local_id: str,
    *,
    include_regions: bool = False,
) -> list[dict[str, Any]]:
    """Project ready canonical nodes of the latest revision into records.

    Leaves expose exactly one canonical block: ``text`` is the block text and
    ``text_hash`` its canonical hash, so every consumer indexes the same unit
    with the same identity.  Regions (``include_regions``) expose their own
    ``summary`` — never the concatenated child text — so a parent body and its
    child fragments are never the same original leaf twice.
    """
    paper_id = str(local_id)
    try:
        row = conn.execute(
            "SELECT MAX(revision) FROM document_revisions WHERE local_id = ? AND state = 'ready'",
            (paper_id,),
        ).fetchone()
    except Exception:  # noqa: BLE001 - un-migrated stores simply have no canonical view
        return []
    revision = int(row[0]) if row and row[0] is not None else None
    if revision is None:
        return []

    records: list[dict[str, Any]] = []
    for (
        node_id,
        node_revision,
        block_id,
        heading_path,
        title,
        text,
        text_hash,
        ordinal,
        line_start,
        line_end,
    ) in conn.execute(_CANONICAL_LEAF_SQL, (paper_id, revision)).fetchall():
        text = str(text or "")
        if not text:
            continue
        heading = _heading_tuple(heading_path)
        records.append(
            {
                "paper_id": paper_id,
                "node_id": str(node_id),
                "node_key": f"{paper_id}:{node_id}",
                "title": _record_title(title, heading),
                "text": text,
                "text_hash": str(text_hash or "")
                or hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "line_start": int(line_start) if line_start is not None else None,
                "line_end": int(line_end) if line_end is not None else None,
                "kind": "leaf",
                "origin": "canonical",
                "node_revision": max(1, int(node_revision or 1)),
                "block_id": str(block_id or ""),
                "ordinal": int(ordinal or 0),
            }
        )
    if not include_regions:
        return records
    for node_id, node_revision, layer, heading_path, title, summary in conn.execute(
        _CANONICAL_REGION_SQL, (paper_id, revision)
    ).fetchall():
        text = str(summary or "").strip()
        if not text:
            continue
        heading = _heading_tuple(heading_path)
        records.append(
            {
                "paper_id": paper_id,
                "node_id": str(node_id),
                "node_key": f"{paper_id}:{node_id}",
                "title": _record_title(title, heading),
                "text": text,
                "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "line_start": None,
                "line_end": None,
                "kind": "region",
                "origin": "canonical",
                "node_revision": max(1, int(node_revision or 1)),
                "layer": int(layer or 0),
            }
        )
    return records


def collect_node_records(
    conn,
    local_id: str,
    *,
    paper_dir: str | Path | None = None,
    tree_json: str | Path | dict[str, Any] | None = None,
    include_regions: bool = False,
) -> list[dict[str, Any]]:
    """The projection consumers call: canonical nodes first, legacy files only
    through the compatibility entry (``collect_tree_node_records``)."""
    records = collect_canonical_node_records(conn, local_id, include_regions=include_regions)
    if records:
        return records
    if paper_dir is None:
        return []
    return collect_tree_node_records(paper_dir, tree_json, paper_id=str(local_id))


def read_node_text(conn, node_id: str) -> str:
    """Exact text of one ready canonical node (leaf block text or summary)."""
    node = str(node_id)
    try:
        row = conn.execute(
            "SELECT b.text FROM tree_nodes n JOIN content_blocks b ON b.block_id = n.block_id "
            "WHERE n.node_id = ? AND n.state = 'ready' AND n.kind = 'leaf'",
            (node,),
        ).fetchone()
        if row and row[0]:
            return str(row[0])
        row = conn.execute(
            "SELECT summary FROM tree_nodes WHERE node_id = ? AND state = 'ready' "
            "AND kind = 'region'",
            (node,),
        ).fetchone()
    except Exception:  # noqa: BLE001 - un-migrated stores have no canonical nodes
        return ""
    return str(row[0] or "") if row else ""


__all__ = [
    "NODE_PROJECTION_VERSION",
    "collect_canonical_node_records",
    "collect_node_records",
    "collect_tree_node_records",
    "read_node_text",
]
