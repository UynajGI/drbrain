"""Extraction input construction: canonical store first (plan T16).

Build and extraction consumers read the document structure and section text
through here.  The canonical store is the first provider; the legacy
``tree.json``/``raw.md`` artifacts remain the compatibility path.  The
knowledge-graph algorithms are untouched — only where the input text comes
from changes, so a paper with canonical content but no files on disk can
still be extracted.
"""

from __future__ import annotations

from typing import Any

from drbrain.storage.node_projection import collect_canonical_node_records


def canonical_node_structure(conn, local_id: str) -> list[dict[str, Any]] | None:
    """Nested ``{node_id, title, nodes?}`` structure from canonical nodes.

    Regions become parents of their membership children; leaves stay childless
    so the extraction pipeline's leaf collection works unchanged.  Returns
    ``None`` when the paper has no canonical nodes.
    """
    records = collect_canonical_node_records(conn, local_id, include_regions=True)
    if not records:
        return None
    return _nest(records, _child_edges(conn, records))


def canonical_extraction_inputs(db, local_id: str) -> tuple[list[dict], dict[str, str]] | None:
    """``(structure, section_texts)`` for extraction, canonical only.

    ``section_texts`` maps every canonical node id to its exact text (leaf
    block text or region summary), so the legacy ``get_node_content`` path is
    never consulted for these nodes.  Returns ``None`` when the paper has no
    canonical content, letting callers keep their legacy flow.
    """
    conn = getattr(db, "conn", None)
    if conn is None:
        return None
    records = collect_canonical_node_records(conn, local_id, include_regions=True)
    if not records:
        return None
    structure = _nest(records, _child_edges(conn, records))
    texts = {str(record["node_id"]): str(record["text"]) for record in records}
    return structure, texts


def _child_edges(conn, records: list[dict[str, Any]]) -> dict[str, list[str]]:
    node_ids = {str(record["node_id"]) for record in records}
    edges: dict[str, list[str]] = {}
    try:
        rows = conn.execute(
            "SELECT parent_id, child_id FROM tree_node_children ORDER BY ordinal, child_id"
        ).fetchall()
    except Exception:  # noqa: BLE001 - un-migrated stores have no membership table
        return edges
    for parent_id, child_id in rows:
        parent, child = str(parent_id), str(child_id)
        if parent in node_ids and child in node_ids:
            edges.setdefault(parent, []).append(child)
    return edges


def _nest(records: list[dict[str, Any]], edges: dict[str, list[str]]) -> list[dict[str, Any]]:
    by_id = {str(record["node_id"]): record for record in records}
    child_ids = {child for children in edges.values() for child in children}
    ordered = [str(record["node_id"]) for record in records]

    def payload(node_id: str) -> dict[str, Any]:
        record = by_id[node_id]
        item: dict[str, Any] = {"node_id": node_id, "title": str(record.get("title") or "")}
        children = edges.get(node_id) or []
        if children:
            item["nodes"] = [payload(child) for child in children]
        return item

    return [payload(node_id) for node_id in ordered if node_id not in child_ids]
