"""Node content retrieval — on-demand loading of node text from markdown files."""

from __future__ import annotations

import json
from pathlib import Path


def _find_node_by_id(structure: list[dict], node_id: str) -> dict | None:
    """Find a node in the tree by its node_id."""
    for node in structure:
        if node.get("node_id") == node_id:
            return node
        found = _find_node_by_id(node.get("nodes", []), node_id)
        if found:
            return found
    return None


def _collect_line_ranges(structure: list[dict]) -> list[dict]:
    """Flatten tree into a sorted list of {node_id, title, line_num} for range calculation."""
    items = []
    for node in structure:
        items.append(
            {
                "node_id": node.get("node_id", ""),
                "title": node.get("title", ""),
                "line_num": node.get("line_num", 0),
            }
        )
        items.extend(_collect_line_ranges(node.get("nodes", [])))
    return sorted(items, key=lambda x: x["line_num"])


def get_node_content(
    md_path: str | Path,
    structure: list[dict],
    node_id: str,
) -> str | None:
    """Load the markdown text content for a specific node by its node_id.

    Reads the MD file and extracts lines from the node's header to the
    next sibling/parent header. Returns None if node_id not found.
    """
    target = _find_node_by_id(structure, node_id)
    if not target:
        return None

    md_path = Path(md_path)
    if not md_path.exists():
        return None

    with open(md_path, encoding="utf-8") as f:
        lines = f.readlines()

    start_line = target["line_num"] - 1  # 0-indexed

    # Find end: next node at same or higher level
    all_nodes = _collect_line_ranges(structure)
    end_line = len(lines)
    found_target = False
    for n in all_nodes:
        if n["line_num"] == target["line_num"]:
            found_target = True
            continue
        if found_target and n["line_num"] > target["line_num"]:
            end_line = n["line_num"] - 1  # exclusive, 1-indexed → 0-indexed
            break

    return "".join(lines[start_line:end_line]).strip()


def get_node_content_by_title(
    md_path: str | Path,
    structure: list[dict],
    title: str,
) -> str | None:
    """Load content for the first node matching the given title."""

    def _find(nodes: list[dict]) -> str | None:
        for node in nodes:
            if node.get("title", "").strip().lower() == title.strip().lower():
                return node.get("node_id")
            found = _find(node.get("nodes", []))
            if found:
                return found
        return None

    node_id = _find(structure)
    if not node_id:
        return None
    return get_node_content(md_path, structure, node_id)


def get_document_structure_json(structure: list[dict]) -> str:
    """Return tree structure as JSON string with text fields removed (saves tokens)."""

    def _remove_text(nodes: list[dict]) -> list[dict]:
        cleaned = []
        for node in nodes:
            c = {k: v for k, v in node.items() if k != "text"}
            if "nodes" in c:
                c["nodes"] = _remove_text(c["nodes"])
            cleaned.append(c)
        return cleaned

    return json.dumps(_remove_text(structure), ensure_ascii=False, indent=2)


def _write_node_id(data, node_id: int = 0) -> int:
    """Assign sequential node_ids to tree nodes."""
    if isinstance(data, dict):
        data["node_id"] = str(node_id).zfill(4)
        node_id += 1
        for key in list(data.keys()):
            if "nodes" in key:
                node_id = _write_node_id(data[key], node_id)
    elif isinstance(data, list):
        for item in data:
            node_id = _write_node_id(item, node_id)
    return node_id


def _format_structure(structure, order: list[str] | None = None):
    """Reorder dict keys and remove empty nodes arrays."""
    if not order:
        return structure
    if isinstance(structure, dict):
        if "nodes" in structure:
            structure["nodes"] = _format_structure(structure["nodes"], order)
        if not structure.get("nodes"):
            structure.pop("nodes", None)
        return {k: structure[k] for k in order if k in structure}
    elif isinstance(structure, list):
        return [_format_structure(item, order) for item in structure]
    return structure


def _create_clean_structure_for_description(structure) -> dict | list:
    """Create structure without text fields for doc description generation."""
    if isinstance(structure, dict):
        clean = {}
        for key in ("title", "node_id", "summary", "prefix_summary"):
            if key in structure:
                clean[key] = structure[key]
        if "nodes" in structure and structure["nodes"]:
            clean["nodes"] = _create_clean_structure_for_description(structure["nodes"])
        return clean
    elif isinstance(structure, list):
        return [_create_clean_structure_for_description(item) for item in structure]
    return structure
