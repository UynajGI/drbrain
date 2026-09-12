"""Canonical PageIndex node-to-text projection.

Every downstream text index should consume this projection.  Keeping the
line-range and inline-text fallbacks here prevents embedding, FTS and
LlamaIndex from silently indexing different representations of one section.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from drbrain.storage.paths import raw_md_path, tree_json_path


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
        if start is not None and end is not None and raw_lines is not None:
            line_start, line_end = int(start), int(end)
            body = "\n".join(raw_lines[line_start:line_end])
        elif node.get("line_num") is not None and raw_lines is not None:
            line_start = int(node["line_num"]) - 1
            line_end = len(raw_lines)
            for following in flat[index + 1 :]:
                if following.get("line_num") is not None:
                    line_end = int(following["line_num"]) - 1
                    break
            line_end = max(line_start, line_end)
            body = "\n".join(raw_lines[line_start:line_end])
        else:
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
                "line_start": line_start,
                "line_end": line_end,
            }
        )
    return records


__all__ = ["collect_tree_node_records"]
