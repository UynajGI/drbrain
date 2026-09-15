"""Display and export body provider: canonical first, legacy fallback (T14).

Display consumers (WebUI paper detail, CLI show/report) and explicit export
read paper content through here instead of touching ``raw.md``/``tree.json``
directly: the canonical store is the first provider, and the read-only legacy
adapter (T13) covers material that has not been migrated yet.  Nothing in
this module writes — a display read must never materialize persistent MD/JSON
artifacts (raw.md, tree.json, report files) just to keep working.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drbrain.storage.content import ContentUnavailableError, read_text, sections

MAX_OUTLINE_NODES = 300
MAX_OUTLINE_DEPTH = 4


@dataclass(frozen=True)
class BodyView:
    """One paper's displayable body; ``source`` names the provider used."""

    local_id: str
    text: str = ""
    source: str = ""  # "canonical" | "raw.md" | "tree.json" | ""
    revision: int | None = None
    sections: tuple[dict[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return bool(self.text) or bool(self.sections)


def _canonical_revision(db, local_id: str) -> int | None:
    row = db.get_document_revision(local_id)
    if row is None:
        return None
    return int(row["revision"])


def read_body(
    db,
    local_id: str,
    *,
    papers_root: str | Path | None = None,
    text_limit: int | None = None,
) -> BodyView:
    """Canonical text/sections first; the legacy adapter covers the rest."""
    revision = _canonical_revision(db, local_id)
    if revision is not None:
        try:
            text = read_text(db, local_id, revision, limit=text_limit)
            canonical_sections = sections(db, local_id, revision)
        except ContentUnavailableError:
            pass
        else:
            return BodyView(
                local_id=local_id,
                text=text,
                source="canonical",
                revision=revision,
                sections=tuple(canonical_sections),
            )
    return _legacy_body(local_id, papers_root)


def _legacy_body(local_id: str, papers_root: str | Path | None) -> BodyView:
    if papers_root is None:
        return BodyView(
            local_id=local_id,
            warnings=("no canonical content and no papers root for the legacy fallback",),
        )
    from drbrain.storage.legacy_content import discover, load

    refs = discover(papers_root, local_id)
    if not refs:
        return BodyView(
            local_id=local_id,
            warnings=("no canonical content and no legacy material",),
        )
    document = load(refs[0])
    return BodyView(
        local_id=local_id,
        text=document.text,
        source=document.text_source,
        sections=tuple(document.sections),
        warnings=tuple(document.warnings),
    )


def body_outline(
    db,
    local_id: str,
    *,
    papers_root: str | Path | None = None,
    max_nodes: int = MAX_OUTLINE_NODES,
) -> list[dict[str, Any]]:
    """Bounded ``{node_id, title, depth, children}`` outline for display."""
    revision = _canonical_revision(db, local_id)
    if revision is not None:
        try:
            canonical_sections = sections(db, local_id, revision)
        except ContentUnavailableError:
            canonical_sections = []
        if canonical_sections:
            return _canonical_outline(canonical_sections, max_nodes=max_nodes)
    if papers_root is None:
        return []
    return _legacy_tree_outline(papers_root, local_id, max_nodes=max_nodes)


def _canonical_outline(
    section_rows: list[dict[str, Any]], *, max_nodes: int
) -> list[dict[str, Any]]:
    paths = [tuple(str(part) for part in (row.get("heading_path") or ())) for row in section_rows]
    outline: list[dict[str, Any]] = []
    for row, path in zip(section_rows, paths):
        if len(outline) >= max_nodes:
            break
        depth = max(0, len(path) - 1)
        if depth > MAX_OUTLINE_DEPTH:
            continue
        children = sum(
            1 for other in paths if len(other) == len(path) + 1 and other[: len(path)] == path
        )
        outline.append(
            {
                "node_id": str(row.get("anchor") or ""),
                "title": path[-1] if path else _first_line(str(row.get("text") or "")),
                "depth": depth,
                "children": children,
            }
        )
    return outline


def _legacy_tree_outline(
    papers_root: str | Path, local_id: str, *, max_nodes: int
) -> list[dict[str, Any]]:
    """The pre-unified tree.json walk, kept for un-migrated material."""
    from drbrain.storage.paths import resolve_paper_dir, tree_json_path

    try:
        directory = resolve_paper_dir(papers_root, local_id)
    except ValueError:
        return []
    if directory is None:
        return []
    path = tree_json_path(directory)
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    # Real tree.json is {"structure": [...]}; a bare list is tolerated for
    # very old files, matching every other legacy consumer.
    structure = payload.get("structure") if isinstance(payload, dict) else payload
    if not isinstance(structure, list):
        return []

    out: list[dict[str, Any]] = []

    def walk(nodes: list, depth: int) -> None:
        if depth > MAX_OUTLINE_DEPTH or len(out) >= max_nodes:
            return
        for node in nodes:
            if not isinstance(node, dict) or len(out) >= max_nodes:
                continue
            out.append(
                {
                    "node_id": str(node.get("node_id") or ""),
                    "title": str(node.get("title") or node.get("summary") or ""),
                    "depth": depth,
                    "children": len(node.get("nodes") or []),
                }
            )
            walk(list(node.get("nodes") or []), depth + 1)

    walk(structure, 0)
    return out


def _first_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return stripped[:120]
    return ""
