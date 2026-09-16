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
#: Bounded section preview used by the literature page (per outline node).
DEFAULT_EXCERPT_CHARS = 400


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


def _excerpt(text: str, limit: int) -> str:
    """A bounded, whitespace-normalised preview of one section's text."""
    collapsed = " ".join(str(text or "").split())
    if limit <= 0 or len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


def body_view(
    db,
    local_id: str,
    *,
    papers_root: str | Path | None = None,
    max_nodes: int = MAX_OUTLINE_NODES,
    excerpt_chars: int = DEFAULT_EXCERPT_CHARS,
) -> dict[str, Any]:
    """Outline + bounded excerpts + provider provenance, in one read.

    This is the literature page's display contract: the canonical store is the
    first provider and the read-only legacy adapter covers un-migrated
    material, exactly like :func:`body_outline`.  With ``excerpt_chars > 0``
    every node additionally carries a bounded ``excerpt`` and, when the
    provider has one, a stable locator (``char_start``/``char_end`` for the
    canonical store, whatever the legacy tree records) so the UI can point at
    a position and let the user copy it.
    """
    limit = max(0, int(excerpt_chars))
    revision = _canonical_revision(db, local_id)
    if revision is not None:
        try:
            canonical_sections = sections(db, local_id, revision)
        except ContentUnavailableError:
            canonical_sections = []
        if canonical_sections:
            return {
                "source": "canonical",
                "revision": revision,
                "warnings": [],
                "available": True,
                "nodes": _canonical_outline(
                    canonical_sections, max_nodes=max_nodes, excerpt_chars=limit
                ),
            }
    if papers_root is None:
        return {
            "source": "",
            "revision": None,
            "warnings": ["no canonical content and no papers root for the legacy fallback"],
            "available": False,
            "nodes": [],
        }
    nodes = _legacy_tree_outline(papers_root, local_id, max_nodes=max_nodes, excerpt_chars=limit)
    return {
        "source": "tree.json" if nodes else "",
        "revision": None,
        "warnings": [] if nodes else ["no canonical content and no legacy tree.json"],
        "available": bool(nodes),
        "nodes": nodes,
    }


def _canonical_outline(
    section_rows: list[dict[str, Any]], *, max_nodes: int, excerpt_chars: int = 0
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
        node: dict[str, Any] = {
            "node_id": str(row.get("anchor") or ""),
            "title": path[-1] if path else _first_line(str(row.get("text") or "")),
            "depth": depth,
            "children": children,
        }
        if excerpt_chars > 0:
            node["excerpt"] = _excerpt(str(row.get("text") or ""), excerpt_chars)
            node["char_start"] = int(row.get("char_start") or 0)
            node["char_end"] = int(row.get("char_end") or 0)
        outline.append(node)
    return outline


def _legacy_tree_outline(
    papers_root: str | Path,
    local_id: str,
    *,
    max_nodes: int,
    excerpt_chars: int = 0,
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
            entry: dict[str, Any] = {
                "node_id": str(node.get("node_id") or ""),
                "title": str(node.get("title") or node.get("summary") or ""),
                "depth": depth,
                "children": len(node.get("nodes") or []),
            }
            if excerpt_chars > 0:
                entry["excerpt"] = _excerpt(
                    str(node.get("text") or node.get("summary") or ""), excerpt_chars
                )
                if node.get("start_index") is not None:
                    entry["char_start"] = int(node["start_index"])
                if node.get("end_index") is not None:
                    entry["char_end"] = int(node["end_index"])
                if node.get("line_num") is not None:
                    entry["line_start"] = int(node["line_num"])
            out.append(entry)
            walk(list(node.get("nodes") or []), depth + 1)

    walk(structure, 0)
    return out


def _first_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return stripped[:120]
    return ""


# ── explicit export (T53) ────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExportBundle:
    """Self-contained export of one paper's body and node tree (T53)."""

    local_id: str
    source: str
    revision: int | None
    text: str
    structure: tuple[dict[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "local_id": self.local_id,
            "source": self.source,
            "revision": self.revision,
            "text": self.text,
            "structure": list(self.structure),
            "warnings": list(self.warnings),
        }


def export_structure(
    db,
    local_id: str,
    *,
    papers_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Nested, self-contained node tree with inline text (explicit export).

    Canonical papers export their region/leaf structure with each node's exact
    text attached, so the tree is usable without ``raw.md``; un-migrated papers
    re-emit their legacy ``tree.json`` structure unchanged.
    """
    revision = _canonical_revision(db, local_id)
    if revision is not None:
        from drbrain.extractor.context import canonical_node_structure

        structure = canonical_node_structure(db.conn, local_id)
        if structure:
            from drbrain.storage.node_projection import collect_canonical_node_records

            records = collect_canonical_node_records(db.conn, local_id, include_regions=True)
            texts = {str(record["node_id"]): str(record["text"]) for record in records}
            return _attach_text(structure, texts)
    if papers_root is None:
        return []
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
    structure = payload.get("structure") if isinstance(payload, dict) else payload
    return list(structure) if isinstance(structure, list) else []


def _attach_text(nodes: list[dict[str, Any]], texts: dict[str, str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        item = dict(node)
        node_id = str(item.get("node_id") or "")
        if node_id in texts:
            item["text"] = texts[node_id]
        children = item.get("nodes") or item.get("children") or []
        if isinstance(children, list) and children:
            item["nodes"] = _attach_text(children, texts)
        out.append(item)
    return out


def export_paper_view(
    db,
    local_id: str,
    *,
    papers_root: str | Path | None = None,
) -> ExportBundle:
    """Explicitly export body + structure; only this path materializes output."""
    body = read_body(db, local_id, papers_root=papers_root)
    structure = export_structure(db, local_id, papers_root=papers_root)
    return ExportBundle(
        local_id=local_id,
        source=body.source,
        revision=body.revision,
        text=body.text,
        structure=tuple(structure),
        warnings=tuple(body.warnings),
    )
