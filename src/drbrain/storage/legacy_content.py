"""Read-only adapters for pre-unified artifacts (plan T13).

Legacy per-paper directories (``raw.md`` + ``tree.json`` + ``source.*``, nested
DOI layouts, SDK outputs) remain readable through the unified read protocol
without ever writing: migration (T49–T53) decides what to import, and default
reads must not create new files, upgrade databases, or silently pick one of
two conflicting copies (ambiguity is an error, not a coin flip).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from drbrain.storage.paths import (
    iter_paper_dirs,
    paper_id_from_dir,
    raw_md_path,
    resolve_paper_dir,
    source_pdf_path,
    tree_json_path,
)


class LegacyContentError(RuntimeError):
    """Legacy material could not be read (ambiguous or malformed)."""


@dataclass(frozen=True)
class LegacyMaterialRef:
    """Where one legacy document lives; no writes ever touch these paths."""

    local_id: str
    paper_dir: Path
    raw_md: Path | None = None
    tree_json: Path | None = None
    source_file: Path | None = None
    layout: str = "canonical"

    def existing_paths(self) -> tuple[Path, ...]:
        return tuple(
            path for path in (self.raw_md, self.tree_json, self.source_file) if path is not None
        )


@dataclass
class LegacyDocument:
    """Normalized, read-only view of one legacy document."""

    ref: LegacyMaterialRef
    text: str = ""
    tree: dict[str, Any] | None = None
    sections: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    text_source: str = ""  # "raw.md" | "tree.json" | ""


def _symlink_safe(path: Path) -> bool:
    try:
        return not path.is_symlink()
    except OSError:
        return False


def discover(
    papers_root: str | Path,
    local_id: str | None = None,
    *,
    limit: int | None = None,
) -> list[LegacyMaterialRef]:
    """Find legacy materials under ``papers_root`` without writing anything."""
    if local_id is not None:
        directory = resolve_paper_dir(papers_root, local_id)
        if directory is None:
            return []
        return [_ref_for(directory, local_id)]
    refs: list[LegacyMaterialRef] = []
    for directory in iter_paper_dirs(papers_root):
        if not _symlink_safe(directory):
            continue
        try:
            resolved_id = paper_id_from_dir(directory, papers_root)
        except Exception:  # noqa: BLE001 - unreadable dirs are skipped, not fatal
            logger.warning("[legacy] cannot derive paper id from {}", directory)
            continue
        refs.append(_ref_for(directory, resolved_id))
        if limit is not None and len(refs) >= limit:
            break
    return refs


def _ref_for(directory: Path, local_id: str) -> LegacyMaterialRef:
    raw = raw_md_path(directory)
    tree = tree_json_path(directory)
    source = source_pdf_path(directory)
    if not source.exists():
        candidates = sorted(
            item
            for item in directory.iterdir()
            if item.is_file() and item.name.startswith("source.")
        )
        source = candidates[0] if candidates else source
    return LegacyMaterialRef(
        local_id=local_id,
        paper_dir=directory,
        raw_md=raw if raw.is_file() else None,
        tree_json=tree if tree.is_file() else None,
        source_file=source if source.is_file() else None,
        layout="legacy" if directory.name != local_id else "canonical",
    )


def _read_tree(path: Path) -> tuple[dict[str, Any] | None, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError) as exc:
        return None, f"unreadable tree.json: {exc}"
    if not isinstance(payload, dict):
        return None, "tree.json is not an object"
    return payload, ""


def _flatten_tree(
    nodes: Sequence[Any],
    heading_path: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Flatten a PageIndex structure into section records with heading paths."""
    flat: list[dict[str, Any]] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        title = str(node.get("title") or "").strip()
        path = (*heading_path, title) if title else heading_path
        entry: dict[str, Any] = {
            "title": title,
            "heading_path": list(path),
            "node_id": str(node.get("node_id") or ""),
            "text": str(node.get("text") or ""),
            "summary": str(node.get("summary") or node.get("prefix_summary") or ""),
        }
        for key in ("line_num", "start_index", "end_index"):
            if node.get(key) is not None:
                entry[key] = node[key]
        flat.append(entry)
        children = node.get("nodes") or node.get("children") or []
        if isinstance(children, list) and children:
            flat.extend(_flatten_tree(children, path))
    return flat


def load(ref: LegacyMaterialRef) -> LegacyDocument:
    """Load one legacy document read-only, preferring raw.md over tree text."""
    document = LegacyDocument(ref=ref)
    if ref.raw_md is not None:
        try:
            document.text = ref.raw_md.read_text(encoding="utf-8", errors="replace")
            document.text_source = "raw.md"
        except OSError as exc:
            document.warnings.append(f"cannot read raw.md: {exc}")
    if ref.tree_json is not None:
        tree, warning = _read_tree(ref.tree_json)
        if warning:
            document.warnings.append(warning)
        else:
            document.tree = tree
            structure = tree.get("structure") if tree else None
            if isinstance(structure, list):
                document.sections = _flatten_tree(structure)
    if not document.text and document.sections:
        document.text = "\n\n".join(
            section["text"] for section in document.sections if section["text"]
        )
        document.text_source = "tree.json"
        document.warnings.append("raw.md missing; text reconstructed from tree.json")
    if not document.text and document.sections == [] and ref.tree_json is None:
        document.warnings.append("no raw.md and no tree.json")
    return document


def iter_sections(document: LegacyDocument) -> Iterable[dict[str, Any]]:
    return iter(document.sections)


def section_text(document: LegacyDocument, node_id: str) -> str:
    """Exact text of one legacy tree node (no synthesized ranges)."""
    for section in document.sections:
        if section["node_id"] == node_id:
            return str(section["text"])
    raise LegacyContentError(f"no legacy node {node_id!r} in {document.ref.local_id}")
