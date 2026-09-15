"""Unified canonical-content read API (plan T10).

Readers never open ``raw.md``/``tree.json``/``pages.json``: everything is
served from the ``content_blocks`` table.  Reads are by document revision
(latest by default) or by explicit revision for audit/old-evidence paths.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from drbrain.storage.database import Database


class ContentUnavailableError(LookupError):
    """The requested document revision or range does not exist."""


@dataclass(frozen=True)
class BlockView:
    block_id: str
    local_id: str
    revision: int
    ordinal: int
    text: str
    text_hash: str
    char_start: int
    char_end: int
    page_start: int | None
    page_end: int | None
    line_start: int | None
    line_end: int | None
    heading_path: tuple[str, ...]
    anchor: str
    kind: str
    parser: str

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> BlockView:
        import json

        raw_path = row.get("heading_path") or "[]"
        try:
            heading = tuple(json.loads(raw_path))
        except (TypeError, ValueError):
            heading = ()
        return cls(
            block_id=str(row["block_id"]),
            local_id=str(row["local_id"]),
            revision=int(row["revision"]),
            ordinal=int(row["ordinal"]),
            text=str(row["text"]),
            text_hash=str(row["text_hash"]),
            char_start=int(row["char_start"]),
            char_end=int(row["char_end"]),
            page_start=row.get("page_start"),
            page_end=row.get("page_end"),
            line_start=row.get("line_start"),
            line_end=row.get("line_end"),
            heading_path=heading,
            anchor=str(row.get("anchor") or ""),
            kind=str(row.get("kind") or "paragraph"),
            parser=str(row.get("parser") or ""),
        )


def resolve_revision(db: Database, local_id: str, revision: int | None = None) -> int:
    """Resolve a revision selector to a concrete revision (latest by default)."""
    row = db.get_document_revision(local_id, revision)
    if row is None:
        raise ContentUnavailableError(
            f"no canonical content for {local_id}"
            + (f"@{revision}" if revision is not None else "")
        )
    return int(row["revision"])


def iter_blocks(db: Database, local_id: str, revision: int | None = None) -> Iterator[BlockView]:
    """Yield blocks in reading order for one (resolved) revision."""
    resolved = resolve_revision(db, local_id, revision)
    for row in db.get_content_blocks(local_id, resolved):
        yield BlockView.from_row(row)


def read_text(
    db: Database, local_id: str, revision: int | None = None, *, limit: int | None = None
) -> str:
    """Reconstruct the canonical text of one revision."""
    parts: list[str] = []
    length = 0
    for block in iter_blocks(db, local_id, revision):
        parts.append(block.text)
        length += len(block.text)
        if limit is not None and length >= limit:
            break
    text = "".join(parts)
    return text if limit is None else text[:limit]


def text_hash(text: str) -> str:
    """Stable hash of a read result (same bytes -> same digest)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_range(
    db: Database,
    local_id: str,
    char_start: int,
    char_end: int,
    revision: int | None = None,
) -> str:
    """Read an exact half-open char range, crossing block boundaries.

    Ranges outside the canonical text are an error rather than a silent
    short read: callers persist source spans and must not receive a
    different amount of text than they recorded.
    """
    resolved = resolve_revision(db, local_id, revision)
    blocks = list(iter_blocks(db, local_id, resolved))
    if not blocks:
        raise ContentUnavailableError(f"{local_id}@{resolved} has no blocks")
    total = blocks[-1].char_end
    if char_start < 0 or char_end <= char_start or char_end > total:
        raise ContentUnavailableError(
            f"range [{char_start}, {char_end}) outside {local_id}@{resolved} [0, {total})"
        )
    pieces: list[str] = []
    for block in blocks:
        if block.char_end <= char_start:
            continue
        if block.char_start >= char_end:
            break
        begin = max(char_start, block.char_start) - block.char_start
        finish = min(char_end, block.char_end) - block.char_start
        pieces.append(block.text[begin:finish])
    return "".join(pieces)


def block_at_offset(
    db: Database, local_id: str, offset: int, revision: int | None = None
) -> BlockView | None:
    resolved = resolve_revision(db, local_id, revision)
    for block in iter_blocks(db, local_id, resolved):
        if block.char_start <= offset < block.char_end:
            return block
    return None


def blocks_in_pages(
    db: Database, local_id: str, page_start: int, page_end: int, revision: int | None = None
) -> list[BlockView]:
    """Blocks overlapping a 1-based inclusive PDF page span (no page fakes)."""
    if page_start < 1 or page_end < page_start:
        raise ValueError("page span must be 1-based inclusive")
    return [
        block
        for block in iter_blocks(db, local_id, revision)
        if block.page_start is not None
        and block.page_end is not None
        and block.page_start <= page_end
        and block.page_end >= page_start
    ]


def blocks_in_lines(
    db: Database, local_id: str, line_start: int, line_end: int, revision: int | None = None
) -> list[BlockView]:
    """Blocks overlapping a 1-based inclusive line span (MD/TeX)."""
    if line_start < 1 or line_end < line_start:
        raise ValueError("line span must be 1-based inclusive")
    return [
        block
        for block in iter_blocks(db, local_id, revision)
        if block.line_start is not None
        and block.line_end is not None
        and block.line_start <= line_end
        and block.line_end >= line_start
    ]


def sections(db: Database, local_id: str, revision: int | None = None) -> list[dict[str, Any]]:
    """Group blocks by heading path; each group is one structural section."""
    resolved = resolve_revision(db, local_id, revision)
    grouped: dict[tuple[str, ...], dict[str, Any]] = {}
    order: list[tuple[str, ...]] = []
    for block in iter_blocks(db, local_id, resolved):
        key = block.heading_path
        if key not in grouped:
            grouped[key] = {
                "heading_path": list(key),
                "anchor": block.anchor,
                "char_start": block.char_start,
                "char_end": block.char_end,
                "block_ids": [],
                "tokens": 0,
            }
            order.append(key)
        entry = grouped[key]
        entry["char_end"] = max(entry["char_end"], block.char_end)
        entry["block_ids"].append(block.block_id)
    result = []
    for key in order:
        entry = grouped[key]
        entry["text"] = read_range(db, local_id, entry["char_start"], entry["char_end"], resolved)
        entry["char_hash"] = text_hash(entry["text"])
        result.append(entry)
    return result


def read_section(
    db: Database,
    local_id: str,
    anchor: str,
    revision: int | None = None,
) -> dict[str, Any]:
    """Read one section by heading anchor (exact match first, then suffix)."""
    for section in sections(db, local_id, revision):
        if section["anchor"] == anchor:
            return section
    for section in sections(db, local_id, revision):
        if section["heading_path"] and section["heading_path"][-1] == anchor:
            return section
    raise ContentUnavailableError(f"no section {anchor!r} in {local_id}")


def summarize_coverage(db: Database, local_id: str, revision: int | None = None) -> dict[str, Any]:
    """Audit one revision's block coverage (contiguity is guaranteed at write)."""
    resolved = resolve_revision(db, local_id, revision)
    blocks: Sequence[BlockView] = list(iter_blocks(db, local_id, resolved))
    return {
        "local_id": local_id,
        "revision": resolved,
        "blocks": len(blocks),
        "chars": blocks[-1].char_end if blocks else 0,
        "kinds": sorted({block.kind for block in blocks}),
        "canonical_hash": text_hash("".join(block.text for block in blocks)),
    }
