"""The single canonical-content write path (plan T22/T51).

Both ingest and the legacy migration write through here: one document
revision, its boundary-faithful blocks, and one published leaf per block,
inside one transaction.  Re-writing the same text for the same document
reuses the existing revision (nothing new is stored), and a changed document
gets a new revision while the old one stays readable — a migration must never
be "completed" by deleting the previous data.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

from loguru import logger

#: media_type values accepted by the content store.
MEDIA_TYPES = ("pdf", "tex", "md")


class CanonicalWriteError(RuntimeError):
    """The canonical write could not be performed."""


def media_type_for_suffix(suffix: str) -> str:
    suffix = str(suffix or "").lower()
    if suffix in ("", ".pdf"):
        return "pdf"
    if suffix in (".tex", ".latex"):
        return "tex"
    return "md"


def write_canonical_content(
    db,
    local_id: str,
    text: str,
    *,
    media_type: str,
    source_hash: str = "",
    parser: str = "",
    parser_revision: str = "",
    page_marks: Sequence[tuple[int, int]] | None = None,
    stale_previous: bool = False,
) -> dict[str, Any]:
    """Write one document revision, its blocks and one published leaf per block.

    ``stale_previous`` marks the previously latest revision ``stale`` after a
    successful write (used when re-deriving inconsistent content); it never
    deletes it.
    """
    if media_type not in MEDIA_TYPES:
        raise CanonicalWriteError(f"unsupported media_type {media_type!r}")
    if not str(text).strip():
        return {"ok": False, "reason": "empty_canonical_text"}

    from drbrain.tree.blocks import build_content_blocks
    from drbrain.tree.contracts import LeafRef, NodeRecord, leaf_node_id

    canonical_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    latest = db.get_document_revision(local_id)
    if (
        latest is not None
        and str(latest.get("canonical_hash")) == canonical_hash
        and str(latest.get("state")) == "ready"
    ):
        count = db.count_content_blocks(local_id, int(latest["revision"]))
        return {
            "ok": True,
            "reused": True,
            "revision": int(latest["revision"]),
            "blocks": count,
            "pages": False,
            "hash": canonical_hash,
        }

    revision = db.next_document_revision(local_id)
    blocks = build_content_blocks(
        text,
        local_id=local_id,
        revision=revision,
        media_type=media_type,
        parser=parser,
        page_marks=page_marks,
    )
    written = 0
    previous_revision = int(latest["revision"]) if latest is not None else None
    with db.transaction():
        db.upsert_document_revision(
            local_id,
            revision,
            source_hash=source_hash,
            canonical_hash=canonical_hash,
            backend=parser,
            media_type=media_type,
            parser_revision=parser_revision,
        )
        written = db.insert_content_blocks(blocks)
        for block in blocks:
            ref = LeafRef(
                local_id=local_id,
                revision=revision,
                block_id=block.block_id,
                char_start=0,
                char_end=len(block.text),
            )
            leaf = NodeRecord(
                node_id=leaf_node_id(ref),
                revision=1,
                kind="leaf",
                state="ready",
                layer=0,
                content_hash=block.text_hash,
                leaf=ref,
                heading_path=block.heading_path,
            )
            db.insert_tree_node(leaf, publish=True)
        if stale_previous and previous_revision is not None:
            try:
                db.set_document_revision_state(local_id, previous_revision, "stale")
            except ValueError:  # pragma: no cover - revision vanished mid-write
                logger.warning(
                    "[tree] could not mark {}@{} stale after rewrite",
                    local_id,
                    previous_revision,
                )
    return {
        "ok": True,
        "reused": False,
        "revision": revision,
        "blocks": written,
        "pages": bool(page_marks),
        "hash": canonical_hash,
    }
