"""In-place block repack: paragraph-sized leaves for existing corpora (T09).

The 10k flow test measured a 118-character median leaf — 23% under 40
characters, 87k bare heading lines — against RAPTOR's 100-token ≈ 400-char
chunking: the canonical partition had become one block per structural line
fragment, so leaf vectors and region summaries saw sentence shards instead
of passages.  This migration re-segments each *ready* revision with the
boundary-faithful builder under the current ``BlockPolicy`` (``min_chars``).

The revision, its ``canonical_hash`` and the text itself are unchanged —
only the boundaries move — and ``insert_content_blocks`` re-proves the
canonical hash after the rewrite, so a merge bug fails closed instead of
silently dropping text.  Page provenance is carried over by deriving
``page_marks`` from the existing blocks' page spans.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from drbrain.tree.blocks import BlockPolicy, build_content_blocks


class RepackError(RuntimeError):
    """A revision could not be re-segmented safely."""


@dataclass
class RepackItem:
    local_id: str
    revision: int
    blocks_before: int
    blocks_after: int


@dataclass
class RepackPlan:
    revisions: int = 0
    items: list[RepackItem] = field(default_factory=list)
    skipped: int = 0
    failed: list[dict[str, Any]] = field(default_factory=list)

    @property
    def blocks_before(self) -> int:
        return sum(item.blocks_before for item in self.items)

    @property
    def blocks_after(self) -> int:
        return sum(item.blocks_after for item in self.items)

    def to_json(self) -> dict[str, Any]:
        return {
            "revisions": self.revisions,
            "repack": [item.__dict__ for item in self.items],
            "skipped": self.skipped,
            "failed": list(self.failed),
            "blocks_before": self.blocks_before,
            "blocks_after": self.blocks_after,
        }


def _page_marks(rows: Sequence[dict]) -> list[tuple[int, int]] | None:
    """Derive PDF page marks from existing block spans (start of each page)."""
    marks: dict[int, int] = {}
    for row in rows:
        page = row.get("page_start")
        if page is None:
            continue
        marks.setdefault(int(page), int(row["char_start"]))
    if not marks:
        return None
    ordered = sorted((page, offset) for page, offset in marks.items())
    if ordered[0][1] != 0:
        return None
    return ordered


def _rebuild(
    local_id: str,
    revision_row: dict[str, Any],
    rows: Sequence[dict],
    policy: BlockPolicy,
) -> list:
    """Re-segment one revision's text; raises when the hash does not match."""
    revision = int(revision_row["revision"])
    text = "".join(str(row["text"]) for row in rows)
    expected = str(revision_row.get("canonical_hash") or "")
    actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if not expected or actual != expected:
        raise RepackError("stored blocks do not reproduce the canonical hash")
    return build_content_blocks(
        text,
        local_id=local_id,
        revision=revision,
        media_type=str(revision_row.get("media_type") or "md"),
        parser=str(revision_row.get("backend") or "repack"),
        page_marks=_page_marks(rows),
        policy=policy,
    )


def _same_segmentation(rows: Sequence[dict], blocks: Sequence) -> bool:
    if len(rows) != len(blocks):
        return False
    for row, block in zip(sorted(rows, key=lambda item: int(item["ordinal"])), blocks):
        if (
            int(row["ordinal"]) != block.ordinal
            or int(row["char_start"]) != block.char_start
            or int(row["char_end"]) != block.char_end
            or str(row["text_hash"]) != block.text_hash
        ):
            return False
    return True


def _ready_revisions(db, local_ids: Sequence[str] | None = None) -> list[tuple[str, int]]:
    sql = (
        "SELECT local_id, revision FROM document_revisions WHERE state = 'ready' "
        "ORDER BY local_id, revision"
    )
    params: tuple = ()
    if local_ids:
        placeholders = ",".join("?" for _ in local_ids)
        sql = (
            f"SELECT local_id, revision FROM document_revisions "
            f"WHERE state = 'ready' AND local_id IN ({placeholders}) ORDER BY local_id, revision"
        )
        params = tuple(local_ids)
    return [(str(row[0]), int(row[1])) for row in db.conn.execute(sql, params).fetchall()]


def plan_repack(
    db,
    *,
    local_ids: Sequence[str] | None = None,
    max_items: int = 0,
    policy: BlockPolicy | None = None,
) -> RepackPlan:
    """Scan ready revisions and record which need re-segmentation.

    Building the blocks here is what decides it (the segmentation is
    deterministic), but the result is only counted — ``apply_repack``
    rebuilds it inside the replacement transaction.
    """
    policy = policy or BlockPolicy()
    plan = RepackPlan()
    for local_id, revision in _ready_revisions(db, local_ids):
        if max_items and len(plan.items) >= max_items:
            break
        plan.revisions += 1
        rows = db.get_content_blocks(local_id, revision)
        if not rows:
            plan.skipped += 1
            continue
        revision_row = db.get_document_revision(local_id, revision)
        if revision_row is None:  # pragma: no cover - listing raced a delete
            plan.skipped += 1
            continue
        try:
            blocks = _rebuild(local_id, revision_row, rows, policy)
        except Exception as exc:  # noqa: BLE001 - per-revision failure accounting
            plan.failed.append({"local_id": local_id, "revision": revision, "error": str(exc)})
            continue
        if _same_segmentation(rows, blocks):
            plan.skipped += 1
            continue
        plan.items.append(
            RepackItem(
                local_id=local_id,
                revision=revision,
                blocks_before=len(rows),
                blocks_after=len(blocks),
            )
        )
    return plan


def apply_repack(db, plan: RepackPlan, *, policy: BlockPolicy | None = None) -> dict[str, Any]:
    """Re-segment each planned revision inside one transaction per revision.

    The returned block counts cover applied revisions only: a revision that
    fails keeps its old segmentation and is reported in ``failed``.
    """
    from drbrain.tree.contracts import LeafRef, NodeRecord, leaf_node_id

    policy = policy or BlockPolicy()
    applied = 0
    blocks_before = 0
    blocks_after = 0
    failed = list(plan.failed)
    for item in plan.items:
        local_id, revision = item.local_id, item.revision
        try:
            rows = db.get_content_blocks(local_id, revision)
            revision_row = db.get_document_revision(local_id, revision)
            if revision_row is None or not rows:
                raise RepackError("revision vanished before apply")
            blocks = _rebuild(local_id, revision_row, rows, policy)
            old_leaves = [
                leaf_node_id(
                    LeafRef(
                        local_id=local_id,
                        revision=revision,
                        block_id=str(row["block_id"]),
                        char_start=0,
                        char_end=len(str(row["text"])),
                    )
                )
                for row in rows
            ]
            with db.transaction():
                db.delete_tree_nodes(old_leaves)
                db.delete_content_blocks(local_id, revision)
                db.insert_content_blocks(blocks)
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
            applied += 1
            blocks_before += len(rows)
            blocks_after += len(blocks)
        except Exception as exc:  # noqa: BLE001 - per-revision failure accounting
            logger.warning("[repack] {}@{} failed: {}", local_id, revision, exc)
            failed.append({"local_id": local_id, "revision": revision, "error": str(exc)})
    return {
        "applied": applied,
        "skipped": plan.skipped,
        "failed": failed,
        "blocks_before": blocks_before,
        "blocks_after": blocks_after,
    }
