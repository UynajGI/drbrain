"""Evidence ranking and validation for the tree leg (plan T41).

Candidates, reads and evidence are three different things and must stay
distinguishable:

* a *candidate* is a ranked node id from the search entry;
* a *read* is an exact span plus its :class:`ReadReceipt`;
* *evidence* is a read that the caller can cite — a leaf span with a receipt,
  or an explicitly-marked summary hint that must be expanded before it can be
  quoted as source text.

Nothing unread may masquerade as evidence, the same canonical span counts
once no matter how many paths reached it (the origins are recorded, not
double-counted), and every entry keeps its origin range so downstream
citation stays exact.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from drbrain.tree.contracts import ReadReceipt


class EvidenceError(RuntimeError):
    """Evidence was assembled from something that was never actually read."""


@dataclass(frozen=True)
class TreeEvidence:
    node_id: str
    node_revision: int
    kind: str
    local_id: str
    block_id: str
    char_start: int
    char_end: int
    content_hash: str
    tokens: int
    source: str  # "leaf" | "summary"
    score: float = 0.0
    query: str = ""
    via: tuple[str, ...] = ()
    text: str = ""

    def __post_init__(self) -> None:
        if self.source not in {"leaf", "summary"}:
            raise ValueError(f"unsupported evidence source {self.source!r}")
        if self.source == "leaf":
            if not self.block_id or self.char_end <= self.char_start:
                raise EvidenceError("leaf evidence requires a concrete span")
        elif self.char_end <= self.char_start:
            raise ValueError("evidence span must be non-empty")

    @property
    def span_key(self) -> tuple[str, str, int, int]:
        return (self.local_id, self.block_id, self.char_start, self.char_end)

    def to_json(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_revision": self.node_revision,
            "kind": self.kind,
            "local_id": self.local_id,
            "block_id": self.block_id,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "content_hash": self.content_hash,
            "tokens": self.tokens,
            "source": self.source,
            "score": round(float(self.score), 6),
            "via": list(self.via),
        }


def evidence_from_navigation(result, *, query: str = "") -> list[TreeEvidence]:
    """Turn a navigator result into evidence, refusing unbacked leaves.

    Items produced by the navigator carry a receipt for leaf reads; an item
    claiming to be leaf evidence without one is a bug, not a citation.
    """
    receipts: dict[tuple[str, int, int], ReadReceipt] = {}
    for receipt in getattr(result, "receipts", []):
        receipts[(receipt.block_id, receipt.char_start, receipt.char_end)] = receipt
    out: list[TreeEvidence] = []
    for item in getattr(result, "evidence", []):
        source = str(item.get("source") or "leaf")
        receipt_data = item.get("receipt") or {}
        block_id = str(receipt_data.get("block_id") or "")
        char_start = int(receipt_data.get("char_start") or 0)
        char_end = int(receipt_data.get("char_end") or 0)
        if source == "leaf":
            receipt = receipts.get((block_id, char_start, char_end))
            if receipt is None:
                raise EvidenceError(
                    f"leaf evidence for {item.get('node_id')!r} has no read receipt"
                )
            content_hash = receipt.content_hash
            tokens = receipt.tokens
        else:
            content_hash = str(receipt_data.get("content_hash") or item.get("content_hash") or "")
            tokens = int(receipt_data.get("tokens") or 0)
        via = tuple(value for value in (str(item.get("via") or ""),) if value)
        out.append(
            TreeEvidence(
                node_id=str(item.get("node_id")),
                node_revision=int(item.get("node_revision") or 0),
                kind=str(item.get("kind") or "leaf"),
                local_id=str(item.get("local_id") or ""),
                block_id=block_id or f"summary:{item.get('node_id')}",
                char_start=char_start,
                char_end=max(char_end, char_start + 1),
                content_hash=content_hash,
                tokens=tokens,
                source=source,
                score=float(item.get("score") or 0.0),
                query=query or getattr(result, "query", ""),
                via=via,
                text=str(item.get("text") or ""),
            )
        )
    return out


def rank_evidence(
    items: Iterable[TreeEvidence], *, collapse_spans: bool = True
) -> list[TreeEvidence]:
    """Order evidence by score; identical canonical spans count once."""
    if not collapse_spans:
        return sorted(items, key=lambda item: (-item.score, item.node_id))
    best: dict[tuple[str, str, int, int], TreeEvidence] = {}
    for item in items:
        key = item.span_key
        existing = best.get(key)
        if existing is None:
            best[key] = item
            continue
        # Keep every node id that reached this span (origins are provenance,
        # not extra evidence) while the strongest reading wins the payload.
        via = tuple(dict.fromkeys([*existing.via, existing.node_id, *item.via, item.node_id]))
        winner = item if item.score > existing.score else existing
        best[key] = TreeEvidence(**{**winner.__dict__, "via": via})
    return sorted(best.values(), key=lambda item: (-item.score, item.node_id))


def validate_evidence(items: Sequence[TreeEvidence]) -> dict[str, Any]:
    """Audit an evidence set: coverage, duplicates, unbacked summaries."""
    spans = {item.span_key for item in items}
    summaries = [item for item in items if item.source == "summary"]
    duplicates = len(items) - len(spans)
    return {
        "items": len(items),
        "unique_spans": len(spans),
        "duplicate_spans": duplicates,
        "summary_hints": len(summaries),
        "leaf_items": len(items) - len(summaries),
        "unbacked_summaries": [item.node_id for item in summaries if not item.via],
    }


def quoted_text(items: Sequence[TreeEvidence], *, allow_summaries: bool = False) -> list[str]:
    """Texts safe to quote as source material (summaries are opt-in)."""
    if allow_summaries:
        return [item.text for item in items if item.text]
    return [item.text for item in items if item.source == "leaf" and item.text]
