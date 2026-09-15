"""Tree tools over published nodes: expand / read / parents / read_scope (T38).

Every tool answers from the unified store and only from *published* nodes:
``staging``/``failed``/``stale`` nodes are invisible, a read returns the exact
canonical text with a :class:`ReadReceipt` (so evidence can be tied back to a
byte range), and budgets are explicit — a truncated answer is reported as
truncated, never silently shortened.  A tool call decides *what to read*, it
never rewrites the body or produces a second summary layer.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from drbrain.tree.contracts import ReadReceipt


class ToolError(RuntimeError):
    """A tool was called with an unusable argument or budget."""


@dataclass
class ToolBudget:
    max_nodes: int = 200
    max_tokens: int = 50_000
    max_calls: int = 500

    def __post_init__(self) -> None:
        for value in (self.max_nodes, self.max_tokens, self.max_calls):
            if value <= 0:
                raise ValueError("tool budgets must be positive")


@dataclass
class ToolState:
    """Per-request bookkeeping the caller can audit."""

    calls: int = 0
    nodes_read: int = 0
    tokens_read: int = 0
    truncated: bool = False
    started_at: float = field(default_factory=time.monotonic)

    def to_json(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "nodes_read": self.nodes_read,
            "tokens_read": self.tokens_read,
            "truncated": self.truncated,
            "elapsed_ms": round((time.monotonic() - self.started_at) * 1000, 3),
        }


def _default_count(text: str) -> int:
    from drbrain.services.tokens import count_tokens

    return int(count_tokens(text))


class TreeTools:
    """Stateless tool surface; state (budgets/receipts) lives in the caller."""

    def __init__(self, db, *, budget: ToolBudget | None = None, count_tokens=None) -> None:
        self.db = db
        self.budget = budget or ToolBudget()
        self.count_tokens = count_tokens or _default_count

    # ── visibility ───────────────────────────────────────────────
    def visible(self, node_id: str) -> bool:
        row = self.db.get_tree_node(node_id)
        return bool(row) and str(row["state"]) == "ready"

    def _require_visible(self, node_id: str) -> dict:
        row = self.db.get_tree_node(node_id)
        if row is None:
            raise ToolError(f"unknown node {node_id!r}")
        if str(row["state"]) != "ready":
            raise ToolError(
                f"node {node_id!r} is {row['state']!r}; only published nodes are readable"
            )
        return row

    def _charge(self, state: ToolState, tokens: int) -> None:
        state.calls += 1
        if state.calls > self.budget.max_calls:
            state.truncated = True
            raise ToolError("tool call budget exhausted")
        if state.tokens_read + tokens > self.budget.max_tokens:
            state.truncated = True
            raise ToolError("token budget exhausted")
        state.tokens_read += tokens

    # ── tools ────────────────────────────────────────────────────
    def node_text(self, node_id: str) -> str:
        """Exact canonical text of one node (leaf body or region summary)."""
        row = self._require_visible(node_id)
        if row["kind"] == "region":
            return str(row["summary"])
        blocks = self.db.get_content_blocks(row["local_id"], int(row["doc_revision"]))
        block = next((item for item in blocks if item["block_id"] == row["block_id"]), None)
        if block is None:
            raise ToolError(f"leaf {node_id!r} references a missing block")
        text = str(block["text"])
        char_start = int(row["char_start"] or 0)
        char_end = int(row["char_end"] or len(text))
        return text[char_start:char_end]

    def expand(self, node_id: str, state: ToolState) -> list[dict[str, Any]]:
        """Direct members of a published region (leaf children show no body)."""
        self._require_visible(node_id)
        self._charge(state, 0)
        children = self.db.get_tree_children(node_id)
        out: list[dict[str, Any]] = []
        for child in children[: self.budget.max_nodes]:
            row = self._require_visible(child["child_id"])
            out.append(
                {
                    "node_id": row["node_id"],
                    "kind": row["kind"],
                    "title": row["title"],
                    "layer": int(row["layer"]),
                    "revision": int(row["revision"]),
                    "local_id": row["local_id"],
                    "summary": row["summary"] if row["kind"] == "region" else "",
                    "weight": child["weight"],
                    "origin": child["origin"],
                }
            )
        if len(children) > self.budget.max_nodes:
            state.truncated = True
        return out

    def read(
        self,
        node_id: str,
        state: ToolState,
        *,
        char_start: int | None = None,
        char_end: int | None = None,
        request_id: str = "",
        tool: str = "read",
    ) -> tuple[str, ReadReceipt]:
        """Read a leaf range (or a region summary) with an exact receipt.

        A sub-range is only allowed inside one canonical block; callers that
        need cross-block text use :meth:`read_scope`, which returns several
        receipts rather than a synthesized one.
        """
        row = self._require_visible(node_id)
        if row["kind"] == "region":
            text = str(row["summary"])
            if char_start is not None or char_end is not None:
                raise ToolError("region summaries are read whole")
            tokens = self.count_tokens(text)
            self._charge(state, tokens)
            state.nodes_read += 1
            receipt = ReadReceipt(
                request_id=request_id or f"r{state.calls}",
                tool="read",
                node_id=node_id,
                node_revision=int(row["revision"]),
                local_id=str(row["local_id"] or ""),
                block_id=f"summary:{node_id}",
                char_start=0,
                char_end=max(1, len(text)),
                content_hash=str(row["content_hash"]),
                tokens=tokens,
            )
            return text, receipt
        blocks = self.db.get_content_blocks(row["local_id"], int(row["doc_revision"]))
        block = next((item for item in blocks if item["block_id"] == row["block_id"]), None)
        if block is None:
            raise ToolError(f"leaf {node_id!r} references a missing block")
        full = str(block["text"])
        node_start = int(row["char_start"] or 0)
        node_end = int(row["char_end"] or len(full))
        start = node_start if char_start is None else node_start + int(char_start)
        end = node_end if char_end is None else node_start + int(char_end)
        if start < node_start or end > node_end or end <= start:
            raise ToolError("requested range is outside the leaf")
        text = full[start:end]
        tokens = self.count_tokens(text)
        self._charge(state, tokens)
        state.nodes_read += 1
        receipt = ReadReceipt(
            request_id=request_id or f"r{state.calls}",
            tool="read" if tool == "read" else "read_scope",
            node_id=node_id,
            node_revision=int(row["revision"]),
            local_id=str(row["local_id"]),
            block_id=str(row["block_id"]),
            char_start=start,
            char_end=end,
            content_hash=str(row["content_hash"]),
            tokens=tokens,
        )
        return text, receipt

    def parents(self, node_id: str, state: ToolState) -> list[dict[str, Any]]:
        """Reverse membership query: every published parent of a node."""
        self._require_visible(node_id)
        self._charge(state, 0)
        out = []
        for parent in self.db.get_tree_parents(node_id):
            row = self.db.get_tree_node(parent["parent_id"])
            if row is None or str(row["state"]) != "ready":
                continue
            out.append(
                {
                    "node_id": row["node_id"],
                    "layer": int(row["layer"]),
                    "title": row["title"],
                    "weight": parent["weight"],
                }
            )
        return out

    def read_scope(
        self,
        node_id: str,
        state: ToolState,
        *,
        request_id: str = "",
        max_blocks: int | None = None,
    ) -> list[tuple[str, ReadReceipt]]:
        """Read every unique canonical span under a node (regions included)."""
        self._require_visible(node_id)
        spans = self._scope_spans(node_id)
        limit = max_blocks or self.budget.max_nodes
        if len(spans) > limit:
            state.truncated = True
            spans = spans[:limit]
        out: list[tuple[str, ReadReceipt]] = []
        for local_id, block_id, char_start, char_end in spans:
            text = self._block_slice(local_id, block_id, char_start, char_end)
            tokens = self.count_tokens(text)
            self._charge(state, tokens)
            state.nodes_read += 1
            out.append(
                (
                    text,
                    ReadReceipt(
                        request_id=request_id or f"scope{state.calls}",
                        tool="read_scope",
                        node_id=node_id,
                        node_revision=int(self.db.get_tree_node(node_id)["revision"]),
                        local_id=local_id,
                        block_id=block_id,
                        char_start=char_start,
                        char_end=char_end,
                        content_hash=self._block_hash(local_id, block_id),
                        tokens=tokens,
                    ),
                )
            )
        return out

    # ── internals ────────────────────────────────────────────────
    def _block_slice(self, local_id: str, block_id: str, char_start: int, char_end: int) -> str:
        block = self._block(local_id, block_id)
        text = str(block["text"])
        return text[char_start:char_end]

    def _block_hash(self, local_id: str, block_id: str) -> str:
        return str(self._block(local_id, block_id)["text_hash"])

    def _block(self, local_id: str, block_id: str) -> dict:
        for block in self.db.get_content_blocks(local_id):
            if block["block_id"] == block_id:
                return block
        raise ToolError(f"missing block {block_id!r}")

    def _scope_spans(self, node_id: str) -> list[tuple[str, str, int, int]]:
        """Unique (local_id, block, range) tuples under a node, collapsed by origin."""
        row = self._require_visible(node_id)
        if row["kind"] == "leaf":
            local_id = str(row["local_id"])
            block_id = str(row["block_id"])
            block = self._block(local_id, block_id)
            return [
                (
                    local_id,
                    block_id,
                    int(row["char_start"] or 0),
                    int(row["char_end"] or len(str(block["text"]))),
                )
            ]
        seen: dict[tuple[str, str, int, int], None] = {}
        for child in self.db.get_tree_children(node_id):
            for span in self._scope_spans(child["child_id"]):
                seen.setdefault(span, None)
        return list(seen)


def receipts_answer(receipts: Iterable[ReadReceipt], spans: Sequence[tuple[str, int, int]]) -> bool:
    """Whether the receipts together actually cover the requested spans."""
    covered = {(receipt.block_id, receipt.char_start, receipt.char_end) for receipt in receipts}
    return all((block_id, start, end) in covered for block_id, start, end in spans)


def _json_dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
