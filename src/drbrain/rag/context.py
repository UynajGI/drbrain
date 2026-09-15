"""Final context selection with a real token budget (plan T44).

The query pipeline is: RRF fuses the three legs **by rank only** (score
spaces never mix), BGE reranks a bounded head (20–50), and this module is the
last step — it dedups by provenance, spreads the selection across papers, and
returns exactly the documents whose cumulative estimated size fits the
configured token budget.  The returned count and ``tokens_used`` are the same
fact: nothing is returned that would push the context over budget, and a
single oversized head document is truncated to the budget instead of leaving
the answer with no context at all.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

CHARS_PER_TOKEN = 4
DEFAULT_MAX_DOCS = 10
DEFAULT_TOKEN_BUDGET = 8000


def estimate_tokens(text: str) -> int:
    """Conservative character-based token estimate (≈4 chars per token)."""
    return max(1, math.ceil(len(str(text or "")) / CHARS_PER_TOKEN))


@dataclass(frozen=True)
class ContextItem:
    node_id: str
    paper_id: str
    text: str
    tokens: int
    score: float
    truncated: bool = False


@dataclass(frozen=True)
class ContextSelection:
    items: tuple[ContextItem, ...]
    tokens_used: int
    token_budget: int
    dropped: int = 0
    deduplicated: int = 0

    @property
    def truncated(self) -> bool:
        return any(item.truncated for item in self.items)

    def to_json(self) -> dict[str, Any]:
        return {
            "count": len(self.items),
            "tokens_used": self.tokens_used,
            "token_budget": self.token_budget,
            "dropped": self.dropped,
            "deduplicated": self.deduplicated,
            "items": [
                {
                    "node_id": item.node_id,
                    "paper_id": item.paper_id,
                    "tokens": item.tokens,
                    "score": item.score,
                    "truncated": item.truncated,
                }
                for item in self.items
            ],
        }


@dataclass(frozen=True)
class _Candidate:
    node_id: str
    paper_id: str
    text: str
    score: float
    origin: Any = field(default=None, compare=False)


def _as_candidate(raw: Any) -> _Candidate | None:
    if isinstance(raw, dict):
        text = str(raw.get("text") or raw.get("excerpt") or "")
        node_id = str(raw.get("node_id") or raw.get("key") or "")
        paper_id = str(raw.get("paper_id") or "")
        score = float(raw.get("score") or 0.0)
    else:
        node = getattr(raw, "node", None)
        meta = dict(getattr(node, "metadata", None) or {}) if node is not None else {}
        text = str(getattr(node, "text", None) or meta.get("text") or "")
        node_id = str(meta.get("node_id") or getattr(node, "node_id", "") or "")
        paper_id = str(meta.get("paper_id") or "")
        score = float(getattr(raw, "score", 0.0) or 0.0)
    if not text:
        return None
    return _Candidate(node_id=node_id, paper_id=paper_id, text=text, score=score, origin=raw)


def _dedup(candidates: Iterable[_Candidate]) -> tuple[list[_Candidate], int]:
    unique: list[_Candidate] = []
    seen_nodes: set[tuple[str, str]] = set()
    seen_texts: set[tuple[str, str]] = set()
    duplicates = 0
    for cand in candidates:
        text_key = (cand.paper_id, hashlib.sha1(cand.text.encode("utf-8")).hexdigest()[:16])
        node_key = (
            cand.paper_id,
            cand.node_id or hashlib.sha1(cand.text.encode("utf-8")).hexdigest()[:16],
        )
        if node_key in seen_nodes or text_key in seen_texts:
            duplicates += 1
            continue
        seen_nodes.add(node_key)
        seen_texts.add(text_key)
        unique.append(cand)
    return unique, duplicates


def _select(
    candidates: Iterable[_Candidate],
    *,
    token_budget: int,
    max_docs: int,
    truncate_head: bool,
) -> tuple[list[tuple[_Candidate, int, bool]], int, int, int]:
    """Returns ``(selected, tokens_used, dropped, deduplicated)``."""
    budget = max(1, int(token_budget))
    limit = max(1, int(max_docs))
    unique, duplicates = _dedup(candidates)

    # Diversity: rotate over papers, keeping each paper's own order.
    rotation: list[str] = []
    buckets: dict[str, list[_Candidate]] = {}
    for cand in unique:
        paper = cand.paper_id
        if paper not in buckets:
            buckets[paper] = []
            rotation.append(paper)
        buckets[paper].append(cand)

    selected: list[tuple[_Candidate, int, bool]] = []
    used = 0
    cursors = dict.fromkeys(rotation, 0)
    while rotation and len(selected) < limit:
        for paper in list(rotation):
            if len(selected) >= limit:
                break
            bucket = buckets[paper]
            if cursors[paper] >= len(bucket):
                rotation.remove(paper)
                continue
            cand = bucket[cursors[paper]]
            cursors[paper] += 1
            tokens = estimate_tokens(cand.text)
            if used + tokens <= budget:
                selected.append((cand, tokens, False))
                used += tokens
    if not selected and unique and truncate_head:
        head = unique[0]
        allowed = max(1, budget * CHARS_PER_TOKEN)
        text = head.text[:allowed]
        tokens = estimate_tokens(text)
        selected.append(
            (_Candidate(head.node_id, head.paper_id, text, head.score, head.origin), tokens, True)
        )
        used = tokens
    dropped = len(unique) - len(selected)
    return selected, used, dropped, duplicates


def select_context(
    candidates: Iterable[Any],
    *,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    max_docs: int = DEFAULT_MAX_DOCS,
    truncate_head: bool = True,
) -> ContextSelection:
    """Dedup, diversify and fit candidates into the token budget."""
    normalized = [cand for cand in (_as_candidate(raw) for raw in candidates) if cand is not None]
    selected, used, dropped, duplicates = _select(
        normalized,
        token_budget=token_budget,
        max_docs=max_docs,
        truncate_head=truncate_head,
    )
    items = tuple(
        ContextItem(
            node_id=cand.node_id,
            paper_id=cand.paper_id,
            text=cand.text,
            tokens=tokens,
            score=cand.score,
            truncated=truncated,
        )
        for cand, tokens, truncated in selected
    )
    return ContextSelection(
        items=items,
        tokens_used=used,
        token_budget=int(token_budget),
        dropped=dropped,
        deduplicated=duplicates,
    )


try:  # pragma: no cover - exercised only without llama-index
    from llama_index.core.postprocessor.node import BaseNodePostprocessor
    from llama_index.core.schema import NodeWithScore
    from pydantic import PrivateAttr

    _LLAMA_INDEX_AVAILABLE = True
except ImportError:
    BaseNodePostprocessor = None  # type: ignore[assignment,misc]
    NodeWithScore = None  # type: ignore[assignment,misc]
    PrivateAttr = None  # type: ignore[assignment,misc]
    _LLAMA_INDEX_AVAILABLE = False


if _LLAMA_INDEX_AVAILABLE:

    class ContextBudgetPostprocessor(BaseNodePostprocessor):
        """Keep only the context that fits the token budget (T44, runs last).

        Dedups by provenance, spreads over papers and returns the nodes whose
        cumulative estimated tokens fit ``token_budget`` (at most
        ``max_docs``), so the node count matches the budget actually used.
        An oversized single head node is truncated to the budget and flagged.
        """

        max_docs: int = DEFAULT_MAX_DOCS
        token_budget: int = DEFAULT_TOKEN_BUDGET
        _last_trace: dict[str, Any] = PrivateAttr(default_factory=dict)

        def __init__(
            self,
            max_docs: int = DEFAULT_MAX_DOCS,
            token_budget: int = DEFAULT_TOKEN_BUDGET,
        ) -> None:
            super().__init__(  # type: ignore[call-arg]
                max_docs=int(max_docs), token_budget=int(token_budget)
            )

        @classmethod
        def class_name(cls) -> str:
            return "ContextBudgetPostprocessor"

        def get_last_trace(self) -> dict[str, Any]:
            return dict(self._last_trace)

        def _postprocess_nodes(self, nodes, query_bundle=None):
            candidates = list(nodes or [])
            normalized = [cand for cand in (_as_candidate(raw) for raw in candidates) if cand]
            selected, used, dropped, duplicates = _select(
                normalized,
                token_budget=self.token_budget,
                max_docs=self.max_docs,
                truncate_head=True,
            )
            out = []
            for cand, tokens, truncated in selected:
                nws = cand.origin
                metadata = {
                    **dict(getattr(nws.node, "metadata", None) or {}),
                    "context_tokens": tokens,
                    "context_truncated": truncated,
                }
                node = nws.node.model_copy(update={"text": cand.text, "metadata": metadata})
                out.append(NodeWithScore(node=node, score=nws.score))
            self._last_trace = {
                "stage": "context_budget",
                "status": "ok",
                "input_nodes": len(candidates),
                "selected": len(out),
                "tokens_used": used,
                "token_budget": self.token_budget,
                "max_docs": self.max_docs,
                "dropped": dropped,
                "deduplicated": duplicates,
            }
            return out


__all__ = [
    "CHARS_PER_TOKEN",
    "ContextBudgetPostprocessor",
    "ContextItem",
    "ContextSelection",
    "DEFAULT_MAX_DOCS",
    "DEFAULT_TOKEN_BUDGET",
    "estimate_tokens",
    "select_context",
]
