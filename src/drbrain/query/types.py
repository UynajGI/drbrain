"""Unified retrieval result type for hybrid search.

``SearchHit`` is the common currency across BM25, embedding, RRF fusion, and
the rerank layer. Each retriever normalizes its native output into
``SearchHit`` (paper-level), fusion merges multiple hit lists, and the
reranker reorders them — all without touching the callers' existing formats.

Design note: fusion happens at the paper granularity. BM25 documents share
``local_id == paper_id`` (see ``build_bm25_index``), and embedding rows carry
``paper_id == paper_dir.name`` (see ``build_tree_vectors``), so both sides
align on ``paper_id``. Section-level detail survives as ``payload``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchHit:
    """A single paper-level retrieval result.

    Attributes:
        paper_id: Paper identifier shared by both BM25 (local_id) and
            embedding (tree_vectors.paper_id) sides.
        score: Original retriever score on input; RRF score on fusion output.
        rank: 1-based rank within the source list; rewritten after fusion.
        source: Provenance tag — ``"bm25"``, ``"embedding"``, or ``"fused"``.
            For fused hits, the originating sources are in ``metadata["sources"]``.
        payload: The originating retriever's raw row (BM25 dict or embedding
            dict). For fused hits, a ``{source: row}`` mapping.
        metadata: Extra debug info. Fusion records per-source contributions
            here as ``{"sources": [...], "contributions": {src: {rank, score}}}``.
    """

    paper_id: str
    score: float
    rank: int = 0
    source: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializable view for CLI / JSON output."""
        return {
            "paper_id": self.paper_id,
            "score": round(self.score, 6),
            "rank": self.rank,
            "source": self.source,
            "payload": self.payload,
            "metadata": self.metadata,
        }


__all__ = ["SearchHit"]
