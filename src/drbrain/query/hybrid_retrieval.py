"""Hybrid retrieval entry point: BM25 + embedding, fused via RRF.

Orchestrates the three query modules without modifying them:

    BM25 ─┐
          ├─ → normalize to paper level ─→ RRF fusion ─→ optional rerank ─→ top_k
    Embed ─┘
                  (skipped when embed_cfg is None or provider="none")

Normalization is necessary because the two retrievers operate at different
granularities: BM25 indexes papers/concepts/arguments keyed by ``local_id``
(== paper_id), while embedding rows are keyed by ``node_id`` (section level)
with a ``paper_id`` column. We collapse embedding hits to the paper level by
keeping each paper's best-scoring node, preserving the node_id in the payload
so downstream callers (PageIndex-style) can still fetch section content.

Fault tolerance: BM25 and embedding are run independently and each wrapped in
try/except — one failing logs a warning and is skipped, never aborting the
whole query. Pure-BM25 mode (``embed_cfg=None``) runs when vectors are off.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from drbrain.query.fusion import reciprocal_rank_fusion
from drbrain.query.rerank import get_reranker
from drbrain.query.types import SearchHit

if TYPE_CHECKING:
    from drbrain.config import EmbedConfig
    from drbrain.storage.database import Database

log = logging.getLogger(__name__)


def hybrid_search(
    query: str,
    db: Database,
    db_path: Path,
    embed_cfg: EmbedConfig | None = None,
    *,
    top_k: int = 10,
    bm25_limit: int = 50,
    embed_limit: int = 50,
    rerank: bool = False,
    rerank_model: str | None = None,
    rrf_k: int = 60,
    rrf_weights: dict[str, float] | None = None,
) -> list[SearchHit]:
    """Run hybrid BM25 + embedding retrieval with RRF fusion.

    Args:
        query: Natural language query.
        db: Database for BM25 index construction.
        db_path: SQLite path for embedding vector search.
        embed_cfg: Embedding config. ``None`` or ``provider="none"`` disables
            the embedding leg (pure BM25 mode).
        top_k: Final number of hits to return.
        bm25_limit: Candidate cap fetched from BM25 before fusion.
        embed_limit: Candidate cap fetched from embedding before fusion.
        rerank: If True, rerank the fused top-N with a cross-encoder (auto
            no-op if unavailable).
        rerank_model: Cross-encoder model id; ``None`` uses the rerank default.
        rrf_k: RRF damping constant.
        rrf_weights: Optional per-source weights (e.g.
            ``{"bm25": 0.4, "embedding": 0.6}``). ``None`` = equal weight.

    Returns:
        Ranked ``SearchHit`` list (length <= ``top_k``), each with a stable
        ``paper_id``/``score``/``rank``/``source``. Empty list on no data.
    """
    bm25_hits = _run_bm25(query, db, bm25_limit)
    embed_hits = _run_embedding(query, db_path, embed_cfg, embed_limit)

    if not bm25_hits and not embed_hits:
        return []

    fused = reciprocal_rank_fusion([bm25_hits, embed_hits], k=rrf_k, weights=rrf_weights)

    if rerank:
        reranker = get_reranker("auto", rerank_model)
        # Rerank a wider window than top_k so good-but-lower-ranked hits
        # survive; the reranker truncates internally.
        reranker.rerank(query, fused, top_n=top_k)
    else:
        fused = fused[:top_k]

    # Defensive: ensure final length and rank integrity.
    fused = fused[:top_k]
    for i, hit in enumerate(fused, start=1):
        hit.rank = i
    return fused


# ── Retriever runners (each independent and fault-tolerant) ──────────────────


def _run_bm25(query: str, db: Database, limit: int) -> list[SearchHit]:
    """Build a fresh BM25 index and return paper-level hits."""
    try:
        from drbrain.query.bm25 import build_bm25_index

        index = build_bm25_index(db)
        rows = index.search(query, limit=limit)
    except Exception as e:  # index build/search failure
        log.warning("[hybrid] BM25 leg failed (%s); skipping", e)
        return []
    return _bm25_to_hits(rows)


def _run_embedding(
    query: str, db_path: Path, cfg: EmbedConfig | None, limit: int
) -> list[SearchHit]:
    """Run vector search and return paper-level hits (or [] if disabled)."""
    if cfg is None:
        return []
    try:
        from drbrain.services.embedding import _embed_provider, search_tree

        if _embed_provider(cfg) == "none":
            return []
        rows = search_tree(query, db_path, top_k=limit, cfg=cfg)
    except Exception as e:  # model load / DB read failure
        log.warning("[hybrid] embedding leg failed (%s); skipping", e)
        return []
    return _embedding_to_hits(rows)


# ── Paper-level normalizers ──────────────────────────────────────────────────


def _bm25_to_hits(rows: list[dict[str, Any]]) -> list[SearchHit]:
    """Collapse BM25 rows to paper level.

    BM25 may return multiple rows for one paper (its title + its concepts +
    its arguments all share ``local_id``). We keep the highest-scoring row
    per paper and derive rank from the post-collapse sort.
    """
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        pid = row.get("local_id") or row.get("id")
        if not pid:
            continue
        score = float(row.get("score", 0.0))
        if pid not in best or score > float(best[pid].get("score", 0.0)):
            best[pid] = row

    hits: list[SearchHit] = [
        SearchHit(
            paper_id=pid,
            score=float(row.get("score", 0.0)),
            source="bm25",
            payload=dict(row),
        )
        for pid, row in best.items()
    ]
    hits.sort(key=lambda h: h.score, reverse=True)
    for i, hit in enumerate(hits, start=1):
        hit.rank = i
    return hits


def _embedding_to_hits(rows: list[dict[str, Any]]) -> list[SearchHit]:
    """Collapse embedding rows to paper level, keeping the best node.

    Each paper may have many embedded sections; we keep the best-scoring one
    and record its ``node_id``/``tree_layer`` in the payload so callers can
    still fetch section content downstream.
    """
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        pid = row.get("paper_id")
        if not pid:
            continue
        score = float(row.get("score", 0.0))
        if pid not in best or score > float(best[pid].get("score", 0.0)):
            best[pid] = row

    hits: list[SearchHit] = [
        SearchHit(
            paper_id=pid,
            score=float(row.get("score", 0.0)),
            source="embedding",
            payload=dict(row),
        )
        for pid, row in best.items()
    ]
    hits.sort(key=lambda h: h.score, reverse=True)
    for i, hit in enumerate(hits, start=1):
        hit.rank = i
    return hits


__all__ = ["hybrid_search"]
