"""Fusion layer: RRF over LlamaIndex ``BaseRetriever`` legs + assembly helpers.

Ticket: T4 (检索器统一). Depends on T2 / T3.

Fusion decision (llama-index-core 0.14.23): ``QueryFusionRetriever`` *is*
importable, but its API no longer fits drbrain's requirements:

* its ``RECIPROCAL_RANK`` mode ignores ``retriever_weights`` (per-source
  weights only apply to ``relative_score``/``dist_based_score``), so the
  config's ``weighted`` fusion mode is impossible through it;
* it dedups by node *content hash* — nodes our custom retrievers construct on
  the fly never share hashes with the vector index's persisted nodes even for
  the same section, so the same paper section would appear twice;
* it generates extra LLM queries by default (``num_queries=4``) — a real LLM
  call per retrieval with no opt-out except ``num_queries=1``, which also
  cripples its relative-score modes; and
* it provides no per-source metadata annotation.

So :class:`FusionRetriever` implements the equivalent NodeWithScore RRF
fusioner the ticket allows (port of ``drbrain.query.fusion.reciprocal_rank_fusion``
semantics): rank-based scoring, ``node_id`` dedup, optional per-source
weights, and ``source``/``sources``/``contributions`` metadata annotation on
every fused node. Each leg is fault-tolerant — one failing retriever is
skipped, never aborting the query.

Assembly helpers:

* :func:`build_fusion_retriever` — wrap BM25 + vector (+ optional custom
  tree/graph) legs in one ``FusionRetriever``, honoring ``llamaindex.fusion_mode``.
* :func:`get_retrievers` — assemble the named leg dict per
  ``llamaindex.retrievers`` config list for T5's query engine.
"""

from __future__ import annotations

import logging
from typing import Any

from drbrain.config import Config
from drbrain.rag.config import get_llamaindex_config
from drbrain.rag.status import RetrievalError, RetrievalStatus, classify_failure

try:
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, QueryBundle

    _LLAMA_INDEX_AVAILABLE = True
except ImportError:  # pragma: no cover - envs without llama-index
    BaseRetriever = None  # type: ignore[assignment,misc]
    NodeWithScore = None  # type: ignore[assignment,misc]
    QueryBundle = None  # type: ignore[assignment,misc]
    _LLAMA_INDEX_AVAILABLE = False

log = logging.getLogger(__name__)

__all__ = [
    "_LLAMA_INDEX_AVAILABLE",
    "FUSION_MODES",
    "FusionRetriever",
    "build_fusion_retriever",
    "get_retrievers",
    "with_acl",
]

#: RRF damping constant (Cormack et al. 2009), same as ``query/fusion.py``.
DEFAULT_K = 60

#: Supported fusion modes: canonical RRF, and RRF with per-source weights.
FUSION_MODES = ("reciprocal_rank", "weighted")


def _rrf_fuse(
    ranked_lists: list[tuple[str, list[NodeWithScore]]],
    k: float = DEFAULT_K,
    weights: dict[str, float] | None = None,
    top_k: int | None = None,
) -> list[NodeWithScore]:
    """Reciprocal Rank Fusion over ``(source, NodeWithScore)`` lists.

    Each source list is sorted by descending score; a node's rank is its
    position in that list. ``rrf(node) = Σ_sources weight[source] / (k + rank)``.
    Nodes are deduplicated by ``node_id`` (content-hash fallback). Every fused
    node's metadata is annotated on a shallow copy (never mutating the source
    node) with ``source`` (primary/joined label), ``sources`` (contributing
    legs) and ``contributions`` ({source: {rank, score, weight}}).
    """
    acc: dict[str, dict[str, Any]] = {}
    for source, nodes in ranked_lists:
        ordered = sorted(
            nodes, key=lambda nws: nws.score if nws.score is not None else 0.0, reverse=True
        )
        for rank, nws in enumerate(ordered, start=1):
            node = nws.node
            nid = node.node_id or node.hash
            if not nid:
                continue
            weight = (weights or {}).get(source, 1.0)
            entry = acc.setdefault(
                nid,
                {
                    "rrf": 0.0,
                    "node": node,
                    "score": nws.score,
                    "sources": [],
                    "contributions": {},
                },
            )
            entry["rrf"] += weight / (k + rank)
            if source not in entry["sources"]:
                entry["sources"].append(source)
            entry["contributions"][source] = {
                "rank": rank,
                "score": nws.score,
                "weight": weight,
            }
            # Keep the best-scoring node object for a duplicated node_id.
            if nws.score is not None and (entry["score"] is None or nws.score > entry["score"]):
                entry["node"] = node
                entry["score"] = nws.score

    fused: list[NodeWithScore] = []
    for entry in acc.values():
        fused.append(
            NodeWithScore(
                node=_annotate_node(entry["node"], entry["sources"], entry["contributions"]),
                score=entry["rrf"],
            )
        )
    fused.sort(key=lambda nws: nws.score if nws.score is not None else 0.0, reverse=True)
    return fused if top_k is None else fused[:top_k]


def _annotate_node(node, sources: list[str], contributions: dict) -> Any:
    """Return a metadata-annotated shallow copy; never mutates the original."""
    if not sources:
        return node
    metadata = dict(node.metadata)
    metadata["source"] = sources[0] if len(sources) == 1 else ",".join(sources)
    metadata["sources"] = list(sources)
    metadata["contributions"] = contributions
    return node.model_copy(update={"metadata": metadata})


def with_acl(nodes: list[Any], acl_filter: dict[str, str] | None) -> list[Any]:
    """Post-filter fused nodes by an ACL context (default-deny).

    ACL enforcement belongs in the retrieval layer, never in the LLM prompt —
    "别泄密" is not a security boundary. Each ``(key, value)`` in
    ``acl_filter`` must be satisfied by the node's metadata:

    * a concrete value must equal ``metadata[key]`` exactly;
    * ``"*"`` is an explicit wildcard: any value for that key passes, but the
      key must still be present;
    * a node *missing* the key is excluded — default-deny is safer than
      assuming an unclassified node is public.

    A falsy/empty ``acl_filter`` is a passthrough (no filtering).
    """
    if not acl_filter:
        return list(nodes or [])
    kept: list[Any] = []
    for nws in nodes or []:
        node = getattr(nws, "node", None)
        meta = dict(getattr(node, "metadata", None) or {}) if node is not None else {}
        if _acl_allowed(meta, acl_filter):
            kept.append(nws)
    return kept


def _acl_allowed(meta: dict[str, Any], acl_filter: dict[str, str]) -> bool:
    """True when ``meta`` satisfies every ACL constraint (default-deny)."""
    for key, value in acl_filter.items():
        if key not in meta:
            return False
        if value == "*":
            continue
        if meta.get(key) != value:
            return False
    return True


if _LLAMA_INDEX_AVAILABLE:

    class FusionRetriever(BaseRetriever):
        """RRF fusion over ``BaseRetriever`` legs (NodeWithScore edition).

        Args:
            retrievers: The leg retrievers to fuse (BM25, vector, tree, graph…).
            sources: One source label per retriever (``bm25``/``vector``/
                ``tree``/``graph``…). Defaults to ``leg0``, ``leg1``, …
            mode: ``"reciprocal_rank"`` (canonical RRF, k=60) or
                ``"weighted"`` (RRF with per-source ``weights``).
            weights: Per-source multipliers for ``mode="weighted"``.
            top_k: Number of fused results to return.
            k: RRF damping constant.
            acl_filter: Optional ``{key: value}`` context (e.g.
                ``{"tenant_id": "company_A"}``) enforced as a post-filter on
                every fused node. ``"*"`` matches any value for a key; a node
                missing the key is excluded (default-deny). ``None``/empty
                disables ACL filtering. See :func:`with_acl`.
        """

        def __init__(
            self,
            retrievers: list[BaseRetriever],
            sources: list[str] | None = None,
            mode: str = "reciprocal_rank",
            weights: dict[str, float] | None = None,
            top_k: int = 10,
            k: float = DEFAULT_K,
            acl_filter: dict[str, str] | None = None,
        ) -> None:
            super().__init__()  # sets callback_manager/object_map (IndexNode resolution)
            self._retrievers = list(retrievers)
            if sources is None:
                sources = [f"leg{i}" for i in range(len(self._retrievers))]
            self._sources = list(sources)
            if len(self._retrievers) != len(self._sources):
                raise ValueError("retrievers and sources must have equal length")
            if mode not in FUSION_MODES:
                raise ValueError(f"unknown fusion mode {mode!r}; expected one of {FUSION_MODES}")
            self.mode = mode
            self.weights = dict(weights) if weights else None
            self.top_k = int(top_k)
            self.k = float(k)
            self._acl_filter = dict(acl_filter) if acl_filter else None

        def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
            ranked: list[tuple[str, list[NodeWithScore]]] = []
            failures: list[tuple[str, RetrievalStatus]] = []
            for retriever, source in zip(self._retrievers, self._sources):
                try:
                    nodes = retriever.retrieve(query_bundle)
                except RetrievalError:
                    # A nested fusion already declared a total outage; propagate
                    # it rather than degrading it into a per-leg empty result.
                    raise
                except Exception as exc:
                    status = classify_failure(exc)
                    failures.append((source, status))
                    log.warning(
                        "[rag] fusion leg %r failed (%s: %s); skipping",
                        source,
                        status.value,
                        exc,
                    )
                    nodes = []
                ranked.append((source, list(nodes or [])))
            if self._retrievers and len(failures) == len(self._retrievers):
                raise RetrievalError(
                    "all retrieval legs failed; refusing to synthesize",
                    failures=failures,
                )
            fused = _rrf_fuse(
                ranked,
                k=self.k,
                weights=self.weights if self.mode == "weighted" else None,
                top_k=self.top_k,
            )
            return with_acl(fused, self._acl_filter)


def _iter_custom_retrievers(custom):
    """Normalize ``custom_retrievers`` to ``[(source, retriever), ...]``."""
    if not custom:
        return []
    if isinstance(custom, dict):
        return list(custom.items())
    return list(custom)  # already a list of (source, retriever) tuples


def build_fusion_retriever(
    cfg: Config,
    vector_index=None,
    bm25_retriever=None,
    custom_retrievers=None,
    top_k: int | None = None,
    weights: dict[str, float] | None = None,
    acl_filter: dict[str, str] | None = None,
):
    """Assemble the unified fusion retriever over BM25 + vector + custom legs.

    Args:
        cfg: Config (``llamaindex.fusion_mode`` + ``embed.top_k``).
        vector_index: A ``VectorStoreIndex`` (converted via ``as_retriever``)
            or an already-built vector ``BaseRetriever``; ``None`` disables.
        bm25_retriever: A ``BM25Retriever`` or ``None``.
        custom_retrievers: ``{source: BaseRetriever}`` (e.g. tree/graph from
            :func:`get_retrievers`) or a list of ``(source, retriever)``.
        top_k: Fusion top-k (defaults to ``embed.top_k``).
        weights: Per-source weights for ``fusion_mode="weighted"``.
        acl_filter: Optional ``{key: value}`` context enforced on every fused
            node (see :class:`FusionRetriever`).

    Returns a :class:`FusionRetriever`, or ``None`` when every leg is missing.
    """
    if not _LLAMA_INDEX_AVAILABLE:
        raise RuntimeError("llama-index is not installed; cannot build fusion retriever")
    li = get_llamaindex_config(cfg)
    top_k = int(top_k or cfg.embed.top_k or 10)

    legs: list[tuple[str, Any]] = []
    if bm25_retriever is not None:
        legs.append(("bm25", bm25_retriever))
    if vector_index is not None:
        if hasattr(vector_index, "as_retriever"):
            vector_index = vector_index.as_retriever(similarity_top_k=top_k)
        legs.append(("vector", vector_index))
    legs.extend(_iter_custom_retrievers(custom_retrievers))

    if not legs:
        return None

    mode = (li.fusion_mode or "reciprocal_rank").strip().lower()
    if mode not in FUSION_MODES:
        log.warning("[rag] unknown fusion_mode %r; falling back to reciprocal_rank", mode)
        mode = "reciprocal_rank"
    return FusionRetriever(
        retrievers=[r for _, r in legs],
        sources=[s for s, _ in legs],
        mode=mode,
        weights=weights,
        top_k=top_k,
        acl_filter=acl_filter,
    )


def get_retrievers(cfg: Config, db=None, graph=None) -> dict[str, Any]:
    """Assemble named retrievers per the ``llamaindex.retrievers`` config list.

    Legs: ``bm25`` (persisted ``BM25Retriever``), ``vector``
    (``index.as_retriever``), ``tree`` (:class:`DrbrainTreeRetriever`),
    ``raptor`` (:class:`DrbrainRAPTORRetriever`), ``graph``
    (:class:`DrbrainGraphRetriever`). Only legs present in the
    config list AND available on disk are returned. T5's query engine combines
    the result with :func:`build_fusion_retriever` for fused retrieval.
    """
    if not _LLAMA_INDEX_AVAILABLE:
        raise RuntimeError("llama-index is not installed; cannot build retrievers")
    from drbrain.rag.indexer import load_index
    from drbrain.rag.retrievers import (
        DrbrainGraphRetriever,
        DrbrainRAPTORRetriever,
        DrbrainTreeRetriever,
    )

    li = get_llamaindex_config(cfg)
    wanted = [str(x).strip() for x in (li.retrievers or ["bm25", "vector"])]
    top_k = int(cfg.embed.top_k or 10)

    out: dict[str, Any] = {}
    if "bm25" in wanted or "vector" in wanted:
        index, bm25 = load_index(cfg)
        if "bm25" in wanted and bm25 is not None:
            out["bm25"] = bm25
        if "vector" in wanted and index is not None:
            out["vector"] = index.as_retriever(similarity_top_k=top_k)

    if "tree" in wanted:
        out["tree"] = DrbrainTreeRetriever(cfg, top_k=top_k, db_path=getattr(db, "path", None))
    if "graph" in wanted:
        out["graph"] = DrbrainGraphRetriever(db=db, graph=graph, top_k=top_k)
    if "raptor" in wanted:
        out["raptor"] = DrbrainRAPTORRetriever(cfg, top_k=top_k, db_path=getattr(db, "path", None))
    return out
