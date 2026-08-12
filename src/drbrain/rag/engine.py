"""Query engine layer: RetrieverQueryEngine assembly + ask compatibility (T5).

Ticket: T5 (查询引擎). Depends on T2 (LLM bridge) / T3 (index layer) / T4
(retrievers + fusion).

Assembly: :func:`build_query_engine` wires the T4 :class:`FusionRetriever` into
a :class:`~llama_index.core.query_engine.RetrieverQueryEngine` with a
``refine`` synthesizer and a fused-score-aware similarity cutoff
postprocessor. :func:`ask_llamaindex` is the legacy-``ask``-compatible entry
point returning ``{question, answer, sources, engine}``; with
``streaming=True`` it behaves as a generator that yields intermediate
``{"chunk": str}`` items before the final result dict.
:func:`build_hybrid_retriever` + :func:`nodes_to_paper_results` back the
``hybrid --engine llamaindex`` / ``query --engine llamaindex`` paper-level
and node-level rankings. :func:`resolve_engine` decides whether the CLI's
``--engine`` switch can actually use llamaindex (design §4.6 fallback rule).

API notes (llama-index-core 0.14.23, verified against the installed wheel):

* ``ResponseSynthesizer`` no longer exists as a class — use the
  :func:`get_response_synthesizer` factory with ``response_mode="refine"``.
  Its ``llm`` defaults to ``Settings.llm``, which
  :func:`~drbrain.rag.llm.init_llamaindex_settings` wires to
  :class:`~drbrain.rag.llm.DrbrainLLM` (so the drbrain fallback chain, ApiCache
  and metrics stay intact through every engine query).
* ``SimilarityPostProcessor`` is now ``SimilarityPostprocessor``, and the
  stock class drops every node whose ``score < cutoff`` plus every
  ``score=None`` node. Fused (RRF) scores are ~``1/(k + rank)`` — far below
  any meaningful similarity threshold — so the stock class would empty the
  result set of a fused engine. Hence :class:`SimilarityCutoffPostprocessor`,
  which applies the cutoff to the best *original per-leg* score (from the
  fusion ``contributions`` annotation) and keeps ``score=None`` nodes (the
  tree/graph legs produce positional/decayed scores, not similarities).
* Streaming: ``query()`` on a streaming synthesizer returns a
  ``StreamingResponse``; ``response_gen`` yields ``str`` chunks (or objects
  exposing ``.delta``/``.text``), and ``source_nodes`` is populated eagerly.
  The current ``DrbrainLLM`` streaming endpoints are single-chunk stubs, so
  "streaming" today surfaces the full answer in one chunk — the protocol is
  ready for real per-token streaming (T9).
"""

from __future__ import annotations

import logging
from typing import Any

from drbrain.config import Config
from drbrain.rag.config import get_llamaindex_config

try:
    from llama_index.core.base.response.schema import (
        Response as _LIXResponse,
    )
    from llama_index.core.base.response.schema import (
        StreamingResponse,
    )
    from llama_index.core.postprocessor.node import BaseNodePostprocessor
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.response_synthesizers import get_response_synthesizer
    from llama_index.core.schema import NodeWithScore

    _LLAMA_INDEX_AVAILABLE = True
except ImportError:  # pragma: no cover - envs without llama-index
    BaseNodePostprocessor = None  # type: ignore[assignment,misc]
    NodeWithScore = None  # type: ignore[assignment,misc]
    RetrieverQueryEngine = None  # type: ignore[assignment,misc]
    StreamingResponse = None  # type: ignore[assignment,misc]
    get_response_synthesizer = None  # type: ignore[assignment,misc]
    _LIXResponse = None  # type: ignore[assignment,misc]
    _LLAMA_INDEX_AVAILABLE = False

log = logging.getLogger(__name__)

__all__ = [
    "_LLAMA_INDEX_AVAILABLE",
    "ENGINE_LEGACY",
    "ENGINE_LLAMAINDEX",
    "ENGINES",
    "SimilarityCutoffPostprocessor",
    "ask_llamaindex",
    "build_hybrid_retriever",
    "build_query_engine",
    "extract_sources",
    "nodes_to_paper_results",
    "resolve_engine",
]

#: Engine names accepted by the CLI ``--engine`` switch.
ENGINE_LLAMAINDEX = "llamaindex"
ENGINE_LEGACY = "legacy"
ENGINES = (ENGINE_LLAMAINDEX, ENGINE_LEGACY)

#: Error message for an unbuildable engine (CLI catches it and falls back).
_ENGINE_UNAVAILABLE_MSG = (
    "llamaindex query engine unavailable (llamaindex.enabled=false, "
    "llama-index missing, or no index built yet)"
)


# ── engine resolution (CLI fallback rule, design §4.6) ──────────────────────


def resolve_engine(cfg: Config, requested: str = ENGINE_LLAMAINDEX) -> str:
    """Resolve a requested engine name to one that is actually usable.

    ``llamaindex`` is returned only when ``requested`` is ``llamaindex`` AND
    the ``llamaindex.enabled`` flag is set AND llama-index is importable.
    Every other combination (explicit ``legacy``, or llamaindex unavailable)
    resolves to ``legacy``. Never raises — this is the CLI's automatic-fallback
    rule (``llamaindex.enabled=false`` or import failure → legacy branch).
    """
    if str(requested).strip().lower() != ENGINE_LLAMAINDEX:
        return ENGINE_LEGACY
    if not _LLAMA_INDEX_AVAILABLE:
        return ENGINE_LEGACY
    if not bool(get_llamaindex_config(cfg).enabled):
        return ENGINE_LEGACY
    return ENGINE_LLAMAINDEX


def _engine_ready(cfg: Config) -> bool:
    """True when the llamaindex engine is both enabled and importable."""
    return resolve_engine(cfg, ENGINE_LLAMAINDEX) == ENGINE_LLAMAINDEX


# ── assembly ────────────────────────────────────────────────────────────────


def build_query_engine(
    cfg: Config,
    db: Any,
    streaming: bool | None = None,
    top_k: int | None = None,
):
    """Assemble the :class:`RetrieverQueryEngine` over the T4 fusion retriever.

    Args:
        cfg: Config (``llamaindex`` section read via
            :func:`~drbrain.rag.config.get_llamaindex_config`).
        db: Database handle for the tree/graph legs (only consumed when those
            legs are configured).
        streaming: Force the synthesizer's streaming flag; ``None`` → the
            configured ``llamaindex.streaming`` value.
        top_k: Fusion top-k (defaults to ``embed.top_k``).

    Returns a ``RetrieverQueryEngine``, or ``None`` when llamaindex is
    disabled/unavailable or there is nothing to retrieve against (no index
    built yet, no legs). The engine's node postprocessors contain the
    :class:`SimilarityCutoffPostprocessor` when a cutoff is configured, plus
    — T8, when ``llamaindex.rerank`` is enabled — the
    :class:`~drbrain.rag.rerank.RerankPostprocessor` (before the cutoff) and
    the :class:`~drbrain.rag.rerank.DeduplicatePostprocessor` (after it):
    coarse truncation at ``rerank_top_k`` → rerank 精排 → similarity cutoff →
    dedup. Rerank is lazy and degrades to Noop, so the engine is safe to
    build even when the reranker model is missing.
    """
    if not _engine_ready(cfg):
        return None
    from drbrain.rag.llm import init_llamaindex_settings

    # Wire Settings.llm/embed_model to DrbrainLLM/DrbrainEmbedding (idempotent,
    # offline-safe): the synthesizer factory reads Settings.llm by default.
    init_llamaindex_settings(cfg)

    li = get_llamaindex_config(cfg)
    # T8: with rerank on, the coarse fusion truncation must be at least as
    # wide as rerank_top_k, otherwise the reranker would only ever see the
    # caller's (smaller) final top-k and reranking would be a no-op.
    if li.rerank:
        top_k = max(int(top_k or cfg.embed.top_k or 10), int(li.rerank_top_k or 20))

    fusion = _build_fusion(cfg, db, top_k=top_k)
    if fusion is None:
        return None

    if streaming is None:
        streaming = bool(li.streaming)
    from llama_index.core.response_synthesizers.type import ResponseMode

    synth = get_response_synthesizer(response_mode=ResponseMode.REFINE, streaming=bool(streaming))

    # T8 postprocessor chain: rerank → cutoff → dedup (order is load-bearing;
    # rerank must run before the cutoff so the cutoff sees reranked order).
    postprocessors: list[Any] = []
    if li.rerank:
        from drbrain.rag.rerank import (
            DeduplicatePostprocessor,
            RerankPostprocessor,
            build_reranker,
        )

        postprocessors.append(
            RerankPostprocessor(top_k=int(li.rerank_top_k or 20), reranker=build_reranker(cfg))
        )
    if li.similarity_cutoff is not None:
        postprocessors.append(SimilarityCutoffPostprocessor(similarity_cutoff=li.similarity_cutoff))
    if li.rerank:
        postprocessors.append(DeduplicatePostprocessor())

    engine = RetrieverQueryEngine(
        retriever=fusion,
        response_synthesizer=synth,
        node_postprocessors=postprocessors or None,
    )
    return engine


def build_hybrid_retriever(cfg: Config, db: Any, top_k: int | None = None):
    """Build the T4 fusion retriever alone (no synthesizer) for hybrid/query.

    Returns a :class:`FusionRetriever`, or ``None`` when llamaindex is
    disabled/unavailable or no fusion legs exist (no index built yet).
    """
    if not _engine_ready(cfg):
        return None
    return _build_fusion(cfg, db, top_k=top_k)


def _build_fusion(cfg: Config, db: Any, top_k: int | None = None):
    """Assemble the named legs (T4 ``get_retrievers``) into one FusionRetriever."""
    from drbrain.rag.fusion import build_fusion_retriever, get_retrievers

    legs = get_retrievers(cfg, db)
    if not legs:
        return None
    vector = legs.pop("vector", None)
    bm25 = legs.pop("bm25", None)
    return build_fusion_retriever(
        cfg,
        vector_index=vector,
        bm25_retriever=bm25,
        custom_retrievers=legs,
        top_k=top_k,
    )


# ── source annotation (design §4.5 引文回链) ─────────────────────────────────


def extract_sources(nodes: list[NodeWithScore] | None) -> list[dict[str, Any]]:
    """Structured back-links from retrieved nodes.

    Each entry is ``{paper_id, node_id, title, score, sources}`` where
    ``sources`` lists the fusion legs that contributed the node (bm25/vector/
    tree/graph…). Missing metadata fields degrade to ``""``/``None`` so the
    CLI JSON stays serializable. ``node_id`` falls back to the node's own id
    when the metadata key is absent.
    """
    out: list[dict[str, Any]] = []
    for nws in nodes or []:
        node = getattr(nws, "node", None)
        meta = dict(getattr(node, "metadata", None) or {})
        out.append(
            {
                "paper_id": str(meta.get("paper_id") or ""),
                "node_id": str(meta.get("node_id") or getattr(node, "node_id", None) or ""),
                "title": str(meta.get("title") or ""),
                "score": float(nws.score) if nws.score is not None else None,
                "sources": list(meta.get("sources") or [])
                or ([meta["source"]] if meta.get("source") else []),
            }
        )
    return out


def nodes_to_paper_results(
    nodes: list[NodeWithScore] | None, top_k: int | None = None
) -> list[dict[str, Any]]:
    """Aggregate fused nodes to paper-level hits (hybrid compat).

    Groups :class:`NodeWithScore` by ``metadata.paper_id`` (falling back to
    the part before ``:`` of the node id), keeping the best score, section
    count, title and the union of contributing legs per paper. Returns
    ``[{paper_id, title, score, sections, sources, rank}]`` sorted by
    descending score — shape-comparable to the legacy hybrid
    ``SearchHit.to_dict()`` output at paper granularity.
    """
    papers: dict[str, dict[str, Any]] = {}
    for nws in nodes or []:
        node = getattr(nws, "node", None)
        meta = dict(getattr(node, "metadata", None) or {})
        pid = str(meta.get("paper_id") or "")
        if not pid:
            nid = str(getattr(node, "node_id", None) or "")
            pid = nid.split(":", 1)[0]
        if not pid:
            continue
        entry = papers.setdefault(
            pid,
            {"paper_id": pid, "title": "", "score": 0.0, "sections": 0, "sources": []},
        )
        score = float(nws.score) if nws.score is not None else 0.0
        if score > entry["score"]:
            entry["score"] = score
            if not entry["title"]:
                entry["title"] = str(meta.get("title") or "")
        entry["sections"] += 1
        for src in list(meta.get("sources") or []) or (
            [meta["source"]] if meta.get("source") else []
        ):
            if src and src not in entry["sources"]:
                entry["sources"].append(src)

    results = sorted(papers.values(), key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(results, start=1):
        r["rank"] = i
    if top_k:
        results = results[: int(top_k)]
    return results


# ── ask compatibility layer ─────────────────────────────────────────────────


def ask_llamaindex(
    cfg: Config,
    db: Any,
    question: str,
    top_k: int = 5,
    streaming: bool = True,
):
    """Legacy-``ask``-compatible query through the LlamaIndex engine.

    Returns the result dict ``{question, answer, sources, engine:
    "llamaindex"}`` (compat layer — legacy ``ask`` emits ``{question, answer,
    context}``, so the shapes line up for the CLI switch).

    With ``streaming=True`` the answer is produced by a lazy
    ``StreamingResponse`` and the return value is a *generator* that first
    yields intermediate ``{"chunk": str}`` items and finally the complete
    result dict. Callers that need a plain dict pass ``streaming=False``
    (the CLI forces this for ``--json`` output).

    Raises :class:`RuntimeError` when the engine cannot be built (disabled,
    llama-index missing, or no index yet) — the CLI catches this and falls
    back to the legacy engine.
    """
    if streaming:
        return _ask_llamaindex_stream(cfg, db, question, top_k=top_k)
    engine = build_query_engine(cfg, db, streaming=False, top_k=top_k)
    if engine is None:
        raise RuntimeError(_ENGINE_UNAVAILABLE_MSG)
    response = engine.query(question)
    return _assemble_answer(
        question,
        _response_text(response),
        _response_sources(response),
    )


def _ask_llamaindex_stream(cfg: Config, db: Any, question: str, top_k: int):
    """Streaming ask: yields ``{"chunk": str}`` items, then the final dict."""
    engine = build_query_engine(cfg, db, streaming=True, top_k=top_k)
    if engine is None:
        raise RuntimeError(_ENGINE_UNAVAILABLE_MSG)
    response = engine.query(question)
    yield from _iter_ask_results(question, response)


def _iter_ask_results(question: str, response: Any):
    """Consume a (possibly streaming) response into chunk + final dict yields."""
    sources = _response_sources(response)
    if isinstance(response, StreamingResponse):
        parts: list[str] = []
        for chunk in response.response_gen:
            text = _chunk_text(chunk)
            parts.append(text)
            yield {"chunk": text}
        answer = "".join(parts)
        if not answer:
            answer = _response_text(response)
    else:
        answer = _response_text(response)
    yield _assemble_answer(question, answer, sources)


def _assemble_answer(question: str, answer: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    """Shape the compat-layer output dict."""
    return {
        "question": question,
        "answer": answer or "",
        "sources": sources,
        "engine": ENGINE_LLAMAINDEX,
    }


def _chunk_text(chunk: Any) -> str:
    """Normalize a streamed chunk (str or an object with ``.delta``/``.text``)."""
    if isinstance(chunk, str):
        return chunk
    delta = getattr(chunk, "delta", None)
    if delta:
        return str(delta)
    return str(getattr(chunk, "text", None) or "")


def _response_text(response: Any) -> str:
    """Best-effort full text of a sync ``Response`` or consumed stream."""
    txt = getattr(response, "response", None)
    if txt is not None:
        return str(txt)
    return getattr(response, "response_txt", None) or ""


def _response_sources(response: Any) -> list[dict[str, Any]]:
    return extract_sources(getattr(response, "source_nodes", None) or [])


# ── node postprocessor (fused-score-aware similarity cutoff) ────────────────


if _LLAMA_INDEX_AVAILABLE:

    class SimilarityCutoffPostprocessor(BaseNodePostprocessor):
        """Similarity cutoff honoring fused-score semantics (module docstring).

        The stock ``SimilarityPostprocessor`` compares ``node.score`` against
        the cutoff; fused (RRF) scores are rank-derived and far below any
        similarity threshold, so this postprocessor instead evaluates the best
        *original per-leg* score recorded by the T4 fusion layer in
        ``node.metadata["contributions"]`` (these are the true
        cosine/BM25/positional scores). Nodes without fusion contributions
        fall back to their own score. ``score=None`` nodes are kept — the
        tree/graph legs legitimately produce positional/decayed scores.
        """

        # Declared pydantic field (BaseNodePostprocessor is a pydantic model;
        # undeclared attrs raise on assignment). ``None`` disables the filter.
        similarity_cutoff: float | None = None

        def __init__(self, similarity_cutoff: float | None = None) -> None:
            # ``similarity_cutoff`` is a field declared on *this* class; mypy's
            # synthesized ``BaseNodePostprocessor.__init__`` (base-class fields
            # only) does not know it, but pydantic validates it at runtime.
            super().__init__(  # type: ignore[call-arg]
                similarity_cutoff=similarity_cutoff
            )

        @classmethod
        def class_name(cls) -> str:
            return "SimilarityCutoffPostprocessor"

        def _postprocess_nodes(self, nodes, query_bundle=None):
            cutoff = self.similarity_cutoff
            if cutoff is None:
                return list(nodes)
            kept = []
            for nws in nodes:
                score = _effective_similarity(nws)
                if score is None or score >= float(cutoff):
                    kept.append(nws)
            return kept


def _effective_similarity(nws: Any) -> float | None:
    """Best original per-leg score of a fused node, else its own score.

    ``contributions`` has shape ``{source: {rank, score, weight}}`` (T4
    FusionRetriever annotation). The per-leg ``score`` is the quantity
    comparable to a similarity threshold.
    """
    score = getattr(nws, "score", None)
    node = getattr(nws, "node", None)
    meta = dict(getattr(node, "metadata", None) or {}) if node is not None else {}
    contributions = meta.get("contributions")
    if isinstance(contributions, dict) and contributions:
        best: float | None = None
        for info in contributions.values():
            s = (info or {}).get("score")
            if s is None:
                continue
            s = float(s)
            if best is None or s > best:
                best = s
        if best is not None:
            return best
    return float(score) if score is not None else None
