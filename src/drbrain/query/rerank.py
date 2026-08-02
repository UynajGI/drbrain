"""Optional rerank layer with graceful degradation.

A reranker reorders ``SearchHit`` results using a query-document relevance
model (typically a cross-encoder). Rerankers are optional: if the backing
model or its dependencies are unavailable, the system must fall back to the
input order — never crash the retrieval pipeline.

Two concrete implementations:
    - ``NoopReranker``: returns ``hits[:top_n]`` unchanged. Always available.
    - ``CrossEncoderReranker``: lazily loads a HuggingFace cross-encoder
      (e.g. ``BAAI/bge-reranker-base``) via ``sentence_transformers``. On
      ``ImportError`` (library or model missing) it logs and degrades to
      no-op, so callers never see an exception.

``get_reranker("auto")`` returns a ``CrossEncoderReranker`` (which self-
degrades) — the recommended default. ``get_reranker("none")`` returns a
``NoopReranker`` for explicit opt-out.

Document text is pulled from each hit's ``payload``: BM25 rows expose
``label``/``text``, embedding rows expose ``node_id``/``paper_id``. Hits
without usable text are kept in their input position (the reranker cannot
score them, so we preserve the retrieval order).
"""

from __future__ import annotations

import logging
from typing import Protocol

from drbrain.query.types import SearchHit

log = logging.getLogger(__name__)

DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"


class Reranker(Protocol):
    """Interface for reranking fused search hits."""

    def rerank(self, query: str, hits: list[SearchHit], top_n: int = 10) -> list[SearchHit]:
        """Return up to ``top_n`` hits reordered by query relevance.

        Implementations must be total: never raise on missing deps/model.
        """
        ...


class NoopReranker:
    """Pass-through reranker. Returns input hits truncated to ``top_n``."""

    def rerank(self, query: str, hits: list[SearchHit], top_n: int = 10) -> list[SearchHit]:  # noqa: ARG002
        return list(hits)[:top_n]


class CrossEncoderReranker:
    """Cross-encoder reranker with automatic no-op fallback.

    The model is loaded lazily on first ``rerank()`` call. If
    ``sentence_transformers`` (and its ``torch`` dependency) is unavailable,
    or the model cannot be loaded, the instance permanently degrades to
    no-op behavior and logs a single warning.
    """

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL) -> None:
        self.model_name = model_name
        self._model = None
        self._failed = False  # sticky: don't retry imports every call

    def _ensure_model(self) -> bool:
        """Lazily load the cross-encoder. Returns False on any failure."""
        if self._model is not None:
            return True
        if self._failed:
            return False
        try:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415
        except ImportError:
            log.warning(
                "[rerank] sentence_transformers unavailable; "
                "falling back to input order (no-op rerank). "
                "Install with: pip install sentence-transformers"
            )
            self._failed = True
            return False
        try:
            self._model = CrossEncoder(self.model_name)
            log.info("[rerank] loaded cross-encoder: %s", self.model_name)
            return True
        except Exception as e:  # model download/load failure
            log.warning(
                "[rerank] failed to load %s (%s); falling back to no-op",
                self.model_name,
                e,
            )
            self._failed = True
            return False

    def rerank(self, query: str, hits: list[SearchHit], top_n: int = 10) -> list[SearchHit]:
        if not hits:
            return []
        if not self._ensure_model():
            return list(hits)[:top_n]

        # Build (query, doc) pairs; track which hits have usable text.
        pairs: list[tuple[int, str]] = []
        for idx, hit in enumerate(hits):
            doc = _hit_to_text(hit)
            if doc:
                pairs.append((idx, doc))

        if not pairs:
            # Nothing scoreable; preserve input order.
            return list(hits)[:top_n]

        try:
            scores = self._model.predict([(query, doc) for _, doc in pairs])
        except Exception as e:  # inference failure mid-flight
            log.warning("[rerank] predict failed (%s); falling back to input order", e)
            return list(hits)[:top_n]

        # Attach rerank scores to the scoreable hits, sort, then splice back
        # the unscoreable ones at the tail (preserving their input order).
        scored: list[tuple[float, int]] = [(float(s), idx) for (idx, _), s in zip(pairs, scores)]
        scored.sort(key=lambda x: x[0], reverse=True)

        scored_ids = {idx for _, idx in scored}
        tail = [hit for idx, hit in enumerate(hits) if idx not in scored_ids]

        ordered = [hits[idx] for _, idx in scored] + tail
        result = ordered[:top_n]
        for i, hit in enumerate(result, start=1):
            hit.rank = i
        return result


def _hit_to_text(hit: SearchHit) -> str:
    """Extract a readable document string from a hit's payload.

    BM25 rows carry ``label``/``text``; embedding rows carry ``node_id``.
    Fused hits carry a ``{source: row}`` payload — we concatenate any
    readable rows, preferring BM25 text when present.
    """
    payload = hit.payload
    if not payload:
        return ""

    # Fused payload is {source: row}; raw payload is the row itself.
    if "bm25" in payload or "embedding" in payload:
        parts: list[str] = []
        for row in payload.values():
            if not isinstance(row, dict):
                continue
            text = _row_to_text(row)
            if text:
                parts.append(text)
        return " | ".join(dict.fromkeys(parts))  # dedupe, keep order

    return _row_to_text(payload)


def _row_to_text(row: dict) -> str:
    """Best-effort text from a single retriever row dict."""
    label = str(row.get("label") or "").strip()
    text = str(row.get("text") or "").strip()
    if text:
        return f"{label}\n{text}" if label else text
    if label:
        return label
    # Embedding rows have no label/text; fall back to node_id as a handle.
    node_id = str(row.get("node_id") or "").strip()
    return node_id


def get_reranker(name: str = "auto", model_name: str | None = None) -> Reranker:
    """Construct a reranker by name.

    Args:
        name: ``"auto"`` (default) returns a ``CrossEncoderReranker`` that
            self-degrades to no-op if deps are missing. ``"none"`` returns
            an explicit ``NoopReranker``.
        model_name: Cross-encoder model id; ignored for ``name="none"``.

    Returns:
        A ``Reranker`` instance. Always returns a usable object.
    """
    if name == "none":
        return NoopReranker()
    return CrossEncoderReranker(model_name or DEFAULT_RERANK_MODEL)


__all__ = [
    "Reranker",
    "NoopReranker",
    "CrossEncoderReranker",
    "get_reranker",
    "DEFAULT_RERANK_MODEL",
]
