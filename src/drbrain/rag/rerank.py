"""Reranking + node postprocessing layer (T8).

Ticket: T8 (重排+后处理). Depends on T4 (fusion) / T5 (query engine).

The ``llama-index-postprocessor-flag-embedding-reranker`` plugin is **not**
installed (and, per the T1 decision, extra plugins are deferred), so the
reranker is self-implemented on the already-present
``sentence-transformers`` CrossEncoder stack (sentence-transformers 5.6.1 +
torch 2.10.0, offline-loadable). Production model: ``Qwen/Qwen3-Reranker-0.6B`` (user decision 2026-08-12: same family as the Qwen3-Embedding-0.6B embed model; loaded via modelscope when the HF hub is unreachable).
(~1.1 GB, downloaded on first use); tests substitute a mock or a tiny cached
model.

Design:

* :class:`CrossEncoderReranker` — thin, **lazy** wrapper over
  ``sentence_transformers.CrossEncoder``. Constructing it costs nothing and
  never touches the network/GPU; the model loads on the first ``rerank``
  call (or ``available`` check). Any load failure degrades to
  ``available == False`` with a single logged warning — the query chain never
  crashes because a reranker model is missing.
* :func:`build_reranker` — factory wired to ``llamaindex.rerank_model`` /
  ``embed.device`` / ``embed.batch_size`` config.
* :class:`RerankPostprocessor` — a ``BaseNodePostprocessor`` that re-scores
  the coarse (fused) top-``top_k`` candidates with the reranker and sorts by
  the new score. Model unavailable / rerank error / no query → **Noop**
  (returns the input unchanged), so reranking can never take the query path
  down.
* :class:`DeduplicatePostprocessor` — keep-first dedup by ``node_id`` with
  content-hash fallback (safety net: fusion already dedups by ``node_id``,
  but tree/graph legs fabricate nodes that can collide after rescoring).

Postprocessor chain order (engine.py assembly): fusion retriever →
:class:`RerankPostprocessor` (top_k = ``llamaindex.rerank_top_k``) →
:class:`~drbrain.rag.engine.SimilarityCutoffPostprocessor` →
:class:`DeduplicatePostprocessor`. Rerank runs *before* the cutoff so the
cutoff sees reranked order but still evaluates the *original* per-leg
similarity scores (``contributions``), not rerank logits (T5 semantics).

Rank-comparison helpers (:func:`top_k_overlap`, :func:`mean_rank_displacement`,
:func:`kendall_tau`) back the A/B tool (``scripts/rerank_ab.py``).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from drbrain.config import Config
from drbrain.rag.config import get_llamaindex_config

try:
    from llama_index.core.postprocessor.node import BaseNodePostprocessor
    from llama_index.core.schema import NodeWithScore
    from pydantic import PrivateAttr

    _LLAMA_INDEX_AVAILABLE = True
except ImportError:  # pragma: no cover - envs without llama-index
    BaseNodePostprocessor = None  # type: ignore[assignment,misc]
    NodeWithScore = None  # type: ignore[assignment,misc]
    PrivateAttr = None  # type: ignore[assignment,misc]
    _LLAMA_INDEX_AVAILABLE = False

log = logging.getLogger(__name__)

__all__ = [
    "_LLAMA_INDEX_AVAILABLE",
    "CrossEncoderReranker",
    "DeduplicatePostprocessor",
    "RerankPostprocessor",
    "build_reranker",
    "kendall_tau",
    "mean_rank_displacement",
    "top_k_overlap",
]


# ── cross-encoder reranker (lazy, offline-safe, degrade-on-failure) ─────────


def _normalize_device(device: str | None) -> str | None:
    """Map drbrain ``embed.device`` values to CrossEncoder's ``device`` arg.

    ``"auto"``/``None``/``""`` → ``None`` (sentence-transformers auto-selects
    CUDA when available); anything else (``cuda``, ``cuda:0``, ``cpu``) passes
    through untouched.
    """
    d = str(device or "").strip().lower()
    if not d or d in ("auto", "none"):
        return None
    return d


def _hf_reachable(timeout: float = 3.0) -> bool:
    """Cheap reachability probe for huggingface.co (bounded by ``timeout``)."""
    import socket

    try:
        with socket.create_connection(("huggingface.co", 443), timeout=timeout):
            return True
    except OSError:
        return False


def _resolve_rerank_model_path(model_name: str, cfg: Config) -> str:
    """Resolve a configured reranker id to a locally-loadable model path.

    Tries, in order: an existing filesystem path; the modelscope cache
    (``embed.cache_dir`` — the production source for Qwen models on this
    host, since huggingface.co is unreachable). Returns the id unchanged when
    nothing local matches, letting sentence-transformers/huggingface_hub
    attempt their normal resolution (gated by the ``_hf_reachable`` download
    check in :class:`CrossEncoderReranker`).
    """
    if "/" not in str(model_name) or Path(str(model_name)).exists():
        return str(model_name)
    try:
        from drbrain.services.embedding import _find_local_model_path

        embed = getattr(cfg, "embed", None)
        cache_dir = getattr(embed, "cache_dir", None) or "~/.cache/modelscope/hub/models"
        local = _find_local_model_path(str(model_name), str(cache_dir))
        if local:
            return local
    except Exception:  # pragma: no cover - resolution is best-effort
        pass
    try:
        # ModelScope's own cache resolution (handles the <org>--<repo> layout
        # under ~/.cache/modelscope/models, where the Qwen models live).
        from modelscope import snapshot_download

        local_path = snapshot_download(str(model_name), local_files_only=True)
        if local_path:
            return str(local_path)
    except Exception:  # pragma: no cover - resolution is best-effort
        pass
    # Last resort: probe the known on-disk layouts directly. snapshot_download
    # can miss an HF-hub-style "<org>--<repo>" directory even when it exists
    # (observed with Qwen3-Reranker-0.6B under ~/.cache/modelscope/models).
    dashed = str(model_name).replace("/", "--")
    for root in (
        "~/.cache/modelscope/models",
        "~/.cache/modelscope/hub/models",
        "~/.cache/huggingface/hub",
    ):
        base = Path(root).expanduser()
        for cand in (base / dashed, base / f"models--{dashed}"):
            if not cand.is_dir():
                continue
            # ModelScope snapshot layout stores weights one level down
            # (<repo>/snapshots/<commit>/...); the loadable dir is that leaf.
            snapshots = cand / "snapshots"
            if snapshots.is_dir():
                leaves = sorted(p for p in snapshots.iterdir() if p.is_dir())
                cand = leaves[-1] if leaves else cand
            if any(cand.glob("*.safetensors")) or any(cand.glob("*.bin")):
                return str(cand)
    return str(model_name)


class CrossEncoderReranker:
    """Lazy wrapper over ``sentence_transformers.CrossEncoder``.

    Attributes:
        model_name: HF model id or local path (e.g. ``Qwen/Qwen3-Reranker-0.6B``).
        device: CrossEncoder device arg (``None`` = auto-select).
        batch_size: predict batch size.
        max_length: token limit per passage (long sections are truncated).

    The model is instantiated only on the first :meth:`rerank` call or
    :attr:`available` check, cache-first (``local_files_only``): a locally
    cached model (or local path) loads instantly and fully offline; a missing
    model only triggers a download when :func:`_hf_reachable` proves
    huggingface.co is reachable. A failed load is recorded once and never
    retried in-process (:attr:`available` becomes ``False``), which is the
    degrade path the query chain relies on — the offline-missing-model case
    degrades in ~seconds instead of huggingface_hub's multi-minute retry
    window.
    """

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        self.model_name = str(model_name)
        self.device = _normalize_device(device)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self._model: Any = None
        self._load_attempted = False
        self._load_error: Exception | None = None

    @property
    def available(self) -> bool:
        """True once the cross-encoder is loaded (triggers the one load attempt)."""
        if self._model is None and not self._load_attempted:
            self._load()
        return self._model is not None

    def _load(self) -> None:
        self._load_attempted = True
        try:
            from sentence_transformers import CrossEncoder

            try:
                # Cache-first: instant and fully offline when the model is
                # present locally (production after the first download).
                self._model = CrossEncoder(
                    self.model_name,
                    device=self.device,
                    max_length=self.max_length,
                    local_files_only=True,
                )
                return
            except Exception as cache_exc:  # noqa: BLE001 - fall through to download
                # Only attempt a download when a quick probe proves HF is
                # reachable. Without this gate, a missing model on an offline
                # host would stall the *first query* for huggingface_hub's
                # full multi-minute retry window — exactly the kind of
                # rerank-related query-chain outage the Noop degrade exists to
                # prevent. With the gate, offline degrade costs ~seconds.
                if not _hf_reachable():
                    raise OSError(
                        f"reranker model {self.model_name!r} not in cache and "
                        f"huggingface.co unreachable ({cache_exc})"
                    ) from cache_exc
                self._model = CrossEncoder(
                    self.model_name,
                    device=self.device,
                    max_length=self.max_length,
                    local_files_only=False,
                )
        except Exception as exc:  # noqa: BLE001 - degrade, never raise
            self._load_error = exc
            log.warning(
                "[rag] reranker model %r load failed (%s); rerank disabled (Noop)",
                self.model_name,
                exc,
            )

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        """Score each ``(query, passage)`` pair; returns one float per passage.

        Raises :class:`RuntimeError` when the model is unavailable — callers
        (e.g. :class:`RerankPostprocessor`) are expected to catch it and
        degrade to the coarse order.
        """
        if not self.available:
            raise RuntimeError(
                f"reranker model {self.model_name!r} unavailable"
                + (f": {self._load_error}" if self._load_error else "")
            )
        pairs = [[query, p] for p in passages]
        scores = self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return [float(s) for s in scores]


def build_reranker(cfg: Config):
    """Build a :class:`CrossEncoderReranker` from ``llamaindex`` + ``embed`` config.

    Never loads the model (lazy) and never fails — an unset model name
    yields ``None`` (callers treat it as "rerank disabled"). The configured
    model id is resolved to a locally-loadable path first (modelscope / HF
    cache) so a pre-cached model loads fully offline.
    """
    if not _LLAMA_INDEX_AVAILABLE:
        return None
    li = get_llamaindex_config(cfg)
    model = str(li.rerank_model or "").strip()
    if not model:
        return None
    embed = getattr(cfg, "embed", None)
    return CrossEncoderReranker(
        model_name=_resolve_rerank_model_path(model, cfg),
        device=getattr(embed, "device", None),
        batch_size=int(getattr(embed, "batch_size", None) or 32),
    )


# ── node postprocessors ─────────────────────────────────────────────────────


if _LLAMA_INDEX_AVAILABLE:

    class RerankPostprocessor(BaseNodePostprocessor):
        """Re-score the coarse top-``top_k`` candidates with a reranker.

        Takes the first ``top_k`` fused nodes (coarse order), scores every
        ``(query, node)`` pair with the :class:`CrossEncoderReranker`, and
        returns them sorted by the new score (nodes beyond ``top_k`` are
        dropped — the "粗排截断 → rerank 精排" contract).

        Degrade paths (all Noop — the input list is returned unchanged, the
        query chain survives): no reranker configured, model load failed,
        no query string, rerank call raised, or the score count mismatches.
        """

        # Declared pydantic fields (BaseNodePostprocessor is a pydantic model).
        top_k: int = 20
        reranker: Any = None  # CrossEncoderReranker | mock | None
        _last_trace: dict[str, Any] = PrivateAttr(default_factory=dict)

        def __init__(self, top_k: int = 20, reranker=None) -> None:
            # ``top_k``/``reranker`` are fields declared on *this* class;
            # mypy's synthesized ``BaseNodePostprocessor.__init__`` (base-class
            # fields only) does not know them, but pydantic validates them via
            # the concrete class at runtime.
            super().__init__(  # type: ignore[call-arg]
                top_k=int(top_k), reranker=reranker
            )

        @classmethod
        def class_name(cls) -> str:
            return "RerankPostprocessor"

        def get_last_trace(self) -> dict[str, Any]:
            """Return JSON-safe telemetry for the latest rerank pass."""
            return dict(self._last_trace)

        def _set_trace(
            self,
            status: str,
            *,
            started: float,
            input_nodes: int,
            candidates: int,
            output_nodes: int,
        ) -> None:
            reranker = self.reranker
            self._last_trace = {
                "stage": "rerank",
                "status": status,
                "input_nodes": input_nodes,
                "candidates": candidates,
                "output_nodes": output_nodes,
                "model": str(getattr(reranker, "model_name", "")) if reranker else None,
                "duration_ms": (time.perf_counter() - started) * 1000.0,
            }

        def _postprocess_nodes(self, nodes, query_bundle=None):
            started = time.perf_counter()
            input_nodes = len(nodes or [])
            if not nodes:
                self._set_trace(
                    "no_candidates",
                    started=started,
                    input_nodes=input_nodes,
                    candidates=0,
                    output_nodes=0,
                )
                return []
            reranker = self.reranker
            if reranker is None:
                self._set_trace(
                    "disabled",
                    started=started,
                    input_nodes=input_nodes,
                    candidates=0,
                    output_nodes=len(nodes),
                )
                return list(nodes)  # no reranker configured → Noop
            if not getattr(reranker, "available", False):
                # load attempted and failed → Noop (warning already logged once
                # by the reranker; don't spam per query)
                self._set_trace(
                    "unavailable",
                    started=started,
                    input_nodes=input_nodes,
                    candidates=0,
                    output_nodes=len(nodes),
                )
                return list(nodes)
            query = getattr(query_bundle, "query_str", None) if query_bundle else None
            if not query:
                self._set_trace(
                    "missing_query",
                    started=started,
                    input_nodes=input_nodes,
                    candidates=0,
                    output_nodes=len(nodes),
                )
                return list(nodes)  # nothing to rerank against → Noop
            top = list(nodes)[: self.top_k]
            passages = [_passage_text(nws.node) for nws in top]
            try:
                scores = reranker.rerank(query, passages)
            except Exception as exc:  # noqa: BLE001 - degrade, never raise
                log.warning("[rag] rerank failed (%s); falling back to coarse order", exc)
                self._set_trace(
                    "error",
                    started=started,
                    input_nodes=input_nodes,
                    candidates=len(top),
                    output_nodes=len(nodes),
                )
                return list(nodes)
            if not scores or len(scores) != len(top):
                log.warning(
                    "[rag] rerank returned %s scores for %s passages; falling back to coarse order",
                    len(scores or []),
                    len(top),
                )
                self._set_trace(
                    "invalid_scores",
                    started=started,
                    input_nodes=input_nodes,
                    candidates=len(top),
                    output_nodes=len(nodes),
                )
                return list(nodes)
            reranked = sorted(
                (
                    NodeWithScore(node=nws.node, score=float(s))
                    for nws, s in zip(top, scores)
                    if s is not None
                ),
                key=lambda x: x.score if x.score is not None else float("-inf"),
                reverse=True,
            )
            self._set_trace(
                "ok",
                started=started,
                input_nodes=input_nodes,
                candidates=len(top),
                output_nodes=len(reranked),
            )
            return reranked

    class DeduplicatePostprocessor(BaseNodePostprocessor):
        """Keep-first dedup by ``node_id`` (content-hash fallback).

        Safety net after rerank: the fusion layer already dedups by
        ``node_id``, but the tree/graph legs fabricate nodes at query time and
        a rescored list can legitimately collide. The first occurrence in the
        current (post-rerank) order wins.
        """

        @classmethod
        def class_name(cls) -> str:
            return "DeduplicatePostprocessor"

        def _postprocess_nodes(self, nodes, query_bundle=None):
            seen: set[Any] = set()
            out = []
            for nws in nodes:
                node = getattr(nws, "node", None)
                if node is None:
                    out.append(nws)
                    continue
                key = node.node_id or node.hash or id(node)
                if key in seen:
                    continue
                seen.add(key)
                out.append(nws)
            return out


def _passage_text(node: Any) -> str:
    """Node text for a rerank pair (plain text, no metadata prefix)."""
    text = getattr(node, "text", None)
    if text:
        return text
    get_content = getattr(node, "get_content", None)
    return str(get_content() if get_content else "")


# ── rank-comparison statistics (back the A/B tool) ──────────────────────────


def top_k_overlap(ids_a: list[str], ids_b: list[str], k: int) -> float:
    """Jaccard overlap of the top-``k`` id sets of two rankings (0.0–1.0)."""
    a = set(ids_a[:k])
    b = set(ids_b[:k])
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def mean_rank_displacement(ids_a: list[str], ids_b: list[str]) -> float:
    """Mean absolute rank shift (1-based) of ids present in both lists."""
    rank_b = {nid: i for i, nid in enumerate(ids_b, start=1)}
    common = [(i, rank_b[nid]) for i, nid in enumerate(ids_a, start=1) if nid in rank_b]
    if not common:
        return 0.0
    return sum(abs(i - j) for i, j in common) / len(common)


def kendall_tau(ids_a: list[str], ids_b: list[str]) -> float:
    """Kendall tau over the ids common to both rankings (−1.0…1.0)."""
    rank_b = {nid: i for i, nid in enumerate(ids_b)}
    common = [rank_b[nid] for nid in ids_a if nid in rank_b]
    concordant = discordant = 0
    n = len(common)
    for i in range(n):
        for j in range(i + 1, n):
            delta = (common[i] - common[j]) * (i - j)
            if delta > 0:
                concordant += 1
            elif delta < 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 0.0
