"""RAPTOR recursive semantic tree (2401.18059).

Builds a hierarchical summary tree on top of PageIndex leaf nodes:
  1. Embed PageIndex nodes via Layer 2 embedding service.
  2. UMAP reduce dimensionality for better GMM clustering.
  3. GMM clustering with BIC automatic k selection.
  4. LLM summarize each cluster → tree_summaries rows.
  5. Re-embed summaries → repeat until convergence.

Usage:
    from drbrain.extractor.raptor import build_raptor_tree
    count = await build_raptor_tree(paper_dir, db_path, cfg.embed)
"""

from __future__ import annotations

import json
import logging
import struct
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from drbrain.config import EmbedConfig

log = logging.getLogger(__name__)

_RAPTOR_MAX_LAYERS = 3
_RAPTOR_UMAP_COMPONENTS = 10
_RAPTOR_MAX_CLUSTERS = 10
_RAPTOR_MIN_CLUSTER_SIZE = 2


# ── BIC ──────────────────────────────────────────────────────────────────────


def _bic_gmm(x: list[list[float]], k: int) -> float:
    """Bayesian Information Criterion for GMM with k clusters.

    BIC = ln(N) * n_params - 2 * ln(L)
    Lower BIC is better.
    """
    import numpy as np
    from sklearn.mixture import GaussianMixture

    x_arr = np.asarray(x, dtype="float32")
    n_samples, n_features = x_arr.shape
    if k < 1 or k > n_samples:
        return float("inf")

    gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        random_state=42,
        max_iter=100,
        n_init=2,
    )
    gmm.fit(x_arr)
    return float(gmm.bic(x_arr))


# ── GMM clustering ───────────────────────────────────────────────────────────


def _gmm_cluster(x: list[list[float]], max_k: int = _RAPTOR_MAX_CLUSTERS) -> list[list[int]]:
    """Cluster embeddings with GMM, selecting k via BIC.

    Returns list of cluster index lists.
    """
    import numpy as np
    from sklearn.mixture import GaussianMixture

    x_arr = np.asarray(x, dtype="float32")
    n_samples = x_arr.shape[0]

    if n_samples < _RAPTOR_MIN_CLUSTER_SIZE + 1:
        return [list(range(n_samples))]

    # Try k from 1 to min(max_k, n_samples)
    max_try = min(max_k, n_samples)
    best_k = 1
    best_bic = float("inf")

    for k in range(1, max_try + 1):
        try:
            bic = _bic_gmm(x, k)
            if bic < best_bic:
                best_bic = bic
                best_k = k
        except Exception:
            continue

    if best_k <= 1:
        return [list(range(n_samples))]

    # Fit GMM with best k
    gmm = GaussianMixture(
        n_components=best_k,
        covariance_type="full",
        random_state=42,
        max_iter=100,
        n_init=3,
    )
    labels = gmm.fit_predict(x_arr)

    # Group indices by cluster label
    clusters: list[list[int]] = [[] for _ in range(best_k)]
    for i, label in enumerate(labels):
        clusters[label].append(i)

    # Filter out empty clusters
    return [c for c in clusters if len(c) >= _RAPTOR_MIN_CLUSTER_SIZE]


# ── UMAP ─────────────────────────────────────────────────────────────────────


def _umap_reduce(
    x: list[list[float]], n_components: int = _RAPTOR_UMAP_COMPONENTS
) -> list[list[float]]:
    """UMAP dimensionality reduction for better GMM performance.

    RAPTOR paper: high-dimensional embeddings cause GMM distance metrics
    to behave poorly. UMAP mitigates this.
    """
    import numpy as np
    import umap

    x_arr = np.asarray(x, dtype="float32")
    n_samples, n_features = x_arr.shape

    # Skip UMAP if too few samples
    if n_samples <= n_components:
        return x

    reducer = umap.UMAP(
        n_components=min(n_components, n_features - 1, n_samples - 1),
        n_neighbors=min(15, n_samples - 1),
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    reduced = reducer.fit_transform(x_arr)
    return reduced.astype("float32").tolist()


# ── Summarization ────────────────────────────────────────────────────────────


async def _summarize_cluster(
    texts: list[str],
    models: list[dict],
) -> str:
    """LLM summarize a cluster of text nodes.

    Uses the same LLM fallback chain as the rest of DrBrain.
    """
    from drbrain.extractor.llm_client import acall_text_with_fallback

    combined = "\n\n---\n\n".join(f"[{i + 1}] {t[:2000]}" for i, t in enumerate(texts))
    prompt = (
        "You are summarizing a group of related text sections from an academic paper. "
        "Write a concise 2-4 sentence summary that captures the key themes and "
        "findings shared across these sections. Focus on what they have in common.\n\n"
        f"Text sections:\n\n{combined}\n\n"
        "Concise summary (2-4 sentences):"
    )
    summary = await acall_text_with_fallback(prompt, models)
    return summary.strip() if summary else " ".join(t.split()[:200] for t in texts)[:500]


# ── Main RAPTOR tree builder ─────────────────────────────────────────────────


async def build_raptor_tree(
    paper_dir: Path,
    db_path: Path,
    embed_cfg: EmbedConfig | None = None,
    models: list[dict] | None = None,
    max_layers: int = _RAPTOR_MAX_LAYERS,
) -> int:
    """Build RAPTOR recursive summary tree for a single paper.

    1. Collect PageIndex leaf nodes + their texts.
    2. Embed all nodes via Layer 2.
    3. UMAP → GMM+BIC → cluster.
    4. LLM summarize each cluster → store in tree_summaries.
    5. Re-embed summaries → repeat until convergence.

    Args:
        paper_dir: Paper directory with tree.json and raw.md.
        db_path: SQLite database path.
        embed_cfg: Embedding configuration.
        models: LLM model list for summarization. If None or empty, falls back
            to raw text concatenation (no LLM summarization).
        max_layers: Maximum recursive layers.

    Returns:
        Total number of summary nodes created.
    """
    import sqlite3

    from drbrain.services.embedding import (
        _collect_tree_nodes,
        _content_hash,
        _embed_batch,
    )

    paper_id = paper_dir.name
    total_summaries = 0

    conn = sqlite3.connect(str(db_path))
    try:
        # Step 1: Read existing PageIndex vectors from DB (already stored by build_tree_vectors)
        rows = conn.execute(
            "SELECT node_id, embedding FROM tree_vectors "
            "WHERE tree_layer = 'pageindex' AND paper_id = ?",
            (paper_id,),
        ).fetchall()

        if len(rows) < 3:
            log.info("Too few PageIndex vectors (%d) for RAPTOR clustering", len(rows))
            return 0

        # Collect node texts from tree.json (needed for LLM summarization)
        nodes = _collect_tree_nodes(paper_dir)
        node_text_map: dict[str, str] = {n["node_id"]: n["text"] for n in nodes}

        node_ids: list[str] = []
        node_texts: list[str] = []
        vectors: list[list[float]] = []
        for row in rows:
            nid = row[0]
            blob = row[1]
            stored_dim = len(blob) // 4  # float32 = 4 bytes
            vec = list(struct.unpack(f"{stored_dim}f", blob))
            node_ids.append(nid)
            node_texts.append(node_text_map.get(nid, ""))
            vectors.append(vec)

        current_texts = node_texts
        current_ids = node_ids
        current_vecs = vectors

        # RAPTOR recursive loop
        for layer in range(1, max_layers + 1):
            if len(current_vecs) < 3:
                break

            # Step 3: UMAP reduce → GMM cluster
            reduced = _umap_reduce(current_vecs)
            clusters = _gmm_cluster(reduced)

            if len(clusters) <= 1:
                log.info(
                    "RAPTOR layer %d: no meaningful clusters found, stopping",
                    layer,
                )
                break

            # Step 4: Summarize each cluster
            layer_count = 0
            for cluster_indices in clusters:
                cluster_texts = [current_texts[i] for i in cluster_indices]
                cluster_source_ids = [current_ids[i] for i in cluster_indices]

                try:
                    summary = await _summarize_cluster(
                        cluster_texts,
                        models or [],
                    )
                except Exception:
                    log.warning(
                        "RAPTOR summarization failed for cluster of %d nodes",
                        len(cluster_texts),
                    )
                    continue

                raptor_id = f"raptor_{paper_id}_L{layer}_{uuid.uuid4().hex[:8]}"

                # Store summary in tree_summaries
                conn.execute(
                    "INSERT OR REPLACE INTO tree_summaries "
                    "(node_id, paper_id, summary_text, source_node_ids, tree_layer) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        raptor_id,
                        paper_id,
                        summary,
                        json.dumps(cluster_source_ids),
                        layer,
                    ),
                )
                layer_count += 1

            if layer_count == 0:
                break

            conn.commit()
            total_summaries += layer_count
            log.info(
                "RAPTOR layer %d: %d clusters → %d summaries",
                layer,
                len(clusters),
                layer_count,
            )

            # Step 5: Re-embed summaries for next layer
            summary_rows = conn.execute(
                "SELECT node_id, summary_text FROM tree_summaries "
                "WHERE paper_id = ? AND tree_layer = ?",
                (paper_id, layer),
            ).fetchall()

            if len(summary_rows) < 3:
                break

            current_texts = [r[1] for r in summary_rows]
            current_ids = [r[0] for r in summary_rows]
            current_vecs = _embed_batch(current_texts, embed_cfg)

            # Store RAPTOR node embeddings
            for nid, txt, vec in zip(current_ids, current_texts, current_vecs):
                blob = struct.pack(f"{len(vec)}f", *vec)
                conn.execute(
                    "INSERT OR REPLACE INTO tree_vectors "
                    "(node_id, paper_id, embedding, content_hash, tree_layer) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        nid,
                        paper_id,
                        blob,
                        _content_hash(txt),
                        f"raptor_L{layer}",
                    ),
                )

        conn.commit()
        return total_summaries

    finally:
        conn.close()
