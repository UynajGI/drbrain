"""Evaluation metrics for temporal link prediction.

Provides ROC/AUC, Precision/Recall@k, and stratification by previous node
distance ``d_prev`` (shortest-path distance in the before-graph) — the paper
shows semantic features especially help at ``d_prev = 3``.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve (0.5 fallback for single-class input)."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, np.asarray(y_score)))


def precision_recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> tuple[float, float]:
    """Precision and Recall among the top-``k`` scored pairs.

    Args:
        y_true: Binary labels.
        y_score: Predicted probabilities.
        k: Number of top predictions to consider.

    Returns:
        ``(precision, recall)``.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    k = min(k, len(y_score))
    if k <= 0:
        return 0.0, 0.0
    order = np.argsort(y_score)[::-1][:k]
    hits = int(y_true[order].sum())
    precision = hits / k
    total_pos = int(y_true.sum())
    recall = hits / total_pos if total_pos > 0 else 0.0
    return precision, recall


def previous_distance(g_before: nx.Graph, u: str, v: str) -> int | None:
    """Shortest-path distance between ``u`` and ``v`` in the before-graph.

    Returns ``None`` when the nodes are disconnected (or absent).
    """
    if u not in g_before or v not in g_before:
        return None
    try:
        return int(nx.shortest_path_length(g_before, u, v))
    except nx.NetworkXNoPath:
        return None


def stratify_by_dprev(
    pairs: list[tuple[str, str]],
    y_true: np.ndarray,
    y_score: np.ndarray,
    g_before: nx.Graph,
) -> dict[int, dict]:
    """Group AUC by previous node distance ``d_prev``.

    Args:
        pairs: Node pairs parallel to ``y_true`` / ``y_score``.
        y_true: Binary labels.
        y_score: Predicted probabilities.
        g_before: The before-graph used to compute distances.

    Returns:
        ``{d_prev: {"count": n, "positives": p, "auc": a}}`` (disconnected pairs
        are bucketed under ``-1``).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    buckets: dict[int, list[int]] = {}
    for i, (u, v) in enumerate(pairs):
        d = previous_distance(g_before, u, v)
        key = d if d is not None else -1
        buckets.setdefault(key, []).append(i)

    out: dict[int, dict] = {}
    for d, idxs in sorted(buckets.items()):
        yt = y_true[idxs]
        ys = y_score[idxs]
        out[d] = {
            "count": len(idxs),
            "positives": int(yt.sum()),
            "auc": roc_auc(yt, ys),
        }
    return out
