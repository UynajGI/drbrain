"""Temporal train/test dataset construction for link prediction.

Splits the co-occurrence graph by publication year: edges accumulated up to
``train_end`` form the "before" graph (used for features — no leakage), and edges
that first appear in ``(train_end, test_end]`` are the positive (emerging) pairs.
Negative pairs are unconnected node pairs sampled from the before-graph.
"""

from __future__ import annotations

import random

import numpy as np

from drbrain.concept_graph.features import yearly_subgraph
from drbrain.storage.database import Database


def temporal_pairs(
    db: Database,
    train_end: int,
    test_end: int,
    *,
    neg_ratio: int = 1,
    seed: int = 42,
    labels: set[str] | None = None,
    exclude_pairs: set[frozenset] | None = None,
) -> dict:
    """Build positive (emerging) and negative node pairs for link prediction.

    Args:
        db: Database handle.
        train_end: Last year of the "before" graph (features may only use <= this).
        test_end: Last year of the test window (positives appear in (train_end, test_end]).
        neg_ratio: Number of negatives sampled per positive.
        seed: RNG seed.
        labels: Optional node whitelist.
        exclude_pairs: Optional set of ``frozenset`` pairs to never emit as
            negatives (used to keep test negatives disjoint from training).

    Returns:
        Dict with ``positives``, ``negatives`` (lists of ``(u, v)`` tuples),
        ``g_before`` (NetworkX graph) and ``nodes``.
    """
    rng = random.Random(seed)
    excluded = exclude_pairs or set()
    g_before = yearly_subgraph(db, train_end, labels=labels)
    g_test = yearly_subgraph(db, test_end, labels=labels)

    before_edges = {frozenset(e) for e in g_before.edges()}
    positives = sorted(
        {frozenset(e) for e in g_test.edges()} - before_edges,
        key=lambda s: tuple(sorted(s)),
    )
    positive_pairs = [tuple(sorted(p)) for p in positives]

    nodes = sorted(g_before.nodes())
    node_set = set(nodes)
    positive_lookup = set(positives)
    negatives: list[tuple[str, str]] = []
    target_neg = len(positive_pairs) * neg_ratio
    attempts = 0
    max_attempts = max(target_neg * 20, 100)
    while len(negatives) < target_neg and attempts < max_attempts and len(nodes) >= 2:
        attempts += 1
        u, v = rng.sample(nodes, 2)
        pair = frozenset((u, v))
        if pair in before_edges or pair in positive_lookup or pair in excluded:
            continue
        if u not in node_set or v not in node_set:
            continue
        negatives.append(tuple(sorted((u, v))))

    return {
        "positives": positive_pairs,
        "negatives": negatives,
        "g_before": g_before,
        "nodes": nodes,
    }


def feature_years(train_end: int, window: int = 5) -> list[int]:
    """Return the ascending list of slice years ending at ``train_end``."""
    start = train_end - window + 1
    return list(range(start, train_end + 1))


def oversample_indices(y: np.ndarray, pos_fraction: float = 0.3, seed: int = 42) -> np.ndarray:
    """Return training indices with positives oversampled to ``pos_fraction``.

    Positives are sampled with replacement; negatives without replacement (or
    with replacement if there are fewer negatives than required).

    Args:
        y: Binary label array.
        pos_fraction: Desired fraction of positives per epoch (paper uses 0.3).
        seed: RNG seed.

    Returns:
        An index array for one oversampled epoch.
    """
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return np.arange(len(y))

    n_pos = len(pos_idx)
    n_neg = int(round(n_pos * (1 - pos_fraction) / pos_fraction))
    pos_sample = rng.choice(pos_idx, size=n_pos, replace=True)
    neg_sample = rng.choice(neg_idx, size=n_neg, replace=len(neg_idx) < n_neg)
    indices = np.concatenate([pos_sample, neg_sample])
    rng.shuffle(indices)
    return indices
