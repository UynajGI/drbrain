"""Tests for temporal link prediction: features, dataset, models, evaluation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import networkx as nx
import numpy as np

from drbrain.concept_graph.dataset import feature_years, oversample_indices, temporal_pairs
from drbrain.concept_graph.eval import (
    precision_recall_at_k,
    previous_distance,
    roc_auc,
    stratify_by_dprev,
)
from drbrain.concept_graph.features import semantic_features, topo_features, yearly_subgraph
from drbrain.concept_graph.link_predict import MixtureEnsemble, MLPLinkClassifier
from drbrain.storage.database import Database


def _tmp_db() -> tuple[Database, tempfile.TemporaryDirectory]:
    td = tempfile.TemporaryDirectory()
    return Database(Path(td.name) / "test.db"), td


def _seed_graph(db: Database) -> None:
    # before (<=2016): a-b, b-c, c-d ; test window adds a-c (2018)
    for src, dst, year in [("a", "b", 2015), ("b", "c", 2015), ("c", "d", 2015), ("a", "c", 2018)]:
        db.insert_cooccurrence(src, dst, year, f"p-{src}{dst}")
    db.conn.commit()


# ── features ─────────────────────────────────────────────────────────────────


def test_yearly_subgraph_accumulates() -> None:
    db, td = _tmp_db()
    try:
        _seed_graph(db)
        g2016 = yearly_subgraph(db, 2016)
        assert set(g2016.edges()) == {("a", "b"), ("b", "c"), ("c", "d")}
        g2019 = yearly_subgraph(db, 2019)
        assert ("a", "c") in g2019.edges()
    finally:
        db.close()
        td.cleanup()


def test_topo_features_length() -> None:
    db, td = _tmp_db()
    try:
        _seed_graph(db)
        feats = topo_features(db, "a", "c", [2015, 2016])
        assert feats.shape == (8,)  # 4 metrics * 2 years
    finally:
        db.close()
        td.cleanup()


def test_semantic_features_concat() -> None:
    emb = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    feat = semantic_features("a", "b", emb)
    assert feat.shape == (4,)
    assert np.allclose(feat, [1.0, 0.0, 0.0, 1.0])


def test_semantic_features_zero_fill() -> None:
    emb = {"a": np.array([1.0, 0.0])}
    feat = semantic_features("a", "missing", emb, dim=2)
    assert np.allclose(feat, [1.0, 0.0, 0.0, 0.0])


# ── dataset ──────────────────────────────────────────────────────────────────


def test_temporal_pairs_emerging_positive() -> None:
    db, td = _tmp_db()
    try:
        _seed_graph(db)
        data = temporal_pairs(db, train_end=2016, test_end=2019, neg_ratio=2)
        assert ("a", "c") in data["positives"]
        # negatives must be unconnected in before-graph and not the positive
        for u, v in data["negatives"]:
            assert frozenset((u, v)) != frozenset(("a", "c"))
            assert not data["g_before"].has_edge(u, v)
    finally:
        db.close()
        td.cleanup()


def test_feature_years_window() -> None:
    assert feature_years(2019, window=5) == [2015, 2016, 2017, 2018, 2019]


def test_oversample_indices_fraction() -> None:
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    idx = oversample_indices(y, pos_fraction=0.5)
    frac = y[idx].mean()
    assert 0.4 <= frac <= 0.6


# ── models ───────────────────────────────────────────────────────────────────


def test_mlp_classifier_learns_separable() -> None:
    rng = np.random.default_rng(0)
    x_pos = rng.normal(loc=2.0, size=(30, 4))
    x_neg = rng.normal(loc=-2.0, size=(30, 4))
    x = np.vstack([x_pos, x_neg]).astype(np.float32)
    y = np.array([1] * 30 + [0] * 30)
    clf = MLPLinkClassifier(max_iter=300).fit(x, y)
    scores = clf.predict_proba(x)
    assert scores[:30].mean() > scores[30:].mean()
    assert roc_auc(y, scores) > 0.9


def test_mixture_ensemble_combines() -> None:
    rng = np.random.default_rng(1)
    xa = np.vstack([rng.normal(2, size=(20, 3)), rng.normal(-2, size=(20, 3))]).astype(np.float32)
    xb = np.vstack([rng.normal(1.5, size=(20, 2)), rng.normal(-1.5, size=(20, 2))]).astype(
        np.float32
    )
    y = np.array([1] * 20 + [0] * 20)
    ens = MixtureEnsemble(
        MLPLinkClassifier(max_iter=300), MLPLinkClassifier(max_iter=300), weight_a=0.6
    )
    ens.fit(xa, xb, y)
    scores = ens.predict_proba(xa, xb)
    assert scores.shape == (40,)
    assert roc_auc(y, scores) > 0.85


# ── evaluation ───────────────────────────────────────────────────────────────


def test_roc_auc_perfect_and_degenerate() -> None:
    assert roc_auc(np.array([0, 1]), np.array([0.1, 0.9])) == 1.0
    assert roc_auc(np.array([1, 1]), np.array([0.5, 0.5])) == 0.5


def test_precision_recall_at_k() -> None:
    y = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    prec, rec = precision_recall_at_k(y, scores, k=2)
    assert prec == 0.5  # top-2 = [pos, neg]
    assert rec == 0.5  # 1 of 2 positives


def test_previous_distance_and_stratify() -> None:
    g = nx.Graph()
    g.add_edges_from([("a", "b"), ("b", "c")])
    assert previous_distance(g, "a", "c") == 2
    assert previous_distance(g, "a", "z") is None
    pairs = [("a", "c"), ("a", "b")]
    y = np.array([1, 0])
    scores = np.array([0.9, 0.2])
    strat = stratify_by_dprev(pairs, y, scores, g)
    assert 2 in strat and strat[2]["positives"] == 1
