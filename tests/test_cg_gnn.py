"""Tests for the optional GNN (GraphSAGE) link predictor.

The whole module is skipped when PyTorch is not installed (``drbrain[gnn]``).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from drbrain.concept_graph.gnn import (  # noqa: E402
    GNNLinkClassifier,
    build_node_topo_features,
    build_normalized_adjacency,
    is_available,
)
from drbrain.storage.database import Database  # noqa: E402


def _tmp_db() -> tuple[Database, tempfile.TemporaryDirectory]:
    td = tempfile.TemporaryDirectory()
    return Database(Path(td.name) / "test.db"), td


def _seed(db: Database) -> None:
    for src, dst, year in [("a", "b", 2015), ("b", "c", 2015), ("c", "d", 2015), ("a", "c", 2018)]:
        pid = f"p-{src}{dst}"
        db.insert_paper(pid, f"Paper {pid}", year, "extracted")
        db.insert_cooccurrence(src, dst, year, pid)
    db.conn.commit()


def test_is_available() -> None:
    assert is_available() is True


def test_build_node_topo_features_shape() -> None:
    db, td = _tmp_db()
    try:
        _seed(db)
        labels = ["a", "b", "c", "d"]
        feat = build_node_topo_features(db, labels, years=[2015, 2016])
        assert feat.shape == (4, 4)  # 2 metrics * 2 years
    finally:
        db.close()
        td.cleanup()


def test_build_normalized_adjacency_row_stochastic() -> None:
    db, td = _tmp_db()
    try:
        _seed(db)
        labels = ["a", "b", "c", "d"]
        adj = build_normalized_adjacency(db, labels, cutoff=2016)
        assert adj.shape == (4, 4)
        sums = adj.sum(dim=1).detach().numpy()
        assert np.allclose(sums, 1.0, atol=1e-5)
    finally:
        db.close()
        td.cleanup()


def test_gnn_fit_predict_proba() -> None:
    db, td = _tmp_db()
    try:
        _seed(db)
        train_pairs = [("a", "c"), ("a", "d"), ("b", "d")]
        y_train = np.array([1, 0, 0])
        gnn = GNNLinkClassifier(epochs=30, hidden_dim=16, embed_dim=8).fit(
            db, train_pairs, y_train, years=[2015, 2016], cutoff=2016
        )
        scores = gnn.predict_proba([("a", "c"), ("a", "d")])
        assert scores.shape == (2,)
        assert np.all((scores >= 0.0) & (scores <= 1.0))
    finally:
        db.close()
        td.cleanup()
