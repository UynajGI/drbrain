"""Link prediction models for the concept graph.

Binary classifiers predicting whether an edge forms between a concept pair:

* :class:`MLPLinkClassifier` — a dense NN (sklearn MLP) used for both the
  topological Baseline and the semantic Embeddings models;
* :class:`MixtureEnsemble` — weighted average of two classifiers' probabilities
  (the paper's best-performing hybrid, e.g. Baseline:Embeddings = 3:2).
"""

from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPClassifier

_DEFAULT_MLP = dict(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    max_iter=200,
    early_stopping=False,
    random_state=42,
)


class MLPLinkClassifier:
    """A dense neural-network binary classifier for link prediction."""

    def __init__(self, **mlp_kwargs):
        params = {**_DEFAULT_MLP, **mlp_kwargs}
        self.clf = MLPClassifier(**params)
        self._fitted = False

    def fit(self, x: np.ndarray, y: np.ndarray) -> MLPLinkClassifier:
        """Fit the classifier on feature matrix ``x`` and binary labels ``y``."""
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y)
        if len(np.unique(y)) < 2:
            # Degenerate single-class training set — record a constant predictor.
            self._constant = float(y[0]) if len(y) else 0.0
            self._fitted = True
            return self
        self.clf.fit(x, y)
        self._fitted = True
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return P(edge) for each row of ``x``."""
        x = np.asarray(x, dtype=np.float32)
        if getattr(self, "_constant", None) is not None:
            return np.full(len(x), self._constant, dtype=np.float32)
        return self.clf.predict_proba(x)[:, 1].astype(np.float32)


class MixtureEnsemble:
    """Weighted probability average of two link classifiers.

    Args:
        model_a: First classifier (e.g. Baseline / GNN).
        model_b: Second classifier (e.g. Embeddings).
        weight_a: Weight for ``model_a`` (``1 - weight_a`` goes to ``model_b``).
    """

    def __init__(
        self, model_a: MLPLinkClassifier, model_b: MLPLinkClassifier, weight_a: float = 0.6
    ):
        self.model_a = model_a
        self.model_b = model_b
        self.weight_a = float(weight_a)

    def fit(self, xa: np.ndarray, xb: np.ndarray, y: np.ndarray) -> MixtureEnsemble:
        """Fit both constituent classifiers on their respective feature matrices."""
        self.model_a.fit(xa, y)
        self.model_b.fit(xb, y)
        return self

    def predict_proba(self, xa: np.ndarray, xb: np.ndarray) -> np.ndarray:
        """Weighted mixture of the two classifiers' P(edge)."""
        pa = self.model_a.predict_proba(xa)
        pb = self.model_b.predict_proba(xb)
        return (self.weight_a * pa + (1 - self.weight_a) * pb).astype(np.float32)
