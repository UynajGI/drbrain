"""Vendored RAPTOR clustering exposed as raw two-stage posteriors (T27/T28).

Upstream computes a GMM posterior and then throws it away, returning only
thresholded labels; its local-to-global remapping also matches float vectors
by equality, and its UMAP has no seed at all.  This adapter keeps the audited
algorithm and fixes exactly those integration defects:

* both stages return the *raw* ``predict_proba`` posterior plus the upstream
  thresholded labels, so ``lambda=0`` can be checked against upstream;
* BIC cluster selection and the strict ``>`` threshold keep upstream
  semantics (including the ``arange(1, min(50, N))`` candidate range);
* subsets of size ``<= dim + 1`` become a single local component exactly as
  upstream does, and local components are namespaced ``"<global>.<local>"``;
* members are propagated by *positional index* instead of float equality, so
  duplicate vectors cannot mis-map rows;
* UMAP and GMM receive explicit random states and the parameters used are
  recorded for the acceptance record (upstream seeds neither).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from drbrain.tree.posteriors import DEFAULT_THRESHOLD, PosteriorStage

#: Fixed version of the vendored algorithm this adapter targets.
UPSTREAM_RAPTOR_COMMIT = "7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767"


class ClusteringError(RuntimeError):
    """The vendored clustering step could not be executed."""


@dataclass(frozen=True)
class ClusteringParams:
    dim: int = 10
    threshold: float = DEFAULT_THRESHOLD
    max_clusters: int = 50
    global_neighbors: int | None = None
    local_neighbors: int = 10
    metric: str = "cosine"
    random_state: int = 224
    gmm_random_state: int = 0

    def __post_init__(self) -> None:
        if self.dim < 2:
            raise ValueError("dim must be >= 2")
        if not (0.0 < self.threshold < 1.0):
            raise ValueError("threshold must lie in (0, 1)")
        if self.max_clusters < 2:
            raise ValueError("max_clusters must be >= 2")


@dataclass
class FittedStage:
    """One fitted stage: raw posterior, upstream labels, and its row map."""

    stage: str
    row_ids: tuple[str, ...]
    component_ids: tuple[str, ...]
    probs: tuple[tuple[float, ...], ...]
    labels: tuple[tuple[int, ...], ...]
    n_components: int
    threshold: float
    subset_of: str | None = None
    reduced_dim: int = 0
    fitted: dict[str, Any] = field(default_factory=dict)

    def posterior_stage(
        self,
        lam: float = 0.0,
        affinity: tuple[tuple[float, ...], ...] | None = None,
    ) -> PosteriorStage:
        """Bridge into the frozen T04 protocol (reweighting/thresholding)."""
        return PosteriorStage(
            stage="global" if self.stage == "global" else "local",
            row_ids=self.row_ids,
            component_ids=self.component_ids,
            probs=self.probs,
            threshold=self.threshold,
            lam=lam,
            affinity=affinity,
            subset_of=self.subset_of,
        )

    def labels_match_upstream(self, upstream_labels: Sequence[Sequence[int]]) -> bool:
        return all(
            tuple(sorted(int(v) for v in row)) == tuple(sorted(mine))
            for row, mine in zip(upstream_labels, self.labels)
        )

    def summary(self) -> dict[str, Any]:
        assigned = sum(1 for row in self.labels if row)
        return {
            "stage": self.stage,
            "rows": len(self.row_ids),
            "components": self.n_components,
            "threshold": self.threshold,
            "subset_of": self.subset_of,
            "reduced_dim": self.reduced_dim,
            "assigned": assigned,
            "unassigned": len(self.row_ids) - assigned,
            "fitted": dict(self.fitted),
        }


def _gmm_posterior(
    embeddings: np.ndarray, threshold: float, params: ClusteringParams
) -> FittedStage:
    """Raw GMM posterior + upstream thresholded labels over the BIC choice."""
    from sklearn.mixture import GaussianMixture

    from drbrain.tree.upstream import load_raptor_module

    cluster_utils = load_raptor_module("cluster_utils")
    n_clusters = int(
        cluster_utils.get_optimal_clusters(
            embeddings, params.max_clusters, random_state=params.random_state
        )
    )
    gm = GaussianMixture(n_components=n_clusters, random_state=params.gmm_random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = tuple(tuple(int(idx) for idx in np.where(prob > threshold)[0]) for prob in probs)
    return FittedStage(
        stage="global",
        row_ids=tuple(str(index) for index in range(len(embeddings))),
        component_ids=tuple(f"g{index}" for index in range(n_clusters)),
        probs=tuple(tuple(float(value) for value in row) for row in probs),
        labels=labels,
        n_components=n_clusters,
        threshold=threshold,
        fitted={
            "bic_clusters": n_clusters,
            "gmm_random_state": params.gmm_random_state,
            "upstream_commit": UPSTREAM_RAPTOR_COMMIT,
        },
    )


def _umap_reduce(
    embeddings: np.ndarray, dim: int, n_neighbors: int, params: ClusteringParams
) -> np.ndarray:
    import umap

    reducer = umap.UMAP(
        n_neighbors=int(n_neighbors),
        n_components=int(dim),
        metric=params.metric,
        random_state=params.random_state,
    )
    return reducer.fit_transform(embeddings)


def global_stage(
    embeddings: np.ndarray,
    row_ids: Sequence[str],
    params: ClusteringParams | None = None,
) -> FittedStage:
    """Global UMAP + BIC/GMM stage over the whole frontier."""
    params = params or ClusteringParams()
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ClusteringError("embeddings must be a 2-D matrix")
    if len(row_ids) != matrix.shape[0]:
        raise ClusteringError("row_ids must match embeddings rows")
    if matrix.shape[0] < 3:
        raise ClusteringError("global clustering needs at least 3 rows")
    dim = int(min(params.dim, matrix.shape[0] - 2))
    neighbors = (
        params.global_neighbors
        if params.global_neighbors is not None
        else int((matrix.shape[0] - 1) ** 0.5)
    )
    neighbors = max(2, min(neighbors, matrix.shape[0] - 1))
    reduced = _umap_reduce(matrix, dim, neighbors, params)
    stage = _gmm_posterior(reduced, params.threshold, params)
    return FittedStage(
        stage="global",
        row_ids=tuple(str(row_id) for row_id in row_ids),
        component_ids=stage.component_ids,
        probs=stage.probs,
        labels=stage.labels,
        n_components=stage.n_components,
        threshold=stage.threshold,
        reduced_dim=dim,
        fitted={**stage.fitted, "global_neighbors": neighbors, "metric": params.metric},
    )


def local_stages(
    embeddings: np.ndarray,
    row_ids: Sequence[str],
    global_fitted: FittedStage,
    params: ClusteringParams | None = None,
) -> dict[str, FittedStage]:
    """Independent local stages inside each global component (upstream order)."""
    params = params or ClusteringParams()
    matrix = np.asarray(embeddings, dtype=np.float32)
    if len(row_ids) != matrix.shape[0]:
        raise ClusteringError("row_ids must match embeddings rows")
    result: dict[str, FittedStage] = {}
    global_labels = global_fitted.labels
    for component_index, component_id in enumerate(global_fitted.component_ids):
        member_rows = [
            index for index, labels in enumerate(global_labels) if component_index in labels
        ]
        if not member_rows:
            continue
        subset = matrix[member_rows]
        subset_ids = tuple(str(row_ids[index]) for index in member_rows)
        if len(member_rows) <= params.dim + 1:
            # Upstream: small subsets stay one local component without fitting.
            probs = tuple((1.0,) for _ in member_rows)
            labels = tuple((0,) for _ in member_rows)
            result[component_id] = FittedStage(
                stage="local",
                row_ids=subset_ids,
                component_ids=(f"{component_id}.l0",),
                probs=probs,
                labels=labels,
                n_components=1,
                threshold=params.threshold,
                subset_of=component_id,
                reduced_dim=0,
                fitted={"small_subset": True, "size": len(member_rows)},
            )
            continue
        reduced = _umap_reduce(subset, params.dim, params.local_neighbors, params)
        fitted = _gmm_posterior(reduced, params.threshold, params)
        result[component_id] = FittedStage(
            stage="local",
            row_ids=subset_ids,
            component_ids=tuple(f"{component_id}.l{index}" for index in range(fitted.n_components)),
            probs=fitted.probs,
            labels=fitted.labels,
            n_components=fitted.n_components,
            threshold=fitted.threshold,
            subset_of=component_id,
            reduced_dim=params.dim,
            fitted={
                **fitted.fitted,
                "size": len(member_rows),
                "local_neighbors": params.local_neighbors,
            },
        )
    return result


def two_stage_posteriors(
    embeddings: np.ndarray,
    row_ids: Sequence[str],
    params: ClusteringParams | None = None,
) -> tuple[FittedStage, dict[str, FittedStage]]:
    """Global stage plus every local stage, each keeping its own posterior."""
    global_fitted = global_stage(embeddings, row_ids, params)
    return global_fitted, local_stages(embeddings, row_ids, global_fitted, params)


def canonical_membership(
    local_map: dict[str, FittedStage],
    row_ids: Sequence[str] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Row id -> namespaced local components; unassigned rows stay present."""
    membership: dict[str, list[str]] = {str(row_id): [] for row_id in (row_ids or ())}
    for _component_id, stage in local_map.items():
        for row_id, labels in zip(stage.row_ids, stage.labels):
            entry = membership.setdefault(row_id, [])
            for label in labels:
                entry.append(stage.component_ids[label])
    return {row_id: tuple(components) for row_id, components in membership.items()}
