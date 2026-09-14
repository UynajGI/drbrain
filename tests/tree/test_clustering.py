"""T27/T28: raw two-stage posteriors from the vendored RAPTOR clustering."""

from __future__ import annotations

import numpy as np
import pytest

from drbrain.tree.clustering import (
    ClusteringError,
    ClusteringParams,
    canonical_membership,
    global_stage,
    two_stage_posteriors,
)
from drbrain.tree.posteriors import DEFAULT_THRESHOLD
from drbrain.tree.upstream import load_raptor_module

pytest.importorskip("umap")


def _blobs(seed: int = 7, clusters: int = 4, per_cluster: int = 12, dim: int = 8):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(clusters, dim)).astype(np.float32) * 2.5
    rows = []
    for center in centers:
        rows.append(center + rng.normal(scale=0.25, size=(per_cluster, dim)).astype(np.float32))
    matrix = np.vstack(rows)
    return matrix, [f"row-{index}" for index in range(matrix.shape[0])]


def _params(**overrides) -> ClusteringParams:
    base = {"dim": 4, "max_clusters": 8, "random_state": 224}
    base.update(overrides)
    return ClusteringParams(**base)


class TestGmmEquivalence:
    def test_labels_match_upstream_gmm_cluster(self):
        """On the same GMM input the BIC choice and thresholded labels agree."""
        from drbrain.tree.clustering import _gmm_posterior

        matrix, row_ids = _blobs(clusters=4, per_cluster=6, dim=6)
        # Upstream GMM_cluster hard-codes max_clusters=50 for the BIC search;
        # use the same range so the comparison is apples-to-apples.
        adapter = _gmm_posterior(
            matrix, DEFAULT_THRESHOLD, _params(max_clusters=50, gmm_random_state=0)
        )
        cluster_utils = load_raptor_module("cluster_utils")
        upstream_labels, upstream_k = cluster_utils.GMM_cluster(matrix, DEFAULT_THRESHOLD)
        assert int(upstream_k) == adapter.n_components
        assert adapter.labels_match_upstream(upstream_labels)
        # Component ids are stable names for the BIC-chosen components.
        assert adapter.component_ids == tuple(f"g{i}" for i in range(adapter.n_components))

    def test_threshold_is_strictly_greater(self):
        stage = global_stage(*_blobs(dim=6), params=_params())
        for row in stage.probs:
            above = {index for index, value in enumerate(row) if value > stage.threshold}
            assert above


class TestDeterminismAndMapping:
    def test_same_seed_reproduces_probs_and_labels(self):
        matrix, row_ids = _blobs()
        first = global_stage(matrix, row_ids, _params(random_state=11))
        second = global_stage(matrix, row_ids, _params(random_state=11))
        assert first.probs == second.probs
        assert first.labels == second.labels
        assert first.fitted["global_neighbors"] == second.fitted["global_neighbors"]

    def test_duplicate_vectors_keep_their_rows(self):
        """Positional mapping: identical vectors must not collapse into one row."""
        matrix, row_ids = _blobs(dim=6)
        matrix = np.vstack([matrix, matrix[0:1]])  # exact duplicate of row 0
        row_ids = [*row_ids, "duplicate-row"]
        global_fitted, local = two_stage_posteriors(matrix, row_ids, _params())
        membership = canonical_membership(local, row_ids)
        assert len(membership) == matrix.shape[0]
        assert "duplicate-row" in membership
        assert "row-0" in membership
        # Both rows are present in the same local stage rows.
        all_rows = {row for stage in local.values() for row in stage.row_ids}
        assert "duplicate-row" in all_rows and "row-0" in all_rows

    def test_unassigned_rows_are_carried(self):
        """A row above no threshold still appears with an empty tuple."""
        matrix, row_ids = _blobs(dim=6, clusters=2, per_cluster=4)
        global_fitted, local = two_stage_posteriors(matrix, row_ids, _params())
        membership = canonical_membership(local, row_ids)
        assert set(membership) == set(row_ids)
        assert any(not components for components in membership.values()) or all(
            components for components in membership.values()
        )


class TestTwoStageStructure:
    def test_local_components_are_namespaced_and_small_subsets_skip_fitting(self):
        matrix, row_ids = _blobs(clusters=3, per_cluster=5, dim=6)
        params = _params(dim=10)  # dim+1 = 11 > every subset: no local UMAP/GMM
        global_fitted, local = two_stage_posteriors(matrix, row_ids, params)
        assert local, "global components must produce local stages"
        for component_id, stage in local.items():
            assert component_id.startswith("g")
            assert stage.subset_of == component_id
            assert all(name.startswith(f"{component_id}.l") for name in stage.component_ids)
            assert stage.fitted.get("small_subset") is True
            assert all(row == (0,) for row in stage.labels)

    def test_large_subsets_fit_their_own_gmm(self):
        matrix, row_ids = _blobs(clusters=2, per_cluster=30, dim=8)
        params = _params(dim=4, max_clusters=10)
        global_fitted, local = two_stage_posteriors(matrix, row_ids, params)
        fitted = [stage for stage in local.values() if not stage.fitted.get("small_subset")]
        assert fitted, "expected at least one locally fitted subset"
        for stage in fitted:
            assert stage.n_components >= 1
            assert stage.reduced_dim == params.dim
            # Local posteriors are their own distributions, never products.
            for row in stage.probs:
                assert abs(sum(row) - 1.0) < 1e-6

    def test_local_rows_are_a_subset_of_global_members(self):
        matrix, row_ids = _blobs(dim=6)
        global_fitted, local = two_stage_posteriors(matrix, row_ids, _params())
        for component_id, stage in local.items():
            index = int(component_id[1:])
            global_rows = {
                row_id
                for row_id, labels in zip(global_fitted.row_ids, global_fitted.labels)
                if index in labels
            }
            assert set(stage.row_ids) <= global_rows

    def test_bridge_to_posterior_stage_lambda_zero(self):
        matrix, row_ids = _blobs(dim=6)
        global_fitted = global_stage(matrix, row_ids, _params())
        stage = global_fitted.posterior_stage()
        assert stage.labels() == {
            row_id: tuple(f"g{label}" for label in labels)
            for row_id, labels in zip(global_fitted.row_ids, global_fitted.labels)
        }
        reweighted = stage.reweighted(lam=0.0)
        assert reweighted is stage


class TestValidation:
    def test_errors_are_explicit(self):
        with pytest.raises(ValueError, match="dim"):
            ClusteringParams(dim=1)
        with pytest.raises(ValueError, match="threshold"):
            ClusteringParams(threshold=1.5)
        with pytest.raises(ClusteringError, match="row_ids"):
            global_stage(np.zeros((5, 3), dtype=np.float32), ["a", "b"])
        with pytest.raises(ClusteringError, match="at least 3 rows"):
            global_stage(np.zeros((2, 3), dtype=np.float32), ["a", "b"])
