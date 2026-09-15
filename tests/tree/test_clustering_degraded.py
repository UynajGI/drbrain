"""GMM degeneracy ladder: float32 → float64+reg_covar → single component."""

from __future__ import annotations

import numpy as np

from drbrain.tree.clustering import ClusteringParams, _fit_gmm


def test_ladder_retries_on_float64_with_reg_covar(monkeypatch) -> None:
    calls = {"n": 0}

    class FakeClusterUtils:
        def get_optimal_clusters(self, matrix, max_clusters, random_state=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError("ill-defined empirical covariance")
            assert matrix.dtype == np.float64
            return 3

    import drbrain.tree.upstream as upstream

    monkeypatch.setattr(upstream, "load_raptor_module", lambda name: FakeClusterUtils())
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(30, 4)).astype(np.float32)
    gm, n_clusters, degraded = _fit_gmm(matrix, ClusteringParams())
    assert calls["n"] == 2
    assert n_clusters == 3
    assert degraded == "float64_reg_covar"
    assert gm.n_components == 3


def test_ladder_collapses_to_one_component_at_the_end(monkeypatch) -> None:
    class AlwaysFailing:
        def get_optimal_clusters(self, *args, **kwargs):
            raise ValueError("ill-defined empirical covariance")

    import drbrain.tree.upstream as upstream

    monkeypatch.setattr(upstream, "load_raptor_module", lambda name: AlwaysFailing())
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(8, 4)).astype(np.float32)
    gm, n_clusters, degraded = _fit_gmm(matrix, ClusteringParams())
    assert n_clusters == 1
    assert degraded == "single_component"
    assert gm.n_components == 1


def test_clean_fit_reports_no_degradation(monkeypatch) -> None:
    class CleanClusterUtils:
        def get_optimal_clusters(self, matrix, max_clusters, random_state=None):
            return 2

    import drbrain.tree.upstream as upstream

    monkeypatch.setattr(upstream, "load_raptor_module", lambda name: CleanClusterUtils())
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(24, 4)).astype(np.float32)
    _gm, n_clusters, degraded = _fit_gmm(matrix, ClusteringParams())
    assert n_clusters == 2
    assert degraded == ""
