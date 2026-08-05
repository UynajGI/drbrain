"""Tests for the interactive concept map exporter (density + communities).

These tests avoid running UMAP (and thus torch) by passing precomputed
coordinates through ``export_html(coords=...)``; only the analysis and
HTML-rendering pipeline is exercised.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from drbrain.concept_graph.map import (
    _community_names,
    _render_html,
    density_field,
    detect_communities,
    export_html,
)


def _two_clusters(n_dense: int = 600, n_sparse: int = 60) -> np.ndarray:
    """Two well-separated Gaussian clusters (dense blob + sparse blob)."""
    rng = np.random.default_rng(7)
    dense = rng.normal(0.0, 0.12, size=(n_dense, 2))
    sparse = rng.normal(6.0, 0.9, size=(n_sparse, 2))
    return np.vstack([dense, sparse])


def test_density_field_high_for_dense_region():
    xy = _two_clusters()
    dens = density_field(xy, bins=96)
    # Dense cluster center should be much denser than sparse-cluster points.
    assert float(dens[:600].mean()) > float(dens[600:].mean()) * 2
    assert dens.min() >= 0.0 and dens.max() <= 1.0


def test_density_field_empty():
    assert density_field(np.zeros((0, 2))).shape == (0,)


def test_detect_communities_two_blobs():
    xy = _two_clusters()
    # max_frac=1.0 disables recursive splitting: the two well-separated blobs
    # must stay as-is under the base DBSCAN pass (recursive splitting only
    # kicks in for the dense centre of a real UMAP layout).
    labels, communities = detect_communities(xy, top_k=15, max_frac=1.0)
    assert len(communities) == 2
    assert sorted(c["id"] for c in communities) == [0, 1]
    # Dense blob is almost entirely one community; the sparse blob (minus a few
    # DBSCAN noise points) must map to the *other* community id.
    assert (labels[:600] == 0).mean() > 0.9
    sparse_clean = {int(lab) for lab in labels[600:] if lab != -1}
    assert sparse_clean == {1}


def test_detect_communities_folds_small_clusters_into_other():
    rng = np.random.default_rng(3)
    big = rng.normal(0.0, 0.15, size=(400, 2))
    tiny = rng.normal(8.0, 0.15, size=(5, 2))
    xy = np.vstack([big, tiny])
    labels, communities = detect_communities(xy, top_k=1, max_frac=1.0)
    # Only the big cluster survives as the top community; the tiny cluster
    # folds into -1. Recursive DBSCAN may split the big blob and re-mark
    # several sub-clusters/noise points as -1 (only top_k=1 is kept).
    assert [c["id"] for c in communities] == [0]
    assert set(labels[400:]) == {-1}
    assert (labels[:400] == 0).mean() > 0.8


def test_community_names_use_top_frequency_labels():
    labels = [f"concept_{i}" for i in range(12)]
    freq = np.asarray([float(i) for i in range(12)], dtype=np.float64)
    comm_labels = np.asarray([0] * 6 + [1] * 6, dtype=np.int64)
    communities = [{"id": 0, "size": 6}, {"id": 1, "size": 6}]
    names = _community_names(labels, freq, comm_labels, communities, top_concepts=2)
    # Community 0 = first 6 labels (freq 0..5), community 1 = last 6 (freq 6..11).
    assert "concept_5 · concept_4" in names[0]
    assert "concept_11 · concept_10" in names[1]


def test_export_html_embeds_sigma_and_communities(tmp_db, tmp_path):
    for label, freq in [("llm", 50), ("retrieval", 30), ("graph", 20), ("noise", 1)]:
        tmp_db.upsert_concept_node(label, doc_freq=freq)
    # Two tight blobs so recursive DBSCAN finds 2 communities.
    rng = np.random.default_rng(11)
    coords: dict[str, list[float]] = {}
    for i, (cx, cy, n) in enumerate([(0.0, 0.0, 40), (5.0, 5.0, 30)]):
        for j in range(n):
            coords[f"c{i}_{j}"] = [
                float(cx + rng.normal(0, 0.15)),
                float(cy + rng.normal(0, 0.15)),
            ]
    coords["noise"] = [20.0, 20.0]

    out = Path(tmp_path) / "map.html"
    path = export_html(tmp_db, out, coords=coords, top_communities=2, density_bins=96)
    html = path.read_text(encoding="utf-8")
    assert "new graphology.Graph()" in html
    assert "new Sigma(" in html
    assert "社区 · 点击开关" in html  # clickable community legend
    assert "const DATA = " in html
    assert "Plotly" not in html  # sigma.js renderer, not Plotly
    assert path.stat().st_size > 100_000  # embedded sigma.js + graphology.js


def test_render_html_empty_is_valid():
    html = _render_html([], [])
    assert "new graphology.Graph()" in html
    assert "new Sigma(" in html
    assert '"nodes": []' in html
