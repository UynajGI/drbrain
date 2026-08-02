"""Tests for concept embeddings and the UMAP concept map."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from drbrain.concept_graph.embeddings import (
    aggregate_vectors,
    compute_concept_embeddings,
    load_concept_embeddings,
    nearest_neighbors,
)
from drbrain.concept_graph.map import export_html, umap_project
from drbrain.storage.database import Database


def _tmp_db() -> tuple[Database, tempfile.TemporaryDirectory]:
    td = tempfile.TemporaryDirectory()
    return Database(Path(td.name) / "test.db"), td


def _seed_nodes(db: Database, labels: list[str]) -> None:
    for label in labels:
        db.upsert_concept_node(label, doc_freq=3, word_count=len(label.split()))
    db.conn.commit()


def test_aggregate_vectors_normalizes() -> None:
    avg = aggregate_vectors([[1.0, 0.0], [1.0, 0.0]])
    assert np.allclose(avg, [1.0, 0.0])
    avg2 = aggregate_vectors([[1.0, 0.0], [0.0, 1.0]])
    assert np.isclose(np.linalg.norm(avg2), 1.0)


def test_aggregate_vectors_empty() -> None:
    assert aggregate_vectors([]).size == 0


def test_compute_concept_embeddings_basic() -> None:
    db, td = _tmp_db()
    try:
        _seed_nodes(db, ["alpha", "beta"])
        fake = lambda texts: [[1.0, 0.0] for _ in texts]  # noqa: E731
        count = compute_concept_embeddings(db, embed_fn=fake, model_name="fake")
        assert count == 2
        loaded = load_concept_embeddings(db)
        assert set(loaded) == {"alpha", "beta"}
        assert loaded["alpha"].shape == (2,)
    finally:
        db.close()
        td.cleanup()


def test_compute_concept_embeddings_provider_disabled() -> None:
    db, td = _tmp_db()
    try:
        _seed_nodes(db, ["alpha"])
        count = compute_concept_embeddings(db, embed_fn=lambda texts: [])
        assert count == 0
    finally:
        db.close()
        td.cleanup()


def test_nearest_neighbors_orders_by_similarity() -> None:
    db, td = _tmp_db()
    try:
        db.insert_concept_embedding("a", np.array([1.0, 0.0], dtype=np.float32).tobytes(), 2)
        db.insert_concept_embedding("b", np.array([0.9, 0.1], dtype=np.float32).tobytes(), 2)
        db.insert_concept_embedding("c", np.array([0.0, 1.0], dtype=np.float32).tobytes(), 2)
        db.conn.commit()
        nn = nearest_neighbors(db, "a", k=2)
        assert nn[0][0] == "b"  # b closest to a
        assert all(label != "a" for label, _ in nn)
    finally:
        db.close()
        td.cleanup()


def test_nearest_neighbors_unknown_label() -> None:
    db, td = _tmp_db()
    try:
        assert nearest_neighbors(db, "missing") == []
    finally:
        db.close()
        td.cleanup()


def test_umap_project_returns_2d_coords() -> None:
    db, td = _tmp_db()
    try:
        rng = np.random.default_rng(0)
        for i in range(6):
            vec = rng.normal(size=8).astype(np.float32)
            db.insert_concept_embedding(f"c{i}", vec.tobytes(), 8)
        db.conn.commit()
        coords = umap_project(db, n_components=2)
        assert len(coords) == 6
        assert all(len(xy) == 2 for xy in coords.values())
    finally:
        db.close()
        td.cleanup()


def test_export_html_writes_file() -> None:
    db, td = _tmp_db()
    try:
        coords = {"x": [0.0, 0.0], "y": [1.0, 1.0]}
        db.upsert_concept_node("x", doc_freq=5, word_count=1)
        db.upsert_concept_node("y", doc_freq=2, word_count=1)
        db.conn.commit()
        out = Path(td.name) / "map.html"
        path = export_html(db, out, coords=coords)
        content = path.read_text(encoding="utf-8")
        assert path.exists()
        assert "DrBrain Concept Map" in content
        assert '"label": "x"' in content
    finally:
        db.close()
        td.cleanup()
