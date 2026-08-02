"""Tests for research-direction recommendation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from drbrain.concept_graph.recommend import llm_curation, own_concepts, recommend_combinations
from drbrain.storage.database import Database


def _tmp_db() -> tuple[Database, tempfile.TemporaryDirectory]:
    td = tempfile.TemporaryDirectory()
    return Database(Path(td.name) / "test.db"), td


def _emb(vec: list[float]) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def _seed(db: Database) -> None:
    # Alice's paper has solar cell + perovskite; Bob's has battery.
    db.insert_paper("p1", "Solar study", 2020, "extracted", authors="Alice Smith; Carol")
    db.insert_paper("p2", "Battery study", 2021, "extracted", authors="Bob Jones")
    for term in ("solar cell", "perovskite"):
        db.insert_paper_term("p1", term, kind="keyword")
    db.insert_paper_term("p2", "battery", kind="keyword")
    # concept nodes
    for label in ("solar cell", "perovskite", "battery", "graphene", "stability"):
        db.upsert_concept_node(label, doc_freq=3, word_count=len(label.split()))
    # known co-occurrence: solar cell — perovskite (Alice's own concepts linked)
    db.insert_cooccurrence("solar cell", "perovskite", 2020, "p1")
    # embeddings
    db.insert_concept_embedding("solar cell", _emb([1.0, 0.0]), 2)
    db.insert_concept_embedding("perovskite", _emb([0.9, 0.1]), 2)
    db.insert_concept_embedding("graphene", _emb([0.8, 0.2]), 2)
    db.insert_concept_embedding("stability", _emb([0.7, 0.3]), 2)
    db.insert_concept_embedding("battery", _emb([0.0, 1.0]), 2)
    db.conn.commit()


def test_own_concepts_intersects_known_nodes() -> None:
    db, td = _tmp_db()
    try:
        _seed(db)
        c_own = own_concepts(db, "Alice")
        assert c_own == {"solar cell", "perovskite"}
        assert own_concepts(db, "Nobody") == set()
    finally:
        db.close()
        td.cleanup()


def test_recommend_excludes_own_and_unrelated() -> None:
    db, td = _tmp_db()
    try:
        _seed(db)
        result = recommend_combinations(db, "Alice", top_k=10, sim_min=0.15, sim_max=0.99)
        suggested = {s["concept"] for s in result["own_x_other"]}
        # related concepts suggested
        assert "graphene" in suggested
        assert "stability" in suggested
        # own concepts and unrelated (battery ~ cosine 0) excluded
        assert "solar cell" not in suggested
        assert "perovskite" not in suggested
        assert "battery" not in suggested
    finally:
        db.close()
        td.cleanup()


def test_recommend_many_own_section() -> None:
    db, td = _tmp_db()
    try:
        _seed(db)
        result = recommend_combinations(db, "Alice", top_k=10, sim_min=0.15, sim_max=0.99)
        many = {m["concept"] for m in result["many_own_x_other"]}
        # graphene is similar to both own concepts -> related_own_count >= 2
        assert "graphene" in many
    finally:
        db.close()
        td.cleanup()


def test_recommend_hub_filter() -> None:
    db, td = _tmp_db()
    try:
        _seed(db)
        # graphene has doc_freq 3; excluding hubs above 2 removes it
        result = recommend_combinations(
            db, "Alice", top_k=10, sim_min=0.15, sim_max=0.99, max_hub_freq=2
        )
        suggested = {s["concept"] for s in result["own_x_other"]}
        assert "graphene" not in suggested
    finally:
        db.close()
        td.cleanup()


def test_llm_curation_graceful_without_models() -> None:
    assert llm_curation([{"concept": "x", "score": 0.5}], models=[]) == ""
    assert llm_curation([], models=[{"model": "m"}]) == ""
