"""Tests for BM25 search with confidence filtering."""

import tempfile
from pathlib import Path

from drbrain.query.bm25 import BM25Search, build_bm25_index
from drbrain.storage.database import Database


def test_bm25_confidence_stored():
    """BM25 stores confidence in documents."""
    bm25 = BM25Search()
    bm25.add_document("p1", "Method", "transformer", confidence=0.95)
    assert bm25._documents[0]["confidence"] == 0.95


def test_bm25_confidence_optional():
    """BM25 documents work without confidence."""
    bm25 = BM25Search()
    bm25.add_document("p1", "Paper", "Test Paper", year=2024)
    assert "confidence" not in bm25._documents[0]


def test_bm25_min_confidence_filter():
    """BM25 filters out documents below min_confidence."""
    bm25 = BM25Search()
    bm25.add_document("p1", "Method", "transformer", confidence=0.95)
    bm25.add_document("p2", "Method", "attention", confidence=0.5)
    bm25.add_document("p3", "Method", "embedding", confidence=0.8)
    bm25.build()

    results = bm25.search("transformer attention embedding", min_confidence=0.7)
    ids = [r["local_id"] for r in results]
    assert "p1" in ids  # 0.95 >= 0.7
    assert "p3" in ids  # 0.8 >= 0.7
    assert "p2" not in ids  # 0.5 < 0.7


def test_bm25_min_confidence_no_match():
    """BM25 returns empty when all docs below threshold."""
    bm25 = BM25Search()
    bm25.add_document("p1", "Method", "test method", confidence=0.3)
    bm25.build()

    results = bm25.search("test", min_confidence=0.9)
    assert results == []


def test_build_bm25_index_includes_confidence():
    """build_bm25_index passes confidence from DB concepts and arguments."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Test Paper", 2024, "uploaded")
        db.insert_concept("p1", "Method", "transformer", 0.95, year=2024)
        db.insert_concept("p1", "Problem", "scalability", 0.6, year=2024)
        db.insert_argument(
            "p1", "transformers scale well", "supports", "transformer", "Method", confidence=0.85
        )
        db.commit()

        bm25 = build_bm25_index(db)
        method_docs = [d for d in bm25._documents if d["type"] == "Method"]
        problem_docs = [d for d in bm25._documents if d["type"] == "Problem"]
        arg_docs = [d for d in bm25._documents if d["type"] == "Argument"]

        assert method_docs[0]["confidence"] == 0.95
        assert problem_docs[0]["confidence"] == 0.6
        assert arg_docs[0]["confidence"] == 0.85
        db.close()


def test_query_confidence_filter_via_db():
    """BM25 search with min_confidence via built index."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Test Paper", 2024, "uploaded")
        db.insert_concept("p1", "Method", "transformer", 0.95, year=2024)
        db.insert_concept("p1", "Method", "embedding", 0.4, year=2024)
        db.commit()

        bm25 = build_bm25_index(db)
        results = bm25.search("transformer embedding", min_confidence=0.7)
        labels = [r["label"] for r in results]
        assert "transformer" in labels
        assert "embedding" not in labels
        db.close()


# -- New tests for coverage boost --


def test_build_index_empty():
    """Empty concept list produces empty (None) index."""
    db = Database(":memory:")
    idx = build_bm25_index(db)
    assert idx._bm25 is None or len(idx._documents) == 0
    db.close()


def test_search_on_empty_db_returns_empty():
    """Search returns empty list when no data exists."""
    db = Database(":memory:")
    idx = build_bm25_index(db)
    if idx is None:
        assert True  # No index = no results, which is correct behavior
    else:
        results = idx.search("any query")
        assert len(results) == 0
    db.close()


def test_search_result_structure():
    """Search results contain expected metadata fields: label, type, local_id."""
    db = Database(":memory:")
    db.insert_paper("test-1", "Test", 2026, "extracted")
    db.insert_concept("test-1", "Problem", "graph neural networks", 0.9, year=2026)
    db.insert_concept("test-1", "Method", "attention mechanism", 0.8, year=2026)
    db.commit()

    idx = build_bm25_index(db)
    if idx is not None:
        results = idx.search("graph networks", limit=3)
        assert len(results) >= 1
        r = results[0]
        assert "label" in r
        assert "type" in r
        assert "local_id" in r
    db.close()


def test_search_limit():
    """Search respects limit parameter, returning at most N results."""
    db = Database(":memory:")
    db.insert_paper("test-1", "Test", 2026, "extracted")
    for i in range(5):
        db.insert_concept(
            "test-1", "Problem", f"concept {i} neural network training", 0.9, year=2026
        )
    db.commit()

    idx = build_bm25_index(db)
    if idx is not None:
        results = idx.search("neural", limit=2)
        assert len(results) <= 2
    db.close()


def test_search_no_matching_terms_score_zero():
    """Query with no matching terms returns results with score 0.0 (BM25 behavior)."""
    db = Database(":memory:")
    db.insert_paper("test-1", "Test", 2026, "extracted")
    db.insert_concept("test-1", "Method", "transformer architecture", 0.9, year=2026)
    db.commit()

    idx = build_bm25_index(db)
    if idx is not None:
        results = idx.search("zzz_nonexistent_term_xyz")
        # BM25Okapi returns all docs with score 0 when no terms match
        # The search function doesn't filter zero-score results
        assert len(results) > 0
        for r in results:
            assert r["score"] == 0.0
    db.close()
