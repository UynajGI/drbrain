"""Tests for BM25 search with confidence filtering."""
import tempfile
from pathlib import Path
from drbrain.storage.database import Database
from drbrain.query.bm25 import BM25Search, build_bm25_index


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
        db.insert_argument("p1", "transformers scale well", "supports", "transformer", "Method", confidence=0.85)
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
