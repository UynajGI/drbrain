"""Tests for BM25 full-text search."""
import tempfile
from pathlib import Path
from brbrain.storage.database import Database
from brbrain.query.bm25 import BM25Search, build_bm25_index

def test_bm25_search_finds_title():
    """BM25 search finds papers by title terms."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Attention Is All You Need", 2017, "uploaded")
        db.insert_paper("p2", "BERT: Pre-training of Deep Bidirectional Transformers", 2018, "uploaded")
        db.insert_paper("p3", "Graph Neural Networks for Text Classification", 2019, "uploaded")
        db.commit()

        index = build_bm25_index(db)
        results = index.search("attention")
        assert len(results) >= 1
        assert results[0]["local_id"] == "p1"

def test_bm25_search_finds_concept_labels():
    """BM25 search finds concepts by label."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Paper One", 2020, "uploaded")
        db.insert_concept("p1", "Method", "transformer architecture", 0.95)
        db.insert_concept("p1", "Problem", "long range dependency", 0.9)
        db.commit()

        index = build_bm25_index(db)
        results = index.search("transformer")
        assert len(results) >= 1
        assert "transformer" in results[0]["label"].lower()

def test_bm25_type_filter():
    """BM25 search respects type filter."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Paper One", 2020, "uploaded")
        db.insert_concept("p1", "Method", "transformer", 0.95)
        db.insert_concept("p1", "Problem", "attention bottleneck", 0.9)
        db.commit()

        index = build_bm25_index(db)
        results = index.search("transformer", type_filter="Method")
        assert len(results) >= 1
        assert results[0]["type"] == "Method"

        results = index.search("transformer", type_filter="Problem")
        assert len(results) == 0

def test_bm25_empty_query():
    """BM25 search returns empty for query with no matches."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Deep Learning", 2015, "uploaded")
        db.commit()
        index = build_bm25_index(db)
        results = index.search("quantum computing")
        assert len(results) == 0
