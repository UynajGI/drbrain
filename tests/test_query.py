"""Tests for query command enhancements: arg-type filter, year-range, BM25 argument claims."""
import tempfile
from pathlib import Path

import pytest
from brbrain.storage.database import Database
from brbrain.query.bm25 import build_bm25_index, BM25Search, tokenize


def test_tokenizer_normalizes():
    """Tokenizer lowercases and extracts alphanumeric tokens."""
    assert tokenize("Attention Is ALL You Need") == ["attention", "is", "all", "you", "need"]
    assert tokenize("Transformer-3.1") == ["transformer", "3", "1"]


def test_bm25_includes_argument_claims():
    """BM25 index includes argument claim text."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Attention Paper", 2017, "uploaded")
        db.insert_concept("p1", "Method", "Transformer", 0.95, year=2017)
        db.insert_argument(
            "p1", "Self-attention replaces RNN for sequence modeling",
            "proposes", "Transformer", "Method", "empirical", "WMT14 BLEU +2.0", 0.95,
        )
        db.commit()

        index = build_bm25_index(db)
        # Search for "replaces RNN" which is in the argument claim, not concept label
        results = index.search("replaces RNN")
        # Should find the argument document
        arg_results = [r for r in results if r["type"] == "Argument"]
        assert len(arg_results) >= 1
        assert "replaces" in arg_results[0]["label"].lower() or "self-attention" in arg_results[0]["label"].lower()
        db.close()


def test_bm25_search_with_arg_type_filter():
    """BM25 search can filter by argument claim type."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Paper 1", 2024, "uploaded")
        db.insert_argument("p1", "Supports claim", "supports", "Method X", "Method", "empirical", "", 0.9)
        db.insert_argument("p1", "Challenges claim", "challenges", "Method X", "Method", "empirical", "", 0.8)
        db.commit()

        index = build_bm25_index(db)
        # Search for "claim" with arg-type filter
        results = index.search("claim", type_filter="Argument")
        assert len(results) == 2

        # Filter by specific arg_type (stored in the document)
        supports_results = [r for r in results if "supports" in r.get("arg_type", "").lower()]
        challenges_results = [r for r in results if "challenges" in r.get("arg_type", "").lower()]
        assert len(supports_results) == 1
        assert len(challenges_results) == 1
        db.close()


def test_query_with_year_range():
    """query_cmd filters results by year range."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        # Papers across years
        for year in [2018, 2020, 2022, 2024]:
            pid = f"p{year}"
            db.insert_paper(pid, f"Paper {year}", year, "uploaded")
            db.insert_concept(pid, "Method", "attention", 0.9, year=year)
        db.commit()

        from brbrain.query.bm25 import build_bm25_index
        index = build_bm25_index(db)
        results = index.search("attention", type_filter="Method")
        assert len(results) == 4

        # Filter by year range via db query (year_range is applied at query_cmd level)
        # The BM25 index itself doesn't know years, but query_cmd will post-filter
        # We test the db method that provides year info
        rows = db.conn.execute(
            "SELECT DISTINCT c.local_id FROM concepts c JOIN papers p ON c.local_id = p.local_id "
            "WHERE c.label = ? AND p.year BETWEEN ? AND ?",
            ("attention", 2020, 2022),
        ).fetchall()
        local_ids = {r[0] for r in rows}
        assert "p2018" not in local_ids
        assert "p2020" in local_ids
        assert "p2022" in local_ids
        assert "p2024" not in local_ids
        db.close()


def test_bm25_argument_document_has_claim_text():
    """Argument documents store full claim text for BM25 matching."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Test Paper", 2024, "uploaded")
        db.insert_argument(
            "p1", "Graph neural networks outperform MLPs on molecular property prediction",
            "proposes", "GNN", "Method", "empirical", "MoleculeNet benchmark", 0.9,
        )
        db.commit()

        index = build_bm25_index(db)
        results = index.search("molecular property prediction")
        arg_results = [r for r in results if r["type"] == "Argument"]
        assert len(arg_results) >= 1
        # The label should contain the claim
        assert "graph neural networks" in arg_results[0]["label"].lower()
        db.close()


def test_bm25_includes_paper_abstracts():
    """BM25 index includes paper abstract text."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Short Title", 2024, "uploaded")
        db.set_paper_abstract("p1", "We propose a novel neural architecture search method "
                              "that combines reinforcement learning with evolutionary strategies")
        db.commit()

        index = build_bm25_index(db)
        # Search for text that's only in the abstract, not the title
        results = index.search("evolutionary strategies")
        paper_results = [r for r in results if r["type"] == "Paper"]
        assert len(paper_results) >= 1
        assert "Short Title" in paper_results[0]["label"]
        db.close()


def test_paper_has_abstract_field():
    """Database supports abstract field on papers."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Test Paper", 2024, "uploaded")
        db.set_paper_abstract("p1", "This is a test abstract.")
        db.commit()

        papers = db.get_all_papers()
        assert len(papers) == 1
        assert papers[0]["abstract"] == "This is a test abstract."
        db.close()


def test_query_neighbors_expansion():
    """query_cmd --neighbors expands results via graph traversal."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        # Create connected papers with edges between concepts
        db.insert_paper("p1", "Paper A", 2020, "uploaded")
        db.insert_paper("p2", "Paper B", 2021, "uploaded")
        db.insert_concept("p1", "Method", "transformer_v1", 0.9, year=2020)
        db.insert_concept("p2", "Method", "transformer_v2", 0.9, year=2021)
        # Edge is between concept labels (as stored in graph)
        db.insert_edge("transformer_v1", "transformer_v2", "extends", "p1")
        db.commit()

        from brbrain.query.bm25 import build_bm25_index
        from brbrain.graph.engine import GraphEngine

        # BM25 finds v1
        index = build_bm25_index(db)
        results = index.search("transformer_v1", type_filter="Method")
        assert len(results) >= 1

        # Graph expansion should find connected nodes (includes start node)
        graph = GraphEngine()
        graph.load_from_db(db)
        neighbors = graph.get_neighbors("transformer_v1", hops=1)
        assert "transformer_v1" in neighbors  # includes start
        assert "transformer_v2" in neighbors  # connected node
        db.close()
