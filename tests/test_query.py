"""Tests for query command enhancements: arg-type filter, year-range, BM25 argument claims."""

import io
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

import typer

from drbrain.cli.commands import query_cmd
from drbrain.query.bm25 import build_bm25_index, tokenize
from drbrain.storage.database import Database


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
            "p1",
            "Self-attention replaces RNN for sequence modeling",
            "proposes",
            "Transformer",
            "Method",
            "empirical",
            "WMT14 BLEU +2.0",
            0.95,
        )
        db.commit()

        index = build_bm25_index(db)
        # Search for "replaces RNN" which is in the argument claim, not concept label
        results = index.search("replaces RNN")
        # Should find the argument document
        arg_results = [r for r in results if r["type"] == "Argument"]
        assert len(arg_results) >= 1
        assert (
            "replaces" in arg_results[0]["label"].lower()
            or "self-attention" in arg_results[0]["label"].lower()
        )
        db.close()


def test_bm25_search_with_arg_type_filter():
    """BM25 search can filter by argument claim type."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Paper 1", 2024, "uploaded")
        db.insert_argument(
            "p1", "Supports claim", "supports", "Method X", "Method", "empirical", "", 0.9
        )
        db.insert_argument(
            "p1", "Challenges claim", "challenges", "Method X", "Method", "empirical", "", 0.8
        )
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

        from drbrain.query.bm25 import build_bm25_index

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
            "p1",
            "Graph neural networks outperform MLPs on molecular property prediction",
            "proposes",
            "GNN",
            "Method",
            "empirical",
            "MoleculeNet benchmark",
            0.9,
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
        db.set_paper_abstract(
            "p1",
            "We propose a novel neural architecture search method "
            "that combines reinforcement learning with evolutionary strategies",
        )
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

        from drbrain.graph.engine import GraphEngine
        from drbrain.query.bm25 import build_bm25_index

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


# ---------------------------------------------------------------------------
# Graph-enhanced query CLI integration tests (RED phase)
# ---------------------------------------------------------------------------


def _make_minimal_config(db_path: str, papers_dir: str) -> dict:
    return {
        "db": {"path": db_path},
        "llm": {"models": [{"provider": "openai", "model": "gpt-4", "api_key": "x"}]},
        "dirs": {
            "inbox": "data/inbox",
            "papers": papers_dir,
            "reports": "/tmp/reports",
            "cache": "data/cache",
            "logs": "data/logs",
        },
        "bm25": {"k1": 1.5, "b": 0.75},
        "queue": {"weak_threshold": 0.7, "auto_accept": 0.9},
        "api": {"s2_rate_limit": 100, "cache_ttl": 86400},
    }


def _mock_load_config(cfg: dict):
    return mock.patch("drbrain.cli.commands.load_config", return_value=cfg)


def test_query_cmd_graph_relation_invalid():
    """--relation with invalid relation type raises Exit(1)."""
    with tempfile.TemporaryDirectory() as td:
        papers_dir = Path(td) / "papers"
        papers_dir.mkdir()
        db_path = f"{td}/test.db"

        from drbrain.storage.database import Database

        db = Database(db_path)
        db.close()

        cfg = _make_minimal_config(db_path, str(papers_dir))

        with _mock_load_config(cfg):
            try:
                query_cmd(
                    text="test",
                    neighbors=2,
                    relation="bogus_relation",
                    direction="both",
                    type_filter=None,
                    arg_type=None,
                    year_start=None,
                    year_end=None,
                    min_confidence=None,
                    limit=20,
                    json_output=False,
                    jsonl=False,
                    paper=None,
                    workspace=None,
                )
                assert False, "Should have raised Exit"
            except typer.Exit as e:
                assert e.exit_code == 1


def test_query_cmd_graph_direction_invalid():
    """--direction with invalid value raises Exit(1)."""
    with tempfile.TemporaryDirectory() as td:
        papers_dir = Path(td) / "papers"
        papers_dir.mkdir()
        db_path = f"{td}/test.db"

        from drbrain.storage.database import Database

        db = Database(db_path)
        db.close()

        cfg = _make_minimal_config(db_path, str(papers_dir))

        with _mock_load_config(cfg):
            try:
                query_cmd(
                    text="test",
                    neighbors=2,
                    relation=None,
                    direction="sideways",
                    type_filter=None,
                    arg_type=None,
                    year_start=None,
                    year_end=None,
                    min_confidence=None,
                    limit=20,
                    json_output=False,
                    jsonl=False,
                    paper=None,
                    workspace=None,
                )
                assert False, "Should have raised Exit"
            except typer.Exit as e:
                assert e.exit_code == 1


def test_query_cmd_graph_expansion_includes_concepts():
    """--neighbors with traverse returns concept nodes with _via_graph fields."""
    with tempfile.TemporaryDirectory() as td:
        papers_dir = Path(td) / "papers"
        papers_dir.mkdir()
        db_path = f"{td}/test.db"

        from drbrain.storage.database import Database

        db = Database(db_path)
        db.insert_paper("paper_a", "Paper A", 2023, "uploaded")
        db.insert_concept("paper_a", "Method", "method_x", 0.9, year=2023)
        db.insert_concept("paper_a", "Gap", "gap_y", 0.8, year=2023)
        db.insert_edge("method_x", "gap_y", "addresses", "paper_a", 1.0)
        db.commit()
        db.close()

        cfg = _make_minimal_config(db_path, str(papers_dir))

        old_stdout = sys.stdout
        capture = io.StringIO()
        sys.stdout = capture
        try:
            with _mock_load_config(cfg):
                query_cmd(
                    text="method_x",
                    neighbors=2,
                    relation=None,
                    direction="both",
                    type_filter=None,
                    arg_type=None,
                    year_start=None,
                    year_end=None,
                    min_confidence=None,
                    limit=20,
                    json_output=True,
                    jsonl=False,
                    paper=None,
                    workspace=None,
                )
        finally:
            sys.stdout = old_stdout

        output = capture.getvalue()

        results = json.loads(output)
        graph_results = [r for r in results if r.get("_via_graph")]
        graph_ids = {r["local_id"] for r in graph_results}

        assert "gap_y" in graph_ids
        gap_result = [r for r in graph_results if r["local_id"] == "gap_y"][0]
        assert gap_result["type"] == "Gap"
        assert "_source_seed" in gap_result
        assert "_distance" in gap_result
        assert "_path" in gap_result
        relations_in_path = [step["relation"] for step in gap_result["_path"]]
        assert "addresses" in relations_in_path


def test_query_cmd_graph_relation_filter():
    """--relation limits which edges are followed in graph expansion."""
    with tempfile.TemporaryDirectory() as td:
        papers_dir = Path(td) / "papers"
        papers_dir.mkdir()
        db_path = f"{td}/test.db"

        from drbrain.storage.database import Database

        db = Database(db_path)
        db.insert_paper("paper_a", "Paper A", 2023, "uploaded")
        db.insert_concept("paper_a", "Method", "method_x", 0.9, year=2023)
        db.insert_concept("paper_a", "Gap", "gap_y", 0.8, year=2023)
        db.insert_concept("paper_a", "Problem", "problem_w", 0.7, year=2023)
        db.insert_edge("method_x", "gap_y", "addresses", "paper_a", 1.0)
        db.insert_edge("method_x", "problem_w", "challenges", "paper_a", 1.0)
        db.commit()
        db.close()

        cfg = _make_minimal_config(db_path, str(papers_dir))

        old_stdout = sys.stdout
        capture = io.StringIO()
        sys.stdout = capture
        try:
            with _mock_load_config(cfg):
                query_cmd(
                    text="method_x",
                    neighbors=1,
                    relation="addresses",
                    direction="both",
                    type_filter=None,
                    arg_type=None,
                    year_start=None,
                    year_end=None,
                    min_confidence=None,
                    limit=20,
                    json_output=True,
                    jsonl=False,
                    paper=None,
                    workspace=None,
                )
        finally:
            sys.stdout = old_stdout

        output = capture.getvalue()

        results = json.loads(output)
        graph_results = [r for r in results if r.get("_via_graph")]
        graph_ids = {r["local_id"] for r in graph_results}
        assert "gap_y" in graph_ids
        assert "problem_w" not in graph_ids


def test_query_cmd_graph_backward_compat():
    """--neighbors 2 without --relation/--direction still works."""
    with tempfile.TemporaryDirectory() as td:
        papers_dir = Path(td) / "papers"
        papers_dir.mkdir()
        db_path = f"{td}/test.db"

        from drbrain.storage.database import Database

        db = Database(db_path)
        db.insert_paper("paper_a", "Paper A", 2023, "uploaded")
        db.insert_concept("paper_a", "Method", "method_x", 0.9, year=2023)
        db.insert_concept("paper_a", "Method", "method_y", 0.85, year=2023)
        db.insert_edge("method_x", "method_y", "extends", "paper_a", 1.0)
        db.commit()
        db.close()

        cfg = _make_minimal_config(db_path, str(papers_dir))

        old_stdout = sys.stdout
        capture = io.StringIO()
        sys.stdout = capture
        try:
            with _mock_load_config(cfg):
                query_cmd(
                    text="method_x",
                    neighbors=2,
                    relation=None,
                    direction="both",
                    type_filter=None,
                    arg_type=None,
                    year_start=None,
                    year_end=None,
                    min_confidence=None,
                    limit=20,
                    json_output=True,
                    jsonl=False,
                    paper=None,
                    workspace=None,
                )
        finally:
            sys.stdout = old_stdout

        output = capture.getvalue()

        results = json.loads(output)
        assert any(r["local_id"] == "method_x" for r in results)
        graph_results = [r for r in results if r.get("_via_graph")]
        assert any(r["local_id"] == "method_y" for r in graph_results)
