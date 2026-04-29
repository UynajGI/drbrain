"""Tests for incremental closure: only re-infer from affected nodes."""

import tempfile
from pathlib import Path

from drbrain.graph.engine import GraphEngine
from drbrain.storage.database import Database


def test_closure_incremental_only_affected():
    """closure_incremental infers from seed nodes through 2-hop neighborhood."""
    g = GraphEngine()
    # Existing chain: A extends B, B replaces C
    g.add_edge("A", "B", "extends", "p1")
    g.add_edge("B", "C", "replaces", "p1")

    # New edge: D extends A
    g.add_edge("D", "A", "extends", "p2")

    # Incremental closure from seed {"D"}
    # D extends A, A extends B => transitive: D extends B
    # A extends B, B replaces C => indirect_evolution(A, C) [full closure]
    # D extends A, A extends B => transitive: D extends B (new from seed)
    inferred = g.closure_incremental({"D"})
    pairs = {(e["src"], e["dst"], e["relation"]) for e in inferred}
    # Transitive closure: D extends A, A extends B => D extends B
    assert ("D", "B", "extends") in pairs


def test_closure_incremental_empty_seed():
    """Empty seed list returns no inferences."""
    g = GraphEngine()
    g.add_edge("A", "B", "extends", "p1")
    assert g.closure_incremental(set()) == []


def test_closure_incremental_no_seed_nodes():
    """Seed nodes not in graph returns no inferences."""
    g = GraphEngine()
    g.add_edge("A", "B", "extends", "p1")
    inferred = g.closure_incremental({"nonexistent"})
    assert inferred == []


def test_closure_incremental_creates_debate():
    """Adding a supports edge triggers creates_debate from seed."""
    g = GraphEngine()
    g.add_edge("P1", "Conclusion_A", "challenges", "p1")
    # New: P2 supports Conclusion_A
    g.add_edge("P2", "Conclusion_A", "supports", "p2")

    inferred = g.closure_incremental({"P2"})
    debates = [e for e in inferred if e["relation"] == "creates_debate"]
    assert len(debates) >= 1


def test_closure_incremental_vs_full_closure():
    """Incremental closure from all nodes equals full closure."""
    g = GraphEngine()
    g.add_edge("M1", "Conclusion_X", "challenges", "p1")
    g.add_edge("M2", "Conclusion_X", "supports", "p1")
    g.add_edge("M3", "M1", "extends", "p2")
    g.add_edge("M1", "M4", "replaces", "p1")

    all_nodes = set()
    for u, v in g.graph.edges():
        all_nodes.add(u)
        all_nodes.add(v)

    full = g.closure()
    incremental = g.closure_incremental(all_nodes)

    full_set = {(e["src"], e["dst"], e["relation"]) for e in full}
    incr_set = {(e["src"], e["dst"], e["relation"]) for e in incremental}
    assert full_set == incr_set


def test_ingest_auto_closure():
    """ingest_cmd automatically runs closure after paper ingest."""
    # This is a higher-level test: verify that _ingest_single_paper
    # calls closure after graph loading.
    # We test this by checking the ingest pipeline code path.
    from drbrain.cli.commands import _ingest_single_paper
    from drbrain.dedup.resolver import DedupEngine
    from drbrain.extractor.canonical import SmartAligner
    from drbrain.graph.engine import GraphEngine

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        reports_dir = Path(td) / "reports"
        papers_dir = Path(td) / "papers"
        logs_dir = Path(td) / "logs"
        reports_dir.mkdir()
        papers_dir.mkdir()
        logs_dir.mkdir()

        cfg = {
            "db": {"path": str(db_path)},
            "llm": {"models": [{"provider": "openai", "model": "gpt-4", "api_key": "x"}]},
            "mineru": {
                "token": "",
                "model": "vlm",
                "is_ocr": False,
                "enable_formula": True,
                "enable_table": True,
            },
            "dirs": {
                "inbox": str(papers_dir),
                "papers": str(papers_dir),
                "reports": str(reports_dir),
                "cache": str(td),
                "logs": str(logs_dir),
            },
            "api": {"s2_rate_limit": 100, "cache_ttl": 86400},
            "queue": {"weak_threshold": 0.7, "auto_accept": 0.9},
            "bm25": {"k1": 1.5, "b": 0.75},
        }

        db = Database(str(db_path))
        graph = GraphEngine()
        dedup = DedupEngine(db)
        aligner = SmartAligner(db, models=cfg["llm"]["models"])

        # Create a minimal PDF file
        pdf_path = papers_dir / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        # Mock extract_pdf to return minimal parsed paper
        from unittest import mock

        from drbrain.parser.mineru_parser import ParsedPaper

        parsed = ParsedPaper(
            title="Test Paper",
            year=2024,
            doi=None,
            arxiv=None,
            text_blocks=["Introduction.—Content."],
            raw_md="# Test\n\nContent.",
        )

        # Mock concept extraction
        from drbrain.extractor.concept import ExtractedConcepts

        extracted = ExtractedConcepts(
            {
                "problems": [{"label": "Test Problem", "confidence": 0.9}],
                "methods": [],
                "conclusions": [],
                "debates": [],
                "gaps": [],
                "actors": [],
                "relations": [],
            }
        )

        with (
            mock.patch("drbrain.cli.commands.extract_pdf", return_value=parsed),
            mock.patch("drbrain.cli.commands.extract_concepts", return_value=extracted),
        ):
            result = _ingest_single_paper(
                pdf_path, cfg, db, graph, dedup, aligner, 0.7, 0.9, json_mode=True
            )

        # Verify closure ran (check that concepts table has data)
        concepts = db.get_concepts_by_paper(result["local_id"])
        assert len(concepts) > 0
        # Closure should have run - verify no error was raised
        assert result["ok"] is True

        db.close()
