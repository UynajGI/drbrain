"""Tests for knowledge boundary detection (Spec §16)."""

import tempfile
from pathlib import Path

from drbrain.graph.engine import GraphEngine
from drbrain.storage.database import Database


def _seed_full_graph(db, graph):
    """Helper: seed papers, concepts, edges, and load graph from db."""
    db.commit()
    graph.load_from_db(db)


def test_technology_cliff():
    """Method with dense extends chain that suddenly stops, with related Gap."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")

        # Papers spanning years
        for i, year in enumerate([2018, 2019, 2020, 2021]):
            db.insert_paper(f"p{i:03d}", f"Paper {i} ({year})", year, "uploaded")
            db.insert_concept(f"p{i:03d}", "Method", "RNN variants", 0.9, year=year)
        # Recent papers with no extends (chain stopped)
        for i, year in enumerate([2024, 2025]):
            pid = f"p{10 + i:03d}"
            db.insert_paper(pid, f"Recent {i} ({year})", year, "uploaded")
            db.insert_concept(pid, "Method", "RNN variants", 0.7, year=year)

        # Extend chain between methods (RNN variants extends LSTM)
        db.insert_paper("p_extra", "LSTM paper", 2017, "uploaded")
        db.insert_concept("p_extra", "Method", "LSTM", 0.9, year=2017)
        db.insert_edge("RNN variants", "LSTM", "extends", "p000")

        # Gap that constrains the method
        db.insert_paper("p200", "Gap paper", 2022, "uploaded")
        db.insert_concept("p200", "Gap", "vanishing gradient", 0.85, year=2022)
        db.insert_edge("vanishing gradient", "RNN variants", "constrains", "p200")

        graph = GraphEngine()
        _seed_full_graph(db, graph)

        seeds = graph.detect_research_seeds(db=db)
        cliff = [s for s in seeds if s["type"] == "technology_cliff"]
        assert len(cliff) >= 1
        assert "RNN variants" in str(cliff[0])
        db.close()


def test_cross_domain_isomorphism():
    """Two disconnected subgraphs share the same Problem label."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")

        # Domain A: NLP papers
        db.insert_paper("pA1", "NLP paper 1", 2023, "uploaded")
        db.insert_concept("pA1", "Problem", "long-range dependency", 0.9, year=2023)
        db.insert_concept("pA1", "Method", "Transformer", 0.95, year=2023)
        db.insert_edge("Transformer", "long-range dependency", "addresses", "pA1")

        db.insert_paper("pA2", "NLP paper 2", 2024, "uploaded")
        db.insert_concept("pA2", "Problem", "long-range dependency", 0.9, year=2024)
        db.insert_edge("Attention", "long-range dependency", "addresses", "pA2")

        # Domain B: Vision papers (same problem, different methods, no connection)
        db.insert_paper("pB1", "Vision paper 1", 2023, "uploaded")
        db.insert_concept("pB1", "Problem", "long-range dependency", 0.85, year=2023)
        db.insert_concept("pB1", "Method", "ViT", 0.9, year=2023)
        db.insert_edge("ViT", "long-range dependency", "addresses", "pB1")

        db.insert_paper("pB2", "Vision paper 2", 2024, "uploaded")
        db.insert_concept("pB2", "Problem", "long-range dependency", 0.88, year=2024)
        db.insert_edge("Swin", "long-range dependency", "addresses", "pB2")

        graph = GraphEngine()
        _seed_full_graph(db, graph)

        seeds = graph.detect_research_seeds(db=db)
        iso = [s for s in seeds if s["type"] == "cross_domain_isomorphism"]
        assert len(iso) >= 1
        assert "long-range dependency" in str(iso[0])
        db.close()


def test_confidence_collapse():
    """Concept with avg_confidence dropping > 0.2 between consecutive 2-year windows."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")

        # Early window (2020-2021): high confidence
        for i in range(4):
            pid = f"p{i:03d}"
            db.insert_paper(pid, f"Early paper {i}", 2020 + (i % 2), "uploaded")
            db.insert_concept(pid, "Method", "GAN", 0.9, year=2020 + (i % 2))

        # Late window (2024-2025): low confidence (paradigm shift)
        for i in range(4):
            pid = f"p{10 + i:03d}"
            db.insert_paper(pid, f"Late paper {i}", 2024 + (i % 2), "uploaded")
            db.insert_concept(pid, "Method", "GAN", 0.55, year=2024 + (i % 2))

        graph = GraphEngine()
        _seed_full_graph(db, graph)

        seeds = graph.detect_research_seeds(db=db)
        collapse = [s for s in seeds if s["type"] == "confidence_collapse"]
        assert len(collapse) >= 1
        assert "GAN" in str(collapse[0])
        db.close()


def test_no_false_positive_confidence_stable():
    """Concept with stable confidence should NOT trigger confidence collapse."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")

        for i in range(6):
            pid = f"p{i:03d}"
            year = 2020 + (i % 6)
            db.insert_paper(pid, f"Stable paper {i}", year, "uploaded")
            db.insert_concept(pid, "Method", "Stable Method", 0.85, year=year)

        graph = GraphEngine()
        _seed_full_graph(db, graph)

        seeds = graph.detect_research_seeds(db=db)
        collapse = [s for s in seeds if s["type"] == "confidence_collapse"]
        assert len(collapse) == 0
        db.close()


def test_existing_patterns_still_work():
    """Existing seed patterns (stale_problem, unaddressed_gap, debate_zone) still detected."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")

        # Unaddressed gap: needs leaves_open edge pointing to the Gap
        db.insert_paper("p1", "Gap paper", 2024, "uploaded")
        db.insert_concept("p1", "Gap", "unresolved issue", 0.8, year=2024)
        db.insert_concept("p1", "Problem", "related problem", 0.8, year=2024)
        db.insert_edge("related problem", "unresolved issue", "leaves_open", "p1")

        graph = GraphEngine()
        _seed_full_graph(db, graph)

        seeds = graph.detect_research_seeds(db=db)
        gaps = [s for s in seeds if s["type"] == "unaddressed_gap"]
        assert len(gaps) >= 1
        db.close()
