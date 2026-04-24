"""Tests for graph engine: closure rules, research seeds, load/persist."""
import tempfile
from pathlib import Path

from brbrain.graph.engine import GraphEngine
from brbrain.storage.database import Database


def test_get_neighbors_1hop():
    """get_neighbors with hops=1 returns start node + immediate neighbors."""
    g = GraphEngine()
    g.add_edge("A", "B", "cites", "p1")
    g.add_edge("B", "C", "cites", "p1")

    # hops=1: start node + direct neighbors
    neighbors = g.get_neighbors("B", hops=1)
    assert neighbors == {"A", "B", "C"}


def test_get_neighbors_2hop():
    """2-hop neighborhood expands to include adjacent nodes."""
    g = GraphEngine()
    g.add_edge("A", "B", "cites", "p1")
    g.add_edge("B", "C", "cites", "p1")
    g.add_edge("C", "D", "cites", "p1")

    neighbors = g.get_neighbors("B", hops=2)
    assert "A" in neighbors
    assert "B" in neighbors
    assert "C" in neighbors


def test_closure_creates_debate():
    """challenges(P, C) & supports(Q, C) => creates_debate(P, Q, C)."""
    g = GraphEngine()
    g.add_edge("P1", "Conclusion_A", "challenges", "paper1")
    g.add_edge("P2", "Conclusion_A", "supports", "paper2")

    inferred = g.closure()
    assert len(inferred) >= 1
    debate = [e for e in inferred if e["relation"] == "creates_debate"]
    assert len(debate) == 1
    assert debate[0]["via"] == "Conclusion_A"


def test_closure_gap_addressed():
    """leaves_open(P, G) & addresses(Q, G) => gap_addressed(G, Q)."""
    g = GraphEngine()
    g.add_edge("P1", "Gap_X", "leaves_open", "paper1")
    g.add_edge("M1", "Gap_X", "addresses", "paper2")

    inferred = g.closure()
    gap_edges = [e for e in inferred if e["relation"] == "gap_addressed"]
    assert len(gap_edges) == 1
    assert gap_edges[0]["src"] == "Gap_X"
    assert gap_edges[0]["dst"] == "M1"


def test_closure_indirect_evolution():
    """extends(M1, M2) & replaces(M2, M3) => indirect_evolution(M1, M3)."""
    g = GraphEngine()
    g.add_edge("M1", "M2", "extends", "paper1")
    g.add_edge("M2", "M3", "replaces", "paper2")

    inferred = g.closure()
    evol = [e for e in inferred if e["relation"] == "indirect_evolution"]
    assert len(evol) == 1
    assert evol[0]["src"] == "M1"
    assert evol[0]["dst"] == "M3"
    assert evol[0]["via"] == "M2"


def test_closure_no_inferred_when_rules_not_met():
    """closure returns empty list when no rules trigger."""
    g = GraphEngine()
    g.add_edge("A", "B", "cites", "paper1")
    g.add_edge("C", "D", "addresses", "paper1")

    inferred = g.closure()
    assert len(inferred) == 0


def test_detect_research_seeds_stale_problem():
    """Problem with high in-degree triggers stale_problem seed."""
    g = GraphEngine()
    # 3 papers addressing the same problem
    g.add_edge("M1", "Problem_X", "addresses", "p1")
    g.add_edge("M2", "Problem_X", "addresses", "p2")
    g.add_edge("M3", "Problem_X", "addresses", "p3")

    seeds = g.detect_research_seeds()
    stale = [s for s in seeds if s["type"] == "stale_problem"]
    assert len(stale) >= 1
    assert stale[0]["concept"] == "Problem_X"


def test_detect_research_seeds_unaddressed_gap():
    """Gap node with no incoming addresses triggers unaddressed_gap seed."""
    g = GraphEngine()
    g.add_edge("G1", "Gap_Y", "leaves_open", "p1")
    g.add_edge("G2", "Gap_Y", "leaves_open", "p2")

    seeds = g.detect_research_seeds()
    gaps = [s for s in seeds if s["type"] == "unaddressed_gap"]
    assert len(gaps) == 1
    assert gaps[0]["concept"] == "Gap_Y"


def test_detect_research_seeds_debate_zone():
    """Conclusion with both supports and challenges triggers debate_zone."""
    g = GraphEngine()
    g.add_edge("P1", "Conclusion_Z", "supports", "paper1")
    g.add_edge("P2", "Conclusion_Z", "challenges", "paper2")

    seeds = g.detect_research_seeds()
    debates = [s for s in seeds if s["type"] == "debate_zone"]
    assert len(debates) == 1
    assert debates[0]["concept"] == "Conclusion_Z"


def test_detect_research_seeds_no_seeds_on_empty_graph():
    """Empty graph returns no research seeds."""
    g = GraphEngine()
    seeds = g.detect_research_seeds()
    assert len(seeds) == 0


def test_load_from_db_and_persist():
    """Graph round-trips edges through database."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        db = Database(str(db_path))

        db.insert_paper("p1", "Test Paper", 2024, "uploaded")
        db.insert_paper("p2", "Another Paper", 2024, "uploaded")
        db.insert_edge("p1", "p2", "cites", "p1", weight=1.0)
        db.commit()

        g = GraphEngine()
        g.load_from_db(db)
        assert g.graph.number_of_edges() == 1

        # Add another edge via graph and persist
        g.add_edge("p2", "p1", "extends", "p2", weight=2.0)
        g.persist_to_db(db)

        g2 = GraphEngine()
        g2.load_from_db(db)
        assert g2.graph.number_of_edges() == 2

        db.close()
