"""Tests for graph engine closure with section-aware confidence."""

from drbrain.graph.engine import GraphEngine


def _make_graph(edges):
    g = GraphEngine()
    for src, dst, rel, paper in edges:
        g.add_edge(src, dst, rel, paper)
    return g


# -- get_neighbors --


def test_get_neighbors_hops_zero():
    """get_neighbors with hops=0 returns only the start node."""
    g = _make_graph(
        [
            ("A", "B", "supports", "p1"),
            ("B", "C", "extends", "p1"),
        ]
    )
    neighbors = g.get_neighbors("A", hops=0)
    assert neighbors == {"A"}


def test_get_neighbors_hops_two():
    """get_neighbors with default hops=2 returns 2-hop neighborhood."""
    g = _make_graph(
        [
            ("A", "B", "supports", "p1"),
            ("B", "C", "extends", "p1"),
            ("C", "D", "replaces", "p1"),
        ]
    )
    neighbors = g.get_neighbors("A", hops=2)
    assert "A" in neighbors
    assert "B" in neighbors
    assert "C" in neighbors
    assert "D" not in neighbors  # 3 hops away


def test_get_neighbors_isolated_node():
    """get_neighbors on a node with no edges returns only itself."""
    g = _make_graph([("A", "B", "supports", "p1")])
    neighbors = g.get_neighbors("X", hops=2)
    assert neighbors == {"X"}


# -- closure rules (no section_map) --


def test_closure_gap_addressed():
    """Rule 2: leaves_open + addresses → gap_addressed."""
    g = _make_graph(
        [
            ("P1", "Gap_X", "leaves_open", "p1"),
            ("M1", "Gap_X", "addresses", "p2"),
        ]
    )
    inferred = g.closure()
    gap_edges = [e for e in inferred if e["relation"] == "gap_addressed"]
    assert len(gap_edges) >= 1
    assert gap_edges[0]["src"] == "Gap_X"


def test_closure_indirect_evolution():
    """Rule 3: extends + replaces chain → indirect_evolution."""
    g = _make_graph(
        [
            ("M1", "M2", "extends", "p1"),
            ("M2", "M3", "replaces", "p2"),
        ]
    )
    inferred = g.closure()
    evolution_edges = [e for e in inferred if e["relation"] == "indirect_evolution"]
    assert len(evolution_edges) >= 1
    assert evolution_edges[0]["src"] == "M1"
    assert evolution_edges[0]["dst"] == "M3"
    assert evolution_edges[0]["via"] == "M2"


def test_closure_gap_to_debate():
    """Rule 4: gap points_to target that has both challenges and supports."""
    g = _make_graph(
        [
            ("Gap1", "Target_A", "points_to", "p1"),
            ("P1", "Target_A", "challenges", "p2"),
            ("P2", "Target_A", "supports", "p2"),
        ]
    )
    inferred = g.closure()
    g2d_edges = [e for e in inferred if e["relation"] == "gap_to_debate"]
    assert len(g2d_edges) >= 1
    assert g2d_edges[0]["src"] == "Gap1"
    assert g2d_edges[0]["dst"] == "Target_A"


def test_closure_shared_actor():
    """Rule 5: papers sharing an Actor form shared_actor edge."""
    g = _make_graph(
        [
            ("Paper_A", "Actor_X", "affiliated_with", "p1"),
            ("Paper_B", "Actor_X", "affiliated_with", "p2"),
        ]
    )
    inferred = g.closure()
    sa_edges = [e for e in inferred if e["relation"] == "shared_actor"]
    assert len(sa_edges) == 1
    assert sa_edges[0]["via"] == "Actor_X"


# -- closure with section_map --


def test_closure_section_aware_decay():
    """Inferred edges get section-aware confidence when section_map provided."""
    g = _make_graph(
        [
            ("P1", "Conclusion_Z", "supports", "p1"),
            ("P2", "Conclusion_Z", "challenges", "p2"),
        ]
    )
    section_map = {"P1": "Methods", "P2": "Discussion"}
    inferred = g.closure(section_map=section_map)
    # creates_debate should be inferred
    debate_edges = [e for e in inferred if e["relation"] == "creates_debate"]
    assert len(debate_edges) >= 1
    # Each inferred edge should have a confidence field
    for edge in debate_edges:
        assert "confidence" in edge
        assert 0 < edge["confidence"] <= 1.0


def test_closure_section_map_missing_src():
    """When section_map lacks a src node, empty string is used (default decay)."""
    g = _make_graph(
        [
            ("P1", "Gap_X", "leaves_open", "p1"),
            ("M1", "Gap_X", "addresses", "p2"),
        ]
    )
    # Non-empty section_map that doesn't include our src nodes
    section_map = {"UnrelatedNode": "Introduction"}
    inferred = g.closure(section_map=section_map)
    for edge in inferred:
        assert "confidence" in edge
        assert 0 < edge["confidence"] <= 1.0


def test_closure_section_map_partial_coverage():
    """section_map with only some nodes still applies confidence to all edges."""
    g = _make_graph(
        [
            ("P1", "Gap_X", "leaves_open", "p1"),
            ("M1", "Gap_X", "addresses", "p2"),
            ("M1", "M2", "extends", "p3"),
            ("M2", "M3", "replaces", "p3"),
        ]
    )
    section_map = {"P1": "Introduction", "M1": "Methods"}
    inferred = g.closure(section_map=section_map)
    # Every inferred edge must have confidence
    for edge in inferred:
        assert "confidence" in edge, f"Missing confidence on {edge['relation']}"


def test_closure_backward_compatible():
    """Without section_map, closure works as before (no confidence field)."""
    g = _make_graph(
        [
            ("P1", "Conclusion_Z", "supports", "p1"),
            ("P2", "Conclusion_Z", "challenges", "p2"),
        ]
    )
    inferred = g.closure()
    debate_edges = [e for e in inferred if e["relation"] == "creates_debate"]
    assert len(debate_edges) >= 1
    # Without section_map, no confidence field on inferred edges
    for edge in debate_edges:
        assert "confidence" not in edge


# -- closure_incremental --


def test_closure_incremental_empty_seeds():
    """closure_incremental with empty seed_nodes returns []."""
    g = _make_graph(
        [
            ("A", "B", "supports", "p1"),
            ("C", "B", "challenges", "p1"),
        ]
    )
    result = g.closure_incremental(set())
    assert result == []


def test_closure_incremental_seed_not_in_graph():
    """closure_incremental with seed nodes not present in graph returns []."""
    g = _make_graph(
        [
            ("A", "B", "supports", "p1"),
        ]
    )
    result = g.closure_incremental({"X", "Y"})
    assert result == []


def test_closure_incremental_with_seeds():
    """closure_incremental runs closure rules on subgraph around seed nodes."""
    g = _make_graph(
        [
            ("P1", "Gap_X", "leaves_open", "p1"),
            ("M1", "Gap_X", "addresses", "p2"),
            # This edge is far from the seed and should be excluded
            ("Far1", "Far2", "supports", "p3"),
        ]
    )
    result = g.closure_incremental({"P1"})
    # Should find gap_addressed from the subgraph
    gap_edges = [e for e in result if e["relation"] == "gap_addressed"]
    assert len(gap_edges) >= 1


# -- detect_research_seeds --


def test_detect_research_seeds_empty_graph():
    """detect_research_seeds with empty graph returns empty list."""
    g = GraphEngine()
    seeds = g.detect_research_seeds()
    assert isinstance(seeds, list)
    assert len(seeds) == 0


def test_detect_research_seeds_unaddressed_gap():
    """detect_research_seeds detects gaps with leaves_open but no addresses."""
    g = _make_graph(
        [
            ("P1", "Gap_Z", "leaves_open", "p1"),
            ("P2", "Gap_Z", "leaves_open", "p2"),
        ]
    )
    seeds = g.detect_research_seeds()
    gap_seeds = [s for s in seeds if s["type"] == "unaddressed_gap"]
    assert len(gap_seeds) >= 1
    assert gap_seeds[0]["concept"] == "Gap_Z"


def test_detect_research_seeds_debate_zone():
    """detect_research_seeds detects targets with both supports and challenges."""
    g = _make_graph(
        [
            ("P1", "Claim_X", "supports", "p1"),
            ("P2", "Claim_X", "challenges", "p2"),
        ]
    )
    seeds = g.detect_research_seeds()
    debate = [s for s in seeds if s["type"] == "debate_zone"]
    assert len(debate) == 1
    assert debate[0]["concept"] == "Claim_X"


def test_detect_research_seeds_no_gaps_when_addressed():
    """A gap with both leaves_open and addresses is NOT flagged as unaddressed."""
    g = _make_graph(
        [
            ("P1", "Gap_Y", "leaves_open", "p1"),
            ("M1", "Gap_Y", "addresses", "p2"),
        ]
    )
    seeds = g.detect_research_seeds()
    gap_seeds = [s for s in seeds if s["type"] == "unaddressed_gap"]
    assert len(gap_seeds) == 0


# -- closure with constrains relation --


def test_closure_constrains_indexing():
    """closure indexes constrains relation without errors (edge case coverage)."""
    g = _make_graph(
        [
            ("Gap1", "Method1", "constrains", "p1"),
            ("P1", "Conclusion_Z", "supports", "p1"),
            ("P2", "Conclusion_Z", "challenges", "p2"),
        ]
    )
    inferred = g.closure()
    # constrains is indexed but has no inference rule — no new edges from it
    debate_edges = [e for e in inferred if e["relation"] == "creates_debate"]
    assert len(debate_edges) >= 1


# -- load_from_db --


def test_load_from_db_all_edges(tmp_db):
    """load_from_db loads all edges from database into the graph."""
    tmp_db.insert_paper("p1", "Test Paper", 2024, "uploaded")
    tmp_db.insert_concept("p1", "Method", "M1", 0.9, year=2024)
    tmp_db.insert_concept("p1", "Method", "M2", 0.8, year=2024)
    tmp_db.insert_edge("M1", "M2", "extends", "p1")
    tmp_db.commit()

    g = GraphEngine()
    g.load_from_db(tmp_db)
    assert g.graph.number_of_edges() >= 1


def test_load_from_db_with_paper_ids(tmp_db):
    """load_from_db filters edges by source_paper when paper_ids provided."""
    tmp_db.insert_paper("p1", "Paper One", 2024, "uploaded")
    tmp_db.insert_paper("p2", "Paper Two", 2024, "uploaded")
    tmp_db.insert_concept("p1", "Method", "M1", 0.9, year=2024)
    tmp_db.insert_concept("p1", "Method", "M2", 0.8, year=2024)
    tmp_db.insert_concept("p2", "Problem", "P1", 0.7, year=2024)
    tmp_db.insert_concept("p2", "Method", "M3", 0.6, year=2024)
    tmp_db.insert_edge("M1", "M2", "extends", "p1")
    tmp_db.insert_edge("M3", "P1", "addresses", "p2")
    tmp_db.commit()

    g = GraphEngine()
    g.load_from_db(tmp_db, paper_ids={"p1"})
    # Only the p1 edge should be loaded
    assert g.graph.number_of_edges() == 1


# -- closure_incremental edge cases --


def test_closure_incremental_seed_isolated_within_subgraph():
    """closure_incremental with seed that has neighbors but no edges among them."""
    g = _make_graph(
        [
            ("Seed", "A", "supports", "p1"),
            ("Seed", "B", "challenges", "p1"),
        ]
    )
    # Seed's neighbors A and B are in subgraph, but there's no A↔B edge
    result = g.closure_incremental({"A"})
    # A is in subgraph with Seed and B, but no creates_debate trigger
    # since supports and challenges target different nodes
    assert isinstance(result, list)


def test_closure_incremental_gap_to_debate():
    """closure_incremental infers gap_to_debate within seed's neighborhood."""
    g = _make_graph(
        [
            ("Gap1", "Target_A", "points_to", "p1"),
            ("P1", "Target_A", "challenges", "p2"),
            ("P2", "Target_A", "supports", "p2"),
        ]
    )
    result = g.closure_incremental({"Gap1"})
    g2d = [e for e in result if e["relation"] == "gap_to_debate"]
    assert len(g2d) == 1
    assert g2d[0]["src"] == "Gap1"


def test_closure_incremental_shared_actor():
    """closure_incremental infers shared_actor in seed's neighborhood."""
    g = _make_graph(
        [
            ("Paper_A", "Actor_X", "affiliated_with", "p1"),
            ("Paper_B", "Actor_X", "affiliated_with", "p2"),
            ("Paper_A", "Target_X", "supports", "p1"),
        ]
    )
    result = g.closure_incremental({"Paper_A"})
    sa = [e for e in result if e["relation"] == "shared_actor"]
    assert len(sa) == 1
    assert sa[0]["via"] == "Actor_X"


def test_closure_incremental_constrains_indexing():
    """closure_incremental indexes constrains relation in subgraph."""
    g = _make_graph(
        [
            ("Gap1", "M1", "constrains", "p1"),
            ("P1", "M1", "supports", "p2"),
        ]
    )
    result = g.closure_incremental({"Gap1"})
    # constrains is indexed but has no direct inference rule
    assert isinstance(result, list)
