from drbrain.graph.engine import GraphEngine
from drbrain.graph.genealogy import (
    _format_provenance,
    _get_concept_provenance,
    analyze_difficulty,
    analyze_frontier,
    detect_paradigm_shifts,
    evolve_concept,
    find_transfer_opportunities,
    format_tree,
    trace_descendants,
)
from drbrain.storage.database import Database


def test_evolve_no_matching_concept():
    """Evolve returns empty list for unknown concept."""
    db = Database(":memory:")
    graph = GraphEngine()
    graph.load_from_db(db)
    result = evolve_concept(graph, db, "nonexistent")
    assert result == []
    db.close()


def test_evolve_single_node():
    """Evolve returns tree with just the root when no edges exist."""
    db = Database(":memory:")
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p1', 'Test', 2026, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) "
        "VALUES ('p1', 'Method', 'Test Concept', 0.9, 'intro')"
    )
    db.commit()
    graph = GraphEngine()
    graph.load_from_db(db)
    result = evolve_concept(graph, db, "Test Concept")
    assert len(result) >= 1
    root = result[0]
    assert root["label"] == "Test Concept"
    assert root["local_id"] == "p1"
    assert root["year"] == 2026
    assert root["children"] == []
    db.close()


def test_evolve_with_descendants():
    """Evolve follows graph edges to find descendants."""
    db = Database(":memory:")
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p1', 'Paper 1', 2020, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p2', 'Paper 2', 2021, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) "
        "VALUES ('p1', 'Method', 'Old Method', 0.9, 'method')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) "
        "VALUES ('p2', 'Method', 'New Method', 0.9, 'method')"
    )
    db.conn.execute(
        "INSERT INTO edges (src_id, dst_id, relation, source_paper, weight) "
        "VALUES ('Old Method', 'New Method', 'extends', 'p1', 0.8)"
    )
    db.commit()

    graph = GraphEngine()
    graph.load_from_db(db)
    result = evolve_concept(graph, db, "Old Method", direction="descendants")
    assert len(result) >= 1
    root = result[0]
    assert len(root.get("children", [])) >= 1
    child = root["children"][0]
    assert child["label"] == "New Method"
    assert child["relation"] == "extends"
    db.close()


def test_format_tree_text():
    """Format tree produces readable text output."""
    node = {
        "label": "Root",
        "type": "Method",
        "local_id": "p1",
        "year": 2020,
        "relation": None,
        "children": [
            {
                "label": "Child",
                "type": "Method",
                "local_id": "p2",
                "year": 2021,
                "relation": "extends",
                "children": [],
            }
        ],
    }
    text = format_tree([node])
    assert "Root" in text
    assert "Child" in text
    assert "extends" in text


def test_format_tree_mermaid():
    """Format tree in Mermaid mode produces valid syntax."""
    node = {
        "label": "Root",
        "type": "Method",
        "local_id": "p1",
        "year": 2020,
        "relation": None,
        "children": [],
    }
    text = format_tree([node], mermaid=True)
    assert text.startswith("graph TD")
    assert "Root" in text


def test_evolve_max_depth():
    """Evolve respects max_depth limit."""
    db = Database(":memory:")
    for i in range(5):
        pid = f"p{i}"
        db.conn.execute(
            f"INSERT INTO papers (local_id, title, year, status) "
            f"VALUES ('{pid}', 'Paper {i}', {2020 + i}, 'extracted')"
        )
        db.conn.execute(
            f"INSERT INTO concepts (local_id, type, label, confidence, section) "
            f"VALUES ('{pid}', 'Method', 'Concept {i}', 0.9, 'method')"
        )
    # Chain: 0->1->2->3->4
    for i in range(4):
        db.conn.execute(
            "INSERT INTO edges (src_id, dst_id, relation, source_paper) VALUES (?, 'extends', ?, ?)",
            (f"Concept {i}", f"Concept {i + 1}", f"p{i}"),
        )
    db.commit()

    graph = GraphEngine()
    graph.load_from_db(db)
    result = evolve_concept(graph, db, "Concept 0", direction="descendants", max_depth=2)
    root = result[0]

    # Count depth of deepest child
    def max_depth_in_tree(node, depth=0):
        m = depth
        for child in node.get("children", []):
            m = max(m, max_depth_in_tree(child, depth + 1))
        return m

    tree_depth = max_depth_in_tree(root)
    assert tree_depth <= 2
    db.close()


def test_evolve_with_ancestors():
    """Evolve follows incoming edges to find ancestors, restructuring tree."""
    db = Database(":memory:")
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p0', 'Paper 0', 2019, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p1', 'Paper 1', 2020, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) "
        "VALUES ('p0', 'Method', 'Old Method', 0.9, 'method')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) "
        "VALUES ('p1', 'Method', 'New Method', 0.9, 'method')"
    )
    db.conn.execute(
        "INSERT INTO edges (src_id, dst_id, relation, source_paper, weight) "
        "VALUES ('Old Method', 'New Method', 'extends', 'p0', 0.8)"
    )
    db.commit()

    graph = GraphEngine()
    graph.load_from_db(db)
    result = evolve_concept(graph, db, "New Method", direction="ancestors")
    assert len(result) >= 1
    root = result[0]
    # Root should be the deepest ancestor ("Old Method") holding the relation
    assert root["label"] == "Old Method"
    assert root["relation"] == "extends"
    assert len(root.get("children", [])) >= 1
    child = root["children"][0]
    assert child["label"] == "New Method"
    # Matched concept is the search target; has no incoming relation
    assert child["relation"] is None
    db.close()


def test_evolve_both_direction():
    """Evolve with direction=both finds ancestors and descendants together."""
    db = Database(":memory:")
    for i in range(3):
        pid = f"p{i}"
        db.conn.execute(
            f"INSERT INTO papers (local_id, title, year, status) "
            f"VALUES ('{pid}', 'Paper {i}', {2020 + i}, 'extracted')"
        )
        db.conn.execute(
            f"INSERT INTO concepts (local_id, type, label, confidence, section) "
            f"VALUES ('{pid}', 'Method', 'Concept {i}', 0.9, 'method')"
        )
    # Concept 0 extends Concept 1; Concept 1 extends Concept 2
    db.conn.execute(
        "INSERT INTO edges (src_id, dst_id, relation, source_paper, weight) "
        "VALUES ('Concept 0', 'Concept 1', 'extends', 'p0', 0.8)"
    )
    db.conn.execute(
        "INSERT INTO edges (src_id, dst_id, relation, source_paper, weight) "
        "VALUES ('Concept 1', 'Concept 2', 'extends', 'p1', 0.8)"
    )
    db.commit()

    graph = GraphEngine()
    graph.load_from_db(db)
    # Search for middle concept — should get ancestor C0 as root, C1 as child, C2 as grandchild
    result = evolve_concept(graph, db, "Concept 1", direction="both")
    assert len(result) >= 1
    root = result[0]
    assert root["label"] == "Concept 0"  # deepest ancestor becomes root
    children = root.get("children", [])
    assert len(children) >= 1
    middle = children[0]
    assert middle["label"] == "Concept 1"
    grandchild = middle.get("children", [])
    assert len(grandchild) >= 1
    assert grandchild[0]["label"] == "Concept 2"
    db.close()


def test_trace_descendants_not_found():
    """trace_descendants returns None for unknown paper."""
    from drbrain.graph.engine import GraphEngine
    from drbrain.graph.genealogy import trace_descendants
    from drbrain.storage.database import Database

    db = Database(":memory:")
    graph = GraphEngine()
    graph.load_from_db(db)
    result = trace_descendants(db, graph, "nonexistent")
    assert result is None
    db.close()


def test_trace_descendants_no_children():
    """Paper with no edges returns root-only tree."""
    from drbrain.graph.engine import GraphEngine
    from drbrain.graph.genealogy import trace_descendants
    from drbrain.storage.database import Database

    db = Database(":memory:")
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p1', 'Lonely Paper', 2020, 'extracted')"
    )
    db.commit()
    graph = GraphEngine()
    graph.load_from_db(db)
    result = trace_descendants(db, graph, "p1")
    assert result is not None
    assert result["local_id"] == "p1"
    assert result["children"] == []
    db.close()


def test_trace_descendants_with_edges():
    """Paper with concept edges returns tree with descendant papers."""
    from drbrain.graph.engine import GraphEngine
    from drbrain.graph.genealogy import trace_descendants
    from drbrain.storage.database import Database

    db = Database(":memory:")
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p1', 'Original', 2020, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p2', 'Descendant Paper', 2021, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) "
        "VALUES ('p1', 'Method', 'Orig Concept', 0.9, 'method')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) "
        "VALUES ('p2', 'Method', 'Desc Concept', 0.9, 'method')"
    )
    db.conn.execute(
        "INSERT INTO edges (src_id, dst_id, relation, source_paper, weight) "
        "VALUES ('Orig Concept', 'Desc Concept', 'extends', 'p1', 0.8)"
    )
    db.commit()

    graph = GraphEngine()
    graph.load_from_db(db)
    result = trace_descendants(db, graph, "p1", generations=1)
    assert result is not None
    assert len(result["children"]) == 1
    child = result["children"][0]
    assert child["local_id"] == "p2"
    assert "Descendant Paper" in child["label"]
    assert child["relation"] == "extends"
    db.close()


def test_trace_descendants_generations_limit():
    """trace_descendants respects generations limit."""
    from drbrain.graph.engine import GraphEngine
    from drbrain.graph.genealogy import trace_descendants
    from drbrain.storage.database import Database

    db = Database(":memory:")
    for i in range(4):
        pid = f"p{i}"
        db.conn.execute(
            f"INSERT INTO papers (local_id, title, year, status) "
            f"VALUES ('{pid}', 'Paper {i}', {2020 + i}, 'extracted')"
        )
        db.conn.execute(
            f"INSERT INTO concepts (local_id, type, label, confidence, section) "
            f"VALUES ('{pid}', 'Method', 'Concept {i}', 0.9, 'method')"
        )
    # Chain: C0 → C1 → C2 → C3
    for i in range(3):
        db.conn.execute(
            "INSERT INTO edges (src_id, dst_id, relation, source_paper, weight) "
            "VALUES (?, ?, 'extends', ?, 0.8)",
            (f"Concept {i}", f"Concept {i + 1}", f"p{i}"),
        )
    db.commit()

    graph = GraphEngine()
    graph.load_from_db(db)

    # generations=1: only direct descendants (p1)
    result = trace_descendants(db, graph, "p0", generations=1)
    assert result is not None
    assert len(result["children"]) == 1
    assert result["children"][0]["local_id"] == "p1"

    # generations=2: p1 and p2
    result = trace_descendants(db, graph, "p0", generations=2)
    assert result is not None
    children = result["children"]
    assert len(children) == 1
    assert children[0]["local_id"] == "p1"
    grandchildren = children[0]["children"]
    assert len(grandchildren) == 1
    assert grandchildren[0]["local_id"] == "p2"

    db.close()


# --- Landscape tests ---


def test_landscape_empty_workspace():
    """Landscape handles empty/nonexistent workspace gracefully."""
    from drbrain.graph.genealogy import landscape_workspace
    from drbrain.storage.database import Database

    db = Database(":memory:")
    result = landscape_workspace(db, workspace_path="nonexistent-ws")
    assert isinstance(result, dict)
    assert "timeline" in result
    assert len(result["timeline"]) == 0
    db.close()


def test_landscape_empty_papers():
    """Landscape with workspace containing no papers returns empty result."""
    from drbrain.graph.genealogy import landscape_workspace
    from drbrain.storage.database import Database

    db = Database(":memory:")
    result = landscape_workspace(db, paper_ids=[])
    assert isinstance(result, dict)
    assert "timeline" in result
    assert "gaps" in result
    assert "debates" in result
    assert len(result["timeline"]) == 0
    assert len(result["gaps"]) == 0
    assert len(result["debates"]) == 0
    db.close()


def test_landscape_basic_timeline():
    """Landscape produces timeline with year-ordered entries."""
    from drbrain.graph.genealogy import landscape_workspace
    from drbrain.storage.database import Database

    db = Database(":memory:")
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p1', 'Old Paper', 2018, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p2', 'New Paper', 2020, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) "
        "VALUES ('p1', 'Method', 'Method A', 0.9, 'method')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) "
        "VALUES ('p2', 'Method', 'Method B', 0.8, 'method')"
    )
    db.conn.execute(
        "INSERT INTO edges (src_id, dst_id, relation, source_paper, weight) "
        "VALUES ('Method A', 'Method B', 'extends', 'p1', 0.7)"
    )
    db.commit()

    result = landscape_workspace(db, paper_ids=["p1", "p2"])
    assert isinstance(result, dict)
    assert "timeline" in result
    assert len(result["timeline"]) >= 2
    years = [e["year"] for e in result["timeline"]]
    assert years == sorted(years)
    db.close()


def test_landscape_no_paper_ids():
    """Landscape with no workspace or paper_ids returns error."""
    from drbrain.graph.genealogy import landscape_workspace
    from drbrain.storage.database import Database

    db = Database(":memory:")
    result = landscape_workspace(db)
    assert isinstance(result, dict)
    assert "error" in result
    db.close()


def test_landscape_papers_without_year():
    """Landscape handles papers with null year gracefully."""
    from drbrain.graph.genealogy import landscape_workspace
    from drbrain.storage.database import Database

    db = Database(":memory:")
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p1', 'No Year Paper', NULL, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) "
        "VALUES ('p1', 'Method', 'Concept X', 0.9, 'method')"
    )
    db.commit()

    result = landscape_workspace(db, paper_ids=["p1"])
    # Papers without year are excluded from timeline
    assert isinstance(result, dict)
    assert len(result["timeline"]) == 0
    db.close()


# --- Paradigm shift tests ---


def test_paradigm_replacement_detected():
    """Type 1: old method declining, new method growing, challenges edge = paradigm shift."""
    from drbrain.graph.genealogy import detect_paradigm_shifts
    from drbrain.storage.database import Database

    db = Database(":memory:")
    # OldMethod: 3 papers in 2018, 1 in 2022 (declining)
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p_old1', 'Old Method Paper', 2018, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p_old2', 'Old Method Paper 2', 2018, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p_old3', 'Old Method Paper 3', 2018, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p_old4', 'Old Method Paper 4', 2022, 'extracted')"
    )
    # NewMethod: 2 papers in 2022, 1 in 2023 (growing)
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p_new1', 'New Method Paper', 2022, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p_new2', 'New Method Paper 2', 2022, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p_new3', 'New Method Paper 3', 2023, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) VALUES ('p_old1', 'Method', 'OldMethod', 0.9, 'method')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) VALUES ('p_old2', 'Method', 'OldMethod', 0.8, 'method')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) VALUES ('p_old3', 'Method', 'OldMethod', 0.7, 'method')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) VALUES ('p_old4', 'Method', 'OldMethod', 0.6, 'method')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) VALUES ('p_new1', 'Method', 'NewMethod', 0.9, 'method')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) VALUES ('p_new2', 'Method', 'NewMethod', 0.8, 'method')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) VALUES ('p_new3', 'Method', 'NewMethod', 0.9, 'method')"
    )
    db.conn.execute(
        "INSERT INTO edges (src_id, dst_id, relation, source_paper, weight) VALUES ('NewMethod', 'OldMethod', 'challenges', 'p_new1', 0.9)"
    )
    db.commit()

    from drbrain.graph.engine import GraphEngine

    graph = GraphEngine()
    graph.load_from_db(db)

    results = detect_paradigm_shifts(graph, db)
    assert isinstance(results, list)
    # Should detect OldMethod -> NewMethod shift
    replacement = [r for r in results if r.get("type") == "replacement"]
    assert len(replacement) >= 1
    r = replacement[0]
    assert r["old_concept"] == "OldMethod"
    assert r["new_concept"] == "NewMethod"
    db.close()


def test_paradigm_no_decline_no_flag():
    """When old method is NOT declining, don't flag as paradigm shift."""
    from drbrain.graph.genealogy import detect_paradigm_shifts
    from drbrain.storage.database import Database

    db = Database(":memory:")
    # Both old and new are in same year (no decline data)
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p1', 'Old', 2022, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p2', 'New', 2022, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) VALUES ('p1', 'Method', 'OldMethod', 0.9, 'method')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) VALUES ('p2', 'Method', 'NewMethod', 0.9, 'method')"
    )
    db.conn.execute(
        "INSERT INTO edges (src_id, dst_id, relation, source_paper, weight) VALUES ('NewMethod', 'OldMethod', 'challenges', 'p2', 0.5)"
    )
    db.commit()

    from drbrain.graph.engine import GraphEngine

    graph = GraphEngine()
    graph.load_from_db(db)

    results = detect_paradigm_shifts(graph, db)
    replacement = [r for r in results if r.get("type") == "replacement"]
    assert len(replacement) == 0  # No decline = no paradigm shift
    db.close()


def test_paradigm_explosion_detected():
    """Type 2: concept explodes 0->many papers in short time with descendants."""
    from drbrain.graph.genealogy import detect_paradigm_shifts
    from drbrain.storage.database import Database

    db = Database(":memory:")
    for i in range(5):
        db.conn.execute(
            f"INSERT INTO papers (local_id, title, year, status) VALUES ('p_exp{i}', 'Explosion Paper {i}', 2022, 'extracted')"
        )
        db.conn.execute(
            f"INSERT INTO concepts (local_id, type, label, confidence, section) VALUES ('p_exp{i}', 'Method', 'ExplodingConcept', 0.95, 'method')"
        )
    # Add descendant edge
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) VALUES ('p_exp0', 'Method', 'DerivedFromExplosion', 0.9, 'method')"
    )
    db.conn.execute(
        "INSERT INTO edges (src_id, dst_id, relation, source_paper, weight) VALUES ('ExplodingConcept', 'DerivedFromExplosion', 'extends', 'p_exp0', 0.8)"
    )
    db.commit()

    from drbrain.graph.engine import GraphEngine

    graph = GraphEngine()
    graph.load_from_db(db)

    results = detect_paradigm_shifts(graph, db, explosion_threshold=5, descendant_threshold=1)
    # With 5 papers in one year + extends edge = explosion-type paradigm
    explosion = [r for r in results if r.get("type") == "explosion"]
    assert len(explosion) >= 1
    e = explosion[0]
    assert e["concept"] == "ExplodingConcept"
    assert e["paper_count"] >= 3
    db.close()


def test_paradigm_cross_domain_invasion():
    """Type 3: method crosses domain boundary via applies edge."""
    from drbrain.graph.genealogy import detect_paradigm_shifts
    from drbrain.storage.database import Database

    db = Database(":memory:")
    # NLP concept applied to CV paper
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p_nlp', 'NLP Paper', 2020, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p_cv1', 'CV Paper 1', 2022, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p_cv2', 'CV Paper 2', 2022, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) VALUES ('p_nlp', 'Method', 'Transformer', 0.95, 'method')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) VALUES ('p_cv1', 'Method', 'VisionTransformer', 0.9, 'method')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) VALUES ('p_cv2', 'Method', 'SwinTransformer', 0.9, 'method')"
    )
    db.conn.execute(
        "INSERT INTO edges (src_id, dst_id, relation, source_paper, weight) VALUES ('Transformer', 'VisionTransformer', 'applies', 'p_nlp', 0.9)"
    )
    db.conn.execute(
        "INSERT INTO edges (src_id, dst_id, relation, source_paper, weight) VALUES ('VisionTransformer', 'SwinTransformer', 'extends', 'p_cv1', 0.8)"
    )
    db.commit()

    from drbrain.graph.engine import GraphEngine

    graph = GraphEngine()
    graph.load_from_db(db)

    results = detect_paradigm_shifts(graph, db)
    invasion = [r for r in results if r.get("type") == "cross_domain"]
    assert len(invasion) >= 1
    iv = invasion[0]
    assert iv["source_concept"] == "Transformer"
    assert iv["target_concept"] == "VisionTransformer"
    assert len(iv.get("cascade", [])) >= 1  # SwinTransformer cascaded
    db.close()


def test_paradigm_empty_graph():
    """Empty graph returns empty list, no crash."""
    from drbrain.graph.engine import GraphEngine
    from drbrain.graph.genealogy import detect_paradigm_shifts
    from drbrain.storage.database import Database

    db = Database(":memory:")
    graph = GraphEngine()
    graph.load_from_db(db)
    results = detect_paradigm_shifts(graph, db)
    assert results == []
    db.close()


# --- Transfer opportunity tests ---


def test_find_transfer_opportunities_explicit():
    """Workspace-based: find Method->Problem transfer candidates."""
    from drbrain.graph.engine import GraphEngine
    from drbrain.graph.genealogy import find_transfer_opportunities
    from drbrain.storage.database import Database

    db = Database(":memory:")
    # Setup two domains with papers
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p_nlp', 'NLP Paper', 2020, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p_cv', 'CV Paper', 2021, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) "
        "VALUES ('p_nlp', 'Method', 'Transformer', 0.95, 'method')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) "
        "VALUES ('p_cv', 'Problem', 'Image Classification', 0.95, 'intro')"
    )
    db.conn.execute(
        "INSERT INTO edges (src_id, dst_id, relation, source_paper, weight) "
        "VALUES ('Transformer', 'Image Classification', 'solves', 'p_nlp', 0.5)"
    )
    db.commit()

    graph = GraphEngine()
    graph.load_from_db(db)

    results = find_transfer_opportunities(
        db, graph, source_paper_ids=["p_nlp"], target_paper_ids=["p_cv"]
    )
    assert isinstance(results, list)
    if results:
        r = results[0]
        assert "source_method" in r
        assert "target_problem" in r
        assert "confidence" in r
    db.close()


def test_find_transfer_opportunities_empty():
    """Empty source or target returns empty list."""
    from drbrain.graph.engine import GraphEngine
    from drbrain.graph.genealogy import find_transfer_opportunities
    from drbrain.storage.database import Database

    db = Database(":memory:")
    graph = GraphEngine()
    graph.load_from_db(db)

    results = find_transfer_opportunities(db, graph, source_paper_ids=[], target_paper_ids=["p1"])
    assert results == []
    db.close()


def test_find_transfer_opportunities_auto():
    """Auto mode: cluster concepts by label similarity as domains."""
    from drbrain.graph.engine import GraphEngine
    from drbrain.graph.genealogy import find_transfer_opportunities_auto
    from drbrain.storage.database import Database

    db = Database(":memory:")
    # NLP domain
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p_nlp', 'NLP Paper', 2020, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) "
        "VALUES ('p_nlp', 'Method', 'Transformer', 0.95, 'method')"
    )
    # CV domain
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p_cv', 'CV Paper', 2021, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) "
        "VALUES ('p_cv', 'Problem', 'Vision Classification', 0.95, 'intro')"
    )
    # Cross-domain edge
    db.conn.execute(
        "INSERT INTO edges (src_id, dst_id, relation, source_paper, weight) "
        "VALUES ('Transformer', 'Vision Classification', 'solves', 'p_nlp', 0.5)"
    )
    db.commit()

    graph = GraphEngine()
    graph.load_from_db(db)

    results = find_transfer_opportunities_auto(db, graph)
    assert isinstance(results, list)
    if results:
        r = results[0]
        assert "confidence" in r
    db.close()


# --- Transfer history tests ---


def test_find_transfer_history():
    """Historical transfer history lists all applies edges with years."""
    from drbrain.graph.engine import GraphEngine
    from drbrain.graph.genealogy import find_transfer_history
    from drbrain.storage.database import Database

    db = Database(":memory:")
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p1', 'Attention Paper', 2017, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p2', 'ViT Paper', 2020, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) "
        "VALUES ('p1', 'Method', 'Transformer', 0.95, 'method')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) "
        "VALUES ('p2', 'Method', 'ViT', 0.95, 'method')"
    )
    db.conn.execute(
        "INSERT INTO edges (src_id, dst_id, relation, source_paper, weight) "
        "VALUES ('Transformer', 'ViT', 'applies', 'p1', 0.9)"
    )
    db.commit()

    graph = GraphEngine()
    graph.load_from_db(db)

    results = find_transfer_history(db, graph)
    assert isinstance(results, list)
    assert len(results) >= 1
    h = results[0]
    assert h["source_concept"] == "Transformer"
    assert h["target_concept"] == "ViT"
    assert h["relation"] == "applies"
    assert h["year"] == 2020  # year from ViT paper
    db.close()


def test_find_transfer_history_empty():
    """No applies edges = empty list."""
    from drbrain.graph.engine import GraphEngine
    from drbrain.graph.genealogy import find_transfer_history
    from drbrain.storage.database import Database

    db = Database(":memory:")
    graph = GraphEngine()
    graph.load_from_db(db)
    results = find_transfer_history(db, graph)
    assert results == []
    db.close()


# -- Provenance helpers --


def test_get_concept_provenance_found():
    """Returns (section, node_id, paper_id) for existing concept."""
    db = Database(":memory:")
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p1', 'Test', 2026, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section, node_id) "
        "VALUES ('p1', 'Gap', 'Scalability', 0.9, '5.2 Limitations', '0005.002')"
    )
    db.commit()
    section, node_id, paper_id = _get_concept_provenance(db, "Scalability", "Gap")
    assert section == "5.2 Limitations"
    assert node_id == "0005.002"
    assert paper_id == "p1"
    db.close()


def test_get_concept_provenance_not_found():
    """Returns empty strings for missing concept."""
    db = Database(":memory:")
    section, node_id, paper_id = _get_concept_provenance(db, "Nonexistent", "Gap")
    assert section == ""
    assert node_id == ""
    assert paper_id == ""
    db.close()


def test_get_concept_provenance_null_type():
    """Lookup with ctype=None matches any type."""
    db = Database(":memory:")
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p1', 'Test', 2026, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section, node_id) "
        "VALUES ('p1', 'Debate', 'Temperature Scaling', 0.85, '4.1 Comparison', '0004.001')"
    )
    db.commit()
    section, node_id, paper_id = _get_concept_provenance(db, "Temperature Scaling", None)
    assert section == "4.1 Comparison"
    assert node_id == "0004.001"
    assert paper_id == "p1"
    db.close()


def test_get_concept_provenance_highest_confidence():
    """Returns the highest-confidence match when multiple exist."""
    db = Database(":memory:")
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p1', 'Paper 1', 2026, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p2', 'Paper 2', 2026, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) "
        "VALUES ('p1', 'Gap', 'Scalability', 0.5, '1 Intro')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) "
        "VALUES ('p2', 'Gap', 'Scalability', 0.95, '5.2 Limitations')"
    )
    db.commit()
    section, node_id, paper_id = _get_concept_provenance(db, "Scalability", "Gap")
    assert section == "5.2 Limitations"  # higher confidence wins
    assert paper_id == "p2"
    db.close()


def test_format_provenance_full():
    """Full provenance with section and paper."""
    result = _format_provenance("3.1 Methods", "0003.001", "paper-abc")
    assert result == "[source: 3.1 Methods of paper-abc]"


def test_format_provenance_section_only():
    """Provenance with section but no paper."""
    result = _format_provenance("5.2 Limitations", "", "")
    assert result == "[source: 5.2 Limitations]"


def test_format_provenance_paper_only():
    """Provenance with paper but no section."""
    result = _format_provenance("", "", "paper-abc")
    assert result == "[source: paper-abc]"


def test_format_provenance_unknown():
    """Fallback when all fields empty."""
    result = _format_provenance("", "", "")
    assert result == "[source: unknown]"


def test_landscape_workspace_includes_provenance():
    """landscape_workspace enriches gaps and debates with provenance."""
    db = Database(":memory:")
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p1', 'Test', 2026, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p2', 'Comment', 2026, 'extracted')"
    )
    # Gap concept with provenance
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section, node_id) "
        "VALUES ('p1', 'Gap', 'Scalability Gap', 0.9, '5.2 Limitations', '0005.002')"
    )
    # Method concept that leaves_open the gap
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence) "
        "VALUES ('p1', 'Method', 'GNN', 0.8)"
    )
    # Debate target
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) "
        "VALUES ('p1', 'Method', 'Temperature Scaling', 0.85, '4.1 Comparison')"
    )
    db.commit()

    # Insert edges into DB (landscape_workspace creates its own GraphEngine)
    db.conn.execute(
        "INSERT INTO edges (src_id, dst_id, relation, source_paper) "
        "VALUES ('GNN', 'Scalability Gap', 'leaves_open', 'p1')"
    )
    db.conn.execute(
        "INSERT INTO edges (src_id, dst_id, relation, source_paper) "
        "VALUES ('Author A', 'Temperature Scaling', 'supports', 'p1')"
    )
    db.conn.execute(
        "INSERT INTO edges (src_id, dst_id, relation, source_paper) "
        "VALUES ('Author B', 'Temperature Scaling', 'challenges', 'p2')"
    )
    db.commit()

    from drbrain.graph.genealogy import landscape_workspace

    result = landscape_workspace(db, paper_ids=["p1", "p2"])

    # Gaps should have provenance
    gaps = result.get("gaps", [])
    assert len(gaps) >= 1
    gap = gaps[0]
    assert "section" in gap
    assert "node_id" in gap
    assert "paper_id" in gap
    assert "provenance" in gap
    assert gap["section"] == "5.2 Limitations"
    assert "[source: 5.2 Limitations of p1]" in gap["provenance"]

    # Debates should have provenance
    debates = result.get("debates", [])
    assert len(debates) >= 1
    debate = debates[0]
    assert "provenance" in debate

    db.close()


# -- evolve provenance --


def test_evolve_includes_section_provenance():
    """evolve_concept tree nodes carry section and node_id."""
    db = Database(":memory:")
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p1', 'Test', 2026, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section, node_id) "
        "VALUES ('p1', 'Method', 'GNN', 0.9, '3.1 Methods', '0003.001')"
    )
    db.commit()
    graph = GraphEngine()
    graph.load_from_db(db)
    result = evolve_concept(graph, db, "GNN")
    assert len(result) >= 1
    assert result[0]["section"] == "3.1 Methods"
    assert result[0]["node_id"] == "0003.001"
    db.close()


# -- descendants provenance --


def test_descendants_includes_concept_bridge():
    """trace_descendants child nodes carry via_concept and via_provenance."""
    db = Database(":memory:")
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p1', 'Parent', 2026, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p2', 'Child', 2026, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section, node_id) "
        "VALUES ('p1', 'Method', 'GNN', 0.9, '3.1 Methods', '0003.001')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section, node_id) "
        "VALUES ('p2', 'Method', 'Improved GNN', 0.85, '2 Methods', '0002.001')"
    )
    db.conn.execute(
        "INSERT INTO edges (src_id, dst_id, relation, source_paper) "
        "VALUES ('GNN', 'Improved GNN', 'extends', 'p2')"
    )
    db.commit()
    graph = GraphEngine()
    graph.load_from_db(db)
    result = trace_descendants(db, graph, "p1")
    assert result is not None
    children = result.get("children", [])
    assert len(children) >= 1
    child = children[0]
    assert "via_concept" in child
    assert "via_provenance" in child
    db.close()


# -- paradigm provenance --


def test_paradigm_replacement_includes_provenance():
    """Replacement shifts carry old_provenance and new_provenance."""
    db = Database(":memory:")
    # OldMethod: 3 papers in 2008, 1 in 2015 → decline
    for i, y in enumerate((2008, 2008, 2008, 2015)):
        pid = f"old-p{i}"
        db.conn.execute(
            "INSERT INTO papers (local_id, title, year, status) VALUES (?, 'Old', ?, 'extracted')",
            (pid, y),
        )
        db.conn.execute(
            "INSERT INTO concepts (local_id, type, label, confidence, section) "
            "VALUES (?, 'Method', 'OldMethod', 0.9, '3 Methods')",
            (pid,),
        )
    # NewMethod: 3 papers in 2024-2026 → growing
    for i, y in enumerate((2024, 2025, 2026)):
        pid = f"new-p{i}"
        db.conn.execute(
            "INSERT INTO papers (local_id, title, year, status) VALUES (?, 'New', ?, 'extracted')",
            (pid, y),
        )
        db.conn.execute(
            "INSERT INTO concepts (local_id, type, label, confidence, section) "
            "VALUES (?, 'Method', 'NewMethod', 0.9, '2 Methods')",
            (pid,),
        )
    db.conn.execute(
        "INSERT INTO edges (src_id, dst_id, relation, source_paper, weight) "
        "VALUES ('NewMethod', 'OldMethod', 'challenges', 'new2026', 0.8)"
    )
    db.commit()
    graph = GraphEngine()
    graph.load_from_db(db)
    results = detect_paradigm_shifts(graph, db, decline_threshold=0.4, growth_threshold=2)
    replacements = [r for r in results if r["type"] == "replacement"]
    assert len(replacements) >= 1
    r = replacements[0]
    assert "old_provenance" in r
    assert "new_provenance" in r
    db.close()


def test_paradigm_explosion_includes_provenance():
    """Explosion shifts carry provenance."""
    db = Database(":memory:")
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p1', 'Test', 2026, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) "
        "VALUES ('p1', 'Method', 'GNN', 0.9, '3.1 Methods')"
    )
    # Additional concepts with same label for explosion threshold
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence) "
        "VALUES ('p1', 'Method', 'GNN', 0.8)"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence) "
        "VALUES ('p1', 'Method', 'GNN', 0.8)"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence) "
        "VALUES ('p1', 'Method', 'GNN', 0.8)"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence) "
        "VALUES ('p1', 'Method', 'GNN', 0.8)"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence) "
        "VALUES ('p1', 'Method', 'GNN', 0.8)"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence) "
        "VALUES ('p1', 'Method', 'GNN', 0.8)"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence) "
        "VALUES ('p1', 'Method', 'GNN', 0.8)"
    )
    # Descendant concepts
    for v in ["GNNv2", "GNNv3", "GNNv4"]:
        db.conn.execute(
            "INSERT INTO concepts (local_id, type, label, confidence) "
            "VALUES ('p1', 'Method', ?, 0.8)",
            (v,),
        )
        db.conn.execute(
            "INSERT INTO edges (src_id, dst_id, relation, source_paper) "
            "VALUES ('GNN', ?, 'extends', 'p1')",
            (v,),
        )
    db.commit()
    graph = GraphEngine()
    graph.load_from_db(db)
    results = detect_paradigm_shifts(
        graph, db, concept="GNN", explosion_threshold=1, descendant_threshold=3
    )
    explosions = [r for r in results if r["type"] == "explosion"]
    assert len(explosions) >= 1
    assert "provenance" in explosions[0]
    assert explosions[0]["section"] == "3.1 Methods"
    db.close()


# -- transfer provenance --


def test_transfers_include_provenance():
    """find_transfer_opportunities enriches output with source_provenance."""
    db = Database(":memory:")
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p1', 'Method Paper', 2026, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p2', 'Problem Paper', 2026, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section, node_id) "
        "VALUES ('p1', 'Method', 'GNN', 0.9, '3.1 Graph Methods', '0003.001')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section, node_id) "
        "VALUES ('p2', 'Problem', 'Scalability', 0.85, '5.2 Limitations', '0005.002')"
    )
    db.conn.execute(
        "INSERT INTO edges (src_id, dst_id, relation, source_paper) "
        "VALUES ('GNN', 'Scalability', 'addresses', 'p1')"
    )
    db.commit()
    graph = GraphEngine()
    graph.load_from_db(db)
    results = find_transfer_opportunities(
        db, graph, source_paper_ids=["p1"], target_paper_ids=["p2"], min_confidence=0.0
    )
    assert len(results) >= 1
    t = results[0]
    assert "source_provenance" in t
    assert t["source_section"] == "3.1 Graph Methods"
    assert t["source_paper_id"] == "p1"
    db.close()


# -- difficulty map --


def test_analyze_difficulty_empty():
    """Empty DB returns all empty categories."""
    db = Database(":memory:")
    result = analyze_difficulty(db)
    assert result["limitation"] == []
    assert result["future_work"] == []
    assert result["discussion"] == []
    assert result["uncategorized"] == []
    db.close()


def test_analyze_difficulty_classifies():
    """Gaps are classified by section title semantics."""
    db = Database(":memory:")
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p1', 'Test', 2026, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) "
        "VALUES ('p1', 'Gap', 'Scalability', 0.9, '5.2 Limitations')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) "
        "VALUES ('p1', 'Gap', 'Better Metrics', 0.8, '6 Future Directions')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) "
        "VALUES ('p1', 'Gap', 'Generalizability', 0.7, '7 Discussion')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) "
        "VALUES ('p1', 'Gap', 'Unknown Gap', 0.6, '3 Methods')"
    )
    db.commit()
    result = analyze_difficulty(db)
    assert len(result["limitation"]) == 1
    assert result["limitation"][0]["label"] == "Scalability"
    assert len(result["future_work"]) == 1
    assert result["future_work"][0]["label"] == "Better Metrics"
    assert len(result["discussion"]) == 1
    assert result["discussion"][0]["label"] == "Generalizability"
    assert len(result["uncategorized"]) == 1
    assert result["uncategorized"][0]["label"] == "Unknown Gap"
    db.close()


def test_analyze_difficulty_provenance():
    """Each gap carries provenance info."""
    db = Database(":memory:")
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p1', 'Test', 2026, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section, node_id) "
        "VALUES ('p1', 'Gap', 'Scalability', 0.9, '5.2 Limitations', '0005.002')"
    )
    db.commit()
    result = analyze_difficulty(db)
    g = result["limitation"][0]
    assert g["provenance"] == "[source: 5.2 Limitations of p1]"
    assert g["node_id"] == "0005.002"
    db.close()


# -- knowledge frontier --


def test_analyze_frontier_empty():
    """Frontier on empty DB returns empty result."""
    db = Database(":memory:")
    result = analyze_frontier(db)
    assert result["active_gaps"] == []
    assert result["stale_gaps"] == []
    assert result["debates"] == []
    assert "limitation=0" in result["summary"]
    db.close()


def test_analyze_frontier_with_data():
    """Frontier synthesizes gaps, debates, and paradigm shifts."""
    db = Database(":memory:")
    # Recent paper with gap
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p1', 'Recent', 2026, 'extracted')"
    )
    # Old paper with gap
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) VALUES ('p2', 'Old', 2010, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) "
        "VALUES ('p1', 'Gap', 'Recent Gap', 0.9, '5 Limitations')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) "
        "VALUES ('p2', 'Gap', 'Old Gap', 0.8, '6 Discussion')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence) VALUES ('p1', 'Method', 'A', 0.9)"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence) VALUES ('p1', 'Method', 'B', 0.9)"
    )
    db.conn.execute(
        "INSERT INTO edges (src_id, dst_id, relation, source_paper) "
        "VALUES ('A', 'Recent Gap', 'leaves_open', 'p1')"
    )
    db.conn.execute(
        "INSERT INTO edges (src_id, dst_id, relation, source_paper) "
        "VALUES ('A', 'Debate Target', 'supports', 'p1')"
    )
    db.conn.execute(
        "INSERT INTO edges (src_id, dst_id, relation, source_paper) "
        "VALUES ('B', 'Debate Target', 'challenges', 'p1')"
    )
    db.commit()
    result = analyze_frontier(db)
    assert len(result["active_gaps"]) == 1
    assert result["active_gaps"][0]["label"] == "Recent Gap"
    assert len(result["stale_gaps"]) == 1
    assert result["stale_gaps"][0]["label"] == "Old Gap"
    assert "1 active gaps" in result["summary"]
    assert "1 stale gaps" in result["summary"]
    db.close()


# --- Additional paradigm shift detection tests (TestParadigmShifts class) ---


class TestParadigmShifts:
    """Focused tests for each paradigm shift detection type."""

    def test_replacement_detected(self):
        """A new method that replaces an old one is detected as replacement."""
        db = Database(":memory:")
        # OldMethod: many papers early, few recently → declining
        for i, y in enumerate((2016, 2017, 2017, 2018)):
            pid = f"old_rep_{i}"
            db.conn.execute(
                "INSERT INTO papers (local_id, title, year, status) VALUES (?, ?, ?, 'extracted')",
                (pid, f"Old paper {pid}", y),
            )
            db.conn.execute(
                "INSERT INTO concepts (local_id, type, label, confidence, section) "
                "VALUES (?, 'Method', 'OldMethod', 0.9, '3 Methods')",
                (pid,),
            )
        # OldMethod: 1 paper in 2023 (recent window)
        db.conn.execute(
            "INSERT INTO papers (local_id, title, year, status) VALUES ('old_recent', 'Old recent', 2023, 'extracted')"
        )
        db.conn.execute(
            "INSERT INTO concepts (local_id, type, label, confidence, section) "
            "VALUES ('old_recent', 'Method', 'OldMethod', 0.9, '3 Methods')"
        )
        # NewMethod: 3 papers in 2023-2024 → growing
        for i, y in enumerate((2022, 2023, 2024)):
            pid = f"new_{i}"
            db.conn.execute(
                "INSERT INTO papers (local_id, title, year, status) VALUES (?, ?, ?, 'extracted')",
                (pid, f"New paper {pid}", y),
            )
            db.conn.execute(
                "INSERT INTO concepts (local_id, type, label, confidence, section) "
                "VALUES (?, 'Method', 'NewMethod', 0.95, '2 Methods')",
                (pid,),
            )
        # 'challenges' edge
        db.conn.execute(
            "INSERT INTO edges (src_id, dst_id, relation, source_paper, weight) "
            "VALUES ('NewMethod', 'OldMethod', 'challenges', 'new_0', 0.85)"
        )
        db.commit()
        graph = GraphEngine()
        graph.load_from_db(db)
        results = detect_paradigm_shifts(graph, db, decline_threshold=0.5, growth_threshold=3)
        replacements = [r for r in results if r["type"] == "replacement"]
        assert len(replacements) == 1
        r = replacements[0]
        assert r["old_concept"] == "OldMethod"
        assert r["new_concept"] == "NewMethod"
        assert r["confidence"] == 0.85
        db.close()

    def test_replacement_new_below_growth_threshold(self):
        """New method with too few papers does NOT trigger replacement."""
        db = Database(":memory:")
        # OldMethod: declining
        for i, y in enumerate((2016, 2016, 2016, 2017)):
            pid = f"old_ng_{i}"
            db.conn.execute(
                "INSERT INTO papers (local_id, title, year, status) VALUES (?, ?, ?, 'extracted')",
                (pid, f"Old {pid}", y),
            )
            db.conn.execute(
                "INSERT INTO concepts (local_id, type, label, confidence, section) "
                "VALUES (?, 'Method', 'OldMethod', 0.9, '3 Methods')",
                (pid,),
            )
        # 1 recent old paper
        db.conn.execute(
            "INSERT INTO papers (local_id, title, year, status) VALUES ('old_r', 'Old recent', 2023, 'extracted')"
        )
        db.conn.execute(
            "INSERT INTO concepts (local_id, type, label, confidence, section) "
            "VALUES ('old_r', 'Method', 'OldMethod', 0.9, '3 Methods')"
        )
        # NewMethod: only 2 papers (below default growth_threshold=3)
        for i, y in enumerate((2023, 2024)):
            pid = f"new_{i}"
            db.conn.execute(
                "INSERT INTO papers (local_id, title, year, status) VALUES (?, ?, ?, 'extracted')",
                (pid, f"New {pid}", y),
            )
            db.conn.execute(
                "INSERT INTO concepts (local_id, type, label, confidence, section) "
                "VALUES (?, 'Method', 'NewMethod', 0.95, '2 Methods')",
                (pid,),
            )
        db.conn.execute(
            "INSERT INTO edges (src_id, dst_id, relation, source_paper, weight) "
            "VALUES ('NewMethod', 'OldMethod', 'challenges', 'new_0', 0.9)"
        )
        db.commit()
        graph = GraphEngine()
        graph.load_from_db(db)
        results = detect_paradigm_shifts(graph, db)
        replacements = [r for r in results if r["type"] == "replacement"]
        assert len(replacements) == 0
        db.close()

    def test_explosion_detected(self):
        """Sudden growth in papers about a concept is detected as explosion."""
        db = Database(":memory:")
        # 10 papers all in 2023 about ExplodingConcept
        for i in range(10):
            pid = f"exp_{i}"
            db.conn.execute(
                "INSERT INTO papers (local_id, title, year, status) VALUES (?, ?, 2023, 'extracted')",
                (pid, f"Explosion paper {i}"),
            )
            db.conn.execute(
                "INSERT INTO concepts (local_id, type, label, confidence, section) "
                "VALUES (?, 'Method', 'ExplodingConcept', 0.95, 'method')",
                (pid,),
            )
        # 4 descendant concepts (extends from ExplodingConcept)
        for v in ("ExplV2", "ExplV3", "ExplV4", "ExplV5"):
            db.conn.execute(
                "INSERT INTO concepts (local_id, type, label, confidence) "
                "VALUES ('exp_0', 'Method', ?, 0.85)",
                (v,),
            )
            db.conn.execute(
                "INSERT INTO edges (src_id, dst_id, relation, source_paper) "
                "VALUES ('ExplodingConcept', ?, 'extends', 'exp_0')",
                (v,),
            )
        db.commit()
        graph = GraphEngine()
        graph.load_from_db(db)
        results = detect_paradigm_shifts(graph, db, explosion_threshold=8, descendant_threshold=3)
        explosions = [r for r in results if r["type"] == "explosion"]
        assert len(explosions) == 1
        e = explosions[0]
        assert e["concept"] == "ExplodingConcept"
        assert e["paper_count"] >= 8
        assert len(e["descendants"]) >= 3
        db.close()

    def test_explosion_no_descendants_not_flagged(self):
        """Concept with many papers but no descendants is NOT explosion."""
        db = Database(":memory:")
        for i in range(10):
            pid = f"nodedesc_{i}"
            db.conn.execute(
                "INSERT INTO papers (local_id, title, year, status) VALUES (?, ?, 2023, 'extracted')",
                (pid, f"No desc paper {i}"),
            )
            db.conn.execute(
                "INSERT INTO concepts (local_id, type, label, confidence, section) "
                "VALUES (?, 'Method', 'PopularNoDescendants', 0.95, 'method')",
                (pid,),
            )
        db.commit()
        graph = GraphEngine()
        graph.load_from_db(db)
        results = detect_paradigm_shifts(graph, db, explosion_threshold=8, descendant_threshold=3)
        explosions = [r for r in results if r["type"] == "explosion"]
        assert len(explosions) == 0
        db.close()

    def test_explosion_steady_growth_not_flagged(self):
        """Concept spread across many years (steady growth) is NOT explosion."""
        db = Database(":memory:")
        # Papers across 5 different years — len(year_counts) = 5 > 2 → not explosion
        for y in (2018, 2019, 2020, 2021, 2022, 2023):
            db.conn.execute(
                "INSERT INTO papers (local_id, title, year, status) VALUES (?, ?, ?, 'extracted')",
                (f"steady_{y}", f"Steady paper {y}", y),
            )
            db.conn.execute(
                "INSERT INTO concepts (local_id, type, label, confidence, section) "
                "VALUES (?, 'Method', 'SteadyGrowth', 0.95, 'method')",
                (f"steady_{y}",),
            )
        db.commit()
        graph = GraphEngine()
        graph.load_from_db(db)
        results = detect_paradigm_shifts(graph, db, explosion_threshold=1, descendant_threshold=1)
        explosions = [r for r in results if r["type"] == "explosion"]
        assert len(explosions) == 0
        db.close()

    def test_explosion_concept_filter(self):
        """Explosion detection can be scoped to a single concept."""
        db = Database(":memory:")
        # Two exploding concepts, only filter on one
        for label in ("ConceptA", "ConceptB"):
            for i in range(5):
                pid = f"{label}_{i}"
                db.conn.execute(
                    "INSERT INTO papers (local_id, title, year, status) VALUES (?, ?, 2023, 'extracted')",
                    (pid, f"{label} paper {i}"),
                )
                db.conn.execute(
                    "INSERT INTO concepts (local_id, type, label, confidence, section) "
                    "VALUES (?, 'Method', ?, 0.95, 'method')",
                    (pid, label),
                )
            # descendants for both
            for v in (f"{label}_d1", f"{label}_d2", f"{label}_d3"):
                db.conn.execute(
                    "INSERT INTO concepts (local_id, type, label, confidence) "
                    "VALUES (?, 'Method', ?, 0.8)",
                    (f"{label}_0", v),
                )
                db.conn.execute(
                    "INSERT INTO edges (src_id, dst_id, relation, source_paper) "
                    "VALUES (?, ?, 'extends', ?)",
                    (label, v, f"{label}_0"),
                )
        db.commit()
        graph = GraphEngine()
        graph.load_from_db(db)
        results = detect_paradigm_shifts(
            graph,
            db,
            concept="ConceptA",
            explosion_threshold=3,
            descendant_threshold=2,
        )
        explosions = [r for r in results if r["type"] == "explosion"]
        assert len(explosions) == 1
        assert explosions[0]["concept"] == "ConceptA"
        db.close()

    def test_explosion_paper_ids_scoping(self):
        """Explosion detection scoped to paper_ids only considers matching concepts."""
        db = Database(":memory:")
        # ConceptA papers in workspace
        for i in range(5):
            pid = f"ws_{i}"
            db.conn.execute(
                "INSERT INTO papers (local_id, title, year, status) VALUES (?, ?, 2023, 'extracted')",
                (pid, f"Workspace paper {i}"),
            )
            db.conn.execute(
                "INSERT INTO concepts (local_id, type, label, confidence, section) "
                "VALUES (?, 'Method', 'WorkspaceConcept', 0.95, 'method')",
                (pid,),
            )
        # ConceptB papers NOT in workspace
        for i in range(5):
            pid = f"other_{i}"
            db.conn.execute(
                "INSERT INTO papers (local_id, title, year, status) VALUES (?, ?, 2023, 'extracted')",
                (pid, f"Other paper {i}"),
            )
            db.conn.execute(
                "INSERT INTO concepts (local_id, type, label, confidence, section) "
                "VALUES (?, 'Method', 'OtherConcept', 0.95, 'method')",
                (pid,),
            )
        # descendants for WorkspaceConcept
        for v in ("ws_d1", "ws_d2", "ws_d3"):
            db.conn.execute(
                "INSERT INTO concepts (local_id, type, label, confidence) "
                "VALUES ('ws_0', 'Method', ?, 0.8)",
                (v,),
            )
            db.conn.execute(
                "INSERT INTO edges (src_id, dst_id, relation, source_paper) "
                "VALUES ('WorkspaceConcept', ?, 'extends', 'ws_0')",
                (v,),
            )
        db.commit()
        graph = GraphEngine()
        graph.load_from_db(db)
        workspace_ids = [f"ws_{i}" for i in range(5)]
        results = detect_paradigm_shifts(
            graph,
            db,
            paper_ids=workspace_ids,
            explosion_threshold=3,
            descendant_threshold=2,
        )
        explosions = [r for r in results if r["type"] == "explosion"]
        assert len(explosions) == 1
        assert explosions[0]["concept"] == "WorkspaceConcept"
        db.close()

    def test_no_shift_on_stable_concepts(self):
        """Concepts with steady growth don't trigger any shifts."""
        db = Database(":memory:")
        # StableMethod: 2 papers per year over 4 years — no decline, no explosion
        for y in (2019, 2020, 2021, 2022):
            for j in range(2):
                pid = f"stable_{y}_{j}"
                db.conn.execute(
                    "INSERT INTO papers (local_id, title, year, status) VALUES (?, ?, ?, 'extracted')",
                    (pid, f"Stable {pid}", y),
                )
                db.conn.execute(
                    "INSERT INTO concepts (local_id, type, label, confidence, section) "
                    "VALUES (?, 'Method', 'StableMethod', 0.9, 'method')",
                    (pid,),
                )
        # No challenges/applies edges, no descendants → nothing should fire
        db.commit()
        graph = GraphEngine()
        graph.load_from_db(db)
        results = detect_paradigm_shifts(graph, db, explosion_threshold=8, descendant_threshold=3)
        assert results == []
        db.close()

    def test_cross_domain_cascade_chain(self):
        """Cross-domain: method A applied to B, B extended to C, C to D → cascade."""
        db = Database(":memory:")
        db.conn.execute(
            "INSERT INTO papers (local_id, title, year, status) VALUES ('pA', 'A Paper', 2020, 'extracted')"
        )
        db.conn.execute(
            "INSERT INTO papers (local_id, title, year, status) VALUES ('pB', 'B Paper', 2021, 'extracted')"
        )
        db.conn.execute(
            "INSERT INTO papers (local_id, title, year, status) VALUES ('pC', 'C Paper', 2022, 'extracted')"
        )
        db.conn.execute(
            "INSERT INTO papers (local_id, title, year, status) VALUES ('pD', 'D Paper', 2023, 'extracted')"
        )
        for pid, label in [
            ("pA", "MethodA"),
            ("pB", "MethodB"),
            ("pC", "MethodC"),
            ("pD", "MethodD"),
        ]:
            db.conn.execute(
                "INSERT INTO concepts (local_id, type, label, confidence, section) "
                "VALUES (?, 'Method', ?, 0.95, 'method')",
                (pid, label),
            )
        # applies chain: A -> B -> C -> D
        db.conn.execute(
            "INSERT INTO edges (src_id, dst_id, relation, source_paper, weight) "
            "VALUES ('MethodA', 'MethodB', 'applies', 'pA', 0.9)"
        )
        db.conn.execute(
            "INSERT INTO edges (src_id, dst_id, relation, source_paper, weight) "
            "VALUES ('MethodB', 'MethodC', 'extends', 'pB', 0.8)"
        )
        db.conn.execute(
            "INSERT INTO edges (src_id, dst_id, relation, source_paper, weight) "
            "VALUES ('MethodC', 'MethodD', 'extends', 'pC', 0.8)"
        )
        db.commit()
        graph = GraphEngine()
        graph.load_from_db(db)
        results = detect_paradigm_shifts(graph, db, cascade_threshold=1)
        invasions = [r for r in results if r["type"] == "cross_domain"]
        assert len(invasions) == 1
        iv = invasions[0]
        assert iv["source_concept"] == "MethodA"
        assert iv["target_concept"] == "MethodB"
        assert len(iv["cascade"]) >= 2  # MethodC and MethodD
        db.close()

    def test_cross_domain_no_cascade_not_flagged(self):
        """Applies edge with no further descendants is NOT cross-domain shift."""
        db = Database(":memory:")
        db.conn.execute(
            "INSERT INTO papers (local_id, title, year, status) VALUES ('pA', 'A', 2020, 'extracted')"
        )
        db.conn.execute(
            "INSERT INTO papers (local_id, title, year, status) VALUES ('pB', 'B', 2021, 'extracted')"
        )
        db.conn.execute(
            "INSERT INTO concepts (local_id, type, label, confidence, section) "
            "VALUES ('pA', 'Method', 'MethodA', 0.95, 'method')"
        )
        db.conn.execute(
            "INSERT INTO concepts (local_id, type, label, confidence, section) "
            "VALUES ('pB', 'Problem', 'ProblemX', 0.95, 'intro')"
        )
        db.conn.execute(
            "INSERT INTO edges (src_id, dst_id, relation, source_paper, weight) "
            "VALUES ('MethodA', 'ProblemX', 'applies', 'pA', 0.9)"
        )
        db.commit()
        graph = GraphEngine()
        graph.load_from_db(db)
        # ProblemX has no neighbors beyond MethodA, so cascade is empty
        results = detect_paradigm_shifts(graph, db, cascade_threshold=1)
        invasions = [r for r in results if r["type"] == "cross_domain"]
        assert len(invasions) == 0
        db.close()

    def test_multiple_shift_types_simultaneously(self):
        """All three shift types can be detected in one call."""
        db = Database(":memory:")
        # --- Replacement setup ---
        # OldMethod: declining (many old, few recent)
        for i, y in enumerate((2016, 2016, 2016, 2017)):
            pid = f"old_multi_{i}"
            db.conn.execute(
                "INSERT INTO papers (local_id, title, year, status) VALUES (?, ?, ?, 'extracted')",
                (pid, f"Old {pid}", y),
            )
            db.conn.execute(
                "INSERT INTO concepts (local_id, type, label, confidence, section) "
                "VALUES (?, 'Method', 'OldMethod', 0.9, '3 Methods')",
                (pid,),
            )
        db.conn.execute(
            "INSERT INTO papers (local_id, title, year, status) VALUES ('old_r', 'Old recent', 2023, 'extracted')"
        )
        db.conn.execute(
            "INSERT INTO concepts (local_id, type, label, confidence, section) "
            "VALUES ('old_r', 'Method', 'OldMethod', 0.9, '3 Methods')"
        )
        # NewMethod: growing
        for i, y in enumerate((2023, 2024, 2025)):
            pid = f"new_{i}"
            db.conn.execute(
                "INSERT INTO papers (local_id, title, year, status) VALUES (?, ?, ?, 'extracted')",
                (pid, f"New {pid}", y),
            )
            db.conn.execute(
                "INSERT INTO concepts (local_id, type, label, confidence, section) "
                "VALUES (?, 'Method', 'NewMethod', 0.95, '2 Methods')",
                (pid,),
            )
        db.conn.execute(
            "INSERT INTO edges (src_id, dst_id, relation, source_paper, weight) "
            "VALUES ('NewMethod', 'OldMethod', 'challenges', 'new_0', 0.9)"
        )
        # --- Explosion setup ---
        for i in range(10):
            pid = f"expl_{i}"
            db.conn.execute(
                "INSERT INTO papers (local_id, title, year, status) VALUES (?, ?, 2023, 'extracted')",
                (pid, f"Expl {i}"),
            )
            db.conn.execute(
                "INSERT INTO concepts (local_id, type, label, confidence, section) "
                "VALUES (?, 'Method', 'ExplodingConcept', 0.95, 'method')",
                (pid,),
            )
        for v in ("ExplD1", "ExplD2", "ExplD3"):
            db.conn.execute(
                "INSERT INTO concepts (local_id, type, label, confidence) "
                "VALUES ('expl_0', 'Method', ?, 0.8)",
                (v,),
            )
            db.conn.execute(
                "INSERT INTO edges (src_id, dst_id, relation, source_paper) "
                "VALUES ('ExplodingConcept', ?, 'extends', 'expl_0')",
                (v,),
            )
        # --- Cross-domain setup ---
        db.conn.execute(
            "INSERT INTO papers (local_id, title, year, status) VALUES ('cdA', 'CDA', 2020, 'extracted')"
        )
        db.conn.execute(
            "INSERT INTO papers (local_id, title, year, status) VALUES ('cdB', 'CDB', 2021, 'extracted')"
        )
        db.conn.execute(
            "INSERT INTO papers (local_id, title, year, status) VALUES ('cdC', 'CDC', 2022, 'extracted')"
        )
        for pid, label in [("cdA", "CrossA"), ("cdB", "CrossB"), ("cdC", "CrossC")]:
            db.conn.execute(
                "INSERT INTO concepts (local_id, type, label, confidence, section) "
                "VALUES (?, 'Method', ?, 0.95, 'method')",
                (pid, label),
            )
        db.conn.execute(
            "INSERT INTO edges (src_id, dst_id, relation, source_paper, weight) "
            "VALUES ('CrossA', 'CrossB', 'applies', 'cdA', 0.9)"
        )
        db.conn.execute(
            "INSERT INTO edges (src_id, dst_id, relation, source_paper, weight) "
            "VALUES ('CrossB', 'CrossC', 'extends', 'cdB', 0.8)"
        )
        db.commit()
        graph = GraphEngine()
        graph.load_from_db(db)
        results = detect_paradigm_shifts(
            graph,
            db,
            decline_threshold=0.5,
            growth_threshold=3,
            explosion_threshold=8,
            descendant_threshold=3,
            cascade_threshold=1,
        )
        types_found = {r["type"] for r in results}
        assert "replacement" in types_found
        assert "explosion" in types_found
        assert "cross_domain" in types_found
        db.close()


class TestAnalyzeDifficulty:
    """Tests for analyze_difficulty keyword classification."""

    def test_weakness_keyword_classified_as_limitation(self):
        """Section with 'weakness' → limitation category."""
        db = Database(":memory:")
        db.conn.execute(
            "INSERT INTO papers (local_id, title, year, status) VALUES ('p1', 'Test', 2026, 'extracted')"
        )
        db.conn.execute(
            "INSERT INTO concepts (local_id, type, label, confidence, section) "
            "VALUES ('p1', 'Gap', 'Robustness Gap', 0.9, '4.1 Weaknesses')"
        )
        db.commit()
        result = analyze_difficulty(db)
        assert len(result["limitation"]) == 1
        assert result["limitation"][0]["label"] == "Robustness Gap"
        db.close()

    def test_shortcoming_keyword_classified_as_limitation(self):
        """Section with 'shortcoming' → limitation category."""
        db = Database(":memory:")
        db.conn.execute(
            "INSERT INTO papers (local_id, title, year, status) VALUES ('p1', 'Test', 2026, 'extracted')"
        )
        db.conn.execute(
            "INSERT INTO concepts (local_id, type, label, confidence, section) "
            "VALUES ('p1', 'Gap', 'Generalization Gap', 0.8, '5.3 Shortcomings')"
        )
        db.commit()
        result = analyze_difficulty(db)
        assert len(result["limitation"]) == 1
        assert result["limitation"][0]["label"] == "Generalization Gap"
        db.close()

    def test_direction_keyword_classified_as_future_work(self):
        """Section with 'direction' → future_work category."""
        db = Database(":memory:")
        db.conn.execute(
            "INSERT INTO papers (local_id, title, year, status) VALUES ('p1', 'Test', 2026, 'extracted')"
        )
        db.conn.execute(
            "INSERT INTO concepts (local_id, type, label, confidence, section) "
            "VALUES ('p1', 'Gap', 'Next Steps', 0.85, '7 Future Research Directions')"
        )
        db.commit()
        result = analyze_difficulty(db)
        assert len(result["future_work"]) == 1
        assert result["future_work"][0]["label"] == "Next Steps"
        db.close()

    def test_open_problem_keyword_classified_as_future_work(self):
        """Section with 'open problem' → future_work category."""
        db = Database(":memory:")
        db.conn.execute(
            "INSERT INTO papers (local_id, title, year, status) VALUES ('p1', 'Test', 2026, 'extracted')"
        )
        db.conn.execute(
            "INSERT INTO concepts (local_id, type, label, confidence, section) "
            "VALUES ('p1', 'Gap', 'Convergence', 0.7, 'Open Problems in GANs')"
        )
        db.commit()
        result = analyze_difficulty(db)
        assert len(result["future_work"]) == 1
        assert result["future_work"][0]["label"] == "Convergence"
        db.close()

    def test_open_question_keyword_classified_as_future_work(self):
        """Section with 'open question' → future_work category."""
        db = Database(":memory:")
        db.conn.execute(
            "INSERT INTO papers (local_id, title, year, status) VALUES ('p1', 'Test', 2026, 'extracted')"
        )
        db.conn.execute(
            "INSERT INTO concepts (local_id, type, label, confidence, section) "
            "VALUES ('p1', 'Gap', 'Scalability', 0.75, 'Open Questions')"
        )
        db.commit()
        result = analyze_difficulty(db)
        assert len(result["future_work"]) == 1
        assert result["future_work"][0]["label"] == "Scalability"
        db.close()

    def test_conclusion_keyword_classified_as_discussion(self):
        """Section with 'conclusion' → discussion category."""
        db = Database(":memory:")
        db.conn.execute(
            "INSERT INTO papers (local_id, title, year, status) VALUES ('p1', 'Test', 2026, 'extracted')"
        )
        db.conn.execute(
            "INSERT INTO concepts (local_id, type, label, confidence, section) "
            "VALUES ('p1', 'Gap', 'Tradeoff', 0.8, '8 Conclusion')"
        )
        db.commit()
        result = analyze_difficulty(db)
        assert len(result["discussion"]) == 1
        assert result["discussion"][0]["label"] == "Tradeoff"
        db.close()

    def test_non_gap_concepts_ignored(self):
        """Only Gap-type concepts are classified; Method/Problem are ignored."""
        db = Database(":memory:")
        db.conn.execute(
            "INSERT INTO papers (local_id, title, year, status) VALUES ('p1', 'Test', 2026, 'extracted')"
        )
        db.conn.execute(
            "INSERT INTO concepts (local_id, type, label, confidence, section) "
            "VALUES ('p1', 'Method', 'GNN', 0.9, '3 Methods')"
        )
        db.conn.execute(
            "INSERT INTO concepts (local_id, type, label, confidence, section) "
            "VALUES ('p1', 'Problem', 'Classification', 0.8, '1 Intro')"
        )
        db.commit()
        result = analyze_difficulty(db)
        assert result["limitation"] == []
        assert result["future_work"] == []
        assert result["discussion"] == []
        assert result["uncategorized"] == []
        db.close()

    def test_null_section_classified_as_uncategorized(self):
        """Gap with NULL section → uncategorized."""
        db = Database(":memory:")
        db.conn.execute(
            "INSERT INTO papers (local_id, title, year, status) VALUES ('p1', 'Test', 2026, 'extracted')"
        )
        db.conn.execute(
            "INSERT INTO concepts (local_id, type, label, confidence) "
            "VALUES ('p1', 'Gap', 'Mystery Gap', 0.7)"
        )
        db.commit()
        result = analyze_difficulty(db)
        assert len(result["uncategorized"]) == 1
        assert result["uncategorized"][0]["label"] == "Mystery Gap"
        db.close()


class TestAnalyzeFrontier:
    """Tests for analyze_frontier composite report."""

    def test_frontier_includes_difficulty_subdict(self):
        """Frontier report contains the difficulty sub-dict."""
        db = Database(":memory:")
        db.conn.execute(
            "INSERT INTO papers (local_id, title, year, status) VALUES ('p1', 'Test', 2026, 'extracted')"
        )
        db.conn.execute(
            "INSERT INTO concepts (local_id, type, label, confidence, section) "
            "VALUES ('p1', 'Gap', 'Scalability', 0.9, '5 Limitations')"
        )
        db.commit()
        result = analyze_frontier(db)
        assert "difficulty" in result
        assert len(result["difficulty"]["limitation"]) == 1
        db.close()

    def test_frontier_stale_gaps_classified(self):
        """Gaps from old papers (beyond 3-year window) go to stale_gaps."""
        db = Database(":memory:")
        db.conn.execute(
            "INSERT INTO papers (local_id, title, year, status) VALUES ('p1', 'Old', 2010, 'extracted')"
        )
        db.conn.execute(
            "INSERT INTO papers (local_id, title, year, status) VALUES ('p2', 'Recent', 2025, 'extracted')"
        )
        db.conn.execute(
            "INSERT INTO concepts (local_id, type, label, confidence, section) "
            "VALUES ('p1', 'Gap', 'Old Gap', 0.8, '5 Limitations')"
        )
        db.conn.execute(
            "INSERT INTO concepts (local_id, type, label, confidence, section) "
            "VALUES ('p2', 'Gap', 'Recent Gap', 0.9, '5 Limitations')"
        )
        db.commit()
        result = analyze_frontier(db)
        assert len(result["stale_gaps"]) == 1
        assert result["stale_gaps"][0]["label"] == "Old Gap"
        assert len(result["active_gaps"]) == 1
        assert result["active_gaps"][0]["label"] == "Recent Gap"
        db.close()

    def test_frontier_gap_without_year_omitted(self):
        """Gaps on papers with NULL year are excluded from active/stale lists."""
        db = Database(":memory:")
        db.conn.execute(
            "INSERT INTO papers (local_id, title, year, status) VALUES ('p1', 'No Year', NULL, 'extracted')"
        )
        db.conn.execute(
            "INSERT INTO concepts (local_id, type, label, confidence, section) "
            "VALUES ('p1', 'Gap', 'Orphan Gap', 0.8, '5 Limitations')"
        )
        db.commit()
        result = analyze_frontier(db)
        # Paper has NULL year → gap not in active_gaps or stale_gaps
        gap_labels = [g["label"] for g in result["active_gaps"] + result["stale_gaps"]]
        assert "Orphan Gap" not in gap_labels
        db.close()
