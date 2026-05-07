from drbrain.graph.engine import GraphEngine
from drbrain.graph.genealogy import evolve_concept, format_tree
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
