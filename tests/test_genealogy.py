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
    assert len(result["children"]) == 1
    assert result["children"][0]["local_id"] == "p1"

    # generations=2: p1 and p2
    result = trace_descendants(db, graph, "p0", generations=2)
    children = result["children"]
    assert len(children) == 1
    assert children[0]["local_id"] == "p1"
    grandchildren = children[0]["children"]
    assert len(grandchildren) == 1
    assert grandchildren[0]["local_id"] == "p2"

    db.close()
