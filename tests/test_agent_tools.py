"""Tests for agent_tools.py — 9 tool functions + execute_tool dispatch."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

from drbrain.extractor.agent_tools import (
    TOOL_DEFINITIONS,
    TOOL_HANDLERS,
    execute_tool,
    find_path,
    get_document_structure,
    get_neighbors,
    get_raptor_summaries,
    get_section_content,
    kg_validate,
    search_concepts,
    search_tree,
)

# ═══════════════════════════════════════════════════════════════════════════════
# TOOL_DEFINITIONS — structural check
# ═══════════════════════════════════════════════════════════════════════════════


def test_tool_definitions_count():
    """TOOL_DEFINITIONS has exactly 7 tool entries."""
    assert len(TOOL_DEFINITIONS) == 7


def test_tool_definitions_names():
    """All expected tool names are present."""
    names = {t["function"]["name"] for t in TOOL_DEFINITIONS}
    expected = {
        "search_concepts",
        "get_neighbors",
        "find_path",
        "get_document_structure",
        "get_section_content",
        "search_tree",
        "get_raptor_summaries",
    }
    assert names == expected


# ═══════════════════════════════════════════════════════════════════════════════
# search_concepts
# ═══════════════════════════════════════════════════════════════════════════════


def test_search_concepts_none_db():
    """search_concepts returns [] when db is None."""
    assert search_concepts(None, query="test") == []


def test_search_concepts_real_db(tmp_db):
    """search_concepts returns matching concepts from a real DB."""
    db = tmp_db
    db.insert_paper("p1", "Attention Paper", 2024, "uploaded")
    # Type must be capitalized per CHECK constraint
    db.insert_concept("p1", "Method", "transformer", 0.95, year=2024)
    db.commit()

    results = search_concepts(db, query="transformer", limit=5)
    assert isinstance(results, list)
    assert len(results) >= 1
    labels = {r["label"] for r in results}
    assert "transformer" in labels
    assert "score" in results[0]


# ═══════════════════════════════════════════════════════════════════════════════
# get_neighbors
# ═══════════════════════════════════════════════════════════════════════════════


def test_get_neighbors_none_graph():
    assert get_neighbors(None, node="A") == []


def test_get_neighbors_real_graph():
    from drbrain.graph.engine import GraphEngine

    g = GraphEngine()
    g.add_edge("A", "B", "related_to", "p1")
    g.add_edge("B", "C", "extends", "p2")

    results = get_neighbors(g, node="B", hops=1, direction="both")
    assert isinstance(results, list)
    assert len(results) >= 1
    for r in results:
        assert "target" in r
        assert "source" in r
        assert "distance" in r
        assert "path" in r


# ═══════════════════════════════════════════════════════════════════════════════
# find_path
# ═══════════════════════════════════════════════════════════════════════════════


def test_find_path_none_graph():
    assert find_path(None, src="A", dst="B") is None


def test_find_path_missing_node():
    from drbrain.graph.engine import GraphEngine

    g = GraphEngine()
    g.add_edge("A", "B", "cites", "p1")
    assert find_path(g, src="A", dst="C") is None


def test_find_path_exists():
    from drbrain.graph.engine import GraphEngine

    g = GraphEngine()
    g.add_edge("A", "B", "cites", "p1")
    g.add_edge("B", "C", "extends", "p1")

    result = find_path(g, src="A", dst="C")
    assert result is not None
    assert "path" in result
    assert "length" in result
    assert result["length"] == 2
    assert result["path"] == ["A", "B", "C"]


def test_find_path_disconnected():
    from drbrain.graph.engine import GraphEngine

    g = GraphEngine()
    g.add_edge("A", "B", "cites", "p1")
    g.add_edge("C", "D", "cites", "p2")

    result = find_path(g, src="A", dst="C")
    assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# get_document_structure
# ═══════════════════════════════════════════════════════════════════════════════


def test_get_document_structure_none_dir():
    assert get_document_structure(None, paper_id="p1") == []


def test_get_document_structure_missing_paper():
    with tempfile.TemporaryDirectory() as td:
        papers_dir = Path(td)
        assert get_document_structure(papers_dir, paper_id="nonexistent") == []


def test_get_document_structure_valid():
    """get_document_structure returns nested tree skeleton."""
    with tempfile.TemporaryDirectory() as td:
        papers_dir = Path(td) / "papers"
        papers_dir.mkdir()

        paper_dir = papers_dir / "test-paper"
        paper_dir.mkdir()

        tree = {
            "structure": [
                {
                    "node_id": "n1",
                    "title": "Introduction",
                    "nodes": [
                        {"node_id": "n1-1", "title": "Background"},
                    ],
                },
                {"node_id": "n2", "title": "Methods"},
            ],
        }
        (paper_dir / "tree.json").write_text(json.dumps(tree), encoding="utf-8")

        result = get_document_structure(papers_dir, paper_id="test-paper")
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["node_id"] == "n1"
        assert result[0]["title"] == "Introduction"
        assert len(result[0]["children"]) == 1
        assert result[0]["children"][0]["node_id"] == "n1-1"


# ═══════════════════════════════════════════════════════════════════════════════
# get_section_content
# ═══════════════════════════════════════════════════════════════════════════════


def test_get_section_content_none_dir():
    assert get_section_content(None, paper_id="p1", node_id="n1") == ""


def test_get_section_content_missing_paper():
    with tempfile.TemporaryDirectory() as td:
        assert get_section_content(Path(td), paper_id="missing", node_id="n1") == ""


def test_get_section_content_valid():
    with tempfile.TemporaryDirectory() as td:
        papers_dir = Path(td) / "papers"
        papers_dir.mkdir()
        paper_dir = papers_dir / "test-paper"
        paper_dir.mkdir()

        (paper_dir / "raw.md").write_text(
            "# Introduction\n\nSome intro text.\n\n# Methods\n\nMethods content.",
            encoding="utf-8",
        )
        tree = {
            "structure": [
                {"node_id": "n1", "title": "Introduction", "line_num": 1},
                {"node_id": "n2", "title": "Methods", "line_num": 5},
            ],
        }
        (paper_dir / "tree.json").write_text(json.dumps(tree), encoding="utf-8")

        content = get_section_content(papers_dir, paper_id="test-paper", node_id="n1")
        assert isinstance(content, str)
        assert "intro text" in content.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# search_tree
# ═══════════════════════════════════════════════════════════════════════════════


def test_search_tree_none_db():
    assert search_tree(None, query="test") == []


def test_search_tree_with_mock(tmp_db):
    """search_tree delegates to query_cross_paper.

    Must patch at the import site inside the function body:
    ``drbrain.query.tree_retrieval.query_cross_paper``.
    """
    fake_result = [
        {"node_id": "n1", "paper_id": "p1", "score": 0.95, "tree_layer": "pageindex"},
    ]
    with mock.patch(
        "drbrain.query.tree_retrieval.query_cross_paper",
        return_value=fake_result,
    ):
        result = search_tree(tmp_db, query="attention")
        assert result == fake_result


# ═══════════════════════════════════════════════════════════════════════════════
# get_raptor_summaries
# ═══════════════════════════════════════════════════════════════════════════════


def test_get_raptor_summaries_none_db():
    assert get_raptor_summaries(None, paper_id="p1") == []


def test_get_raptor_summaries_real_db(tmp_db):
    """get_raptor_summaries returns summaries ordered by tree_layer."""
    db = tmp_db
    db.conn.execute(
        "INSERT INTO tree_summaries (node_id, paper_id, summary_text, source_node_ids, tree_layer) "
        "VALUES (?, ?, ?, ?, ?)",
        ("r_p1_L2_def", "p1", "Higher-level summary.", json.dumps(["r_p1_L1_abc"]), 2),
    )
    db.conn.execute(
        "INSERT INTO tree_summaries (node_id, paper_id, summary_text, source_node_ids, tree_layer) "
        "VALUES (?, ?, ?, ?, ?)",
        ("r_p1_L1_abc", "p1", "Layer 1 summary.", json.dumps(["n1", "n2"]), 1),
    )
    db.commit()

    results = get_raptor_summaries(db, paper_id="p1")
    assert len(results) == 2
    assert results[0]["tree_layer"] == 1
    assert results[1]["tree_layer"] == 2
    assert results[0]["summary_text"] == "Layer 1 summary."
    assert isinstance(results[0]["source_node_ids"], list)


def test_get_raptor_summaries_empty(tmp_db):
    """Returns empty list for paper with no summaries."""
    assert get_raptor_summaries(tmp_db, paper_id="nonexistent") == []


# ═══════════════════════════════════════════════════════════════════════════════
# kg_validate
# ═══════════════════════════════════════════════════════════════════════════════


def test_kg_validate_none_graph():
    """kg_validate returns consistent=True when graph is None."""
    result = kg_validate("Transformers are great", db=None, graph=None)
    assert result["consistent"] is True
    assert result["violations"] == []
    assert result["patterns"] == []


def test_kg_validate_empty_graph():
    from drbrain.graph.engine import GraphEngine

    g = GraphEngine()  # empty graph
    result = kg_validate("Some hypothesis", db=None, graph=g)
    assert result["consistent"] is True


def test_kg_validate_single_entity():
    """kg_validate returns early when fewer than 2 entities are found."""
    from drbrain.graph.engine import GraphEngine

    g = GraphEngine()
    g.add_edge("Transformer", "Attention", "uses", "p1")

    result = kg_validate("Transformer is a method.", db=None, graph=g)
    assert result["consistent"] is True
    assert result["violations"] == []


def test_kg_validate_debate_pattern():
    """kg_validate detects debate pattern when two nodes challenge same target."""
    from drbrain.graph.engine import GraphEngine

    g = GraphEngine()
    g.add_edge("ModelA", "TheoryX", "challenges", "p1")
    g.add_edge("ModelB", "TheoryX", "challenges", "p2")

    result = kg_validate("ModelA and ModelB both challenge TheoryX", db=None, graph=g)
    assert result["consistent"] is True  # challenges is not asymmetric
    assert any(p["type"] == "debate" for p in result["patterns"])
    debate = next(p for p in result["patterns"] if p["type"] == "debate")
    assert "TheoryX" in debate["description"]


def test_kg_validate_gap_pattern(tmp_db):
    """kg_validate detects gap pattern when two mentioned entities have no edge."""
    from drbrain.graph.engine import GraphEngine

    db = tmp_db
    # concepts.local_id has FK referencing papers — insert papers first
    db.insert_paper("p1", "Paper One", 2024, "uploaded")
    db.insert_paper("p2", "Paper Two", 2024, "uploaded")
    db.commit()

    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, first_seen, last_seen) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("p1", "Method", "MethodX", 0.9, 2024, 2024),
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, first_seen, last_seen) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("p2", "Problem", "ProblemY", 0.9, 2024, 2024),
    )
    db.commit()

    g = GraphEngine()
    # Two entities exist but with no direct edge between them
    g.add_edge("MethodX", "Other", "uses", "p1")
    g.add_edge("ProblemY", "Other2", "addresses", "p2")

    result = kg_validate("MethodX addresses ProblemY", db=db, graph=g)
    gaps = [p for p in result["patterns"] if p["type"] == "gap"]
    assert len(gaps) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL_HANDLERS mapping — structural check
# ═══════════════════════════════════════════════════════════════════════════════


def test_tool_handlers_has_all_tools():
    """TOOL_HANDLERS maps all 7 tool names to functions."""
    expected = {
        "search_concepts",
        "get_neighbors",
        "find_path",
        "get_document_structure",
        "get_section_content",
        "search_tree",
        "get_raptor_summaries",
    }
    assert set(TOOL_HANDLERS.keys()) == expected


# ═══════════════════════════════════════════════════════════════════════════════
# execute_tool — dispatch
# ═══════════════════════════════════════════════════════════════════════════════

# NOTE: ``execute_tool`` looks up the handler from ``TOOL_HANDLERS``, which stores
# direct references to the real functions.  Patching the module-level name does
# NOT affect the dict entry.  We patch ``TOOL_HANDLERS`` entries instead.


def test_execute_tool_unknown_name():
    """execute_tool returns [] for unknown tool name."""
    result = execute_tool("nonexistent", {}, db=None, graph=None)
    assert result == []


def test_execute_tool_search_concepts():
    """execute_tool routes search_concepts with db kwarg."""
    fake_fn = mock.MagicMock(return_value=[{"label": "X"}])
    with mock.patch.dict(TOOL_HANDLERS, {"search_concepts": fake_fn}):
        result = execute_tool("search_concepts", {"query": "test", "limit": 3}, db="fake_db")
    fake_fn.assert_called_once_with("fake_db", query="test", limit=3)
    assert result == [{"label": "X"}]


def test_execute_tool_get_neighbors():
    """execute_tool routes get_neighbors with graph kwarg."""
    fake_fn = mock.MagicMock(return_value=[{"target": "B"}])
    with mock.patch.dict(TOOL_HANDLERS, {"get_neighbors": fake_fn}):
        fake_graph = mock.MagicMock()
        result = execute_tool("get_neighbors", {"node": "A"}, graph=fake_graph)
    fake_fn.assert_called_once_with(fake_graph, node="A")
    assert result == [{"target": "B"}]


def test_execute_tool_find_path():
    """execute_tool routes find_path with graph kwarg."""
    fake_fn = mock.MagicMock(return_value={"path": ["A", "B"], "length": 1})
    with mock.patch.dict(TOOL_HANDLERS, {"find_path": fake_fn}):
        fake_graph = mock.MagicMock()
        result = execute_tool("find_path", {"src": "A", "dst": "B"}, graph=fake_graph)
    fake_fn.assert_called_once_with(fake_graph, src="A", dst="B")
    assert result["length"] == 1


def test_execute_tool_get_document_structure():
    """execute_tool routes get_document_structure with papers_dir kwarg."""
    fake_fn = mock.MagicMock(return_value=[{"node_id": "n1"}])
    with mock.patch.dict(TOOL_HANDLERS, {"get_document_structure": fake_fn}):
        result = execute_tool(
            "get_document_structure",
            {"paper_id": "p1"},
            papers_dir=Path("/fake/papers"),
        )
    fake_fn.assert_called_once_with(Path("/fake/papers"), paper_id="p1")
    assert result == [{"node_id": "n1"}]


def test_execute_tool_get_section_content():
    """execute_tool routes get_section_content with papers_dir kwarg."""
    fake_fn = mock.MagicMock(return_value="section text")
    with mock.patch.dict(TOOL_HANDLERS, {"get_section_content": fake_fn}):
        result = execute_tool(
            "get_section_content",
            {"paper_id": "p1", "node_id": "n1"},
            papers_dir=Path("/fake/papers"),
        )
    fake_fn.assert_called_once_with(Path("/fake/papers"), paper_id="p1", node_id="n1")
    assert result == "section text"


def test_execute_tool_search_tree():
    """execute_tool routes search_tree with db kwarg."""
    fake_fn = mock.MagicMock(return_value=[{"node_id": "n1"}])
    with mock.patch.dict(TOOL_HANDLERS, {"search_tree": fake_fn}):
        result = execute_tool("search_tree", {"query": "test"}, db="fake_db")
    fake_fn.assert_called_once_with("fake_db", query="test")
    assert result == [{"node_id": "n1"}]


def test_execute_tool_get_raptor_summaries():
    """execute_tool routes get_raptor_summaries with db kwarg."""
    fake_fn = mock.MagicMock(return_value=[{"summary_text": "summary"}])
    with mock.patch.dict(TOOL_HANDLERS, {"get_raptor_summaries": fake_fn}):
        result = execute_tool("get_raptor_summaries", {"paper_id": "p1"}, db="fake_db")
    fake_fn.assert_called_once_with("fake_db", paper_id="p1")
    assert result == [{"summary_text": "summary"}]


# ── End-to-end execute_tool tests with real dependencies ────────────────


def test_execute_tool_real_search_concepts(tmp_db):
    """execute_tool search_concepts end-to-end with a real DB."""
    db = tmp_db
    db.insert_paper("p1", "Unrelated Title", 2024, "uploaded")
    db.insert_concept("p1", "Method", "transformer", 0.95, year=2024)
    db.commit()

    result = execute_tool("search_concepts", {"query": "transformer", "limit": 5}, db=db)
    assert isinstance(result, list)
    assert len(result) >= 1
    labels = {r["label"] for r in result}
    assert "transformer" in labels


def test_execute_tool_real_get_neighbors():
    """execute_tool get_neighbors end-to-end with a real graph."""
    from drbrain.graph.engine import GraphEngine

    g = GraphEngine()
    g.add_edge("A", "B", "cites", "p1")
    result = execute_tool("get_neighbors", {"node": "A", "hops": 1}, graph=g)
    assert isinstance(result, list)
    assert len(result) >= 1


def test_execute_tool_real_find_path():
    """execute_tool find_path end-to-end with a real graph."""
    from drbrain.graph.engine import GraphEngine

    g = GraphEngine()
    g.add_edge("A", "B", "cites", "p1")
    g.add_edge("B", "C", "extends", "p1")
    result = execute_tool("find_path", {"src": "A", "dst": "C"}, graph=g)
    assert result is not None
    assert result["length"] == 2


def test_execute_tool_real_get_document_structure():
    """execute_tool get_document_structure end-to-end with real files."""
    with tempfile.TemporaryDirectory() as td:
        papers_dir = Path(td) / "papers"
        papers_dir.mkdir()
        paper_dir = papers_dir / "test-paper"
        paper_dir.mkdir()

        tree = {"structure": [{"node_id": "n1", "title": "Intro"}]}
        (paper_dir / "tree.json").write_text(json.dumps(tree), encoding="utf-8")

        result = execute_tool(
            "get_document_structure",
            {"paper_id": "test-paper"},
            papers_dir=papers_dir,
        )
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["node_id"] == "n1"


def test_execute_tool_real_get_section_content():
    """execute_tool get_section_content end-to-end with real files."""
    with tempfile.TemporaryDirectory() as td:
        papers_dir = Path(td) / "papers"
        papers_dir.mkdir()
        paper_dir = papers_dir / "test-paper"
        paper_dir.mkdir()

        (paper_dir / "raw.md").write_text("# Intro\n\nHello.\n", encoding="utf-8")
        tree = {"structure": [{"node_id": "n1", "title": "Intro", "line_num": 1}]}
        (paper_dir / "tree.json").write_text(json.dumps(tree), encoding="utf-8")

        result = execute_tool(
            "get_section_content",
            {"paper_id": "test-paper", "node_id": "n1"},
            papers_dir=papers_dir,
        )
        assert isinstance(result, str)
        assert "Hello" in result


def test_execute_tool_real_get_raptor_summaries(tmp_db):
    """execute_tool get_raptor_summaries end-to-end with a real DB."""
    db = tmp_db
    db.conn.execute(
        "INSERT INTO tree_summaries (node_id, paper_id, summary_text, source_node_ids, tree_layer) "
        "VALUES (?, ?, ?, ?, ?)",
        ("r_p1_L1_abc", "p1", "Summary text.", json.dumps(["n1"]), 1),
    )
    db.commit()

    result = execute_tool("get_raptor_summaries", {"paper_id": "p1"}, db=db)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["summary_text"] == "Summary text."
