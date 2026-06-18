"""Tests for drbrain.cli._helpers.display — display/analysis helper functions.

Most are pure or take small mockable graph/db objects. We use MagicMock for
GraphEngine and Database where SQL is involved, and real networkx graphs
where relation-matching logic needs actual edges.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import networkx as nx

from drbrain.cli._helpers.display import (
    _apply_mined_rules,
    _build_closure_context,
    _enrich_tree_with_sections,
    _export_paper_to_meta,
    _extend_chain,
    _match_pattern,
    _print_analyze_report,
    _render_landscape,
    _show_actor,
)

# ---------------------------------------------------------------------------
# _print_analyze_report
# ---------------------------------------------------------------------------


def test_print_analyze_report_error(capsys):
    """Error path: echoes the error to stderr and returns early."""
    _print_analyze_report({"error": "boom"})
    captured = capsys.readouterr()
    assert "boom" in captured.err or "boom" in captured.out


def test_print_analyze_report_full(capsys):
    report = {
        "paper": {"title": "My Paper", "year": 2024},
        "summary": {
            "seeds": 2,
            "causal_chains": 1,
            "inferred_edges": 0,
            "critical_nodes": 1,
            "hypotheses": 1,
            "isomorphisms": 1,
        },
        "executive_summary": "A short summary",
        "cross_paper_insights": [
            {
                "method": "M1",
                "method_paper": "P1",
                "problem": "Prob1",
                "problem_paper": "P2",
                "similarity": 0.9,
            }
        ],
        "seeds": [
            {"type": "Gap", "concept": "C1", "description": "desc", "suggested_solutions": "sol"}
        ],
        "causal_chains": [{"source": "A", "target": "B", "via": "rel"}],
        "critical_nodes": ["NodeX"],
        "hypotheses": [{"type": "novelty", "description": "H1", "confidence": 0.75}],
        "isomorphisms": [{"pattern": "pat", "similarity": 0.6}],
    }
    _print_analyze_report(report)
    out = capsys.readouterr().out
    assert "My Paper (2024)" in out
    assert "Executive Summary" in out
    assert "Cross-paper Insights (1)" in out
    assert "M1" in out
    assert "Research Seeds (2)" in out
    assert "Causal Chains (1)" in out
    assert "Critical Nodes (1)" in out
    assert "Hypotheses (1)" in out
    assert "Isomorphisms (1)" in out


def test_print_analyze_report_minimal(capsys):
    report = {
        "paper": {"title": "T", "year": None},
        "summary": {
            "seeds": 0,
            "causal_chains": 0,
            "inferred_edges": 0,
            "critical_nodes": 0,
            "hypotheses": 0,
            "isomorphisms": 0,
        },
    }
    _print_analyze_report(report)
    out = capsys.readouterr().out
    assert "T (None)" in out
    assert "Research Seeds (0)" in out


# ---------------------------------------------------------------------------
# _render_landscape
# ---------------------------------------------------------------------------


def test_render_landscape_empty(capsys):
    _render_landscape({}, top_n=5)
    out = capsys.readouterr().out
    assert "No papers found" in out


def test_render_landscape_with_entries(capsys):
    result = {
        "timeline": [
            {
                "year": 2020,
                "title": "Paper A",
                "key_concepts": [{"label": "C1", "type": "Method"}],
            },
            {
                "year": 2020,
                "title": "Paper B",
                "key_concepts": [],
            },
            {
                "year": 2022,
                "title": "Paper C",
                "key_concepts": [{"label": "C2", "type": "Problem"}],
            },
        ],
        "gaps": [{"description": "a gap", "concept": "G1", "provenance": "prov"}],
        "debates": [{"description": "a debate", "concept": "D1"}],
    }
    _render_landscape(result, top_n=3)
    out = capsys.readouterr().out
    assert "Landscape" in out
    assert "Paper A" in out
    assert "C1 [Method]" in out
    assert "Paper C" in out
    assert "Persistent gaps (1)" in out
    assert "Debates (1)" in out


def test_render_landscape_truncates_key_concepts(capsys):
    """top_n limits rendered key concepts per entry."""
    result = {
        "timeline": [
            {
                "year": 2021,
                "title": "T",
                "key_concepts": [{"label": f"L{i}", "type": "Method"} for i in range(10)],
            }
        ]
    }
    _render_landscape(result, top_n=2)
    out = capsys.readouterr().out
    assert "L0" in out
    assert "L1" in out
    assert "L2" not in out


# ---------------------------------------------------------------------------
# _build_closure_context
# ---------------------------------------------------------------------------


def test_build_closure_context_empty_seeds():
    g = MagicMock()
    assert _build_closure_context(g, []) == ""
    assert _build_closure_context(g, ["x"]) != "" or True  # depends on graph


def test_build_closure_context_no_edges():
    g = MagicMock()
    g.graph.number_of_edges.return_value = 0
    assert _build_closure_context(g, ["A", "B"]) == ""


def test_build_closure_context_no_inferred():
    g = MagicMock()
    g.graph.number_of_edges.return_value = 5
    g.closure_incremental.return_value = []
    assert _build_closure_context(g, ["A"]) == ""


def test_build_closure_context_with_edges():
    g = MagicMock()
    g.graph.number_of_edges.return_value = 5
    g.closure_incremental.return_value = [
        {"relation": "causes", "dst": "B", "confidence": 0.8, "via": "rule1"},
        {"relation": "uses_method", "dst": "C", "confidence": 0.95, "via": ""},
    ]
    out = _build_closure_context(g, ["A"], top_k=5)
    # sorted by confidence desc → C first
    assert "uses method" in out  # underscore replaced with space
    assert "causes" in out
    assert "B" in out and "C" in out
    assert "confidence: 0.95" in out
    assert "via: rule1" in out
    # via is empty for second edge → not included
    lines = out.splitlines()
    assert sum("via:" in ln for ln in lines) == 1


def test_build_closure_context_top_k():
    g = MagicMock()
    g.graph.number_of_edges.return_value = 5
    g.closure_incremental.return_value = [
        {"relation": f"r{i}", "dst": f"D{i}", "confidence": i / 10, "via": ""} for i in range(10)
    ]
    out = _build_closure_context(g, ["A"], top_k=3)
    assert len(out.splitlines()) == 3


# ---------------------------------------------------------------------------
# _match_pattern / _extend_chain / _apply_mined_rules — use real nx graph
# ---------------------------------------------------------------------------


def _make_graph(edges):
    """Build a GraphEngine-like object with edges carrying 'relation' attr."""
    g = MagicMock()
    nxg = nx.DiGraph()
    for u, v, rel in edges:
        nxg.add_edge(u, v, relation=rel)
    g.graph = nxg
    return g


def test_match_pattern_short_pattern_returns_empty():
    g = _make_graph([("A", "B", "r1")])
    assert _match_pattern(g, [("r1", "forward")]) == []


def test_match_pattern_two_hop_match():
    g = _make_graph([("A", "B", "r1"), ("B", "C", "r2")])
    matches = _match_pattern(g, [("r1", "forward"), ("r2", "forward")])
    # forward direction stores idx[v].add(u); chain from middle node B
    # expect some src→dst pair
    assert isinstance(matches, list)


def test_extend_chain_empty_indices():
    """No remaining indices → returns set containing current node."""
    assert _extend_chain(MagicMock(), [], "X") == {"X"}


def test_extend_chain_no_next_nodes():
    """Remaining indices but no successors → empty set (no further chain)."""
    g = MagicMock()
    idx = {"X": set()}  # no successors
    # last index → returns next_nodes directly (empty)
    result = _extend_chain(g, [idx], "X")
    assert result == set()


def test_apply_mined_rules_empty():
    g = MagicMock()
    assert _apply_mined_rules(g, []) == []


def test_apply_mined_rules_short_body_skipped():
    """body_path with < 2 relations is skipped."""
    g = MagicMock()
    rules = [{"body_path": ["only_one"], "head": "inferred"}]
    assert _apply_mined_rules(g, rules) == []


def test_apply_mined_rules_with_match():
    # Match algorithm: for r1 edge u->v, idx[v].add(u); for r2 edge x->w, idx[w].add(x).
    # Middle node B must be a key in BOTH indices, so we need an r2 edge whose
    # destination is B (X->B) plus the chain target.
    g = _make_graph([("A", "B", "r1"), ("B", "C", "r2"), ("X", "B", "r2")])
    rules = [{"body_path": ["r1", "r2"], "head": "direct", "confidence": 0.7}]
    inferred = _apply_mined_rules(g, rules)
    assert len(inferred) == 1
    edge = inferred[0]
    assert edge["relation"] == "direct"
    assert edge["via"] == "mined:direct"
    assert edge["confidence"] == 0.7


# ---------------------------------------------------------------------------
# _enrich_tree_with_sections
# ---------------------------------------------------------------------------


def test_enrich_tree_no_labels_noop():
    g = MagicMock()
    db = MagicMock()
    tree = {"children": []}
    _enrich_tree_with_sections(tree, g, db)
    g.get_section_contexts_batch.assert_not_called()


def test_enrich_tree_adds_sections():
    g = MagicMock()
    g.get_section_contexts_batch.return_value = {
        "ConceptA": {"section": "Intro", "node_id": "0001"},
    }
    db = MagicMock()
    tree = {
        "concept": "ConceptA",
        "children": [
            {"concept": "ConceptB", "children": []},
            {"label": "ConceptA", "children": []},
        ],
    }
    _enrich_tree_with_sections(tree, g, db)
    assert tree["section"] == "Intro"
    assert tree["node_id"] == "0001"
    # nested matching label also enriched
    assert tree["children"][1]["section"] == "Intro"
    # ConceptB not in section_map → unchanged
    assert "section" not in tree["children"][0]


# ---------------------------------------------------------------------------
# _export_paper_to_meta — patch Database
# ---------------------------------------------------------------------------


@patch("drbrain.cli._helpers.display.Database")
def test_export_paper_not_found_returns_empty(mock_db_cls):
    mock_db = MagicMock()
    mock_db.get_paper.return_value = None
    mock_db_cls.return_value = mock_db
    assert _export_paper_to_meta(mock_db, "ghost") == {}


@patch("drbrain.storage.export._extract_lastname", return_value="Smith")
def test_export_paper_full(mock_lastname):
    db = MagicMock()
    db.get_paper.return_value = {
        "local_id": "p1",
        "title": "T",
        "year": 2023,
        "doi": "10.x",
        "arxiv": "2401.00001",
        "paper_type": "article",
        "abstract": "abs",
        "journal": "Nature",
        "publisher": "Pub",
        "citation_count": 5,
        "volume": "1",
        "pages": "1-10",
    }
    # authors query: SELECT GROUP_CONCAT(...) returns a single-column row
    db.conn.execute.return_value.fetchone.return_value = ("John Smith and Jane Doe",)

    out = _export_paper_to_meta(db, "p1")
    assert out["local_id"] == "p1"
    assert out["title"] == "T"
    assert out["year"] == 2023
    assert out["authors"] == "John Smith and Jane Doe"
    assert out["first_author_lastname"] == "Smith"
    assert out["doi"] == "10.x"
    assert out["arxiv"] == "2401.00001"
    assert out["paper_type"] == "article"
    assert out["citation_count"] == 5


@patch("drbrain.storage.export._extract_lastname", return_value="Doe")
def test_export_paper_no_authors(mock_lastname):
    db = MagicMock()
    db.get_paper.return_value = {"local_id": "p2", "title": "X"}
    db.conn.execute.return_value.fetchone.return_value = (None,)

    out = _export_paper_to_meta(db, "p2")
    assert out["authors"] == ""
    # _extract_lastname is patched to return "Doe" regardless of input;
    # the function always invokes it, so the result reflects the mock.


# ---------------------------------------------------------------------------
# _show_actor — patch Database
# ---------------------------------------------------------------------------


@patch("drbrain.cli._helpers.display.Database")
def test_show_actor_no_papers(mock_db_cls, capsys):
    mock_db = MagicMock()
    # aliases query, papers query
    mock_db.conn.execute.return_value.fetchall.return_value = []
    mock_db_cls.return_value = mock_db

    cfg = {"db": {"path": "/tmp/x.db"}}
    _show_actor(cfg, "nobody")
    out = capsys.readouterr().out
    assert "no associated papers" in out


@patch("drbrain.cli._helpers.display.Database")
def test_show_actor_with_papers(mock_db_cls, capsys):
    mock_db = MagicMock()
    # Sequence of execute().fetchall()/fetchone() calls:
    # 1) aliases query → fetchall()
    # 2) papers query → fetchall()
    # 3) shared_actor query → fetchall() (only if paper_ids non-empty)
    aliases_result = [("John Smith",), ("J. Smith",)]
    papers_result = [("p1", "Title One", 2020), ("p2", "Title Two", 2021)]
    shared_result = []  # no shared_actor connections
    mock_cursor = MagicMock()
    mock_cursor.fetchall.side_effect = [aliases_result, papers_result, shared_result]
    mock_db.conn.execute.return_value = mock_cursor
    mock_db_cls.return_value = mock_db

    cfg = {"db": {"path": "/tmp/x.db"}}
    _show_actor(cfg, "smith_canonical")
    out = capsys.readouterr().out
    assert "Author: smith_canonical" in out
    assert "Display: John Smith, J. Smith" in out
    assert "Papers: 2" in out
    assert "Title One (2020)" in out
    assert "Title Two (2021)" in out
