"""Tests for LLM agent graph reasoner."""
import tempfile
from pathlib import Path
from unittest import mock

import asyncio

from drbrain.graph.engine import GraphEngine
from drbrain.extractor.reasoner import ReasonerAgent


def test_reasoner_tool_definitions():
    """Reasoner has search, neighbors, and path tools."""
    agent = ReasonerAgent(graph_engine=None, models=[])
    tools = agent.tool_definitions()
    tool_names = [t["function"]["name"] for t in tools]
    assert "search_concepts" in tool_names
    assert "get_neighbors" in tool_names
    assert "find_path" in tool_names


def test_reasoner_search_tool():
    """search_concepts tool calls BM25 search."""
    from drbrain.storage.database import Database
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Test", 2024, "uploaded")
        db.insert_concept("p1", "Method", "transformer", 0.9, year=2024)
        db.commit()

        agent = ReasonerAgent(db=db, graph_engine=None, models=[])
        result = agent._search_concepts("transformer", limit=3)
        assert len(result) > 0
        db.close()


def test_reasoner_graph_tools():
    """get_neighbors and find_path work on graph."""
    g = GraphEngine()
    g.add_edge("A", "B", "addresses", "p1")
    g.add_edge("B", "C", "extends", "p1")

    agent = ReasonerAgent(db=None, graph_engine=g, models=[])

    neighbors = agent._get_neighbors("B", hops=1, direction="both")
    assert len(neighbors) >= 1

    path = agent._find_path("A", "C")
    assert path is not None
    assert path["length"] == 2


def test_reasoner_find_path_no_path():
    """find_path returns None for disconnected nodes."""
    g = GraphEngine()
    g.add_edge("A", "B", "cites", "p1")

    agent = ReasonerAgent(db=None, graph_engine=g, models=[])
    result = agent._find_path("A", "X")  # X not in graph
    assert result is None


def test_reasoner_no_graph_no_crash():
    """Reasoner tools handle None graph gracefully."""
    agent = ReasonerAgent(db=None, graph_engine=None, models=[])
    assert agent._get_neighbors("A") == []
    assert agent._find_path("A", "B") is None
    assert agent._search_concepts("test") == []
