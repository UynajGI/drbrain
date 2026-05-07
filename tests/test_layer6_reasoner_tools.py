"""Tests for Layer 6: ReasonerAgent tree tools."""

import tempfile
from pathlib import Path
from unittest import mock

import pytest

# ── Tool definitions ─────────────────────────────────────────────────────────


def test_tree_tools_in_tool_definitions():
    """ReasonerAgent exposes 4 tree tools: get_document_structure, get_section_content, search_tree, get_raptor_summaries."""
    from drbrain.extractor.reasoner import ReasonerAgent

    agent = ReasonerAgent(models=[{"provider": "openai", "model": "gpt-4o"}])
    tools = agent.tool_definitions()

    tool_names = {t["function"]["name"] for t in tools}
    assert "get_document_structure" in tool_names
    assert "get_section_content" in tool_names
    assert "search_tree" in tool_names
    assert "get_raptor_summaries" in tool_names
    # Existing tools still present
    assert "search_concepts" in tool_names
    assert "get_neighbors" in tool_names
    assert "find_path" in tool_names


# ── Tool handlers ────────────────────────────────────────────────────────────


def test_get_document_structure_handler():
    """_get_document_structure returns tree skeleton for a paper."""
    with tempfile.TemporaryDirectory() as td:
        import json

        # _papers_dir() = db.path.parent / "papers"
        papers_dir = Path(td) / "data" / "papers" / "test-paper"
        papers_dir.mkdir(parents=True)

        tree = {
            "structure": [
                {"node_id": "n1", "title": "Introduction"},
                {"node_id": "n2", "title": "Methods"},
            ]
        }
        (papers_dir / "tree.json").write_text(json.dumps(tree), encoding="utf-8")

        from drbrain.extractor.reasoner import ReasonerAgent
        from drbrain.storage.database import Database

        db_path = Path(td) / "data" / "drbrain.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db = Database(db_path)
        db.conn.execute(
            "INSERT INTO papers (local_id, title) VALUES (?, ?)",
            ("test-paper", "Test Paper"),
        )
        db.conn.commit()

        agent = ReasonerAgent(models=[], db=db)
        result = agent._get_document_structure(paper_id="test-paper")
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["node_id"] == "n1"


def test_get_document_structure_missing_paper():
    """Returns empty list for unknown paper."""
    from drbrain.extractor.reasoner import ReasonerAgent

    agent = ReasonerAgent(models=[])
    result = agent._get_document_structure(paper_id="nonexistent")
    assert result == []


def test_get_section_content_handler():
    """_get_section_content returns text for a tree node."""
    with tempfile.TemporaryDirectory() as td:
        import json

        papers_dir = Path(td) / "data" / "papers" / "test-paper"
        papers_dir.mkdir(parents=True)
        (papers_dir / "raw.md").write_text(
            "# Introduction\n\nSome intro text.\n\n# Methods\n\nMethods content.",
            encoding="utf-8",
        )

        tree = {
            "structure": [
                {"node_id": "n1", "title": "Introduction", "line_num": 1},
                {"node_id": "n2", "title": "Methods", "line_num": 5},
            ]
        }
        (papers_dir / "tree.json").write_text(json.dumps(tree), encoding="utf-8")

        from drbrain.extractor.reasoner import ReasonerAgent
        from drbrain.storage.database import Database

        db_path = Path(td) / "data" / "drbrain.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db = Database(db_path)
        db.conn.execute(
            "INSERT INTO papers (local_id, title) VALUES (?, ?)",
            ("test-paper", "Test Paper"),
        )
        db.conn.commit()

        agent = ReasonerAgent(models=[], db=db)
        content = agent._get_section_content(paper_id="test-paper", node_id="n1")
        assert "intro text" in content.lower()


def test_search_tree_handler():
    """_search_tree uses collapsed tree retrieval."""
    with mock.patch(
        "drbrain.query.tree_retrieval.query_cross_paper",
        return_value=[
            {"node_id": "n1", "paper_id": "p1", "score": 0.95, "tree_layer": "pageindex"},
            {"node_id": "r1", "paper_id": "p1", "score": 0.88, "tree_layer": "raptor_L1"},
        ],
    ):
        from drbrain.extractor.reasoner import ReasonerAgent
        from drbrain.storage.database import Database

        with tempfile.TemporaryDirectory() as td:
            db = Database(Path(td) / "test.db")
            agent = ReasonerAgent(models=[], db=db)
            results = agent._search_tree(query="attention mechanism")
            assert len(results) == 2
            assert results[0]["score"] == 0.95


def test_get_raptor_summaries_handler():
    """_get_raptor_summaries returns RAPTOR summaries from tree_summaries table."""
    import json

    from drbrain.extractor.reasoner import ReasonerAgent
    from drbrain.storage.database import Database

    with tempfile.TemporaryDirectory() as td:
        papers_dir = Path(td) / "data" / "papers" / "test-paper"
        papers_dir.mkdir(parents=True)
        (papers_dir / "tree.json").write_text(
            json.dumps({"structure": [{"node_id": "n1", "title": "Test"}]}),
            encoding="utf-8",
        )

        db_path = Path(td) / "data" / "drbrain.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db = Database(db_path)
        db.conn.execute(
            "INSERT INTO papers (local_id, title) VALUES (?, ?)",
            ("test-paper", "Test Paper"),
        )
        # Insert RAPTOR summaries
        db.conn.execute(
            "INSERT INTO tree_summaries (node_id, paper_id, summary_text, source_node_ids, tree_layer) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                "raptor_test-paper_L1_abc",
                "test-paper",
                "This section cluster discusses attention mechanisms and their variants.",
                json.dumps(["n1", "n2"]),
                1,
            ),
        )
        db.conn.execute(
            "INSERT INTO tree_summaries (node_id, paper_id, summary_text, source_node_ids, tree_layer) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                "raptor_test-paper_L2_def",
                "test-paper",
                "Higher-level summary of methodology across all sections.",
                json.dumps(["raptor_test-paper_L1_abc"]),
                2,
            ),
        )
        db.conn.commit()

        agent = ReasonerAgent(models=[], db=db)
        result = agent._get_raptor_summaries(paper_id="test-paper")

        assert isinstance(result, list)
        assert len(result) == 2
        # Should be ordered by tree_layer
        assert result[0]["tree_layer"] == 1
        assert result[0]["paper_id"] == "test-paper"
        assert "attention mechanisms" in result[0]["summary_text"]
        assert isinstance(result[0]["source_node_ids"], list)
        assert "n1" in result[0]["source_node_ids"]
        assert result[1]["tree_layer"] == 2


def test_get_raptor_summaries_empty_for_unknown_paper():
    """Returns empty list when paper has no RAPTOR summaries."""
    from drbrain.extractor.reasoner import ReasonerAgent
    from drbrain.storage.database import Database

    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        agent = ReasonerAgent(models=[], db=db)
        result = agent._get_raptor_summaries(paper_id="nonexistent")
        assert result == []


# ── Tool dispatch in reason loop ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reason_dispatches_tree_tools():
    """reason method correctly dispatches tree tool calls."""
    from drbrain.extractor.reasoner import ReasonerAgent
    from drbrain.storage.database import Database

    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")

        # Mock the tree tool handlers
        agent = ReasonerAgent(models=[{"provider": "openai", "model": "gpt-4o"}], db=db)

        # Mock _call_llm to return a tool call for get_document_structure
        async def _fake_call_llm(messages):
            return mock.MagicMock(
                content=None,
                tool_calls=[
                    mock.MagicMock(
                        function=mock.MagicMock(
                            name="get_document_structure",
                            arguments='{"paper_id": "test-paper"}',
                        )
                    )
                ],
            )

        with (
            mock.patch.object(agent, "_call_llm", side_effect=_fake_call_llm),
            mock.patch.object(agent, "_get_document_structure", return_value=[]),
            mock.patch.object(agent, "_kg_validate", return_value={"consistent": True}),
        ):
            result = await agent.reason("What sections discuss attention?")
            # Should complete without error
            assert "No LLM models" not in result


@pytest.mark.asyncio
async def test_reason_dispatches_raptor_summaries():
    """reason dispatches get_raptor_summaries tool calls correctly."""
    from drbrain.extractor.reasoner import ReasonerAgent
    from drbrain.storage.database import Database

    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")

        agent = ReasonerAgent(models=[{"provider": "openai", "model": "gpt-4o"}], db=db)

        async def _fake_call_llm(messages):
            return mock.MagicMock(
                content=None,
                tool_calls=[
                    mock.MagicMock(
                        function=mock.MagicMock(
                            name="get_raptor_summaries",
                            arguments='{"paper_id": "test-paper"}',
                        )
                    )
                ],
            )

        with (
            mock.patch.object(agent, "_call_llm", side_effect=_fake_call_llm),
            mock.patch.object(
                agent,
                "_get_raptor_summaries",
                return_value=[
                    {
                        "node_id": "raptor_test_L1_abc",
                        "paper_id": "test-paper",
                        "summary_text": "Cross-section attention summary.",
                        "source_node_ids": ["n1", "n2"],
                        "tree_layer": 1,
                    }
                ],
            ),
            mock.patch.object(agent, "_kg_validate", return_value={"consistent": True}),
        ):
            result = await agent.reason("What is the overall theme?")
            assert "No LLM models" not in result
