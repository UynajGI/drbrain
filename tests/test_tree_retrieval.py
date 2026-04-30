"""Tests for structure-first retrieval using PageIndex tree."""

import json
from unittest import mock

from drbrain.query.tree_retrieval import _ask_llm_for_relevant_nodes, query_by_structure


@mock.patch("drbrain.query.tree_retrieval.acall_with_fallback")
def test_ask_llm_for_relevant_nodes_returns_ids(mock_acall):
    """_ask_llm_for_relevant_nodes parses LLM response as node_id list."""
    mock_acall.return_value = ["0001", "0003"]

    import asyncio

    result = asyncio.run(
        _ask_llm_for_relevant_nodes(
            "What is the methodology?",
            '{"title": "Paper", "nodes": []}',
            [{"provider": "openai", "model": "gpt-4"}],
        )
    )
    assert result == ["0001", "0003"]


@mock.patch("drbrain.query.tree_retrieval.acall_with_fallback")
def test_ask_llm_for_relevant_nodes_limits_to_5(mock_acall):
    """_ask_llm_for_relevant_nodes returns at most 5 node IDs."""
    mock_acall.return_value = ["0001", "0002", "0003", "0004", "0005", "0006", "0007"]

    import asyncio

    result = asyncio.run(
        _ask_llm_for_relevant_nodes(
            "question",
            "{}",
            [],
        )
    )
    assert len(result) == 5


@mock.patch("drbrain.query.tree_retrieval.acall_with_fallback")
def test_ask_llm_for_relevant_nodes_handles_dict_response(mock_acall):
    """_ask_llm_for_relevant_nodes handles wrapped dict responses."""
    mock_acall.return_value = {"node_ids": ["0001", "0002"]}

    import asyncio

    result = asyncio.run(_ask_llm_for_relevant_nodes("question", "{}", []))
    assert result == ["0001", "0002"]


@mock.patch("drbrain.query.tree_retrieval.acall_with_fallback")
def test_ask_llm_for_relevant_nodes_handles_failure(mock_acall):
    """_ask_llm_for_relevant_nodes returns empty list on LLM failure."""
    mock_acall.return_value = None

    import asyncio

    result = asyncio.run(_ask_llm_for_relevant_nodes("question", "{}", []))
    assert result == []


@mock.patch("drbrain.query.tree_retrieval.get_node_content")
@mock.patch("drbrain.query.tree_retrieval._ask_llm_for_relevant_nodes")
def test_query_by_structure_basic(mock_ask, mock_get_content, tmp_path):
    """query_by_structure loads content from relevant sections."""
    # Setup files
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    (paper_dir / "tree.json").write_text(
        '{"doc_name": "test", "line_count": 10, "structure": [{"title": "A", "node_id": "0000", "nodes": []}]}'
    )
    (paper_dir / "raw.md").write_text("# A\nContent here.\n")

    # Mock LLM to return a node_id
    mock_ask.return_value = ["0000"]
    mock_get_content.return_value = "Full content of section A."

    import asyncio

    result = asyncio.run(query_by_structure("What is in section A?", paper_dir, []))
    assert result is not None
    assert "Full content of section A" in result


@mock.patch("drbrain.query.tree_retrieval._ask_llm_for_relevant_nodes")
def test_query_by_structure_no_relevant_sections(mock_ask, tmp_path):
    """Returns None when LLM identifies no relevant sections."""
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    (paper_dir / "tree.json").write_text('{"doc_name": "test", "line_count": 10, "structure": []}')
    (paper_dir / "raw.md").write_text("# Empty\n")

    mock_ask.return_value = []

    import asyncio

    result = asyncio.run(query_by_structure("question", paper_dir, []))
    assert result is None


def test_query_by_structure_missing_files(tmp_path):
    """Returns None when tree.json or raw.md is missing."""
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()

    import asyncio

    result = asyncio.run(query_by_structure("question", paper_dir, []))
    assert result is None


@mock.patch("drbrain.query.tree_retrieval.get_node_content")
@mock.patch("drbrain.query.tree_retrieval._ask_llm_for_relevant_nodes")
def test_query_by_structure_multiple_sections(mock_ask, mock_get_content, tmp_path):
    """Multiple relevant sections are concatenated with separator."""
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    structure = [
        {"title": "A", "node_id": "0000", "nodes": []},
        {"title": "B", "node_id": "0001", "nodes": []},
    ]
    (paper_dir / "tree.json").write_text(
        f'{{"doc_name": "test", "line_count": 10, "structure": {json.dumps(structure)}}}'
    )
    (paper_dir / "raw.md").write_text("# A\nContent A\n\n# B\nContent B\n")

    mock_ask.return_value = ["0000", "0001"]
    mock_get_content.side_effect = ["Content A.", "Content B."]

    import asyncio

    result = asyncio.run(query_by_structure("question", paper_dir, []))
    assert result is not None
    assert "Content A." in result
    assert "Content B." in result
    assert "---" in result
