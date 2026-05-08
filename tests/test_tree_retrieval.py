"""Tests for structure-first retrieval using PageIndex tree."""

import json
import tempfile
from pathlib import Path
from unittest import mock

import typer

from drbrain.query.tree_retrieval import (
    _ask_llm_for_relevant_nodes,
    _build_remaining_structure,
    _build_top_level_structure,
    _collect_all_leaf_ids,
    _expand_branch,
    _get_node_title,
    query_by_structure,
)


def _make_minimal_config(db_path: str, papers_dir: str) -> dict:
    return {
        "db": {"path": db_path},
        "llm": {"models": [{"provider": "openai", "model": "gpt-4", "api_key": "x"}]},
        "dirs": {
            "inbox": "data/spool/inbox",
            "papers": papers_dir,
            "reports": "/tmp/reports",
            "cache": "data/cache",
            "logs": "data/logs",
        },
        "bm25": {"k1": 1.5, "b": 0.75},
        "queue": {"weak_threshold": 0.7, "auto_accept": 0.9},
        "api": {"s2_rate_limit": 100, "cache_ttl": 86400},
    }


def _make_ctx(cfg: dict):
    """Create a minimal typer.Context mock with config pre-loaded."""
    ctx = mock.MagicMock(spec=typer.Context)
    ctx.obj = {"config": cfg}
    return ctx


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
@mock.patch("drbrain.query.tree_retrieval.acall_with_fallback")
def test_query_by_structure_basic(mock_acall, mock_get_content, tmp_path):
    """query_by_structure returns structured section data."""
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    (paper_dir / "tree.json").write_text(
        '{"doc_name": "test", "line_count": 10, "structure": [{"title": "A", "node_id": "0000", "nodes": []}]}'
    )
    (paper_dir / "raw.md").write_text("# A\nContent here.\n")

    # Mock LLM round 1: return node_ids for small skeleton
    mock_acall.return_value = {"node_ids": ["0000"]}
    mock_get_content.return_value = "Full content of section A."

    import asyncio

    result = asyncio.run(query_by_structure("What is in section A?", paper_dir, []))
    assert result is not None
    assert len(result) == 1
    assert result[0]["content"] == "Full content of section A."
    assert result[0]["node_id"] == "0000"


@mock.patch("drbrain.query.tree_retrieval.acall_with_fallback")
def test_query_by_structure_no_relevant_sections(mock_acall, tmp_path):
    """Returns None when LLM identifies no relevant sections."""
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    (paper_dir / "tree.json").write_text('{"doc_name": "test", "line_count": 10, "structure": []}')
    (paper_dir / "raw.md").write_text("# Empty\n")

    mock_acall.return_value = None

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
@mock.patch("drbrain.query.tree_retrieval.acall_with_fallback")
def test_query_by_structure_multiple_sections(mock_acall, mock_get_content, tmp_path):
    """Multiple relevant sections are returned as structured list."""
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    structure = [
        {"title": "Section A", "node_id": "0000", "nodes": []},
        {"title": "Section B", "node_id": "0001", "nodes": []},
    ]
    (paper_dir / "tree.json").write_text(
        f'{{"doc_name": "test", "line_count": 10, "structure": {json.dumps(structure)}}}'
    )
    (paper_dir / "raw.md").write_text("# A\nContent A\n\n# B\nContent B\n")

    mock_acall.return_value = {"node_ids": ["0000", "0001"]}
    mock_get_content.side_effect = ["Content A.", "Content B."]

    import asyncio

    result = asyncio.run(query_by_structure("question", paper_dir, []))
    assert result is not None
    assert len(result) == 2
    assert result[0]["content"] == "Content A."
    assert result[0]["node_id"] == "0000"
    assert result[0]["title"] == "Section A"
    assert result[1]["content"] == "Content B."
    assert result[1]["node_id"] == "0001"
    assert result[1]["title"] == "Section B"


# -- CLI integration: query_cmd --paper flag --


def test_query_cmd_tree_paper_missing_tree():
    """--paper with a paper directory that has no tree.json shows error."""
    from drbrain.cli.commands import query_cmd

    with tempfile.TemporaryDirectory() as td:
        papers_dir = Path(td) / "papers"
        papers_dir.mkdir()
        paper_dir = papers_dir / "test_paper"
        paper_dir.mkdir()
        # raw.md exists but tree.json does not
        (paper_dir / "raw.md").write_text("# Test\nContent.\n")

        cfg = _make_minimal_config(f"{td}/test.db", str(papers_dir))

        ctx = _make_ctx(cfg)
        try:
            query_cmd(
                ctx,
                "What is the methodology?",
                paper="test_paper",
            )
            assert False, "Should have raised Exit"
        except typer.Exit as e:
            assert e.exit_code == 1


@mock.patch("drbrain.cli.query_commands.query_by_structure_hybrid")
def test_query_cmd_tree_success(mock_qbs):
    """--paper with valid tree.json invokes tree retrieval and shows content."""
    from drbrain.cli.commands import query_cmd

    mock_qbs.return_value = [
        {"node_id": "0000", "title": "A", "content": "Sample retrieved content from section 3."}
    ]

    with tempfile.TemporaryDirectory() as td:
        papers_dir = Path(td) / "papers"
        papers_dir.mkdir()
        paper_dir = papers_dir / "test_paper"
        paper_dir.mkdir()
        (paper_dir / "tree.json").write_text(
            '{"doc_name": "test", "line_count": 10, "structure": [{"title": "A", "node_id": "0000", "nodes": []}]}'
        )
        (paper_dir / "raw.md").write_text("# A\nContent.\n")

        cfg = _make_minimal_config(f"{td}/test.db", str(papers_dir))

        # Capture stdout
        import io
        import sys

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        ctx = _make_ctx(cfg)
        query_cmd(ctx, "What is the methodology?", paper="test_paper")

        output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        assert mock_qbs.called
        assert "Sample retrieved content" in output


@mock.patch("drbrain.cli.query_commands.query_by_structure_hybrid")
def test_query_cmd_tree_no_relevant(mock_qbs):
    """--paper with valid paper but no relevant sections shows message."""
    from drbrain.cli.commands import query_cmd

    mock_qbs.return_value = None

    with tempfile.TemporaryDirectory() as td:
        papers_dir = Path(td) / "papers"
        papers_dir.mkdir()
        paper_dir = papers_dir / "test_paper"
        paper_dir.mkdir()
        (paper_dir / "tree.json").write_text(
            '{"doc_name": "test", "line_count": 10, "structure": []}'
        )
        (paper_dir / "raw.md").write_text("# Empty\n")

        cfg = _make_minimal_config(f"{td}/test.db", str(papers_dir))

        import io
        import sys

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        ctx = _make_ctx(cfg)
        query_cmd(
            ctx,
            "Does not exist in this paper",
            paper="test_paper",
            type_filter=None,
            arg_type=None,
            year_start=None,
            year_end=None,
            min_confidence=None,
            limit=20,
            neighbors=0,
            json_output=False,
            jsonl=False,
        )

        output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        assert mock_qbs.called
        assert "No relevant sections" in output


@mock.patch("drbrain.cli.query_commands.query_by_structure_hybrid")
def test_query_cmd_tree_json_output(mock_qbs):
    """--paper --json outputs structured JSON."""
    from drbrain.cli.commands import query_cmd

    mock_qbs.return_value = [
        {"node_id": "0000", "title": "A", "content": "Content from section A."}
    ]

    with tempfile.TemporaryDirectory() as td:
        papers_dir = Path(td) / "papers"
        papers_dir.mkdir()
        paper_dir = papers_dir / "test_paper"
        paper_dir.mkdir()
        (paper_dir / "tree.json").write_text(
            '{"doc_name": "test", "line_count": 10, "structure": [{"title": "A", "node_id": "0000", "nodes": []}]}'
        )
        (paper_dir / "raw.md").write_text("# A\nContent.\n")

        cfg = _make_minimal_config(f"{td}/test.db", str(papers_dir))

        import io
        import sys

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        ctx = _make_ctx(cfg)
        query_cmd(ctx, "question", paper="test_paper", json_output=True)

        output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        # Should be valid JSON
        data = json.loads(output)
        assert data["mode"] == "hybrid"
        assert data["paper"] == "test_paper"
        assert len(data["sections"]) == 1
        assert data["sections"][0]["node_id"] == "0000"


def test_query_cmd_tree_paper_not_found():
    """--paper with nonexistent local_id raises Exit."""
    from drbrain.cli.commands import query_cmd

    with tempfile.TemporaryDirectory() as td:
        papers_dir = Path(td) / "papers"
        papers_dir.mkdir()

        cfg = _make_minimal_config(f"{td}/test.db", str(papers_dir))

        ctx = _make_ctx(cfg)
        try:
            query_cmd(ctx, "question", paper="nonexistent_paper")
            assert False, "Should have raised Exit"
        except typer.Exit as e:
            assert e.exit_code == 1


def test_query_cmd_bm25_unchanged():
    """Without --paper, normal BM25 behavior is preserved."""
    from drbrain.cli.commands import query_cmd
    from drbrain.storage.database import Database

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        papers_dir = Path(td) / "papers"
        papers_dir.mkdir()

        db = Database(str(db_path))
        db.insert_paper("p1", "Test Paper", 2024, "uploaded")
        db.insert_concept("p1", "Problem", "Sample concept", 0.9, year=2024)
        db.commit()
        db.close()

        cfg = _make_minimal_config(str(db_path), str(papers_dir))

        import io
        import sys

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        ctx = _make_ctx(cfg)
        # No --paper flag → should use BM25
        query_cmd(
            ctx,
            "sample",
            paper=None,
            type_filter=None,
            arg_type=None,
            year_start=None,
            year_end=None,
            min_confidence=None,
            limit=20,
            neighbors=0,
            json_output=False,
            jsonl=False,
        )

        output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        assert "Query:" in output


# -- Edge case: LLM returns no relevant sections when structure exists --


@mock.patch("drbrain.query.tree_retrieval._ask_llm_for_relevant_nodes")
def test_query_by_structure_llm_returns_empty_with_structure(mock_ask, tmp_path):
    """Returns None when LLM finds no relevant sections despite valid structure."""
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    (paper_dir / "tree.json").write_text(
        '{"doc_name": "test", "line_count": 10, "structure": [{"title": "A", "node_id": "0000", "nodes": []}]}'
    )
    (paper_dir / "raw.md").write_text("# A\nContent here.\n")

    mock_ask.return_value = []

    import asyncio

    result = asyncio.run(query_by_structure("question", paper_dir, []))
    assert result is None


# -- Edge case: get_node_content returns empty/None for all returned IDs --


@mock.patch("drbrain.query.tree_retrieval.get_node_content")
@mock.patch("drbrain.query.tree_retrieval._ask_llm_for_relevant_nodes")
def test_query_by_structure_ids_have_no_content(mock_ask, mock_get_content, tmp_path):
    """Returns None when LLM picks IDs but all have empty content."""
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    (paper_dir / "tree.json").write_text(
        '{"doc_name": "test", "line_count": 10, "structure": [{"title": "A", "node_id": "0000", "nodes": []}]}'
    )
    (paper_dir / "raw.md").write_text("# A\nContent.\n")

    mock_ask.return_value = ["0000"]
    mock_get_content.return_value = None  # No content for any node

    import asyncio

    result = asyncio.run(query_by_structure("question", paper_dir, []))
    assert result is None


# -- Edge case: _get_node_title with nonexistent node_id --


def test_get_node_title_nonexistent_node():
    """_get_node_title returns '' for a node_id not in the tree."""
    structure = [{"title": "Section A", "node_id": "0000", "nodes": []}]
    result = _get_node_title(structure, "nonexistent")
    assert result == ""


def test_get_node_title_in_nested_structure():
    """_get_node_title finds title in nested nodes."""
    structure = [
        {
            "title": "Root",
            "node_id": "0000",
            "nodes": [
                {"title": "Child", "node_id": "0000-0", "nodes": []},
            ],
        }
    ]
    assert _get_node_title(structure, "0000-0") == "Child"
    assert _get_node_title(structure, "0000") == "Root"


# -- Additional helper function tests for coverage --


def test_get_node_title_empty_structure():
    """_get_node_title returns '' for empty structure."""
    assert _get_node_title([], "n1") == ""


def test_get_node_title_missing_node():
    """_get_node_title returns '' for node not in structure."""
    structure = [{"title": "A", "node_id": "n1", "nodes": []}]
    assert _get_node_title(structure, "n2") == ""


def test_collect_all_leaf_ids_single():
    """_collect_all_leaf_ids finds single leaf."""
    structure = [
        {
            "title": "root",
            "node_id": "root",
            "nodes": [{"title": "leaf1", "node_id": "leaf1", "nodes": []}],
        }
    ]
    leaves = _collect_all_leaf_ids(structure)
    assert "leaf1" in leaves
    assert "root" not in leaves


def test_collect_all_leaf_ids_nested():
    """_collect_all_leaf_ids finds all leaves in nested tree."""
    structure = [
        {
            "title": "root",
            "node_id": "root",
            "nodes": [
                {
                    "title": "mid",
                    "node_id": "mid",
                    "nodes": [{"title": "leaf1", "node_id": "leaf1", "nodes": []}],
                },
                {"title": "leaf2", "node_id": "leaf2", "nodes": []},
            ],
        }
    ]
    leaves = _collect_all_leaf_ids(structure)
    assert "leaf1" in leaves
    assert "leaf2" in leaves
    assert "root" not in leaves
    assert "mid" not in leaves


def test_build_remaining_structure_filters_read():
    """_build_remaining_structure excludes already-read leaf nodes."""
    structure = [
        {"title": "Intro", "node_id": "n1", "nodes": []},
        {"title": "Methods", "node_id": "n2", "nodes": []},
    ]
    result = _build_remaining_structure(structure, {"n1"})
    parsed = json.loads(result)
    assert len(parsed) == 1
    assert parsed[0]["node_id"] == "n2"


def test_build_remaining_structure_empty():
    """_build_remaining_structure returns '{}' when all nodes read."""
    structure = [{"title": "A", "node_id": "n1", "nodes": []}]
    result = _build_remaining_structure(structure, {"n1"})
    assert result == "{}"


def test_build_top_level_structure_depth1():
    """_build_top_level_structure with depth=1 shows only top-level with child_count."""
    structure = [
        {
            "title": "A",
            "node_id": "n1",
            "nodes": [
                {"title": "A1", "node_id": "n1a", "nodes": []},
                {"title": "A2", "node_id": "n1b", "nodes": []},
            ],
        },
        {"title": "B", "node_id": "n2", "nodes": []},
    ]
    result = _build_top_level_structure(structure, depth=1)
    assert any(n["node_id"] == "n1" for n in result)
    assert any(n["node_id"] == "n2" for n in result)
    # n1 should have child_count since depth=1 collapses children
    n1 = [n for n in result if n["node_id"] == "n1"][0]
    assert n1["child_count"] == 2
    assert n1["children"] == []


def test_build_top_level_structure_depth2():
    """_build_top_level_structure with depth=2 includes one level of children."""
    structure = [
        {
            "title": "A",
            "node_id": "n1",
            "nodes": [
                {"title": "A1", "node_id": "n1a", "nodes": []},
                {"title": "A2", "node_id": "n1b", "nodes": []},
            ],
        },
        {"title": "B", "node_id": "n2", "nodes": []},
    ]
    result = _build_top_level_structure(structure, depth=2)
    assert any(n["node_id"] == "n1" for n in result)
    assert any(n["node_id"] == "n2" for n in result)
    # Children should be expanded at depth=2
    n1 = [n for n in result if n["node_id"] == "n1"][0]
    assert len(n1["children"]) == 2


def test_expand_branch_includes_children():
    """_expand_branch returns children of selected nodes."""
    structure = [
        {
            "title": "root",
            "node_id": "root",
            "nodes": [
                {
                    "title": "child1",
                    "node_id": "child1",
                    "nodes": [{"title": "grandchild1", "node_id": "grandchild1", "nodes": []}],
                },
                {"title": "child2", "node_id": "child2", "nodes": []},
            ],
        }
    ]
    result = _expand_branch(structure, {"root"})
    node_ids = {n["node_id"] for n in result}
    assert "child1" in node_ids
    assert "child2" in node_ids
    # root itself should NOT be in result (its children are returned)
    assert "root" not in node_ids


def test_expand_branch_leaf_node():
    """_expand_branch on a leaf node returns the node itself."""
    structure = [{"title": "leaf", "node_id": "leaf1", "nodes": []}]
    result = _expand_branch(structure, {"leaf1"})
    assert len(result) == 1
    assert result[0]["node_id"] == "leaf1"


def test_expand_branch_no_match():
    """_expand_branch with non-matching IDs returns empty list."""
    structure = [{"title": "A", "node_id": "n1", "nodes": []}]
    result = _expand_branch(structure, {"nonexistent"})
    assert result == []
