"""Tests for concept extraction — flat and tree-based."""

from unittest import mock

from drbrain.extractor.concept import (
    ExtractedConcepts,
    _collect_leaf_nodes,
    _merge_concepts,
    extract_concepts_from_tree,
    extract_section_concepts,
)

# -- Leaf node collection --


def test_collect_leaf_nodes_simple():
    """Leaf nodes are nodes with no children."""
    structure = [
        {"title": "A", "node_id": "0000", "line_num": 1, "nodes": []},
        {"title": "B", "node_id": "0001", "line_num": 5, "nodes": []},
    ]
    leaves = _collect_leaf_nodes(structure)
    assert len(leaves) == 2
    assert leaves[0]["title"] == "A"
    assert leaves[1]["title"] == "B"


def test_collect_leaf_nodes_nested():
    """Only deepest nodes are leaves."""
    structure = [
        {
            "title": "Parent",
            "node_id": "0000",
            "line_num": 1,
            "nodes": [
                {"title": "Child1", "node_id": "0001", "line_num": 3, "nodes": []},
                {"title": "Child2", "node_id": "0002", "line_num": 7, "nodes": []},
            ],
        }
    ]
    leaves = _collect_leaf_nodes(structure)
    assert len(leaves) == 2
    assert leaves[0]["title"] == "Child1"
    assert leaves[1]["title"] == "Child2"


def test_collect_leaf_nodes_mixed():
    """Mix of leaf and parent nodes."""
    structure = [
        {"title": "Flat", "node_id": "0000", "line_num": 1, "nodes": []},
        {
            "title": "Parent",
            "node_id": "0001",
            "line_num": 5,
            "nodes": [
                {"title": "Deep", "node_id": "0002", "line_num": 7, "nodes": []},
            ],
        },
    ]
    leaves = _collect_leaf_nodes(structure)
    assert len(leaves) == 2
    assert leaves[0]["title"] == "Flat"
    assert leaves[1]["title"] == "Deep"


def test_collect_leaf_nodes_preserves_summary():
    """Leaf nodes carry summary if present."""
    structure = [
        {"title": "A", "node_id": "0000", "line_num": 1, "summary": "sum A", "nodes": []},
    ]
    leaves = _collect_leaf_nodes(structure)
    assert leaves[0]["summary"] == "sum A"


# -- Merge concepts --


def test_merge_concepts_dedup_by_label():
    """Duplicate labels are merged, keeping highest confidence."""
    r1 = ExtractedConcepts({"problems": [{"label": "Long range", "confidence": 0.7}]})
    r2 = ExtractedConcepts({"problems": [{"label": "Long Range", "confidence": 0.9}]})
    merged = _merge_concepts([r1, r2])
    assert len(merged.problems) == 1
    assert merged.problems[0]["confidence"] == 0.9


def test_merge_concepts_different_labels():
    """Different labels are kept."""
    r1 = ExtractedConcepts({"methods": [{"label": "Transformer", "confidence": 0.9}]})
    r2 = ExtractedConcepts({"methods": [{"label": "CNN", "confidence": 0.8}]})
    merged = _merge_concepts([r1, r2])
    assert len(merged.methods) == 2


def test_merge_concepts_relations_dedup():
    """Duplicate relations are merged."""
    r1 = ExtractedConcepts({"relations": [{"head": "A", "rel": "proposes", "tail": "B"}]})
    r2 = ExtractedConcepts({"relations": [{"head": "A", "rel": "proposes", "tail": "B"}]})
    merged = _merge_concepts([r1, r2])
    assert len(merged.relations) == 1


def test_merge_concepts_arguments_dedup():
    """Duplicate arguments are merged by (claim, target), keeping highest confidence."""
    r1 = ExtractedConcepts(
        {
            "arguments": [
                {
                    "claim": "X solves Y",
                    "claim_type": "solves",
                    "target": "Y",
                    "target_type": "Problem",
                    "confidence": 0.7,
                }
            ]
        }
    )
    r2 = ExtractedConcepts(
        {
            "arguments": [
                {
                    "claim": "X solves Y",
                    "claim_type": "solves",
                    "target": "Y",
                    "target_type": "Problem",
                    "confidence": 0.95,
                }
            ]
        }
    )
    merged = _merge_concepts([r1, r2])
    assert len(merged.arguments) == 1
    assert merged.arguments[0].confidence == 0.95


def test_merge_concepts_empty():
    """Merging empty list returns empty ExtractedConcepts."""
    merged = _merge_concepts([])
    assert merged.problems == []
    assert merged.methods == []


def test_merge_concepts_categories():
    """All six categories are merged independently."""
    r1 = ExtractedConcepts(
        {
            "problems": [{"label": "P1", "confidence": 0.9}],
            "methods": [{"label": "M1", "confidence": 0.8}],
            "conclusions": [{"label": "C1", "confidence": 0.7}],
            "debates": [{"label": "D1", "confidence": 0.6}],
            "gaps": [{"label": "G1", "confidence": 0.5}],
            "actors": [{"label": "A1", "confidence": 0.4}],
        }
    )
    r2 = ExtractedConcepts(
        {
            "problems": [{"label": "P2", "confidence": 0.85}],
            "gaps": [{"label": "G2", "confidence": 0.6}],
        }
    )
    merged = _merge_concepts([r1, r2])
    assert len(merged.problems) == 2
    assert len(merged.methods) == 1
    assert len(merged.gaps) == 2


# -- Section extraction --


@mock.patch("drbrain.extractor.concept.acall_with_fallback")
def test_extract_section_concepts_calls_llm(mock_acall):
    """extract_section_concepts sends section title + content to LLM."""
    mock_acall.return_value = {
        "problems": [{"label": "Test", "confidence": 0.9}],
        "methods": [],
        "conclusions": [],
        "debates": [],
        "gaps": [],
        "actors": [],
        "relations": [],
        "arguments": [],
    }

    import asyncio

    result = asyncio.run(
        extract_section_concepts(
            "Methods",
            "We use transformer architecture.",
            '{"title": "Paper", "nodes": []}',
            [{"provider": "openai", "model": "gpt-4"}],
        )
    )

    assert result is not None
    assert len(result.problems) == 1
    # Verify the prompt included section title and structure
    call_args = mock_acall.call_args
    user_prompt = call_args.kwargs.get("prompt") or call_args[0][0]
    assert "Methods" in user_prompt
    assert "We use transformer architecture" in user_prompt
    assert "Document Structure" in user_prompt


@mock.patch("drbrain.extractor.concept.acall_with_fallback")
def test_extract_section_concepts_returns_none_on_failure(mock_acall):
    """extract_section_concepts returns None when LLM fails."""
    mock_acall.return_value = None

    import asyncio

    result = asyncio.run(extract_section_concepts("Title", "text", "{}", []))
    assert result is None


# -- Tree-based extraction --


@mock.patch("drbrain.extractor.concept.get_node_content")
@mock.patch("drbrain.extractor.concept.acall_with_fallback")
def test_extract_concepts_from_tree_basic(mock_acall, mock_get_content, tmp_path):
    """extract_concepts_from_tree extracts from each leaf node."""
    md = tmp_path / "test.md"
    md.write_text("# Title\n\nAbstract.\n\n## Methods\n\nWe use transformers.\n")

    structure = [
        {
            "title": "Title",
            "node_id": "0000",
            "line_num": 1,
            "nodes": [
                {"title": "Abstract", "node_id": "0001", "line_num": 1, "nodes": []},
                {"title": "Methods", "node_id": "0002", "line_num": 3, "nodes": []},
            ],
        },
    ]

    # Mock get_node_content to return content for each leaf (must be >50 chars)
    def mock_content(path, struct, node_id):
        if node_id == "0001":
            return "This paper addresses long-range dependencies in sequence modeling tasks using novel architectures."
        if node_id == "0002":
            return "We use transformer architecture with self-attention mechanism for parallel computation of representations."
        return None

    mock_get_content.side_effect = mock_content

    # Mock LLM to return different results for each section
    mock_acall.return_value = {
        "problems": [{"label": "Long-range dependency", "confidence": 0.9}],
        "methods": [],
        "conclusions": [],
        "debates": [],
        "gaps": [],
        "actors": [],
        "relations": [],
        "arguments": [],
    }

    import asyncio

    result = asyncio.run(
        extract_concepts_from_tree(md, structure, [{"provider": "openai", "model": "gpt-4"}])
    )
    assert result is not None
    # LLM called for each leaf with content
    assert mock_acall.call_count == 2


@mock.patch("drbrain.extractor.concept.get_node_content")
def test_extract_concepts_from_tree_no_content(mock_get_content, tmp_path):
    """Returns None when no leaf node has content."""
    md = tmp_path / "test.md"
    md.write_text("# Empty\n")
    structure = [{"title": "Empty", "node_id": "0000", "line_num": 1, "nodes": []}]
    mock_get_content.return_value = ""

    import asyncio

    result = asyncio.run(
        extract_concepts_from_tree(md, structure, [{"provider": "openai", "model": "gpt-4"}])
    )
    assert result is None


@mock.patch("drbrain.extractor.concept.get_node_content")
@mock.patch("drbrain.extractor.concept.acall_with_fallback")
def test_extract_concepts_from_tree_merges_results(mock_acall, mock_get_content, tmp_path):
    """Results from multiple sections are merged with dedup."""
    md = tmp_path / "test.md"
    md.write_text("# T\n\n## A\n\nText A\n\n## B\n\nText B\n")
    structure = [
        {"title": "A", "node_id": "0000", "line_num": 1, "nodes": []},
        {"title": "B", "node_id": "0001", "line_num": 3, "nodes": []},
    ]

    mock_get_content.side_effect = lambda p, s, n: (
        "This is substantial content for section "
        + n
        + " with enough text to pass the minimum length threshold."
    )

    # Return different results for each call
    mock_acall.side_effect = [
        {
            "problems": [{"label": "Shared Problem", "confidence": 0.7}],
            "methods": [],
            "conclusions": [],
            "debates": [],
            "gaps": [],
            "actors": [],
            "relations": [],
            "arguments": [],
        },
        {
            "problems": [{"label": "Shared Problem", "confidence": 0.95}],
            "methods": [{"label": "New Method", "confidence": 0.8}],
            "conclusions": [],
            "debates": [],
            "gaps": [],
            "actors": [],
            "relations": [],
            "arguments": [],
        },
    ]

    import asyncio

    result = asyncio.run(
        extract_concepts_from_tree(md, structure, [{"provider": "openai", "model": "gpt-4"}])
    )
    assert result is not None
    # "Shared Problem" should appear once with confidence 0.95
    assert len(result.problems) == 1
    assert result.problems[0]["confidence"] == 0.95
    # "New Method" should be present
    assert len(result.methods) == 1


def test_extract_concepts_from_tree_no_models(tmp_path):
    """Returns None when no models provided."""
    md = tmp_path / "test.md"
    md.write_text("# T\n")

    import asyncio

    result = asyncio.run(extract_concepts_from_tree(md, [], []))
    assert result is None
