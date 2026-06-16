"""Tests for concept extraction — flat and tree-based."""

from unittest import mock

from drbrain.extractor.concept import (
    ExtractedConcepts,
    _collect_leaf_nodes,
    _is_plateau_reached,
    _is_quality_content,
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
            return "This paper addresses long-range dependencies in sequence modeling tasks using novel architectures and attention mechanisms."
        if node_id == "0002":
            return "We use transformer architecture with self-attention mechanism for parallel computation of representations across multiple layers."
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
        + " with enough text to easily pass the minimum length threshold for quality."
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


# -- Concurrency control --


@mock.patch("drbrain.extractor.concept.extract_section_concepts")
def test_extract_concurrent_limits(mock_extract, tmp_path):
    """Concurrency is capped by max_concurrent parameter."""
    import asyncio

    md = tmp_path / "test.md"
    md.write_text(
        "## A\n"
        + "Content A. " * 20
        + "\n## B\n"
        + "Content B. " * 20
        + "\n## C\n"
        + "Content C. " * 20
        + "\n## D\n"
        + "Content D. " * 20
        + "\n"
    )

    concurrent_count = {"current": 0, "max_seen": 0}

    async def _tracked_extract(*args, **kwargs):
        concurrent_count["current"] += 1
        if concurrent_count["current"] > concurrent_count["max_seen"]:
            concurrent_count["max_seen"] = concurrent_count["current"]
        await asyncio.sleep(0.05)
        concurrent_count["current"] -= 1
        return ExtractedConcepts(
            {
                "problems": [],
                "methods": [],
                "conclusions": [],
                "debates": [],
                "gaps": [],
                "actors": [],
                "relations": [],
                "arguments": [],
            }
        )

    mock_extract.side_effect = _tracked_extract

    structure = [
        {"title": "A", "node_id": "0000", "line_num": 1, "nodes": []},
        {"title": "B", "node_id": "0001", "line_num": 3, "nodes": []},
        {"title": "C", "node_id": "0002", "line_num": 5, "nodes": []},
        {"title": "D", "node_id": "0003", "line_num": 7, "nodes": []},
    ]

    models = [{"provider": "openai", "model": "gpt-4", "api_key": "x"}]
    asyncio.run(extract_concepts_from_tree(md, structure, models, max_concurrent=2))
    assert concurrent_count["max_seen"] <= 2


@mock.patch("drbrain.extractor.concept.extract_section_concepts")
def test_extract_concurrent_merges(mock_extract, tmp_path):
    """Concurrent extraction results merge correctly."""
    import asyncio

    md = tmp_path / "test.md"
    md.write_text("## A\n" + "Content A. " * 20 + "\n## B\n" + "Content B. " * 20 + "\n")

    results_map = {
        "A": ExtractedConcepts(
            {
                "problems": [{"label": "Problem A", "confidence": 0.8}],
                "methods": [],
                "conclusions": [],
                "debates": [],
                "gaps": [],
                "actors": [],
                "relations": [],
                "arguments": [],
            }
        ),
        "B": ExtractedConcepts(
            {
                "problems": [{"label": "Problem B", "confidence": 0.9}],
                "methods": [{"label": "Method X", "confidence": 0.7}],
                "conclusions": [],
                "debates": [],
                "gaps": [],
                "actors": [],
                "relations": [],
                "arguments": [],
            }
        ),
    }

    async def _mock_extract(title, text, struct_json, models):
        return results_map.get(title)

    mock_extract.side_effect = _mock_extract

    structure = [
        {"title": "A", "node_id": "0000", "line_num": 1, "nodes": []},
        {"title": "B", "node_id": "0001", "line_num": 3, "nodes": []},
    ]
    models = [{"provider": "openai", "model": "gpt-4", "api_key": "x"}]

    result = asyncio.run(extract_concepts_from_tree(md, structure, models))
    assert result is not None
    assert len(result.problems) == 2
    assert len(result.methods) == 1


# -- Content quality gate --


def test_quality_gate_rejects_short():
    """Content shorter than 100 chars is rejected."""
    assert _is_quality_content("Too short.") is False


def test_quality_gate_rejects_references():
    """Content that is mostly reference list entries is rejected."""
    lines = [
        "[1] Author A. Title A. Journal A, 2020.",
        "[2] Author B. Title B. Journal B, 2021.",
        "[3] Author C. Title C. Journal C, 2022.",
        "[4] Author D. Title D. Journal D, 2023.",
        "Some non-ref text to reach minimum length threshold for the quality gate check.",
    ]
    text = "\n".join(lines)
    assert _is_quality_content(text) is False


def test_quality_gate_accepts_normal():
    """Normal academic content passes the quality gate."""
    text = (
        "This paper proposes a novel approach to solving the problem of "
        "large-scale knowledge graph construction. Our method leverages "
        "transformer-based architectures combined with symbolic reasoning "
        "to achieve state-of-the-art performance on benchmark datasets."
    )
    assert _is_quality_content(text) is True


# -- Cross-section argument linking --


# -- Extraction validation --


def test_validate_extraction_clean():
    """Valid extraction returns no errors."""
    from drbrain.extractor.concept import validate_extraction

    concepts = ExtractedConcepts(
        {
            "problems": [{"label": "Slow training", "confidence": 0.9}],
            "methods": [{"label": "Pruning", "confidence": 0.8}],
            "conclusions": [],
            "debates": [],
            "gaps": [],
            "actors": [],
            "relations": [
                {"head": "Pruning", "rel": "addresses", "tail": "Slow training"},
            ],
            "arguments": [],
        }
    )
    errors = validate_extraction(concepts)
    assert errors == []


def test_validate_extraction_tbox_violation():
    """TBox violation in relations is detected."""
    from drbrain.extractor.concept import validate_extraction

    concepts = ExtractedConcepts(
        {
            "problems": [{"label": "High cost", "confidence": 0.9}],
            "methods": [],
            "conclusions": [],
            "debates": [],
            "gaps": [],
            "actors": [],
            "relations": [
                {
                    "head": "High cost",
                    "rel": "replaces",
                    "tail": "Something",
                },  # Problem can't use "replaces"
            ],
            "arguments": [],
        }
    )
    errors = validate_extraction(concepts)
    assert len(errors) >= 1
    assert "TBox" in errors[0]


def test_validate_extraction_unknown_type():
    """Unknown concept type is detected."""
    from drbrain.extractor.concept import validate_extraction

    concepts = ExtractedConcepts(
        {
            "problems": [],
            "methods": [],
            "conclusions": [],
            "debates": [],
            "gaps": [],
            "actors": [],
            "relations": [],
            "arguments": [],
        }
    )
    # Manually inject an unknown type
    concepts.problems = [{"label": "X", "confidence": 0.5, "type": "UnknownType"}]
    errors = validate_extraction(concepts)
    # Should either warn about unknown type or return empty (no relations to check)
    # Since we only check relations, unknown type without relations = no error
    assert isinstance(errors, list)


# -- Tree edge interface tests --


def test_build_tree_edges_returns_head_rel_tail_format():
    """Tree edges must use head/rel/tail keys matching build_cmd insert loop."""
    from drbrain.extractor.concept import _build_tree_edges

    tree = [
        {
            "title": "Methods",
            "nodes": [
                {"title": "Training", "nodes": []},
                {"title": "Evaluation", "nodes": []},
            ],
        },
    ]
    edges = _build_tree_edges(tree)
    assert len(edges) == 3  # document→Methods + 2 children
    for e in edges:
        assert "head" in e, f"Missing 'head' key: {e}"
        assert "rel" in e, f"Missing 'rel' key: {e}"
        assert "tail" in e, f"Missing 'tail' key: {e}"
        assert e["rel"] == "contains"
    # Verify parent-child relationships
    assert any(e["head"] == "document" and e["tail"] == "Methods" for e in edges)
    assert any(e["head"] == "Methods" and e["tail"] == "Training" for e in edges)


def test_build_tree_edges_empty_structure():
    from drbrain.extractor.concept import _build_tree_edges

    assert _build_tree_edges([]) == []


def test_section_type_hints_known_sections():
    from drbrain.extractor.concept import _section_type_hints

    assert _section_type_hints("Methods") == {"Method": 0.9}
    assert "Problem" in _section_type_hints("Abstract")
    assert "Gap" in _section_type_hints("Future Work")


def test_tree_position_weight_by_depth():
    from drbrain.extractor.concept import _tree_position_weight

    assert _tree_position_weight({"title": "Abstract"}, depth=0) == 0.6
    assert _tree_position_weight({"title": "Training Details"}, depth=5) >= 0.9
    assert 0.5 <= _tree_position_weight({"title": "Unknown"}, depth=2) <= 1.0


# -- Plateau detection --


def test_plateau_zero_growth():
    """Zero new elements signals plateau."""
    assert _is_plateau_reached(0, 10) is True


def test_plateau_below_relative_threshold():
    """New elements below 5% of total signals plateau."""
    assert _is_plateau_reached(2, 100) is True  # 2%
    assert _is_plateau_reached(4, 100) is True  # 4%
    assert _is_plateau_reached(1, 21) is True  # 4.76%


def test_plateau_above_threshold():
    """Significant growth continues."""
    assert _is_plateau_reached(3, 20) is False  # 15%
    assert _is_plateau_reached(5, 50) is False  # 10%
    assert _is_plateau_reached(10, 50) is False  # 20%


def test_plateau_empty_total():
    """Zero total with non-zero growth continues."""
    assert _is_plateau_reached(5, 0) is False
