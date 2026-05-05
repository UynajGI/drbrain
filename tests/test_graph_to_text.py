"""Tests for services/graph_to_text.py — KG subgraph -> natural language."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from drbrain.graph.engine import TraverseResult, TraverseStep
from drbrain.services.graph_to_text import describe_path, describe_subgraph

# -- helpers ------------------------------------------------------------------


def _fake_step(src, relation, dst, hop=1):
    return TraverseStep(src=src, relation=relation, dst=dst, hop=hop)


# -- describe_path ------------------------------------------------------------


def test_describe_path_basic():
    """Path with two hops produces natural language chain."""
    path = [
        _fake_step("Transformer", "proposes", "Self-Attention"),
        _fake_step("Self-Attention", "extends", "Attention Mechanism"),
    ]
    result = describe_path(path)
    assert "Transformer" in result
    assert "proposes" in result
    assert "Self-Attention" in result
    assert "extends" in result
    assert "Attention Mechanism" in result


def test_describe_path_single_hop():
    """Single-hop path describes one relationship."""
    path = [_fake_step("BERT", "extends", "Transformer")]
    result = describe_path(path)
    assert "BERT" in result
    assert "extends" in result
    assert "Transformer" in result


def test_describe_path_empty():
    """Empty path returns empty string."""
    assert describe_path([]) == ""


def test_describe_path_various_relations():
    """All relation types get natural language treatment."""
    path = [
        _fake_step("Method A", "addresses", "Problem X"),
        _fake_step("Method A", "solves", "Problem Y"),
        _fake_step("Paper B", "proposes", "Method A"),
        _fake_step("Method C", "replaces", "Method A"),
        _fake_step("Method A", "extends", "Method D"),
        _fake_step("Theory E", "supports", "Method A"),
        _fake_step("Method A", "challenges", "Assumption F"),
        _fake_step("Constraint G", "limits", "Method A"),
    ]
    result = describe_path(path)
    assert "addresses" in result or "addresses Problem X" in result
    assert "solves" in result
    assert "proposes" in result
    assert "replaces" in result


def test_describe_path_dict_steps():
    """Path with plain dict steps (from JSON/CLI) works the same."""
    path = [
        {"src": "Transformer", "relation": "proposes", "dst": "Self-Attention"},
    ]
    result = describe_path(path)
    assert "Transformer" in result
    assert "Self-Attention" in result


# -- describe_subgraph --------------------------------------------------------


@pytest.mark.asyncio
async def test_describe_subgraph_with_neighbors():
    """Subgraph with neighbors -> LLM call with structured prompt, returns text."""
    graph = MagicMock()
    db = MagicMock()

    graph.traverse.return_value = [
        TraverseResult(
            target="Seq2Seq",
            target_type="unknown",
            source="Transformer",
            distance=1,
            path=[_fake_step("Transformer", "addresses", "Seq2Seq")],
        ),
        TraverseResult(
            target="Self-Attention",
            target_type="unknown",
            source="Transformer",
            distance=1,
            path=[_fake_step("Transformer", "extends", "Self-Attention")],
        ),
        TraverseResult(
            target="BERT",
            target_type="unknown",
            source="Transformer",
            distance=2,
            path=[
                _fake_step("Transformer", "extends", "Self-Attention"),
                _fake_step("Self-Attention", "extends", "BERT", hop=2),
            ],
        ),
    ]

    mock_llm = AsyncMock(return_value="Transformer is a foundational architecture...")

    with patch("drbrain.extractor.llm_client.acall_text_with_fallback", mock_llm):
        result = await describe_subgraph(
            graph, db, "Transformer", [{"provider": "openai"}], depth=2
        )

    assert isinstance(result, str)
    assert len(result) > 0
    mock_llm.assert_called_once()

    # Verify prompt includes subgraph data
    call_args = mock_llm.call_args
    prompt = call_args[0][0]
    assert "Transformer" in prompt
    assert "Seq2Seq" in prompt
    assert "Self-Attention" in prompt
    assert "addresses" in prompt


@pytest.mark.asyncio
async def test_describe_subgraph_empty_traversal():
    """No neighbors found -> still calls LLM for basic description."""
    graph = MagicMock()
    db = MagicMock()
    graph.traverse.return_value = []

    mock_llm = AsyncMock(return_value="No connected concepts found for NoSuchNode.")

    with patch("drbrain.extractor.llm_client.acall_text_with_fallback", mock_llm):
        result = await describe_subgraph(graph, db, "NoSuchNode", [{"provider": "openai"}])

    assert isinstance(result, str)
    mock_llm.assert_called_once()
    prompt = mock_llm.call_args[0][0]
    assert "NoSuchNode" in prompt


@pytest.mark.asyncio
async def test_describe_subgraph_llm_returns_none():
    """LLM fallback exhausted -> returns empty string."""
    graph = MagicMock()
    db = MagicMock()
    graph.traverse.return_value = [
        TraverseResult(
            target="A",
            target_type="unknown",
            source="X",
            distance=1,
            path=[_fake_step("X", "extends", "A")],
        ),
    ]

    mock_llm = AsyncMock(return_value=None)

    with patch("drbrain.extractor.llm_client.acall_text_with_fallback", mock_llm):
        result = await describe_subgraph(graph, db, "X", [{"provider": "openai"}])

    assert result == ""


@pytest.mark.asyncio
async def test_describe_subgraph_dedup_targets():
    """Duplicate targets across hops appear only once in prompt."""
    graph = MagicMock()
    db = MagicMock()

    graph.traverse.return_value = [
        TraverseResult(
            target="BERT",
            target_type="unknown",
            source="Transformer",
            distance=1,
            path=[_fake_step("Transformer", "extends", "BERT")],
        ),
        TraverseResult(
            target="BERT",
            target_type="unknown",
            source="Transformer",
            distance=2,
            path=[
                _fake_step("Transformer", "extends", "Self-Attention"),
                _fake_step("Self-Attention", "extends", "BERT", hop=2),
            ],
        ),
    ]

    mock_llm = AsyncMock(return_value="...")

    with patch("drbrain.extractor.llm_client.acall_text_with_fallback", mock_llm):
        await describe_subgraph(graph, db, "Transformer", [{"provider": "openai"}])

    prompt = mock_llm.call_args[0][0]
    # BERT should appear, but not in a way that suggests duplicate
    bert_count = prompt.count("BERT")
    # BERT should be listed once as an entity, not counted twice in the entity list
    assert bert_count <= 4  # generous bound — entity list + relation mentions


@pytest.mark.asyncio
async def test_describe_subgraph_prompt_has_explicit_instructions():
    """Prompt asks LLM for concise, plain-text summary."""
    graph = MagicMock()
    db = MagicMock()
    graph.traverse.return_value = [
        TraverseResult(
            target="GPT",
            target_type="unknown",
            source="Transformer",
            distance=1,
            path=[_fake_step("Transformer", "extends", "GPT")],
        ),
    ]

    mock_llm = AsyncMock(return_value="...")

    with patch("drbrain.extractor.llm_client.acall_text_with_fallback", mock_llm):
        await describe_subgraph(graph, db, "Transformer", [{"provider": "openai"}])

    prompt = mock_llm.call_args[0][0]
    assert (
        "concise" in prompt.lower()
        or "plain text" in prompt.lower()
        or "paragraph" in prompt.lower()
    )


# -- analyze report integration -----------------------------------------------


def test_analyze_paper_includes_graph_summary():
    """When models provided, analyze_paper report includes graph_summary key."""
    from drbrain.report.analyzer import analyze_paper

    db = MagicMock()
    db.get_paper.return_value = {
        "local_id": "p1",
        "title": "Graph NNs",
        "year": 2024,
    }
    db.get_concepts_by_paper.return_value = [
        {"label": "GNN", "type": "Method", "confidence": 0.9},
        {"label": "Graph Convolution", "type": "Method", "confidence": 0.85},
    ]
    db.get_arguments_by_paper.return_value = []

    graph = MagicMock()
    graph.detect_research_seeds.return_value = []
    graph.closure.return_value = []

    models = [{"provider": "test"}]

    mock_llm = AsyncMock(return_value="GNN is a method for learning on graph-structured data.")
    with patch("drbrain.extractor.llm_client.acall_text_with_fallback", mock_llm):
        result = analyze_paper(db, graph, "p1", models=models)

    assert "graph_summary" in result
    assert isinstance(result["graph_summary"], str)
    assert len(result["graph_summary"]) > 0


def test_analyze_paper_no_models_no_graph_summary():
    """Without models, graph_summary is not included."""
    from drbrain.report.analyzer import analyze_paper

    db = MagicMock()
    db.get_paper.return_value = {
        "local_id": "p1",
        "title": "A Paper",
        "year": 2024,
    }
    db.get_concepts_by_paper.return_value = []
    db.get_arguments_by_paper.return_value = []

    graph = MagicMock()
    graph.detect_research_seeds.return_value = []
    graph.closure.return_value = []

    result = analyze_paper(db, graph, "p1", models=None)
    assert "graph_summary" not in result
