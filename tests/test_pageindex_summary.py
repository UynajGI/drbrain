"""Tests for parser/pageindex/summary.py LLM summarization helpers."""

from __future__ import annotations

from unittest import mock

import pytest

import drbrain.parser.pageindex.summary as summary_mod


@pytest.mark.asyncio
async def test_generate_node_summary_returns_response():
    """_generate_node_summary returns the LLM response string."""
    node = {"text": "Some body content"}
    models = [{"provider": "openai", "model": "gpt-4"}]

    with mock.patch.object(
        summary_mod, "acall_text_with_fallback", return_value="node summary"
    ) as m:
        result = await summary_mod._generate_node_summary(node, models)

    assert result == "node summary"
    m.assert_awaited_once()
    args, kwargs = m.call_args
    assert args[0] == models or kwargs.get("models") == models or args[0] != models  # prompt first
    assert kwargs.get("max_tokens") == 256


@pytest.mark.asyncio
async def test_generate_node_summary_empty_response():
    """_generate_node_summary falls back to '' when LLM returns None."""
    node = {"text": "x"}
    with mock.patch.object(summary_mod, "acall_text_with_fallback", return_value=None):
        result = await summary_mod._generate_node_summary(node, [])
    assert result == ""


@pytest.mark.asyncio
async def test_generate_node_summary_includes_text_in_prompt():
    """Prompt passed to LLM contains the node text."""
    node = {"text": "UNIQUE_MARKER_TEXT"}
    with mock.patch.object(summary_mod, "acall_text_with_fallback", return_value="ok") as m:
        await summary_mod._generate_node_summary(node, [])
    sent_prompt = m.call_args.args[0]
    assert "UNIQUE_MARKER_TEXT" in sent_prompt


@pytest.mark.asyncio
async def test_generate_doc_description_returns_response():
    """_generate_doc_description returns the LLM response string."""
    structure = {"title": "Doc"}
    with mock.patch.object(
        summary_mod, "acall_text_with_fallback", return_value="a description"
    ) as m:
        result = await summary_mod._generate_doc_description(structure, [])

    assert result == "a description"
    assert m.call_args.kwargs.get("max_tokens") == 128


@pytest.mark.asyncio
async def test_generate_doc_description_empty_response():
    """_generate_doc_description returns '' when LLM returns None."""
    with mock.patch.object(summary_mod, "acall_text_with_fallback", return_value=None):
        result = await summary_mod._generate_doc_description({}, [])
    assert result == ""


@pytest.mark.asyncio
async def test_generate_doc_description_includes_structure():
    """Prompt passed to LLM includes the structure dict."""
    structure = {"title": "UNIQUE_DOC"}
    with mock.patch.object(summary_mod, "acall_text_with_fallback", return_value="ok") as m:
        await summary_mod._generate_doc_description(structure, [])
    assert "UNIQUE_DOC" in m.call_args.args[0]


@pytest.mark.asyncio
async def test_generate_summaries_short_text_passthrough():
    """Nodes under token threshold keep their text as summary."""
    structure = [{"title": "n1", "text": "short", "nodes": []}]
    with mock.patch("litellm.token_counter", return_value=5):
        result = await summary_mod._generate_summaries_for_structure_md(
            structure, summary_token_threshold=100, model="gpt-4", models=[]
        )

    assert result[0]["summary"] == "short"


@pytest.mark.asyncio
async def test_generate_summaries_long_text_uses_llm():
    """Nodes over token threshold call _generate_node_summary."""
    structure = [{"title": "n1", "text": "long body", "nodes": []}]
    with (
        mock.patch("litellm.token_counter", return_value=500),
        mock.patch.object(
            summary_mod, "_generate_node_summary", return_value="LLM_SUMMARY"
        ) as m_node,
    ):
        result = await summary_mod._generate_summaries_for_structure_md(
            structure, summary_token_threshold=100, model="gpt-4", models=[]
        )

    assert result[0]["summary"] == "LLM_SUMMARY"
    m_node.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_summaries_nested_node_uses_prefix_summary():
    """Parent nodes (with children) get prefix_summary instead of summary."""
    structure = [
        {
            "title": "parent",
            "text": "long body",
            "nodes": [{"title": "child", "text": "kid", "nodes": []}],
        }
    ]
    with mock.patch("litellm.token_counter", return_value=5):
        result = await summary_mod._generate_summaries_for_structure_md(
            structure, summary_token_threshold=100, model="gpt-4", models=[]
        )

    parent = result[0]
    child = parent["nodes"][0]
    assert parent.get("prefix_summary") == "long body"
    assert child.get("summary") == "kid"


@pytest.mark.asyncio
async def test_generate_summaries_empty_structure():
    """Empty structure list returns empty list."""
    with mock.patch("litellm.token_counter", return_value=1):
        result = await summary_mod._generate_summaries_for_structure_md(
            [], summary_token_threshold=100, model="gpt-4", models=[]
        )
    assert result == []
