"""Tests for ReasonerAgent.reason() — the multi-turn reasoning orchestrator."""

from __future__ import annotations

from unittest import mock

import pytest


def _client(response=None, side_effect=None, error=None):
    """Async OpenAI client mock wired to the reasoner's factory seam."""
    client = mock.MagicMock()
    if error is not None:
        client.chat.completions.create = mock.AsyncMock(side_effect=error)
    elif side_effect is not None:
        client.chat.completions.create = mock.AsyncMock(side_effect=side_effect)
    else:
        client.chat.completions.create = mock.AsyncMock(return_value=response)
    return client


# ---------------------------------------------------------------------------
# Helpers — build fake OpenAI SDK response objects
# ---------------------------------------------------------------------------


def _make_response(content=None, tool_calls=None):
    """Build a fake OpenAI ``ChatCompletion`` response.

    *content* is the assistant's text reply (used when no tool calls).
    *tool_calls* is a list of dicts ``{"name": str, "arguments": str}``.
    """
    msg = mock.MagicMock()
    msg.content = content
    if tool_calls:
        msg.tool_calls = []
        for i, tc in enumerate(tool_calls):
            func = mock.MagicMock()
            func.name = tc["name"]
            func.arguments = tc["arguments"]
            call = mock.MagicMock()
            call.id = f"call_{i}"
            call.function = func
            msg.tool_calls.append(call)
    else:
        msg.tool_calls = None

    choice = mock.MagicMock()
    choice.message = msg
    resp = mock.MagicMock()
    resp.choices = [choice]
    return resp


# NOTE: the OpenAI client factory is imported *inside* the ``reason()``
# method body, so we patch the single factory seam
# (``drbrain.extractor.llm_client._aopenai_client``) instead of any
# reasoner-module attribute.


# ---------------------------------------------------------------------------
# Test 1: No models configured → early return
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reason_no_models():
    """Returns 'No LLM models configured.' when models list is empty."""
    from drbrain.extractor.reasoner import ReasonerAgent

    agent = ReasonerAgent(models=[])
    result = await agent.reason("What is attention?")
    assert result == "No LLM models configured."


# ---------------------------------------------------------------------------
# Test 2: Single-turn — LLM returns final answer without tool calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reason_single_turn():
    """LLM returns a final answer on the first call; no tool invocations."""
    from drbrain.extractor.reasoner import ReasonerAgent

    models = [{"provider": "openai", "model": "gpt-4o"}]
    agent = ReasonerAgent(models=models)

    fake_response = _make_response(content="Attention is a mechanism in neural networks.")

    with (
        mock.patch(
            "drbrain.extractor.llm_client._aopenai_client",
            return_value=_client(response=fake_response),
        ) as mock_acompletion,
    ):
        result = await agent.reason("What is attention?", max_turns=5)

    assert result == "Attention is a mechanism in neural networks."
    mock_acompletion.return_value.chat.completions.create.assert_awaited_once()


# ---------------------------------------------------------------------------
# Test 3: Multi-turn — LLM requests tool, then returns answer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reason_multi_turn():
    """LLM calls search_concepts, receives result, then returns answer."""
    from drbrain.extractor.reasoner import ReasonerAgent

    models = [{"provider": "openai", "model": "gpt-4o"}]
    agent = ReasonerAgent(models=models)

    tool_response = _make_response(
        tool_calls=[
            {"name": "search_concepts", "arguments": '{"query": "transformer", "limit": 3}'}
        ],
    )
    final_response = _make_response(content="Transformers are a type of neural architecture.")

    with (
        mock.patch(
            "drbrain.extractor.llm_client._aopenai_client",
            return_value=_client(side_effect=[tool_response, final_response]),
        ) as mock_acompletion,
        mock.patch.object(
            agent,
            "_search_concepts",
            return_value=[{"label": "Transformer", "type": "method", "score": 0.95}],
        ),
    ):
        result = await agent.reason("What is a transformer?", max_turns=5)

    assert result == "Transformers are a type of neural architecture."
    assert mock_acompletion.return_value.chat.completions.create.await_count == 2


# ---------------------------------------------------------------------------
# Test 4: Max turns exhausted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reason_max_turns():
    """After max_turns tool-calling iterations, returns exhaustion message."""
    from drbrain.extractor.reasoner import ReasonerAgent

    models = [{"provider": "openai", "model": "gpt-4o"}]
    agent = ReasonerAgent(models=models)

    # Every response requests another tool call — never gives a final answer
    tool_response = _make_response(
        tool_calls=[{"name": "get_neighbors", "arguments": '{"node": "A"}'}],
    )

    with (
        mock.patch(
            "drbrain.extractor.llm_client._aopenai_client",
            return_value=_client(response=tool_response),
        ) as mock_acompletion,
        mock.patch.object(agent, "_get_neighbors", return_value=[]),
    ):
        result = await agent.reason("Explore the graph", max_turns=2)

    assert result == "Unable to answer after maximum reasoning turns."
    # With max_turns=2, acompletion is called exactly twice
    assert mock_acompletion.return_value.chat.completions.create.await_count == 2


# ---------------------------------------------------------------------------
# Test 5: Exception in the LLM call LLM call → returns error string
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reason_llm_error():
    """Exceptions from the LLM call are caught and returned as error."""
    from drbrain.extractor.reasoner import ReasonerAgent

    models = [{"provider": "openai", "model": "gpt-4o"}]
    agent = ReasonerAgent(models=models)

    with (
        mock.patch(
            "drbrain.extractor.llm_client._aopenai_client",
            return_value=_client(error=RuntimeError("API down")),
        ),
    ):
        result = await agent.reason("Test question")

    assert "Reasoning error:" in result
    assert "API down" in result


# ---------------------------------------------------------------------------
# Test 6: Unknown tool name in tool_calls → empty result appended, loop continues
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reason_unknown_tool():
    """Unknown tool names get an empty result list and the loop continues."""
    from drbrain.extractor.reasoner import ReasonerAgent

    models = [{"provider": "openai", "model": "gpt-4o"}]
    agent = ReasonerAgent(models=models)

    unknown_tool_response = _make_response(
        tool_calls=[{"name": "nonexistent_tool", "arguments": "{}"}],
    )
    final_response = _make_response(content="Here is my answer.")

    with (
        mock.patch(
            "drbrain.extractor.llm_client._aopenai_client",
            return_value=_client(side_effect=[unknown_tool_response, final_response]),
        ),
    ):
        result = await agent.reason("Question", max_turns=5)

    assert result == "Here is my answer."


# ---------------------------------------------------------------------------
# Test 7: closure_context is appended to system prompt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reason_closure_context_in_system_prompt():
    """When closure_context is set, it appears in the system message."""
    from drbrain.extractor.reasoner import ReasonerAgent

    models = [{"provider": "openai", "model": "gpt-4o"}]
    agent = ReasonerAgent(models=models, closure_context="A --[inferred: subsumes]--> B")

    fake_response = _make_response(content="Answer")

    with (
        mock.patch(
            "drbrain.extractor.llm_client._aopenai_client",
            return_value=_client(response=fake_response),
        ) as mock_acompletion,
    ):
        await agent.reason("Test", max_turns=1)

    call_kwargs = mock_acompletion.return_value.chat.completions.create.call_args[1]
    system_msg = call_kwargs["messages"][0]["content"]
    assert "inferred" in system_msg
    assert "A --[inferred: subsumes]--> B" in system_msg


# ---------------------------------------------------------------------------
# Test 8: Multiple tool calls in a single turn
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reason_multiple_tools_single_turn():
    """LLM requests two tool calls in one turn; both are executed before next LLM call."""
    from drbrain.extractor.reasoner import ReasonerAgent

    models = [{"provider": "openai", "model": "gpt-4o"}]
    agent = ReasonerAgent(models=models)

    multi_tool_response = _make_response(
        tool_calls=[
            {"name": "search_concepts", "arguments": '{"query": "transformer"}'},
            {"name": "get_neighbors", "arguments": '{"node": "Transformer"}'},
        ],
    )
    final_response = _make_response(content="Combined answer using both tools.")

    with (
        mock.patch(
            "drbrain.extractor.llm_client._aopenai_client",
            return_value=_client(side_effect=[multi_tool_response, final_response]),
        ) as mock_acompletion,
        mock.patch.object(
            agent,
            "_search_concepts",
            return_value=[{"label": "Transformer", "type": "method", "score": 0.9}],
        ),
        mock.patch.object(
            agent,
            "_get_neighbors",
            return_value=[
                {"target": "Attention", "source": "Transformer", "distance": 1, "path": []},
            ],
        ),
    ):
        result = await agent.reason("What is a transformer?", max_turns=5)

    assert result == "Combined answer using both tools."
    assert mock_acompletion.return_value.chat.completions.create.await_count == 2


# ---------------------------------------------------------------------------
# Test 9: model config with api_key and base_url forwarded to the OpenAI client factory OpenAI client factory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reason_model_config_forwarded():
    """api_key and base_url from model config reach the OpenAI client factory."""
    from drbrain.extractor.reasoner import ReasonerAgent

    models = [
        {
            "provider": "openai",
            "model": "gpt-4o",
            "api_key": "sk-test",
            "base_url": "https://example.com/v1",
        },
    ]
    agent = ReasonerAgent(models=models)

    fake_response = _make_response(content="ok")
    with (
        mock.patch(
            "drbrain.extractor.llm_client._aopenai_client",
            return_value=_client(response=fake_response),
        ) as mock_acompletion,
    ):
        await agent.reason("Test", max_turns=1)

    mock_acompletion.assert_called_once_with("sk-test", "https://example.com/v1")
    call_kwargs = mock_acompletion.return_value.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "gpt-4o"  # bare wire name; routing via base_url
    assert call_kwargs["temperature"] == 0.3
    assert call_kwargs["max_tokens"] == 1024
