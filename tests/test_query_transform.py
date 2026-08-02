"""Tests for HyDE query transform.

Uses an injectable LLM caller — no real model required. Verifies:
  - successful transform replaces the query with the hypothesis
  - LLM failure / empty models returns the original query unchanged
  - the retriever never sees an exception (graceful degradation)
  - multiple generated docs are merged
  - async variant behaves identically
"""

from __future__ import annotations

import asyncio

from drbrain.query.query_transform import HydeResult, ahyde_transform, hyde_transform

# ── sync ─────────────────────────────────────────────────────────────────────


def _stub_caller(return_value: str | None, *, fail: bool = False):
    """Build a sync caller stub. Records call count for inspection."""

    def _call(*, prompt, models, system_prompt, max_tokens):  # noqa: ANN001
        _call.count += 1  # type: ignore[attr-defined]
        if fail:
            raise RuntimeError("simulated LLM failure")
        return return_value

    _call.count = 0  # type: ignore[attr-defined]
    return _call


def test_hyde_transforms_query_when_llm_succeeds():
    """A successful LLM call replaces the query with the hypothesis."""
    caller = _stub_caller("Turbulent drag reduction is achieved via polymer additives.")

    result = hyde_transform("How to reduce drag?", [{"provider": "x", "model": "y"}], caller=caller)

    assert isinstance(result, HydeResult)
    assert result.transformed is True
    assert result.query == "Turbulent drag reduction is achieved via polymer additives."
    assert result.original == "How to reduce drag?"
    assert result.hypothesis is not None
    assert caller.count == 1


def test_hyde_returns_original_on_llm_failure():
    """An LLM exception must not propagate; original query returned."""
    caller = _stub_caller(None, fail=True)

    result = hyde_transform("any question", [{"provider": "x", "model": "y"}], caller=caller)

    assert result.transformed is False
    assert result.query == "any question"
    assert result.hypothesis is None


def test_hyde_returns_original_on_none_response():
    """A None response (all models failed) falls back to original query."""
    caller = _stub_caller(None)

    result = hyde_transform("question", [{"provider": "x", "model": "y"}], caller=caller)

    assert result.transformed is False
    assert result.query == "question"


def test_hyde_empty_models_returns_original():
    """No models configured → no LLM call, original query preserved."""
    result = hyde_transform("question", [], caller=_stub_caller("should not be called"))

    assert result.transformed is False
    assert result.query == "question"


def test_hyde_empty_question_returns_original():
    """Blank question short-circuits without calling the LLM."""
    caller = _stub_caller("hypothesis")

    result = hyde_transform("   ", [{"provider": "x", "model": "y"}], caller=caller)

    assert result.transformed is False
    assert result.query == "   "
    assert caller.count == 0


def test_hyde_merges_multiple_docs():
    """n_docs>1 generates several hypotheses and concatenates them."""
    caller = _stub_caller("doc text")

    result = hyde_transform("question", [{"provider": "x", "model": "y"}], n_docs=3, caller=caller)

    assert result.transformed is True
    assert caller.count == 3
    # Three identical "doc text" merged → joined by blank lines
    assert result.hypothesis == "doc text\n\ndoc text\n\ndoc text"


# ── async ────────────────────────────────────────────────────────────────────


def _async_stub(return_value: str | None, *, fail: bool = False):
    """Build an async caller stub."""

    async def _call(*, prompt, models, system_prompt, max_tokens, _cache=None):  # noqa: ANN001
        _call.count += 1  # type: ignore[attr-defined]
        if fail:
            raise RuntimeError("async LLM failure")
        return return_value

    _call.count = 0  # type: ignore[attr-defined]
    return _call


def test_ahyde_transforms_query_when_llm_succeeds():
    """Async variant transforms identically to sync on success."""
    caller = _async_stub("Hypothesis paragraph about the method.")

    result = asyncio.run(
        ahyde_transform("question", [{"provider": "x", "model": "y"}], caller=caller)
    )

    assert result.transformed is True
    assert result.query == "Hypothesis paragraph about the method."


def test_ahyde_returns_original_on_failure():
    """Async variant degrades gracefully on LLM exception."""
    caller = _async_stub(None, fail=True)

    result = asyncio.run(
        ahyde_transform("question", [{"provider": "x", "model": "y"}], caller=caller)
    )

    assert result.transformed is False
    assert result.query == "question"
