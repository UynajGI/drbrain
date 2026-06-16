"""Tests for LLM response caching wired into call_with_fallback / acall_with_fallback.

Verifies that:
- Second identical call short-circuits (no network) when ApiCache is provided
- Different prompts produce cache misses
- acall_text_with_fallback stores/retrieves text wrapped in a dict
- Callers without _cache behave exactly as before (backward compat)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from drbrain.extractor.cache import ApiCache
from drbrain.extractor.llm_client import (
    _build_litellm_kwargs,
    _cache_key,
    acall_text_with_fallback,
    acall_with_fallback,
    acall_with_messages,
    call_with_fallback,
    call_with_messages,
)

MODELS = [{"provider": "openai", "model": "gpt-4o-mini", "api_key": "sk-test"}]


def _mock_response(content: str):
    """Build a fake litellm response object."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
    return resp


# ── _cache_key ────────────────────────────────────────────────────────────


class TestCacheKey:
    def test_same_inputs_same_key(self):
        k1 = _cache_key("m1", "sys", "prompt", 100)
        k2 = _cache_key("m1", "sys", "prompt", 100)
        assert k1 == k2

    def test_different_model_different_key(self):
        assert _cache_key("m1", "sys", "p", 100) != _cache_key("m2", "sys", "p", 100)

    def test_different_prompt_different_key(self):
        assert _cache_key("m", "sys", "p1", 100) != _cache_key("m", "sys", "p2", 100)

    def test_different_max_tokens_different_key(self):
        assert _cache_key("m", "sys", "p", 100) != _cache_key("m", "sys", "p", 200)

    def test_key_is_16_hex_chars(self):
        key = _cache_key("m", "s", "p", 1)
        assert len(key) == 16
        int(key, 16)  # must be valid hex


# ── _build_litellm_kwargs prompt caching ─────────────────────────────────


class TestBuildKwargsPromptCaching:
    """Anthropic prompt caching: cache_control on long system prompts."""

    def test_anthropic_long_system_prompt_gets_cache_control(self):
        """System prompt >= 4000 chars on Anthropic gets cache_control block."""
        cfg = {"provider": "anthropic", "model": "claude-sonnet-4-20250514"}
        long_sys = "x" * 5000
        kwargs = _build_litellm_kwargs(cfg, "hi", long_sys, 100)
        sys_msg = kwargs["messages"][0]
        assert sys_msg["role"] == "system"
        assert isinstance(sys_msg["content"], list)
        block = sys_msg["content"][0]
        assert block["cache_control"] == {"type": "ephemeral"}
        assert block["text"] == long_sys

    def test_anthropic_short_system_prompt_no_cache_control(self):
        """Short system prompt (<4000) stays plain string even on Anthropic."""
        cfg = {"provider": "anthropic", "model": "claude-sonnet-4-20250514"}
        kwargs = _build_litellm_kwargs(cfg, "hi", "short sys", 100)
        sys_msg = kwargs["messages"][0]
        assert isinstance(sys_msg["content"], str)

    def test_non_anthropic_no_cache_control(self):
        """OpenAI/others never get cache_control even with long prompts."""
        cfg = {"provider": "openai", "model": "gpt-4o"}
        long_sys = "x" * 5000
        kwargs = _build_litellm_kwargs(cfg, "hi", long_sys, 100)
        sys_msg = kwargs["messages"][0]
        assert isinstance(sys_msg["content"], str)

    def test_claude_in_model_name_also_triggers(self):
        """Provider 'openai' but model name contains 'claude' (proxy) triggers."""
        cfg = {"provider": "openai", "model": "claude-3-opus", "api_base": "https://proxy"}
        long_sys = "x" * 4500
        kwargs = _build_litellm_kwargs(cfg, "hi", long_sys, 100)
        sys_msg = kwargs["messages"][0]
        assert isinstance(sys_msg["content"], list)
        assert sys_msg["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_no_system_prompt_skips_block(self):
        """Empty system_prompt → no system message at all."""
        cfg = {"provider": "anthropic", "model": "claude-sonnet-4-20250514"}
        kwargs = _build_litellm_kwargs(cfg, "hi", "", 100)
        assert kwargs["messages"][0]["role"] == "user"


# ── call_with_fallback (sync) ────────────────────────────────────────────


class TestCallWithFallbackCache:
    def test_second_call_hits_cache(self, tmp_path):
        """Second identical call must NOT invoke litellm.completion."""
        cache = ApiCache(str(tmp_path / "llm_cache"), ttl=3600)
        call_count = [0]

        def _fake_completion(**kwargs):
            call_count[0] += 1
            return _mock_response('{"concepts": []}')

        with patch("drbrain.extractor.llm_client.litellm") as mock_litellm:
            mock_litellm.completion.side_effect = _fake_completion
            r1 = call_with_fallback("prompt-a", MODELS, "sys", _cache=cache)
            r2 = call_with_fallback("prompt-a", MODELS, "sys", _cache=cache)

        assert r1 == {"concepts": []}
        assert r2 == {"concepts": []}
        assert call_count[0] == 1  # second call hit cache

    def test_different_prompts_both_invoke_llm(self, tmp_path):
        cache = ApiCache(str(tmp_path / "llm_cache"), ttl=3600)
        call_count = [0]

        def _fake_completion(**kwargs):
            call_count[0] += 1
            return _mock_response('{"ok": true}')

        with patch("drbrain.extractor.llm_client.litellm") as mock_litellm:
            mock_litellm.completion.side_effect = _fake_completion
            call_with_fallback("prompt-a", MODELS, "sys", _cache=cache)
            call_with_fallback("prompt-b", MODELS, "sys", _cache=cache)

        assert call_count[0] == 2

    def test_no_cache_backward_compat(self):
        """Without _cache, every call hits litellm (original behavior)."""
        call_count = [0]

        def _fake_completion(**kwargs):
            call_count[0] += 1
            return _mock_response('{"v": 1}')

        with patch("drbrain.extractor.llm_client.litellm") as mock_litellm:
            mock_litellm.completion.side_effect = _fake_completion
            r1 = call_with_fallback("p", MODELS)
            r2 = call_with_fallback("p", MODELS)

        assert r1 == {"v": 1}
        assert r2 == {"v": 1}
        assert call_count[0] == 2


# ── acall_with_fallback (async) ──────────────────────────────────────────


class TestAcallWithFallbackCache:
    @pytest.mark.asyncio
    async def test_second_call_hits_cache(self, tmp_path):
        cache = ApiCache(str(tmp_path / "llm_cache"), ttl=3600)
        call_count = [0]

        async def _fake_acompletion(**kwargs):
            call_count[0] += 1
            return _mock_response('{"async": true}')

        with patch("drbrain.extractor.llm_client.litellm") as mock_litellm:
            mock_litellm.acompletion.side_effect = _fake_acompletion
            r1 = await acall_with_fallback("p", MODELS, "sys", _cache=cache)
            r2 = await acall_with_fallback("p", MODELS, "sys", _cache=cache)

        assert r1 == {"async": True}
        assert r2 == {"async": True}
        assert call_count[0] == 1


# ── acall_text_with_fallback ─────────────────────────────────────────────


class TestAcallTextWithFallbackCache:
    @pytest.mark.asyncio
    async def test_text_cached_and_retrieved(self, tmp_path):
        cache = ApiCache(str(tmp_path / "text_cache"), ttl=3600)
        call_count = [0]

        async def _fake_acompletion(**kwargs):
            call_count[0] += 1
            return _mock_response("plain text response")

        with patch("drbrain.extractor.llm_client.litellm") as mock_litellm:
            mock_litellm.acompletion.side_effect = _fake_acompletion
            r1 = await acall_text_with_fallback("p", MODELS, "sys", _cache=cache)
            r2 = await acall_text_with_fallback("p", MODELS, "sys", _cache=cache)

        assert r1 == "plain text response"
        assert r2 == "plain text response"
        assert call_count[0] == 1  # second call hit cache


# ── call_with_messages cache ─────────────────────────────────────────────


def _mock_messages_response(text: str):
    """Build a fake litellm response for call_with_messages."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = text
    resp.choices[0].message.tool_calls = None
    resp.usage = MagicMock(prompt_tokens=20, completion_tokens=10)
    return resp


class TestCallWithMessagesCache:
    def test_second_identical_call_hits_cache(self, tmp_path):
        """Second identical call_with_messages must NOT invoke litellm.completion."""
        cache = ApiCache(str(tmp_path / "msg_cache"), ttl=3600)
        call_count = [0]
        msgs = [{"role": "user", "content": "hello"}]

        def _fake_completion(**kwargs):
            call_count[0] += 1
            return _mock_messages_response("hi there")

        with patch("drbrain.extractor.llm_client.litellm") as mock_litellm:
            mock_litellm.completion.side_effect = _fake_completion
            r1 = call_with_messages(msgs, MODELS, _cache=cache, temperature=0)
            r2 = call_with_messages(msgs, MODELS, _cache=cache, temperature=0)

        assert r1 == {"text": "hi there", "tool_calls": None, "usage": {"in": 20, "out": 10}}
        assert r2 == {"text": "hi there", "tool_calls": None, "usage": {"in": 20, "out": 10}}
        assert call_count[0] == 1  # second call hit cache

    def test_different_messages_produce_cache_miss(self, tmp_path):
        """Different messages must NOT hit cache — each invokes the LLM."""
        cache = ApiCache(str(tmp_path / "msg_cache"), ttl=3600)
        call_count = [0]
        msgs_a = [{"role": "user", "content": "hello"}]
        msgs_b = [{"role": "user", "content": "goodbye"}]

        def _fake_completion(**kwargs):
            call_count[0] += 1
            return _mock_messages_response("response")

        with patch("drbrain.extractor.llm_client.litellm") as mock_litellm:
            mock_litellm.completion.side_effect = _fake_completion
            call_with_messages(msgs_a, MODELS, _cache=cache)
            call_with_messages(msgs_b, MODELS, _cache=cache)

        assert call_count[0] == 2

    def test_no_cache_backward_compat_messages(self):
        """Without _cache, every call_with_messages hits litellm."""
        call_count = [0]
        msgs = [{"role": "user", "content": "hello"}]

        def _fake_completion(**kwargs):
            call_count[0] += 1
            return _mock_messages_response("response")

        with patch("drbrain.extractor.llm_client.litellm") as mock_litellm:
            mock_litellm.completion.side_effect = _fake_completion
            r1 = call_with_messages(msgs, MODELS)
            r2 = call_with_messages(msgs, MODELS)

        assert r1["text"] == "response"
        assert r2["text"] == "response"
        assert call_count[0] == 2


# ── acall_with_messages cache ───────────────────────────────────────────


class TestAcallWithMessagesCache:
    @pytest.mark.asyncio
    async def test_second_identical_call_hits_cache(self, tmp_path):
        """Second identical acall_with_messages must NOT invoke litellm.acompletion."""
        cache = ApiCache(str(tmp_path / "async_msg_cache"), ttl=3600)
        call_count = [0]
        msgs = [{"role": "user", "content": "hello"}]

        async def _fake_acompletion(**kwargs):
            call_count[0] += 1
            return _mock_messages_response("async hi")

        with patch("drbrain.extractor.llm_client.litellm") as mock_litellm:
            mock_litellm.acompletion.side_effect = _fake_acompletion
            r1 = await acall_with_messages(msgs, MODELS, _cache=cache, temperature=0)
            r2 = await acall_with_messages(msgs, MODELS, _cache=cache, temperature=0)

        assert r1["text"] == "async hi"
        assert r2["text"] == "async hi"
        assert call_count[0] == 1  # second call hit cache
