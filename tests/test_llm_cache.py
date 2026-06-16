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
    _cache_key,
    acall_text_with_fallback,
    acall_with_fallback,
    call_with_fallback,
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
