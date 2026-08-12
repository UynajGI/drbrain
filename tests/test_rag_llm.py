"""T2 tests: ``DrbrainLLM`` bridge — construction, delegation, fallback chain,
cache behaviour, and ``Settings.llm`` wiring.

Mocked unit tests must pass offline (no network, no GPU). The live-call test
is marked ``integration`` and skipped by default (``-m "not integration"``).
"""

import importlib.util
from pathlib import Path

import pytest

from drbrain.config import Config, LlamaIndexConfig
from drbrain.rag.llm import DrbrainLLM, init_llamaindex_settings

_HAS_LLAMA_INDEX = importlib.util.find_spec("llama_index") is not None

pytestmark = pytest.mark.skipif(not _HAS_LLAMA_INDEX, reason="llama_index not installed")

MODELS = [
    {"provider": "openai", "model": "gpt-4o", "api_key": "k1", "base_url": None},
    {"provider": "openai", "model": "gpt-4o-mini", "api_key": "k2", "base_url": None},
]


def _cfg(tmp_path=None, models=None, cache_ttl=0) -> Config:
    c = Config()
    c.llm.models = list(models) if models is not None else list(MODELS)
    c.api.cache_ttl = cache_ttl
    if tmp_path is not None:
        c.dirs.cache = str(tmp_path)
    return c


class _StubMetrics:
    def record_llm(self, **kwargs):  # noqa: ANN001, ANN002, ANN003
        pass


class _FakeMessage:
    content = "fallback ok"


class _FakeChoice:
    message = _FakeMessage()


class _FakeResponse:
    choices = [_FakeChoice()]
    usage = None


# ── construction ──


def test_constructor_defaults():
    llm = DrbrainLLM(_cfg())
    assert llm.temperature == 0.1  # drbrain JSON/text default
    assert llm.max_tokens == 4096
    assert llm.context_window == 16384
    assert llm.model_name == "openai/gpt-4o"  # first entry in the chain


def test_constructor_temperature_max_tokens_overrides():
    llm = DrbrainLLM(_cfg(), temperature=0.7, max_tokens=128)
    assert llm.temperature == 0.7
    assert llm.max_tokens == 128


def test_constructor_empty_models():
    llm = DrbrainLLM(_cfg(models=[]))
    assert llm.model_name == "drbrain/unknown"


def test_metadata():
    llm = DrbrainLLM(_cfg(), max_tokens=256)
    md = llm.metadata
    assert md.is_chat_model is True
    assert md.is_function_calling_model is False
    assert md.model_name == "openai/gpt-4o"
    assert md.num_output == 256
    assert md.context_window == 16384


# ── completion delegation ──


def test_complete_sync_delegates(monkeypatch):
    monkeypatch.setattr("drbrain.metrics.get_metrics", lambda: _StubMetrics())
    captured = {}

    def fake(prompt, models, max_tokens=1024, **kwargs):
        captured["models"] = models
        captured["max_tokens"] = max_tokens
        return "sync text"

    monkeypatch.setattr("drbrain.extractor.llm_client.call_text_with_fallback", fake)
    llm = DrbrainLLM(_cfg())
    resp = llm.complete("hello")
    assert resp.text == "sync text"
    assert resp.delta == "sync text"
    assert resp.additional_kwargs["model"] == "openai/gpt-4o"
    assert captured["models"] == MODELS  # full fallback chain forwarded
    assert captured["max_tokens"] == llm.max_tokens


async def test_acomplete_delegates(monkeypatch):
    monkeypatch.setattr("drbrain.metrics.get_metrics", lambda: _StubMetrics())
    captured = {}

    async def fake(prompt, models, max_tokens=1024, **_cache):
        captured["models"] = models
        captured["max_tokens"] = max_tokens
        return "async text"

    monkeypatch.setattr("drbrain.extractor.llm_client.acall_text_with_fallback", fake)
    llm = DrbrainLLM(_cfg())
    resp = await llm.acomplete("hello")
    assert resp.text == "async text"
    assert captured["models"] == MODELS


async def test_acomplete_cache_disabled_when_ttl_zero(monkeypatch, tmp_path):
    monkeypatch.setattr("drbrain.metrics.get_metrics", lambda: _StubMetrics())
    captured = {}

    async def fake(prompt, models, max_tokens=1024, **kw):
        captured["cache"] = kw.get("_cache")
        return "x"

    monkeypatch.setattr("drbrain.extractor.llm_client.acall_text_with_fallback", fake)
    llm = DrbrainLLM(_cfg(tmp_path, cache_ttl=0))
    await llm.acomplete("hello")
    assert captured["cache"] is None


# ── fallback chain: first model fails → second succeeds ──


async def test_fallback_chain_first_fails_second_succeeds(monkeypatch, tmp_path):
    """DrbrainLLM routes through the REAL llm_client chain; litellm mocked."""
    monkeypatch.setattr("drbrain.metrics.get_metrics", lambda: _StubMetrics())
    called: list[str] = []

    async def fake_acompletion(**kwargs):
        called.append(kwargs["model"])
        if len(called) == 1:  # first model in the chain fails
            raise RuntimeError("provider down")
        return _FakeResponse()

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)
    llm = DrbrainLLM(_cfg(tmp_path), max_tokens=64)
    resp = await llm.acomplete("retry me")
    assert resp.text == "fallback ok"
    assert called == ["openai/gpt-4o", "openai/gpt-4o-mini"]  # order preserved


async def test_cache_hit_skips_network(monkeypatch, tmp_path):
    """Real llm_client + real ApiCache: 2nd identical call never hits network."""
    monkeypatch.setattr("drbrain.metrics.get_metrics", lambda: _StubMetrics())
    calls = {"n": 0}

    async def fake_acompletion(**kwargs):
        calls["n"] += 1
        return _FakeResponse()

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)
    llm = DrbrainLLM(_cfg(tmp_path, cache_ttl=86400), max_tokens=64)
    r1 = await llm.acomplete("same prompt")
    r2 = await llm.acomplete("same prompt")
    assert r1.text == r2.text == "fallback ok"
    assert calls["n"] == 1  # second call served from ApiCache


# ── chat delegation ──


def test_chat_converts_messages_and_forwards_temperature(monkeypatch):
    from llama_index.core.base.llms.types import ChatMessage, MessageRole

    monkeypatch.setattr("drbrain.metrics.get_metrics", lambda: _StubMetrics())
    captured = {}

    def fake(messages, models, max_tokens=1024, temperature=0.3, **_kw):
        captured["messages"] = messages
        captured["temperature"] = temperature
        captured["max_tokens"] = max_tokens
        return {"text": "assistant reply", "tool_calls": None, "usage": {"in": 5, "out": 9}}

    monkeypatch.setattr("drbrain.extractor.llm_client.call_with_messages", fake)
    llm = DrbrainLLM(_cfg(), temperature=0.7)
    resp = llm.chat(
        [
            ChatMessage(role=MessageRole.SYSTEM, content="sys"),
            ChatMessage(role=MessageRole.USER, content="hi"),
        ]
    )
    assert captured["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    assert captured["temperature"] == 0.7  # temperature passthrough
    assert captured["max_tokens"] == llm.max_tokens
    assert resp.message.role == MessageRole.ASSISTANT
    assert resp.message.content == "assistant reply"
    assert resp.delta == "assistant reply"


async def test_achat_forwards_cache(monkeypatch, tmp_path):
    from llama_index.core.base.llms.types import ChatMessage, MessageRole

    monkeypatch.setattr("drbrain.metrics.get_metrics", lambda: _StubMetrics())
    captured = {}

    async def fake(messages, models, max_tokens=1024, temperature=0.3, **kw):
        captured["cache"] = kw.get("_cache")
        captured["temperature"] = temperature
        return {"text": "ok", "tool_calls": None}

    monkeypatch.setattr("drbrain.extractor.llm_client.acall_with_messages", fake)
    llm = DrbrainLLM(_cfg(tmp_path, cache_ttl=86400))
    await llm.achat([ChatMessage(role=MessageRole.USER, content="hi")])
    from drbrain.extractor.cache import ApiCache

    assert isinstance(captured["cache"], ApiCache)
    assert captured["temperature"] == 0.1


def test_chat_tool_calls_surface_in_message_kwargs(monkeypatch):
    from llama_index.core.base.llms.types import ChatMessage, MessageRole

    monkeypatch.setattr("drbrain.metrics.get_metrics", lambda: _StubMetrics())

    def fake(messages, models, max_tokens=1024, temperature=0.3, **_kw):
        return {"text": "", "tool_calls": [{"id": "c1", "type": "function"}], "usage": None}

    monkeypatch.setattr("drbrain.extractor.llm_client.call_with_messages", fake)
    llm = DrbrainLLM(_cfg())
    resp = llm.chat([ChatMessage(role=MessageRole.USER, content="hi")])
    assert resp.message.additional_kwargs["tool_calls"] == [{"id": "c1", "type": "function"}]


# ── streaming: real per-token through litellm (T9) ──────────────────────────


class _FakeStreamChunk:
    """One litellm-style stream chunk with a text delta."""

    def __init__(self, content: str) -> None:
        self.choices = [type("_C", (), {"delta": type("_D", (), {"content": content})})()]


class _FakeStream:
    """An iterable of stream chunks (``for chunk in response`` works)."""

    def __init__(self, parts: list[str]) -> None:
        self.parts = parts
        self.usage = None

    def __iter__(self):
        for p in self.parts:
            yield _FakeStreamChunk(p)


class _FakeAsyncStream:
    """An async-iterable of stream chunks."""

    def __init__(self, parts: list[str]) -> None:
        self.parts = parts
        self.usage = None

    def __aiter__(self):
        self._i = 0
        return self

    async def __anext__(self):
        if self._i >= len(self.parts):
            raise StopAsyncIteration
        p = self.parts[self._i]
        self._i += 1
        return _FakeStreamChunk(p)


def _stub_metrics(monkeypatch):
    monkeypatch.setattr("drbrain.metrics.get_metrics", lambda: _StubMetrics())


def test_stream_complete_yields_real_tokens(monkeypatch):
    """Per-token completion streaming: every delta surfaces as a chunk."""
    _stub_metrics(monkeypatch)
    captured = {}

    def fake_completion(**kwargs):
        captured["kwargs"] = kwargs
        return _FakeStream(["chunk ", "one ", "two"])

    monkeypatch.setattr("litellm.completion", fake_completion)
    llm = DrbrainLLM(_cfg(), max_tokens=64)
    chunks = list(llm.stream_complete("q"))
    assert [c.delta for c in chunks] == ["chunk ", "one ", "two"]
    assert [c.text for c in chunks] == ["chunk ", "one ", "two"]
    # the first chunk arrives (generator is lazy: no chunk produced by the
    # previous yield means streaming would appear stalled)
    assert chunks[0].delta  # first chunk non-empty
    assert captured["kwargs"]["stream"] is True
    assert captured["kwargs"]["max_tokens"] == 64
    assert captured["kwargs"]["model"] == "openai/gpt-4o"
    # full reply reconstructs from the deltas
    assert "".join(c.delta for c in chunks) == "chunk one two"


def test_stream_chat_yields_real_tokens(monkeypatch):
    """Per-token chat streaming: delta + message.content both carry the token."""
    from llama_index.core.base.llms.types import ChatMessage, MessageRole

    _stub_metrics(monkeypatch)
    captured = {}

    def fake_completion(**kwargs):
        captured["temperature"] = kwargs["temperature"]
        return _FakeStream(["to", "ken", "s"])

    monkeypatch.setattr("litellm.completion", fake_completion)
    llm = DrbrainLLM(_cfg(), max_tokens=64)
    chunks = list(llm.stream_chat([ChatMessage(role=MessageRole.USER, content="hi")]))
    assert [c.delta for c in chunks] == ["to", "ken", "s"]
    assert [c.message.content for c in chunks] == ["to", "ken", "s"]
    assert chunks[0].delta  # first chunk non-empty
    assert captured["temperature"] == 0.1  # DrbrainLLM default forwarded


async def test_astream_chat_yields_real_tokens(monkeypatch):
    from llama_index.core.base.llms.types import ChatMessage, MessageRole

    _stub_metrics(monkeypatch)
    captured = {}

    async def fake_acompletion(**kwargs):
        captured["stream"] = kwargs.get("stream")
        return _FakeAsyncStream(["a", "b", "c"])

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)
    llm = DrbrainLLM(_cfg(), max_tokens=64)
    chunks = [r async for r in llm.astream_chat([ChatMessage(role=MessageRole.USER, content="q")])]
    assert [c.delta for c in chunks] == ["a", "b", "c"]
    assert chunks[0].delta  # first chunk arrives
    assert captured["stream"] is True


async def test_astream_complete_yields_real_tokens(monkeypatch):
    _stub_metrics(monkeypatch)

    async def fake_acompletion(**kwargs):
        return _FakeAsyncStream(["x", "y"])

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)
    llm = DrbrainLLM(_cfg(), max_tokens=64)
    chunks = [r async for r in llm.astream_complete("q")]
    assert [c.delta for c in chunks] == ["x", "y"]


def test_stream_chat_skips_empty_deltas(monkeypatch):
    """Empty/None deltas (reasoning chunks, finish markers) are not yielded."""
    from llama_index.core.base.llms.types import ChatMessage, MessageRole

    _stub_metrics(monkeypatch)
    monkeypatch.setattr(
        "litellm.completion",
        lambda **kw: _FakeStream(["real", "", None, "text"]),
    )
    llm = DrbrainLLM(_cfg(), max_tokens=64)
    chunks = list(llm.stream_chat([ChatMessage(role=MessageRole.USER, content="q")]))
    assert [c.delta for c in chunks] == ["real", "text"]


def test_stream_chat_falls_back_across_models(monkeypatch):
    """First model's stream raises → second model streams (fallback chain)."""
    from llama_index.core.base.llms.types import ChatMessage, MessageRole

    _stub_metrics(monkeypatch)
    called: list[str] = []

    def fake_completion(**kwargs):
        called.append(kwargs["model"])
        if len(called) == 1:
            raise RuntimeError("stream broke mid-flight")
        return _FakeStream(["fallback", " reply"])

    monkeypatch.setattr("litellm.completion", fake_completion)
    llm = DrbrainLLM(_cfg(), max_tokens=64)
    chunks = list(llm.stream_chat([ChatMessage(role=MessageRole.USER, content="q")]))
    assert called == ["openai/gpt-4o", "openai/gpt-4o-mini"]  # order preserved
    assert "".join(c.delta for c in chunks) == "fallback reply"


def test_stream_complete_all_models_fail_raises(monkeypatch):
    _stub_metrics(monkeypatch)

    def fake_completion(**kwargs):
        raise RuntimeError("all down")

    monkeypatch.setattr("litellm.completion", fake_completion)
    llm = DrbrainLLM(_cfg(), max_tokens=64)
    with pytest.raises(RuntimeError, match="streaming completion failed"):
        list(llm.stream_complete("q"))


def test_stream_chat_cache_hit_replays_cached_reply(monkeypatch, tmp_path):
    """temperature==0: a cached full reply replays as one chunk, no network."""
    from llama_index.core.base.llms.types import ChatMessage, MessageRole

    _stub_metrics(monkeypatch)
    calls = {"n": 0}

    def fake_completion(**kwargs):
        calls["n"] += 1
        return _FakeStream(["cached", " reply"])

    monkeypatch.setattr("litellm.completion", fake_completion)
    llm = DrbrainLLM(_cfg(tmp_path, cache_ttl=86400), temperature=0, max_tokens=64)
    msgs = [ChatMessage(role=MessageRole.USER, content="same q")]

    chunks1 = list(llm.stream_chat(msgs))
    assert "".join(c.delta for c in chunks1) == "cached reply"
    # identical second call: single-chunk replay from ApiCache, zero network
    chunks2 = list(llm.stream_chat(msgs))
    assert "".join(c.delta for c in chunks2) == "cached reply"
    assert calls["n"] == 1
    assert len(chunks2) == 1  # cached replay is one chunk


def test_stream_complete_cache_hit_skips_network(monkeypatch, tmp_path):
    """Text-path streaming caches unconditionally (drbrain text rule)."""
    _stub_metrics(monkeypatch)
    calls = {"n": 0}

    def fake_completion(**kwargs):
        calls["n"] += 1
        return _FakeStream(["same", "text"])

    monkeypatch.setattr("litellm.completion", fake_completion)
    llm = DrbrainLLM(_cfg(tmp_path, cache_ttl=86400), max_tokens=64)
    first = list(llm.stream_complete("same prompt"))
    assert "".join(c.delta for c in first) == "sametext"
    second = list(llm.stream_complete("same prompt"))
    assert "".join(c.delta for c in second) == "sametext"
    assert calls["n"] == 1  # second call served from ApiCache


# ── Settings wiring ──


def test_init_disabled_does_not_touch_settings_llm():
    from llama_index.core import Settings
    from llama_index.core.llms.mock import MockLLM

    marker = MockLLM()  # Settings.llm setter resolves via resolve_llm (must be an LLM)
    Settings.llm = marker
    cfg = Config(llamaindex=LlamaIndexConfig(enabled=False))
    assert init_llamaindex_settings(cfg) is False
    assert Settings.llm is marker


def test_init_enabled_sets_settings_llm():
    from llama_index.core import Settings

    from drbrain.rag.llm import DrbrainEmbedding

    cfg = Config(llamaindex=LlamaIndexConfig(enabled=True))
    assert init_llamaindex_settings(cfg) is True
    assert isinstance(Settings.llm, DrbrainLLM)
    assert isinstance(Settings.embed_model, DrbrainEmbedding)
    assert Settings.llm.temperature == 0.1


def test_init_llm_failure_returns_false(monkeypatch):
    cfg = Config(llamaindex=LlamaIndexConfig(enabled=True))

    def _boom(cfg, **kwargs):
        raise RuntimeError("construct failed")

    monkeypatch.setattr("drbrain.rag.llm.DrbrainLLM", _boom)
    assert init_llamaindex_settings(cfg) is False


# ── integration: one live call with the opencode test key ──


@pytest.mark.integration
async def test_integration_real_call_hits_cache(tmp_path, monkeypatch):
    """Live ask-scenario call; 2nd identical call must hit ApiCache (no network).

    Uses the opencode.ai ``deepseek-v4-flash`` test key from ``test-run/``
    (never hardcoded here). Skipped unless that config file exists.
    """
    test_cfg_path = Path(__file__).resolve().parents[1] / "test-run" / "config.yaml"
    if not test_cfg_path.exists():
        pytest.skip("test-run/config.yaml (opencode test key) not present")
    # Explicit nonexistent local_path: skip the repo-root config.local.yaml
    # overlay so the opencode test key stays models[0].
    cfg = Config.from_yaml(
        str(test_cfg_path), local_path=test_cfg_path.parent / "config.local.yaml"
    )
    assert cfg.llm.models, "test-run config must define llm.models"
    assert "opencode" in (cfg.llm.models[0].get("base_url") or ""), (
        "expected opencode.ai test key as models[0]"
    )
    cfg.dirs.cache = str(tmp_path)

    llm = DrbrainLLM(cfg, max_tokens=64)
    r1 = await llm.acomplete("Reply with exactly one word: pong")
    assert r1.text, "live call returned empty response"

    # The cache lookup lives inside the real fallback function (before any
    # network), so patching litellm's completion proves call 2 never goes out.
    async def no_network(**kwargs):
        raise AssertionError("second call must not touch the network")

    monkeypatch.setattr("litellm.acompletion", no_network)
    r2 = await llm.acomplete("Reply with exactly one word: pong")
    assert r2.text == r1.text
