"""LlamaIndex ``Settings``/LLM bridge (T1: Settings init; T2: LLM adapter wired).

Bridges drbrain's own assets into LlamaIndex: the embedding provider stays
drbrain's (lazy sentence-transformers adapter) and the LLM fallback chain stays
drbrain's (``extractor/llm_client.py`` — multi-model fallback + ApiCache +
metrics recording), exposed to LlamaIndex as a custom :class:`DrbrainLLM`.
Everything in this module degrades gracefully when llama-index is not
installed, so the CLI and existing tests never break.

T2 conclusion: ``llama-index-llms-litellm``'s ``LiteLLM`` class takes a single
``model: str`` plus one ``api_key``/``api_base`` — it has no notion of a model
list, no ApiCache, and no drbrain metrics. So the bridge is a custom ``LLM``
subclass that delegates to ``llm_client`` and keeps all three assets intact
(design §4.1 anticipated this: "若不支持多模型 fallback,则自定义 BaseLLM").
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any, ClassVar

from drbrain.config import Config

try:
    from llama_index.core.base.llms.types import (
        ChatMessage,
        ChatResponse,
        CompletionResponse,
        LLMMetadata,
        MessageRole,
    )
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.llms import LLM

    _LLAMA_INDEX_AVAILABLE = True
except ImportError:  # pragma: no cover - envs without llama-index
    BaseEmbedding = None  # type: ignore[assignment,misc]
    ChatMessage = None  # type: ignore[assignment]
    ChatResponse = None  # type: ignore[assignment]
    CompletionResponse = None  # type: ignore[assignment]
    LLMMetadata = None  # type: ignore[assignment]
    MessageRole = None  # type: ignore[assignment]
    LLM = None  # type: ignore[assignment]
    _LLAMA_INDEX_AVAILABLE = False

log = logging.getLogger(__name__)


def _stream_delta(chunk: Any) -> str:
    """Extract the text delta from an OpenAI-compatible stream chunk.

    Accepts litellm stream chunks (``chunk.choices[0].delta.content``) and
    degenerate shapes (``None`` / no choices) — malformed chunks stream as an
    empty delta rather than crashing the consumer.
    """
    if chunk is None:
        return ""
    choices = getattr(chunk, "choices", None)
    if not choices:
        return ""
    delta = getattr(choices[0], "delta", None)
    if delta is None:
        return ""
    return str(getattr(delta, "content", None) or "")


class _DrbrainEmbedMixin:
    """Embed logic delegating to drbrain's own embed provider.

    Lazy by design: the configured sentence-transformer model is only loaded on
    the first embed call, so constructing the adapter never touches the network
    or GPU (keeps CLI startup and offline smoke tests fast).
    """

    def __init__(self, cfg: Config) -> None:
        # Only call AFTER the pydantic ``super().__init__()`` (see
        # DrbrainEmbedding.__init__): pydantic's BaseModel init rebuilds
        # ``__dict__`` from declared fields and would wipe a plain attribute
        # set before it (T3 found ``_cfg`` silently missing). object.__setattr__
        # bypasses pydantic's validation of undeclared attributes.
        object.__setattr__(self, "_cfg", cfg)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        from drbrain.services.embedding import _embed_batch

        return _embed_batch(texts, self._cfg.embed)

    def _embed_one(self, text: str) -> list[float]:
        embs = self._embed([text])
        return embs[0] if embs else []


if _LLAMA_INDEX_AVAILABLE:

    class DrbrainEmbedding(BaseEmbedding, _DrbrainEmbedMixin):
        """LlamaIndex ``BaseEmbedding`` adapter over drbrain's embed provider.

        ``model_name`` mirrors the configured ``embed.model`` so downstream
        LlamaIndex components see the real model identity.
        """

        def __init__(self, cfg: Config, **kwargs: Any) -> None:
            super().__init__(
                model_name=cfg.embed.model,
                embed_batch_size=cfg.embed.batch_size,
                **kwargs,
            )
            # Must run AFTER super().__init__ — pydantic's BaseModel init
            # rebuilds __dict__ from declared fields, wiping _cfg otherwise.
            _DrbrainEmbedMixin.__init__(self, cfg)

        def _get_text_embedding(self, text: str) -> list[float]:
            return self._embed_one(text)

        def _get_query_embedding(self, query: str) -> list[float]:
            return self._embed_one(query)

        async def _aget_query_embedding(self, query: str) -> list[float]:
            return self._embed_one(query)

        def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
            return self._embed(texts)

    class DrbrainLLM(LLM):
        """LlamaIndex ``LLM`` adapter over drbrain's multi-model fallback chain.

        Every completion delegates to :mod:`drbrain.extractor.llm_client` so the
        existing fallback chain (try each ``cfg.llm.models`` entry in order),
        ``ApiCache`` (sha256-keyed, opt-in) and metrics recording
        (``MetricsStore.record_llm``) keep working unchanged.

        ``temperature`` defaults to drbrain's 0.1 (JSON/text default) and is
        forwarded to the messages-based calls (``call_with_messages`` family).
        The prompt-based text helpers hardcode their own temperatures inside
        ``llm_client`` (0 / 0.1); changing that is out of scope here. The four
        streaming endpoints (``stream_chat``/``stream_complete``/``astream_*``)
        stream real per-token deltas through litellm's native streaming,
        keeping the model fallback chain, ApiCache (chat: temperature == 0;
        text: unconditional) and metrics intact (T9: replaces the T2
        single-chunk stubs).
        """

        DEFAULT_CONTEXT_WINDOW: ClassVar[int] = 16384
        DEFAULT_MAX_TOKENS: ClassVar[int] = 4096
        DEFAULT_TEMPERATURE: ClassVar[float] = 0.1

        # Declared as pydantic fields so ``DrbrainLLM(cfg, temperature=0.7)``
        # validates; private (underscore) attrs carry cfg/models/cache.
        temperature: float = DEFAULT_TEMPERATURE
        max_tokens: int = DEFAULT_MAX_TOKENS
        context_window: int = DEFAULT_CONTEXT_WINDOW

        def __init__(
            self,
            cfg: Config,
            temperature: float = DEFAULT_TEMPERATURE,
            max_tokens: int = DEFAULT_MAX_TOKENS,
            **kwargs: Any,
        ) -> None:
            super().__init__(temperature=temperature, max_tokens=max_tokens, **kwargs)
            # Underscore attrs assigned after pydantic init: pydantic's
            # validate_python(self_instance=...) rebuilds field state and can
            # drop pre-set plain attrs, but setting invalid-field names goes
            # through the plain-setattr path (no error).
            self._cfg = cfg
            self._models = list(cfg.llm.models)
            self._cache = None  # ApiCache | None, built lazily on first call

        # ── identity ────────────────────────────────────────────────────

        @property
        def model_name(self) -> str:
            """``provider/model`` of the first entry in the fallback chain."""
            if not self._models:
                return "drbrain/unknown"
            first = self._models[0]
            return f"{first.get('provider', 'openai')}/{first.get('model', 'unknown')}"

        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata(
                context_window=self.context_window,
                num_output=self.max_tokens,
                is_chat_model=True,
                is_function_calling_model=False,
                model_name=self.model_name,
                system_role=MessageRole.SYSTEM,
            )

        # ── cache ───────────────────────────────────────────────────────

        def _get_cache(self):
            """ApiCache built lazily from config (``dirs.cache`` + TTL)."""
            if self._cache is None:
                api = getattr(self._cfg, "api", None)
                ttl = getattr(api, "cache_ttl", 0) or 0
                if ttl > 0:
                    dirs = getattr(self._cfg, "dirs", None)
                    cache_dir = getattr(dirs, "cache", None) or "data/cache"
                    from drbrain.extractor.cache import ApiCache

                    self._cache = ApiCache(cache_dir, ttl=ttl)
            return self._cache

        # ── LlamaIndex protocol ─────────────────────────────────────────

        def complete(
            self, prompt: str, formatted: bool = False, **kwargs: Any
        ) -> CompletionResponse:
            """Sync completion via drbrain's text fallback chain (no cache)."""
            from drbrain.extractor.llm_client import call_text_with_fallback

            max_tokens = kwargs.pop("max_tokens", self.max_tokens)
            text = call_text_with_fallback(prompt, self._models, max_tokens=max_tokens)
            return self._completion(text)

        async def acomplete(
            self, prompt: str, formatted: bool = False, **kwargs: Any
        ) -> CompletionResponse:
            """Async completion; ApiCache hit short-circuits the network."""
            from drbrain.extractor.llm_client import acall_text_with_fallback

            max_tokens = kwargs.pop("max_tokens", self.max_tokens)
            text = await acall_text_with_fallback(
                prompt, self._models, max_tokens=max_tokens, _cache=self._get_cache()
            )
            return self._completion(text)

        def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
            """Sync chat via drbrain's messages fallback chain (no cache)."""
            from drbrain.extractor.llm_client import call_with_messages

            result = call_with_messages(
                self._to_litellm_messages(messages),
                self._models,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return self._chat(result)

        async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
            """Async chat; cache applies when temperature == 0 (drbrain rule)."""
            from drbrain.extractor.llm_client import acall_with_messages

            result = await acall_with_messages(
                self._to_litellm_messages(messages),
                self._models,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                _cache=self._get_cache(),
            )
            return self._chat(result)

        # ── streaming: real per-token via litellm native streaming (T9) ────

        def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any):
            """Stream a chat completion token-by-token through litellm.

            Tries each model in the fallback chain (a mid-stream failure falls
            through to the next model; the first model that produces output
            wins). Each yielded :class:`ChatResponse` carries the delta in
            both ``delta`` and ``message.content`` (consumers of either
            convention accumulate the full reply). Caching follows the drbrain
            chat rule (``temperature == 0``): a cached full reply is replayed
            as a single chunk, and a streamed reply is cached on completion.
            """
            litellm_msgs = self._to_litellm_messages(messages)
            cache, key = self._chat_stream_cache_state(litellm_msgs)
            if key is not None:
                cached = cache.get(key)
                if isinstance(cached, dict):
                    log.info("[llm] stream_chat cache hit (key=%s)", key)
                    yield self._chat(cached)
                    return

            import litellm

            last_exc: Exception | None = None
            for model_cfg in self._models:
                name = f"{model_cfg['provider']}/{model_cfg['model']}"
                try:
                    start = time.monotonic()
                    response = litellm.completion(
                        **self._stream_kwargs(model_cfg, litellm_msgs, temperature=self.temperature)
                    )
                    parts: list[str] = []
                    for chunk in response:
                        delta = _stream_delta(chunk)
                        if not delta:
                            continue
                        parts.append(delta)
                        yield self._stream_chat_chunk(delta, name)
                    self._record_stream(model_cfg, response, start)
                    if key is not None:
                        cache.set(
                            key,
                            {
                                "text": "".join(parts),
                                "tool_calls": None,
                                "usage": {"in": 0, "out": 0},
                            },
                        )
                    return
                except Exception as exc:  # noqa: BLE001 - fallback chain catches all
                    last_exc = exc
                    log.warning(
                        "[llm] stream_chat %s failed (attempt %d/%d): %s",
                        name,
                        self._models.index(model_cfg) + 1,
                        len(self._models),
                        exc,
                    )
                    continue
            raise RuntimeError(
                f"streaming chat failed after {len(self._models)} model(s): {last_exc}"
            )

        def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any):
            """Stream a text completion token-by-token through litellm.

            Cache follows the drbrain text rule (unconditional when a cache is
            configured): a cached reply is replayed as a single chunk, and a
            streamed reply is cached on completion.
            """
            messages = [{"role": "user", "content": prompt}]
            cache, key = self._text_stream_cache_state(prompt)
            if key is not None:
                cached = cache.get(key)
                if cached is not None and isinstance(cached, dict) and "__text__" in cached:
                    log.info("[llm] stream_complete cache hit (key=%s)", key)
                    yield self._completion(cached["__text__"])
                    return

            import litellm

            last_exc: Exception | None = None
            for model_cfg in self._models:
                name = f"{model_cfg['provider']}/{model_cfg['model']}"
                try:
                    start = time.monotonic()
                    response = litellm.completion(
                        **self._stream_kwargs(model_cfg, messages, temperature=0.1)
                    )
                    parts: list[str] = []
                    for chunk in response:
                        delta = _stream_delta(chunk)
                        if not delta:
                            continue
                        parts.append(delta)
                        yield self._stream_complete_chunk(delta, name)
                    self._record_stream(model_cfg, response, start)
                    if key is not None:
                        cache.set(key, {"__text__": "".join(parts)})
                    return
                except Exception as exc:  # noqa: BLE001 - fallback chain catches all
                    last_exc = exc
                    log.warning(
                        "[llm] stream_complete %s failed (attempt %d/%d): %s",
                        name,
                        self._models.index(model_cfg) + 1,
                        len(self._models),
                        exc,
                    )
                    continue
            raise RuntimeError(
                f"streaming completion failed after {len(self._models)} model(s): {last_exc}"
            )

        async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any):
            """Async per-token streaming chat (same semantics as stream_chat)."""
            litellm_msgs = self._to_litellm_messages(messages)
            cache, key = self._chat_stream_cache_state(litellm_msgs)
            if key is not None:
                cached = cache.get(key)
                if isinstance(cached, dict):
                    log.info("[llm] astream_chat cache hit (key=%s)", key)
                    yield self._chat(cached)
                    return

            import litellm

            last_exc: Exception | None = None
            for model_cfg in self._models:
                name = f"{model_cfg['provider']}/{model_cfg['model']}"
                try:
                    start = time.monotonic()
                    response = await litellm.acompletion(
                        **self._stream_kwargs(model_cfg, litellm_msgs, temperature=self.temperature)
                    )
                    parts: list[str] = []
                    async for chunk in response:
                        delta = _stream_delta(chunk)
                        if not delta:
                            continue
                        parts.append(delta)
                        yield self._stream_chat_chunk(delta, name)
                    self._record_stream(model_cfg, response, start)
                    if key is not None:
                        cache.set(
                            key,
                            {
                                "text": "".join(parts),
                                "tool_calls": None,
                                "usage": {"in": 0, "out": 0},
                            },
                        )
                    return
                except Exception as exc:  # noqa: BLE001 - fallback chain catches all
                    last_exc = exc
                    log.warning(
                        "[llm] astream_chat %s failed (attempt %d/%d): %s",
                        name,
                        self._models.index(model_cfg) + 1,
                        len(self._models),
                        exc,
                    )
                    continue
            raise RuntimeError(
                f"async streaming chat failed after {len(self._models)} model(s): {last_exc}"
            )

        async def astream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any):
            """Async per-token streaming completion (same semantics as stream_complete)."""
            messages = [{"role": "user", "content": prompt}]
            cache, key = self._text_stream_cache_state(prompt)
            if key is not None:
                cached = cache.get(key)
                if cached is not None and isinstance(cached, dict) and "__text__" in cached:
                    log.info("[llm] astream_complete cache hit (key=%s)", key)
                    yield self._completion(cached["__text__"])
                    return

            import litellm

            last_exc: Exception | None = None
            for model_cfg in self._models:
                name = f"{model_cfg['provider']}/{model_cfg['model']}"
                try:
                    start = time.monotonic()
                    response = await litellm.acompletion(
                        **self._stream_kwargs(model_cfg, messages, temperature=0.1)
                    )
                    parts: list[str] = []
                    async for chunk in response:
                        delta = _stream_delta(chunk)
                        if not delta:
                            continue
                        parts.append(delta)
                        yield self._stream_complete_chunk(delta, name)
                    self._record_stream(model_cfg, response, start)
                    if key is not None:
                        cache.set(key, {"__text__": "".join(parts)})
                    return
                except Exception as exc:  # noqa: BLE001 - fallback chain catches all
                    last_exc = exc
                    log.warning(
                        "[llm] astream_complete %s failed (attempt %d/%d): %s",
                        name,
                        self._models.index(model_cfg) + 1,
                        len(self._models),
                        exc,
                    )
                    continue
            raise RuntimeError(
                f"async streaming completion failed after {len(self._models)} model(s): {last_exc}"
            )

        # ── streaming helpers ────────────────────────────────────────────

        def _stream_kwargs(self, model_cfg: dict, messages: list[dict], temperature: float) -> dict:
            """litellm streaming kwargs for one model (fallback chain entry)."""
            kwargs: dict[str, Any] = {
                "model": f"{model_cfg['provider']}/{model_cfg['model']}",
                "messages": messages,
                "temperature": temperature,
                "max_tokens": self.max_tokens,
                "timeout": 60,
                "stream": True,
            }
            if model_cfg.get("api_key"):
                kwargs["api_key"] = model_cfg["api_key"]
            if model_cfg.get("base_url"):
                kwargs["api_base"] = model_cfg["base_url"]
            return kwargs

        def _chat_stream_cache_state(self, litellm_msgs: list[dict]):
            """(cache, key) for stream_chat; cache applies only at temperature 0."""
            cache = self._get_cache()
            if cache is None or not self._models or self.temperature != 0:
                return None, None
            from drbrain.extractor.llm_client import _messages_cache_key

            key = _messages_cache_key(self._models, litellm_msgs, self.max_tokens, self.temperature)
            return cache, key

        def _text_stream_cache_state(self, prompt: str):
            """(cache, key) for stream_complete; text path caches unconditionally."""
            cache = self._get_cache()
            if cache is None or not self._models:
                return None, None
            from drbrain.extractor.llm_client import _cache_key

            key = _cache_key(
                f"{self._models[0]['provider']}/{self._models[0]['model']}",
                "",
                prompt,
                self.max_tokens,
            )
            return cache, key

        @staticmethod
        def _stream_chat_chunk(delta: str, model_name: str) -> ChatResponse:
            """One ChatResponse carrying the token delta (both fields)."""
            return ChatResponse(
                message=ChatMessage(role=MessageRole.ASSISTANT, content=delta),
                delta=delta,
                raw=delta,
                additional_kwargs={"model": model_name},
            )

        @staticmethod
        def _stream_complete_chunk(delta: str, model_name: str) -> CompletionResponse:
            """One CompletionResponse carrying the token delta."""
            return CompletionResponse(
                text=delta,
                delta=delta,
                raw=delta,
                additional_kwargs={"model": model_name},
            )

        @staticmethod
        def _record_stream(model_cfg: dict, response: Any, start: float) -> None:
            """Best-effort metrics for a consumed stream (usage may be absent)."""
            try:
                from drbrain.extractor.llm_client import _record_llm

                _record_llm(model_cfg["model"], model_cfg.get("provider", ""), response, start)
            except Exception:  # pragma: no cover - metrics are best-effort
                pass

        # ── helpers ─────────────────────────────────────────────────────

        @staticmethod
        def _to_litellm_messages(messages: Sequence[ChatMessage]) -> list[dict[str, str]]:
            """Convert LlamaIndex ``ChatMessage``s to litellm dicts."""
            out = []
            for msg in messages:
                role = getattr(msg.role, "value", str(msg.role))
                out.append({"role": role, "content": msg.content or ""})
            return out

        def _completion(self, text: str | None) -> CompletionResponse:
            text = text or ""
            return CompletionResponse(
                text=text,
                delta=text,
                raw=text,
                additional_kwargs={"model": self.model_name},
            )

        def _chat(self, result: dict | None) -> ChatResponse:
            text = (result or {}).get("text", "") or ""
            tool_calls = (result or {}).get("tool_calls")
            message_kwargs = {"tool_calls": tool_calls} if tool_calls else {}
            return ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=text,
                    additional_kwargs=message_kwargs,
                ),
                delta=text,
                raw=result,
                additional_kwargs={"model": self.model_name},
            )


else:  # pragma: no cover - exercised in environments without llama-index

    class DrbrainEmbedding(_DrbrainEmbedMixin):
        """Duck-typed embed adapter used when llama-index is not installed."""

        def __init__(self, cfg: Config) -> None:
            super().__init__(cfg)
            self.model_name = cfg.embed.model
            self.embed_batch_size = cfg.embed.batch_size

        def get_text_embedding(self, text: str) -> list[float]:
            return self._embed_one(text)

        async def aget_text_embedding(self, text: str) -> list[float]:
            return self._embed_one(text)

        def get_query_embedding(self, query: str) -> list[float]:
            return self._embed_one(query)

        async def aget_query_embedding(self, query: str) -> list[float]:
            return self._embed_one(query)

        def get_text_embedding_batch(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
            return self._embed(texts)


def init_llamaindex_settings(cfg: Config) -> bool:
    """Initialize LlamaIndex ``Settings`` from config; return availability.

    Returns ``True`` only when ``llamaindex.enabled`` is set AND llama_index is
    importable AND both the embed adapter and the :class:`DrbrainLLM` bridge
    could be installed. Never raises: on any failure (or ``enabled=false``) it
    returns ``False`` so callers fall back to legacy implementations.

    Offline-safe: the embed adapter is lazy (no model download at init) and
    ``DrbrainLLM`` only reads config (no network, no cache dir creation until
    the first call).
    """
    from drbrain.rag.config import get_llamaindex_config

    if not get_llamaindex_config(cfg).enabled:
        return False
    settings = _import_settings()
    if settings is None:
        return False
    try:
        settings.embed_model = DrbrainEmbedding(cfg)
        settings.llm = DrbrainLLM(cfg)
    except Exception:
        return False
    return True


def _import_settings():
    """Import ``llama_index.core.Settings``; return ``None`` when missing."""
    try:
        from llama_index.core import Settings
    except ImportError:
        return None
    return Settings
