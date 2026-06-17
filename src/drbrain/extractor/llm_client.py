"""LLM client with YAML-configured fallback chain + token tracking.

Response caching: when a caller passes an ``ApiCache`` instance, the first
model's response is cached keyed by ``sha256(model + system_prompt + prompt)``
and subsequent identical calls short-circuit without hitting the network.
Caching is opt-in via keyword-only ``_cache``; existing callers are unaffected.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import TYPE_CHECKING

import litellm
from loguru import logger

if TYPE_CHECKING:
    from drbrain.extractor.cache import ApiCache


def _cache_key(model_name: str, system_prompt: str, prompt: str, max_tokens: int) -> str:
    """Stable hash key for an LLM call (model + prompts + max_tokens).

    Returns the first 16 hex chars of sha256 — collision probability is
    negligible for practical prompt spaces, and keeps filenames short.
    """
    raw = f"{model_name}\x00{system_prompt}\x00{prompt}\x00{max_tokens}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _messages_cache_key(
    models: list[dict], messages: list[dict], max_tokens: int, temperature: float
) -> str:
    """Stable hash key for call_with_messages / acall_with_messages.

    Returns the first 16 hex chars of sha256 over model name, messages,
    max_tokens, and temperature.
    """
    raw = json.dumps(
        {
            "model": f"{models[0]['provider']}/{models[0]['model']}",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class LLMClient:
    """Calls LLM with provider/model from config, supports fallback chain."""

    def __init__(self, models: list[dict]):
        self.models = models

    def call(self, prompt: str, system_prompt: str = "", max_tokens: int = 16384) -> dict | None:
        return call_with_fallback(prompt, self.models, system_prompt, max_tokens)


def _build_litellm_kwargs(
    model_cfg: dict, prompt: str, system_prompt: str, max_tokens: int
) -> dict:
    name = f"{model_cfg['provider']}/{model_cfg['model']}"
    messages = []
    # Anthropic prompt caching: mark long system prompts as ephemeral cache
    # points. Anthropic bills cached input tokens at ~10% of normal rate,
    # so reusing a shared system prompt across many calls is a big saving.
    # Threshold ~4000 chars ≈ 1000 tokens (Anthropic's minimum cacheable block).
    provider = model_cfg.get("provider", "")
    is_anthropic = provider in ("anthropic", "claude") or "claude" in model_cfg.get("model", "")
    if system_prompt and is_anthropic and len(system_prompt) >= 4000:
        messages.append(
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
                ],
            }
        )
    elif system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    kwargs = {
        "model": name,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "timeout": 60,
    }
    if model_cfg.get("api_key"):
        kwargs["api_key"] = model_cfg["api_key"]
    if model_cfg.get("base_url"):
        kwargs["api_base"] = model_cfg["base_url"]
    return kwargs


def _record_llm(model_name: str, provider: str, response, start_time: float) -> None:
    """Record LLM usage to metrics store."""
    try:
        from drbrain.metrics import get_metrics

        duration_ms = int((time.monotonic() - start_time) * 1000)
        usage = getattr(response, "usage", None)
        tokens_in = getattr(usage, "prompt_tokens", 0) if usage else 0
        tokens_out = getattr(usage, "completion_tokens", 0) if usage else 0
        get_metrics().record_llm(
            model=model_name,
            provider=provider,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            duration_ms=duration_ms,
        )
    except Exception:
        pass


def call_with_fallback(
    prompt: str,
    models: list[dict],
    system_prompt: str = "",
    max_tokens: int = 16384,
    *,
    _cache: ApiCache | None = None,
) -> dict | None:
    """Try models in order, return first successful parsed JSON response.

    When ``_cache`` is provided, the first model's successful response is
    cached and reused on identical subsequent calls.
    """
    logger.info("[llm] call starting — %d models in chain", len(models))
    if _cache is not None and models:
        key = _cache_key(
            f"{models[0]['provider']}/{models[0]['model']}", system_prompt, prompt, max_tokens
        )
        cached = _cache.get(key)
        if cached is not None:
            logger.info(f"[llm] cache hit (key={key})")
            return cached
    for i, model_cfg in enumerate(models):
        name = f"{model_cfg['provider']}/{model_cfg['model']}"
        try:
            start = time.monotonic()
            kwargs = _build_litellm_kwargs(model_cfg, prompt, system_prompt, max_tokens)
            response = litellm.completion(**kwargs)
            _record_llm(model_cfg["model"], model_cfg.get("provider", ""), response, start)
            content = response.choices[0].message.content
            elapsed = int((time.monotonic() - start) * 1000)
            logger.info(f"[llm] success: {name} in {elapsed}ms")
            parsed = json.loads(content)
            if _cache is not None:
                _cache.set(key, parsed)
            return parsed
        except Exception as e:
            logger.warning(f"[llm] {name} failed (attempt {i + 1}/{len(models)}): {e}")
            continue
    logger.error(f"[llm] all {len(models)} models exhausted")
    return None


async def acall_with_fallback(
    prompt: str,
    models: list[dict],
    system_prompt: str = "",
    max_tokens: int = 16384,
    *,
    _cache: ApiCache | None = None,
) -> dict | list | None:
    """Async version of call_with_fallback.

    When ``_cache`` is provided, the first model's successful response is
    cached and reused on identical subsequent calls.
    """
    if _cache is not None and models:
        key = _cache_key(
            f"{models[0]['provider']}/{models[0]['model']}", system_prompt, prompt, max_tokens
        )
        cached = _cache.get(key)
        if cached is not None:
            logger.info(f"[llm] async cache hit (key={key})")
            return cached
    for i, model_cfg in enumerate(models):
        name = f"{model_cfg['provider']}/{model_cfg['model']}"
        try:
            start = time.monotonic()
            kwargs = _build_litellm_kwargs(model_cfg, prompt, system_prompt, max_tokens)
            response = await litellm.acompletion(**kwargs)
            _record_llm(model_cfg["model"], model_cfg.get("provider", ""), response, start)
            content = response.choices[0].message.content
            elapsed = int((time.monotonic() - start) * 1000)
            logger.info(f"[llm] async success: {name} in {elapsed}ms")
            parsed = json.loads(content)
            if _cache is not None:
                _cache.set(key, parsed)
            return parsed
        except Exception as e:
            logger.warning(f"[llm] async {name} failed (attempt {i + 1}/{len(models)}): {e}")
            continue
    logger.error(f"[llm] async all {len(models)} models exhausted")
    return None


def call_text_with_fallback(
    prompt: str,
    models: list[dict],
    system_prompt: str = "",
    max_tokens: int = 2048,
) -> str | None:
    """Sync text call with fallback. Returns raw text (not JSON)."""
    import litellm

    for i, model_cfg in enumerate(models):
        name = f"{model_cfg['provider']}/{model_cfg['model']}"
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            kwargs = {
                "model": name,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": max_tokens,
                "timeout": 60,
                "extra_body": {"thinking": {"type": "disabled"}},
            }
            if model_cfg.get("api_key"):
                kwargs["api_key"] = model_cfg["api_key"]
            if model_cfg.get("base_url"):
                kwargs["api_base"] = model_cfg["base_url"]
            response = litellm.completion(**kwargs)
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Text model {name} failed (attempt {i + 1}/{len(models)}): {e}")
            continue
    logger.error(f"All {len(models)} models failed for text call")
    return None


async def acall_text_with_fallback(
    prompt: str,
    models: list[dict],
    system_prompt: str = "",
    max_tokens: int = 1024,
    *,
    _cache: ApiCache | None = None,
) -> str | None:
    """Async text call with fallback. Returns raw text (not JSON).

    When ``_cache`` is provided, the response is cached (wrapped in a dict
    to satisfy ApiCache's JSON-serializable contract) and reused on hit.
    """
    if _cache is not None and models:
        key = _cache_key(
            f"{models[0]['provider']}/{models[0]['model']}", system_prompt, prompt, max_tokens
        )
        cached = _cache.get(key)
        if cached is not None and isinstance(cached, dict) and "__text__" in cached:
            logger.info(f"[llm] text cache hit (key={key})")
            return cached["__text__"]
    for i, model_cfg in enumerate(models):
        name = f"{model_cfg['provider']}/{model_cfg['model']}"
        try:
            start = time.monotonic()
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            kwargs = {
                "model": name,
                "messages": messages,
                "temperature": 0,
                "max_tokens": max_tokens,
                "timeout": 60,
            }
            if model_cfg.get("api_key"):
                kwargs["api_key"] = model_cfg["api_key"]
            if model_cfg.get("base_url"):
                kwargs["api_base"] = model_cfg["base_url"]
            response = await litellm.acompletion(**kwargs)
            _record_llm(model_cfg["model"], model_cfg.get("provider", ""), response, start)
            logger.debug(
                f"LLM text call success: {name} in {int((time.monotonic() - start) * 1000)}ms"
            )
            text = response.choices[0].message.content.strip()
            if _cache is not None:
                _cache.set(key, {"__text__": text})
            return text
        except Exception as e:
            logger.warning(f"Model {name} failed (attempt {i + 1}/{len(models)}): {e}")
            continue
    logger.error(f"All {len(models)} models failed")
    return None


def call_with_messages(
    messages: list[dict],
    models: list[dict],
    tools: list[dict] | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.3,
    timeout: int = 60,
    *,
    _cache: ApiCache | None = None,
) -> dict | None:
    """Call LLM with pre-built messages list (supports multi-turn conversation).

    Unlike call_with_fallback which builds [system, user] from scratch,
    this accepts an arbitrary messages list that may contain previous
    assistant/tool messages for multi-turn tool-calling loops.

    Returns:
        {"text": str, "tool_calls": list | None, "usage": {"in": int, "out": int}}
        or None if all models fail.
    """
    logger.info("[llm] call_with_messages — %d models, %d messages", len(models), len(messages))

    # Cache lookup — only for deterministic responses (temperature == 0)
    key: str | None = None
    if _cache is not None and models and temperature == 0:
        key = _messages_cache_key(models, messages, max_tokens, temperature)
        cached = _cache.get(key)
        if cached is not None:
            logger.info(f"[llm] call_with_messages cache hit (key={key})")
            return cached

    for i, model_cfg in enumerate(models):
        name = f"{model_cfg['provider']}/{model_cfg['model']}"
        try:
            start = time.monotonic()
            kwargs: dict = {
                "model": name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": timeout,
            }
            if model_cfg.get("api_key"):
                kwargs["api_key"] = model_cfg["api_key"]
            if model_cfg.get("base_url"):
                kwargs["api_base"] = model_cfg["base_url"]
            if tools:
                kwargs["tools"] = tools
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

            response = litellm.completion(**kwargs)
            _record_llm(model_cfg["model"], model_cfg.get("provider", ""), response, start)

            msg = response.choices[0].message
            elapsed = int((time.monotonic() - start) * 1000)
            logger.info(f"[llm] call_with_messages success: {name} in {elapsed}ms")

            usage = response.usage
            result = {
                "text": msg.content or "",
                "tool_calls": _extract_tool_calls(msg),
                "usage": {
                    "in": usage.prompt_tokens if usage else 0,
                    "out": usage.completion_tokens if usage else 0,
                },
            }
            if _cache is not None and key is not None:
                _cache.set(key, result)
            return result
        except Exception as e:
            logger.warning(
                f"[llm] call_with_messages {name} failed (attempt {i + 1}/{len(models)}): {e}"
            )
            continue
    logger.error(f"[llm] call_with_messages all {len(models)} models exhausted")
    return None


async def acall_with_messages(
    messages: list[dict],
    models: list[dict],
    tools: list[dict] | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.3,
    timeout: int = 60,
    *,
    _cache: ApiCache | None = None,
) -> dict | None:
    """Async version of call_with_messages."""
    logger.info("[llm] acall_with_messages — %d models, %d messages", len(models), len(messages))

    # Cache lookup
    key: str | None = None
    if _cache is not None and models and temperature == 0:
        key = _messages_cache_key(models, messages, max_tokens, temperature)
        cached = _cache.get(key)
        if cached is not None:
            logger.info(f"[llm] acall_with_messages cache hit (key={key})")
            return cached

    for i, model_cfg in enumerate(models):
        name = f"{model_cfg['provider']}/{model_cfg['model']}"
        try:
            start = time.monotonic()
            kwargs: dict = {
                "model": name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": timeout,
            }
            if model_cfg.get("api_key"):
                kwargs["api_key"] = model_cfg["api_key"]
            if model_cfg.get("base_url"):
                kwargs["api_base"] = model_cfg["base_url"]
            if tools:
                kwargs["tools"] = tools
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

            response = await litellm.acompletion(**kwargs)
            _record_llm(model_cfg["model"], model_cfg.get("provider", ""), response, start)

            msg = response.choices[0].message
            elapsed = int((time.monotonic() - start) * 1000)
            logger.info(f"[llm] acall_with_messages success: {name} in {elapsed}ms")

            usage = response.usage
            result = {
                "text": msg.content or "",
                "tool_calls": _extract_tool_calls(msg),
                "usage": {
                    "in": usage.prompt_tokens if usage else 0,
                    "out": usage.completion_tokens if usage else 0,
                },
            }
            if _cache is not None and key is not None:
                _cache.set(key, result)
            return result
        except Exception as e:
            logger.warning(
                f"[llm] acall_with_messages {name} failed (attempt {i + 1}/{len(models)}): {e}"
            )
            continue
    logger.error(f"[llm] acall_with_messages all {len(models)} models exhausted")
    return None


def _extract_tool_calls(msg) -> list[dict] | None:
    """Extract tool calls from a litellm message into a serializable list."""
    raw = getattr(msg, "tool_calls", None)
    if not raw:
        return None
    result = []
    for tc in raw:
        item = {
            "id": getattr(tc, "id", ""),
            "type": "function",
            "function": {
                "name": getattr(tc.function, "name", ""),
                "arguments": getattr(tc.function, "arguments", ""),
            },
        }
        result.append(item)
    return result or None
