"""LLM client with YAML-configured fallback chain + token tracking.

Response caching: when a caller passes an ``ApiCache`` instance, the first
model's response is cached keyed by ``sha256(model + system_prompt + prompt)``
and subsequent identical calls short-circuit without hitting the network.
Caching is opt-in via keyword-only ``_cache``; existing callers are unaffected.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from typing import TYPE_CHECKING, Any

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


class KeyRotator:
    """Rotate API keys to spread rate-limit load across accounts.

    Independent, additive component — the existing fallback-chain logic
    (``call_with_fallback`` / ``acall_with_fallback``) is untouched.  Two
    strategies, both distilled from the research-side batch scripts:

    - ``round_robin``: keys are picked in a cycle via ``next(itertools.count())
      % n`` — mirrors ``research/scripts/cg_recipe_extract_api.py``
      ``_next_headers``.  Spreads concurrent batch traffic evenly; single-key
      429 rate limits are the recurring pain point in those batch jobs.
    - ``hash``: ``hash(key_hint) % n`` maps the same entity string to the same
      key deterministically — mirrors ``research/scripts/cg_recipe_conditions_backfill.py``
      ``hash(material) % len(KEYS)``.  Pair with a per-key connection pool so
      one entity always reuses the same pool/connection.

    Note: ``hash()`` on strings is salted per process (``PYTHONHASHSEED``), so
    the ``hash`` strategy guarantees stable mapping within one process, not
    across process restarts.

    Not thread-safe by itself.  ``round_robin`` is effectively safe under the
    CPython GIL (``next`` on ``itertools.count`` is a single C-level call);
    ``hash`` is stateless.  Callers sharing one rotator across threads that
    need stronger guarantees should guard calls with a lock.
    """

    def __init__(self, keys: list[str], strategy: str = "round_robin"):
        """Initialize with the API keys and a rotation strategy.

        Args:
            keys: Non-empty list of API key strings.
            strategy: ``"round_robin"`` (default) or ``"hash"``.

        Raises:
            ValueError: if ``keys`` is empty or ``strategy`` is unknown.
        """
        if not keys:
            raise ValueError("KeyRotator requires at least one key")
        if strategy not in ("round_robin", "hash"):
            raise ValueError(f"unknown strategy {strategy!r} (expected 'round_robin' or 'hash')")
        self._keys = list(keys)
        self._strategy = strategy
        self._counter = itertools.count()

    def next(self, key_hint: str | None = None) -> str:
        """Return the next key to use.

        Args:
            key_hint: Entity string the key will be used for.  Required for the
                ``hash`` strategy (determines the stable mapping); ignored for
                ``round_robin``.

        Raises:
            ValueError: if the ``hash`` strategy is used without a key_hint.
        """
        if self._strategy == "round_robin":
            return self._keys[next(self._counter) % len(self._keys)]
        if key_hint is None:
            raise ValueError("hash strategy requires a non-None key_hint")
        return self._keys[hash(key_hint) % len(self._keys)]


# Module-level rotators keyed by the keys tuple so each model's key list gets a
# stable round-robin cursor across calls within a process.
_API_KEY_ROTATORS: dict[tuple, KeyRotator] = {}


def _resolve_api_key(model_cfg: dict) -> str | None:
    """Return the api_key for one call, rotating when ``api_keys`` (a list) is set.

    ``api_keys`` spreads rate-limit load across accounts (round-robin); a bare
    ``api_key`` string is used as-is. Returns ``None`` when neither is present.
    """
    keys = model_cfg.get("api_keys")
    if keys:
        key_tuple = tuple(str(k) for k in keys if k)
        if not key_tuple:
            return model_cfg.get("api_key")
        rotator = _API_KEY_ROTATORS.get(key_tuple)
        if rotator is None:
            rotator = KeyRotator(list(key_tuple), strategy="round_robin")
            _API_KEY_ROTATORS[key_tuple] = rotator
        return rotator.next()
    return model_cfg.get("api_key")


# Per-agent key counter: one agent (an LLM instance) pins ONE key for its whole
# multi-turn conversation; different agents get different keys via this counter.
_AGENT_KEY_COUNTER = itertools.count()


def resolve_agent_key(model_cfg: dict) -> dict:
    """Resolve a model's ``api_keys`` list to ONE fixed ``api_key`` (per agent).

    A multi-turn agent conversation re-sends the same system prompt every round,
    so rotating keys *per call* would shatter upstream (per-key) cache hits.
    Instead each agent pins one key for its lifetime; different agents pick
    different keys round-robin. Returns a shallow copy so the shared config dict
    is never mutated.
    """
    model_cfg = dict(model_cfg)
    keys = model_cfg.get("api_keys")
    if isinstance(keys, list):
        keys = [str(k) for k in keys if k]
        if keys:
            model_cfg["api_key"] = keys[next(_AGENT_KEY_COUNTER) % len(keys)]
    model_cfg.pop("api_keys", None)
    return model_cfg


class LLMClient:
    """Calls LLM with provider/model from config, supports fallback chain."""

    def __init__(self, models: list[dict]):
        """Initialize with a list of model configs for fallback chaining."""
        self.models = models

    def call(self, prompt: str, system_prompt: str = "", max_tokens: int = 16384) -> dict | None:
        """Synchronous LLM call with automatic fallback. Returns parsed JSON or None."""
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
    api_key = _resolve_api_key(model_cfg)
    if api_key:
        kwargs["api_key"] = api_key
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


def _prompt_hash(prompt: str) -> str:
    """SHA1 of the prompt text — a stable fingerprint without the content."""
    return hashlib.sha1((prompt or "").encode()).hexdigest()[:16]


def _messages_prompt_hash(messages: list[dict]) -> str:
    """SHA1 of the serialized messages — fingerprint without the content."""
    return hashlib.sha1(json.dumps(messages, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _to_int(value: Any) -> int:
    """Coerce a token/duration value to int (0 when missing or non-numeric)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _log_llm_call(
    *,
    model: str,
    provider: str,
    status: str,
    prompt_hash: str,
    n_messages: int,
    tokens_in: int = 0,
    tokens_out: int = 0,
    duration_ms: int = 0,
    error: str = "",
) -> None:
    """Append one LLM call trace line to ``data/logs/llm_calls.jsonl``.

    Best-effort and safe: never raises, never logs the api_key or prompt/messages
    content (only a hash + counts). Logging must never affect the call path.
    """
    try:
        from pathlib import Path

        from drbrain.log import get_session_id

        path = Path("data/logs/llm_calls.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": time.time(),
            "session_id": get_session_id(),
            "model": model,
            "provider": provider,
            "status": status,
            "prompt_hash": prompt_hash,
            "n_messages": _to_int(n_messages),
            "tokens_in": _to_int(tokens_in),
            "tokens_out": _to_int(tokens_out),
            "duration_ms": _to_int(duration_ms),
            "error": str(error)[:200],
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 — logging must never break the call path
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
        if isinstance(cached, dict):
            logger.info(f"[llm] cache hit (key={key})")
            return cached
    for i, model_cfg in enumerate(models):
        name = f"{model_cfg['provider']}/{model_cfg['model']}"
        start = time.monotonic()
        try:
            kwargs = _build_litellm_kwargs(model_cfg, prompt, system_prompt, max_tokens)
            response = litellm.completion(**kwargs)
            _record_llm(model_cfg["model"], model_cfg.get("provider", ""), response, start)
            content = response.choices[0].message.content
            elapsed = int((time.monotonic() - start) * 1000)
            logger.info(f"[llm] success: {name} in {elapsed}ms")
            usage = getattr(response, "usage", None)
            _log_llm_call(
                model=model_cfg["model"],
                provider=model_cfg.get("provider", ""),
                status="success",
                prompt_hash=_prompt_hash(prompt),
                n_messages=1,
                tokens_in=getattr(usage, "prompt_tokens", 0) if usage else 0,
                tokens_out=getattr(usage, "completion_tokens", 0) if usage else 0,
                duration_ms=elapsed,
            )
            parsed = json.loads(content)
            if _cache is not None:
                _cache.set(key, parsed)
            return parsed
        except Exception as e:
            elapsed = int((time.monotonic() - start) * 1000)
            logger.warning(f"[llm] {name} failed (attempt {i + 1}/{len(models)}): {e}")
            _log_llm_call(
                model=model_cfg["model"],
                provider=model_cfg.get("provider", ""),
                status="error",
                prompt_hash=_prompt_hash(prompt),
                n_messages=1,
                duration_ms=elapsed,
                error=str(e)[:200],
            )
            continue
    logger.error(f"[llm] all {len(models)} models exhausted")
    _log_llm_call(
        model="all",
        provider="",
        status="error",
        prompt_hash=_prompt_hash(prompt),
        n_messages=1,
        error="all models exhausted",
    )
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
        if isinstance(cached, dict):
            logger.info(f"[llm] async cache hit (key={key})")
            return cached
    for i, model_cfg in enumerate(models):
        name = f"{model_cfg['provider']}/{model_cfg['model']}"
        start = time.monotonic()
        try:
            kwargs = _build_litellm_kwargs(model_cfg, prompt, system_prompt, max_tokens)
            response = await litellm.acompletion(**kwargs)
            _record_llm(model_cfg["model"], model_cfg.get("provider", ""), response, start)
            content = response.choices[0].message.content
            elapsed = int((time.monotonic() - start) * 1000)
            logger.info(f"[llm] async success: {name} in {elapsed}ms")
            usage = getattr(response, "usage", None)
            _log_llm_call(
                model=model_cfg["model"],
                provider=model_cfg.get("provider", ""),
                status="success",
                prompt_hash=_prompt_hash(prompt),
                n_messages=1,
                tokens_in=getattr(usage, "prompt_tokens", 0) if usage else 0,
                tokens_out=getattr(usage, "completion_tokens", 0) if usage else 0,
                duration_ms=elapsed,
            )
            parsed = json.loads(content)
            if _cache is not None:
                _cache.set(key, parsed)
            return parsed
        except Exception as e:
            elapsed = int((time.monotonic() - start) * 1000)
            logger.warning(f"[llm] async {name} failed (attempt {i + 1}/{len(models)}): {e}")
            _log_llm_call(
                model=model_cfg["model"],
                provider=model_cfg.get("provider", ""),
                status="error",
                prompt_hash=_prompt_hash(prompt),
                n_messages=1,
                duration_ms=elapsed,
                error=str(e)[:200],
            )
            continue
    logger.error(f"[llm] async all {len(models)} models exhausted")
    _log_llm_call(
        model="all",
        provider="",
        status="error",
        prompt_hash=_prompt_hash(prompt),
        n_messages=1,
        error="all models exhausted",
    )
    return None


def call_text_with_fallback(
    prompt: str,
    models: list[dict],
    system_prompt: str = "",
    max_tokens: int = 4096,
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
            api_key = _resolve_api_key(model_cfg)
            if api_key:
                kwargs["api_key"] = api_key
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
            api_key = _resolve_api_key(model_cfg)
            if api_key:
                kwargs["api_key"] = api_key
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
        if isinstance(cached, dict):
            logger.info(f"[llm] call_with_messages cache hit (key={key})")
            return cached

    for i, model_cfg in enumerate(models):
        name = f"{model_cfg['provider']}/{model_cfg['model']}"
        start = time.monotonic()
        try:
            kwargs: dict = {
                "model": name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": timeout,
            }
            api_key = _resolve_api_key(model_cfg)
            if api_key:
                kwargs["api_key"] = api_key
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
            _log_llm_call(
                model=model_cfg["model"],
                provider=model_cfg.get("provider", ""),
                status="success",
                prompt_hash=_messages_prompt_hash(messages),
                n_messages=len(messages),
                tokens_in=usage.prompt_tokens if usage else 0,
                tokens_out=usage.completion_tokens if usage else 0,
                duration_ms=elapsed,
            )
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
            elapsed = int((time.monotonic() - start) * 1000)
            logger.warning(
                f"[llm] call_with_messages {name} failed (attempt {i + 1}/{len(models)}): {e}"
            )
            _log_llm_call(
                model=model_cfg["model"],
                provider=model_cfg.get("provider", ""),
                status="error",
                prompt_hash=_messages_prompt_hash(messages),
                n_messages=len(messages),
                duration_ms=elapsed,
                error=str(e)[:200],
            )
            continue
    logger.error(f"[llm] call_with_messages all {len(models)} models exhausted")
    _log_llm_call(
        model="all",
        provider="",
        status="error",
        prompt_hash=_messages_prompt_hash(messages),
        n_messages=len(messages),
        error="all models exhausted",
    )
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
        if isinstance(cached, dict):
            logger.info(f"[llm] acall_with_messages cache hit (key={key})")
            return cached

    for i, model_cfg in enumerate(models):
        name = f"{model_cfg['provider']}/{model_cfg['model']}"
        start = time.monotonic()
        try:
            kwargs: dict = {
                "model": name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": timeout,
            }
            api_key = _resolve_api_key(model_cfg)
            if api_key:
                kwargs["api_key"] = api_key
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
            _log_llm_call(
                model=model_cfg["model"],
                provider=model_cfg.get("provider", ""),
                status="success",
                prompt_hash=_messages_prompt_hash(messages),
                n_messages=len(messages),
                tokens_in=usage.prompt_tokens if usage else 0,
                tokens_out=usage.completion_tokens if usage else 0,
                duration_ms=elapsed,
            )
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
            elapsed = int((time.monotonic() - start) * 1000)
            logger.warning(
                f"[llm] acall_with_messages {name} failed (attempt {i + 1}/{len(models)}): {e}"
            )
            _log_llm_call(
                model=model_cfg["model"],
                provider=model_cfg.get("provider", ""),
                status="error",
                prompt_hash=_messages_prompt_hash(messages),
                n_messages=len(messages),
                duration_ms=elapsed,
                error=str(e)[:200],
            )
            continue
    logger.error(f"[llm] acall_with_messages all {len(models)} models exhausted")
    _log_llm_call(
        model="all",
        provider="",
        status="error",
        prompt_hash=_messages_prompt_hash(messages),
        n_messages=len(messages),
        error="all models exhausted",
    )
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
