"""LLM client with YAML-configured fallback chain + token tracking.

Response caching: when a caller passes an ``ApiCache`` instance, the first
model's response is cached keyed by ``sha256(model + system_prompt + prompt)``
and subsequent identical calls short-circuit without hitting the network.
Caching is opt-in via keyword-only ``_cache``; existing callers are unaffected.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import threading
import time
from enum import Enum
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import litellm
from loguru import logger
from openai import AsyncOpenAI
from openai import OpenAI as SyncOpenAI

if TYPE_CHECKING:
    from drbrain.extractor.cache import ApiCache

# OpenAI SDK 客户端缓存(base_url+api_key 复用连接,序列化稳定使 provider 前缀缓存可命中)
_openai_clients: dict[str, SyncOpenAI] = {}
_aopenai_clients: dict[str, AsyncOpenAI] = {}


def _openai_client(api_key: str, base_url: str) -> SyncOpenAI:
    ck = f"{base_url}|{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"
    if ck not in _openai_clients:
        _openai_clients[ck] = SyncOpenAI(api_key=api_key, base_url=base_url)
    return _openai_clients[ck]


def _aopenai_client(api_key: str, base_url: str) -> AsyncOpenAI:
    ck = f"{base_url}|{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"
    if ck not in _aopenai_clients:
        _aopenai_clients[ck] = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return _aopenai_clients[ck]


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

    Keys in cooldown (per-key rate-limit state machine) are skipped: the
    rotator advances until it finds a key not in cooldown, so one exhausted key
    in a pool doesn't block the healthy ones. Returns ``None`` when every key
    in the pool is in cooldown (caller should skip the model).
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
        for _ in range(len(key_tuple)):
            candidate = rotator.next()
            if _RATE_LIMIT_SM.is_key_available(model_cfg, candidate):
                return candidate
        # every key in the pool is in cooldown → model effectively unavailable
        return None
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
            # Pin from the keys NOT in cooldown: a plain round-robin counter
            # assigns ~N_dead/N keys of the agents to rate-limited keys for
            # their whole lifetime. Fall back to plain round-robin only when
            # every key is cooling down (state recovers on its own).
            available = [k for k in keys if _RATE_LIMIT_SM.is_key_available(model_cfg, k)]
            pool = available or keys
            model_cfg["api_key"] = pool[next(_AGENT_KEY_COUNTER) % len(pool)]
    model_cfg.pop("api_keys", None)
    return model_cfg


# ── Per-model RPM throttle ─────────────────────────────────────────────────────
# config 里 models[i].rpm = 每分钟调用上限（如 runinfra qwen3-8-27b: 60 → 取 30）。
# 无 rpm 字段 → 不限流。sync 路径用锁 + time.sleep；async 路径单独维护时间戳
# 避免跨线程竞态。所有 fallback 入口共享同一模块级状态，保证整体节奏 ≤ rpm。

_rpm_lock = threading.Lock()
_rpm_last_call: dict[str, float] = {}
_rpm_async_last_call: dict[str, float] = {}


def _rpm_interval_secs(model_cfg: dict) -> float:
    rpm = model_cfg.get("rpm")
    if not rpm:
        return 0.0
    try:
        rpm = max(1, int(rpm))
    except (TypeError, ValueError):
        return 0.0
    return 60.0 / rpm


def _throttle(model_cfg: dict) -> None:
    """Sync throttle: sleep so calls to this model stay within its rpm."""
    interval = _rpm_interval_secs(model_cfg)
    if interval <= 0:
        return
    name = f"{model_cfg.get('provider', '')}/{model_cfg.get('model', '')}"
    with _rpm_lock:
        last = _rpm_last_call.get(name, 0.0)
        now = time.monotonic()
        wait = interval - (now - last)
        if wait > 0:
            _rpm_last_call[name] = now + wait  # 预订下一个时间槽
            time.sleep(wait)
        else:
            _rpm_last_call[name] = now


async def _athrottle(model_cfg: dict) -> None:
    """Async throttle — same pacing as ``_throttle``, no blocking of the loop."""
    interval = _rpm_interval_secs(model_cfg)
    if interval <= 0:
        return
    name = f"{model_cfg.get('provider', '')}/{model_cfg.get('model', '')}"
    last = _rpm_async_last_call.get(name, 0.0)
    now = time.monotonic()
    wait = interval - (now - last)
    if wait > 0:
        _rpm_async_last_call[name] = now + wait
        await asyncio.sleep(wait)
    else:
        _rpm_async_last_call[name] = now


# ── Per-model concurrency cap ─────────────────────────────────────────────────
# 部分 API 有并发上限（实测 runinfra qwen3-8-27b: 7 concurrent hosted requests）。
# models[i].max_concurrent 可配置；默认 4 留余量。sync 用 threading.Semaphore；
# async 用 per-event-loop asyncio.Semaphore（Semaphore 不能跨 loop 复用）。

_sync_sems: dict[str, threading.Semaphore] = {}
_sync_sems_lock = threading.Lock()
_async_sems: dict[tuple[str, int], asyncio.Semaphore] = {}
_async_sems_lock = threading.Lock()


def _max_concurrent(model_cfg: dict) -> int:
    mc = model_cfg.get("max_concurrent")
    try:
        return max(1, int(mc)) if mc else 4
    except (TypeError, ValueError):
        return 4


def _sync_sem(model_cfg: dict) -> threading.Semaphore:
    name = f"{model_cfg.get('provider', '')}/{model_cfg.get('model', '')}"
    sem = _sync_sems.get(name)
    if sem is None:
        with _sync_sems_lock:
            sem = _sync_sems.setdefault(name, threading.Semaphore(_max_concurrent(model_cfg)))
    return sem


def _async_sem(model_cfg: dict) -> asyncio.Semaphore:
    name = f"{model_cfg.get('provider', '')}/{model_cfg.get('model', '')}"
    loop_id = id(asyncio.get_running_loop())
    key = (name, loop_id)
    sem = _async_sems.get(key)
    if sem is None:
        with _async_sems_lock:
            sem = _async_sems.setdefault(key, asyncio.Semaphore(_max_concurrent(model_cfg)))
    return sem


# ── Retry on transient errors ─────────────────────────────────────────────────
# 超时/限流/连接抖动（runinfra 实测：182s 超时、Connection error、RateLimit）
# 应重试当前模型而不是立刻 fallback 到无关模型。models[i].retries 可配置
# （默认 2 = 最多 3 次尝试）；仅对可重试错误重试，其余照旧 fallback。


def _max_attempts(model_cfg: dict) -> int:
    retries = model_cfg.get("retries")
    try:
        return max(1, int(retries) + 1) if retries else 3
    except (TypeError, ValueError):
        return 3


def _is_retryable(e: Exception) -> bool:
    msg = str(e).lower()
    return any(
        k in msg
        for k in (
            "ratelimit",
            "rate limit",
            "timed out",
            "timeout",
            "connection",
            "internal server",
            "bad gateway",
            "overloaded",
            "try again",
            "temporarily unavailable",
            "502",
            "503",
            "504",
            "429",
        )
    )


def _is_rate_limit(e: Exception) -> bool:
    """True when the error is a hard quota/rate-limit (not a transient blip).

    These warrant a cooldown skip rather than a short retry: the model is
    exhausted for a window, and hammering it with 2s/4s backoff just burns
    time and re-triggers the same error (observed with free-tier models).
    """
    msg = str(e).lower()
    return any(
        k in msg
        for k in (
            "ratelimit",
            "rate limit",
            "429",
            "capacity is limited",
            "too many requests",
            "quota",
            "out of credits",
            "insufficient",
            "free model capacity",
            "recharging",
            "usage limit",
            "monthly usage",
        )
    )


# ── Rate-limit state machine ──────────────────────────────────────────────────
# 自动路由：当模型命中限流/配额（_is_rate_limit）时，进入 COOLDOWN 状态并跳过该
# 模型一段可配置时间（models[i].cooldown_secs，默认 60s），而不是用短退避反复锤打
# 一个仍在限流的模型。连续限流 → 冷却时间指数翻倍（封顶 max_cooldown_secs，默认
# 600s）；一次成功调用 → 复位 NORMAL。状态机模块级共享，跨进程内所有调用一致。
#
# 路由行为：处于 COOLDOWN 的模型在冷却期内被跳过（不调用），fallback 链继续走
# 下一个可用模型；冷却到期后自动回到 NORMAL 重新参与路由。


class _RateLimitState(Enum):
    NORMAL = "normal"
    COOLDOWN = "cooldown"


class RateLimitStateMachine:
    """Per-(model, key) cooldown state machine for automatic rate-limit routing.

    Tracks cooldown per API key (not just per model): when one key in a
    round-robin pool hits a quota/rate limit, only that key is skipped while
    the other keys keep serving. This matters for monthly-usage quotas that are
    per-key (e.g. opencode mimo-v2.5: 7/8 keys fine, 1 key exhausted) — putting
    the whole model in cooldown would wrongly block the healthy keys.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # (model_name, key) -> (state, cooldown_until_monotonic, consecutive_rate_limits)
        self._states: dict[tuple[str, str], tuple[_RateLimitState, float, int]] = {}

    @staticmethod
    def _name(model_cfg: dict) -> str:
        return f"{model_cfg.get('provider', '')}/{model_cfg.get('model', '')}"

    @staticmethod
    def _cooldown_secs(model_cfg: dict) -> float:
        try:
            return max(0.0, float(model_cfg.get("cooldown_secs", 60)))
        except (TypeError, ValueError):
            return 60.0

    @staticmethod
    def _max_cooldown_secs(model_cfg: dict) -> float:
        try:
            return max(0.0, float(model_cfg.get("max_cooldown_secs", 600)))
        except (TypeError, ValueError):
            return 600.0

    def _state(self, model_cfg: dict, api_key: str) -> tuple[_RateLimitState, float, int]:
        return self._states.get((self._name(model_cfg), api_key), (_RateLimitState.NORMAL, 0.0, 0))

    def is_key_available(self, model_cfg: dict, api_key: str) -> bool:
        """True if this specific key is not in cooldown (or cooldown expired)."""
        with self._lock:
            state, until, _ = self._state(model_cfg, api_key)
            if state is _RateLimitState.NORMAL:
                return True
            if time.monotonic() >= until:
                self._states[(self._name(model_cfg), api_key)] = (
                    _RateLimitState.NORMAL,
                    0.0,
                    0,
                )
                return True
            return False

    def remaining(self, model_cfg: dict, api_key: str) -> float:
        """Seconds left in cooldown for this key (0 when not in cooldown)."""
        with self._lock:
            state, until, _ = self._state(model_cfg, api_key)
            if state is _RateLimitState.COOLDOWN:
                return max(0.0, until - time.monotonic())
            return 0.0

    def on_rate_limit(self, model_cfg: dict, api_key: str) -> float:
        """Record a rate limit for this key; enter COOLDOWN and return wait secs."""
        name = self._name(model_cfg)
        base = self._cooldown_secs(model_cfg)
        cap = self._max_cooldown_secs(model_cfg)
        with self._lock:
            _, _, consecutive = self._state(model_cfg, api_key)
            consecutive += 1
            wait = min(base * (2 ** (consecutive - 1)), cap)
            self._states[(name, api_key)] = (
                _RateLimitState.COOLDOWN,
                time.monotonic() + wait,
                consecutive,
            )
            return wait

    def on_success(self, model_cfg: dict, api_key: str) -> None:
        """A successful call resets this key back to NORMAL."""
        with self._lock:
            self._states[(self._name(model_cfg), api_key)] = (
                _RateLimitState.NORMAL,
                0.0,
                0,
            )


_RATE_LIMIT_SM = RateLimitStateMachine()


class LLMClient:
    """Calls LLM with provider/model from config, supports fallback chain."""

    def __init__(self, models: list[dict]):
        """Initialize with a list of model configs for fallback chaining."""
        self.models = models

    def call(self, prompt: str, system_prompt: str = "", max_tokens: int = 16384) -> dict | None:
        """Synchronous LLM call with automatic fallback. Returns parsed JSON or None."""
        return call_with_fallback(prompt, self.models, system_prompt, max_tokens)


def _thinking_extra_body(model_cfg: dict) -> dict:
    """thinking 禁用参数。ox-alpha-free 等模型强制 thinking 不可禁用
    （报错 [1210] This model always engages in thinking），配置 disable_thinking: false
    时返回空 dict 不传 thinking 参数。"""
    if model_cfg.get("disable_thinking") is False:
        return {}
    return {"thinking": {"type": "disabled"}}


def _build_litellm_kwargs(
    model_cfg: dict,
    prompt: str,
    system_prompt: str,
    max_tokens: int,
    *,
    disable_thinking: bool = False,
    api_key: str | None = None,
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
        # 长全文抽取（8k chars prompt）27b 可能几十秒 —— timeout 可 per-model 覆盖。
        "timeout": model_cfg.get("timeout", 60),
    }
    # A per-model body is authoritative. Otherwise the lean concept extractor
    # can request the Qwen/Zhipu switch without changing the fallback client's
    # existing provider-specific thinking policy.
    extra_body = model_cfg.get("extra_body")
    if extra_body is not None:
        kwargs["extra_body"] = extra_body
    elif disable_thinking and model_cfg.get("thinking_param") == "enable_thinking":
        kwargs["extra_body"] = {"enable_thinking": False}
    else:
        kwargs["extra_body"] = _thinking_extra_body(model_cfg)
    if api_key is None:
        api_key = _resolve_api_key(model_cfg)
    if api_key:
        kwargs["api_key"] = api_key
    if model_cfg.get("base_url"):
        kwargs["api_base"] = model_cfg["base_url"]
    return kwargs


class _ChatShim:
    """Responses API 结果 → chat-completions 形适配器（choices/message/usage），
    让既有 content 提取、``_record_llm``、tool_calls 消费代码零改动。"""

    def __init__(self, content: str, tool_calls: list | None, usage):
        msg = SimpleNamespace(content=content, tool_calls=tool_calls)
        self.choices = [SimpleNamespace(message=msg)]
        self.usage = usage


def _responses_tools(chat_tools: list[dict] | None) -> list[dict] | None:
    """chat tools（``{type:function,function:{name,...}}``）→ Responses 扁平格式。"""
    if not chat_tools:
        return None
    out = []
    for t in chat_tools:
        fn = t.get("function") or t
        out.append(
            {
                "type": "function",
                "name": fn.get("name"),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return out


def _messages_to_responses_input(messages: list[dict]) -> tuple[str, list[dict]]:
    """chat messages → ``(instructions, responses input 数组)``。

    assistant.tool_calls → ``function_call`` 项；role=tool → ``function_call_output`` 项；
    system 合并为 ``instructions``（Responses API 的系统提示位）。
    """
    instructions = ""
    input_items: list[dict] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content")
        if isinstance(content, list):  # anthropic 缓存分块 → 拼纯文本
            content = "".join(
                c.get("text", "")
                for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            )
        if role == "system":
            instructions = content or ""
        elif role == "assistant" and m.get("tool_calls"):
            if content:
                input_items.append({"role": "assistant", "content": content})
            for tc in m["tool_calls"]:
                fn = tc.get("function") or {}
                input_items.append(
                    {
                        "type": "function_call",
                        "call_id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "arguments": fn.get("arguments", "{}"),
                    }
                )
        elif role == "tool":
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": m.get("tool_call_id", ""),
                    "output": content
                    if isinstance(content, str)
                    else json.dumps(content, ensure_ascii=False),
                }
            )
        else:
            input_items.append({"role": role, "content": content or ""})
    return instructions, input_items


def _responses_output_to_chat(resp) -> tuple[str, list | None, SimpleNamespace]:
    """Responses 返回 → ``(text, chat 形 tool_calls, usage)``。"""
    text_parts: list[str] = []
    tool_calls: list = []
    for item in getattr(resp, "output", None) or []:
        itype = getattr(item, "type", None)
        if itype == "message":
            for c in getattr(item, "content", None) or []:
                if getattr(c, "type", "") == "output_text":
                    text_parts.append(getattr(c, "text", "") or "")
        elif itype == "function_call":
            tool_calls.append(
                SimpleNamespace(
                    id=getattr(item, "call_id", "") or f"call_{len(tool_calls)}",
                    type="function",
                    function=SimpleNamespace(
                        name=getattr(item, "name", ""),
                        arguments=getattr(item, "arguments", "{}"),
                    ),
                )
            )
    usage = getattr(resp, "usage", None)
    usage_details = getattr(usage, "input_tokens_details", None)
    usage_ns = SimpleNamespace(
        prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
        completion_tokens=getattr(usage, "output_tokens", 0) or 0,
        cached_tokens=getattr(usage_details, "cached_tokens", 0) or 0,
    )
    return "".join(text_parts), (tool_calls or None), usage_ns


def _responses_kwargs(model_cfg: dict, kwargs: dict, api_key: str | None) -> dict:
    """由 chat completions kwargs 构造 ``litellm.responses`` kwargs。

    gpt-5 系不接受 temperature / response_format，确定性靠 prompt 纪律
    （drbrain 各 prompt 均明写"只返回 JSON"）。
    """
    instructions, input_items = _messages_to_responses_input(kwargs.get("messages") or [])
    rk: dict = dict(
        model=f"openai/{model_cfg['model']}",
        input=input_items,
        max_output_tokens=_to_int(kwargs.get("max_tokens")) or 4096,
        timeout=kwargs.get("timeout", 60),
    )
    if instructions:
        rk["instructions"] = instructions
    tools = _responses_tools(kwargs.get("tools"))
    if tools:
        rk["tools"] = tools
    if api_key:
        rk["api_key"] = api_key
    if kwargs.get("api_base"):
        rk["api_base"] = kwargs["api_base"]
    return rk


def _responses_call_openai(rk: dict, model_cfg: dict, api_key: str | None):
    """OpenAI SDK 直连 Responses API——跳过 litellm 序列化层,前缀缓存可命中。"""
    base_url = model_cfg.get("base_url") or rk.pop("api_base", None) or "https://api.openai.com/v1"
    rk.pop("api_base", None)  # 无条件移除 litellm 专有字段
    rk.pop("api_key", None)
    # 剥 litellm 的 openai/ 前缀
    if "model" in rk and rk["model"].startswith("openai/"):
        rk["model"] = rk["model"][7:]
    client = _openai_client(api_key or "", base_url)
    return client.responses.create(**rk)


async def _aresponses_call_openai(rk: dict, model_cfg: dict, api_key: str | None):
    """:func:`_responses_call_openai` 的 async 版。"""
    base_url = model_cfg.get("base_url") or rk.pop("api_base", None) or "https://api.openai.com/v1"
    rk.pop("api_base", None)  # 无条件移除 litellm 专有字段
    rk.pop("api_key", None)
    if "model" in rk and rk["model"].startswith("openai/"):
        rk["model"] = rk["model"][7:]
    client = _aopenai_client(api_key or "", base_url)
    return await client.responses.create(**rk)


def _invoke_llm(model_cfg: dict, kwargs: dict, api_key: str | None):
    """统一调用入口：``wire_api: responses`` 的模型走 Responses API（含 tools 桥接），
    其余走 chat completions。返回 ``(content, response)``。

    中转示例（Sub2API，wire_api=responses）::

        - provider: sub2api
          model: gpt-5.6-sol
          base_url: https://www.kapestack.top/Sub2API/v1
          api_key: sk-...
          wire_api: responses
    """
    if model_cfg.get("wire_api") != "responses":
        resp = litellm.completion(**kwargs)
        return resp.choices[0].message.content, resp
    rk = _responses_kwargs(model_cfg, kwargs, api_key)
    resp = _responses_call_openai(rk, model_cfg, api_key)
    content, tool_calls, usage = _responses_output_to_chat(resp)
    return content, _ChatShim(content, tool_calls, usage)


async def _ainvoke_llm(model_cfg: dict, kwargs: dict, api_key: str | None):
    """:func:`_invoke_llm` 的 async 版（``litellm.aresponses``）。"""
    if model_cfg.get("wire_api") != "responses":
        resp = await litellm.acompletion(**kwargs)
        return resp.choices[0].message.content, resp
    rk = _responses_kwargs(model_cfg, kwargs, api_key)
    resp = await _aresponses_call_openai(rk, model_cfg, api_key)
    content, tool_calls, usage = _responses_output_to_chat(resp)
    return content, _ChatShim(content, tool_calls, usage)


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


def _cached_tokens(usage: Any) -> int:
    """Provider-reported prefix-cache hits; 0 unless the field is a real number."""
    value = getattr(usage, "cached_tokens", None) if usage is not None else None
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def _log_llm_call(
    *,
    model: str,
    provider: str,
    status: str,
    prompt_hash: str,
    n_messages: int,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cached_tokens: int = 0,
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
            "cached_tokens": _to_int(cached_tokens),
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
    disable_thinking: bool = False,
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
        _throttle(model_cfg)
        with _sync_sem(model_cfg):
            for attempt in range(_max_attempts(model_cfg)):
                # 解析本次调用用的 key（跳过冷却中的 key）；全池冷却 → 跳过该模型。
                api_key = _resolve_api_key(model_cfg)
                if model_cfg.get("api_keys") and api_key is None:
                    logger.warning(f"[llm] {name} all keys in cooldown — skipping")
                    break
                start = time.monotonic()
                try:
                    kwargs = _build_litellm_kwargs(
                        model_cfg,
                        prompt,
                        system_prompt,
                        max_tokens,
                        disable_thinking=disable_thinking,
                        api_key=api_key,
                    )
                    content, response = _invoke_llm(model_cfg, kwargs, api_key)
                    _record_llm(model_cfg["model"], model_cfg.get("provider", ""), response, start)
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
                        cached_tokens=_cached_tokens(usage),
                        duration_ms=elapsed,
                    )
                    parsed = json.loads(content)
                    if api_key:
                        _RATE_LIMIT_SM.on_success(model_cfg, api_key)
                    if _cache is not None:
                        _cache.set(key, parsed)
                    return parsed
                except Exception as e:
                    if _is_rate_limit(e):
                        # 只把失败的 key 打入冷却，继续尝试池里下一个 key（不跳过整个模型）。
                        wait = _RATE_LIMIT_SM.on_rate_limit(model_cfg, api_key or "")
                        logger.warning(
                            f"[llm] {name} key rate-limited — cooldown {wait:.0f}s, trying next key"
                        )
                        _log_llm_call(
                            model=model_cfg["model"],
                            provider=model_cfg.get("provider", ""),
                            status="error",
                            prompt_hash=_prompt_hash(prompt),
                            n_messages=1,
                            duration_ms=int((time.monotonic() - start) * 1000),
                            error=str(e)[:200],
                        )
                        continue  # 换下一个 key 重试
                    if _is_retryable(e) and attempt < _max_attempts(model_cfg) - 1:
                        logger.warning(
                            f"[llm] {name} retry {attempt + 1}/{_max_attempts(model_cfg)}: {e}"
                        )
                        time.sleep(2**attempt)
                        continue
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
                    break
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
    disable_thinking: bool = False,
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
        await _athrottle(model_cfg)
        async with _async_sem(model_cfg):
            for attempt in range(_max_attempts(model_cfg)):
                # 解析本次调用用的 key（跳过冷却中的 key）；全池冷却 → 跳过该模型。
                api_key = _resolve_api_key(model_cfg)
                if model_cfg.get("api_keys") and api_key is None:
                    logger.warning(f"[llm] async {name} all keys in cooldown — skipping")
                    break
                start = time.monotonic()
                try:
                    kwargs = _build_litellm_kwargs(
                        model_cfg,
                        prompt,
                        system_prompt,
                        max_tokens,
                        disable_thinking=disable_thinking,
                        api_key=api_key,
                    )
                    content, response = await _ainvoke_llm(model_cfg, kwargs, api_key)
                    _record_llm(model_cfg["model"], model_cfg.get("provider", ""), response, start)
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
                        cached_tokens=_cached_tokens(usage),
                        duration_ms=elapsed,
                    )
                    parsed = json.loads(content)
                    if api_key:
                        _RATE_LIMIT_SM.on_success(model_cfg, api_key)
                    if _cache is not None:
                        _cache.set(key, parsed)
                    return parsed
                except Exception as e:
                    if _is_rate_limit(e):
                        # 只把失败的 key 打入冷却，继续尝试池里下一个 key（不跳过整个模型）。
                        wait = _RATE_LIMIT_SM.on_rate_limit(model_cfg, api_key or "")
                        logger.warning(
                            f"[llm] async {name} key rate-limited — cooldown {wait:.0f}s, "
                            f"trying next key"
                        )
                        _log_llm_call(
                            model=model_cfg["model"],
                            provider=model_cfg.get("provider", ""),
                            status="error",
                            prompt_hash=_prompt_hash(prompt),
                            n_messages=1,
                            duration_ms=int((time.monotonic() - start) * 1000),
                            error=str(e)[:200],
                        )
                        continue  # 换下一个 key 重试
                    if _is_retryable(e) and attempt < _max_attempts(model_cfg) - 1:
                        logger.warning(
                            f"[llm] async {name} retry {attempt + 1}/{_max_attempts(model_cfg)}: {e}"
                        )
                        await asyncio.sleep(2**attempt)
                        continue
                    elapsed = int((time.monotonic() - start) * 1000)
                    logger.warning(
                        f"[llm] async {name} failed (attempt {i + 1}/{len(models)}): {e}"
                    )
                    _log_llm_call(
                        model=model_cfg["model"],
                        provider=model_cfg.get("provider", ""),
                        status="error",
                        prompt_hash=_prompt_hash(prompt),
                        n_messages=1,
                        duration_ms=elapsed,
                        error=str(e)[:200],
                    )
                    break
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

    for i, model_cfg in enumerate(models):
        name = f"{model_cfg['provider']}/{model_cfg['model']}"
        _throttle(model_cfg)
        with _sync_sem(model_cfg):
            for attempt in range(_max_attempts(model_cfg)):
                # 解析本次调用用的 key（跳过冷却中的 key）；全池冷却 → 跳过该模型。
                api_key = _resolve_api_key(model_cfg)
                if model_cfg.get("api_keys") and api_key is None:
                    logger.warning(f"Text model {name} all keys in cooldown — skipping")
                    break
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
                        "extra_body": _thinking_extra_body(model_cfg),
                    }
                    if api_key:
                        kwargs["api_key"] = api_key
                    if model_cfg.get("base_url"):
                        kwargs["api_base"] = model_cfg["base_url"]
                    content, _resp = _invoke_llm(model_cfg, kwargs, api_key)
                    if api_key:
                        _RATE_LIMIT_SM.on_success(model_cfg, api_key)
                    if content is None:
                        raise ValueError("empty LLM response content")
                    return content.strip()
                except Exception as e:
                    if _is_rate_limit(e):
                        # 只把失败的 key 打入冷却，继续尝试池里下一个 key。
                        wait = _RATE_LIMIT_SM.on_rate_limit(model_cfg, api_key or "")
                        logger.warning(
                            f"Text model {name} key rate-limited — cooldown {wait:.0f}s, "
                            f"trying next key"
                        )
                        continue  # 换下一个 key 重试
                    logger.warning(f"Text model {name} failed (attempt {i + 1}/{len(models)}): {e}")
                    break
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
        await _athrottle(model_cfg)
        async with _async_sem(model_cfg):
            for attempt in range(_max_attempts(model_cfg)):
                # 解析本次调用用的 key（跳过冷却中的 key）；全池冷却 → 跳过该模型。
                api_key = _resolve_api_key(model_cfg)
                if model_cfg.get("api_keys") and api_key is None:
                    logger.warning(f"Model {name} all keys in cooldown — skipping")
                    break
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
                        # thinking 模型（hy3/qwen3）默认开推理链，纯文本请求会
                        # 把 max_tokens 全吃进 thinking 导致 content 空 —— 统一禁用。
                        # ox-alpha-free 等强制 thinking 的模型配置 disable_thinking: false 跳过。
                        "extra_body": _thinking_extra_body(model_cfg),
                    }
                    if api_key:
                        kwargs["api_key"] = api_key
                    if model_cfg.get("base_url"):
                        kwargs["api_base"] = model_cfg["base_url"]
                    content, response = await _ainvoke_llm(model_cfg, kwargs, api_key)
                    _record_llm(model_cfg["model"], model_cfg.get("provider", ""), response, start)
                    logger.debug(
                        f"LLM text call success: {name} in {int((time.monotonic() - start) * 1000)}ms"
                    )
                    if api_key:
                        _RATE_LIMIT_SM.on_success(model_cfg, api_key)
                    if content is None:
                        raise ValueError("empty LLM response content")
                    text = content.strip()
                    if _cache is not None:
                        _cache.set(key, {"__text__": text})
                    return text
                except Exception as e:
                    if _is_rate_limit(e):
                        # 只把失败的 key 打入冷却，继续尝试池里下一个 key。
                        wait = _RATE_LIMIT_SM.on_rate_limit(model_cfg, api_key or "")
                        logger.warning(
                            f"Model {name} key rate-limited — cooldown {wait:.0f}s, trying next key"
                        )
                        continue  # 换下一个 key 重试
                    logger.warning(f"Model {name} failed (attempt {i + 1}/{len(models)}): {e}")
                    break
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
        _throttle(model_cfg)
        with _sync_sem(model_cfg):
            for attempt in range(_max_attempts(model_cfg)):
                # 解析本次调用用的 key（跳过冷却中的 key）；全池冷却 → 跳过该模型。
                api_key = _resolve_api_key(model_cfg)
                if model_cfg.get("api_keys") and api_key is None:
                    logger.warning(
                        f"[llm] call_with_messages {name} all keys in cooldown — skipping"
                    )
                    break
                start = time.monotonic()
                try:
                    kwargs: dict = {
                        "model": name,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "timeout": timeout,
                    }
                    if api_key:
                        kwargs["api_key"] = api_key
                    if model_cfg.get("base_url"):
                        kwargs["api_base"] = model_cfg["base_url"]
                    if tools:
                        kwargs["tools"] = tools
                        kwargs["extra_body"] = _thinking_extra_body(model_cfg)

                    content, response = _invoke_llm(model_cfg, kwargs, api_key)
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
                        cached_tokens=_cached_tokens(usage),
                        duration_ms=elapsed,
                    )
                    if api_key:
                        _RATE_LIMIT_SM.on_success(model_cfg, api_key)
                    result = {
                        "text": msg.content or "",
                        "tool_calls": _extract_tool_calls(msg),
                        "usage": {
                            "in": usage.prompt_tokens if usage else 0,
                            "out": usage.completion_tokens if usage else 0,
                            "cached": _cached_tokens(usage),
                        },
                    }
                    if _cache is not None and key is not None:
                        _cache.set(key, result)
                    return result
                except Exception as e:
                    if _is_rate_limit(e):
                        # 只把失败的 key 打入冷却，继续尝试池里下一个 key。
                        wait = _RATE_LIMIT_SM.on_rate_limit(model_cfg, api_key or "")
                        logger.warning(
                            f"[llm] call_with_messages {name} key rate-limited — "
                            f"cooldown {wait:.0f}s, trying next key"
                        )
                        _log_llm_call(
                            model=model_cfg["model"],
                            provider=model_cfg.get("provider", ""),
                            status="error",
                            prompt_hash=_messages_prompt_hash(messages),
                            n_messages=len(messages),
                            duration_ms=int((time.monotonic() - start) * 1000),
                            error=str(e)[:200],
                        )
                        continue  # 换下一个 key 重试
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
                    break
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
        await _athrottle(model_cfg)
        async with _async_sem(model_cfg):
            for attempt in range(_max_attempts(model_cfg)):
                # 解析本次调用用的 key（跳过冷却中的 key）；全池冷却 → 跳过该模型。
                api_key = _resolve_api_key(model_cfg)
                if model_cfg.get("api_keys") and api_key is None:
                    logger.warning(
                        f"[llm] acall_with_messages {name} all keys in cooldown — skipping"
                    )
                    break
                start = time.monotonic()
                try:
                    kwargs: dict = {
                        "model": name,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "timeout": timeout,
                    }
                    if api_key:
                        kwargs["api_key"] = api_key
                    if model_cfg.get("base_url"):
                        kwargs["api_base"] = model_cfg["base_url"]
                    if tools:
                        kwargs["tools"] = tools
                        kwargs["extra_body"] = _thinking_extra_body(model_cfg)

                    content, response = await _ainvoke_llm(model_cfg, kwargs, api_key)
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
                        cached_tokens=_cached_tokens(usage),
                        duration_ms=elapsed,
                    )
                    if api_key:
                        _RATE_LIMIT_SM.on_success(model_cfg, api_key)
                    result = {
                        "text": msg.content or "",
                        "tool_calls": _extract_tool_calls(msg),
                        "usage": {
                            "in": usage.prompt_tokens if usage else 0,
                            "out": usage.completion_tokens if usage else 0,
                            "cached": _cached_tokens(usage),
                        },
                    }
                    if _cache is not None and key is not None:
                        _cache.set(key, result)
                    return result
                except Exception as e:
                    if _is_rate_limit(e):
                        # 只把失败的 key 打入冷却，继续尝试池里下一个 key。
                        wait = _RATE_LIMIT_SM.on_rate_limit(model_cfg, api_key or "")
                        logger.warning(
                            f"[llm] acall_with_messages {name} key rate-limited — "
                            f"cooldown {wait:.0f}s, trying next key"
                        )
                        _log_llm_call(
                            model=model_cfg["model"],
                            provider=model_cfg.get("provider", ""),
                            status="error",
                            prompt_hash=_messages_prompt_hash(messages),
                            n_messages=len(messages),
                            duration_ms=int((time.monotonic() - start) * 1000),
                            error=str(e)[:200],
                        )
                        continue  # 换下一个 key 重试
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
                    break
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
