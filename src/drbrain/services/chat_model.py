"""Explicit endpoint-bound chat adapter for the online roles (T19).

Design: ``docs/unified-tree-rag-design.md`` §7 — tree navigation and the final
answer both run on the ``chat_model`` role (DeepSeek today), while every
build-time call stays on ``index_model``.  This adapter is deliberately narrow:

* **Chat Completions only.**  The wire protocol is fixed at construction; there
  is no protocol switch and no per-provider API-surface inspection, so no
  provider can silently be called through a second API surface.
* **The endpoint travels with the instance.**  ``base_url`` and ``api_key`` are
  bound per instance from :func:`drbrain.services.model_roles.resolve_model_role`;
  no global environment variable and no module-level default switches them.
* **No hidden fallback.**  When ``llm.chat`` is configured it is authoritative —
  a failing chat endpoint raises instead of silently answering from
  ``llm.models``.  (``llm.models`` is only the documented legacy source when no
  chat role and no chat chain exist at all — see ``model_roles``.)
* **Missing credentials fail immediately**, at construction or on the first
  call, with an actionable message instead of an opaque 401 later.
* Each call returns its own ``usage``, so navigation and answering are metered
  separately and are recorded against the chat endpoint (never the index one).

The transport is injectable (``transport=``) so tests never touch the network;
the default transport reuses the shared client's pooled ``AsyncOpenAI`` instance
per ``(base_url, api_key)``.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from drbrain.extractor.llm_client import (
    _aopenai_client,
    _cached_tokens,
    _extract_tool_calls,
    _is_retryable,
    _log_llm_call,
    _max_attempts,
    _messages_prompt_hash,
    _record_llm,
)
from drbrain.security import safe_error
from drbrain.services.index_model import endpoint_gate
from drbrain.services.model_roles import (
    ROLE_CHAT,
    ModelRole,
    ModelRoleError,
    resolve_model_role,
)

#: A transport receives the prepared Chat Completions request and returns a
#: mapping with ``text``, ``tool_calls`` (wire shape), ``finish_reason``,
#: ``usage`` and — for real traffic — ``raw``.  Sync and async callables are
#: both accepted.
ChatTransport = Callable[[Mapping[str, Any]], Any]

_MAX_BACKOFF_SECS = 4.0


class ChatModelError(RuntimeError):
    """Typed failure from the endpoint-bound chat adapter.

    ``reason`` is ``"credentials"`` (missing/unresolved key), ``"transport"``
    (connection/HTTP/protocol failure), ``"truncated"`` (the answer budget was
    consumed before any text/tool call — common on reasoning endpoints) or
    ``"protocol"`` (malformed response).
    """

    REASONS = ("credentials", "transport", "truncated", "protocol")

    def __init__(
        self,
        reason: str,
        message: str,
        *,
        endpoint_name: str = "",
        model: str = "",
    ) -> None:
        if reason not in self.REASONS:
            raise ValueError(f"unknown ChatModelError reason {reason!r}")
        self.reason = reason
        self.endpoint_name = endpoint_name
        self.model = model
        target = endpoint_name or model or "chat endpoint"
        super().__init__(f"[{reason}] {target}: {message}")


@dataclass
class ChatToolCall:
    """One function call requested by the model."""

    id: str
    name: str
    arguments: str

    def arguments_dict(self) -> dict[str, Any]:
        """Best-effort parsed arguments (``{}`` when the model emitted junk)."""
        try:
            parsed = json.loads(self.arguments or "{}")
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def as_wire(self) -> dict[str, Any]:
        """OpenAI wire shape, for appending back to ``messages``."""
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass
class ChatResult:
    """One successful chat-completions response."""

    text: str
    tool_calls: list[ChatToolCall] = field(default_factory=list)
    finish_reason: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    endpoint_name: str = ""
    model: str = ""

    def as_assistant_message(self) -> dict[str, Any]:
        """Assistant message to append before sending tool results back."""
        message: dict[str, Any] = {"role": "assistant", "content": self.text or None}
        if self.tool_calls:
            message["tool_calls"] = [call.as_wire() for call in self.tool_calls]
        return message


def _run_blocking(coro: Awaitable[Any]) -> Any:
    """Run a coroutine from a sync caller without breaking a running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]
    # Inside a running loop the coroutine runs on a worker thread with its own
    # loop; the concurrency gate is thread-safe, so the cap still holds.
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro)).result()  # type: ignore[arg-type]


def _ms_since(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _usage_counts(usage: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key, value in dict(usage or {}).items():
        try:
            counts[str(key)] = int(value or 0)
        except (TypeError, ValueError):
            counts[str(key)] = 0
    return counts


class ChatModel:
    """Chat Completions client bound to the resolved ``chat_model`` endpoint."""

    def __init__(
        self,
        role: ModelRole | None = None,
        *,
        cfg: Any = None,
        transport: ChatTransport | None = None,
        max_attempts: int | None = None,
        slot_timeout: float | None = 600.0,
    ) -> None:
        """Bind one chat endpoint.

        Args:
            role: A pre-resolved ``chat_model`` role.
            cfg: Typed config (or dict); used to resolve the role when ``role``
                is omitted.  ``llm.chat`` / ``llm.roles.chat_model`` are
                authoritative whenever they exist.
            transport: Test seam — an async/sync callable receiving the prepared
                request.  When omitted, the real pooled OpenAI client is used
                and credentials are required up front.
            max_attempts: Total attempts per call (``None`` = shared policy).
            slot_timeout: Seconds to wait for a free slot on this endpoint's
                process-wide concurrency gate (``None`` waits forever).
        """
        if role is None:
            if cfg is None:
                raise ModelRoleError(ROLE_CHAT, "ChatModel needs a resolved role or a config")
            role = resolve_model_role(cfg, ROLE_CHAT)
        if role.role != ROLE_CHAT:
            raise ModelRoleError(
                ROLE_CHAT,
                f"ChatModel was given a {role.role!r} role",
                action="resolve the chat role with resolve_model_role(cfg, 'chat_model')",
            )
        self.role = role
        self._transport = transport
        self._external_transport = transport is not None
        if not self._external_transport and not role.base_url:
            raise ChatModelError(
                "credentials",
                "endpoint has no base_url",
                endpoint_name=role.endpoint_name,
                model=role.model,
            )
        if not self._external_transport:
            missing = role.missing_credential_reason
            if missing:
                raise ChatModelError(
                    "credentials",
                    missing,
                    endpoint_name=role.endpoint_name,
                    model=role.model,
                )
        self.max_attempts = max(1, int(max_attempts)) if max_attempts else _max_attempts({})
        self.slot_timeout = slot_timeout
        self._gate = endpoint_gate(role.identity, role.max_concurrent)

    # -- calls --------------------------------------------------------------

    async def achat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
        max_tokens: int = 1024,
        *,
        temperature: float = 0.3,
        timeout: float | None = None,
    ) -> ChatResult:
        """One chat-completions round trip (Chat Completions protocol only)."""
        request: dict[str, Any] = {
            "model": self.role.model,
            "messages": [dict(message) for message in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout if timeout is not None else self.role.timeout_secs,
        }
        if tools:
            request["tools"] = [dict(tool) for tool in tools]
            if tool_choice is not None:
                request["tool_choice"] = tool_choice
        await self._acquire()
        try:
            return await self._achat_with_retries(request)
        finally:
            self._gate.release()

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> ChatResult:
        """Sync entry point; shares the endpoint's process-wide gate."""
        return _run_blocking(self.achat(messages, tools, tool_choice, max_tokens, **kwargs))

    async def aprobe(self, *, timeout: float | None = None) -> dict[str, Any]:
        """One minimal real call, returned as redacted metadata.

        The budget is deliberately not tiny: reasoning endpoints (DeepSeek's
        ``deepseek-flash`` among them) spend the output budget on hidden
        reasoning first, so a 16-token probe would report a false failure.
        """
        started = time.monotonic()
        result = await self.achat(
            [{"role": "user", "content": "Reply with the single word: ok"}],
            max_tokens=128,
            temperature=0.0,
            timeout=timeout if timeout is not None else min(self.role.timeout_secs, 30.0),
        )
        return {
            **self.role.redacted(),
            "ok": True,
            "latency_ms": _ms_since(started),
            "finish_reason": result.finish_reason,
            "tokens_out": result.usage.get("out", 0),
        }

    def probe(self, *, timeout: float | None = None) -> dict[str, Any]:
        """Sync :meth:`aprobe` (used by ``drbrain check``)."""
        return _run_blocking(self.aprobe(timeout=timeout))

    # -- internals ----------------------------------------------------------

    async def _acquire(self) -> None:
        try:
            await self._gate.aslot(self.slot_timeout)
        except TimeoutError as exc:
            raise ChatModelError(
                "transport",
                f"no concurrency slot within {self.slot_timeout}s "
                f"(cap {self._gate.capacity}, in use {self._gate.in_use})",
                endpoint_name=self.role.endpoint_name,
                model=self.role.model,
            ) from exc

    async def _achat_with_retries(self, request: Mapping[str, Any]) -> ChatResult:
        last_error = ""
        for attempt in range(1, self.max_attempts + 1):
            started = time.monotonic()
            try:
                payload = self._transport_call(request)
                if inspect.isawaitable(payload):
                    payload = await payload
            except Exception as exc:  # noqa: BLE001 - classified below
                last_error = safe_error(exc, secrets=(self.role.api_key,))
                self._observe(request, status="error", started=started, error=exc)
                if attempt < self.max_attempts and _is_retryable(exc):
                    delay = min(_MAX_BACKOFF_SECS, 0.5 * 2 ** (attempt - 1))
                    logger.debug(
                        "[chat-model] {} attempt {}/{} failed ({}), retrying in {:.1f}s",
                        self.role.endpoint_name or self.role.model,
                        attempt,
                        self.max_attempts,
                        last_error,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise ChatModelError(
                    "transport",
                    last_error or "transport failed",
                    endpoint_name=self.role.endpoint_name,
                    model=self.role.model,
                ) from exc
            result = self._to_result(payload)
            self._observe(
                request,
                status="success",
                started=started,
                raw=payload.get("raw") if isinstance(payload, Mapping) else None,
            )
            return result
        raise ChatModelError(  # pragma: no cover - the loop returns or raises
            "transport",
            last_error or "transport failed",
            endpoint_name=self.role.endpoint_name,
            model=self.role.model,
        )

    def _to_result(self, payload: Any) -> ChatResult:
        if not isinstance(payload, Mapping):
            raise ChatModelError(
                "protocol",
                "transport returned a non-mapping response",
                endpoint_name=self.role.endpoint_name,
                model=self.role.model,
            )
        raw_calls = payload.get("tool_calls") or []
        tool_calls = [
            ChatToolCall(
                id=str(call.get("id") or ""),
                name=str((call.get("function") or {}).get("name") or ""),
                arguments=str((call.get("function") or {}).get("arguments") or ""),
            )
            for call in raw_calls
            if isinstance(call, Mapping)
        ]
        text = str(payload.get("text") or "")
        finish_reason = str(payload.get("finish_reason") or "")
        if finish_reason in {"length", "max_tokens"}:
            raise ChatModelError(
                "truncated",
                f"output budget exhausted before any answer (finish_reason={finish_reason}); "
                "reasoning endpoints spend this budget on hidden reasoning first",
                endpoint_name=self.role.endpoint_name,
                model=self.role.model,
            )
        if not text and not tool_calls:
            raise ChatModelError(
                "protocol",
                "endpoint returned neither text nor tool calls",
                endpoint_name=self.role.endpoint_name,
                model=self.role.model,
            )
        return ChatResult(
            text=text,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=_usage_counts(payload.get("usage")),
            endpoint_name=self.role.endpoint_name,
            model=self.role.model,
        )

    def _transport_call(self, request: Mapping[str, Any]) -> Any:
        if self._transport is not None:
            return self._transport(request)
        return self._default_transport(request)

    async def _default_transport(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """Real traffic: Chat Completions on the pooled ``AsyncOpenAI`` client."""
        client = _aopenai_client(self.role.api_key, self.role.base_url)
        response = await client.chat.completions.create(**dict(request))
        choice = response.choices[0]
        message = choice.message
        usage = getattr(response, "usage", None)
        return {
            "text": getattr(message, "content", "") or "",
            "tool_calls": _extract_tool_calls(message) or [],
            "finish_reason": getattr(choice, "finish_reason", "") or "",
            "usage": {
                "in": getattr(usage, "prompt_tokens", 0) if usage else 0,
                "out": getattr(usage, "completion_tokens", 0) if usage else 0,
                "cached": _cached_tokens(usage),
            },
            "raw": response,
        }

    def _observe(
        self,
        request: Mapping[str, Any],
        *,
        status: str,
        started: float,
        raw: Any = None,
        error: BaseException | None = None,
    ) -> None:
        """Record metrics/trace for real endpoint traffic only."""
        if self._external_transport:
            return
        messages = list(request.get("messages") or [])
        usage = getattr(raw, "usage", None) if raw is not None else None
        _log_llm_call(
            model=self.role.model,
            provider=self.role.provider,
            status=status,
            prompt_hash=_messages_prompt_hash([dict(message) for message in messages]),
            n_messages=len(messages),
            tokens_in=getattr(usage, "prompt_tokens", 0) if usage else 0,
            tokens_out=getattr(usage, "completion_tokens", 0) if usage else 0,
            cached_tokens=_cached_tokens(usage),
            duration_ms=_ms_since(started),
            error=str(error or ""),
            secrets=(self.role.api_key,),
        )
        if raw is not None:
            _record_llm(self.role.model, self.role.provider, raw, started)
