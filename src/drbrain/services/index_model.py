"""Shared build-time index-model client and scheduler (T18).

Design: ``docs/unified-tree-rag-design.md`` §7 — every build-time judgment
(structure decisions, title/region summaries, structural and semantic group
summaries) goes to the ``index_model`` role, and nowhere else.  This module is
the single client for that role:

* The endpoint is bound at construction from
  :func:`drbrain.services.model_roles.resolve_model_role`; no global environment
  variable switches it.
* ONE process-wide concurrency gate per endpoint identity is shared by all
  callers (structure tasks and summary tasks cannot add up to more than the
  endpoint's ``max_concurrent`` cap).  Sync callers run their own event loop in
  a worker thread and acquire the same gate, so the cap holds across threads.
* Timeouts are bounded per attempt and retries follow the shared client's
  policy (``retries`` semantics of ``extractor.llm_client``, 3 total attempts by
  default).
* Failures are typed: :class:`IndexModelError` carries ``reason`` one of
  ``"empty"`` (no usable text), ``"truncated"`` (cut off at ``max_tokens``),
  ``"transport"`` (credentials/connection/HTTP/protocol) or ``"budget"`` (no
  concurrency slot within the wait budget).  Callers can tell them apart; an
  empty or truncated response is never reported as success.

The transport is injectable (``transport=``) so tests never touch the network.
The default transport reuses the shared client's pooled ``AsyncOpenAI`` instance
per ``(base_url, api_key)`` with Chat Completions; metrics and the JSONL call
trace are recorded for real traffic only (an injected fake is a test seam).
"""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

# The private helpers below are the existing client's own retry/metrics seams.
# They are reused deliberately: the public ``acall_with_messages`` does not
# expose ``finish_reason``, which T18 requires for truncation detection.
from drbrain.extractor.llm_client import (
    _aopenai_client,
    _cached_tokens,
    _is_retryable,
    _log_llm_call,
    _max_attempts,
    _messages_prompt_hash,
    _record_llm,
)
from drbrain.security import safe_error
from drbrain.services.model_roles import (
    ROLE_INDEX,
    ModelRole,
    ModelRoleError,
    resolve_model_role,
)

#: A transport receives the prepared chat-completions request (``model``,
#: ``messages``, ``temperature``, ``max_tokens``, ``timeout``) and returns a
#: mapping with ``text``, ``finish_reason``, ``usage`` and — for real traffic —
#: ``raw`` (the provider response object, used for metrics).  Async and sync
#: callables are both accepted.
IndexTransport = Callable[[Mapping[str, Any]], Any]

_MAX_BACKOFF_SECS = 4.0

#: Process-wide gates keyed by endpoint identity (see ``ModelRole.identity``).
_gates: dict[str, _EndpointGate] = {}
_gates_lock = threading.Lock()


class IndexModelError(RuntimeError):
    """Typed failure from the shared index-model client.

    ``reason`` distinguishes the cases callers must handle differently:

    * ``"empty"`` — the endpoint returned no usable text
    * ``"truncated"`` — the response was cut off at ``max_tokens``
    * ``"transport"`` — credentials, connection, HTTP or protocol failure
    * ``"budget"`` — the per-endpoint concurrency budget was exhausted
    """

    REASONS = ("empty", "truncated", "transport", "budget")

    def __init__(
        self,
        reason: str,
        message: str,
        *,
        endpoint_name: str = "",
        model: str = "",
    ) -> None:
        if reason not in self.REASONS:
            raise ValueError(f"unknown IndexModelError reason {reason!r}")
        self.reason = reason
        self.endpoint_name = endpoint_name
        self.model = model
        target = endpoint_name or model or "index endpoint"
        super().__init__(f"[{reason}] {target}: {message}")


@dataclass
class IndexModelResult:
    """One successful index-model response."""

    text: str
    finish_reason: str
    usage: dict[str, int] = field(default_factory=dict)
    endpoint_name: str = ""
    model: str = ""


class _EndpointGate:
    """Process-wide concurrency gate for one endpoint identity.

    A single counter is shared by every caller; waiters park on their own event
    loop (or on a worker-thread loop created by :func:`_run_blocking`), so no
    thread is blocked while waiting and the cap cannot be exceeded by mixing
    sync and async callers.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, int(capacity))
        self._lock = threading.Lock()
        self._in_use = 0
        self._waiters: list[asyncio.Future] = []

    @property
    def in_use(self) -> int:
        with self._lock:
            return self._in_use

    async def aslot(self, timeout: float | None = None) -> None:
        """Acquire one slot on the caller's event loop."""
        with self._lock:
            if self._in_use < self.capacity:
                self._in_use += 1
                return
            waiter = asyncio.get_running_loop().create_future()
            self._waiters.append(waiter)
        try:
            await asyncio.wait_for(waiter, timeout)
        except (TimeoutError, asyncio.CancelledError):
            if waiter.done() and not waiter.cancelled():
                # Granted concurrently with the timeout/cancel: we own a slot
                # and the caller's ``finally`` releases it.
                return
            with self._lock:
                try:
                    self._waiters.remove(waiter)
                except ValueError:
                    pass
            raise

    def release(self) -> None:
        """Release one slot, handing it to the next waiter if any."""
        with self._lock:
            self._in_use = max(0, self._in_use - 1)
            while self._waiters:
                waiter = self._waiters.pop(0)
                if waiter.done():
                    continue  # cancelled/timed-out waiter - skip it
                self._in_use += 1
                try:
                    waiter.get_loop().call_soon_threadsafe(_resolve_waiter, waiter)
                except RuntimeError:  # event loop already closed
                    self._in_use -= 1
                    continue
                return


def _resolve_waiter(waiter: asyncio.Future) -> None:
    if not waiter.done():
        waiter.set_result(None)


def endpoint_gate(identity: str, capacity: int) -> _EndpointGate:
    """Return the process-wide gate for one endpoint identity.

    The first resolved capacity wins for the process lifetime: two configs that
    disagree about one endpoint's cap must not silently double the allowed
    concurrency.
    """
    with _gates_lock:
        gate = _gates.get(identity)
        if gate is None:
            gate = _EndpointGate(capacity)
            _gates[identity] = gate
        elif gate.capacity != max(1, int(capacity)):
            logger.debug(
                "[index-model] endpoint {} already has cap {}; ignoring requested {}",
                identity,
                gate.capacity,
                capacity,
            )
        return gate


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


class IndexModel:
    """The build-time ``index_model`` client (one endpoint, one shared cap)."""

    def __init__(
        self,
        role: ModelRole | None = None,
        *,
        cfg: Any = None,
        transport: IndexTransport | None = None,
        max_attempts: int | None = None,
        slot_timeout: float | None = 600.0,
    ) -> None:
        """Bind one endpoint.

        Args:
            role: A pre-resolved ``index_model`` role (preferred inside a build
                where the role was already resolved).
            cfg: Typed config (or dict); used to resolve the role when ``role``
                is omitted.
            transport: Test seam — an async/sync callable receiving the prepared
                request.  When omitted, the real pooled OpenAI client is used
                and credentials are required up front.
            max_attempts: Total attempts per call.  ``None`` reuses the shared
                client's default policy.
            slot_timeout: Seconds to wait for a free concurrency slot before
                raising ``IndexModelError("budget")``.  ``None`` waits forever.
        """
        if role is None:
            if cfg is None:
                raise ModelRoleError(ROLE_INDEX, "IndexModel needs a resolved role or a config")
            role = resolve_model_role(cfg, ROLE_INDEX)
        if role.role != ROLE_INDEX:
            raise ModelRoleError(
                ROLE_INDEX,
                f"IndexModel was given a {role.role!r} role",
                action="resolve the index role with resolve_model_role(cfg, 'index_model')",
            )
        self.role = role
        self._transport = transport
        self._external_transport = transport is not None
        if not self._external_transport and role.requires_api_key and not role.api_key:
            raise IndexModelError(
                "transport",
                "no API key for this endpoint",
                endpoint_name=role.endpoint_name,
                model=role.model,
            )
        self.max_attempts = max(1, int(max_attempts)) if max_attempts else _max_attempts({})
        self.slot_timeout = slot_timeout
        self._gate = endpoint_gate(role.identity, role.max_concurrent)

    # -- calls --------------------------------------------------------------

    async def acall(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        timeout: float | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> IndexModelResult:
        """One index-model call.  Raises :class:`IndexModelError` on failure."""
        request: dict[str, Any] = {
            "model": self.role.model,
            "messages": [dict(message) for message in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout if timeout is not None else self.role.timeout_secs,
        }
        if extra_body is not None:
            request["extra_body"] = dict(extra_body)

        await self._acquire()
        try:
            return await self._acall_with_retries(request)
        finally:
            self._gate.release()

    async def acall_text(
        self,
        prompt: str,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> IndexModelResult:
        """Convenience wrapper for a single user turn."""
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return await self.acall(messages, **kwargs)

    def call(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> IndexModelResult:
        """Sync entry point; shares the same process-wide gate."""
        return _run_blocking(self.acall(messages, **kwargs))

    def call_text(self, prompt: str, system_prompt: str = "", **kwargs: Any) -> IndexModelResult:
        """Sync entry point for a single user turn."""
        return _run_blocking(self.acall_text(prompt, system_prompt, **kwargs))

    async def aprobe(self, *, timeout: float | None = None) -> dict[str, Any]:
        """One minimal real call, returned as redacted metadata.

        Raises :class:`IndexModelError` (typed) when the endpoint is unusable.
        """
        started = time.monotonic()
        result = await self.acall_text(
            "Reply with the single word: ok",
            max_tokens=32,
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
            raise IndexModelError(
                "budget",
                f"no concurrency slot within {self.slot_timeout}s "
                f"(cap {self._gate.capacity}, in use {self._gate.in_use})",
                endpoint_name=self.role.endpoint_name,
                model=self.role.model,
            ) from exc

    async def _acall_with_retries(self, request: Mapping[str, Any]) -> IndexModelResult:
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
                        "[index-model] {} attempt {}/{} failed ({}), retrying in {:.1f}s",
                        self.role.endpoint_name or self.role.model,
                        attempt,
                        self.max_attempts,
                        last_error,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise IndexModelError(
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
        raise IndexModelError(  # pragma: no cover - the loop returns or raises
            "transport",
            last_error or "transport failed",
            endpoint_name=self.role.endpoint_name,
            model=self.role.model,
        )

    def _to_result(self, payload: Any) -> IndexModelResult:
        if not isinstance(payload, Mapping):
            raise IndexModelError(
                "transport",
                "transport returned a non-mapping response",
                endpoint_name=self.role.endpoint_name,
                model=self.role.model,
            )
        text = str(payload.get("text") or "")
        finish_reason = str(payload.get("finish_reason") or "")
        if finish_reason in {"length", "max_tokens"}:
            raise IndexModelError(
                "truncated",
                f"response hit the max_tokens budget (finish_reason={finish_reason})",
                endpoint_name=self.role.endpoint_name,
                model=self.role.model,
            )
        if not text.strip():
            raise IndexModelError(
                "empty",
                "endpoint returned no usable text",
                endpoint_name=self.role.endpoint_name,
                model=self.role.model,
            )
        return IndexModelResult(
            text=text,
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
        """Real traffic: the pooled ``AsyncOpenAI`` client (Chat Completions)."""
        client = _aopenai_client(self.role.api_key, self.role.base_url)
        response = await client.chat.completions.create(**dict(request))
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        return {
            "text": getattr(choice.message, "content", "") or "",
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
