"""T18 contract tests for the shared index-model client.

Every test injects a transport (or stubs the pooled client factory), so the
suite never touches the network.  Covered: the process-wide per-endpoint
concurrency cap shared by two callers, ``finish_reason`` mapping, the typed
error reasons (empty / truncated / transport / budget), retry behaviour,
cancellation bookkeeping and the probe's redacted metadata.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import threading
import time
from types import SimpleNamespace

import pytest

import drbrain.services.index_model as index_model_module
from drbrain.services.index_model import (
    IndexModel,
    IndexModelError,
    endpoint_gate,
)
from drbrain.services.model_roles import ModelRole, ModelRoleError

KEY = "sk-index-SECRET-11aa22"

# Every call gets its own endpoint identity: the gate registry is process-wide,
# so distinct ports keep tests independent (and `_role` reuse keeps them shared
# where the test wants one endpoint).
_PORTS = itertools.count(8101)


def _role(
    *,
    suffix: str = "a",
    max_concurrent: int = 1,
    api_key: str = KEY,
    base_url: str | None = None,
) -> ModelRole:
    """A distinct endpoint per call unless ``base_url`` is pinned."""
    return ModelRole(
        role="index_model",
        endpoint_name=f"spark_{suffix}_{next(_PORTS)}",
        provider="openai",
        model="spark-x25-4b",
        base_url=base_url or f"http://127.0.0.1:{next(_PORTS)}/v1",
        api_key=api_key,
        max_concurrent=max_concurrent,
        source="roles",
    )


def _ok(text: str = "ok", finish_reason: str = "stop", usage: dict | None = None):
    def transport(request):
        return {"text": text, "finish_reason": finish_reason, "usage": usage or {"in": 3, "out": 1}}

    return transport


# -- happy path ---------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("suffix", ["a1", "a2"])
async def test_success_maps_finish_reason_and_identity(suffix):
    """A ``stop`` response maps to a typed result bound to its endpoint."""
    role = _role(suffix=suffix)
    result = await IndexModel(role=role, transport=_ok("summary")).acall_text("hi")
    assert result.text == "summary"
    assert result.finish_reason == "stop"
    assert result.usage == {"in": 3, "out": 1}
    assert result.endpoint_name == role.endpoint_name
    assert result.model == role.model


@pytest.mark.asyncio
async def test_async_call_maps_finish_reason():
    """The async entry point returns the same typed result."""
    result = await IndexModel(role=_role(suffix="b1"), transport=_ok("x")).acall_text("hi")
    assert (result.text, result.finish_reason) == ("x", "stop")


@pytest.mark.asyncio
async def test_system_prompt_is_sent_as_its_own_message():
    seen: list[dict] = []

    def transport(request):
        seen.append(dict(request))
        return {"text": "x", "finish_reason": "stop", "usage": {}}

    await IndexModel(role=_role(suffix="b2"), transport=transport).acall_text(
        "question", system_prompt="you are terse", max_tokens=64
    )
    assert [message["role"] for message in seen[0]["messages"]] == ["system", "user"]
    assert seen[0]["max_tokens"] == 64
    assert seen[0]["model"] == "spark-x25-4b"


# -- concurrency --------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrency_cap_is_shared_across_two_callers():
    """Two clients on one endpoint cannot exceed the endpoint's cap together."""
    peak = {"current": 0, "max": 0}

    async def transport(request):
        peak["current"] += 1
        peak["max"] = max(peak["max"], peak["current"])
        await asyncio.sleep(0.02)
        peak["current"] -= 1
        return {"text": "ok", "finish_reason": "stop", "usage": {}}

    role = _role(suffix="c1", max_concurrent=2)
    structure_tasks = IndexModel(role=role, transport=transport)
    summary_tasks = IndexModel(role=role, transport=transport)
    results = await asyncio.gather(
        *[structure_tasks.acall_text(f"s{i}") for i in range(4)],
        *[summary_tasks.acall_text(f"m{i}") for i in range(4)],
    )
    assert len(results) == 8
    assert peak["max"] == 2  # never 3+, the cap is shared, not per instance


@pytest.mark.asyncio
async def test_concurrency_cap_is_per_endpoint_not_per_model_name():
    """Same model name at two URLs keeps two independent budgets."""
    peak = {"current": 0, "max": 0}

    async def transport(request):
        peak["current"] += 1
        peak["max"] = max(peak["max"], peak["current"])
        await asyncio.sleep(0.02)
        peak["current"] -= 1
        return {"text": "ok", "finish_reason": "stop", "usage": {}}

    first = IndexModel(
        role=_role(suffix="d1", max_concurrent=1, base_url="http://127.0.0.1:9101/v1"),
        transport=transport,
    )
    second = IndexModel(
        role=_role(suffix="d2", max_concurrent=1, base_url="http://127.0.0.1:9102/v1"),
        transport=transport,
    )
    await asyncio.gather(first.acall_text("a"), second.acall_text("b"))
    assert peak["max"] == 2


@pytest.mark.asyncio
async def test_sync_caller_shares_the_same_gate():
    """A worker-thread (sync) call cannot bypass the async callers' cap."""
    role = _role(suffix="e1", max_concurrent=1)
    gate = endpoint_gate(role.identity, 1)
    await gate.aslot()  # hold the only slot
    model = IndexModel(role=role, transport=_ok(), slot_timeout=0.05)
    try:
        started = time.monotonic()
        with pytest.raises(IndexModelError) as excinfo:
            await asyncio.get_running_loop().run_in_executor(None, model.call_text, "sync")
        assert excinfo.value.reason == "budget"
        assert time.monotonic() - started < 5
    finally:
        gate.release()


# -- typed failures -----------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_response_is_a_typed_error():
    model = IndexModel(role=_role(suffix="f1"), transport=_ok(text="   "))
    with pytest.raises(IndexModelError) as excinfo:
        await model.acall_text("hi")
    assert excinfo.value.reason == "empty"


@pytest.mark.asyncio
async def test_truncated_response_is_a_typed_error():
    model = IndexModel(
        role=_role(suffix="f2"), transport=_ok(text="partial", finish_reason="length")
    )
    with pytest.raises(IndexModelError) as excinfo:
        await model.acall_text("hi")
    assert excinfo.value.reason == "truncated"
    assert "max_tokens" in str(excinfo.value)


@pytest.mark.asyncio
async def test_transport_failure_is_a_typed_error_and_is_retried():
    attempts = {"count": 0}

    def transport(request):
        attempts["count"] += 1
        raise RuntimeError("connection reset by peer")

    model = IndexModel(role=_role(suffix="f3"), transport=transport, max_attempts=2)
    with pytest.raises(IndexModelError) as excinfo:
        await model.acall_text("hi")
    assert excinfo.value.reason == "transport"
    assert attempts["count"] == 2  # retryable errors use the shared retry policy
    assert "connection reset" in str(excinfo.value)


@pytest.mark.asyncio
async def test_retry_then_success_returns_the_result():
    attempts = {"count": 0}

    def transport(request):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("503 service temporarily unavailable")
        return {"text": "recovered", "finish_reason": "stop", "usage": {}}

    result = await IndexModel(role=_role(suffix="f4"), transport=transport).acall_text("hi")
    assert result.text == "recovered"
    assert attempts["count"] == 2


@pytest.mark.asyncio
async def test_slot_budget_exhaustion_is_a_typed_error():
    role = _role(suffix="f5", max_concurrent=1)
    gate = endpoint_gate(role.identity, 1)
    await gate.aslot()
    try:
        model = IndexModel(role=role, transport=_ok(), slot_timeout=0.05)
        with pytest.raises(IndexModelError) as excinfo:
            await model.acall_text("hi")
        assert excinfo.value.reason == "budget"
        assert gate.in_use == 1  # our own held slot only - no leak
    finally:
        gate.release()


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_leak_a_slot():
    role = _role(suffix="f6", max_concurrent=1)
    gate = endpoint_gate(role.identity, 1)
    await gate.aslot()
    model = IndexModel(role=role, transport=_ok(), slot_timeout=5.0)
    waiting = asyncio.create_task(model.acall_text("hi"))
    await asyncio.sleep(0.02)  # let it park on the gate
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert gate.in_use == 1
    gate.release()
    assert gate.in_use == 0

    # The gate is still usable afterwards.
    result = await model.acall_text("hi")
    assert result.text == "ok"


@pytest.mark.asyncio
async def test_missing_credential_fails_fast_without_a_transport():
    role = _role(suffix="f7", api_key="", base_url="https://api.deepseek.com/v1")
    with pytest.raises(IndexModelError) as excinfo:
        IndexModel(role=role)
    assert excinfo.value.reason == "transport"
    assert "API key" in str(excinfo.value)


def test_index_model_rejects_a_non_index_role():
    other = ModelRole(
        role="chat_model",
        endpoint_name="deepseek",
        provider="openai",
        model="deepseek-flash",
        base_url="https://api.deepseek.com/v1",
        api_key=KEY,
        source="roles",
    )
    with pytest.raises(ModelRoleError, match="chat_model"):
        IndexModel(role=other, transport=_ok())


def test_resolves_from_config_when_role_is_omitted():
    cfg = {
        "llm": {
            "endpoints": {
                "spark_local": {
                    "provider": "openai",
                    "model": "spark-x25-4b",
                    "api_key": KEY,
                    "base_url": "http://127.0.0.1:8010/v1",
                    "max_concurrent": 3,
                }
            },
            "roles": {"index_model": "spark_local"},
        }
    }
    model = IndexModel(cfg=cfg, transport=_ok())
    assert model.role.endpoint_name == "spark_local"
    assert model._gate.capacity == 3


# -- transport seam and observability ----------------------------------------


@pytest.mark.asyncio
async def test_default_transport_uses_the_pooled_client_without_network(monkeypatch):
    """The real transport path is exercised with a stubbed SDK client."""
    seen: dict = {}
    traces: list[dict] = []
    metrics: list[tuple] = []

    async def create(**kwargs):
        seen["request"] = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="summary text"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3, cached_tokens=0),
        )

    class _Client:
        chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    def fake_client(api_key: str, base_url: str):
        seen["api_key"] = api_key
        seen["base_url"] = base_url
        return _Client()

    monkeypatch.setattr(index_model_module, "_aopenai_client", fake_client)
    # The trace/metrics sinks write files; stub them so the test stays hermetic.
    monkeypatch.setattr(index_model_module, "_log_llm_call", lambda **kwargs: traces.append(kwargs))
    monkeypatch.setattr(
        index_model_module, "_record_llm", lambda *args, **kwargs: metrics.append(args)
    )

    role = _role(suffix="g1", base_url="http://127.0.0.1:8201/v1")
    result = await IndexModel(role=role).acall_text("hi")

    assert result.text == "summary text"
    assert seen["api_key"] == KEY
    assert seen["base_url"] == role.base_url
    assert seen["request"]["model"] == role.model
    assert seen["request"]["messages"][0]["role"] == "user"
    assert traces[0]["status"] == "success"
    assert traces[0]["tokens_in"] == 7
    assert metrics  # provider usage was recorded for real traffic


@pytest.mark.asyncio
async def test_probe_returns_redacted_metadata(monkeypatch):
    monkeypatch.setattr(index_model_module, "_log_llm_call", lambda **kwargs: None)
    monkeypatch.setattr(index_model_module, "_record_llm", lambda *args, **kwargs: None)
    role = _role(suffix="h1", base_url="http://127.0.0.1:8301/v1")
    payload = await IndexModel(role=role, transport=_ok("ok", usage={"in": 1, "out": 1})).aprobe()
    assert payload["ok"] is True
    assert payload["endpoint_name"] == role.endpoint_name
    assert payload["finish_reason"] == "stop"
    assert payload["tokens_out"] == 1
    assert KEY not in json.dumps(payload, default=str)


@pytest.mark.asyncio
async def test_injected_transport_is_never_used_for_metrics(monkeypatch):
    """An injected transport is a test seam: no file/log side effects."""
    written: list[str] = []
    monkeypatch.setattr(index_model_module, "_log_llm_call", lambda **kw: written.append("trace"))
    monkeypatch.setattr(
        index_model_module, "_record_llm", lambda *a, **kw: written.append("metric")
    )
    monkeypatch.setattr(
        index_model_module,
        "_aopenai_client",
        lambda *a, **kw: pytest.fail("network client must not be constructed"),
    )
    await IndexModel(role=_role(suffix="h2"), transport=_ok()).acall_text("hi")
    assert written == []


def test_sync_call_from_a_plain_thread_works():
    """The sync entry point runs its own loop and still counts on the gate."""
    role = _role(suffix="h3", max_concurrent=2)
    model = IndexModel(role=role, transport=_ok("threaded"))
    result: dict = {}

    def run():
        result["value"] = model.call_text("hi")

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert result["value"].text == "threaded"
    assert endpoint_gate(role.identity, 2).in_use == 0
