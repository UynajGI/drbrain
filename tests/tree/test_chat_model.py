"""T19 contract tests for the endpoint-bound chat adapter.

Every test injects a transport (or stubs the pooled client factory), so the
suite never touches the network.  Covered: base_url/api_key travel with the
instance, a full tool round trip, fail-fast missing credentials, the fixed Chat
Completions protocol (no Responses API), and no hidden fallback to ``llm.models``
when ``llm.chat`` is configured.
"""

from __future__ import annotations

import asyncio
import inspect
import itertools
import json
from types import SimpleNamespace

import pytest

import drbrain.services.chat_model as chat_model_module
from drbrain.services.chat_model import ChatModel, ChatModelError
from drbrain.services.model_roles import ModelRole, ModelRoleError, resolve_model_role

KEY = "sk-chat-SECRET-33bb44"
KEY_B = "sk-chat-SECRET-55cc66"

_PORTS = itertools.count(8301)


def _role(
    *,
    suffix: str = "a",
    api_key: str = KEY,
    base_url: str | None = None,
    max_concurrent: int = 4,
) -> ModelRole:
    """A distinct endpoint identity per call (the gate registry is process-wide)."""
    return ModelRole(
        role="chat_model",
        endpoint_name=f"chat_{suffix}_{next(_PORTS)}",
        provider="openai",
        model="deepseek-flash",
        base_url=base_url or "https://api.deepseek.com/v1",
        api_key=api_key,
        max_concurrent=max_concurrent,
        source="roles",
    )


def _tool_call_response():
    return {
        "text": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "kg_lookup", "arguments": '{"concept": "attention"}'},
            }
        ],
        "finish_reason": "tool_calls",
        "usage": {"in": 12, "out": 4},
    }


def _answer_response():
    return {
        "text": "Attention is a weighted sum.",
        "tool_calls": [],
        "finish_reason": "stop",
        "usage": {"in": 30, "out": 9},
    }


# -- instance-bound endpoint --------------------------------------------------


@pytest.mark.asyncio
async def test_base_url_and_api_key_travel_with_the_instance(monkeypatch):
    """Two instances bound to two endpoints never share credentials."""
    seen: list[tuple[str, str]] = []
    requests: list[dict] = []

    async def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, cached_tokens=0),
        )

    class _Client:
        chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    monkeypatch.setattr(
        chat_model_module,
        "_aopenai_client",
        lambda api_key, base_url: (seen.append((api_key, base_url)), _Client())[1],
    )
    monkeypatch.setattr(chat_model_module, "_log_llm_call", lambda **kwargs: None)
    monkeypatch.setattr(chat_model_module, "_record_llm", lambda *args, **kwargs: None)

    first = ChatModel(role=_role(suffix="p1", api_key=KEY, base_url="https://a.example/v1"))
    second = ChatModel(role=_role(suffix="p2", api_key=KEY_B, base_url="https://b.example/v1"))
    await first.achat([{"role": "user", "content": "hi"}])
    await second.achat([{"role": "user", "content": "hi"}])

    assert seen == [(KEY, "https://a.example/v1"), (KEY_B, "https://b.example/v1")]
    assert first.role.base_url != second.role.base_url
    assert len(requests) == 2


@pytest.mark.asyncio
async def test_chat_request_uses_only_the_chat_completions_protocol():
    """The wire request is chat-completions shaped - never a Responses payload."""
    seen: list[dict] = []

    def transport(request):
        seen.append(dict(request))
        return _answer_response()

    model = ChatModel(role=_role(suffix="p3"), transport=transport)
    await model.achat([{"role": "user", "content": "hi"}], max_tokens=128)
    request = seen[0]
    assert "messages" in request
    assert "input" not in request and "instructions" not in request
    assert request["max_tokens"] == 128
    assert request["model"] == model.role.model

    source = inspect.getsource(chat_model_module)
    assert "wire_api" not in source
    assert ".responses" not in source


# -- tool round trip ----------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_round_trip():
    """A tool call is returned typed and can be fed back as a message."""
    seen: list[dict] = []

    def transport(request):
        seen.append(dict(request))
        return _tool_call_response() if len(seen) == 1 else _answer_response()

    model = ChatModel(role=_role(suffix="t1"), transport=transport)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "kg_lookup",
                "description": "look up a concept",
                "parameters": {"type": "object", "properties": {"concept": {"type": "string"}}},
            },
        }
    ]
    first = await model.achat(
        [{"role": "user", "content": "what is attention?"}], tools=tools, tool_choice="auto"
    )
    assert first.finish_reason == "tool_calls"
    assert first.text == ""
    assert [call.name for call in first.tool_calls] == ["kg_lookup"]
    assert first.tool_calls[0].arguments_dict() == {"concept": "attention"}
    assert seen[0]["tools"] == tools
    assert seen[0]["tool_choice"] == "auto"

    messages = [
        {"role": "user", "content": "what is attention?"},
        first.as_assistant_message(),
        {"role": "tool", "tool_call_id": first.tool_calls[0].id, "content": "graph evidence"},
    ]
    second = await model.achat(messages, tools=tools)
    assert second.text == "Attention is a weighted sum."
    assert second.finish_reason == "stop"
    assert second.usage == {"in": 30, "out": 9}
    assert [message["role"] for message in seen[1]["messages"]] == ["user", "assistant", "tool"]
    assert seen[1]["messages"][1]["tool_calls"][0]["function"]["name"] == "kg_lookup"


def test_tool_call_arguments_are_parsed_defensively():
    model = ChatModel(
        role=_role(suffix="t2"),
        transport=lambda request: {
            "text": "",
            "tool_calls": [{"id": "c", "function": {"name": "f", "arguments": "not-json"}}],
            "finish_reason": "tool_calls",
            "usage": {},
        },
    )
    result = model.chat([{"role": "user", "content": "hi"}])
    assert result.tool_calls[0].arguments_dict() == {}


# -- routing and credentials --------------------------------------------------


def test_chat_chain_is_used_and_models_is_not_consulted():
    """With llm.chat configured, the chat role comes from it (no models fallback)."""
    cfg = {
        "llm": {
            "endpoints": {
                "ds": {
                    "provider": "openai",
                    "model": "deepseek-flash",
                    "api_key": KEY,
                    "base_url": "https://api.deepseek.com/v1",
                }
            },
            "roles": {"rag_chat": "ds"},
            "chat": [
                {
                    "provider": "openai",
                    "model": "deepseek-flash",
                    "api_key": KEY,
                    "base_url": "https://api.deepseek.com/v1",
                }
            ],
            "models": [
                {"provider": "openai", "model": "gpt-4o", "api_key": KEY_B, "base_url": None}
            ],
        }
    }
    model = ChatModel(cfg=cfg, transport=lambda request: _answer_response())
    assert model.role.endpoint_name == "ds"
    assert model.role.model == "deepseek-flash"
    assert "api.deepseek.com" in model.role.base_url
    assert model.role.source == "roles"


@pytest.mark.asyncio
async def test_no_hidden_fallback_to_generic_models_on_failure():
    """A failing chat endpoint raises; it never answers from llm.models."""
    calls: list[str] = []

    def transport(request):
        calls.append(request["model"])
        raise RuntimeError("connection refused")

    model = ChatModel(role=_role(suffix="r1"), transport=transport, max_attempts=1)
    with pytest.raises(ChatModelError) as excinfo:
        await model.achat([{"role": "user", "content": "hi"}])
    assert excinfo.value.reason == "transport"
    assert calls == [model.role.model]


def test_missing_api_key_raises_immediately():
    role = _role(suffix="c1", api_key="")
    with pytest.raises(ChatModelError) as excinfo:
        ChatModel(role=role)
    assert excinfo.value.reason == "credentials"
    assert "api_key" in str(excinfo.value)


def test_unresolved_env_placeholder_api_key_raises_immediately():
    role = _role(suffix="c2", api_key="${DEEPSEEK_API_KEY}")
    with pytest.raises(ChatModelError) as excinfo:
        ChatModel(role=role)
    message = str(excinfo.value)
    assert excinfo.value.reason == "credentials"
    assert "DEEPSEEK_API_KEY" in message and "export" in message


def test_keyless_local_endpoint_needs_no_credential():
    role = _role(suffix="c3", api_key="", base_url="http://127.0.0.1:8010/v1")
    model = ChatModel(role=role, transport=lambda request: _answer_response())
    assert model.role.api_key == ""
    assert model.chat([{"role": "user", "content": "hi"}]).finish_reason == "stop"


def test_chat_model_rejects_a_non_chat_role():
    index_role = ModelRole(
        role="index_model",
        endpoint_name="spark",
        provider="openai",
        model="spark-x25-4b",
        base_url="http://127.0.0.1:8010/v1",
        api_key=KEY,
        source="roles",
    )
    with pytest.raises(ModelRoleError, match="index_model"):
        ChatModel(role=index_role, transport=lambda request: _answer_response())


# -- shared budget, failures, probe -------------------------------------------


@pytest.mark.asyncio
async def test_chat_endpoint_cap_is_shared_between_callers():
    peak = {"current": 0, "max": 0}

    async def transport(request):
        peak["current"] += 1
        peak["max"] = max(peak["max"], peak["current"])
        await asyncio.sleep(0.02)
        peak["current"] -= 1
        return _answer_response()

    role = _role(suffix="g1", max_concurrent=1)
    navigation = ChatModel(role=role, transport=transport)
    answering = ChatModel(role=role, transport=transport)
    await asyncio.gather(
        *[navigation.achat([{"role": "user", "content": "n"}]) for _ in range(3)],
        *[answering.achat([{"role": "user", "content": "a"}]) for _ in range(3)],
    )
    assert peak["max"] == 1


@pytest.mark.asyncio
async def test_response_without_text_or_tool_calls_is_a_protocol_error():
    model = ChatModel(
        role=_role(suffix="g2"),
        transport=lambda request: {"text": "", "tool_calls": [], "finish_reason": "stop"},
    )
    with pytest.raises(ChatModelError) as excinfo:
        await model.achat([{"role": "user", "content": "hi"}])
    assert excinfo.value.reason == "protocol"


@pytest.mark.asyncio
async def test_budget_starved_response_is_a_typed_truncation():
    """A reasoning endpoint can spend the whole budget before answering."""
    model = ChatModel(
        role=_role(suffix="g5"),
        transport=lambda request: {"text": "", "tool_calls": [], "finish_reason": "length"},
    )
    with pytest.raises(ChatModelError) as excinfo:
        await model.achat([{"role": "user", "content": "hi"}], max_tokens=16)
    assert excinfo.value.reason == "truncated"
    assert "reasoning" in str(excinfo.value)


@pytest.mark.asyncio
async def test_probe_budget_survives_a_reasoning_prefix():
    """The probe asks for room to reason *and* answer."""
    seen: list[dict] = []

    def transport(request):
        seen.append(dict(request))
        return _answer_response()

    await ChatModel(role=_role(suffix="g6"), transport=transport).aprobe()
    assert seen[0]["max_tokens"] >= 64


@pytest.mark.asyncio
async def test_probe_returns_redacted_metadata():
    model = ChatModel(role=_role(suffix="g3"), transport=lambda request: _answer_response())
    payload = await model.aprobe()
    assert payload["ok"] is True
    assert payload["endpoint_name"] == model.role.endpoint_name
    assert payload["finish_reason"] == "stop"
    assert KEY not in json.dumps(payload, default=str)


def test_probe_failure_is_typed_and_key_free():
    recorder: dict = {}

    def transport(request):
        recorder["request"] = dict(request)
        raise RuntimeError(f"401 unauthorized for key {KEY}")

    model = ChatModel(role=_role(suffix="g4"), transport=transport, max_attempts=1)
    with pytest.raises(ChatModelError) as excinfo:
        model.probe()
    assert excinfo.value.reason == "transport"
    assert KEY not in str(excinfo.value)
    assert recorder["request"]["messages"][0]["role"] == "user"


def test_resolves_legacy_generic_models_when_nothing_else_is_configured():
    cfg = {"llm": {"models": [{"provider": "openai", "model": "gpt-4o", "api_key": KEY}]}}
    role = resolve_model_role(cfg, "chat_model")
    assert role.source == "models"
    model = ChatModel(cfg=cfg, transport=lambda request: _answer_response())
    assert model.role.model == "gpt-4o"
