from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from drbrain.capabilities import (
    APIAdapter,
    CapabilityCatalog,
    CapabilityDescriptor,
    CapabilityExecution,
    CapabilityJobMethods,
    CLIAdapter,
    InvocationResult,
    InvocationStatus,
    ModelAdapter,
    discover_skills,
    function_tool_name,
    input_digest,
    parse_skill,
    validate_instance,
)
from drbrain.plugins import Plugin, PluginRegistry, PluginResult, ResultStatus
from drbrain.rag import mcp_tools
from drbrain.rag.mcp_tools import mcp_descriptor_to_capability


def test_descriptor_round_trip_and_stable_input_digest():
    descriptor = CapabilityDescriptor(
        id="plugin:demo",
        name="demo",
        description="demo capability",
        kind="plugin",
        input_schema={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        },
    )
    assert CapabilityDescriptor.from_dict(descriptor.to_dict()) == descriptor
    assert input_digest({"b": 2, "a": 1}) == input_digest({"a": 1, "b": 2})
    assert validate_instance(descriptor.input_schema, {"value": 1}) == ()
    assert validate_instance(descriptor.input_schema, {"value": "bad"})
    future = CapabilityDescriptor(
        id="future:tool",
        name="tool",
        description="future protocol",
        kind="future_protocol",
    )
    assert future.kind == "future_protocol"


def test_descriptor_deserialization_reports_missing_identity_fields():
    with pytest.raises(ValueError, match="missing required field.*name"):
        CapabilityDescriptor.from_dict({"id": "plugin:demo"})


def test_schema_validation_sorts_mixed_object_and_array_paths():
    schema = {
        "type": "object",
        "properties": {
            "a": {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "additionalProperties": {"type": "integer"},
            }
        },
    }
    errors = validate_instance(schema, {"a": {0: "bad", "x": "bad"}})
    assert len(errors) == 2
    assert any("a[0]" in error for error in errors)
    assert any("a.x" in error for error in errors)


def test_function_tool_names_are_provider_safe_and_collision_safe():
    used = {"mcp_server_search"}
    assert function_tool_name("mcp:server:search", used) == "mcp_server_search_2"
    assert function_tool_name("model:bandgap", used) == "model_bandgap"


def test_plugin_registration_and_call_use_shared_contract():
    registry = PluginRegistry(process_isolation=False)
    plugin = Plugin(
        name="demo",
        description="demo",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        },
        side_effect="read",
        version="1",
    )
    registry.register(plugin, lambda args: {"value": args["value"]})
    assert registry.list_capabilities()[0].id == "plugin:demo"
    assert registry.call("demo", {"value": "bad"}).status is ResultStatus.INVALID_INPUT
    with pytest.raises(ValueError, match="already registered"):
        registry.register(plugin, lambda args: {}, replace=False)
    registry.register(plugin, lambda args: {"replaced": True}, replace=True)
    assert registry.call("demo", {"value": 1}).data == {"replaced": True}


def test_no_result_is_completed_but_legacy_ok_stays_false():
    result = PluginResult(ResultStatus.NO_RESULT)
    assert result.completed
    assert not result.ok
    neutral = result.to_invocation_result()
    assert neutral.status is InvocationStatus.NO_RESULT
    assert neutral.completed


def test_mcp_descriptor_keeps_namespace_and_rich_metadata():
    descriptor = mcp_descriptor_to_capability(
        {"id": "paper-server", "transport": "stdio", "command": "echo"},
        {
            "id": "mcp:paper-server:search",
            "name": "search",
            "description": "search papers",
            "inputSchema": {"type": "object"},
            "outputSchema": {"type": "object"},
            "annotations": {"readOnlyHint": True},
            "_meta": {"vendor": "demo"},
        },
    )
    assert descriptor.id == "mcp:paper-server:search"
    assert descriptor.annotations.read_only is True
    assert descriptor.output_schema == {"type": "object"}
    assert descriptor.metadata["_meta"] == {"vendor": "demo"}


def test_skill_adapter_validates_and_never_executes_files(tmp_path):
    skill_dir = tmp_path / "demo-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: A reusable skill\nallowed-tools: [search]\n---\n\nInstructions.\n",
        encoding="utf-8",
    )
    (skill_dir / "run.py").write_text("raise RuntimeError('must not execute')\n", encoding="utf-8")
    descriptor = parse_skill(skill_dir)
    assert descriptor.id == "skill:demo-skill"
    assert descriptor.metadata["resources"] == ["run.py"]
    assert descriptor.metadata["body"] == "Instructions."
    assert discover_skills(tmp_path) == [descriptor]


def test_skill_discovery_skips_malformed_yaml_by_default(tmp_path):
    skill_dir = tmp_path / "broken-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: [broken\ndescription: invalid\n---\n",
        encoding="utf-8",
    )

    assert discover_skills(tmp_path) == []
    with pytest.raises(ValueError, match="not valid YAML"):
        discover_skills(tmp_path, strict=True)


def test_mcp_result_preserves_empty_structured_content_and_error_text(monkeypatch):
    raw = SimpleNamespace(
        content=[{"type": "text", "text": "server detail"}],
        structuredContent={},
        isError=True,
    )
    policy = SimpleNamespace(server_id="paper-server", transport="stdio")
    result = mcp_tools._call_result_from_mcp(raw, policy, "search")
    assert result.status is InvocationStatus.ERROR
    assert result.structured_content == {}
    monkeypatch.setattr(mcp_tools, "_call", lambda *_args, **_kwargs: result)
    monkeypatch.setattr(mcp_tools, "_run_coro", lambda value: value)
    assert mcp_tools.call_mcp_tool({"command": "echo"}, "search", {}) == "server detail"


def test_mcp_discovery_timeout_uses_specific_error(monkeypatch):
    async def timeout(*_args, **_kwargs):
        raise TimeoutError("deadline")

    monkeypatch.setattr(mcp_tools, "_discover_unbounded", timeout)
    policy = SimpleNamespace(server_id="paper-server", timeout_seconds=1)
    with pytest.raises(mcp_tools.MCPTimeoutError):
        asyncio.run(mcp_tools._discover(policy))


def test_cli_timeout_returns_timeout_result(monkeypatch):
    def raise_timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="demo", timeout=0.1)

    monkeypatch.setattr(subprocess, "run", raise_timeout)
    adapter = CLIAdapter(name="demo", description="demo", command=("demo",), timeout_seconds=0.1)
    result = adapter.invoke({})
    assert result.status is InvocationStatus.TIMEOUT


def test_catalog_async_invoke_and_host_evidence_precedence():
    catalog = CapabilityCatalog()

    async def invoke(_arguments):
        return InvocationResult(
            InvocationStatus.OK,
            data={"ok": True},
            evidence={"capability_id": "spoofed", "runtime": "spoofed"},
        )

    catalog.register(
        CapabilityDescriptor(
            id="model:async",
            name="async",
            description="async capability",
            kind="model",
        ),
        invoke,
    )
    result = asyncio.run(catalog.ainvoke("model:async", {}))
    assert result.data == {"ok": True}
    assert result.evidence["capability_id"] == "model:async"
    assert result.evidence["runtime"]


def test_mcp_url_only_descriptor_uses_validated_policy_metadata():
    descriptor = mcp_descriptor_to_capability(
        {"url": "https://example.test/mcp"},
        {"name": "search", "inputSchema": {"type": "object"}},
    )
    assert descriptor.id == "mcp:https://example.test/mcp:search"
    assert descriptor.metadata["server_id"] == "https://example.test/mcp"
    assert descriptor.resource_scope["transport"] == "streamable_http"


def test_invocation_result_is_json_serializable():
    result = InvocationResult(
        InvocationStatus.OK,
        structured_content={"answer": 1},
        content=({"type": "text", "text": "answer"},),
    )
    json.dumps(result.to_dict(), ensure_ascii=False)


def test_catalog_unifies_model_api_cli_and_recommendation():
    catalog = CapabilityCatalog()
    catalog.register_adapter(
        APIAdapter(
            name="paper-search",
            description="search paper metadata",
            url="https://example.test/search",
            provider="literature",
            input_schema={"type": "object"},
            request=lambda **_kwargs: (200, {"papers": ["demo"]}),
        )
    )
    catalog.register_adapter(
        CLIAdapter(
            name="local-parser",
            description="parse a local document",
            command=(sys.executable, "-c", "import json; print(json.dumps({'ok': True}))"),
        )
    )
    catalog.register_adapter(
        ModelAdapter(
            name="bandgap-model",
            description="predict material band gap",
            predict=lambda args: {"bandgap": args["x"] * 2},
            input_schema={
                "type": "object",
                "properties": {"x": {"type": "number"}},
                "required": ["x"],
            },
        )
    )
    assert {item.kind for item in catalog.list()} == {"api", "cli", "model"}
    assert catalog.recommend("predict bandgap")[0].id == "model:bandgap-model"
    assert catalog.invoke("api:literature:paper-search", {}).data == {"papers": ["demo"]}
    assert catalog.invoke("model:bandgap-model", {"x": 3}).data == {"bandgap": 6}
    assert (
        catalog.invoke("model:bandgap-model", {"x": "bad"}).status is InvocationStatus.INVALID_INPUT
    )
    assert catalog.invoke("cli:local-parser", {}).data == {"ok": True}


def test_catalog_job_contract_normalizes_states_and_idempotency():
    calls = {"submit": 0}

    def submit(_args):
        calls["submit"] += 1
        return "job-1"

    state = {"value": "running"}
    catalog = CapabilityCatalog()
    descriptor = CapabilityDescriptor(
        id="model:long-run",
        name="long-run",
        description="long model run",
        kind="model",
        execution=CapabilityExecution(mode="async"),
    )
    catalog.register(
        descriptor,
        lambda _args: None,
        jobs=CapabilityJobMethods(
            submit=submit,
            poll=lambda _job: {"status": state["value"], "result": {"x": 1}},
            cancel=lambda _job: True,
        ),
    )
    first = catalog.submit_job("model:long-run", {}, idempotency_key="request-1")
    second = catalog.submit_job("model:long-run", {}, idempotency_key="request-1")
    assert first == second == "job-1"
    assert calls["submit"] == 1
    assert catalog.poll_job("model:long-run", "job-1").status is InvocationStatus.RUNNING
    state["value"] = "done"
    assert catalog.poll_job("model:long-run", "job-1").status is InvocationStatus.OK
    assert catalog.cancel_job("model:long-run", "../unsafe") is False


def test_catalog_persists_idempotency_records(tmp_path):
    calls = {"count": 0}
    descriptor = CapabilityDescriptor(
        id="cli:persisted",
        name="persisted",
        description="persisted job",
        kind="cli",
        execution=CapabilityExecution(mode="async"),
    )
    jobs = CapabilityJobMethods(
        submit=lambda _args: calls.__setitem__("count", calls["count"] + 1) or "job-1",
        poll=lambda _job: {"status": "pending"},
        cancel=lambda _job: True,
    )
    first = CapabilityCatalog(state_dir=tmp_path)
    first.register(descriptor, lambda _args: None, jobs=jobs)
    assert first.submit_job("cli:persisted", {}, idempotency_key="same") == "job-1"
    second = CapabilityCatalog(state_dir=tmp_path)
    second.register(descriptor, lambda _args: None, jobs=jobs)
    assert second.submit_job("cli:persisted", {}, idempotency_key="same") == "job-1"
    assert calls["count"] == 1
