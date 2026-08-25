"""Focused contracts for the durable autoresearch ToolBroker."""

from __future__ import annotations

import asyncio
import time

from drbrain.loop.policy import ToolDefinition, ToolPolicy
from drbrain.loop.store import RunLedger
from drbrain.loop.tool_broker import ToolBroker, ToolCallStatus
from drbrain.loop.transitions import TransitionService
from drbrain.loop.workflow import ResearchLoopWorkflow
from drbrain.plugins.protocol import Plugin, PluginResult, ResultStatus


def _broker(
    tmp_path,
    *,
    step_capabilities: dict[str, set[str]] | None = None,
    lease_seconds: float = 60,
):
    ledger = RunLedger(tmp_path / "ledger.sqlite3")
    run = ledger.get_or_create_run("broker topic")
    transitions = TransitionService(ledger)
    transitions.start_run(run.run_id)
    step_id = transitions.begin_cycle(
        run.run_id,
        cycle=1,
        worker_id="worker-a",
        lease_seconds=lease_seconds,
    )
    attempt_id = ledger.active_attempt_id(step_id)
    assert attempt_id is not None
    broker = ToolBroker(
        ledger=ledger,
        run_id=run.run_id,
        step_id=step_id,
        attempt_id=attempt_id,
        worker_id="worker-a",
        lease_seconds=lease_seconds,
        policy=ToolPolicy(step_capabilities=step_capabilities or {"retrieve": {"graph:read"}}),
    )
    return broker, ledger, run.run_id


def _read_tool() -> ToolDefinition:
    return ToolDefinition(
        name="search_graph",
        source="graph",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        side_effect="read",
        required_capabilities=("graph:read",),
    )


def test_broker_writes_redacted_intent_before_handler_and_observation(tmp_path):
    broker, ledger, run_id = _broker(tmp_path)

    def handler():
        calls = ledger.tool_calls(run_id)
        assert len(calls) == 1
        assert calls[0].status == ToolCallStatus.INTENT
        return {"answer": "evidence"}

    observation = asyncio.run(
        broker.execute(
            node_name="retrieve",
            definition=_read_tool(),
            arguments={"query": "flat band", "api_key": "must-not-persist"},
            executor=handler,
        )
    )

    assert observation.status is ToolCallStatus.SUCCEEDED
    calls = ledger.tool_calls(run_id)
    assert len(calls) == 1
    assert calls[0].status == ToolCallStatus.SUCCEEDED
    assert calls[0].proposal["arguments"]["api_key"] == "[REDACTED]"
    assert calls[0].observation["output"] == {"answer": "evidence"}
    assert [event.event_type for event in ledger.events(run_id)][-2:] == [
        "tool_intended",
        "tool_observed",
    ]


def test_policy_denial_is_durable_and_never_reaches_the_handler(tmp_path):
    broker, ledger, run_id = _broker(tmp_path, step_capabilities={"retrieve": {"graph:read"}})
    invoked = False
    legacy_plugin = ToolDefinition(
        name="legacy_plugin",
        source="plugin",
        input_schema={"type": "object"},
        side_effect="unspecified",
        required_capabilities=("plugin:legacy_plugin",),
    )

    def handler():
        nonlocal invoked
        invoked = True
        return {"unexpected": True}

    observation = asyncio.run(
        broker.execute(
            node_name="retrieve",
            definition=legacy_plugin,
            arguments={},
            executor=handler,
        )
    )

    assert observation.status is ToolCallStatus.DENIED
    assert not invoked
    calls = ledger.tool_calls(run_id)
    assert calls[0].status == ToolCallStatus.DENIED
    assert ledger.events(run_id)[-1].event_type == "tool_denied"


def test_verifier_is_read_only_even_when_host_grants_a_write_capability(tmp_path):
    broker, _, _ = _broker(tmp_path, step_capabilities={"verify": {"plugin:mutate"}})
    definition = ToolDefinition(
        name="overwrite_experiment",
        source="plugin",
        input_schema={"type": "object"},
        side_effect="write",
        required_capabilities=("plugin:mutate",),
        supports_idempotency=True,
    )
    called = False

    def handler():
        nonlocal called
        called = True
        return {"changed": True}

    observation = asyncio.run(
        broker.execute(
            node_name="verify",
            definition=definition,
            arguments={},
            executor=handler,
            approved=True,
        )
    )

    assert observation.status is ToolCallStatus.DENIED
    assert called is False


def test_compute_output_is_resolved_to_its_durable_tool_call(tmp_path):
    broker, _, _ = _broker(tmp_path, step_capabilities={"compute": {"plugin:compute"}})
    definition = ToolDefinition(
        name="run_python",
        source="plugin",
        input_schema={"type": "object"},
        side_effect="write",
        required_capabilities=("plugin:compute",),
        supports_idempotency=True,
    )

    observation = asyncio.run(
        broker.execute(
            node_name="compute",
            definition=definition,
            arguments={},
            executor=lambda: {"job_id": "job-durable"},
            approved=True,
        )
    )

    assert broker.tool_call_id_for_output("job-durable") == observation.tool_call_id


def test_durable_mcp_requires_trust_and_a_nonempty_allowlist(tmp_path):
    broker, ledger, run_id = _broker(tmp_path, step_capabilities={"retrieve": {"mcp:echo"}})
    invoked = False
    mcp_tool = ToolDefinition(
        name="echo",
        source="mcp",
        input_schema={"type": "object"},
        side_effect="read",
        required_capabilities=("mcp:echo",),
        trusted=True,
    )

    def handler():
        nonlocal invoked
        invoked = True
        return "unexpected"

    observation = asyncio.run(
        broker.execute(
            node_name="retrieve",
            definition=mcp_tool,
            arguments={},
            executor=handler,
        )
    )

    assert observation.status is ToolCallStatus.DENIED
    assert not invoked
    assert ledger.tool_calls(run_id)[0].status == ToolCallStatus.DENIED


def test_read_timeout_retries_but_write_timeout_becomes_unknown(tmp_path):
    broker, ledger, run_id = _broker(
        tmp_path,
        step_capabilities={"retrieve": {"graph:read"}, "compute": {"plugin:writer"}},
    )
    attempts = 0

    async def eventually_reads():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("temporary read timeout")
        return {"answer": "retried"}

    read_observation = asyncio.run(
        broker.execute(
            node_name="retrieve",
            definition=_read_tool(),
            arguments={"query": "retry"},
            executor=eventually_reads,
        )
    )
    assert read_observation.status is ToolCallStatus.SUCCEEDED
    assert attempts == 2

    async def timed_out_write():
        raise TimeoutError("external write timeout")

    write_observation = asyncio.run(
        broker.execute(
            node_name="compute",
            definition=ToolDefinition(
                name="writer",
                source="plugin",
                input_schema={"type": "object"},
                side_effect="write",
                required_capabilities=("plugin:writer",),
                supports_idempotency=True,
            ),
            arguments={},
            executor=timed_out_write,
        )
    )
    assert write_observation.status is ToolCallStatus.UNKNOWN
    assert ledger.tool_calls(run_id)[-1].status == ToolCallStatus.UNKNOWN


def test_sync_executor_timeout_obeys_the_same_read_retry_policy(tmp_path):
    broker, _, _ = _broker(tmp_path)
    slow_read = ToolDefinition(
        name="slow_search",
        source="graph",
        input_schema={"type": "object"},
        side_effect="read",
        required_capabilities=("graph:read",),
        timeout_s=0.001,
    )

    observation = asyncio.run(
        broker.execute(
            node_name="retrieve",
            definition=slow_read,
            arguments={},
            executor=lambda: time.sleep(0.05),
        )
    )

    assert observation.status is ToolCallStatus.TIMED_OUT
    assert observation.attempts == 2


def test_broker_redacts_model_facing_tool_results(tmp_path):
    broker, _, _ = _broker(tmp_path)

    observation = asyncio.run(
        broker.execute(
            node_name="retrieve",
            definition=_read_tool(),
            arguments={"query": "safe"},
            executor=lambda: {
                "api_key": "live-secret",
                "note": "Authorization: Bearer also-secret",
            },
        )
    )

    message = observation.to_llm_message()
    assert "live-secret" not in message
    assert "also-secret" not in message
    assert "[REDACTED]" in message


def test_idempotency_records_never_contain_secret_arguments(tmp_path):
    broker, ledger, run_id = _broker(tmp_path)
    idempotent_read = ToolDefinition(
        name="search_graph",
        source="graph",
        input_schema={"type": "object"},
        side_effect="read",
        required_capabilities=("graph:read",),
        supports_idempotency=True,
    )

    executions = 0

    def handler():
        nonlocal executions
        executions += 1
        return {"answer": executions}

    first = asyncio.run(
        broker.execute(
            node_name="retrieve",
            definition=idempotent_read,
            arguments={"api_key": "must-not-persist"},
            executor=handler,
        )
    )
    second = asyncio.run(
        broker.execute(
            node_name="retrieve",
            definition=idempotent_read,
            arguments={"api_key": "different-credential"},
            executor=handler,
        )
    )

    calls = ledger.tool_calls(run_id)
    assert executions == 2
    assert first.reused is False
    assert second.reused is False
    assert calls[0].idempotency_key is not None
    assert calls[0].idempotency_key != calls[1].idempotency_key
    assert "must-not-persist" not in calls[0].idempotency_key
    assert "different-credential" not in calls[1].idempotency_key
    assert all("must-not-persist" not in str(event.payload) for event in ledger.events(run_id))
    assert all("different-credential" not in str(event.payload) for event in ledger.events(run_id))


def test_broker_renews_lease_during_a_long_tool_call(tmp_path):
    broker, ledger, run_id = _broker(tmp_path, lease_seconds=1)

    async def slow_read():
        await asyncio.sleep(1.1)
        return {"answer": "completed under renewed lease"}

    observation = asyncio.run(
        broker.execute(
            node_name="retrieve",
            definition=_read_tool(),
            arguments={"query": "long tool"},
            executor=slow_read,
        )
    )

    assert observation.status is ToolCallStatus.SUCCEEDED
    assert ledger.tool_calls(run_id)[0].status == ToolCallStatus.SUCCEEDED


def test_broker_accumulates_reported_resource_usage_across_retries(tmp_path):
    broker, ledger, run_id = _broker(tmp_path)
    attempts = 0

    def handler():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return PluginResult(
                ResultStatus.TIMEOUT,
                error="retry once",
                resource_usage={"gpu_seconds": 1.0},
            )
        return PluginResult(
            ResultStatus.OK,
            data={"answer": "completed"},
            resource_usage={"gpu_seconds": 2.0},
        )

    observation = asyncio.run(
        broker.execute(
            node_name="retrieve",
            definition=_read_tool(),
            arguments={"query": "retry resources"},
            executor=handler,
        )
    )

    assert attempts == 2
    assert observation.status is ToolCallStatus.SUCCEEDED
    assert observation.resource_usage["gpu_seconds"] == 3.0
    assert ledger.tool_calls(run_id)[0].observation["resource_usage"]["gpu_seconds"] == 3.0


def test_workflow_direct_search_routes_classified_plugin_through_broker(tmp_path):
    broker, ledger, run_id = _broker(
        tmp_path,
        step_capabilities={"retrieve": {"plugin:search_papers"}},
    )

    class Registry:
        plugin = Plugin(
            name="search_papers",
            description="Search local papers",
            input_schema={"type": "object"},
            side_effect="read",
            required_capabilities=("plugin:search_papers",),
        )

        def get(self, name):
            assert name == "search_papers"
            return self.plugin

        def call(self, name, arguments):
            assert name == "search_papers"
            assert arguments == {"query": "flat band", "limit": 10}
            calls = ledger.tool_calls(run_id)
            assert len(calls) == 1
            assert calls[0].status == ToolCallStatus.INTENT
            return PluginResult(ResultStatus.OK, data={"papers": [{"title": "A paper"}]})

    workflow = ResearchLoopWorkflow(tool_broker=broker)
    workflow._plugin_registry = Registry()  # noqa: SLF001 - explicit adapter boundary contract

    assert asyncio.run(workflow._direct_search("flat band")) == ["A paper"]  # noqa: SLF001
    assert ledger.tool_calls(run_id)[0].status == ToolCallStatus.SUCCEEDED
