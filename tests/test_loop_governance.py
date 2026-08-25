"""Operational controls and read-only audit contracts for durable runs."""

from __future__ import annotations

import asyncio

import pytest

from drbrain.loop.director import ResearchDirector
from drbrain.loop.governance import RunGovernance
from drbrain.loop.policy import ToolDefinition, ToolPolicy
from drbrain.loop.store import RunExecutionBlockedError, RunLedger
from drbrain.loop.tool_broker import ToolBroker, ToolCallStatus
from drbrain.loop.transitions import TransitionService
from drbrain.loop.workflow import ResearchLoopWorkflow


def _running(tmp_path, *, budget: dict[str, int | float] | None = None):
    ledger = RunLedger(tmp_path / "ledger.sqlite3")
    run = ledger.get_or_create_run("governed topic", budget=budget)
    transitions = TransitionService(ledger)
    transitions.start_run(run.run_id)
    step_id = transitions.begin_cycle(run.run_id, cycle=1, worker_id="worker", lease_seconds=60)
    attempt_id = ledger.active_attempt_id(step_id)
    assert attempt_id is not None
    broker = ToolBroker(
        ledger=ledger,
        run_id=run.run_id,
        step_id=step_id,
        attempt_id=attempt_id,
        worker_id="worker",
        lease_seconds=60,
        policy=ToolPolicy(step_capabilities={"compute": {"plugin:compute"}}),
    )
    return ledger, run.run_id, broker


def _compute_tool() -> ToolDefinition:
    return ToolDefinition(
        name="run_python",
        source="plugin",
        input_schema={"type": "object"},
        side_effect="write",
        required_capabilities=("plugin:compute",),
        supports_idempotency=True,
    )


def test_status_trace_and_audit_are_read_only_and_cancel_keeps_evidence(tmp_path):
    ledger, run_id, _ = _running(tmp_path)
    control = RunGovernance(ledger)
    before = len(ledger.events(run_id))

    status = control.status("governed topic")
    trace = control.trace(run_id)
    audit = control.audit_summary(run_id)

    assert status["run_id"] == run_id
    assert trace["events"]
    assert audit["event_count"] == before
    assert len(ledger.events(run_id)) == before

    control.cancel(run_id, reason="operator_stop")
    assert control.status(run_id)["status"] == "cancelled"
    assert control.trace(run_id)["events"][-1]["event_type"] == "run_cancelled"


def test_pause_blocks_new_broker_side_effects_and_resume_restores_execution(tmp_path):
    ledger, run_id, broker = _running(tmp_path)
    control = RunGovernance(ledger)
    control.pause(run_id, reason="operator_pause")
    invoked = False

    def handler():
        nonlocal invoked
        invoked = True
        return {"job_id": "unexpected"}

    blocked = asyncio.run(
        broker.execute(
            node_name="compute",
            definition=_compute_tool(),
            arguments={},
            executor=handler,
            approved=True,
        )
    )
    assert blocked.status is ToolCallStatus.DENIED
    assert invoked is False

    control.resume(run_id)
    allowed = asyncio.run(
        broker.execute(
            node_name="compute",
            definition=_compute_tool(),
            arguments={},
            executor=lambda: {"job_id": "allowed"},
            approved=True,
        )
    )
    assert allowed.status is ToolCallStatus.SUCCEEDED
    assert len(ledger.tool_calls(run_id)) == 1


def test_budget_blocks_new_tool_calls_and_records_an_explanation(tmp_path):
    ledger, run_id, broker = _running(tmp_path, budget={"max_tool_calls": 1})
    first = asyncio.run(
        broker.execute(
            node_name="compute",
            definition=_compute_tool(),
            arguments={"job": "one"},
            executor=lambda: {"job_id": "one"},
            approved=True,
        )
    )
    second = asyncio.run(
        broker.execute(
            node_name="compute",
            definition=_compute_tool(),
            arguments={"job": "two"},
            executor=lambda: {"job_id": "two"},
            approved=True,
        )
    )

    assert first.status is ToolCallStatus.SUCCEEDED
    assert second.status is ToolCallStatus.DENIED
    audit = RunGovernance(ledger).audit_summary(run_id)
    assert audit["budget"]["usage"]["tool_calls"] == 1
    assert audit["budget"]["exhausted"] is True


def test_approval_decision_reuses_the_waiting_call_contract(tmp_path):
    ledger, run_id, broker = _running(tmp_path)
    control = RunGovernance(ledger)
    tool = ToolDefinition(
        name="needs_approval",
        source="plugin",
        input_schema={"type": "object"},
        side_effect="irreversible",
        required_capabilities=("plugin:compute",),
        supports_idempotency=True,
    )
    waiting = asyncio.run(
        broker.execute(
            node_name="compute",
            definition=tool,
            arguments={"target": "fixture"},
            executor=lambda: {"changed": True},
        )
    )
    assert waiting.status is ToolCallStatus.WAITING_APPROVAL

    approved = control.approve(waiting.tool_call_id, actor="operator")
    retried = asyncio.run(
        broker.execute(
            node_name="compute",
            definition=tool,
            arguments={"target": "fixture"},
            executor=lambda: {"changed": True},
        )
    )

    assert approved["decision"] == "approved"
    assert retried.status is ToolCallStatus.SUCCEEDED


def test_approval_authorizes_only_one_retry_and_can_be_granted_again(tmp_path):
    ledger, run_id, broker = _running(tmp_path)
    control = RunGovernance(ledger)
    tool = ToolDefinition(
        name="needs_approval",
        source="plugin",
        input_schema={"type": "object"},
        side_effect="irreversible",
        required_capabilities=("plugin:compute",),
        supports_idempotency=True,
    )
    executions = 0

    def handler():
        nonlocal executions
        executions += 1
        return {"changed": executions}

    waiting = asyncio.run(
        broker.execute(
            node_name="compute", definition=tool, arguments={"target": "fixture"}, executor=handler
        )
    )
    control.approve(waiting.tool_call_id, actor="operator")
    assert (
        asyncio.run(
            broker.execute(
                node_name="compute",
                definition=tool,
                arguments={"target": "fixture"},
                executor=handler,
            )
        ).status
        is ToolCallStatus.SUCCEEDED
    )

    step_id = ledger.tool_calls(run_id)[-1].step_id
    transitions = TransitionService(ledger)
    transitions.complete_cycle(
        run_id,
        step_id=step_id,
        cycle_result={},
        state_snapshot={},
        research_state=None,
        worker_id="worker",
    )
    next_step = transitions.begin_cycle(run_id, cycle=2, worker_id="worker", lease_seconds=60)
    next_attempt = ledger.active_attempt_id(next_step)
    assert next_attempt is not None
    next_broker = ToolBroker(
        ledger=ledger,
        run_id=run_id,
        step_id=next_step,
        attempt_id=next_attempt,
        worker_id="worker",
        lease_seconds=60,
        policy=ToolPolicy(step_capabilities={"compute": {"plugin:compute"}}),
    )
    second_waiting = asyncio.run(
        next_broker.execute(
            node_name="compute", definition=tool, arguments={"target": "fixture"}, executor=handler
        )
    )

    assert second_waiting.status is ToolCallStatus.WAITING_APPROVAL
    assert executions == 1
    control.approve(second_waiting.tool_call_id, actor="operator")
    assert (
        asyncio.run(
            next_broker.execute(
                node_name="compute",
                definition=tool,
                arguments={"target": "fixture"},
                executor=handler,
            )
        ).status
        is ToolCallStatus.SUCCEEDED
    )
    assert executions == 2


def test_idempotent_replay_uses_cached_result_without_spending_budget(tmp_path):
    ledger, run_id, broker = _running(tmp_path, budget={"max_tool_calls": 1})
    executions = 0

    def handler():
        nonlocal executions
        executions += 1
        return {"job_id": "cached"}

    first = asyncio.run(
        broker.execute(
            node_name="compute",
            definition=_compute_tool(),
            arguments={"job": "cached"},
            executor=handler,
            approved=True,
        )
    )
    replay = asyncio.run(
        broker.execute(
            node_name="compute",
            definition=_compute_tool(),
            arguments={"job": "cached"},
            executor=handler,
            approved=True,
        )
    )

    assert first.status is ToolCallStatus.SUCCEEDED
    assert replay.status is ToolCallStatus.SUCCEEDED
    assert replay.reused is True
    assert executions == 1
    audit = RunGovernance(ledger).audit_summary(run_id)
    assert audit["budget"]["usage"]["tool_calls"] == 1
    assert audit["budget"]["exhausted"] is False


def test_model_budget_blocks_agent_before_it_is_called(tmp_path):
    ledger, run_id, _ = _running(tmp_path, budget={"max_model_calls": 0})
    invoked = False

    class _Agent:
        def run(self, **_kwargs):
            nonlocal invoked
            invoked = True

            async def result():
                return type("Result", (), {"response": type("Response", (), {"content": "ok"})()})()

            return result()

    workflow = ResearchLoopWorkflow(
        budget_reserver=lambda amounts: ledger.reserve_budget(run_id, amounts)
    )
    with pytest.raises(RunExecutionBlockedError):
        asyncio.run(workflow.run_agent(_Agent(), "do not execute"))

    assert invoked is False
    assert RunGovernance(ledger).status(run_id)["status"] == "failed"


def test_each_function_agent_turn_reserves_model_budget_before_calling_the_model(tmp_path):
    ledger, run_id, _ = _running(tmp_path, budget={"max_model_calls": 2})
    model_turns = 0

    class _Agent:
        async def take_step(self):
            nonlocal model_turns
            model_turns += 1

        def run(self, **_kwargs):
            async def result():
                await self.take_step()
                await self.take_step()
                await self.take_step()
                return type(
                    "Result", (), {"response": type("Response", (), {"content": "unexpected"})()}
                )()

            return result()

    workflow = ResearchLoopWorkflow(
        budget_reserver=lambda amounts: ledger.reserve_budget(run_id, amounts)
    )
    with pytest.raises(RunExecutionBlockedError):
        asyncio.run(workflow.run_agent(_Agent(), "count each turn"))

    assert model_turns == 2
    budget = RunGovernance(ledger).audit_summary(run_id)["budget"]
    assert budget["usage"]["model_calls"] == 2
    assert budget["exhausted"] is True


def test_cancel_revokes_the_active_lease_and_prevents_cycle_completion(tmp_path):
    ledger, run_id, broker = _running(tmp_path)
    step_id = ledger.active_leased_steps(run_id)[0]
    RunGovernance(ledger).cancel(run_id, reason="operator_stop")

    assert ledger.active_attempt_id(step_id) is None
    with pytest.raises(RunExecutionBlockedError):
        TransitionService(ledger).complete_cycle(
            run_id,
            step_id=step_id,
            cycle_result={},
            state_snapshot={},
            research_state=None,
            worker_id="worker",
        )
    invoked = False

    def handler():
        nonlocal invoked
        invoked = True
        return {"unexpected": True}

    blocked = asyncio.run(
        broker.execute(
            node_name="compute",
            definition=_compute_tool(),
            arguments={"job": "after-cancel"},
            executor=handler,
            approved=True,
        )
    )
    assert blocked.status is ToolCallStatus.DENIED
    assert blocked.execution_blocked is True
    assert invoked is False


def test_director_discards_an_active_cycle_when_a_runtime_budget_is_exhausted(tmp_path):
    director = ResearchDirector(cfg=object(), run_dir=tmp_path / "runs")

    async def exhaust_runtime_budget(*_args, **_kwargs):
        ledger = RunLedger(tmp_path / "runs" / "ledger.sqlite3")
        run = ledger.get_run("runtime budget")
        assert run is not None
        ledger.reserve_budget(run.run_id, {"model_calls": 1})
        raise AssertionError("budget reservation should have raised")

    director._run_cycle = exhaust_runtime_budget  # type: ignore[method-assign]
    state = asyncio.run(director.run("runtime budget", max_cycles=1, budget={"max_model_calls": 0}))

    ledger = RunLedger(tmp_path / "runs" / "ledger.sqlite3")
    run = ledger.get_run("runtime budget")
    assert run is not None
    event_types = [event.event_type for event in ledger.events(run.run_id)]
    assert state["cycles"] == 0
    assert run.status == "failed"
    assert "cycle_failed" in event_types
    assert "cycle_completed" not in event_types


def test_director_stops_before_claiming_a_cycle_when_attempt_budget_is_exhausted(tmp_path):
    director = ResearchDirector(cfg=object(), run_dir=tmp_path / "runs")

    state = asyncio.run(director.run("budgeted topic", max_cycles=3, budget={"max_attempts": 0}))

    control = RunGovernance(RunLedger(tmp_path / "runs" / "ledger.sqlite3"))
    assert state["cycles"] == 0
    assert control.status("budgeted topic")["status"] == "failed"
    assert control.audit_summary("budgeted topic")["budget"]["exhausted"] is True
