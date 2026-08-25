"""Operational controls and read-only audit contracts for durable runs."""

from __future__ import annotations

import asyncio

from drbrain.loop.director import ResearchDirector
from drbrain.loop.governance import RunGovernance
from drbrain.loop.policy import ToolDefinition, ToolPolicy
from drbrain.loop.store import RunLedger
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
    answer = asyncio.run(workflow.run_agent(_Agent(), "do not execute"))

    assert answer is None
    assert invoked is False
    assert RunGovernance(ledger).status(run_id)["status"] == "failed"


def test_director_stops_before_claiming_a_cycle_when_attempt_budget_is_exhausted(tmp_path):
    director = ResearchDirector(cfg=object(), run_dir=tmp_path / "runs")

    state = asyncio.run(director.run("budgeted topic", max_cycles=3, budget={"max_attempts": 0}))

    control = RunGovernance(RunLedger(tmp_path / "runs" / "ledger.sqlite3"))
    assert state["cycles"] == 0
    assert control.status("budgeted topic")["status"] == "failed"
    assert control.audit_summary("budgeted topic")["budget"]["exhausted"] is True
