# Autoresearch durable-run operations

The legacy workspace files (`champion.md`, `dead_ends.md`, `results/`, and
`knowledge/`) remain compatibility projections. The SQLite run ledger is the
operational source of truth for status, trace, controls, and audit summaries.

## Read and control a run

```python
from drbrain.loop import RunGovernance
from drbrain.loop.store import RunLedger

control = RunGovernance(RunLedger("workspace/autoresearch/ledger.sqlite3"))
status = control.status("research topic")     # topic or stable run_id
trace = control.trace(status["run_id"])       # read-only event and tool trace
audit = control.audit_summary(status["run_id"])

control.pause(status["run_id"], reason="operator_pause")
control.resume(status["run_id"])
control.cancel(status["run_id"], reason="operator_cancel")
```

`pause` and `cancel` never remove evidence or artifacts. New ToolBroker calls
are denied while a run is paused, cancelled, or budget-exhausted. `resume`
reopens only a paused run; checkpoint recovery remains owned by
`ResearchDirector`.

## Tool approvals

An irreversible or otherwise approval-gated idempotent tool call enters
`waiting_approval`. An operator can decide its deterministic retry contract:

```python
control.approve(tool_call_id, actor="operator")
# or
control.reject(tool_call_id, actor="operator", reason="outside experiment scope")
```

Approval never invokes a handler directly. It only allows (or denies) an
identical later ToolBroker retry, preserving the intent/observation audit
boundary.

## Budgets

`ResearchDirector.run()` accepts an additive `budget=` mapping for a new run.
Supported enforced limits are `max_attempts`, `max_tool_calls`,
`max_rag_calls`, `max_model_calls`, and `max_wall_seconds`.

```python
await director.run(
    "topic",
    budget={"max_attempts": 6, "max_tool_calls": 20, "max_model_calls": 12},
)
```

The ledger records every reservation and transitions the run to an explanatory
failed terminal state before a limit would be exceeded. `audit_summary()`
reports both configured limits and consumed usage. Existing callers that omit
`budget` retain their historical unbounded behavior.
