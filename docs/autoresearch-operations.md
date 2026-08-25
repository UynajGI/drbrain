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
`max_rag_calls`, `max_model_calls`, `max_wall_seconds`, `max_tokens`,
`max_cpu_seconds`, and `max_gpu_seconds`.
The shorter boundary names (`attempts`, `tool_calls`, `rag_calls`,
`model_calls`, `wall_seconds`, `tokens`, `cpu_seconds`, and `gpu_seconds`) are
accepted as additive aliases and are stored canonically with the `max_` prefix.

```python
await director.run(
    "topic",
    budget={
        "max_attempts": 6,
        "max_tool_calls": 20,
        "max_model_calls": 12,
        "max_tokens": 100_000,
        "max_gpu_seconds": 3_600,
    },
)
```

The ledger records every reservation and transitions the run to an explanatory
failed terminal state before a limit would be exceeded. `audit_summary()`
reports both configured limits and consumed usage. Existing callers that omit
`budget` retain their historical unbounded behavior.

Budget exhaustion is intentionally terminal: it is a hard cost/side-effect
ceiling rather than a pause. Start a new run with a higher limit after auditing
the exhausted run. For an opaque agent that cannot expose per-turn callbacks,
the loop reserves its configured maximum trajectory up front; failed model
requests still count as attempted external calls.

Call and attempt limits are reserved before their boundary executes. Token and
resource limits are additionally charged from facts available only afterward:
LlamaIndex agent turns use provider-reported `usage` metadata when present;
they are never estimated from text. ToolBroker records
`PluginResult.resource_usage` (`tokens`, `cpu_seconds`, `gpu_seconds`) and otherwise records
CPU only for bounded synchronous adapters running in an isolated worker thread.
A provider-reported CPU value takes precedence because it can attribute
subprocess work more precisely, and GPU time must be reported by the plugin
because the host cannot reliably attribute GPU work. If a bounded synchronous
adapter times out, its worker may continue after cancellation; the broker
conservatively charges its timeout duration as a single-thread CPU upper bound,
so retries cannot evade the CPU cap. If an observed amount crosses its
limit, that completed boundary stays in the audit trace and the run becomes
terminal before any later model, RAG, or tool boundary is admitted.
