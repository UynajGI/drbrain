# Autoresearch durable-run operations

The legacy workspace files (`champion.md`, `dead_ends.md`, `results/`, and
`knowledge/`) remain compatibility projections. The SQLite run ledger is the
operational source of truth for status, trace, controls, and audit summaries.

## Start or resume a run

Autoresearch is opt-in. Configure the additive `autoresearch` section, then
use the supported CLI entry point:

```yaml
autoresearch:
  enabled: true
  run_dir: workspace/autoresearch
  max_cycles: 3
  require_rag_evidence: true
```

```bash
drbrain autoresearch run "research topic"
# Machine-readable run summary:
drbrain autoresearch run "research topic" --json
```

The topic is the durable run identity: invoking the command again with the
same topic resumes that run rather than creating a second ledger record. The
CLI only adds this operator entry; existing `ResearchDirector` and workflow
callers retain their previous defaults.

## External tools and RAG evidence mode

When `plugins_dir` or `mcp_servers` is configured, the CLI creates the
existing durable `ToolPolicy`/`ToolBroker` boundary. External tools are denied
unless their capability is explicitly granted to the relevant workflow step:

```yaml
autoresearch:
  plugins_dir: path/to/plugins
  step_capabilities:
    retrieve: ["rag:read", "plugin:search_papers"]
```

MCP tools additionally require the existing trusted-server and non-empty
allowlist contract. The configuration above does not grant a catch-all: an
empty `step_capabilities` map intentionally exposes no external tool.

`require_rag_evidence: true` is for a deliberately RAG-grounded run. In that
mode an unreferenced retrieval cannot promote a claim beyond `prediction`.
The strictness and retained RAG generation are persisted with the run; the
tool policy also forms part of the checkpoint manifest. A resumed run therefore
retains its original strictness, and a checkpoint whose strictness or policy
differs is rejected. Older checkpoints without the additive strictness field
retain their historical `false` semantics.

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

## Internal-beta live acceptance

The repository CI and focused offline regressions validate the operator path,
but they do not prove a real provider, plugin, and MCP deployment. When a
configured internal-beta environment is available, run one bounded topic with
`require_rag_evidence: true` and an explicitly capability-scoped read tool.
Record the run ID and verify: the ledger has a generation-pinned evidence
bundle, tool intent/observation records, claim-to-evidence links, and a resume
that preserves the same strict evidence setting. This is a targeted acceptance
run, not a new large-scale test program.
