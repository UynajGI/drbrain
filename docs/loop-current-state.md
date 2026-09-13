# Research loop

## Role

The loop is a bounded, resumable workflow for evidence-backed research. It coordinates retrieval, capability calls, discussion, computation, verification, and settlement. It does not discover arbitrary tools or persist raw writes itself; those concerns belong to `LoopToolSpace`, `CapabilityCatalog`, and `RunLedger`.

## Execution model

Each run has a durable run ID, ordered event log, step/attempt records, leases, tool intents/results, checkpoints, and final settlement. Events are versioned JSON envelopes with sequence and idempotency checks. Checkpoints include workflow state, reducer version, and RAG evidence mode so incompatible resumes are rejected.

## Tool selection

At each step, `LoopToolSpace` derives a scoped tool view from the capability catalog. Visibility is determined by role, policy, permissions, side effects, and resource scope. Recommendations are deterministic lexical matches over descriptor metadata; invocation still performs schema and policy validation.

## Control phases

1. **Discover**: gather relevant evidence and capabilities.
2. **Discuss**: record analyst/critic messages and queue decisions.
3. **Execute**: run approved retrieval, model, API, CLI, or compute capabilities.
4. **Verify**: check outputs and evidence against the step contract.
5. **Settle**: persist only verified claims and report artifacts.

Workers use leases and idempotency keys. A retry may replay an equivalent event or tool call, but a conflicting payload is rejected.

## Failure behavior

Transient work can retry within its policy. Expired leases are reclaimed once. Invalid checkpoints, conflicting sequences, invalid tool arguments, and failed verification move the run to an explicit failure or manual-review state; they are not silently treated as success.

## Extension point

New workflow steps consume typed events and a `LoopToolSpace`; they should not import MCP, PluginRegistry, or provider-specific dispatch code. New persistence fields belong in `RunLedger` and its schema migration path.

## Verification

```bash
uv run pytest tests/test_loop* -q
uv run ruff check src/drbrain/loop
```

Authoritative implementation lives in `src/drbrain/loop/`, especially `research_events.py`, `checkpointing.py`, `tool_space.py`, `tool_broker.py`, and `store.py`.
