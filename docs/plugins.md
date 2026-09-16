# Capabilities and plugins

## Purpose

All agent-callable tools enter DrBrain through `CapabilityCatalog`. The catalog makes Python plugins, model providers, APIs, CLI programs, MCP tools, and Skill packages discoverable through one contract.

## Descriptor

`CapabilityDescriptor` contains a stable namespaced `id`, display name, description, kind, version, JSON input/output schemas, annotations, execution mode, permissions, resource scope, and provenance. Provenance can include code/resource digests, runtime, dependency lock digest, determinism, and seed.

## Invocation

Adapters expose `descriptor()` and `invoke(arguments)`. Registration validates schemas, rejects duplicate IDs by default, and isolates failures from optional sources. `CapabilityCatalog.invoke()` validates input and returns an `InvocationResult` envelope with status, value, error, usage, and provenance.

Long-running capabilities provide submit/poll/cancel methods. Job IDs are path-safe and bounded; idempotency keys return the existing job; state can be persisted atomically in a catalog state directory.

## Built-in adapters

- `PluginRegistry` adapts existing Python plugins while preserving its public API.
- MCP discovery preserves structured output, annotations, metadata, and namespaced IDs.
- Skills are instruction/resource descriptors and deliberately return `DENIED` when invoked; they are not executable tools.
- Future adapters can wrap an API, CLI, model, subprocess, or another protocol without changing loop code.

## Loop integration

`LoopToolSpace` creates a role- and step-scoped view over the catalog. Visibility applies permissions, side-effect annotations, resource scope, and policy. Tool definitions are derived from descriptors, so the planner never needs protocol-specific dispatch logic.

## Authoring checklist

1. Choose a stable namespace and capability ID.
2. Provide complete JSON schemas and safety annotations.
3. Return the standard invocation envelope, including failures.
4. Declare execution mode and job semantics when applicable.
5. Include provenance sufficient to identify the implementation and environment.
6. Add contract tests for discovery, validation, invocation, and failure behavior.

See the implementation in `src/drbrain/capabilities/` and the contract tests in `tests/test_capability_contract.py`.
