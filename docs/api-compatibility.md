# API compatibility policy

Effective immediately, DrBrain upgrades follow an **additive-only** public API
policy. Existing callers must continue to work without source, configuration,
or output rewrites.

## Protected public contracts

- Python import paths, public classes/functions, parameter names, parameter
  kinds, requiredness, and existing default values.
- CLI command names, existing flags, short aliases, defaults, exit semantics,
  and machine-readable JSON fields.
- YAML configuration keys and their existing default behavior, including a
  configuration file that omits a newer section.
- Serialized model fields, status/enum values, and persisted artifact layouts
  that callers may already read.

## Permitted evolution

- Add new import paths, commands, optional flags, optional keyword parameters,
  configuration keys with defaults, enum values, and JSON fields.
- Add richer metadata or health information without changing current result
  keys or their meanings.
- Deprecate an API only with a compatible implementation retained and a clear
  migration path. Removal or renaming requires a separately approved major
  compatibility decision.

## Enforcement

Every touched subsystem must carry a regression guard for its established
public surface. The first guard is
`tests/test_rag_public_compat.py`: it requires the RAG baseline to remain a
subset of the current API and permits only optional additions.

## RAG production additions

- RAG answers now add `evidence_ids`. If no stable retrieved evidence can be
  bound to a final answer, both the query engine and agent return the stable
  `status: "insufficient_evidence"` with an empty evidence list instead.
- `reason_llamaindex(..., principal=...)` adds optional session ownership
  binding. Omitting `principal` retains the local-CLI session behavior;
  supplying one denies access to a session owned by another principal (or an
  unowned legacy session).
- `load_mcp_tools(..., require_trusted=True)` and
  `build_agent(..., require_trusted_mcp=True)` opt into fail-closed MCP
  loading. A production MCP entry then requires `trusted: true`, a non-empty
  `allowed_tools` list, and a bounded optional `timeout_seconds`. Existing
  direct and local MCP calls keep their historic defaults.
