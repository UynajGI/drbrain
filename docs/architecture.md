# DrBrain architecture

This document describes the current implementation. The knowledge-graph subsystem and WebUI are separate scopes and are intentionally excluded.

## System shape

```text
source files / URLs
        │
        ▼
parser + provider adapters ──► paper workspace + SQLite
        │                              │
        ├──────────────► RAG indexes   ├──► capability catalog
        │                              │          │
        └──────────────► extracted data│          ▼
                                       │    loop tool space
                                       ▼          │
                                  research loop ──┘
                                       │
                                       ▼
                              evidence-backed report
```

Each boundary is typed or persisted. Optional failures are recorded without preventing unrelated material from entering storage.

## Runtime boundaries

### Runtime and configuration

`RuntimeContext` resolves the project root and constrains runtime paths. `Config` loads YAML, expands environment variables, and exposes typed model, provider, storage, RAG, and loop settings. CLI commands pass the resolved configuration through Typer context.

### Storage

`drbrain.storage.database.Database` is the application write boundary. SQLite uses WAL mode and foreign keys. Database methods own write SQL and transaction semantics; callers may issue read-only queries for reporting. The file workspace stores PDFs, parsed Markdown, PageIndex trees, reports, and caches. The database stores identities, extracted records, retrieval metadata, sessions, claims, and evidence references.

Backups include runtime/schema metadata and an immutable SQLite snapshot. Restore validates the manifest before changing the destination and supports explicit legacy archives.

### Parsing and providers

Parser adapters return a normalized `Parsed` result. PDF ingestion prefers the configured MinerU/PageIndex path and falls back locally when unavailable. Provider adapters fetch metadata, documents, citations, patents, and web content behind bounded retries and cacheable responses. Provider failures remain attached to the ingestion result so partial material can still be stored.

### Retrieval (RAG)

The RAG layer combines BM25, vector, PageIndex/tree, RAPTOR, and graph-aware sources through fusion and optional reranking. Results retain source and section provenance. The RAG agent exposes retrieval and validation tools, while MCP tools are adapted into the same capability model as local plugins and Skills.

### Capabilities and plugins

`CapabilityDescriptor` is the neutral description of an invokable capability. `CapabilityCatalog` is the single discovery, recommendation, validation, invocation, and job entry point. Adapters cover Python plugins, MCP servers, non-executable Skills, model providers, APIs, and CLI commands. Descriptors include schemas, annotations, permissions, execution mode, job support, and provenance. Names are namespaced and duplicate registration is rejected unless replacement is explicit.

Invocation results share one envelope containing status, value, error, usage, and provenance. Async capabilities use bounded job IDs, optional idempotency keys, normalized states, and durable state when configured.

### Loop and tool space

The research loop is a bounded workflow. Each step receives a typed event, a scoped tool space, and a persisted run ledger. The tool-space layer asks the capability catalog for tools allowed by role, permissions, resource scope, and query relevance. The loop records calls, evidence, decisions, and checkpoints for resumption and audit.

Discovery, discussion, execution, verification, and settlement are separate phases. Compute and verification steps require explicit evidence before settlement can write claims.

## Typical request flow

1. The CLI resolves `RuntimeContext` and typed configuration.
2. Ingestion stores the source artifact and invokes parser/provider adapters.
3. Normalized data and provenance are committed through `Database` methods.
4. RAG indexes are built or incrementally refreshed.
5. A loop run creates a ledger and requests a role-scoped tool space from `CapabilityCatalog`.
6. Retrieval and capability calls return typed envelopes; failures remain attached to the step.
7. Verification checks evidence and execution outputs before claims are settled.
8. The CLI renders a report and preserves run artifacts.

## Fallback and failure policy

- Metadata providers fail independently; cached or local metadata remains usable.
- Parser fallback may reduce fidelity, but the source artifact is retained.
- RAG legs are isolated; fusion proceeds with valid legs.
- Discovery skips an unavailable adapter and records the reason.
- A job becomes terminal only after adapter confirmation or loop timeout policy.
- Writes use explicit transactions and atomic temporary-file replacement where partial output is unsafe.

## Extension rules

New protocols implement a capability adapter rather than an agent-specific dispatch path. The adapter provides a descriptor, invocation, normalized result, schema validation, and provenance; transport details remain inside it. New writes become `Database` methods. New CLI commands are registered in `src/drbrain/cli/main.py` and documented in [cli-reference.md](cli-reference.md).

## Verification

```bash
uv run drbrain --help
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/drbrain
uv run pytest -m "not integration"
```

When this document and implementation details diverge, the source modules and tests are authoritative; update this document with the boundary change.
