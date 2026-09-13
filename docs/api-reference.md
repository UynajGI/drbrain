# API reference

The public Python boundaries are grouped by responsibility.

| Module | Primary API |
| --- | --- |
| `drbrain.config` | `Config`, typed sub-configurations, `load_config()` |
| `drbrain.storage.database` | `Database` and centralized write methods |
| `drbrain.storage.backup` | `create_backup()`, `restore_backup()`, `list_backups()` |
| `drbrain.parser` | normalized parser results and parser adapters |
| `drbrain.rag` | retrievers, fusion, indexing, reranking, evidence extraction |
| `drbrain.capabilities` | `CapabilityDescriptor`, `CapabilityCatalog`, invocation/job contracts |
| `drbrain.loop` | workflow, tool space, event log, checkpoints, run ledger |
| `drbrain.extractor.session_agent` | `SessionAgent` lifecycle and conversation methods |

## Compatibility rule

Prefer typed dataclasses and protocol interfaces at boundaries. Adapters may preserve legacy dictionaries for compatibility, but new integrations should use descriptors and invocation envelopes. Database writes must go through `Database` methods.

## Errors and results

Capability calls return normalized status and provenance. Parser, provider, and retrieval failures are reported at their boundary. CLI commands use non-zero exit codes and stderr for errors; `--json` output is intended for automation.

For exact signatures, inspect the module docstrings and type annotations in `src/drbrain/`; this page intentionally documents stable boundaries rather than duplicating generated API detail.
