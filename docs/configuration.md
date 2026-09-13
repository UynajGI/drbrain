# Configuration

## Loading order

DrBrain loads the base YAML configuration, then an optional `config.local.yaml`, then resolves environment substitutions. CLI `--config PATH` selects another file and `--root PATH` selects the runtime namespace for relative data paths.

## Main sections

| Section | Purpose |
| --- | --- |
| `dirs` | inbox, pending, papers, reports, cache, logs, backups, citation styles |
| `db` | SQLite database path |
| `llm` | default and role-specific model fallback chains; node overrides |
| `mineru` | PDF parser backend, token, OCR and model |
| `pageindex` | PageIndex backend and tree settings |
| `embedding` | local, OpenAI-compatible, or disabled embedding provider |
| `rag` | retriever legs, fusion, reranking and evidence policy |
| `api` | Crossref, Semantic Scholar and DeepXiv credentials/rate limits |
| `backup` | rsync binaries and named remote targets |
| `autoresearch` | durable loop, leases, checkpoint and tool policy settings |
| `admin` | local authentication hash used by the optional application layer |

Use `config.example.yaml` as the shape reference. `Config` and its typed sub-configurations provide dict-style access for compatibility with older callers.

## Secrets

Put credentials in `config.local.yaml` or environment variables. The file is ignored by Git. Logs, event payloads, CLI output, and capability provenance pass through the sensitive-value redaction helpers before persistence or display.

## Runtime paths

Relative paths resolve below the runtime root. When `DRBRAIN_ROOT` or `DRBRAIN_RUNTIME_ROOT` is set, path validation rejects symlinks and paths escaping that root. This applies to database, workspace, backup, report, and loop-ledger paths.

## Validation

```bash
uv run drbrain setup --quick
uv run drbrain check
```

Configuration errors should be fixed before ingestion. Provider credentials are optional unless the selected command requires them; local parsing, local storage, and provider fallbacks remain usable without every external key.
