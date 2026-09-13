# Troubleshooting

## Start with diagnostics

```bash
uv run drbrain check
uv run drbrain --help
```

## Common failures

| Symptom | Action |
| --- | --- |
| No models configured | Run `setup`, then configure `llm.models` or role-specific chains |
| Ingest cannot parse a PDF | Check MinerU settings; retry with the local parser fallback |
| Query returns no vector results | Run `embed --tree`, or use lexical `query` without the hybrid engine |
| Provider rate limit | Configure keys/rate limits or rely on cached/local metadata |
| Capability unavailable | Inspect descriptor discovery and adapter logs; other sources remain isolated |
| Session not found | Verify the session ID with `session list` |
| Restore rejects archive | Check manifest/checksum; use `--allow-legacy` only for trusted old archives |
| Loop cannot resume | Inspect checkpoint/run ledger compatibility and lease ownership |

Logs are under the configured `dirs.logs` path. Secrets and tool payloads are redacted before logging and persistence. When reporting a bug, include the command, exit code, sanitized log excerpt, and configuration section involved.
