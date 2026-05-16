# Troubleshooting

Common problems and how to fix them.

## Environment

### `ModuleNotFoundError: No module named 'drbrain'`

Editable install missing. After `uv sync`, run:

```bash
uv pip install -e .
```

### `drbrain: command not found`

The CLI entry point wasn't installed. Verify:

```bash
uv pip install -e .
which drbrain
```

### Config not found

`drbrain setup` generates `config.local.yaml` from `config.example.yaml`.
If you skipped setup:

```bash
cp config.example.yaml config.yaml
drbrain setup --quick
```

---

## PDF Ingest

### MinerU API unreachable

Symptoms: ingest hangs or "MinerU API timeout" in logs.

- Check `mineru.token` in config
- Verify network access to MinerU endpoint
- **Fallback**: DrBrain auto-falls back to PyMuPDF (pymupdf4llm). No action needed unless you need formula/table quality.

### PDF encrypted / corrupted

Symptoms: "PDF encryption detected" or ingest fails immediately.

- Remove DRM from PDF before ingest
- `drbrain check` runs PDF validation by default

### Ingest succeeded but paper is empty

Check `data/papers/<id>/raw.md` — if empty, MinerU parsing failed. Check logs at
`data/logs/drbrain.log` for the specific error. The PDF may be scanned (image-only);
try enabling `mineru.is_ocr: true`.

---

## LLM API

### All models failed / timeout

Symptoms: `All N models failed` in logs, command returns empty or exits.

1. `drbrain check` — verifies LLM connectivity
2. Check `llm.models` in config — first model is primary, rest are fallbacks
3. Verify API keys are set (use `${ENV_VAR}` syntax, not plain text in config.yaml)
4. Check network access to provider base URLs
5. Increase timeout: litellm defaults to 60s per call

### Rate limiting (429)

- Add more models to the fallback chain
- Reduce `extract.max_concurrent`
- Semantic Scholar: reduce `s2_rate_limit` or add `s2_api_key`

### JSON parse error from LLM

Models occasionally return malformed JSON. DrBrain logs the raw response and fails
over to the next model in the chain. If ALL models fail JSON parsing consistently,
try a different model (e.g. switch from a small local model to a cloud one).

---

## Database

### Database locked

SQLite WAL mode should prevent this. If it persists:

- No other process should hold a write lock
- `data/drbrain.db-wal` and `data/drbrain.db-shm` are normal WAL files — don't delete them
- Kill any hung `drbrain` processes

### Schema migration error

DrBrain auto-migrates on `Database()` init. If a migration fails:

- Check `schema_versions` table for current version
- Restore from backup: `drbrain backup` creates `data/backups/drbrain-<timestamp>.tar.gz`
- File a bug with the schema version and error message from logs

---

## Embedding

### Model download stuck

First `drbrain embed --tree` downloads the model (~1.2 GB for Qwen3-Embedding-0.6B).

- `source: "modelscope"` — faster in China, slower elsewhere
- `source: "huggingface"` — set `hf_endpoint` if behind firewall
- Set `device: "cpu"` if GPU driver issues

### CUDA out of memory

- Reduce `embed.batch_size` (try 8 or 16)
- Set `embed.device: "cpu"` to skip GPU entirely
- GPU profiler auto-tunes on next run; profile cache at `~/.cache/drbrain/gpu_profile.json`

### openai-compat returns empty

- Verify `embed.api_base` ends with `/v1` (no trailing slash)
- Test with curl: `curl -H "Authorization: Bearer $KEY" $BASE/models`
- Check `embed.api_key` is set (use `${ENV_VAR}`)

### Dimension mismatch

Happens when switching embedding models. Re-run to regenerate all vectors:

```bash
drbrain embed --tree
```

---

## Workspaces

### Workspace not found

```bash
drbrain ws list   # see all workspaces
drbrain ws show <name>   # inspect one
```

### Rename failed

Workspace names must be valid directory names (no `/`, `..`, or special chars).

---

## Logs

### Where to find logs

| What | Where |
|------|-------|
| App log (structured) | `data/logs/drbrain.log` |
| Per-session ID | Log entries include `session_id` UUID4 |
| LLM token tracking | `data/metrics.db` (`llm_calls` table) |
| API cache | `data/cache/` (safe to delete) |

### Increasing log verbosity

Set `LOGURU_LEVEL=DEBUG` environment variable before running any command:

```bash
LOGURU_LEVEL=DEBUG drbrain ingest
```

---

## Recovery

### Restore from backup

```bash
drbrain backup              # creates tar.gz in data/backups/
# To restore:
tar xzf data/backups/drbrain-<timestamp>.tar.gz
```

### Rebuild search index

```bash
drbrain index
```

### Reset everything

```bash
drbrain clean               # removes DB and cache, keeps papers/
drbrain setup --quick       # re-initialize
drbrain index               # rebuild search index
```
