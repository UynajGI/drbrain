# Logging Guidelines

## Overview

- **Library**: Loguru (`from loguru import logger`).
- **Setup**: `setup_logging(level="DEBUG", log_path="data/logs/drbrain.log")` in `src/drbrain/log.py`.
- **Session ID**: `get_session_id()` returns UUID4, logged on setup, reused across process lifetime.
- **Canonical output**: `ui(message)` — writes to both stdout and log file via `logger.opt(depth=1).info()`.
- **Path**: Configurable via `cfg.dirs.logs`, defaults to `data/logs/drbrain.log`.
- **Rotation**: 10 MB, retain 5 backups.

## Log Levels

| Level | When |
|-------|------|
| `DEBUG` | Development details, chunk sizes, API request bodies |
| `INFO` | Key operations: ingest start, translation progress, tree extraction |
| `WARNING` | Recoverable issues: API timeout, missing optional metadata |
| `ERROR` | Failures: all LLM models exhausted, extraction failed |

## Structured Logging

- Loguru's `logger.bind(name=name)` for module-scoped loggers.
- `logger.opt(depth=1)` in helper functions to show caller location.
- `logger.exception()` for errors with full traceback (not `logger.error()`).

## Module Prefix Convention

Every log message MUST carry a `[module]` prefix for grep-ability. Use short, unique tags:

```
[ingest]    [build]    [closure]   [embed]     [llm]
[parse]     [tree]     [citations] [audit]     [repair]
[backup]    [setup]    [genealogy] [tree-retrieval]
[import]    [translate] [reasoner]  [db]
```

Format: `logger.info("[module] message — key=value")`.
Use em-dash (`—`) before key-value metadata, not colon.

```python
# Good — grep-able, key-value at end
logger.info("[ingest] Stage 1/4 parse: %s", pdf_path.name)
logger.info("[embed] model loaded: %s device=%s dim=%d", model_name, device, dim)

# Bad — no prefix, mixed formatting
logger.info("Starting ingest of %s" % pdf_path)
```

## Pipeline Stage Timing

Every multi-stage pipeline MUST log per-stage elapsed time. Use `time.monotonic()` (not `time.time()`).

```python
import time as _time

_t0 = _time.monotonic()
# ... Stage 1 ...
_t1 = _time.monotonic()
logger.info("[build] Stage 1/5 ontology done in %.1fs — %d types", _t1 - _t0, len(ontology))

# ... Stage 2 ...
_t2 = _time.monotonic()
logger.info("[build] Stage 2/5 entities done in %.1fs — %d concepts", _t2 - _t1, len(concepts))
```

Stage labels MUST use `N/M` format (e.g., `1/4`, `2/5`) so readers know progress position.

## Completion Summaries

Operations that produce counts MUST log aggregated results. Use dict/group-by for category breakdowns.

```python
# By-severity summary
_by_sev: dict[str, int] = {}
for i in issues:
    _by_sev[i["severity"]] = _by_sev.get(i["severity"], 0) + 1
logger.info("[audit] done in %.1fs — %d issues: %s", _t_done, len(issues), dict(_by_sev))

# By-rule summary
_by_rule: dict[str, int] = {}
for e in inferred:
    _by_rule[e["relation"]] = _by_rule.get(e["relation"], 0) + 1
logger.info("[closure] done in %.1fs — %d edges inferred: %s", _t_done, len(inferred), dict(_by_rule))
```

## Fallback Tracing

When a primary path fails and a fallback activates, log BOTH the failure and the fallback:

```python
# Good — both paths logged
if out_dir is not None:
    raw_md = self._read_output_md(out_dir)
    _parse_log.info("[parse] MinerU succeeded for %s", pdf_path.name)
else:
    _parse_log.warning("[parse] MinerU unavailable, falling back to PyMuPDF for %s", pdf_path.name)
    raw_md = self._fallback_pymupdf(pdf_path)
```

## Import Pattern

Module-level import when logging is pervasive; local import when logging is sparse.

```python
# Module-level (multiple functions use it)
from loguru import logger

# Local — single function, avoids import overhead for cold paths
def _embed_batch_openai_compat(texts, cfg):
    from loguru import logger as _my_log
```

When using local imports, alias to avoid shadowing callers (`_xxx_log` convention).

## What to Log

- CLI invocations: `logger.info(f"CLI invoked: {cmd}")` with session_id.
- API failures: `logger.exception("OpenAlex API error")` before fallback.
- Pipeline stages: entry with config, per-stage timing, exit with counts.
- Model loading: path, device, dimension.
- Batch progress: `N/M` iteration counts for multi-item operations.
- Fallback activation: when primary path fails and secondary is used.

## What NOT to Log

- API keys or tokens (stored in config.local.yaml, gitignored).
- Full paper text (use char counts instead).
- Passwords, secrets, `os.environ` values.
- Every iteration of a tight loop — log only aggregated counts.

## Common Mistakes

- **`time.time()` for elapsed time** — use `time.monotonic()`, not `time.time()`. `time.monotonic()` is monotonic and unaffected by system clock changes.
- **No prefix** — un-grep-able logs. Always use `[module]` prefix.
- **Colon instead of em-dash** — `—` separates message from metadata; `:` separates label from value within metadata.
- **Logging inside tight loops** — log once after the loop with aggregated counts.
- **Forgetting fallback trace** — if a primary path fails silently and fallback activates, the log should show both transitions.
