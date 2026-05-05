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

## What to Log

- CLI invocations: `logger.info(f"CLI invoked: {cmd}")` with session_id.
- API failures: `logger.exception("OpenAlex API error")` before fallback.
- Translation progress: "Starting translation: N chunks", "Translation progress: X/N".
- Ingest events: stage transitions (parse → identify → tree), quality gate failures.

## What NOT to Log

- API keys or tokens (stored in config.local.yaml, gitignored).
- Full paper text (use char counts instead).
- Passwords, secrets, `os.environ` values.
