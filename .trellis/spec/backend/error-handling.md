# Error Handling

## Overview

- **Exception hierarchy**: `src/drbrain/exceptions.py` — `DrBrainError` base, `ConfigError`, `APIError`/`APIRateLimitError`, `ExtractionError`, `StorageError`.
- **WorkspaceError** inherits from `DrBrainError` (moved from `workspace.py`).
- All API client exceptions must call `logger.exception()` before returning fallback values.

## Error Types

```python
class DrBrainError(Exception): ...          # base
class ConfigError(DrBrainError): ...        # config loading/validation
class APIError(DrBrainError): ...           # external API failure
class APIRateLimitError(APIError): ...      # rate limit exceeded
class ExtractionError(DrBrainError): ...    # concept/argument extraction
class StorageError(DrBrainError): ...       # DB or file storage
```

## Error Handling Patterns

**API clients** (openalex.py, crossref.py, citation.py, repair.py):
```python
try:
    data = api_call(...)
except Exception:
    logger.exception("OpenAlex API error")
    return None  # or [], or 0 — preserve existing fallback
```
- Always `logger.exception()` (includes traceback) before returning fallback.
- Wrap logger call in its own try/except to prevent logging failures from crashing.
- Keep existing return values — don't change logic, only ADD logging.

**CLI commands**: Use `raise typer.Exit(1)` on user-facing errors.
**Do NOT**: `except Exception: pass` — this was common pre-T4, now prohibited.

## API Error Responses

- CLI errors: `typer.echo("message", err=True)` + `raise typer.Exit(1)`.
- JSON mode: `{"error": "message"}` with exit 1.
- Internal functions: return `None`, `[]`, or `{}` on failure, log via `logger.exception()`.

## Common Mistakes

- Silent exception swallowing (`except Exception: pass`) — fixed in T4 audit.
- Forgetting `logger.exception()` in API retry loops — all calls now have it.
- Using bare `except:` without type — never used in this codebase.
- Raising exceptions without logging — always log first, then raise/re-raise.
