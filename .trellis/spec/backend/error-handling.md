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

## Multi-Stage Pipeline Resilience

The build pipeline (`build_graph_from_tree`) runs 5 sequential LLM stages. Each stage must be resilient to failure.

### Idempotency

- Re-running `drbrain build` on a paper that partially completed a prior run must be safe.
- Each stage must check for pre-existing output before starting: if stage N data already exists and is valid, skip to stage N+1.
- Ontology extension (Stage 1) is shared across papers — concurrent builds must not produce duplicate ontology entries.

### Stage-Level Error Recovery

- If Stage N fails, Stages 1..N-1 intermediate results must be preserved (not discarded).
- On re-run, resume from the failed stage, not from scratch.
- Intermediate results stored in DB with `status` field (`pending`, `in_progress`, `complete`, `failed`).

### Rollback

- Stage 5 (Refinement) corrects earlier extraction output. If refinement degrades quality, the pre-refinement state must be recoverable.
- Store a `refinement_diff` (before/after) so refinement can be reverted.

### Contract

```python
class StageStatus(enum.StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"

async def run_stage(
    stage: str,
    paper_id: str,
    input_data: dict,
    *,
    resume: bool = True,
) -> StageResult:
    """Run a build stage with idempotency guard.

    Returns StageResult with .status, .data, .diff (if refinement).
    Raises ExtractionError on unrecoverable failure.
    """
```

## Common Mistakes

- Silent exception swallowing (`except Exception: pass`) — fixed in T4 audit.
- Forgetting `logger.exception()` in API retry loops — all calls now have it.
- Using bare `except:` without type — never used in this codebase.
- Raising exceptions without logging — always log first, then raise/re-raise.
- Multi-stage pipeline that discards partial results on failure — each stage must persist before proceeding to the next.
