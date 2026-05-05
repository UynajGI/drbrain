# Engineering Hardening T2-T10

## Context

DrBrain 827 tests pass, 24 commands. Gap audit against scholaraio found 8 engineering weaknesses.
T1 (typed Config dataclass) already complete. This task covers T2 through T10.

## Requirements

### T2: Logging — session_id + ui()
- `get_session_id()` returns UUID4, lifecycle-scoped
- `ui()` writes to both console and log
- Log path from config, not hardcoded
- Keep Loguru

### T3: Metrics — WAL + thread-safety + remove dead code
- Remove: `record_api()`, `get_llm_stats()`, `LLMTimer`, `timed_llm()`, `api_calls` table
- Add: `PRAGMA journal_mode=WAL`, `check_same_thread=False`, `threading.Lock`
- Add: `timer()` context manager, `timed()` decorator, `session_id` field

### T4: Error Handling — custom exceptions + logger.exception()
- New `src/drbrain/exceptions.py` with 6 exception classes
- Audit openalex.py, crossref.py, citation.py — add `logger.exception()` in catch blocks
- `WorkspaceError` inherits from `DrBrainError`

### T5: Test Infra — conftest.py
- Shared `tmp_db` and `cfg_dict` fixtures
- Migrate 5 most-used test files first, rest gradually

### T6: API Clients — requests.Session + Retry
- Shared `_http_session()` with urllib3.Retry in openalex.py, crossref.py, citation.py
- MinerU exponential backoff
- PDF pre-validation

### T7: Storage — schema version + WAL + path accessors
- `schema_versions` table with version tracking
- `PRAGMA journal_mode=WAL`
- `storage/paths.py` with centralized path accessors
- Atomic writes everywhere

### T8: CLI — config cached once
- Load config in `_main_callback`, store in typer.Context
- Commands read from context instead of calling `load_config()`

### T9: Dependencies — check_import_error()
- `cli/dependencies.py` with install hints dictionary
- Used in setup.py and extractor optional imports
- Atomic write sweep for remaining write_text() calls

### T10: neat-freak — docs sync
- Update CLAUDE.md, CHANGELOG.md, README.md

## Success Criteria
- All 827+ tests pass (no regression)
- `drbrain check` works
- `drbrain query "test"` works
