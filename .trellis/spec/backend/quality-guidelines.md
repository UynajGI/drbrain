# Quality Guidelines

## Overview

- **Linter**: Ruff (`uv run ruff check .`), config in `pyproject.toml`.
- **Formatter**: Ruff format (`uv run ruff format .`).
- **Test runner**: pytest with `asyncio_mode = "auto"`, 1000+ tests.
- **Pre-commit hooks**: Ruff check + format on all Python files.

## Forbidden Patterns

- `except Exception: pass` — always log the exception (T4 audit fixed all instances).
- `except:` without exception type — never used.
- Direct `print()` in library code — use `logger.info()` or `ui()`.
- `cfg["key"]["path"]` without fallback — use `cfg.db.path` (typed Config dataclass).
- `write_text()` without tmp→rename — use atomic writes for all filesystem writes.
- `urllib.request` — use `requests.Session` with `urllib3.Retry` adapter.
- `from .module import *` — wildcard imports hide dependency chains and pollute namespaces. Always import explicit names.
- `json_str.replace('None', 'null')` or similar string-replace on LLM JSON output — corrupts string values containing those substrings. Use `json_repair` library or schema-constrained generation instead.
- Logger that re-serializes and writes entire log file on every `.info()` call — O(n²) I/O. For JSONL/metrics logs, append one line at a time; for structured logs, batch writes or use loguru.

## Required Patterns

- **Atomic writes**: `tmp_path.write_text(data); tmp_path.replace(path)` for all file writes.
- **Backward compat**: Config changes must support dict-style access (`__getitem__`, `get()`).
- **Non-blocking gates**: Quality checks log warnings but never block operations.
- **TDD**: Write tests before implementation, run `uv run pytest` before commit.
- **Session ID**: All log entries include session_id for traceability.

## Testing Requirements

- **Unit tests**: Required for all new functions. Use `tests/conftest.py` fixtures (`tmp_db`, `cfg_dict`).
- **Integration tests**: Marked with `@pytest.mark.integration`, skipped in fast test runs.
- **Coverage target**: 84% overall, higher for new modules.
- **No mocking of database layer** — tests hit real SQLite (in-memory or temp file).
- **Mock strategies**: Use `unittest.mock.patch.object` for LLM calls, `monkeypatch` for env vars.

## Code Review Checklist

- [ ] Tests pass: `uv run pytest`
- [ ] Lint clean: `uv run ruff check .`
- [ ] No bare `except:` or `except Exception: pass`
- [ ] Logger.exception() in all API error handlers
- [ ] Atomic writes for all filesystem writes
- [ ] Config uses typed access, not raw dict
- [ ] No new dependencies without pyproject.toml entry
- [ ] CHANGELOG entry for user-facing changes
