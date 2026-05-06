# Test Hardening — Fix Failures + Smoke Tests + Coverage Gate

## A. Fix 14 pre-existing test failures (easiest → hardest)

### Easy: Mock path fixes (5 tests)
- `tests/test_batch_ingest.py` (4 tests): `drbrain.cli.commands.extract_concepts` → `drbrain.extractor.concept.extract_concepts`
- `tests/test_incremental_closure.py::test_ingest_auto_closure`: same mock path
- Find all with: `grep -rn "drbrain.cli.commands.extract_concepts" tests/`

### Medium: Analyzer seed tests (3 tests)
- `test_analyze_paper_basic`, `test_summary_counts_match_arrays`, `test_seeds_limited_to_10`
- Root cause: fake graph has insufficient edges for `detect_research_seeds` to find anything
- Fix: add enough edges to the test fixtures to produce seeds. Check what `detect_research_seeds` expects — likely a certain minimum graph density.

### Medium: Parser helper tests (5 tests)
- `test_fetch_arxiv_metadata_success`, `test_fetch_arxiv_metadata_single_title`, `test_fallback_pymupdf_empty_markdown_uses_text`, `test_parser_full_extract_flow_with_fallback`, `test_extract_single_with_mineru_success`
- Root cause: mocks don't properly intercept the actual function calls (import path changed)
- Fix: check where the functions being mocked are actually called from now, update mock paths

### Easy: MinerU test (1 test)
- `test_parser_succeeds_on_first_try`: assertion issue
- Fix: check actual vs expected, adjust assertion or mock

## B. Smoke test

Create `tests/test_smoke.py` — doesn't need external APIs, runs fast:

```python
def test_cli_help():
    """drbrain --help works"""
    ...

def test_database_creates():
    """Database creates tables correctly"""
    ...

def test_config_loads():
    """Config loads without errors"""
    ...

def test_auth_roundtrip():
    """Password hash → verify works"""
    ...
```

## C. Coverage gate in CI

Add `--cov-fail-under=65` to pytest in `.github/workflows/ci.yml`.

## Acceptance
- 0 test failures
- Smoke tests pass
- CI enforces 65% coverage minimum
