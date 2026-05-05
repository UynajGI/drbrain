# Data Quality Pipeline

## Context
scholaraio has 15+ rule library audit, 7-stage ingest quality gates, PDF pre-validation.
DrBrain has none. This adds three layers of data quality assurance.

## Requirements

### T1: drbrain audit (new command)
- `src/drbrain/services/audit.py` — 15 rules, severity-graded
- Rules: missing_title, missing_md, missing_doi, missing_abstract, missing_year, missing_journal, missing_authors, short_md, empty_tree, low_concept_count, no_edges, placeholder_status, old_placeholder, unresolved_env, duplicate_title
- CLI: `drbrain audit [--severity] [--workspace] [--json]`

### T2: PDF pre-validation
- `_validate_pdf()` in mineru_parser.py
- Check: page count > 0, not encrypted, openable by PyMuPDF
- Insert before MinerU CLI call, fall through to pymupdf on failure

### T3: Ingest quality gates
- Gate 1: Markdown size > 200 bytes
- Gate 2: Title + year + external ID present
- Gate 3: >= 1 concept + >= 1 edge after build
- All non-blocking (warn but save)

### T4: Tests
- Each audit rule tested
- PDF validation tests
- Ingest quality gate tests

## Success Criteria
- `drbrain audit` produces report
- PDF validation catches corrupt PDFs
- All tests pass, no regressions
