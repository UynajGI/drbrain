# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Conventional Commits](https://www.conventionalcommits.org/).

## [Unreleased] — dev/feature

### Added
- **Structured reasoning workflow engine**: `extractor/session_agent.py` — 7 built-in workflows (review, gap-analysis, impact, compare, frontier, lineage, paradigm). Workflow orchestrator with step-level result caching. CLI via `drbrain reason --workflow`.
- **Workflow visualizer**: Pipeline diagrams and result summaries for reasoning workflows.
- **Batch-fetch command**: `drbrain batch-fetch` — process DOI/URL lists in bulk with progress tracking.
- **Graph export formats**: GraphML, JSON-LD, and Cypher export via `drbrain graph export --format graphml|jsonld|cypher`.
- **Backup restore**: `drbrain restore` — restore from tar.gz backups.
- **Standalone search command**: `drbrain search` — quick BM25 keyword search independent of graph-aware `query`.
- **Session management CLI**: `drbrain session new/ask/chat/list/delete/export` — persistent DB-backed session CRUD.
- **HTTP retry decorator**: `http_retry` with exponential backoff in `services/http_utils.py`.
- **Workflow-level result caching**: Non-deterministic queries skip cache; temperature=0 results cached.

### Changed
- **Parser splits**: `mineru_parser.py` → `parser/mineru/` subpackage; `pageindex_parser.py` → `parser/pageindex/` subpackage.
- **Graph engine split**: `engine.py` split into `engine.py` (core), `engine_closure.py` (rule inference), `engine_embeddings.py` (embedding validation), `query_embeddings.py` (complex queries).
- **Genealogy subpackage**: `genealogy.py` (1011 lines) → `graph/genealogy/` — `lineage.py`, `paradigm.py`, `landscape.py`, `transfer.py`, `display.py`.
- **Concept extraction subpackage**: `concept.py` → `extractor/concept/` with `pipeline.py`, `dedup.py`, `merge.py`, `tree_helpers.py`, `types.py`.
- **`open_db()` context manager**: CLI modules migrated to shared `open_db()` for DB lifecycle management, eliminating ~100 lines of boilerplate.
- **Stats consolidation**: `Database.get_stats()` centralizes statistics queries.

### Performance
- **LLM response caching**: `call_with_messages` / `acall_with_messages` now check ApiCache before calling LLM. Cache disabled for `temperature > 0`.
- **search_tree filtering**: `search_tree()` accepts optional `paper_id` to avoid full-table BLOB scan.
- **DOI enrichment parallelization**: ThreadPoolExecutor for 5-source metadata resolution.

### Changed
- **LLM client consolidation**: sync and async call paths unified in `extractor/llm_client.py`. `_base_url` helper for provider-specific URL normalization.
- **Reasoning base refactoring**: `_run_async()` helper for safe event-loop handling. Workflow cache key now fingerprints graph + DB state.

### Fixed
- `acall_with_fallback` / `acall_with_messages`: async LLM calls now properly synced with litellm event loop handling.
- **DeepSeek compat**: `base_url` normalization for OpenAI-compatible providers (DeepSeek expects no trailing `/v1`).
- `reasoner` model fallback: respects config's model list order when primary fails.
- `backup_cmd --list`: tolerates missing `ctx.obj`.
- `reason_cmd`: normalized `OptionInfo` for all options.
- `metrics_panel`: migrated to `connect_wal()` for thread-safe DB access.
- Workflow steps: standardized error handling across all 7 workflows (errors set result to `None`, don't halt pipeline).

