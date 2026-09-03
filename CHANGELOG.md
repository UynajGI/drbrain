# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Conventional Commits](https://www.conventionalcommits.org/).

## [Unreleased] — dev/feature

### Added
- **WebUI** (`src/drbrain/webui/`, `drbrain webui`): local research workbench over the existing CLI capabilities — BM25 search, RAG ask, autoresearch loop (start a run from a research goal, watch ledger events / claims live), compute jobs and data/plugin assets. Standard-library HTTP server, one static page, no new dependencies.
- **LlamaIndex RAG layer** (`src/drbrain/rag/`): BM25 + vector + tree retrieval with RRF fusion, rerank, FunctionAgent tool calling, and an eval harness (retriever metrics / ragas / semantic / qagen). CLI: `drbrain rag index` (incremental index build), `drbrain rag eval`, and `drbrain hybrid` (fused retrieval).
- **Concept graph subsystem** (`src/drbrain/concept_graph/`, `drbrain cg …`): corpus-scale concept co-occurrence graph with ingest/build/embed/neighbors/map/predict/recommend; UMAP interactive map export; leakage-free yearly prediction snapshots.
- **Research loop** (`src/drbrain/loop/`, PR #14): multi-agent orchestration closed loop (workflow nodes + analyst/critic/compute/verifier roles + discussion layer). _Note: this subsystem is under heavy active development upstream; internals will change._
- **sqlite-vec ANN index layer**: `tree_vectors_vec` virtual tables with optional int8 quantization for corpus-scale vector search over tree nodes.
- **CLI**: `drbrain hybrid`, `drbrain survey` (one-shot markdown literature survey), `drbrain check-citations`, `drbrain ingest-link`; `cg` and `rag` sub-apps.
- **Schema migrations v9–v15**: `concept_graph`, `concept_node_columns`, `concept_epistemic`, `knowledge_snapshots`, `answer_records`, `evidence`, `claims`.

### Changed
- **RAG engine consolidation**: LlamaIndex is the sole hybrid-retrieval engine (legacy `hybrid_search` path removed); `drbrain ask` / `hybrid` route through it.
- **Extraction robustness**: per-key rate-limit state machine and fallback chains in `extractor/llm_client.py`.

### Fixed
- Eval qagen migrated to LlamaIndex `DatasetGenerator`; assorted ruff/mypy/CI fixes.

_Entries above summarize commits between `0.1.0a3` and the current `main`; per-commit details live in `git log`._

## [0.1.0a3] — 2026-06-19

### Added
- **Structured reasoning workflow engine**: `extractor/session_agent.py` — 7 built-in workflows (review, gap-analysis, impact, compare, frontier, lineage, paradigm). Workflow orchestrator with step-level result caching. CLI via `drbrain reason --workflow`.
- **Workflow visualizer**: Pipeline diagrams and result summaries for reasoning workflows.
- **Batch-fetch command**: `drbrain batch-fetch` — process DOI/URL lists in bulk with progress tracking.
- **Graph export formats**: GraphML, JSON-LD, and Cypher export via `drbrain graph export --format graphml|jsonld|cypher`.
- **OKF export**: `drbrain export-okf <bundle>` — export the knowledge graph as an Open Knowledge Format v0.1 markdown bundle (directory tree + YAML frontmatter + markdown cross-links). Human- and agent-readable; `git clone`/`cat` consumable. Supports `--paper` / `--workspace` subgraph filtering.
- **Backup restore**: `drbrain restore` — restore from tar.gz backups.
- **Standalone search command**: `drbrain search` — quick BM25 keyword search independent of graph-aware `query`.
- **Session management CLI**: `drbrain session new/ask/chat/list/delete/export` — persistent DB-backed session CRUD.
- **HTTP retry decorator**: `http_retry` with exponential backoff in `services/http_utils.py`.
- **Workflow-level result caching**: Non-deterministic queries skip cache; temperature=0 results cached.
- **Incremental update system**: schema v8 migration adds `updated_at` to `papers`/`concepts`/`edges`; `Database` gains `get_dirty_papers` / `get_papers_since` / `get_last_run` / `set_last_run` / `touch_paper` for change tracking. `build`, `closure`, `embed` (TransE), and `index` (BM25) are now incremental-by-default with stage watermarks. `pipeline` defaults to incremental; use `--full` to force the legacy full-rebuild behavior.
- **Centralized Database write methods**: 13 new methods (`set_external_id`, `insert_citation_cache`, `set_paper_field`, `set_paper_type`, `delete_concept`, `redirect_edge_endpoint`, `accept_queue_by_label`, `upsert_build_stage`, `merge_papers`, `insert_agent_session`/`soft_delete_session`/`touch_session`/`insert_agent_message`) so application-layer code never writes raw SQL.

### Changed
- **Parser splits**: `mineru_parser.py` → `parser/mineru/` subpackage; `pageindex_parser.py` → `parser/pageindex/` subpackage.
- **Graph engine split**: `engine.py` split into `engine.py` (core), `engine_closure.py` (rule inference), `engine_embeddings.py` (embedding validation), `query_embeddings.py` (complex queries).
- **Genealogy subpackage**: `genealogy.py` (1011 lines) → `graph/genealogy/` — `lineage.py`, `paradigm.py`, `landscape.py`, `transfer.py`, `display.py`.
- **Concept extraction subpackage**: `concept.py` → `extractor/concept/` with `pipeline.py`, `dedup.py`, `merge.py`, `tree_helpers.py`, `types.py`.
- **`open_db()` context manager**: CLI modules migrated to shared `open_db()` for DB lifecycle management, eliminating ~100 lines of boilerplate.
- **Stats consolidation**: `Database.get_stats()` centralizes statistics queries.
- **LLM client consolidation**: sync and async call paths unified in `extractor/llm_client.py`. `_base_url` helper for provider-specific URL normalization.
- **Reasoning base refactoring**: `_run_async()` helper for safe event-loop handling. Workflow cache key now fingerprints graph + DB state.
- **SQL write centralization**: 35 raw SQL writes (INSERT/UPDATE/DELETE) outside `database.py` routed through centralized `Database` methods — `citation.py` (13), `repair.py` (9), `db_ingest.py` (8), `session_agent.py` (4), `dedup.py` (3), `agent.py` (2), `queue.py` (1). `Database` is now the sole write surface.
- **`closure_cmd`** new `--incremental/--full` flag (default `--incremental`): runs rules on the 2-hop neighborhood of changed concepts instead of the full graph. Wires up the previously dead `GraphEngine.closure_incremental()`.
- **`index_cmd`** now incremental: skips rebuild when no paper changed since the last index run (`--rebuild` forces full).
- **`build_cmd`** default selection now includes papers touched since the last build run (not just `status == 'uploaded'`), and records `set_last_run('build')`.
- **`delete_paper`** now removes edges by `source_paper` (was incorrectly matching `src_id`/`dst_id`, which hold concept labels), touches neighbor papers sharing the deleted concepts, and clears closure/embed/index watermarks.

### Performance
- **LLM response caching**: `call_with_messages` / `acall_with_messages` now check ApiCache before calling LLM. Cache disabled for `temperature > 0`.
- **search_tree filtering**: `search_tree()` accepts optional `paper_id` to avoid full-table BLOB scan.
- **DOI enrichment parallelization**: ThreadPoolExecutor for 5-source metadata resolution.
- **Incremental TransE training**: `TransE.train_incremental(graph, new_edges, ...)` trains only on edges from dirty papers with a shortened epoch budget, preserving all existing entity/relation vectors (the old path cleared and retrained everything). Relations are now warm-started instead of discarded.
- **Incremental pipeline**: adding one paper to an N-paper library no longer forces N LLM extractions — only the new paper builds, closure scans its neighborhood, embeddings micro-adjust.

### Fixed
- `acall_with_fallback` / `acall_with_messages`: async LLM calls now properly synced with litellm event loop handling.
- **DeepSeek compat**: `base_url` normalization for OpenAI-compatible providers (DeepSeek expects no trailing `/v1`).
- `reasoner` model fallback: respects config's model list order when primary fails.
- `backup_cmd --list`: tolerates missing `ctx.obj`; accepts both `Config` dataclass and raw dict config.
- `backup_cmd --json`: output order fixed — no longer emits header text before the JSON payload.
- `reason_cmd`: normalized `OptionInfo` for all options.
- `metrics_panel`: migrated to `connect_wal()` for thread-safe DB access.
- Workflow steps: standardized error handling across all 7 workflows (errors set result to `None`, don't halt pipeline).
- **`GraphEngine.__init__`**: now initializes `self._transE = None` (regression from the engine split caused `AttributeError` on first embedding/closure-hybrid access; affected 8 tests).
- **`index_cmd` TypeError**: was calling `build_bm25_index(db, force=rebuild)` but the signature is `(db, k1, b)` — every `drbrain index` invocation crashed.
- **`uspto_ppubs._ensure_session`**: first `opener.open(req1)` now wrapped in try/except so session-establishment failures raise `PpubsError` instead of leaking raw `HTTPError`.
- **`citation.expand_citations_multi` counts**: `refs_added`/`citing_added` no longer increment before the DB write succeeds (previously inflated on failure); `except Exception` narrowed to `(sqlite3.IntegrityError, sqlite3.OperationalError)`.
- **`_merge_papers` atomicity**: the 6 separate UPDATEs/DELETEs are now a single transaction (`Database.merge_papers`), preventing torn state (concepts migrated but source paper intact) on mid-sequence failure.
- **`repair.py` updated_at**: repair writes now bump `updated_at` via `set_paper_field` — previously repaired papers were invisible to the incremental-update system.
- **`delete_paper` edge deletion**: matches `source_paper` instead of `src_id`/`dst_id` (which hold concept labels, not paper ids).

### Tests
- **Incremental update coverage**: 19 tests in `test_incremental_updates.py` (schema v8 migration, change-tracking queries, `delete_paper` neighbor touching, `TransE.train_incremental`, `closure_incremental`, centralized write methods).
- **OKF export coverage**: 12 tests in `test_okf_export.py` (slugify, frontmatter conformance, cross-link rendering, arguments, paper export, index, filtering, broken-link tolerance).
- **batch_ingest test rewrite**: mocked at `_ingest_single_paper` boundary instead of only `extract_pdf`, so tests no longer hang on unmocked network/async calls.

