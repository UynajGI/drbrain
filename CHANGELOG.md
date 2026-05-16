# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Conventional Commits](https://www.conventionalcommits.org/).

## [0.1.0a1] — 2026-05-16

### Added
- **openai-compat embedding provider**: `_embed_batch_openai_compat()` calls any OpenAI-compatible `/v1/embeddings` API. Config via `embed.api_base` + `embed.api_key`. Retry with exponential backoff on 429/5xx. Chunked batching by `embed.batch_size`. `provider` setting now fully supports `"openai-compat"` alongside `"local"` and `"none"`.
- **GraphEngine embedding persistence**: `learn_embeddings(dim, epochs, lr)`, `entity_embedding(label)`, `predict_link(head, relation, top_k)`, `similar_entities(label, top_k)` on GraphEngine. TransE vectors persisted to `embeddings` table; hybrid closure reuses persistent embeddings instead of inline training. (2202.07412-inspired)
- **KG closure edges in LLM context**: `ask`/`reason` commands now inject closure-inferred edges (`--[inferred: rel]-->` format) into LLM context, with top-k confidence limiting. `_build_closure_context()` helper in `cli/_common.py`. (2306.08302 F1-inspired)
- **RAPTOR two-stage tree traversal**: `tree_traversal_search()` — layer-by-layer top-k descent with collapsed tree fallback, per RAPTOR Figure 2. More token-efficient than flat cosine across all layers for deep trees.
- **Embedding GPU batch auto-tuning**: `_compute_batch_size()`, `_estimate_mem_per_sample()`, GPU memory profiling with cached results. Automatic optimal batch size selection on CUDA devices.
- **Embedding post_filter**: `_post_filter()` — score threshold and empty-text filtering on `search_tree()` results.
- **Multi-source model download**: `_resolve_model_path()` — ModelScope primary + HuggingFace fallback for embedding models. (ScholarAIO vectors.py-inspired)
- **`evolve --stats` flag**: Temporal evolution signal classification (emerging/established/declining/contested/resurging) and year-by-year counts with trend indicators (growing/declining/stable). New `get_concept_signal()` and `get_concept_evolution()` methods in `Database` class.

### Changed
- **CLI module split**: `cli/commands.py` (4331 lines) split into 8 focused modules — `ingest_commands.py`, `query_commands.py`, `export_commands.py`, `check_commands.py`, `ws_commands.py`, `repair_commands.py`, `build_commands.py`, `analysis_commands.py`. Shared helpers extracted to `cli/_common.py`. `cli/commands.py` retained as backward-compatible re-export shim. `ws_commands.py` uses `@ws_app.command()` decorator pattern matching `graph_commands.py`. All existing imports and tests remain compatible.

### Removed
- **Dead code**: `_embed_signature` (services/embedding.py), `expand_citations_oa` (extractor/citation.py), `_pick_main_pdf` (services/zotero_import.py). None had any callers.
- **`timeline` command**: Removed `drbrain timeline`. Functionality superseded by `drbrain evolve --stats`.

### Added
- **Tree-graph provenance (Layer 1)**: `node_id` column on concepts and arguments tables linking back to PageIndex tree nodes. New tables: `tree_vectors` (per-node embeddings), `tree_summaries` (RAPTOR recursive summaries), `vector_metadata` (signature tracking). Migration v4 auto-applies on next `Database()` init. `insert_concept()` and `insert_argument()` accept optional `node_id` parameter.
- **Embedding engine (Layer 2)**: `EmbedConfig` in config.py with provider (local/openai-compat/none), default model Qwen3-Embedding-0.6B. `services/embedding.py` — `build_tree_vectors()` for tree node embeddings with content_hash incremental updates, `search_tree()` for cosine similarity retrieval. `provider=none` gracefully disables all vectors. Schema: `tree_vectors` table with BLOB embeddings, `vector_metadata` for signature tracking. Pattern: ScholarAIO vectors.py.
- **RAPTOR recursive semantic tree (Layer 3)**: `extractor/raptor.py` — recursive embedding→UMAP→GMM+BIC clustering→LLM summarization on PageIndex leaf nodes. Builds multi-layer summary tree in `tree_summaries` with `source_node_ids` provenance chains. New dependencies: numpy, scikit-learn, umap-learn.
- **PageIndex provenance Layer 2 injection**: All 5 knowledge genealogy features (landscape, evolve, descendants, paradigm, transfers) carry PageIndex provenance via `_get_concept_provenance()`. Text tree and Mermaid renderers show section/paper provenance inline.
- **Knowledge genealogy CLI**: `drbrain evolve` (concept lineage trees), `drbrain descendants` (academic offspring tracking), `drbrain landscape` (domain timeline with gaps/debates/technology cliffs), `drbrain paradigm` (paradigm shift detection via paper-age analysis), `drbrain transfers` (cross-domain method migration via workspace clustering).
- **Cross-domain isomorphism**: `drbrain isomorphism` finds structurally similar concepts across domains via subgraph relation signature + Jaccard similarity + label similarity scoring. RAPTOR cross-section context enrichment via `enrich_isomorphisms_with_raptor`.
- **Difficulty map**: `drbrain difficulty` classifies knowledge gaps by originating section type (introduction, methods, results, etc.) via section semantics analysis, with composite difficulty scoring.
- **Knowledge frontier report**: `drbrain frontier` combines research seeds, debate zones, technology cliffs, difficulty scores, and confidence collapse patterns into a single composite report.
- **Adaptive plateau detection**: Iterative convergence detection (`_is_plateau_reached`) replaces fixed-threshold iteration termination in ontology extension for smarter LLM cost control.
- **PageIndex-RAPTOR-tree full pipeline integration**: `build_paper_tree_vectors` bridges PageIndex embeddings + RAPTOR recursive summaries. `query_by_structure_hybrid` provides LLM-primary tree navigation with optional vector pre-filtering. ReasonerAgent `get_raptor_summaries` tool exposes cross-section summaries to LLM agent. `drbrain embed --tree` executes both layers in one pass.
- **KG reasoning enhancement**: TransE-based complex query answering with ∧,∨,¬ operators (`graph query` command, `drbrain graph query`). LLM↔KG bidirectional iterative reasoning with TBox/RBox validation feedback loop (`--bidirectional` flag). Embedding-driven path rule mining from TransE relations (`drbrain closure --mine-rules`). LLM-powered subgraph-to-text description (`drbrain graph describe`).
- **Data quality pipeline**: Full-library audit with 15 severity-graded rules (`drbrain audit`). PDF pre-validation (encryption/corruption check via PyMuPDF) before MinerU. 3 non-blocking ingest quality gates (markdown size, metadata completeness, extraction quality).
- **PageIndex TOC verification**: LLM-based section title position verification with auto-correction loop (max 2 retries). Inspired by PageIndex's verify_toc + fix_incorrect_toc.
- **Engineering hardening T1-T10**: Typed `Config` dataclass with env var resolution (T1). Session-aware logging with `get_session_id()`/`ui()` (T2). Metrics with WAL/thread-safety/timer/timed + dead code removal (T3). Custom exception hierarchy + `logger.exception()` audit across 5 API modules (T4). Shared `conftest.py` test fixtures — `tmp_db`, `cfg_dict` (T5). API clients upgraded to `requests.Session` with `urllib3.Retry` exponential backoff (T6). Schema-versioned migrations + WAL + centralized path accessors `storage/paths.py` (T7). CLI config cached once in typer.Context (T8). `cli/dependencies.py` — `check_import_error()` with install hints + atomic write sweep (T9). Docs sync (T10).
- **Translate refactor**: Placeholder-protected chunk splitting (code blocks, LaTeX math, images preserved across chunk boundaries). Heuristic language detection (CJK + Latin stopwords for de/fr/es). Concurrent chunk translation via ThreadPoolExecutor. Workdir-based state persistence with resume-from-interruption. Exponential backoff retry with timeout subdivision. Terminology annotation rules for zh/ja/ko.
- **Workspace hardening**: `validate_workspace_name()` prevents path traversal. Atomic writes (tmp→rename) for `refs/papers.json`. `schema_version: 1` in workspace.yaml. `ws rename` command.
- **Venue metadata enrichment**: Ingest now fetches journal, publisher, and citation_count from OpenAlex, CrossRef, S2, and DeepXiv APIs. Stored in papers table for complete BibTeX/RIS export. Placeholder papers upgraded on ingest also receive updated venue metadata.
- **Cross-paper concept dedup**: automatic exact+similar label merging after `drbrain build`. Word-overlap similarity detection. Based on 2511.11017 ontology-driven approach.
- **3-layer KG reasoning stack**: TransE embeddings (`drbrain embed`), hybrid closure (`drbrain closure --mode hybrid`), LLM agent reasoning (`drbrain reason`). Based on 2202.07412, 2306.08302, 2511.11017.
- **Pipeline refactor**: Two-phase. `drbrain ingest` (lightweight) + `drbrain build` (5-stage extraction). Based on 2306.08302/2511.11017.
- **Graph search — directed traversal**: `query --neighbors` now uses `GraphEngine.traverse()` with `--relation` (comma-separated edge type filter) and `--direction` (forward/backward/both) flags. Graph expansion returns concept nodes (Problem/Method/Gap/etc.) with full path trace, not just paper neighbors.
- **Graph search — direct queries**: `drbrain graph neighbors <node>` traverses graph without BM25 text search. `drbrain graph path <src> <dst>` finds shortest path with edge direction/recovery from MultiDiGraph.
- **Closure filtering**: `drbrain closure --rule <name>` (repeatable, 11 rules supported) and `--dry-run` (read-only, does not persist).
- **Multi-paper concept analysis**: `drbrain graph related <id...>` with 3 modes — `concepts` (SQL label intersection + coverage), `graph` (1-hop neighbor intersection via traverse), `edges` (shared relation-target patterns).
- **Hybrid ranking**: `drbrain query --hybrid` applies multiplicative PageRank boost [1.0, 2.0] to re-rank BM25 results by graph centrality. Pure Python PageRank, no scipy dependency.
- **Metadata cross-validation**: `_resolve_metadata` cross-checks 5 sources — arXiv, CrossRef, S2, OpenAlex, DeepXiv (TLDR + keywords + citations). Title+year consistency, text-year anchor. Stores doi, s2_id, openalex_id. Abstract from tree.json.
- **Extraction concurrency**: `extract.max_concurrent` in config.yaml controls parallel LLM calls during concept extraction (default 10)
- **Library management**: Inbox auto-classification (paper/thesis/preprint/book/review/document), spool/pending queue, workspace CRUD (`drbrain ws`), BibTeX/RIS/Markdown export, tar.gz backup, delete with `--rm-files`
- **Citation graph**: Shared-reference analysis (`drbrain citations --type shared-refs`), citation verification against library (`drbrain check-citations`), citation_cache table with S2 write-through
- **Knowledge frontier analysis**: `drbrain analyze` with 4 selection modes, LLM executive summary + seed solution suggestions + cross-paper method migration detection
- **PageIndex tree-based ingestion**: TOC fallback (header → PDF outline → LLM segmentation), tree validation/repair, concurrent leaf-node extraction, content quality gate, cross-section argument linking
- **Section-aware reasoning**: TBox validation in extract, section contradiction detection, section-aware confidence propagation in graph closure
- **Check command enhancements**: Library stats, disk space monitoring, MinerU API connectivity test, parser path recommendation
- **PDF parser fallback**: Replaced pypdfium2 with PyMuPDF (fitz) for richer markdown extraction
- **Agent skills**: 5 knowledge frontier skills (research-analysis, paper-ingest, paper-query, citation-tracking, workspace-analysis)
- **Metadata repair**: `drbrain repair` auto-fixes titles, years, DOIs, authors via CrossRef/arXiv
- **Zotero import**: `drbrain import zotero` and `drbrain import bibtex` for external library migration
- **Logging & metrics**: loguru structured logging with rotating files, SQLite LLM token tracking
- **Paper translation**: `drbrain translate` via LLM with section-aware chunking
- Pre-commit hook: ruff check + format on staged Python files
- Commit-msg hook: enforce conventional commit message format
- Pre-push hook: run tests before pushing to main
- Prepare-commit-msg: auto-generate commit template from staged changes

### Changed
- PDF parser fallback: pypdfium2 → PyMuPDF (fitz)
- Default ingest path: `data/inbox/` → `data/spool/inbox/`
- Data directory layout: renamed inbox, added spool/pending, workspace/, backups/
- CLI: `expand` command replaced by `citations`
- CLI: `serve` command (Streamlit UI) removed — not a current priority

### Fixed
- Fix: `isomorphism_cmd` now wires `enrich_isomorphisms_with_raptor` — RAPTOR cross-section context was implemented in extraction layer but not called from CLI (fields always empty before).
- Fix: `build_raptor_tree` reads existing PageIndex vectors from `tree_vectors` DB table instead of re-embedding identical nodes — eliminates 2x embedding compute for papers with RAPTOR trees.
- Import: journal and citation_count now passed to `insert_paper()` when importing from BibTeX/Zotero
- Repair: journal repairs from CrossRef now written to DB (was returned in list but not applied)
- Repair: reports "Paper not found" error instead of silently producing 0 repairs
- PDF parsing: replaced pypdfium2 with PyMuPDF (fitz); use `pymupdf4llm` for markdown extraction with proper heading/table structure; plain text fallback
- LLM client: 60s timeout prevents indefinite hangs; `drbrain check` now tests LLM API connectivity
- Ingest: PDF removed from inbox after successful ingest (was left behind)
- TBox schema expanded: Method gains `supports/challenges/limits/constrains`; Conclusion gains `extends`; all types accept `cross_section_support/cross_section_challenge`. Validation rejections downgraded to WARNING.
- `seed_cmd` dict key access: `seed['node']`→`seed['concept']`, `seed['signal']`→`seed['description']`
- `test_closure_cmd_backward_compat`: insufficient test data (single extends edge produces no inferred edges; use 3-node transitive chain)
- `clean_cmd`: targeted individual DB/metrics files instead of entire `data/` directory
- `check_cmd`: creates missing directories; tests LLM API connectivity; tests MinerU CLI presence; PyMuPDF fallback warning only when no MinerU path available
- `check_cmd`: fallback directory paths updated from `data/inbox` to `data/spool/inbox` (stale reference from before restructure)
- `citations`: multi-source expansion (OpenAlex+S2+CrossRef), placeholder papers for new refs/citing, auto-upgrade on later ingest, configurable `--limit`/`--sort`
- `embed`: incremental training by default — new entities initialized randomly, existing ones warm-started from DB. `--retrain` for full rebuild.
- `_link_cross_section_arguments`: no longer creates edges to fake nodes; information preserved as debug log only
- `setup` / `check`: DeepXiv token (data.rag.ac.cn) + S2 API key (semanticscholar.org) registration prompts. Ingest exports deepxiv_token to environment for library use.
- `main.py`: fixed `brbrain` → `drbrain` import
- `setup_cmd`: upgraded from config-only wizard to full env initializer (config + dirs + validation + readiness summary). `--quick` flag for non-interactive mode. Validate-only mode when config exists

