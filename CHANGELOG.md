# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Conventional Commits](https://www.conventionalcommits.org/).

## [Unreleased]

### Added
- **Graph search — directed traversal**: `query --neighbors` now uses `GraphEngine.traverse()` with `--relation` (comma-separated edge type filter) and `--direction` (forward/backward/both) flags. Graph expansion returns concept nodes (Problem/Method/Gap/etc.) with full path trace, not just paper neighbors.
- **Graph search — direct queries**: `drbrain graph neighbors <node>` traverses graph without BM25 text search. `drbrain graph path <src> <dst>` finds shortest path with edge direction/recovery from MultiDiGraph.
- **Closure filtering**: `drbrain closure --rule <name>` (repeatable, 11 rules supported) and `--dry-run` (read-only, does not persist).
- **Multi-paper concept analysis**: `drbrain graph related <id...>` with 3 modes — `concepts` (SQL label intersection + coverage), `graph` (1-hop neighbor intersection via traverse), `edges` (shared relation-target patterns).
- **Hybrid ranking**: `drbrain query --hybrid` applies multiplicative PageRank boost [1.0, 2.0] to re-rank BM25 results by graph centrality. Pure Python PageRank, no scipy dependency.
- **Config example**: `config.example.yaml` with 9 LLM provider templates (OpenAI, Anthropic, DeepSeek, Zhipu, Bailian, MiniMax, Moonshot, Ollama, vLLM) — uncomment the one you need.
- **Extraction concurrency**: `extract.max_concurrent` in config.yaml controls parallel LLM calls during concept extraction (default 10)
- **Library management**: Inbox auto-classification (paper/thesis/preprint/book/review/document), spool/pending queue, workspace CRUD (`drbrain ws`), BibTeX/RIS/Markdown export, tar.gz backup, delete with `--rm-files`
- **Citation graph**: Shared-reference analysis (`drbrain citations --type shared-refs`), citation verification against library (`drbrain check-citations`), citation_cache table with S2 write-through
- **Knowledge frontier analysis**: `drbrain analyze` command orchestrating seeds, causal chains, counterfactual, hypotheses, and isomorphism detection
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

### Fixed
- PDF parsing: replaced pypdfium2 with PyMuPDF (fitz); use `pymupdf4llm` for markdown extraction with proper heading/table structure; plain text fallback
- LLM client: 60s timeout prevents indefinite hangs; `drbrain check` now tests LLM API connectivity
- Ingest: PDF removed from inbox after successful ingest (was left behind)
- `seed_cmd` dict key access: `seed['node']`→`seed['concept']`, `seed['signal']`→`seed['description']`
- `test_closure_cmd_backward_compat`: insufficient test data (single extends edge produces no inferred edges; use 3-node transitive chain)
- `clean_cmd`: targeted individual DB/metrics files instead of entire `data/` directory
- `check_cmd`: now creates missing directories instead of just reporting them
- `setup_cmd`: upgraded from config-only wizard to full env initializer (config + dirs + validation + readiness summary). `--quick` flag for non-interactive mode. Validate-only mode when config exists

## [v0.1.0] - 2026-04-28

Initial release of DrBrain — academic knowledge graph system (vector-free, symbol-driven).

### Added
- PDF parsing via MinerU with pypdfium2 fallback
- Semantic Scholar citation expansion with OpenAlex/CrossRef fallbacks
- BM25 + LLM hybrid concept alignment (SmartAligner) with stopwords normalization
- TBox/RBox schema validation for academic ontology
- Confidence queue with auto_accept/weak_threshold routing
- Boundary detection patterns (stale_problem, debate_zone, etc.)
- Graph traversal queries (--neighbors, BM25 confidence filtering)
- CLI commands: ingest, query, stats, dedup, validate, queue resolve, delete
- Web UI via Streamlit
- File-based API cache with TTL
- Transaction rollback on ingestion failure
- CrossRef DOI enrichment from title/arXiv
- Batch queue resolution with type filtering

### Changed
- Package renamed from brbrain to drbrain
- Citation placeholders use batch insert per batch

### Fixed
- pypdfium2 fallback for PDF parsing
- CLI command compatibility with typer
- BM25 search now indexes paper abstracts
- OpenAlex fallback when S2 returns 429/no data
- Duplicate get_concept_evolution definition
- Ambiguous variable names flagged by linter
