# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Conventional Commits](https://www.conventionalcommits.org/).

## [Unreleased]

### Added
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
- Missing Python package check in `drbrain check` updated from pypdfium2 to pymupdf

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
