# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Conventional Commits](https://www.conventionalcommits.org/).

## [Unreleased]

### Added
- Pre-commit hook: ruff check + format on staged Python files
- Commit-msg hook: enforce conventional commit message format
- Pre-push hook: run tests before pushing to main
- Prepare-commit-msg: auto-generate commit template from staged changes

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
