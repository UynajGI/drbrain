# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Commands

```bash
uv sync                        # install all deps (including dev)
uv run drbrain <command>       # CLI entry point
uv run pytest                   # run all tests
uv run pytest -m "not integration"  # skip slow integration tests
uv run pytest tests/test_xxx.py::test_name  # single test case
uv run ruff check .             # lint
uv run ruff format .            # format
uv run pytest --cov=drbrain --cov-report=term  # coverage report
```

Key user commands: `setup`, `ingest`, `query`, `graph`, `analyze`, `citations`, `ws`, `export`, `backup`, `check`, `seed`, `closure`, `repair`, `import`, `translate`, `clean`.

## Architecture

DrBrain is an **academic knowledge graph system** — vector-free, symbol-driven research discovery. It ingests PDFs, extracts structured concepts/arguments via LLM, deduplicates identities, and infers new relationships through rule-based graph closure.

### Ingestion Pipeline (2-phase)

**Phase 1 — `drbrain ingest`**: Lightweight PDF-to-library. No concept extraction.

1. **Parse** (`parser/mineru_parser.py`): MinerU CLI → Markdown. Falls back to `pymupdf4llm.to_markdown()`. PDFs >150 pages split into chunks.
2. **Identify** (`dedup/resolver.py`, `mineru_parser.py:_resolve_metadata`): 5-source cross-validation (arXiv, CrossRef, S2, OpenAlex, DeepXiv). Stores title, year, doi, arxiv, s2_id, openalex_id in `paper_ids`. Extracts abstract from tree.json.
3. **Tree** (`parser/pageindex_parser.py`): LLM structures markdown into `tree.json` with section summaries.
4. **Record**: Insert paper with status `uploaded`.

**Phase 2 — `drbrain build [paper_id...]`**: 5-stage LLM graph extraction.

1. **Ontology Extension**: LLM suggests domain-specific subcategories under 6 TBox types.
2. **Entity Extraction**: Per leaf-node concept extraction with subcategory labels. 10-way concurrency.
3. **Relation Extraction**: LLM connects concepts using TBox relations.
4. **Coreference Resolution**: LLM merges duplicate entity labels.
5. **Iterative Refinement**: LLM self-reviews extraction for contradictions and errors (skippable via `--skip-refine`).

Paper statuses: `uploaded` → `extracted` (after build). `placeholder` for citation-only papers.

### Reasoning & Discovery Modules (post-ingestion)

- **Causal Chain** (`extractor/causal_chain.py`): Builds X→Y(via Z) chains from argument mechanism fields. DFS chain discovery sorts candidates by section adjacency (Introduction→Methods→Results→Discussion). `build_causal_chains()`, `find_chains_from()`, `find_path()`.
- **Confidence Propagation** (`extractor/confidence_propagation.py`): Multi-hop confidence decay (default 0.85 per hop), multi-path merging via probabilistic OR: `P = 1 - prod(1 - p_i)`. Section-aware variant: `propagate_confidence_with_section()` — Methods/Results decay 0.90, Discussion/Related Work 0.80.
- **Counterfactual Queries** (`extractor/counterfactual.py`): "What if X didn't exist?" — measures node removal impact on closure inferences. Section-weighted variant: `find_critical_nodes_weighted()`. `run_counterfactual()`, `find_critical_nodes()`.
- **Cross-domain Isomorphism** (`extractor/isomorphism.py`): Finds structurally similar subgraphs via relation signature Jaccard similarity. Section-aware signatures: `"in:supports@Methods"`. `find_similar_problems()`, `find_isomorphic_patterns()`.
- **Hypothesis Generation** (`extractor/hypothesis.py`): Generates research hypotheses from unaddressed gaps, debate zones, and technology cliffs. Evidence strings include section provenance. `detect_section_contradictions()` finds supports/challenges from different sections. `generate_hypotheses()`, `score_hypothesis()`.
- **Structure-first Retrieval** (`query/tree_retrieval.py`): PageIndex-style retrieval via `query --paper`. Returns structured `[{"node_id", "title", "content"}]`.
- **Graph-Enhanced Search** (`query` + `graph/engine.py`): `query --neighbors` for directed graph expansion from BM25 results. `query --hybrid` applies multiplicative PageRank boost [1.0, 2.0] to re-rank results by graph centrality. Returns concept nodes with `_via_graph`, `_source_seed`, `_distance`, `_path` fields.
- **Citation Graph** (`storage/citation_graph.py`): Shared-reference analysis, ref/citing/shared-refs queries. `find_shared_refs()` detects papers sharing references but not citing each other (knowledge frontier signal).
- **Citation Verification** (`extractor/citation_check.py`): Extracts (Author, Year) patterns from text and matches against the local library.
- **Library Management** (`storage/inbox.py`, `storage/workspace.py`, `storage/export.py`, `storage/backup.py`): Inbox scanning (with pending queue), workspace CRUD, BibTeX/RIS/Markdown export, tar.gz backup.
- **Paper Type Detection** (`extractor/detection.py`): Heuristic + LLM classification into paper/review/thesis/preprint/book/document.
- **Knowledge Frontier Analysis** (`report/analyzer.py`): Orchestrates all reasoning modules into unified report via `drbrain analyze`.
- **Metadata Repair** (`services/repair.py`): Auto-fix paper metadata via CrossRef/arXiv APIs. Title normalization, missing DOI resolution, author/journal backfill.
- **Zotero Import** (`services/zotero_import.py`): Import papers from Zotero SQLite databases and BibTeX `.bib` files.
- **Paper Translation** (`services/translate.py`): LLM-powered paper translation with section-aware chunking.
- **Graph Query** (`cli/graph_commands.py`): Direct graph traversal without BM25. `drbrain graph neighbors <node>` traverses with direction/relation filtering via `GraphEngine.traverse()`. `drbrain graph path <src> <dst>` finds shortest path using `nx.shortest_path()` on undirected copy, recovers edge direction/relation from original directed graph. `drbrain closure` supports `--rule` (filter by inference rule name) and `--dry-run` (read-only, no DB persist). `drbrain graph related <paper_id...>` analyzes shared concepts across papers in 3 modes: `concepts` (SQL label intersection), `graph` (1-hop neighbor intersection via traverse), `edges` (shared edge patterns).
- **Logging** (`log.py`, `metrics.py`): loguru-based structured logging with rotating files. SQLite-backed LLM token usage tracking in `data/metrics.db`.

### Data Directory Layout

```
data/
├── spool/inbox/       # PDFs awaiting ingest (auto-classified)
├── spool/pending/     # Failed ingests + pending.jsonl
├── papers/<id>/       # Per-paper: source.pdf, raw.md, images/, tree.json
├── drbrain.db          # SQLite database
├── metrics.db          # LLM token tracking
├── backups/            # tar.gz backups
├── cache/             # API cache (rebuildable)
├── logs/              # validation.log
└── reports/           # Per-paper JSON reports
workspace/<name>/      # Paper subsets: workspace.yaml + refs/papers.json
```

### Key Design Points

- **Config**: `config.yaml` (checked in, all non-secret settings) overlayed by `config.local.yaml` (gitignored, secrets only — api_key, token, email). Env var placeholders via `${VAR_NAME}` syntax. Deep-merge at the dict level. `config.example.yaml` has 9 LLM provider templates.
- **LLM fallback chain**: `acall_with_fallback()` iterates through configured model list in `config.local.yaml`; first successful parse wins, `None` if all exhausted. Supports any litellm provider (OpenAI, Anthropic, Ollama, plus OpenAI-compatible endpoints like DeepSeek/Zhipu/Bailian).
- **No vector embeddings**: BM25 (`query/bm25.py`) for search over concepts + arguments. No vector DB dependency.
- **Symbol-driven reasoning**: Graph closure rules, transitive closure, asymmetric detection, causal chains, confidence propagation, counterfactuals, isomorphism detection — all rule-based, zero embeddings.
- **Ecosystem enrichment**: `arxiv` library for arXiv metadata; CrossRef API (`crossref.py`) for DOI resolution and cross-validation; `pyalex` library for OpenAlex title search and author identity; Semantic Scholar (with API key support). Rate-limited with configurable cache TTL.
- **Graph-based discovery**: `detect_research_seeds()` finds stale problems, unaddressed gaps, debate zones, technology cliffs, cross-domain isomorphism, and confidence collapse patterns. `generate_hypotheses()` produces actionable research hypotheses from these patterns.
- **Section provenance**: `section` field flows from LLM extraction → DB → L1-L4 reasoning modules. Enables section-aware confidence decay, counterfactual weighting, isomorphism signatures, hypothesis evidence grounding, and contradiction detection.
- **Streamlit UI**: `drbrain serve` launches interactive graph visualization at `http://127.0.0.1:8501`.
- **Clean command**: `drbrain clean --force` removes all data files except PDFs.

### Testing

- Tests use pytest with `asyncio_mode = "auto"`.
- Integration tests are marked with `@pytest.mark.integration`; run with `-m "not integration"` to skip.
- Tests hit a real SQLite database (in-memory or temp file) — no mocking of the database layer.
- 1094 tests total. TDD: tests-first for all new modules. 84% coverage.

### Gotchas

- **Editable install**: After `uv sync`, run `uv pip install -e .` once if `ModuleNotFoundError: No module named 'drbrain'` appears. The src-layout package needs an editable install for imports to resolve.
- **typer OptionInfo in tests**: When calling command functions directly (not via CLI), typer `Option` default values appear as `OptionInfo` objects. Use `isinstance(param, typer.models.OptionInfo)` to extract `.default`. All commands use the `_resolve_workspace_papers()` or equivalent normalization.

### Skills

Project skills in `skills/` directory. Available skills: `research-analysis` (knowledge frontier analysis), `paper-ingest`, `paper-query`, `citation-tracking`, `workspace-analysis`.

## Behavioral Guidelines

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
