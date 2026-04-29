# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Commands

```bash
uv sync                        # install all deps (including dev)
uv run drbrain <command>       # CLI entry point
uv run pytest                   # run all tests
uv run pytest tests/test_xxx.py             # single test file
uv run pytest tests/test_xxx.py::test_name  # single test case
uv run ruff check .             # lint
uv run ruff format .            # format
```

## Architecture

DrBrain is an **academic knowledge graph system** — vector-free, symbol-driven research discovery. It ingests PDFs, extracts structured concepts/arguments via LLM, deduplicates identities, and infers new relationships through rule-based graph closure.

### Ingestion Pipeline (9 stages)

1. **Parse** (`parser/mineru_parser.py`): MinerU CLI converts PDF → Markdown. Falls back to pypdfium2 text extraction. Chapter-filtered to high-signal sections (abstract, intro, methods, conclusion, etc.). PDFs >150 pages are split into chunks.

2. **Identify** (`dedup/resolver.py`): Resolve paper identity via priority chain: DOI → arXiv → S2 ID → OpenAlex ID → title+year fuzzy match. Creates or upgrades paper record.

3. **Extract** (`extractor/concept.py`, `extractor/llm_client.py`): LLM extracts structured concepts (Problem, Method, Conclusion, Debate, Gap, Actor) and typed arguments with mechanism fields. Uses a fallback chain across configured models. Prompt template lives in `prompts/extract_concepts.txt`.

4. **Validate** (`validator/schema.py`): **TBox** enforces which relations each concept type can use (e.g., Problem can `addresses`/`leaves_open`/`points_to`). **RBox** enforces transitivity, asymmetry, irreflexivity on specific relations. Rejected items go to `validation.log`.

5. **Queue** (`extractor/queue.py`): Low-confidence concepts (< `weak_threshold`, default 0.7) are routed to `confidence_queue` table for manual review via `drbrain queue` / `drbrain queue resolve`.

6. **Align** (`extractor/canonical.py`): Canonical ID alignment — BM25 similarity match with LLM arbitration for ambiguous cases (score 0.3-0.8).

7. **Ingest** (`storage/database.py`): Inserts concepts, arguments (with `mechanism` field for causal chains), edges into SQLite. Basic SQLite wrapper with auto-schema init — no ORM.

8. **Expand** (`extractor/citation.py`): Fetches references and citations from Semantic Scholar / CrossRef / OpenAlex. Creates placeholder papers for external references.

9. **Closure** (`graph/engine.py`): NetworkX MultiDiGraph in-memory. Rule-based relationship inference (8 rules): `creates_debate`, `gap_addressed`, `indirect_evolution`, `gap_to_debate`, `shared_actor`, transitive closure, asymmetric detection, multi-hop path rules. Supports both full-graph and incremental (2-hop subgraph) closure.

### Reasoning & Discovery Modules (post-ingestion)

- **Causal Chain** (`extractor/causal_chain.py`): Builds X→Y(via Z) chains from argument mechanism fields. `build_causal_chains()`, `find_chains_from()`, `find_path()`.
- **Confidence Propagation** (`extractor/confidence_propagation.py`): Multi-hop confidence decay (default 0.85 per hop), multi-path merging via probabilistic OR: `P = 1 - prod(1 - p_i)`.
- **Counterfactual Queries** (`extractor/counterfactual.py`): "What if X didn't exist?" — measures node removal impact on closure inferences. `run_counterfactual()`, `find_critical_nodes()`.
- **Cross-domain Isomorphism** (`extractor/isomorphism.py`): Finds structurally similar subgraphs via relation signature Jaccard similarity. `find_similar_problems()`, `find_isomorphic_patterns()`.
- **Hypothesis Generation** (`extractor/hypothesis.py`): Generates research hypotheses from unaddressed gaps, debate zones, and technology cliffs. `generate_hypotheses()`, `score_hypothesis()`.

### Key Design Points

- **Config**: `config.yaml` (checked in) overlayed by `config.local.yaml` (gitignored). Env var placeholders via `${VAR_NAME}` syntax. Deep-merge at the dict level.
- **LLM fallback chain**: `acall_with_fallback()` iterates through configured model list; returns first successful parse, `None` if all exhausted.
- **No vector embeddings**: BM25 (`query/bm25.py`) for search over concepts + arguments. No vector DB dependency.
- **Symbol-driven reasoning**: Graph closure rules, transitive closure, asymmetric detection, causal chains, confidence propagation, counterfactuals, isomorphism detection — all rule-based, zero embeddings.
- **Ecosystem enrichment**: CrossRef, Semantic Scholar, OpenAlex APIs for citation expansion, DOI resolution, author identity. Rate-limited with configurable cache TTL.
- **Graph-based discovery**: `detect_research_seeds()` finds stale problems, unaddressed gaps, debate zones, technology cliffs, cross-domain isomorphism, and confidence collapse patterns. `generate_hypotheses()` produces actionable research hypotheses from these patterns.
- **Streamlit UI**: `drbrain serve` launches interactive graph visualization at `http://127.0.0.1:8501`.

### Testing

- Tests use pytest with `asyncio_mode = "auto"`.
- Integration tests are marked with `@pytest.mark.integration`; run with `-m "not integration"` to skip.
- Tests hit a real SQLite database (in-memory or temp file) — no mocking of the database layer.
- 401 tests total (356 base + 45 from L1-L4 features). TDD: tests-first for all new modules.
