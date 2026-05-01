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

1. **Parse** (`parser/mineru_parser.py`): MinerU CLI converts PDF → Markdown. Falls back to PyMuPDF markdown extraction. Chapter-filtered to high-signal sections (abstract, intro, methods, conclusion, etc.). PDFs >150 pages are split into chunks.

2. **Identify** (`dedup/resolver.py`): Resolve paper identity via priority chain: DOI → arXiv → S2 ID → OpenAlex ID → title+year fuzzy match. Creates or upgrades paper record.

3. **Extract** (`extractor/concept.py`, `extractor/llm_client.py`): LLM extracts structured concepts (Problem, Method, Conclusion, Debate, Gap, Actor) and typed arguments with `mechanism` and `section` fields. Uses a fallback chain across configured models. Prompt template lives in `prompts/extract_concepts.txt`.

   **PageIndex tree extraction** (`parser/pageindex_parser.py`): Documents are first structured into a tree (`tree.json` in Stage 2.5). `extract_concepts_from_tree()` then extracts per-leaf-node with the tree skeleton as LLM context, replacing flat `text[:8000]` truncation. Concurrency is capped via `asyncio.Semaphore(max_concurrent=3)`. Content quality gate (`_is_quality_content()`) filters short text and reference lists before LLM calls. After merge, `_link_cross_section_arguments()` links arguments sharing targets across sections.

4. **Validate** (`validator/schema.py`): **TBox** enforces which relations each concept type can use (e.g., Problem can `addresses`/`leaves_open`/`points_to`). **RBox** enforces transitivity, asymmetry, irreflexivity on specific relations. Pre-insertion TBox check via `validate_extraction()` in concept.py. Rejected items go to `validation.log`.

5. **Queue** (`extractor/queue.py`): Low-confidence concepts (< `weak_threshold`, default 0.7) are routed to `confidence_queue` table for manual review via `drbrain queue` / `drbrain queue resolve`.

6. **Align** (`extractor/canonical.py`): Canonical ID alignment — BM25 similarity match with LLM arbitration for ambiguous cases (score 0.3-0.8).

7. **Ingest** (`storage/database.py`): Inserts concepts, arguments (with `mechanism` and `section` fields), edges into SQLite. Basic SQLite wrapper with auto-schema init — no ORM.

8. **Expand** (`extractor/citation.py`): Fetches references and citations from Semantic Scholar / CrossRef / OpenAlex. Creates placeholder papers for external references.

9. **Closure** (`graph/engine.py`): NetworkX MultiDiGraph in-memory. Rule-based relationship inference (8 rules): `creates_debate`, `gap_addressed`, `indirect_evolution`, `gap_to_debate`, `shared_actor`, transitive closure, asymmetric detection, multi-hop path rules. Supports both full-graph and incremental (2-hop subgraph) closure. When `section_map` is provided, inferred edges get section-aware confidence via `propagate_confidence_with_section()`.

### Reasoning & Discovery Modules (post-ingestion)

- **Causal Chain** (`extractor/causal_chain.py`): Builds X→Y(via Z) chains from argument mechanism fields. DFS chain discovery sorts candidates by section adjacency (Introduction→Methods→Results→Discussion). `build_causal_chains()`, `find_chains_from()`, `find_path()`.
- **Confidence Propagation** (`extractor/confidence_propagation.py`): Multi-hop confidence decay (default 0.85 per hop), multi-path merging via probabilistic OR: `P = 1 - prod(1 - p_i)`. Section-aware variant: `propagate_confidence_with_section()` — Methods/Results decay 0.90, Discussion/Related Work 0.80.
- **Counterfactual Queries** (`extractor/counterfactual.py`): "What if X didn't exist?" — measures node removal impact on closure inferences. Section-weighted variant: `find_critical_nodes_weighted()`. `run_counterfactual()`, `find_critical_nodes()`.
- **Cross-domain Isomorphism** (`extractor/isomorphism.py`): Finds structurally similar subgraphs via relation signature Jaccard similarity. Section-aware signatures: `"in:supports@Methods"`. `find_similar_problems()`, `find_isomorphic_patterns()`.
- **Hypothesis Generation** (`extractor/hypothesis.py`): Generates research hypotheses from unaddressed gaps, debate zones, and technology cliffs. Evidence strings include section provenance. `detect_section_contradictions()` finds supports/challenges from different sections. `generate_hypotheses()`, `score_hypothesis()`.
- **Structure-first Retrieval** (`query/tree_retrieval.py`): PageIndex-style retrieval via `query --paper`. Reads tree skeleton → LLM selects relevant node_ids → loads content on-demand. Returns structured `[{"node_id", "title", "content"}]`.

### Key Design Points

- **Config**: `config.yaml` (checked in) overlayed by `config.local.yaml` (gitignored). Env var placeholders via `${VAR_NAME}` syntax. Deep-merge at the dict level.
- **LLM fallback chain**: `acall_with_fallback()` iterates through configured model list; returns first successful parse, `None` if all exhausted.
- **No vector embeddings**: BM25 (`query/bm25.py`) for search over concepts + arguments. No vector DB dependency.
- **Symbol-driven reasoning**: Graph closure rules, transitive closure, asymmetric detection, causal chains, confidence propagation, counterfactuals, isomorphism detection — all rule-based, zero embeddings.
- **Ecosystem enrichment**: CrossRef, Semantic Scholar, OpenAlex APIs for citation expansion, DOI resolution, author identity. Rate-limited with configurable cache TTL.
- **Graph-based discovery**: `detect_research_seeds()` finds stale problems, unaddressed gaps, debate zones, technology cliffs, cross-domain isomorphism, and confidence collapse patterns. `generate_hypotheses()` produces actionable research hypotheses from these patterns.
- **Section provenance**: `section` field flows from LLM extraction → DB → L1-L4 reasoning modules. Enables section-aware confidence decay, counterfactual weighting, isomorphism signatures, hypothesis evidence grounding, and contradiction detection.
- **Streamlit UI**: `drbrain serve` launches interactive graph visualization at `http://127.0.0.1:8501`.
- **Clean command**: `drbrain clean --force` removes all data files except PDFs.

### Testing

- Tests use pytest with `asyncio_mode = "auto"`.
- Integration tests are marked with `@pytest.mark.integration`; run with `-m "not integration"` to skip.
- Tests hit a real SQLite database (in-memory or temp file) — no mocking of the database layer.
- 542 tests total. TDD: tests-first for all new modules.

### Skills

Project skills in `skills/` directory. Available skills: `research-analysis` (knowledge frontier analysis), `paper-ingest`, `paper-query`, `citation-tracking`, `workspace-analysis`.

### Behavioral Guidelines

From Karpathy guidelines — bias toward caution over speed.

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
