# CLAUDE.md

## Quick Commands

```bash
uv sync                                  # install all deps
uv run drbrain <command>                 # CLI entry point
uv run pytest                            # all tests
uv run pytest -m "not integration"       # skip slow integration tests
uv run pytest tests/test_xxx.py::name    # single test
uv run ruff check . && uv run ruff format .
uv run pytest --cov=drbrain --cov-report=term
```

Commands: `setup`, `ingest`, `build`, `query`, `graph`, `analyze`, `citations`, `ws`, `export`, `backup`, `check`, `audit`, `seed`, `closure`, `repair`, `import`, `translate`, `clean`, `ask`, `index`, `show`, `fetch`, `embed`, `reason`, `evolve`, `descendants`, `landscape`.

## Architecture

DrBrain is a **vector-free, symbol-driven academic knowledge graph**. Ingest PDFs → extract concepts/arguments via LLM → deduplicate → infer new edges via rule-based closure.

### Pipeline

**Ingest** (`drbrain ingest`): PDF→markdown (MinerU CLI, fallback pymupdf4llm). 5-source cross-validation (arXiv, CrossRef, S2, OpenAlex, DeepXiv) for metadata + venue (journal/publisher/citation_count). LLM tree-structures markdown → `tree.json`. Status: `uploaded`.

**Build** (`drbrain build [id...]`): 5-stage LLM extraction — ontology extension → entity extraction (10-way concurrent) → relation extraction → coreference → refinement (`--skip-refine` to skip). Status: `extracted`.

### Key Modules

| Area         | Key files                                                                                                                                                | What                                                                                              |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Graph engine | `graph/engine.py`, `graph/embedding.py`                                                                                                                  | TransE embeddings, rule closure (8+4 rules), hybrid scoring, t-norm grounding                     |
| Extraction   | `extractor/concept.py`, `extractor/reasoner.py`                                                                                                          | 5-stage LLM extraction, bidirectional LLM↔KG reasoning                                            |
| Reasoning    | `extractor/causal_chain.py`, `extractor/confidence_propagation.py`, `extractor/counterfactual.py`, `extractor/isomorphism.py`, `extractor/hypothesis.py` | Causal chains, confidence decay, counterfactuals, cross-domain isomorphism, hypothesis generation |
| Search       | `query/bm25.py`, `query/tree_retrieval.py`                                                                                                               | BM25 over concepts+arguments; PageIndex tree-search                                               |
| Quality      | `services/audit.py`, `services/repair.py`                                                                                                                | 15 audit rules, metadata enrichment via OpenAlex                                                  |
| Import       | `services/zotero_import.py`, `services/translate.py`                                                                                                     | Zotero/BibTeX/Endnote import, LLM translation with resume                                         |
| Storage      | `storage/database.py`, `storage/export.py`, `storage/workspace.py`                                                                                       | SQLite WAL + schema versions, BibTeX/RIS export, workspace CRUD                                   |
| CLI          | `cli/commands.py`, `cli/graph_commands.py`, `cli/main.py`                                                                                                | Typer CLI, graph traversal, KGQA (`ask`)                                                          |

### Data Layout

```
data/
├── spool/inbox/        PDFs awaiting ingest
├── spool/pending/      Failed ingests
├── papers/<id>/        source.pdf, raw.md, tree.json, images/
├── drbrain.db          SQLite (WAL mode, schema_versions)
├── metrics.db          LLM token tracking
├── cache/              API cache (rebuildable)
└── reports/            Per-paper JSON
workspace/<name>/       workspace.yaml + refs/papers.json
```

### Design Points

- **Config**: `config.py` typed dataclass hierarchy. `config.yaml` + `config.local.yaml` (gitignored). Env var `${VAR_NAME}` resolution. Sub-configs support dict-style `[]` access.
- **Logging/Metrics**: loguru + `get_session_id()` (UUID4), `ui()` for user output. SQLite metrics with WAL + thread-safety, `timer()` / `timed()`.
- **API clients**: `requests.Session` + `urllib3.Retry` on 429/5xx. MinerU exponential backoff.
- **LLM**: `acall_with_fallback()` iterates model list in config; any litellm provider.
- **No vectors**: BM25 search, rule-based reasoning. Zero embedding dependency for core discovery.
- **Section provenance**: `section` field flows from LLM extraction → DB → all reasoning layers (confidence decay, counterfactuals, etc.).
- **Atomic writes**: tmp→rename pattern throughout. `storage/paths.py` for centralized paths.

### Testing

- pytest, `asyncio_mode = "auto"`. Real SQLite (in-memory/temp), no DB mocking.
- `@pytest.mark.integration` on slow tests. `-m "not integration"` to skip.

### Gotchas

- **Editable install**: `uv pip install -e .` once after `uv sync` if `ModuleNotFoundError: No module named 'drbrain'`.
- **typer OptionInfo**: In tests, typer `Option` defaults appear as `OptionInfo` objects — use `isinstance(param, typer.models.OptionInfo)` to extract `.default`.

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

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool                        | Use when                                               |
| --------------------------- | ------------------------------------------------------ |
| `detect_changes`            | Reviewing code changes — gives risk-scored analysis    |
| `get_review_context`        | Need source snippets for review — token-efficient      |
| `get_impact_radius`         | Understanding blast radius of a change                 |
| `get_affected_flows`        | Finding which execution paths are impacted             |
| `query_graph`               | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes`     | Finding functions/classes by name or keyword           |
| `get_architecture_overview` | Understanding high-level codebase structure            |
| `refactor_tool`             | Planning renames, finding dead code                    |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.
