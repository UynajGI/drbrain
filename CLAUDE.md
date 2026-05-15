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

Commands: `setup`, `ingest`, `build`, `query`, `graph`, `analyze`, `citations`, `check-citations`, `ws`, `export`, `backup`, `check`, `audit`, `seed`, `closure`, `repair`, `import`, `translate`, `clean`, `ask`, `index`, `show`, `fetch`, `embed`, `reason`, `evolve`, `descendants`, `landscape`, `paradigm`, `transfers`, `isomorphism`, `difficulty`, `frontier`, `report`, `list`, `stats`, `queue`, `delete`, `lineage`.

Sub-apps: `graph` (neighbors, path, related, describe, query, traverse-from), `ws` (create, add, remove, list, show, delete, rename). `queue` has subcommands `resolve` and `resolve-all`.

## Architecture

DrBrain is a **symbol-driven academic knowledge graph with lightweight vector retrieval**. Ingest PDFs → extract concepts/arguments via LLM → deduplicate → infer new edges via rule-based closure. Vectors used only for semantically-complete tree nodes, never arbitrary chunks.

### Pipeline

**Ingest** (`drbrain ingest`): PDF→markdown (MinerU CLI, fallback pymupdf4llm). 5-source cross-validation (arXiv, CrossRef, S2, OpenAlex, DeepXiv) for metadata + venue (journal/publisher/citation_count). LLM tree-structures markdown → `tree.json`. Status: `uploaded`.

**Build** (`drbrain build [id...]`): 5-stage LLM extraction — ontology extension → entity extraction (10-way concurrent) → relation extraction → coreference → refinement (`--skip-refine` to skip). Status: `extracted`.

### Key Modules

| Area         | Key files                                                                                                                                                | What                                                                                              |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Graph engine | `src/drbrain/graph/engine.py`, `src/drbrain/graph/embedding.py`, `src/drbrain/graph/genealogy.py`                                                                                                                  | TransE embeddings (learn_embeddings, entity_embedding, predict_link, similar_entities), rule closure (8+4 rules), hybrid scoring, concept lineage/landscape/paradigm detection                     |
| Extraction   | `src/drbrain/extractor/concept.py`, `src/drbrain/extractor/agent.py`, `src/drbrain/extractor/reasoner.py`, `src/drbrain/extractor/raptor.py`                                                                                  | 5-stage LLM extraction (agent-based), bidirectional LLM↔KG reasoning, RAPTOR recursive semantic tree            |
| Reasoning    | `src/drbrain/extractor/causal_chain.py`, `src/drbrain/extractor/confidence_propagation.py`, `src/drbrain/extractor/counterfactual.py`, `src/drbrain/extractor/isomorphism.py`, `src/drbrain/extractor/hypothesis.py` | Causal chains, confidence decay, counterfactuals, cross-domain isomorphism, hypothesis generation |
| Search       | `src/drbrain/query/bm25.py`, `src/drbrain/query/tree_retrieval.py`                                                                                                               | BM25 over concepts+arguments; PageIndex tree-search + RAPTOR two-stage traversal (layer descent + collapsed fallback)                                               |
| Embedding    | `src/drbrain/services/embedding.py`                                                                                                                                  | Tree node embeddings (sentence-transformers), FAISS cosine search, GPU batch auto-tuning, post_filter, multi-source download (ModelScope+HuggingFace), provider=none grace  |
| Quality      | `src/drbrain/services/audit.py`, `src/drbrain/services/repair.py`                                                                                                                | 15 audit rules, metadata enrichment via OpenAlex                                                  |
| Import       | `src/drbrain/services/zotero_import.py`, `src/drbrain/services/translate.py`                                                                                                     | Zotero/BibTeX/Endnote import, LLM translation with resume                                         |
| Storage      | `src/drbrain/storage/database.py`, `src/drbrain/storage/export.py`, `src/drbrain/storage/workspace.py`                                                                                       | SQLite WAL + schema versions, BibTeX/RIS export, workspace CRUD                                   |
| CLI          | `src/drbrain/cli/main.py` (registration), `src/drbrain/cli/commands.py` (re-exports), `src/drbrain/cli/_common.py`, `src/drbrain/cli/{ingest,query,export,check,ws,repair,build,analysis,graph}_commands.py`, `src/drbrain/cli/setup.py`, `src/drbrain/cli/dependencies.py` | Typer CLI, graph traversal, KGQA (`ask`), setup validation                                                          |

### Data Layout

```
data/
├── spool/inbox/        PDFs awaiting ingest
├── spool/pending/      Failed ingests
├── papers/<id>/        source.pdf, raw.md, tree.json, images/
├── drbrain.db          SQLite (WAL mode, schema_versions)
├── metrics.db          LLM token tracking (created on first use)
├── cache/              API cache (rebuildable)
├── logs/               loguru rotating logs
├── backups/            tar.gz exports
└── reports/            Per-paper JSON
workspace/<name>/       workspace.yaml + refs/papers.json
```

### Design Points

- **Config**: `src/drbrain/config.py` typed dataclass hierarchy. `config.yaml` + `config.local.yaml` (gitignored). Env var `${VAR_NAME}` resolution. Sub-configs support dict-style `[]` access.
- **Logging/Metrics**: loguru + `get_session_id()` (UUID4), `ui()` for user output. SQLite metrics with WAL + thread-safety, `timer()` / `timed()`.
- **API clients**: `requests.Session` + `urllib3.Retry` on 429/5xx. MinerU exponential backoff.
- **LLM**: `acall_with_fallback()` iterates model list in config; any litellm provider.
- **Lightweight vectors**: Vectors for semantically-complete tree nodes only (PageIndex sections, RAPTOR summaries). Stored in `tree_vectors` table with FAISS. `provider=none` disables — pure BM25 + LLM navigation. Never chunk-level embedding. Reference: ScholarAIO embedding engine.
- **Section provenance**: `section` and `node_id` fields flow from LLM extraction → DB → all reasoning layers (confidence decay, counterfactuals, etc.). `node_id` links back to PageIndex tree nodes.
- **DB tables**: concepts, arguments, edges, aliases, embeddings, tree_vectors, tree_summaries, vector_metadata, papers, paper_ids, confidence_queue, citation_cache, research_seeds, build_stages, schema_versions.
- **Atomic writes**: tmp→rename pattern throughout. `src/drbrain/storage/paths.py` for centralized paths.

### Testing

- pytest, `asyncio_mode = "auto"`. Real SQLite (in-memory/temp), no DB mocking.
- `@pytest.mark.integration` on slow tests. `-m "not integration"` to skip.

### Gotchas

- **Editable install**: `uv pip install -e .` once after `uv sync` if `ModuleNotFoundError: No module named 'drbrain'`.
- **typer OptionInfo**: In tests, typer `Option` defaults appear as `OptionInfo` objects — use `isinstance(param, typer.models.OptionInfo)` to extract `.default`.

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
