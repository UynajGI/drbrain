# Directory Structure

## Directory Layout

```
src/drbrain/
├── cli/             # Typer CLI: commands.py (~3000 lines), graph_commands.py, main.py, setup.py, dependencies.py
├── config.py        # Typed Config dataclass hierarchy (LLC, MinerU, API, Dirs, DB, Extract, BM25, Queue)
├── log.py           # Loguru setup, get_session_id(), ui()
├── metrics.py       # SQLite metrics store with WAL, timer(), timed(), session_id tracking
├── exceptions.py    # DrBrainError hierarchy
├── dedup/           # Paper identity resolution (DOI, arXiv, S2, OpenAlex, title+year)
├── extractor/       # LLM-powered extraction + reasoning modules
│   ├── llm_client.py        # Async LLM fallback chain (acall_with_fallback)
│   ├── concept.py           # Entity/relation extraction pipeline
│   ├── argument.py          # Causal argument units
│   ├── reasoner.py          # LLM agent tool-calling + bidirectional reasoning
│   ├── rule_miner.py        # Embedding-driven path rule mining
│   ├── openalex.py          # OpenAlex API client (requests.Session + Retry)
│   ├── crossref.py          # CrossRef API client
│   ├── citation.py          # Multi-source citation expansion
│   └── ...
├── graph/           # Knowledge graph engine
│   ├── engine.py            # NetworkX graph, closure rules, BFS traversal
│   ├── embedding.py         # TransE entity/relation embeddings
│   ├── query_embeddings.py  # Embedding-based complex query operators
│   └── ...
├── parser/          # PDF parsing and document tree extraction
│   ├── mineru_parser.py     # MinerU CLI + PyMuPDF fallback, PDF pre-validation
│   └── pageindex_parser.py  # Markdown→tree, TOC verification, node summaries
├── query/           # BM25 search + tree retrieval
├── report/          # Knowledge frontier analysis reports
├── services/        # Domain services
│   ├── audit.py             # Data quality scan (15 rules)
│   ├── graph_to_text.py     # Subgraph→NL LLM description
│   ├── repair.py            # Metadata auto-fix via CrossRef/arXiv
│   ├── translate.py         # Placeholder-protected chunk translation with resume
│   └── ...
├── storage/         # SQLite database + paths
│   ├── database.py          # Schema, migrations, CRUD
│   ├── paths.py             # Centralized path accessors
│   └── workspace.py         # Workspace management (create/add/remove/rename)
├── validator/       # TBox/RBox knowledge graph schema validation
└── ...
```

## Module Organization

- **CLI commands**: One function per command in `cli/commands.py`, registered in `main.py`.
- **Graph commands**: Subcommands in `cli/graph_commands.py` under `graph_app` Typer instance.
- **Services**: Pure logic, no CLI code. Accept config/DB as parameters.
- **Parser**: PDF→markdown→tree pipeline, no CLI knowledge.
- **Extractor**: LLM callers + reasoning, imported by CLI and services.
- **New feature?** If it's user-facing, add to `cli/`. If it's logic, add to `services/` or `extractor/`.

## Naming Conventions

- Module files: `snake_case.py` (`graph_commands.py`, `llm_client.py`).
- Internal helpers: `_underscore_prefix()` for private functions.
- Command functions: `{name}_cmd()` (`ingest_cmd`, `query_cmd`).

## Examples

- Well-organized service: `src/drbrain/services/translate.py` (chunking, retry, resume, concurrency).
- Well-organized CLI: `src/drbrain/cli/main.py` (app registration, callback hook).
- Path accessor pattern: `src/drbrain/storage/paths.py` (centralized `paper_dir()`, `raw_md_path()`, etc.).
