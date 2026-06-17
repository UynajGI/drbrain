# Contributing

## Codebase Tour

```
src/drbrain/
├── cli/                  # Typer CLI commands
│   ├── main.py           # App definition, thin command registration
│   ├── commands.py       # Backward-compatible re-exports from split modules
│   ├── _helpers/         # Shared CLI utilities (db, ingest helpers, display)
│   ├── _common.py        # Shared private helpers
│   ├── ingest_commands.py    # ingest, fetch, batch-fetch, citations, closure, report
│   ├── query_commands.py     # query, search, index, list, show, stats, seed
│   ├── export_commands.py    # export, backup, restore, delete, queue, lineage
│   ├── check_commands.py     # check, audit, analyze, clean
│   ├── ws_commands.py        # workspace CRUD (typer sub-app)
│   ├── repair_commands.py    # repair, import
│   ├── build_commands.py     # build, embed, translate
│   ├── analysis_commands.py  # ask, reason, evolve, descendants, landscape, paradigm, transfers, isomorphism, difficulty, frontier
│   ├── graph_commands.py     # graph subcommands (neighbors, path, related, describe, query, traverse-from, export)
│   ├── session_commands.py   # session subcommands (new, ask, chat, list, delete, export)
│   ├── setup.py          # Setup wizard
│   ├── _setup_i18n.py     # Bilingual setup (EN/ZH)
│   └── dependencies.py   # Import check helpers
├── extractor/            # LLM extraction, reasoning, and API clients
│   ├── concept/          # 5-stage graph extraction pipeline (subpackage)
│   │   ├── pipeline.py   # Main pipeline orchestration
│   │   ├── dedup.py      # Concept deduplication
│   │   ├── merge.py      # Cross-paper concept merging
│   │   ├── tree_helpers.py # Tree-to-graph conversion helpers
│   │   └── types.py      # Extraction type definitions
│   ├── agent.py          # LLM agent base class
│   ├── agent_tools.py    # Shared tool definitions (TOOL_DEFINITIONS, kg_validate)
│   ├── session_agent.py  # Persistent DB-backed SessionAgent for multi-turn reasoning
│   ├── reasoner.py       # Stateless ReasonerAgent with tool-calling
│   ├── raptor.py         # RAPTOR recursive semantic tree summarization
│   ├── llm_client.py     # acall_with_fallback(), litellm wrappers
│   ├── openalex.py       # OpenAlex API client
│   ├── crossref.py       # CrossRef API client
│   ├── cache.py          # API response cache
│   ├── causal_chain.py   # Causal chain reasoning
│   ├── confidence_propagation.py # Multi-hop confidence decay
│   ├── counterfactual.py # Node removal impact analysis
│   ├── isomorphism.py    # Cross-domain subgraph similarity
│   ├── hypothesis.py     # Hypothesis generation
│   ├── rule_miner.py     # Embedding-driven path rule mining
│   ├── citation.py       # Citation expansion (OpenAlex + S2 + CrossRef)
│   ├── citation_check.py # In-text citation verification
│   ├── detection.py      # Paper type classification
│   ├── argument.py       # Argument unit extraction and validation
│   ├── canonical.py      # Label normalization + SmartAligner for dedup
│   └── queue.py          # Confidence queue resolution
├── graph/                # Graph engine, embeddings, and reasoning
│   ├── engine.py         # GraphEngine: load, save, traverse, neighborhoods
│   ├── engine_closure.py # Symbolic rule closure (8+4 rules) + hybrid scoring
│   ├── engine_embeddings.py # Embedding-grounded validation for inferred edges
│   ├── embedding.py      # TransE training and link prediction
│   ├── query_embeddings.py # Complex query operators (project, intersect, union, negate)
│   ├── path_reasoning.py # Hybrid tree+graph path reasoning
│   └── genealogy/        # Knowledge genealogy subpackage
│       ├── lineage.py    # Concept evolution trees + descendants
│       ├── paradigm.py   # Paradigm shift detection
│       ├── landscape.py  # Domain landscape: gaps, debates, cliffs
│       ├── transfer.py   # Cross-domain method transfer discovery
│       └── display.py    # Text tree + Mermaid rendering
├── parser/               # PDF parsing and content structuring
│   ├── mineru_parser.py  # Thin re-export wrapper
│   ├── mineru/           # MinerU CLI + PyMuPDF fallback (subpackage)
│   │   ├── parser.py     # PDF → Markdown conversion
│   │   ├── fallback.py   # PyMuPDF fallback path
│   │   └── metadata.py   # Multi-source metadata resolution
│   ├── pageindex_parser.py # Thin re-export wrapper
│   └── pageindex/        # LLM tree structuring (subpackage)
│       ├── builder.py    # Tree construction from markdown
│       ├── summary.py    # LLM node summarization
│       ├── validation.py # Tree validation and repair
│       └── retrieval.py  # Content retrieval
├── query/                # Search and retrieval
│   ├── bm25.py           # BM25 index over concepts + arguments
│   └── tree_retrieval.py # PageIndex tree search (adaptive depth)
├── providers/            # External service clients
│   ├── webtools.py       # Web extraction (qt-web-extractor)
│   ├── uspto_odp.py      # USPTO ODP patent API (key required)
│   └── uspto_ppubs.py    # USPTO PPUBS client (free, session-based)
├── storage/              # Database and I/O
│   ├── database.py       # SQLite DB with WAL, schema_versions, CRUD
│   ├── paths.py          # Centralized path accessors
│   ├── export.py         # BibTeX, RIS, Markdown export
│   ├── workspace.py      # Workspace CRUD
│   ├── backup.py         # tar.gz + rsync backup
│   ├── inbox.py          # Inbox scanning and pending queue
│   ├── citation_graph.py # Citation graph queries
│   ├── proceedings.py    # Conference proceedings registry
│   ├── graph_export.py   # GraphML / JSON-LD / Cypher export
│   ├── connection.py     # WAL connection helper
│   └── explore.py        # Literature discovery silos (JSONL)
├── services/             # Higher-level services
│   ├── audit.py          # 15-rule data quality scan
│   ├── repair.py         # Metadata repair via APIs
│   ├── enrich.py         # CrossRef backfill + scrub detection
│   ├── translate.py      # LLM paper translation
│   ├── zotero_import.py  # Zotero/BibTeX/Endnote import
│   ├── graph_to_text.py  # Subgraph-to-text LLM description
│   ├── citation_styles.py # APA/Vancouver/Chicago/MLA + custom
│   ├── document.py       # Office doc inspection (DOCX/PPTX/XLSX)
│   ├── fsearch.py        # Federated search (local + arXiv)
│   ├── pipeline.py       # Step chaining with presets
│   ├── metrics_panel.py  # User behavior analytics
│   └── parser_benchmark.py # PDF parser comparison harness
├── report/               # Analysis reports
│   ├── analyzer.py       # Knowledge frontier analyzer
│   └── generator.py      # Report generation utilities
├── config.py             # Typed Config dataclass
├── log.py                # loguru-based structured logging
├── metrics.py            # LLM token usage tracking
└── exceptions.py         # DrBrainError hierarchy

tests/                    # pytest suite (real SQLite, no DB mocking)
├── test_ingest.py
├── test_build.py
├── test_query.py
├── test_graph_engine.py
├── test_embedding.py
├── test_closure.py
├── test_export.py
├── test_workspace.py
├── test_citations.py
├── test_analyze.py
├── test_audit.py
├── test_translate.py
├── test_repair.py
└── ...

skills/                   # AgentSkills.io skills (27 total)
├── paper-ingest/
├── paper-query/
├── citation-tracking/
├── research-analysis/
├── workspace-analysis/
├── show/
├── fsearch/
├── patent-search/
├── explore/
├── proceedings/
├── pipeline/
├── enrich/
├── metrics/
├── document/
├── citation-styles/
├── backup/
├── ingest-link/
├── audit/
├── export/
├── translate/
├── graph/
├── import/
└── index/
```

## Development Setup

```bash
# Clone and install
git clone https://github.com/UynajGI/DrBrain.git
cd DrBrain
uv sync
uv pip install -e .

# Set up DrBrain config
drbrain setup --quick

# Verify
drbrain check
```

Key development commands:

```bash
uv run pytest                          # all tests
uv run pytest -m "not integration"     # skip slow integration tests
uv run pytest tests/test_xxx.py::test_name  # single test
uv run ruff check .                    # lint
uv run ruff format .                   # format
uv run pytest --cov=drbrain --cov-report=term  # coverage
```

## How to Add a CLI Command

### 1. Create the command function

Add a new function in the appropriate module (e.g., `src/drbrain/cli/commands.py` for core commands, `src/drbrain/cli/graph_commands.py` for graph subcommands). Use the project's typer pattern:

```python
def my_command(
    ctx: typer.Context,
    arg1: str = typer.Argument(..., help="Description"),
    flag1: bool = typer.Option(False, "--flag1", help="Description"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """One-line docstring describing what the command does."""
    cfg = ctx.obj["config"]
    db = Database(cfg["db"]["path"])

    # ... implementation ...

    db.close()

    if json_output:
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        typer.echo("Human-readable output")
```

Key conventions:
- Accept `ctx: typer.Context` as the first parameter
- Always access config via `ctx.obj["config"]`
- Always support `--json` for machine-readable output
- Always close the database connection before returning
- Use `typer.Exit(1)` for errors (not `sys.exit()`)

### 2. Register in `cli/main.py`

```python
# At the top of main.py, import your function
from drbrain.cli.commands import my_command

# Register with the app
app.command("my-command")(my_command)
```

### 3. Add tests

Create or extend a test file in `tests/`:

```python
def test_my_command():
    db = Database(":memory:")
    # ... seed test data ...
    # Call the command function directly
    my_command(ctx_with_config, arg1="test")
    # ... assert results ...
```

### 4. Add to `docs/cli-reference.md`

Document the command with its description, flags, and examples. Follow the existing format.

## How to Add a Reasoning Module

### 1. Create the module in `extractor/`

```python
# src/drbrain/extractor/my_module.py

def analyze_my_pattern(graph, db, **kwargs) -> list[dict]:
    """Analyze something in the knowledge graph."""
    results = []
    for node in graph.graph.nodes:
        # ... analysis ...
        results.append({"node": node, "finding": ...})
    return results
```

### 2. Wire into the analyzer

In `src/drbrain/report/analyzer.py`, import and call your module:

```python
from drbrain.extractor.my_module import analyze_my_pattern

def analyze_paper(db, graph, paper_id, full=False, models=None):
    # ... existing analysis ...
    if full:
        my_results = analyze_my_pattern(graph, db)
        report["my_findings"] = my_results
    return report
```

### 3. Add tests with real graph fixtures

```python
def test_my_module():
    db = Database(":memory:")
    graph = GraphEngine()
    # Seed test data with known patterns
    db.insert_concept("p1", "Problem", "Overfitting", 0.9)
    db.insert_concept("p1", "Method", "Dropout", 0.9)
    graph.load_from_db(db)
    results = analyze_my_pattern(graph, db)
    assert len(results) > 0
```

### 4. Document in `docs/architecture.md`

Add a section under "Reasoning Modules" describing what your module does.

## Testing Patterns

- **Real SQLite, no mocking:** Tests use in-memory or temporary file databases. The database layer is never mocked.
- **Mark slow tests:** `@pytest.mark.integration` for tests that call external APIs or run LLM extraction.
- **Direct function calls:** Tests call command functions directly with `typer.Context` objects, not through subprocess.
- **OptionInfo normalization:** When calling typer commands directly, `Option` defaults come through as `OptionInfo` objects. Use `isinstance(param, typer.models.OptionInfo)` to extract `.default`.

```bash
# Fast tests (no integration):
uv run pytest -m "not integration"

# All tests:
uv run pytest

# Single test:
uv run pytest tests/test_graph_engine.py::test_traverse_forward
```

## Code Style

- **Linter/formatter:** `ruff check .` and `ruff format .`
- **Docstrings:** Google-style for public API functions
- **Commit messages:** Conventional Commits (`feat:`, `fix:`, `chore:`, `refactor:`, `docs:`, `test:`)
- **Language:** English for all code, comments, and documentation
- **Type hints:** Use `from __future__ import annotations` and modern Python typing

## Documentation Standards

- **CLI Reference:** Every new command gets an entry in `docs/cli-reference.md` with description, flags table, and examples.
- **Architecture:** New modules are documented in `docs/architecture.md` under the appropriate section.
- **Getting Started:** Keep `docs/getting-started.md` current if the setup or first-pipeline flow changes.
- **Docstrings:** Public API functions should have Google-style docstrings with `Args:`, `Returns:`, and `Raises:` sections.

## Pre-commit Hooks

Install pre-commit hooks for automated linting and formatting:

```bash
pre-commit install
```
