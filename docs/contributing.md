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
│   ├── base.py           # Shared PatentBase ABC + google_patents_url helper
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

## How to Add a New Module

### Extractor Module

1. Create `src/drbrain/extractor/my_module.py` with a public function:

```python
def analyze_my_pattern(graph, db, **kwargs) -> list[dict]:
    """Analyze something in the knowledge graph."""
    results = []
    for node in graph.graph.nodes:
        # ... analysis ...
        results.append({"node": node, "finding": ...})
    return results
```

2. Wire into the analyzer in `src/drbrain/report/analyzer.py`:

```python
from drbrain.extractor.my_module import analyze_my_pattern

def analyze_paper(db, graph, paper_id, full=False, models=None):
    if full:
        report["my_findings"] = analyze_my_pattern(graph, db)
```

3. Add CLI support in the appropriate commands module.
4. Add tests in `tests/test_my_module.py`.
5. Document in `docs/architecture.md` under "Reasoning Modules".
6. Run `uv run ruff check . && uv run ruff format .` before committing.

### Service Module

Services are higher-level modules in `src/drbrain/services/` that compose multiple lower-level modules. Follow the same pattern: create the module, wire it in, add CLI, add tests, document.

### Graph Module

Graph modules live in `src/drbrain/graph/`. Standard interface: accept `GraphEngine` and `Database` as parameters. New inference rules go in `engine_closure.py` (for `closure()`) or `path_reasoning.py` (for path-level rules).

## PR Process

1. **Branch**: `feature/description` or `fix/description` from `main`
2. **Commits**: Conventional Commits format (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`)
3. **Before PR**:
   ```bash
   uv run ruff check . && uv run ruff format .    # lint + format
   uv run pytest                                   # all tests
   uv run pytest -m "not integration"              # fast suite
   ```
4. **PR title**: Conventional Commit format (e.g. `feat(reasoning): add contradiction detection workflow`)
5. **PR description**: what, why, how tested, screenshots/CLI output if relevant
6. **Review**: at least one approving review before merge. Reviewer checks: correctness, test coverage, docs updated, no unrelated changes.

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md` — move `[Unreleased]` entries to a new version section with date
3. Create a git tag: `git tag -a v0.1.0 -m "v0.1.0"`
4. Push tag: `git push origin v0.1.0`
5. Build and publish to PyPI (TBD when PyPI publishing is set up)

## Testing Guide

### Test Database

Use `Database(":memory:")` for fast isolated tests:

```python
from drbrain.storage.database import Database

def test_something():
    db = Database(":memory:")
    db.insert_concept("paper1", "Problem", "Overfitting", 0.9)
    # ... test ...
    db.close()
```

### CLI Command Testing

When testing typer commands directly, construct a minimal context:

```python
import typer
from drbrain.config import Config

def test_my_command():
    cfg = Config.default()
    ctx = typer.Context(typer.Typer(), obj={"config": cfg})
    result = my_command(ctx, arg1="value")
    assert result is not None
```

If using `Option` defaults, they come through as `OptionInfo` objects:

```python
assert isinstance(param, typer.models.OptionInfo)
default_value = param.default
```

### Integration Tests

Mark slow tests with `@pytest.mark.integration`:

```python
@pytest.mark.integration
def test_llm_extraction():
    ...
```

Run without integration tests: `uv run pytest -m "not integration"`

### Test Fixtures

Common patterns:
- `Database(":memory:")` for isolated DB tests
- `GraphEngine()` loads from DB, tests against loaded graph
- Seed data manually: insert known concepts/edges, then verify analysis output
- Never mock the database layer. Real SQLite only.

## Skill Development

DrBrain skills live in `skills/<name>/SKILL.md`. Each skill wraps one or more CLI commands for use with AI coding agents.

### Adding a New Skill

1. Create `skills/my-skill/SKILL.md`:

```markdown
---
name: my-skill
description: Short description shown in skill list
---

# My Skill

## Quick Start
```bash
drbrain my-command --flag value
```

## Usage
...
```

2. The skill name should match the primary CLI command it wraps.
3. Include realistic examples in the skill — the AI agent uses these as few-shot prompts.
4. Test the skill: `npx skills add ./skills/my-skill` then use it in an AI agent.

### Skill Categories

Skills are organized by function:
- **Data In** (ingest, fetch, import, translate) — getting papers into DrBrain
- **KG Build** (build, embed, closure) — constructing the knowledge graph
- **Query & Explore** (query, search, graph, explore) — searching and navigating
- **Analysis** (analyze, evolve, landscape, paradigm, transfers, isomorphism) — knowledge discovery
- **Library Management** (list, show, stats, export, backup, proceedings) — maintaining the library
- **Quality** (audit, repair, check) — data quality assurance

## Pre-commit Hooks

Install pre-commit hooks for automated linting and formatting:

```bash
pre-commit install
```
