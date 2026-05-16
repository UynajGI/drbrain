# Getting Started

## Prerequisites

- **Python 3.12+** (required)
- **git** (for source install)
- An AI coding agent (Claude Code, Codex, Cursor, Cline, Windsurf, Qwen, or GitHub Copilot) -- optional but recommended

## Installation

DrBrain is a CLI tool. **pipx** or **uv tool install** are the recommended ways to install
it — both create an isolated environment so DrBrain doesn't interfere with your other projects.

### pipx

```bash
# Coming in beta — not yet on PyPI
pipx install drbrain
```

### uv

```bash
# Coming in beta — not yet on PyPI
uv tool install drbrain
```

### pip

```bash
# Coming in beta — not yet on PyPI
pip install drbrain
```

### From source

```bash
git clone https://github.com/UynajGI/DrBrain.git
cd DrBrain
uv sync && uv pip install -e .
drbrain setup
```

## Data Directory

By default, DrBrain stores everything under `~/DrBrain/`:

| Platform | Default path |
|----------|-------------|
| Linux | `~/DrBrain/` |
| macOS | `~/DrBrain/` |
| Windows | `%USERPROFILE%\DrBrain\` |

Inside: `inbox/` (drop PDFs here), `papers/` (processed papers), `drbrain.db` (knowledge graph).

## What `drbrain setup` Does

`drbrain setup` is an interactive wizard that configures your environment:

1. **Configures LLM** -- any litellm provider (OpenAI, Anthropic, Ollama, DeepSeek, Zhipu, Bailian, etc.)
2. **Configures MinerU PDF parser** -- optional; PyMuPDF fallback works without it
3. **Sets up API keys** -- Semantic Scholar, CrossRef, OpenAlex (optional; higher rate limits)
4. **Creates data directories** -- `data/spool/inbox/`, `data/papers/`, `data/cache/`, etc.
5. **Detects AI platforms** and injects agent entry files so your coding agent can use DrBrain skills

Use `--quick` to skip interactive prompts and accept defaults:

```bash
drbrain setup --quick
```

## Your First Pipeline

The full pipeline turns a PDF into a queryable knowledge graph in two phases.

### Step 1: Drop a PDF

Copy a research paper PDF into the inbox:

```bash
cp ~/Downloads/deepseek-r1.pdf data/spool/inbox/
```

### Step 2: Ingest

`drbrain ingest` parses the PDF, extracts metadata from 5 sources (arXiv, CrossRef, Semantic Scholar, OpenAlex, DeepXiv), and structures the content into a section tree.

```bash
drbrain ingest
```

This processes all PDFs in `data/spool/inbox/` by default. You can also ingest specific files:

```bash
drbrain ingest paper1.pdf paper2.pdf
```

After ingest, the paper status is `uploaded`. A `papers/<id>/tree.json` file contains the structured section tree.

### Step 3: Build

`drbrain build` extracts concepts and relations via a 5-stage LLM pipeline:

1. **Ontology Extension** -- domain-specific subcategories under 6 TBox types
2. **Entity Extraction** -- per leaf-node concept extraction, 10-way concurrent
3. **Relation Extraction** -- TBox-typed connections between concepts
4. **Coreference Resolution** -- merge duplicate entity labels
5. **Iterative Refinement** -- LLM self-review for contradictions (skippable with `--skip-refine`)

```bash
drbrain build
# Or target a specific paper:
drbrain build p6a321e
# Or all papers:
drbrain build --all
# Skip the iterative refinement to save time/LLM cost:
drbrain build --skip-refine
```

After build, paper status becomes `extracted`. Your knowledge graph now has typed concepts linked by semantic relations.

### Step 4: Query

Search your knowledge graph with keyword-based BM25 retrieval:

```bash
drbrain query "transformer attention mechanism"
```

Add graph-enhanced discovery:

```bash
# Expand results by 1-hop graph traversal
drbrain query "reinforcement learning" --neighbors 1
# Boost results by graph centrality (PageRank)
drbrain query "knowledge distillation" --hybrid
# Filter by concept type
drbrain query "optimization" --type-filter Method
# Limit to specific years
drbrain query "deep learning" --year-start 2020 --year-end 2024
```

Search the content tree of a specific paper (PageIndex retrieval):

```bash
drbrain query "residual connections" --paper p6a321e
```

### Step 5: Analyze

Run a knowledge frontier analysis to discover research seeds, causal chains, hypotheses, and cross-domain patterns:

```bash
drbrain analyze p6a321e --full
drbrain analyze --workspace nlp --full
drbrain analyze --query "large language models" --full
```

### Step 6: Reason (LLM Agent)

Ask natural language questions that the LLM agent answers by exploring the knowledge graph:

```bash
drbrain reason "What are the main approaches to reducing hallucination in LLMs?"
```

With bidirectional mode (LLM forms hypotheses, validates against graph constraints):

```bash
drbrain reason "Is chain-of-thought more effective than direct prompting for math?" --bidirectional
```

## Verify Your Environment

```bash
drbrain check   # diagnostics: packages, external tools, API connectivity
drbrain audit   # data quality scan (15 rules, 3 severity levels)
```

## Where Data Lives

After running the pipeline, your data directory looks like this:

```
data/
  spool/inbox/        # PDFs awaiting ingest
  spool/pending/      # Failed ingests
  papers/<id>/          # Per-paper: source.pdf, raw.md, tree.json, images/
  drbrain.db            # SQLite database (concepts, relations, edges)
  metrics.db            # LLM token tracking
  cache/                # API cache
  reports/              # Per-paper JSON analysis reports
  backups/              # tar.gz backups
workspace/<name>/       # Paper subsets: workspace.yaml + refs/papers.json
```

## Next Steps

- [CLI Reference](cli-reference.md) -- every command with examples
- [Architecture](architecture.md) -- how the system works under the hood
- [Contributing](contributing.md) -- add commands, modules, and skills
