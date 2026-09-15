# Getting started

## Prerequisites

- Python 3.12
- `uv`
- An LLM provider configuration for extraction, chat, or loop tasks
- Optional: MinerU, embedding service, and external provider keys

## Install

The base install contains the lightweight CLI, configuration, storage, and
graph surface. Feature stacks are opt-in profiles:

| Profile | Adds |
| --- | --- |
| `pdf` | PyMuPDF and `pymupdf4llm` |
| `ingest` | PDF parsing, PageIndex, LLM bridge, and metadata providers |
| `models` | sentence-transformers and ModelScope for local BGE models |
| `rag` | LlamaIndex, BM25, and Zvec ANN retrieval |
| `full` | All runtime profiles, OCR, Office, analytics, and PyTorch/GNN |

```bash
# Minimal CLI
uv sync
uv pip install -e .
uv run drbrain --help

# Typical local PDF + BGE + Zvec workflow
uv sync --extra ingest --extra models --extra rag
```

## Initialize a runtime

```bash
uv run drbrain setup --quick
```

Use `--root PATH` on CLI commands to isolate data from the current directory. Keep secrets in `config.local.yaml` or environment variables; do not commit them.

## First paper

```bash
uv run drbrain ingest path/to/paper.pdf
uv run drbrain index build        # prepare + publish the searchable index
uv run drbrain search "your research question"   # evidence, no answer
uv run drbrain ask "your research question"      # evidence + synthesized answer
```

`index build` fills the enabled retrieval legs incrementally — lexical BM25,
canonical FTS, shared vectors, and the unified tree hierarchy — and publishes
one queryable generation; `index status` reports the ingested / indexed /
retrievable states per leg, and `index verify` re-checks what `search` and
`ask` actually read. For the optional knowledge-graph branch, run
`drbrain graph build`, then `drbrain graph embed`, then `drbrain graph closure`.

The historical names still work during the compatibility period (hidden from
the help page, one migration line on stderr): `rag prepare`/`rag prepare
--unified`/`rag index` → `index build`, `rag health` → `index status`,
`query`/`hybrid` → `search`, `fsearch` → `search --source all`, `search` →
`library search`, and `build`/`embed`/`closure` → `graph build`/`graph
embed`/`graph closure`. See [cli-reference.md](cli-reference.md).

## Research loop

```bash
uv run drbrain reason "How do these methods differ?"
uv run drbrain reason --workflow review PAPER_ID
```

The loop uses the capability catalog to select a scoped tool space. Runs, events, tool calls, checkpoints, and reports are persisted for inspection and resume.

## Backups

```bash
uv run drbrain backup
uv run drbrain restore data/backups/drbrain-YYYYMMDD-HHMMSS.tar.gz --target restore-dir
```

Archives without a manifest require the explicit `--allow-legacy` option.

## Verification

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/drbrain
uv run pytest -m "not integration"
```

If a command fails, run `uv run drbrain check` and consult [troubleshooting.md](troubleshooting.md).
