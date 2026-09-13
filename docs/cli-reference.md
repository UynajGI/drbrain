# CLI reference

All commands are invoked as `uv run drbrain COMMAND`. Use `--help` on any command for complete option details. `--config PATH` selects a configuration file and `--root PATH` selects the runtime root.

## Setup and ingestion

| Command | Purpose |
| --- | --- |
| `setup [--quick]` | Create configuration, directories, and validate the environment |
| `ingest PATH...` | Parse PDF/Markdown/text/LaTeX into the paper store |
| `fetch IDENTIFIER` | Find and download an open paper, then ingest it |
| `batch-fetch FILE` | Fetch identifiers from a list |
| `import FILE` | Import Zotero, BibTeX, or Endnote records |
| `translate` | Translate stored Markdown with resume support |

## Retrieval and research

| Command | Purpose |
| --- | --- |
| `query TEXT` | Search local concepts and arguments; supports filters and JSON output |
| `ask TEXT` | Natural-language question answering over the local corpus |
| `reason TEXT` | Run bounded tool-using reasoning; supports sessions and workflows |
| `survey TEXT` | Generate a one-shot literature survey |
| `search TEXT` | Quick BM25 search over papers, concepts, and arguments |
| `fsearch TEXT` | Federated local and arXiv search |
| `frontier` | Report active gaps, debates, and frontier signals |
| `landscape` | Summarize domain timeline and open problems |

## Build and maintenance

| Command | Purpose |
| --- | --- |
| `build [PAPER_ID...]` | Extract structured records from ingested papers |
| `embed` | Build graph or tree text embeddings |
| `closure` | Run rule-based closure |
| `audit` | Run data-quality checks |
| `repair` / `enrich` | Repair or enrich metadata |
| `index` | Rebuild local search indexes |
| `clean` | Clear rebuildable data while preserving inbox sources |

## Library and export

`list`, `show`, `stats`, `report`, and `delete` manage papers. `export` writes BibTeX, RIS, or Markdown; `citations` queries citation relations; `style` manages citation styles; `document` inspects Office files; `patent-search` queries patent sources; `proceedings` manages proceedings; and `explore` manages lightweight discovery silos.

## Sessions and workspaces

`session new|ask|chat|list|delete|export` manages persistent reasoning sessions. `ws create|add|remove|list|show|delete|rename` manages workspace collections.

## Backup and diagnostics

```bash
uv run drbrain backup [--target NAME] [--dry-run]
uv run drbrain restore ARCHIVE --target PATH [--force] [--allow-legacy]
uv run drbrain check
```

`graph` commands and `webui` are documented separately because they are outside the current storage/RAG/plugin/loop scope.

## Machine-readable output

Commands that support `--json` emit a stable object for automation. `query` additionally supports `--jsonl`. Errors use a non-zero exit code and write diagnostics to stderr.
