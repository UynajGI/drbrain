# CLI reference

All commands are invoked as `uv run drbrain COMMAND`. Use `--help` on any command for complete option details. `--config PATH` selects a configuration file and `--root PATH` selects the runtime root.

## Main line: ingest → index build → search / ask

| Command | Purpose |
| --- | --- |
| `setup [--quick]` | Create configuration, directories, and validate the environment |
| `ingest PATH...` | Parse PDF/Markdown/text/LaTeX into the paper store and register the canonical body (no per-paper MD/tree for new papers) |
| `index build [--force] [--tree-storage PATH] [--db PATH] [--json]` | Prepare every enabled retrieval leg and publish one queryable generation: incremental lexical BM25, canonical FTS, shared leaf/region vectors and the unified tree hierarchy (FTS → vectors → hierarchy → publish), plus the LlamaIndex generation when `rag_engine: llamaindex` (the deprecated SQL snapshot path is not republished). Exit 1 when any stage failed — a failed stage is never reported as ready |
| `index status [--json]` | Read-only readiness report: separates **ingested / indexed / retrievable**, per-leg `ready` + reasons + versions, pending backlog and failure reasons. Exit 0 (the report is the result) |
| `index verify [--json]` | Read-only consistency check over exactly what `search`/`ask` read: FTS, node-vector backlog, leaf reachability, the published generation (manifest, embedding-profile identity, live watermarks). Exit 1 on any error. The broad storage audit stays `storage audit` |
| `search QUERY... [--limit N] [--paper ID...] [--source local\|arxiv\|all] [--json]` | Three-leg evidence retrieval over the same chain as `ask` (bm25/vector/tree by default), without answer synthesis. Returns sources, text locators, route and index generation. `--paper` scopes every leg to the given papers; `--source arxiv\|all` adds external rows (`source="arxiv"`, url/doi/arxiv_id, no local locator). A missing index exits 1 with the `drbrain index build` remedy |
| `ask TEXT` | Natural-language question answering (retrieval + REFINE synthesis over the same route) |
| `pipeline [--preset full\|quick\|embed\|full-rag] [--steps S1,S2]` | Chain steps in sequence; each step runs the main-line child (`ingest`, `graph build`, `graph embed`, `graph closure`, `index build`) |

## Knowledge graph (`graph` namespace)

| Command | Purpose |
| --- | --- |
| `graph build [PAPER_ID...] [--all]` | 5-stage LLM extraction (ontology → entities → relations → coref → refine); hidden top-level `build` is the same command |
| `graph embed [--dim N] [--epochs N] [--retrain] [--papers ...]` | Train TransE graph embeddings (bare `embed` is the same command; not a silent behavior change) |
| `graph closure [--incremental\|--full] [--mode symbolic\|hybrid] [--mine-rules]` | Rule-based inference (8 symbolic + 4 embedding rules) |
| `graph neighbors/path/related/describe/traverse-from/export` | Direct graph traversal, description and export |
| `graph query` | TransE complex query (∧∨¬ operators) |

## Retrieval and research

| Command | Purpose |
| --- | --- |
| `library search TEXT [--limit N] [--type T] [--json]` | Bibliographic BM25 search over papers, concepts, and argument claims (the historical `search`) |
| `reason TEXT` | Run bounded tool-using reasoning; supports sessions and workflows |
| `survey TEXT` | Generate a one-shot literature survey |
| `frontier` / `landscape` | Active gaps, debates, frontier signals / domain timeline |
| `evolve`, `descendants`, `analyze`, `seed`, `difficulty`, `transfers`, `isomorphism`, `paradigm` | Graph genealogy and analysis |

## Index evaluation and operations

| Command | Purpose |
| --- | --- |
| `rag eval` | Golden-set evaluation (retriever HitRate/MRR and/or RAGAS-style) |
| `rag baselines` | Evaluation-only baselines over a golden split: `unified_tree_flat` (unified-tree ablation) and provenance-gated `raptor_collapsed` |
| `rag pageindex-index` / `rag pageindex-chat` | PageIndex native filesystem materialisation / per-paper chat |
| `storage audit` | Read-only storage-integrity audit (unified store + legacy artifacts) |

## Build and maintenance

| Command | Purpose |
| --- | --- |
| `audit` | Run data-quality checks |
| `repair` / `enrich` | Repair or enrich metadata |
| `clean` | Clear rebuildable data while preserving inbox sources |
| `check` / `metrics` / `backup` / `restore` | Environment checks, analytics, backups |

## Library and export

`list`, `show`, `stats`, `report`, and `delete` manage papers. `export` writes BibTeX, RIS, or Markdown; `export-okf` writes an OKF bundle; `citations` queries citation relations; `style` manages citation styles; `document` inspects Office files; `patent-search` queries patent sources; `proceedings` manages proceedings; and `explore` manages lightweight discovery silos.

## Sessions and workspaces

`session new|ask|chat|list|delete|export` manages persistent reasoning sessions. `ws create|add|remove|list|show|delete|rename` manages workspace collections. `webui` starts the local browser workbench.

## Compatibility aliases (hidden from the help page)

The main-line convergence is additive: every historical entry point stays
callable with its original flags, defaults, exit codes and JSON contracts, is
hidden from the help listing, and prints one migration line on stderr.

| Historical entry | Use instead |
| --- | --- |
| `query`, `hybrid` | `search` |
| `fsearch` | `search --source all` (or `--source arxiv`) |
| `search` (BM25 over papers/concepts/arguments) | `library search` |
| `rag prepare`, `rag prepare --unified`, `rag index`, `embed --tree` | `index build` |
| `rag health` | `index status` |
| `build` | `graph build` |
| `embed` | `graph embed` |
| `closure` | `graph closure` |
| `index` (bare, lexical BM25 rebuild) | still the lexical stage; `index build` runs it together with the other legs |

## Machine-readable output

Commands that support `--json` emit a stable object for automation. `query` additionally supports `--jsonl`. Errors use a non-zero exit code and write diagnostics to stderr; migration notices for hidden aliases are written to stderr so stdout stays parseable.
