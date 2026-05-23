# CLI Reference

All DrBrain commands, grouped by function. Every command supports `--json` for machine-readable output unless otherwise noted.

---

## Setup and Maintenance

### `drbrain setup`

Initialize DrBrain -- generate config, create directories, validate environment.

| Flag | Short | Description |
|------|-------|-------------|
| `--quick` | `-q` | Skip interactive prompts, read from env vars |
| `--change-password` | | Change the admin password |

```bash
drbrain setup
drbrain setup --quick
```

### `drbrain check`

Check dependencies, configuration, and environment variables. Reports package versions, external tool availability, API connectivity, and data directory status.

```bash
drbrain check
```

### `drbrain audit`

Scan the library for data quality issues using 15 severity-graded rules.

| Flag | Short | Description |
|------|-------|-------------|
| `--severity` | `-s` | Minimum severity: `error`, `warning`, `info` (default: `warning`) |
| `--workspace` | `-w` | Limit audit to a workspace |
| `--json` | `-j` | Output as JSON |

```bash
drbrain audit
drbrain audit --severity error
drbrain audit --workspace nlp --json
```

### `drbrain ingest-link`

Ingest web URLs by extracting rendered content via an external qt-web-extractor service.

| Flag | Short | Description |
|------|-------|-------------|
| `--pdf` | | Force PDF extraction mode |
| `--dry-run` | | Preview only, no save |
| `--json` | | Output JSON |

```bash
drbrain ingest-link https://example.com/page
drbrain ingest-link https://example.com/report.pdf --pdf
drbrain ingest-link https://a.com https://b.com --dry-run
```

### `drbrain fetch`

Fetch a paper from open access sources — find PDF → download → ingest. Uses a 5-stage fallback: arXiv, OpenAlex OA, Unpaywall, direct DOI resolution, title-based arXiv search.

| Flag | Short | Description |
|------|-------|-------------|
| `--arxiv` | | Treat identifier as arXiv ID |

```bash
drbrain fetch 10.1234/example.doi
drbrain fetch --arxiv 1706.03762
drbrain fetch "Attention Is All You Need"
```

### `drbrain patent-search`

Search USPTO patents via PPUBS (free) or ODP (API key required).

| Flag | Short | Description |
|------|-------|-------------|
| `--source` | `-s` | Search source: `ppubs` (free) or `odp` (API key) |
| `--application` | `-a` | Lookup by application number (ODP only) |
| `--limit` | `-n` | Max results (default: 10) |
| `--api-key` | | USPTO ODP API key |
| `--json` | | Output JSON |

```bash
drbrain patent-search "machine learning"
drbrain patent-search "quantum computing" --source odp --api-key $USPTO_ODP_API_KEY
drbrain patent-search --application 17123456 --source odp
```

### `drbrain clean`

Clear data directories (database, cache, logs, papers, reports). Keeps inbox PDFs intact.

| Flag | Short | Description |
|------|-------|-------------|
| `--force` | `-f` | Skip confirmation prompt |
| `--config` | `-c` | Config file path (default: `config.yaml`) |

```bash
drbrain clean --force
```

---

## Ingestion

### `drbrain ingest`

Parse PDFs, extract metadata from 5 sources, and build section tree. Accepts PDF files, directories, or defaults to `data/spool/inbox/`.

| Flag | Description |
|------|-------------|
| `--json` | Output machine-readable JSON |

```bash
drbrain ingest                           # all PDFs in inbox
drbrain ingest paper.pdf                 # single file
drbrain ingest dir1/ dir2/               # multiple directories
drbrain ingest paper1.pdf paper2.pdf     # multiple files
drbrain ingest --json                    # JSON output for pipelines
```

### `drbrain build`

Extract concepts and relations via 5-stage LLM pipeline. Processes all `uploaded` papers by default. When `--session` is provided, injects a structured extraction summary into the session so subsequent `reason --session` calls have full build context.

| Flag | Short | Description |
|------|-------|-------------|
| `--all` | | Build graph for all papers in database |
| `--skip-refine` | | Skip iterative refinement stage (saves LLM cost) |
| `--session` | `-s` | Session ID ("new" to create, or existing ID). Injects extraction summary after build. |
| `--json` | | Output JSON |

```bash
drbrain build                      # all unprocessed papers
drbrain build p6a321e              # single paper
drbrain build p6a321e p3b452c      # multiple papers
drbrain build --all                # re-extract everything
drbrain build --skip-refine        # faster, less polished
drbrain build --session new        # create session, inject summary
drbrain build p6a321e -s sess-xxx  # inject into existing session
drbrain build p6a321e --json
```

---

## Query and Search

### `drbrain query`

Query concepts and arguments with BM25 keyword search and optional graph enhancement.

| Flag | Short | Description |
|------|-------|-------------|
| `--type-filter` | | Filter by concept type (Problem, Method, etc.) |
| `--arg-type` | | Filter by argument claim type |
| `--year-start` | | Minimum year |
| `--year-end` | | Maximum year |
| `--min-confidence` | | Minimum confidence threshold |
| `--limit` | | Maximum results (default: 20) |
| `--neighbors` | `-n` | Expand results by N hops of graph traversal |
| `--relation` | `-R` | Comma-separated relation types to follow |
| `--direction` | `-D` | Traversal direction: forward, backward, both (default: both) |
| `--hybrid` | | Boost results by graph centrality (PageRank) |
| `--paper` | | Paper local_id for hybrid tree retrieval (LLM primary + vector auxiliary) |
| `--workspace` | `-w` | Limit to workspace |
| `--json` / `--jsonl` | | Output format |

```bash
drbrain query "transformer attention"
drbrain query "deep learning" --type-filter Method --year-start 2020
drbrain query "reinforcement learning" --neighbors 2 --hybrid
drbrain query "residual connections" --paper p6a321e
drbrain query "model compression" --workspace ml-systems --json
```

### `drbrain ask`

Ask a natural language question -- retrieves relevant concepts from the KG and generates a concise answer with citations.

| Flag | Short | Description |
|------|-------|-------------|
| `--top` | `-k` | Number of graph concepts to retrieve (default: 5) |
| `--json` | | Output JSON |

```bash
drbrain ask "Is attention better than CNN for NLP?"
drbrain ask "What causes catastrophic forgetting?" --top 10 --json
```

### `drbrain fsearch`

Federated search — local library + arXiv with automatic ingested annotation.

| Flag | Short | Description |
|------|-------|-------------|
| `--arxiv` | | Also search arXiv |
| `--arxiv-only` | | Search arXiv only |
| `--limit` | `-n` | Max results per source (default: 20) |
| `--json` | | Output JSON |

```bash
drbrain fsearch "attention mechanism"
drbrain fsearch "transformer" --arxiv
drbrain fsearch "graph neural network" --arxiv-only --json
```

### `drbrain index`

Build or rebuild the BM25 search index. Run this after adding or modifying papers if query results seem stale.

| Flag | Description |
|------|-------------|
| `--rebuild` | Force full index rebuild |
| `--json` | Output JSON |

```bash
drbrain index
drbrain index --rebuild
drbrain index --rebuild --json
```

---

## Graph Exploration

### `drbrain graph neighbors`

Traverse the graph from a node, showing neighbors with path information.

| Flag | Short | Description |
|------|-------|-------------|
| `--hops` | `-n` | Number of hops (default: 1) |
| `--relation` | `-R` | Comma-separated relation types to filter by |
| `--direction` | `-D` | forward, backward, or both (default: both) |
| `--json` | | Output JSON |
| `--workspace` | `-w` | Limit to workspace |

```bash
drbrain graph neighbors Attention
drbrain graph neighbors "Transformer" --hops 2
drbrain graph neighbors "Curriculum Learning" --relation extends,addresses --direction forward
drbrain graph neighbors p6a321e --workspace nlp --json
```

### `drbrain graph path`

Find the shortest path between two nodes in the knowledge graph.

| Flag | Description |
|------|-------------|
| `--max-length` | Maximum path length (default: 6) |
| `--json` | Output JSON |
| `--workspace` | `-w` | Limit to workspace |

```bash
drbrain graph path Attention "Transformer"
drbrain graph path "Dropout" "Batch Normalization" --max-length 4 --json
```

### `drbrain graph related`

Analyze shared concepts and connections across multiple papers.

| Flag | Short | Description |
|------|-------|-------------|
| `--mode` | `-m` | Analysis mode: `concepts` (SQL label intersection), `graph` (1-hop neighbor intersection), `edges` (shared edge patterns) |
| `--min-shared` | | Minimum number of papers a concept must appear in (default: 2) |
| `--json` | | Output JSON |
| `--workspace` | `-w` | Limit to workspace |

```bash
drbrain graph related p6a321e p3b452c
drbrain graph related p6a321e p3b452c p9c781d --mode edges --min-shared 3
drbrain graph related p6a321e p3b452c --mode graph --json
```

### `drbrain graph describe`

Generate a natural language description of the subgraph centered on a node.

| Flag | Short | Description |
|------|-------|-------------|
| `--depth` | `-n` | Number of hops to traverse (default: 1) |
| `--json` | | Output JSON |
| `--workspace` | `-w` | Limit to workspace |

```bash
drbrain graph describe Attention
drbrain graph describe "Reinforcement Learning" --depth 2
```

### `drbrain graph query`

Execute embedding-based complex queries over TransE embeddings. Requires trained embeddings (`drbrain embed`).

**Query DSL types:** `project`, `intersect`, `union`, `negate`

| Flag | Short | Description |
|------|-------|-------------|
| `--top` | `-k` | Number of results (default: 10) |
| `--json` | | Output JSON |

```bash
drbrain graph query '{"type":"project","entity":"Attention","relation":"addresses"}'
drbrain graph query '{"type":"intersect","queries":[{"type":"project","entity":"LLM","relation":"addresses"},{"type":"project","entity":"Scaling","relation":"addresses"}]}' --top 5 --json
```

### `drbrain graph traverse-from`

Traverse the graph starting from a section title.

```bash
drbrain graph traverse-from "Related Work" --depth 2
drbrain graph traverse-from "Experiments" --direction forward --json
```

---

## Citations

### `drbrain citations`

Query the citation graph for a paper: references, citing papers, and shared-reference analysis.

| Flag | Short | Description |
|------|-------|-------------|
| `--type` | `-t` | Query type: `refs`, `citing`, `shared-refs`, `all` (default: `all`) |
| `--limit` | `-l` | Max results per type (default: 200) |
| `--sort` | `-s` | Sort: `cited_by_count:desc`, `publication_date:desc`, `relevance_score:desc` |
| `--workspace` | `-w` | Limit to workspace |
| `--json` | | Output JSON |

```bash
drbrain citations p6a321e
drbrain citations p6a321e --type citing --sort publication_date:desc
drbrain citations p6a321e --type shared-refs --workspace nlp
```

### `drbrain check-citations`

Verify in-text citations against your local library. Extracts (Author, Year) patterns from text and matches them.

| Flag | Short | Description |
|------|-------|-------------|
| `--file` | `-f` | Read text from file |
| `--json` | | Output JSON |

```bash
drbrain check-citations "Transformer (Vaswani et al., 2017) introduced..."
drbrain check-citations --file draft.md --json
```

---

## Analysis

### `drbrain evolve`

Show how a concept evolved — ancestors and descendants in the knowledge graph via BFS traversal. Follows `extends`, `refines`, `applies` edges.

| Flag | Short | Description |
|------|-------|-------------|
| `--direction` | `-d` | `ancestors`, `descendants`, or `both` (default: `both`) |
| `--max-depth` | `-n` | Max traversal depth (default: 3) |
| `--mermaid` | | Output as Mermaid diagram |
| `--stats` | | Show temporal evolution signal (emerging/established/declining/contested/resurging) and year-by-year counts |
| `--json` | | Output as JSON |

```bash
drbrain evolve "Transformer"
drbrain evolve "graph neural network" --direction descendants --max-depth 5
drbrain evolve "Attention" --mermaid
drbrain evolve "Dropout" --stats
```

### `drbrain descendants`

Trace a paper's academic offspring — who cited, extended, refined, or challenged it recursively.

| Flag | Short | Description |
|------|-------|-------------|
| `--generations` | `-g` | Number of generations to trace (default: 3) |
| `--mermaid` | | Mermaid diagram output |
| `--json` | | JSON output |
| `--sections` | | Show section provenance for each concept |

```bash
drbrain descendants p3f8a2
drbrain descendants p3f8a2 --generations 5 --mermaid
drbrain descendants p3f8a2 --sections
```

### `drbrain landscape`

Domain panorama — year-ordered timeline with key concepts, persistent gaps, and active debates. Workspace-scoped.

| Flag | Short | Description |
|------|-------|-------------|
| `--top-n` | | Top papers per year (default: 5) |
| `--json` | | JSON output |

```bash
drbrain landscape my-workspace
drbrain landscape my-workspace --top-n 10
```

### `drbrain paradigm`

Detect paradigm shifts — replacement (old declining, new growing), explosion (concept burst with descendants), or cross-domain invasion.

| Flag | Short | Description |
|------|-------|-------------|
| `--workspace` | `-w` | Scan entire workspace |
| `--json` | | JSON output |

```bash
drbrain paradigm "Transformer"
drbrain paradigm --workspace nlp-ws
```

### `drbrain transfers`

Discover cross-domain method migration opportunities. Two modes: explicit workspaces (`--from`/`--to`) or auto-clustering (`--auto`).

| Flag | Short | Description |
|------|-------|-------------|
| `--from` | | Source workspace (methods) |
| `--to` | | Target workspace (problems) |
| `--auto` | | Auto-detect domains via label clustering |
| `--history` | | Show historical cross-domain transfers |
| `--min-confidence` | | Minimum transfer confidence (default: 0.3) |
| `--json` | | JSON output |
| `--sections` | | Show section provenance for transferred concepts |

```bash
drbrain transfers --from nlp-ws --to cv-ws
drbrain transfers --auto
drbrain transfers --history
```

### `drbrain isomorphism`

Find structurally isomorphic subgraphs -- concepts with similar relation patterns across domains.

| Flag | Short | Description |
|------|-------|-------------|
| `--min-confidence` | | Minimum confidence threshold (default: 0.5) |
| `--json` | | Output as JSON (includes RAPTOR context) |

```bash
drbrain isomorphism "Attention Mechanism"
drbrain isomorphism --min-confidence 0.7
drbrain isomorphism --json
```

### `drbrain difficulty`

Classify knowledge gaps by section semantics. Maps each gap to its originating section type (introduction, methods, results, etc.) and computes a composite difficulty score.

| Flag | Short | Description |
|------|-------|-------------|
| `--json` | | Output as JSON |

```bash
drbrain difficulty
drbrain difficulty --json
```

### `drbrain frontier`

Composite knowledge frontier report combining research seeds, debate zones, technology cliffs, difficulty scores, and confidence collapse patterns.

| Flag | Short | Description |
|------|-------|-------------|
| `--json` | | Output as JSON |

```bash
drbrain frontier
drbrain frontier --json
```

### `drbrain analyze`

Run a knowledge frontier analysis including research seeds, causal chains, hypotheses, and cross-domain patterns.

**Paper selection (mutually exclusive, first match wins):**
- `<local_id>` -- single paper
- `--papers p1,p2,...` -- specific papers
- `--query "text"` -- BM25 search then analyze matches
- `--discover "question"` -- LLM graph exploration to find relevant papers
- `--workspace ws` -- all papers in workspace

| Flag | Short | Description |
|------|-------|-------------|
| `--papers` | | Comma-separated paper IDs |
| `--query` | | BM25 search query |
| `--discover` | | LLM graph discovery question |
| `--workspace` | `-w` | Workspace boundary scan |
| `--full` | `-f` | Full analysis (slower, more thorough) |
| `--json` | | Output JSON |

```bash
drbrain analyze p6a321e --full
drbrain analyze --papers p6a321e,p3b452c --full
drbrain analyze --query "large language models" --full
drbrain analyze --discover "knowledge distillation in NLP"
drbrain analyze --workspace nlp --full --json
```

### `drbrain seed`

Detect research seeds from graph patterns. Seeds include stale problems, unaddressed gaps, debate zones, technology cliffs, and confidence collapse patterns.

| Flag | Short | Description |
|------|-------|-------------|
| `--workspace` | `-w` | Limit to workspace |
| `--json` | | Output JSON |

```bash
drbrain seed
drbrain seed --workspace cv --json
```

### `drbrain closure`

Run rule-based closure on the full graph to infer new edges via symbolic or hybrid inference.

| Flag | Description |
|------|-------------|
| `--dry-run` | Output inferred edges but do not persist to database |
| `--rule` | Run only named rule(s). Repeatable. Omit for all. |
| `--workspace` | `-w` | Limit to workspace |
| `--mode` | Inference mode: `symbolic` or `hybrid` (default: `symbolic`) |
| `--mine-rules` | Mine path rules from TransE embeddings |
| `--min-confidence` | Minimum confidence for mined rules (default: 0.6) |
| `--ground` | Ground transitive rules as concrete triples (t-norm) |
| `--json` | Output JSON |

**Available rules:** `creates_debate`, `gap_addressed`, `indirect_evolution`, `gap_to_debate`, `shared_actor`, `transitive_closure`, `asymmetric_violations`, `method_supersedes_problem`, `challenge_chain`, `gap_inheritance`, `indirect_support`

```bash
drbrain closure
drbrain closure --rule transitive_closure --rule challenge_chain
drbrain closure --mode hybrid --workspace nlp
drbrain closure --dry-run --json
drbrain closure --mine-rules --min-confidence 0.7 --ground
```

### `drbrain reason`

LLM agent that reasons over the knowledge graph using tool-calling. The agent has access to `search_concepts`, `get_neighbors`, and `find_path` tools.

When `--session` is provided, uses persistent `SessionAgent` (DB-backed) instead of stateless `ReasonerAgent`. Session context accumulates across CLI invocations — build results, previous questions, and tool calls are all preserved.

| Flag | Short | Description |
|------|-------|-------------|
| `--bidirectional` | `-b` | Use bidirectional LLM-KG iterative reasoning (validates hypotheses against graph constraints) |
| `--max-rounds` | `-r` | Maximum hypothesis-revision rounds (default: 3) |
| `--session` | `-s` | Use persistent session. "new" to create, or existing session ID. |

```bash
drbrain reason "What are the main approaches to reducing hallucination?"
drbrain reason "Is dropout effective for transformer regularization?" --bidirectional --max-rounds 5
drbrain reason -s sess-xxx "Compare concepts from yesterday's build"
drbrain reason -s new -b "Explain the debate around scaling laws"
```

### `drbrain embed`

Train TransE graph embeddings for link prediction and entity similarity.
Use `--tree` to generate text embeddings for PageIndex and RAPTOR tree nodes.

Provider is configured via `embed.provider` in config: `local` (default, sentence-transformers),
`openai-compat` (OpenAI-compatible `/v1/embeddings` API), or `none` (disable).

| Flag | Description |
|------|-------------|
| `--dim` | Embedding dimension (default: 128) |
| `--epochs` | Training epochs (default: 100) |
| `--retrain` | Force retrain from scratch |
| `--tree` | Generate tree node text embeddings (PageIndex + RAPTOR) |

```bash
drbrain embed
drbrain embed --dim 256 --epochs 200
drbrain embed --retrain
drbrain embed --tree
```

### `drbrain report`

Display a single-paper report with graph coverage and concept statistics.

```bash
drbrain report p6a321e
drbrain report p6a321e --json
```

### `drbrain lineage`

Explore author/research lineage via OpenAlex deduplicated IDs.

| Flag | Short | Description |
|------|-------|-------------|
| `--list` | | List all actors with paper counts |
| `--name` | `-n` | Search actors by display name |
| `--json` | | Output JSON |

```bash
drbrain lineage --list
drbrain lineage --name "Hinton"
drbrain lineage A5023806754 --json
```

---

## Export and Import

### `drbrain export`

Export paper metadata to BibTeX, RIS, or Markdown with citation style support.

| Flag | Short | Description |
|------|-------|-------------|
| `--format` | `-f` | Export format: `bib`, `ris`, `md` (default: `bib`) |
| `--style` | `-s` | Citation style for Markdown: `apa`, `vancouver`, `chicago-author-date`, `mla` (default: `apa`) |
| `--all` | | Export all papers |
| `--output` | `-o` | Output file path |
| `--json` | | Output JSON |

```bash
drbrain export p6a321e --format bib
drbrain export --all --format ris -o library.ris
drbrain export --all --format md --style vancouver
drbrain export p6a321e --format md --json
```

### `drbrain style`

Manage citation styles for Markdown export.

| Flag | Short | Description |
|------|-------|-------------|
| `--list` | `-l` | List available citation styles |
| `--show` | | Show source of a specific style |

```bash
drbrain style --list
drbrain style --show chicago-author-date
```

### `drbrain import`

Import papers from Zotero, BibTeX, or Endnote.

| Flag | Description |
|------|-------------|
| `--dry-run` | Preview only |
| `--json` | Output JSON |
| `--list-collections` | List collections and exit (Zotero only) |
| `--collection` | Filter by collection key (Zotero only) |
| `--api-key` | Zotero API key (for Web API mode) |
| `--library-id` | Zotero library ID (for Web API mode) |
| `--library-type` | Zotero library type: `user` or `group` |
| `--no-pdf` | Skip PDF detection/download |
| `--import-collections` | Create workspaces per collection after import |

```bash
drbrain import zotero ~/zotero.sqlite
drbrain import zotero ~/zotero.sqlite --collection ABC123 --import-collections
drbrain import zotero . --api-key xyz --library-id 12345
drbrain import bibtex references.bib
drbrain import endnote export.xml
drbrain import endnote library.ris --dry-run
```

---

## Library Management

### `drbrain list`

List all papers in the database.

```bash
drbrain list
drbrain list --json
```

### `drbrain show`

Show detailed view of a single paper: metadata, concepts (grouped by type), arguments, outgoing/incoming edges.

```bash
drbrain show p6a321e
drbrain show p6a321e --json
```

### `drbrain stats`

Display database statistics: paper counts, concept counts by type, relation distribution, recent activity.

| Flag | Short | Description |
|------|-------|-------------|
| `--workspace` | `-w` | Limit to workspace |
| `--json` | | Output JSON |

```bash
drbrain stats
drbrain stats --workspace nlp
```

### `drbrain delete`

Delete a paper and all its associated data from the graph.

| Flag | Short | Description |
|------|-------|-------------|
| `--force` | `-f` | Skip confirmation prompt |
| `--rm-files` | | Also delete paper directory |
| `--json` | | Output JSON |

```bash
drbrain delete p6a321e --force
drbrain delete p6a321e --rm-files --force
```

### `drbrain repair`

Repair paper metadata via CrossRef, arXiv, and OpenAlex APIs. Fixes missing DOIs, author names, journal info, and more.

| Flag | Short | Description |
|------|-------|-------------|
| `--all` | | Repair all papers |
| `--workspace` | `-w` | Limit to workspace |
| `--dry-run` | | Preview only, no changes |
| `--json` | | Output JSON |

```bash
drbrain repair p6a321e
drbrain repair --all --dry-run
drbrain repair --workspace nlp --json
```

### `drbrain translate`

Translate a paper's markdown via LLM with placeholder-protected chunking, concurrent translation, and resume-from-interruption.

| Flag | Short | Description |
|------|-------|-------------|
| `--lang` | `-l` | Target language code: `zh`, `en`, `ja`, etc. (default: `zh`) |
| `--force` | `-f` | Force re-translation even if output exists |
| `--json` | | Output JSON |

```bash
drbrain translate p6a321e --lang zh
drbrain translate p6a321e --lang en --force
drbrain translate p6a321e --lang ja --json
```

### `drbrain backup`

Create tar.gz backups or sync to rsync remote targets.

| Flag | Short | Description |
|------|-------|-------------|
| `--output` | `-o` | Custom output path (tar.gz mode) |
| `--list` | | List existing backups and rsync targets |
| `--target` | `-t` | Rsync backup target name |
| `--dry-run` | | Rsync dry-run (no transfer) |
| `--json` | | Output JSON |

```bash
drbrain backup                                    # tar.gz local backup
drbrain backup -o ~/backups/drbrain.tar.gz
drbrain backup --list                              # list local + remote targets
drbrain backup --target myserver                   # rsync to remote
drbrain backup --target myserver --dry-run         # preview rsync
```

### `drbrain enrich`

Enrich paper metadata from CrossRef and detect scrub-worthy records.

| Flag | Short | Description |
|------|-------|-------------|
| `--all` | | Check all papers |
| `--dry-run` | | Check without backfilling |
| `--json` | | Output JSON |

```bash
drbrain enrich p6a321e
drbrain enrich p6a321e --dry-run
drbrain enrich --all
drbrain enrich --all --dry-run --json
```

### `drbrain document`

Inspect an Office document (DOCX, PPTX, XLSX) — structured text summary.

| Flag | Short | Description |
|------|-------|-------------|
| `--format` | `-f` | Override format detection (`docx`, `pptx`, `xlsx`) |

```bash
drbrain document report.docx
drbrain document presentation.pptx
drbrain document data.xlsx --format xlsx
```

### `drbrain metrics`

Show user behavior analytics — top keywords, most-read papers, weekly trends.

| Flag | Short | Description |
|------|-------|-------------|
| `--json` | | Output JSON |

```bash
drbrain metrics
drbrain metrics --json
```

### `drbrain queue`

List all pending confidence queue items (concepts, aliases, or relations flagged for human review).

```bash
drbrain queue
drbrain queue --json
```

### `drbrain queue resolve`

Resolve a queue item: accept or reject.

| Flag | Description |
|------|-------------|
| `--accept` | Accept the queue item |
| `--reject` | Reject the queue item |
| `--json` | Output JSON |

```bash
drbrain queue resolve 42 --accept
drbrain queue resolve 42 --reject
```

### `drbrain queue resolve-all`

Batch resolve all pending queue items.

| Flag | Description |
|------|-------------|
| `--accept` / `--reject` | Accept or reject all |
| `--type` | Filter by item type (concept, alias, relation) |
| `--max-conf` | Only process items with confidence <= this value |
| `--json` | Output JSON |

```bash
drbrain queue resolve-all --accept --type concept --max-conf 0.3
drbrain queue resolve-all --reject --type alias
```

---

## Pipeline

### `drbrain pipeline`

Chain multiple processing steps in sequence via presets or custom step lists.

| Flag | Short | Description |
|------|-------|-------------|
| `--preset` | `-p` | Preset: `full`, `quick`, `embed` |
| `--steps` | `-s` | Comma-separated step names |
| `--list` | | List available steps and presets |
| `--dry-run` | | Preview steps without executing |

```bash
drbrain pipeline --preset full
drbrain pipeline --preset quick
drbrain pipeline --steps build,embed
drbrain pipeline --preset full --dry-run
drbrain pipeline --list
```

**Available steps:** `ingest`, `build`, `embed`, `closure`

---

## Meetings and Discovery

### `drbrain proceedings`

Manage conference proceedings — create, list, show, and associate papers.

| Flag | Short | Description |
|------|-------|-------------|
| `--list` | `-l` | List all proceedings |
| `--create` | | Create proceeding: `"Name Year"` |
| `--show` | | Show proceeding by ID |
| `--add` | | Add paper: `PROCEEDING_ID PAPER_ID` |
| `--json` | | Output JSON |

```bash
drbrain proceedings --list
drbrain proceedings --create "NeurIPS 2024"
drbrain proceedings --show abc12345
drbrain proceedings --add abc12345 p6a321e
```

### `drbrain explore`

Manage explore silos — lightweight literature discovery collections.

| Flag | Short | Description |
|------|-------|-------------|
| `--list` | `-l` | List all silos |
| `--create` | | Create a new silo |
| `--delete` | | Delete a silo |
| `--name` | `-n` | Silo name for `--show` or `--search` |
| `--show` | | Show silo papers |
| `--search` | `-s` | Search papers within a silo |
| `--json` | | Output JSON |

```bash
drbrain explore --list
drbrain explore --create transformers
drbrain explore --name transformers --show
drbrain explore --name transformers --search "attention"
drbrain explore --delete transformers
```

---

## Workspace Management (`drbrain ws`)

Workspaces are named subsets of papers for focused analysis.

### `drbrain ws create`

```bash
drbrain ws create nlp --description "NLP papers"
```

### `drbrain ws list`

```bash
drbrain ws list
```

### `drbrain ws show`

```bash
drbrain ws show nlp
drbrain ws show nlp --json
```

### `drbrain ws add`

```bash
drbrain ws add nlp p6a321e p3b452c
```

### `drbrain ws remove`

```bash
drbrain ws remove nlp p3b452c
```

### `drbrain ws rename`

```bash
drbrain ws rename nlp nlp-transformers
```

### `drbrain ws delete`

```bash
drbrain ws delete nlp
```
