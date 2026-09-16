---
name: kg-build
description: >
  Build the knowledge graph from ingested papers — 5-stage LLM extraction, TransE graph embeddings,
  and rule-based closure inference. Use this skill whenever the user wants to process ingested papers,
  build the knowledge graph, train embeddings, run inference, or needs to regenerate the graph after
  adding new papers. Also use when the user asks "process my papers", "extract concepts", "build the
  graph", "train embeddings", "run closure", "infer new edges", or needs the KG populated before
  querying or analysis. This is the mandatory second step after ingest — papers must be built before
  they can be searched or analyzed. Trigger proactively when the user has ingested papers and wants
  to work with the knowledge graph.
---

# KG Build

Build the knowledge graph from ingested papers in three stages: LLM extraction (`graph build`), TransE
embedding training (`graph embed`), and rule-based closure inference (`graph closure`). After this
pipeline, concepts, arguments, edges, and inferred relations are ready for query and analysis.

## Prerequisites

Papers must be ingested first (status: `uploaded`). Check with:

```bash
drbrain list
drbrain stats
```

## Workflow

### Step 1: Extract concepts and relations

5-stage LLM extraction: ontology extension → entity extraction (10-way concurrent) → relation
extraction → coreference resolution → iterative refinement.

```bash
drbrain graph build                  # all unprocessed papers
drbrain graph build p3f8a2 p7b1c4    # specific papers
drbrain graph build --all            # rebuild everything
drbrain graph build --skip-refine    # skip refinement (faster, lower quality)
```

Status changes: `uploaded` → `extracted`. Check `drbrain stats` to confirm.

### Step 2: Train graph embeddings

Train TransE embeddings on the extracted graph for link prediction and complex queries:

```bash
drbrain graph embed                  # default: dim=128, epochs=100
drbrain graph embed --dim 256 --epochs 200
drbrain graph embed --retrain        # force retrain even if embeddings exist
```

Embeddings enable `graph query` (complex ∧∨¬ queries) and `graph closure --mode hybrid`.

### Step 3: Prepare the searchable index (optional but recommended)

`index build` prepares and publishes the searchable index in one incremental run: lexical BM25,
canonical FTS, shared text vectors and the unified tree hierarchy (text embeddings for retrieval):

```bash
drbrain index build
```

Enables `search` / `search --paper <id>` (paper-scoped evidence retrieval) and `ask`.

### Step 4: Run graph closure

Infer new edges via rule-based reasoning. Symbolic mode uses 8 hard rules; hybrid mode adds 4
embedding-based rules (requires `drbrain graph embed` first):

```bash
drbrain graph closure                          # symbolic, all rules
drbrain graph closure --mode hybrid            # embedding-aware inference
drbrain graph closure --dry-run                # preview without persisting
drbrain graph closure --rule extends --rule replaces  # specific rules only
drbrain graph closure --mine-rules             # mine path rules from TransE embeddings
drbrain graph closure --ground                 # ground transitive rules as concrete triples
drbrain graph closure -w my-workspace          # scope to workspace
```

## Full pipeline

```bash
drbrain ingest                    # Step 0: add papers (see paper-ingest skill)
drbrain graph build               # Step 1: extract
drbrain index build               # Step 2-3: searchable index (vectors + tree)
drbrain graph closure --mode hybrid     # Step 4: inference
```

## Examples

**Build everything from scratch:**
```bash
drbrain graph build --all && drbrain index build --force && drbrain graph closure --mode hybrid
```

**Incremental update after adding papers:**
```bash
drbrain graph build              # only unprocessed
drbrain graph embed --retrain    # retrain with new entities
drbrain graph closure            # re-run inference
```

**Preview inferred edges before committing:**
```bash
drbrain graph closure --mode hybrid --dry-run --json | jq '.'
```

## Next Steps

After building the KG, you can:

- **Search**: `drbrain search` (see `paper-query` skill)
- **Ask questions**: `drbrain ask`, `drbrain reason` (see `kg-reason` skill)
- **Analyze**: `drbrain seed`, `drbrain evolve`, `drbrain frontier` (see `knowledge-cartography` skill)
- **Explore graph**: `drbrain graph neighbors/path/related` (see `graph` skill)
- **Audit quality**: `drbrain audit` (see `audit` skill)

## CLI Reference

| Command | What it does |
|---------|--------------|
| `drbrain graph build` | Extract from all unprocessed papers |
| `drbrain graph build <id...>` | Extract from specific papers |
| `drbrain graph build --all` | Rebuild all papers |
| `drbrain graph build --skip-refine` | Skip iterative refinement |
| `drbrain graph embed` | Train TransE graph embeddings |
| `drbrain index build` | Prepare the searchable index (lexical + FTS + shared vectors + tree) |
| `drbrain graph embed --retrain` | Force retrain |
| `drbrain graph embed --dim 256 --epochs 200` | Custom training params |
| `drbrain graph closure` | Symbolic rule inference (8 rules) |
| `drbrain graph closure --mode hybrid` | Symbolic + embedding rules (12 rules) |
| `drbrain graph closure --dry-run` | Preview without persisting |
| `drbrain graph closure --mine-rules` | Mine path rules from embeddings |
| `drbrain graph closure --ground` | Ground transitive rules |
| `drbrain graph closure --rule X` | Run specific rule(s) only |
| `drbrain graph closure -w <ws>` | Scope to workspace |
