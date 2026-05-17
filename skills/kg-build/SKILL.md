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

Build the knowledge graph from ingested papers in three stages: LLM extraction (`build`), TransE
embedding training (`embed`), and rule-based closure inference (`closure`). After this pipeline,
concepts, arguments, edges, and inferred relations are ready for query and analysis.

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
drbrain build                  # all unprocessed papers
drbrain build p3f8a2 p7b1c4    # specific papers
drbrain build --all            # rebuild everything
drbrain build --skip-refine    # skip refinement (faster, lower quality)
```

Status changes: `uploaded` → `extracted`. Check `drbrain stats` to confirm.

### Step 2: Train graph embeddings

Train TransE embeddings on the extracted graph for link prediction and complex queries:

```bash
drbrain embed                  # default: dim=128, epochs=100
drbrain embed --dim 256 --epochs 200
drbrain embed --retrain        # force retrain even if embeddings exist
```

Embeddings enable `graph query` (complex ∧∨¬ queries) and `closure --mode hybrid`.

### Step 3: Train text embeddings (optional but recommended)

Generate PageIndex tree node + RAPTOR recursive summary embeddings for tree retrieval:

```bash
drbrain embed --tree
```

Enables `query --paper <id>` (tree-based section retrieval).

### Step 4: Run graph closure

Infer new edges via rule-based reasoning. Symbolic mode uses 8 hard rules; hybrid mode adds 4
embedding-based rules (requires `drbrain embed` first):

```bash
drbrain closure                          # symbolic, all rules
drbrain closure --mode hybrid            # embedding-aware inference
drbrain closure --dry-run                # preview without persisting
drbrain closure --rule extends --rule replaces  # specific rules only
drbrain closure --mine-rules             # mine path rules from TransE embeddings
drbrain closure --ground                 # ground transitive rules as concrete triples
drbrain closure -w my-workspace          # scope to workspace
```

## Full pipeline

```bash
drbrain ingest                    # Step 0: add papers (see paper-ingest skill)
drbrain build                     # Step 1: extract
drbrain embed --tree              # Step 2-3: embeddings
drbrain closure --mode hybrid     # Step 4: inference
```

## Examples

**Build everything from scratch:**
```bash
drbrain build --all && drbrain embed --retrain --tree && drbrain closure --mode hybrid
```

**Incremental update after adding papers:**
```bash
drbrain build              # only unprocessed
drbrain embed --retrain    # retrain with new entities
drbrain closure            # re-run inference
```

**Preview inferred edges before committing:**
```bash
drbrain closure --mode hybrid --dry-run --json | jq '.'
```

## Next Steps

After building the KG, you can:

- **Search**: `drbrain query` (see `paper-query` skill)
- **Ask questions**: `drbrain ask`, `drbrain reason` (see `kg-reason` skill)
- **Analyze**: `drbrain seed`, `drbrain evolve`, `drbrain frontier` (see `knowledge-cartography` skill)
- **Explore graph**: `drbrain graph neighbors/path/related` (see `graph` skill)
- **Audit quality**: `drbrain audit` (see `audit` skill)

## CLI Reference

| Command | What it does |
|---------|--------------|
| `drbrain build` | Extract from all unprocessed papers |
| `drbrain build <id...>` | Extract from specific papers |
| `drbrain build --all` | Rebuild all papers |
| `drbrain build --skip-refine` | Skip iterative refinement |
| `drbrain embed` | Train TransE graph embeddings |
| `drbrain embed --tree` | Train PageIndex+RAPTOR text embeddings |
| `drbrain embed --retrain` | Force retrain |
| `drbrain embed --dim 256 --epochs 200` | Custom training params |
| `drbrain closure` | Symbolic rule inference (8 rules) |
| `drbrain closure --mode hybrid` | Symbolic + embedding rules (12 rules) |
| `drbrain closure --dry-run` | Preview without persisting |
| `drbrain closure --mine-rules` | Mine path rules from embeddings |
| `drbrain closure --ground` | Ground transitive rules |
| `drbrain closure --rule X` | Run specific rule(s) only |
| `drbrain closure -w <ws>` | Scope to workspace |
