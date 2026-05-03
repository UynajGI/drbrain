# Pipeline Refactor: Lightweight Ingest + Multi-Stage Graph Build

## Scope

Separate the monolithic ingest pipeline into two phases:
- **Phase 1 — `drbrain ingest`**: lightweight PDF-to-library (parse + metadata + tree.json). No concept extraction. LLM only for tree structuring (PageIndex).
- **Phase 2 — `drbrain build [paper_id...]`**: graph construction with 5-stage LLM extraction pipeline.

## 1. Architecture

```
drbrain ingest                    drbrain build [paper_id...]
──────────────────                ──────────────────────────────
PDF → MinerU / pymupdf4llm        Load paper(s) with raw.md + tree.json
  → raw.md                        ──────────────────────────────
  → tree.json (LLM summaries)     Stage 1: Ontology Extension
  → metadata (_resolve_metadata)    LLM suggests subcategories under
  → paper record (DB)               TBox 6 types. Multi-paper shared.
                                  ──────────────────────────────
                                  Stage 2: Entity Extraction
                                    Per leaf node: LLM extracts
                                    concepts with subcategory labels
                                  ──────────────────────────────
                                  Stage 3: Relation Extraction
                                    LLM links concepts across sections
                                  ──────────────────────────────
                                  Stage 4: Coreference Resolution
                                    LLM merges duplicate concept labels
                                  ──────────────────────────────
                                  Stage 5: Iterative Refinement
                                    LLM reviews output, corrects errors
                                  ──────────────────────────────
                                  Validate → Insert → Closure
```

## 2. CLI Interface

### `drbrain ingest` (modified)

Removes: LLM concept/argument extraction, validation, closure, report.
Keeps: PDF parsing, metadata enrichment, tree structuring (LLM), paper DB insert.

### `drbrain build` (new)

```bash
drbrain build [paper_id...]  # Omit for all unprocessed (status=uploaded)
```

| Flag | Default | Description |
|---|---|---|
| `--skip-refine` | false | Skip stage 5 |
| `--json` | false | JSON output |

## 3. Five Stages

### Stage 1: Ontology Extension
LLM reads tree.json → suggests subcategories under 6 TBox types.
e.g. Method → attention, GNN, RL. Multi-paper: shared ontology.

### Stage 2: Entity Extraction
Per leaf node: LLM extracts concepts with subcategory labels + confidence.

### Stage 3: Relation Extraction
LLM links concepts using TBox relations. Target must be existing or created concept.

### Stage 4: Coreference Resolution
LLM merges duplicate entity labels across sections.

### Stage 5: Iterative Refinement
LLM self-reviews extraction, flags contradictions, outputs corrections.

## 4. Database

New paper status: `extracted`. Build only processes `uploaded` papers.

## 5. Files

| File | Change |
|---|---|
| `cli/commands.py` | Trim ingest_cmd, add build_cmd |
| `extractor/concept.py` | Add 5-stage pipeline functions |
| `prompts/` | New: ontology, entities, relations, coreference, refine |
| `storage/database.py` | `extracted` status |
| `validator/schema.py` | Subcategory field (optional) |
