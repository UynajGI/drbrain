# Pipeline Refactor: Lightweight Ingest + Multi-Stage Graph Build

## Scope

Separate the monolithic ingest pipeline into two phases:
- **Phase 1 — `drbrain ingest`**: lightweight PDF-to-library (parse + metadata + tree.json). No concept extraction. LLM only for tree structuring (PageIndex).
- **Phase 2 — `drbrain build [paper_id...]`**: graph construction with 5-stage LLM extraction pipeline based on [2306.08302](https://arxiv.org/abs/2306.08302) and [2511.11017](https://arxiv.org/abs/2511.11017).

## Non-Goals

- Backward compatibility with old `--full` mode
- Real-time collaborative editing
- Changes to graph query/closure/search (these consume the improved graph)

---

## 1. Architecture

```
drbrain ingest                    drbrain build [paper_id...]
──────────────────                ──────────────────────────────
PDF → MinerU / pymupdf4llm        Load paper(s) with raw.md + tree.json
  → raw.md                        ──────────────────────────────
  → tree.json (LLM summaries)     Stage 1: Ontology Extension
  → metadata (_resolve_metadata)    LLM reviews tree.json → suggests
  → paper record (DB)               subcategories under TBox 6 types
                                    e.g. Method → attention, GNN, RL
                                  ──────────────────────────────
                                  Stage 2: Entity Extraction
                                    Per leaf node: LLM extracts
                                    concepts with subcategory labels
                                  ──────────────────────────────
                                  Stage 3: Relation Extraction
                                    LLM links concepts across sections
                                    using TBox-allowed relations
                                  ──────────────────────────────
                                  Stage 4: Coreference Resolution
                                    LLM merges duplicate entities
                                    (same concept, different labels)
                                  ──────────────────────────────
                                  Stage 5: Iterative Refinement
                                    LLM reviews own output, flags
                                    contradictions, corrects errors
                                  ──────────────────────────────
                                  Validate → Insert → Closure
```

## 2. CLI Interface

### `drbrain ingest` (modified)

```bash
drbrain ingest [paths...]    # Default: data/spool/inbox/
```

Removes: LLM concept extraction, argument extraction, validation, closure, report generation.
Keeps: PDF parsing, metadata enrichment, tree structuring (LLM), paper DB insert.

Output:
```
Parsing: 1706.03762v7.pdf
  Title: Attention Is All You Need
  Year: 2017
  arXiv: 1706.03762
  Sections: 2 high-signal blocks
  [new] p1a2b3c
  Document tree: 26 sections -> tree.json
  Ingested: p1a2b3c
```

### `drbrain build` (new command)

```bash
drbrain build [paper_id...]  # Omit for all unprocessed papers
```

| Flag | Default | Description |
|---|---|---|
| `--papers` / `-p` | (all unprocessed) | Specific paper IDs |
| `--skip-refine` | false | Skip iterative refinement stage |
| `--json` | false | JSON output |

Output:
```
Building graph for 1 paper(s)...

p1a2b3c: Attention Is All You Need
  Stage 1: Ontology... 3 subcategories (Method: attention, positional encoding, feed-forward)
  Stage 2: Entities...   157 concepts extracted
  Stage 3: Relations...   66 edges
  Stage 4: Coreference... 12 merges
  Stage 5: Refine...      3 corrections

Valid items: 145 | Rejected: 8
Concepts inserted: 145 | Edges: 58
```

## 3. Data Flow

### Stage 1: Ontology Extension

**Input**: tree.json structure (section titles + summaries)
**LLM Prompt**: "Given this paper's structure, suggest domain-specific subcategories for these concept types: Problem, Method, Conclusion, Gap, Debate, Actor. Return JSON."
**Output**: `{type: [subcategory_label, ...]}` dict, merged into TBox for this build session.
**Multi-paper**: when building 2+ papers together, ontology is shared across all.

### Stage 2: Entity Extraction

**Input**: per leaf node content + ontology from Stage 1
**LLM Prompt**: Extract concepts of each type, with optional subcategory from the ontology.
**Output**: concepts with `{label, type, subcategory, confidence}`

### Stage 3: Relation Extraction

**Input**: all extracted entities + tree structure
**LLM Prompt**: Connect entities using allowed TBox relations. Target entity must exist or be newly created.
**Output**: relations with `{head, rel, tail, confidence}`

### Stage 4: Coreference Resolution

**Input**: all extracted entities
**LLM Prompt**: "These concepts may refer to the same thing. Merge duplicates."
**Output**: merge map `{canonical_label: [variant_labels]}`

### Stage 5: Iterative Refinement

**Input**: full extraction result (entities + relations + merges)
**LLM Prompt**: "Review this knowledge graph extraction. Find contradictions, redundancies, or missing relations. Output corrections."
**Output**: corrections as add/remove/modify operations.

## 4. Database Changes

New `paper_status` enum value: `extracted` (in addition to `uploaded`, `placeholder`, `merged`).

```
uploaded   → paper ingested (has raw.md + tree.json + metadata)
extracted  → concepts + edges built (has graph data)
placeholder → citation-only placeholder (existing)
merged     → merged with another paper (existing)
```

`drbrain build` only processes papers with `status = 'uploaded'` (has content, no graph yet).

## 5. Files Touched

| File | Change |
|---|---|
| `cli/commands.py` | Trim `ingest_cmd` (remove extraction+validation+closure); add `build_cmd` |
| `extractor/concept.py` | Add 5-stage pipeline (`build_graph_from_tree`, `_extract_ontology`, `_extract_entities`, `_extract_relations`, `_resolve_coreferences`, `_refine`) |
| `extractor/llm_client.py` | May need `acall_text_with_fallback` tweaks for new prompts |
| `prompts/` | New prompts: ontology.txt, extract_entities.txt, extract_relations.txt, coreference.txt, refine.txt |
| `storage/database.py` | Add `extracted` status support |
| `validator/schema.py` | Add subcategory field support (optional) |

## 6. Testing

- `ingest` produces paper with raw.md + tree.json, no concepts
- `build` on single paper produces concepts + edges
- `build` on multiple papers shares ontology
- `build --skip-refine` skips stage 5
- `build` only processes papers with status=`uploaded`
- TBox violations correctly rejected
- Coreference merges correctly update concept labels
