# Architecture

## Philosophy

DrBrain is **symbol-driven with lightweight vectors**. BM25 and rule-based symbolic reasoning form the core. Vectors are used only for semantically-complete tree nodes (PageIndex sections, RAPTOR summaries) to enhance retrieval -- never for arbitrary text chunks. There is no vector database dependency. `provider=none` disables vectors entirely, falling back to BM25 + LLM navigation.

Every design decision follows from a single principle: **the knowledge graph is the source of truth**. Concepts, relations, and inference rules are explicit, auditable, and human-readable. Vectors serve retrieval, not knowledge representation.

---

## System Overview

```
PDF → [Phase 1: Ingest] → Markdown + Metadata + tree.json
       ↓
     [Phase 2: Build] → Concepts + Relations → Knowledge Graph
       ↓
     [Reasoning Stack] → Causal Chains, Seeds, Hypotheses, Isomorphisms
       ↓
     [Search] → BM25 + Graph-enhanced + PageIndex Tree Retrieval
```

---

## Ingestion Pipeline

### Phase 1: Ingest (Lightweight)

The `drbrain ingest` command runs a lightweight pipeline that extracts the paper's structure and metadata. No concept extraction happens here.

```
PDF → MinerU/PyMuPDF → Markdown
    → 5-source metadata cross-validation → paper_ids + papers tables
    → LLM tree structuring → tree.json
    → Status: uploaded
```

**Steps:**

1. **Parse** (`parser/mineru_parser.py`): MinerU CLI converts PDF to Markdown. Falls back to `pymupdf4llm.to_markdown()` when MinerU is unavailable. PDFs over 150 pages are split into chunks.

2. **Identify** (`dedup/resolver.py`, `parser/mineru_parser.py:_resolve_metadata`): Cross-validates metadata from 5 sources -- arXiv, CrossRef, Semantic Scholar, OpenAlex, DeepXiv. Stores `title`, `year`, `doi`, `arxiv`, `s2_id`, `openalex_id` in `paper_ids`, and `journal`, `publisher`, `citation_count` in `papers`. Extracts abstract from `tree.json`.

3. **Tree** (`parser/pageindex_parser.py`): LLM structures the markdown into a hierarchical tree with section summaries. Each node has a title, summary, and optional content. The tree is stored as `papers/<id>/tree.json`.

4. **Record**: Paper inserted with status `uploaded`.

### Phase 2: Build (5-Stage LLM Extraction)

The `drbrain build` command runs a 5-stage LLM pipeline that extracts structured knowledge from the paper's section tree.

```
tree.json → [Stage 1: Ontology Extension]
          → [Stage 2: Entity Extraction]  (10-way concurrent)
          → [Stage 3: Relation Extraction]
          → [Stage 4: Coreference Resolution]
          → [Stage 5: Iterative Refinement] (skippable)
          → Status: extracted
```

**Stage 1 -- Ontology Extension:** The LLM suggests domain-specific subcategories under the 6 TBox types. For example, under `Method` it might suggest `OptimizationAlgorithm`, `RegularizationTechnique`, or `ArchitectureDesign`.

**Stage 2 -- Entity Extraction:** Per leaf-node concept extraction with subcategory labels. Runs 10 leaf nodes concurrently for throughput.

**Stage 3 -- Relation Extraction:** The LLM connects concepts using TBox-defined relations (e.g., `addresses`, `extends`, `challenges`, `solves`). Relations are typed and directed.

**Stage 4 -- Coreference Resolution:** The LLM identifies duplicate entity labels across sections and merges them.

**Stage 5 -- Iterative Refinement:** The LLM self-reviews the extraction for contradictions and errors. Skippable via `--skip-refine` to save time and API cost.

Paper status transitions: `uploaded` -> `extracted` (after successful build). Papers with status `placeholder` are citation-only records that haven't been ingested.

---

## Knowledge Graph

### TBox (Type-Level)

The type-level ontology defines 6 concept types:

| Type | Description |
|------|-------------|
| **Problem** | A research problem or challenge |
| **Method** | A technique, algorithm, or approach |
| **Conclusion** | A concluding insight or takeaway |
| **Gap** | An identified gap in the literature |
| **Debate** | A controversy or disagreement in the field |
| **Actor** | A person, organization, or research group |

Edge relations between concepts:

| Relation | Meaning |
|----------|---------|
| `addresses` | Method addresses a Problem |
| `leaves_open` | Approach leaves a Gap open |
| `points_to` | Concept points to future work or another concept |
| `proposes` | Method or Actor proposes something |
| `extends` | Method extends an existing method |
| `replaces` | Method replaces an older method |
| `solves` | Method solves a Problem |
| `supports` | Concept supports another |
| `challenges` | Concept challenges another |
| `limits` | Concept limits/restricts another |
| `constrains` | Concept constrains another |
| `affiliated_with` | Actor is affiliated with an institution |
| `contains` | Section contains a sub-section or concept (structural, auto-generated) |

### RBox (Relation-Level)

The relation-level inference system has 11 rules:

**Rules in `closure()`:**
- `transitive_closure` -- if A→B and B→C then A→C
- `creates_debate` -- if A supports X and B challenges X, creates a debate
- `gap_addressed` -- if a Gap has leaves_open and addresses, it is addressed
- `indirect_evolution` -- method lineage through extends + replaces chains
- `gap_to_debate` -- a Gap pointing to a debated target
- `shared_actor` -- papers sharing an Actor create implicit connections
- `asymmetric_violations` -- detects symmetry violations (supports vs challenges) (logged, not inferred)

**Rules in `apply_path_rules()`:**
- `method_supersedes_problem` -- a method that solves a problem supersedes it
- `challenge_chain` -- transitive challenge propagation
- `gap_inheritance` -- gaps propagate through extends relations
- `indirect_support` -- multi-hop support chains

**Hybrid closure:** The `--mode hybrid` variant weights inferred edges by TransE embedding scores. Path confidence is computed via relation composition distance.

### 3-Layer Reasoning Stack

The reasoning stack is inspired by the architecture described in papers 2202.07412, 2306.08302, and 2511.11017.

```
Layer 1: TransE Embeddings
  - Train entity/relation vectors (drbrain embed)
  - Link prediction: predict_link()
  - Entity similarity: similar_entities()
  - Storage: embeddings SQLite table

Layer 2: Hybrid Closure
  - Rule-based inference (symbolic + hybrid)
  - Confidence-weighted edges
  - T-norm transitive path materialization
  - Embedding-driven path rule mining (--mine-rules)

Layer 3: LLM Agent Reasoning
  - Tool-calling agent: search_concepts, get_neighbors, find_path
  - Bidirectional mode: hypothesis formation -> KG validation -> revision loop
  - Hypothesis generation from gap/debate/technology-cliff patterns
```

---

## Reasoning Modules

### Causal Chains
`extractor/causal_chain.py` -- `build_causal_chains()`, `find_chains_from()`, `find_path()`. Traces causal relationships through the graph: which concepts cause (or are caused by) which other concepts, with multi-hop chain building.

### Confidence Propagation
`extractor/confidence_propagation.py` -- Multi-hop confidence decay with default decay factor 0.85. Section-aware variant weights confidence differently based on which section the concept came from.

### Counterfactual Analysis
`extractor/counterfactual.py` -- Node removal impact analysis. What happens to the graph if a concept is removed? Measures connectivity impact and identifies critical nodes.

### Cross-Domain Isomorphism
`extractor/isomorphism.py` -- Subgraph similarity by relation signature. Finds structurally similar subgraphs across domains, enabling cross-domain knowledge transfer.

### Hypothesis Generation
`extractor/hypothesis.py` -- Generates actionable research hypotheses from gaps, debates, technology cliffs, and confidence collapse patterns.

### Knowledge Genealogy
`graph/genealogy.py` -- Concept lineage trees (`evolve`), academic offspring tracking (`descendants`), domain landscape with timeline/gaps/debates (`landscape`), paradigm shift detection (`paradigm`), cross-domain method transfer discovery (`transfers`). All five features include PageIndex provenance (`[source: <section> of <paper>]`) via `_get_concept_provenance()`. Text tree and Mermaid renderers show provenance inline.

### Structure-First Retrieval
`query/tree_retrieval.py` -- Full PageIndex implementation. Iterative tree-search with adaptive depth navigation. Small skeletons get one-shot selection; large skeletons get top-level -> branch selection -> leaf selection.

### RAPTOR Recursive Semantic Tree
`extractor/raptor.py` -- Implements RAPTOR (2401.18059). Recursive embedding → UMAP → GMM+BIC clustering → LLM summarization on PageIndex leaf nodes. Builds multi-layer summary tree with `source_node_ids` provenance chains. Stored in `tree_summaries` and `tree_vectors` tables.

### Rule Mining
`extractor/rule_miner.py` -- Mines path rules from TransE relation vectors. Activated via `drbrain closure --mine-rules`. Discovers patterns like "if X addresses Y and Y extends Z, then X is likely to address Z."

---

## Search

### BM25 (`query/bm25.py`)
Standard BM25 over concept labels, arguments, and paper metadata. No vector embeddings required. The index is rebuilt via `drbrain index`.

### Graph-Enhanced Search
- `--neighbors N`: After BM25 retrieval, expand by N hops of directed graph traversal. Results include `_via_graph`, `_source_seed`, `_distance`, `_path` metadata.
- `--hybrid`: Applies multiplicative PageRank boost [1.0, 2.0] to re-rank BM25 results by graph centrality.

### PageIndex Tree Retrieval
- `--paper <id>`: Bypasses BM25. Performs hierarchical tree search on a specific paper's section tree, using LLM-guided branch/leaf selection.
- Collapsed tree mode (planned): flatten all tree nodes (PageIndex + RAPTOR summaries), embed query, cosine similarity retrieval. Replaces LLM navigation when vectors are available.

### RAPTOR Semantic Tree
`extractor/raptor.py` — Recursive embedding → UMAP → GMM+BIC clustering → LLM summarization on PageIndex leaf nodes. Multi-layer summary tree with provenance chains. Collapsed tree retrieval across papers (Layer 4, planned). Inspired by RAPTOR (2401.18059).

### Embedding Queries (`graph query`)
Complex queries over TransE embeddings: projection, intersection, union, negation. Requires `drbrain embed`.

### Lightweight Text Embeddings (planned)
`drbrain embed` will also generate SBERT embeddings for tree nodes (PageIndex leaves + RAPTOR summaries). Stored in `tree_vectors` table. FAISS IndexFlatIP for cosine similarity search. Reference: ScholarAIO embedding engine (Qwen3-Embedding-0.6B, local inference).

---

## Data Layout

```
data/
├── spool/
│   ├── inbox/            # PDFs awaiting ingest (auto-classified)
│   └── pending/          # Failed ingests + pending.jsonl
├── papers/<id>/          # Per-paper directory
│   ├── source.pdf        # Original PDF
│   ├── raw.md            # Parsed markdown (MinerU/PyMuPDF)
│   ├── tree.json         # Structured section tree (PageIndex)
│   └── images/           # Extracted figures and tables
├── drbrain.db            # Main SQLite database (WAL mode)
├── metrics.db            # LLM token usage tracking
├── cache/                # API response cache (rebuildable)
├── logs/                 # Application logs (loguru, rotating)
├── backups/              # tar.gz backups
└── reports/              # Per-paper JSON analysis reports

workspace/<name>/         # Paper subsets
├── workspace.yaml        # Metadata
└── refs/papers.json      # Paper list

config.yaml               # Base config (checked in)
config.local.yaml         # Local overrides + secrets (gitignored)
```

### Database Schema (`storage/database.py`)

Key tables:
- `papers` -- title, year, journal, paper_type, status, abstract, citation_count
- `paper_ids` -- doi, arxiv, s2_id, openalex_id (cross-reference)
- `concepts` -- label, type, confidence, section, node_id, source_paper
- `edges` -- src_id, dst_id, relation, source_paper, confidence, node_id, section
- `arguments` -- claim, claim_type, target, section, node_id, source_paper
- `aliases` -- canonical_id, variant (for dedup)
- `embeddings` -- TransE entity/relation vectors
- `tree_vectors` -- per-node embeddings (node_id, paper_id, embedding BLOB, content_hash, tree_layer)
- `tree_summaries` -- RAPTOR recursive summaries (node_id, paper_id, summary_text, source_node_ids, tree_layer)
- `vector_metadata` -- embedding signature tracking (key, value)
- `citation_cache` -- expanded citations from APIs
- `queue` -- pending confidence items for human review
- `build_stages` -- per-paper pipeline stage status (paper_id, stage, status, result_json) for agent idempotency
- `schema_versions` -- versioned migrations

The database uses **WAL mode** for concurrent read/write access. Schema migrations are stored in `schema_versions` and applied automatically.

### Atomic Writes

All file writes use the **tmp -> rename** pattern for crash safety:
1. Write to `<path>.tmp`
2. `os.rename(<path>.tmp, <path>)`

---

## Key Design Decisions

### Lightweight Vectors for Retrieval
Vectors are used only for semantically-complete tree nodes (PageIndex sections, RAPTOR summaries) to accelerate retrieval. Never for arbitrary text chunks. `provider=none` disables all vectors, falling back to pure BM25 + LLM navigation. Embeddings are stored in SQLite alongside everything else -- no separate vector database.

### SQLite with WAL
A single SQLite file with WAL mode is the only database. Simple, portable, no server needed. Concurrent reads work well under WAL mode. Appropriate for a personal research tool.

### Atomic Writes (tmp -> rename)
Throughout the codebase, file writes go to a temporary file first, then atomically rename into place. This prevents partial writes from corrupting state.

### Typed Config
`Config` is a typed dataclass (`config.py`) with sub-configs: `LLMConfig`, `MinerUConfig`, `APIConfig`, `DirsConfig`, `DBConfig`, `ExtractConfig`, `BM25Config`, `QueueConfig`. Loaded from `config.yaml` (checked in) overlaid by `config.local.yaml` (gitignored, contains secrets). Environment variable placeholders `${VAR_NAME}` are resolved at load time.

### LLM Fallback Chain
`acall_with_fallback()` iterates through the configured model list. First successful parse wins. Returns `None` if all models are exhausted. Supports any litellm provider (OpenAI, Anthropic, Ollama, plus OpenAI-compatible endpoints like DeepSeek, Zhipu, Bailian).

### Section Provenance
The `section` field flows from LLM extraction through the database and into all reasoning layers. This enables section-aware confidence decay, counterfactual weighting, isomorphism signatures, and hypothesis evidence grounding.

### Symbol-Driven Reasoning
Graph closure rules, transitive closure, asymmetric detection, causal chains, confidence propagation, counterfactuals, and isomorphism detection are all rule-based. Zero embeddings required for core reasoning.

### Agent-Based Pipeline
The 5-stage LLM pipeline (`extractor/agent.py`) wraps each stage as a dedicated `BuildAgent` subclass (OntologyAgent, EntityAgent, RelationAgent, CorefAgent, RefineAgent). Agents have independent system prompts, input/output validation contracts, and idempotency guards via `build_stages` DB table. Inspired by 2511.11017's agent-based KG construction workflow.

### Concurrent Extraction
Stage 2 (entity extraction) runs with 10-way concurrency on leaf nodes. Translation uses ThreadPoolExecutor for concurrent chunk translation.

### Data Quality
`drbrain audit` applies 15 severity-graded rules covering paper metadata, concept integrity, edge consistency, and graph structure. PDF pre-validation detects encryption and corruption before ingest. Three non-blocking quality gates run during ingest.
