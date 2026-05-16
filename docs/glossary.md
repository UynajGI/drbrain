# Glossary

Domain terms used throughout DrBrain's codebase and documentation.

## Knowledge Graph

**TBox (Type Box)**
The type-level ontology. Defines what kinds of things exist. DrBrain's TBox has 6 types:
Problem, Method, Conclusion, Gap, Debate, Actor.

**RBox (Relation Box)**
The relation-level rules. Defines how things connect. 11 inference rules (8 in `closure()`,
4 in `apply_path_rules()`).

**Closure**
The process of applying inference rules to derive new edges from existing ones.
Modes: `symbolic` (rule-only), `hybrid` (TransE-weighted).

**Hybrid Scoring**
Combines rule-based closure confidence with TransE embedding scores. Multiplicative
weighting of symbolic and embedding signals.

**Confidence**
A 0.0–1.0 score on every concept, argument, and edge. Initially from LLM extraction,
modified by confidence propagation and closure rules.

**Confidence Propagation**
Multi-hop decay: as you traverse further from an extracted concept, confidence drops
(decay factor 0.85 per hop). Section-aware variant weights confidence by section type.

**Provenance**
Tracing a concept or edge back to its origin: which paper, which section (`section` field),
which tree node (`node_id` field). Flows from extraction through the DB to all reasoning
layers.

**Dedup**
Cross-paper concept deduplication. Exact label match or word-overlap similarity merges
duplicate concepts into canonical identities.

---

## Graph Embeddings

**TransE**
Translating Embeddings model (Bordes et al., 2013). Represents entities and relations as
vectors such that `head + relation ≈ tail`. Used for link prediction, entity similarity,
and path rule mining.

**Link Prediction**
Given a head entity and relation, predict the most likely tail entity using TransE vector
arithmetic. `predict_link(head, relation, top_k)`.

**Entity Similarity**
Cosine similarity between TransE entity vectors. `similar_entities(label, top_k)`.

---

## Tree & Retrieval

**PageIndex**
A tree-structured document representation. Builds a hierarchical section tree from a
document's table of contents. Each node has a title, summary, and content range.
Reference: answerdotai/pageindex.

**RAPTOR**
Recursive Abstractive Processing for Tree-Organized Retrieval (Sarthi et al., 2401.18059).
Recursive embedding → UMAP → GMM+BIC clustering → LLM summarization on leaf nodes.
Builds a multi-layer summary tree. DrBrian stores summaries in `tree_summaries` table.

**Tree Node**
A single section or summary in the PageIndex/RAPTOR tree. Identified by `node_id`.
Leaf nodes = original sections. Internal nodes = RAPTOR summaries.

**Tree Layer**
The depth level in the tree. `pageindex` = original leaves (layer 0). `raptor_L1`,
`raptor_L2`, etc. = recursive summary layers.

**Tree Traversal**
Two-stage retrieval per RAPTOR Figure 2: layer-by-layer top-k descent (select best
nodes at each layer) → collapsed tree fallback (brute-force cosine over all nodes
if traversal misses).

**Collapsed Tree**
All tree nodes (PageIndex + all RAPTOR layers) flattened into a single pool for
brute-force cosine similarity search. Used as fallback and for cross-paper queries.

---

## Reasoning

**Causal Chain**
A multi-hop path through the graph tracing cause-effect relationships. `build_causal_chains()`
finds chains of arbitrary length.

**Counterfactual**
"What if" analysis: remove a node from the graph and measure impact on connectivity, path
lengths, and closure output.

**Isomorphism**
Cross-domain subgraph similarity. Two subgraphs are isomorphic if they share the same
relation signature. Enables knowledge transfer between domains.

**Knowledge Frontier**
The boundary between known and unknown in a domain. Composite report combining: research
seeds, debate zones, technology cliffs, difficulty scores, and confidence collapse patterns.

**Paradigm Shift**
Detection via paper-age analysis: clusters of new concepts that rapidly displace older
ones indicate paradigm shifts.

**Transfers**
Cross-domain method migration: finding methods from one domain applied in another,
detected through workspace clustering.

**Difficulty Map**
Gap classification by originating section type (introduction, methods, results, etc.)
with composite difficulty scoring. `drbrain difficulty`.

---

## Pipeline

**Ingest**
Phase 1: PDF → Markdown + Metadata + tree.json. Lightweight, no concept extraction.
Paper status: `uploaded`.

**Build**
Phase 2: tree.json → 5-stage LLM extraction → concepts + relations + edges.
Paper status: `extracted`.

**Ontology Extension**
Build Stage 1: LLM suggests domain-specific subtypes under the 6 TBox categories
(e.g. `OptimizationAlgorithm` under `Method`).

**Concurrent Extraction**
Build Stage 2: per-leaf-node concept extraction, 10-way parallel by default.

**Iterative Refinement**
Build Stage 5: LLM self-reviews extraction output for contradictions. Skippable via
`--skip-refine`.

**Placeholder Paper**
A citation-only record (status: `placeholder`). Created when expanding citations.
Upgraded to `uploaded` if the paper is later ingested.

**Quality Gate**
Non-blocking checks during ingest: markdown size, metadata completeness, extraction
quality. Log warnings but never block.

---

## Embedding (Text)

**Tree Node Embedding**
SBERT vector for a PageIndex/RAPTOR tree node. Stored in `tree_vectors` table as BLOB.
Used for cosine similarity search.

**Content Hash**
SHA256 of node text, truncated to 16 chars. Used for incremental update: if hash
matches existing, skip re-embedding.

**Provider**
The embedding backend: `local` (sentence-transformers, default), `openai-compat`
(any `/v1/embeddings` API), `none` (disable vectors).

**GPU Profile**
One-time memory benchmark per model+GPU combination. Cached to disk. Feeds adaptive
batch sizing to avoid CUDA OOM.

---

## Storage

**WAL Mode**
SQLite Write-Ahead Logging. Enables concurrent reads during writes. `drbrain.db-wal`
and `drbrain.db-shm` files are normal.

**Atomic Write**
tmp→rename pattern for all filesystem writes. Prevents partial writes from corrupting
state.

**Schema Version**
Tracked in `schema_versions` table. Migrations auto-apply on `Database()` init.

**Workspace**
A named subset of papers. Stored under `workspace/<name>/` with `workspace.yaml` and
`refs/papers.json`.

---

## CLIs & Tools

**typer**
Python CLI framework used by DrBrain. Commands are typer functions registered in
`cli/main.py`.

**litellm**
LLM abstraction library. DrBrain uses it for all LLM calls with provider-agnostic
fallback chains.

**MinerU**
PDF parsing service. Primary parser; PyMuPDF is the fallback.

**loguru**
Structured logging library. Rotating file logs at `data/logs/drbrain.log`.
