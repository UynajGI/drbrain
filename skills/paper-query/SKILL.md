---
name: paper-query
description: >
  Search and explore papers in the DrBrain library using BM25 keyword search, graph-enhanced hybrid
  ranking, and tree-based retrieval. Use this skill whenever the user asks "what papers do I have
  about X?", "find papers on...", "search for...", "show me papers about...", "look up concept X",
  "browse my library", wants to explore their collection by topic, or needs to find specific concept
  types or arguments. Also use when the user wants to deep-read a paper's sections via tree-based
  retrieval, track concept evolution over time, or expand search results with graph neighborhood
  traversal. Trigger proactively for any kind of library search or exploration.
---

# Paper Query

Search and explore the DrBrain library. Two search modes: BM25 keyword search over concepts and
arguments (topic-based), and PageIndex tree-based retrieval (deep-reading specific paper sections).

## Prerequisites

Knowledge graph must be built (`kg-build` skill). BM25 index must be current (run `drbrain index` if search returns stale results). Tree-based retrieval requires `drbrain embed --tree` (PageIndex+RAPTOR embeddings).

## Search modes

### BM25 keyword search

Find papers and concepts across the entire library:

```bash
drbrain query "graph neural networks"
drbrain query "transformer attention" --type-filter Method
drbrain query "over-smoothing" --min-confidence 0.8
drbrain query "knowledge distillation" --year-start 2020 --limit 10
```

Combine filters: `--type-filter` (Problem/Method/Conclusion/Debate/Gap/Actor), `--arg-type`
(supports/challenges/extends), `--year-start/--year-end`, `--min-confidence`, `--limit`.

Expand results with graph traversal:

```bash
drbrain query "graph attention" --neighbors 2
```

### Tree-based retrieval

Deep-read a specific paper's sections using the PageIndex tree structure:

```bash
drbrain query "how does the proposed method handle overfitting" --paper p3f8a2
```

### Hybrid ranking

Boost BM25 results with graph centrality (PageRank):

```bash
drbrain query "graph attention" --hybrid
```

## After finding papers

- `drbrain show p3f8a2` — inspect a paper's full contents
- `drbrain ask "what is the main contribution of this paper?"` — natural language Q&A over the KG
- `drbrain reason "compare approach A and B"` — deep LLM agent reasoning (see kg-reason skill)
- `drbrain analyze p3f8a2` — run knowledge frontier analysis
- `drbrain citations p3f8a2 --type shared-refs` — find related work
- `drbrain ws add attention-methods p3f8a2` — save to a workspace

## Examples

**Topic search with type and confidence filters:**
```bash
drbrain query "contrastive learning" --type-filter Method --min-confidence 0.7 --limit 20
```

**Graph-expanded search:**
```bash
drbrain query "attention mechanism" --neighbors 2 --json | jq '.[]._distance'
```

**Section-level deep reading:**
```bash
drbrain query "regularization strategy" --paper p3f8a2
```

## CLI Reference

| Command | What it does |
|---------|--------------|
| `drbrain query <terms>` | BM25 keyword search |
| `drbrain query <terms> --type-filter Method` | Filter by concept type |
| `drbrain query <terms> --neighbors N` | Graph-expanded results |
| `drbrain query <terms> --paper <id>` | Tree-based section retrieval |
| `drbrain query <terms> --hybrid` | PageRank-boosted ranking |
| `drbrain ask "<question>"` | Natural language KGQA |
