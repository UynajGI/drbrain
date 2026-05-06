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

### Timeline view

Track concept evolution across papers:

```bash
drbrain timeline "Self-Attention"
```

## After finding papers

- `drbrain show p3f8a2` — inspect a paper's full contents
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
| `drbrain timeline <concept>` | Year-by-year concept evolution |
