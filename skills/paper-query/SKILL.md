---
name: paper-query
description: >
  Search and explore papers in the DrBrain library. Use this skill whenever the user wants to find
  papers, search by keyword or concept, browse their collection, explore a specific paper's content
  via tree-based retrieval, or look up what the library contains. Also use when the user asks "what
  papers do I have about X?", "find me papers on...", "show me what's in my library", or wants to
  explore the knowledge graph neighborhood of a concept.
---

# Paper Query

Search and explore the DrBrain library. Two search modes are available: BM25 keyword search over
concepts and arguments (best for finding papers by topic), and PageIndex tree-based retrieval (best
for deep-reading a specific paper's sections).

## Search modes

### BM25 keyword search

Best for finding papers and concepts across the library:

```bash
drbrain query "graph neural networks"
drbrain query "transformer attention" --type-filter Method
drbrain query "over-smoothing" --min-confidence 0.8
drbrain query "knowledge distillation" --year-start 2020 --limit 10
```

Filters can be combined: `--type-filter` (Problem/Method/Conclusion/Debate/Gap/Actor),
`--arg-type` (supports/challenges/extends), `--year-start/--year-end`, `--min-confidence`,
`--limit`.

Expand results by graph traversal to see related papers:

```bash
drbrain query "graph attention" --neighbors 2
```

### Tree-based retrieval

For deep-reading a specific paper. Uses the PageIndex tree structure to find relevant sections:

```bash
drbrain query "how does the proposed method handle overfitting" --paper <local_id>
```

This sends the paper's section tree to an LLM, which selects relevant sections, then loads their
content on-demand. Much more targeted than reading the full paper.

### Timeline view

Track how a concept evolved across papers:

```bash
drbrain timeline "Self-Attention"
```

Shows year-by-year paper count and confidence trends, plus a signal classification (emerging,
established, declining, contested, resurging).

## After finding papers

Once you've found interesting papers:
- Use `drbrain report <local_id>` for a single-paper summary
- Use `drbrain analyze <local_id>` for knowledge frontier analysis
- Use `drbrain citations <local_id> --type shared-refs` to find related work
- Use `drbrain ws add <workspace> <local_id>` to save papers for later analysis
