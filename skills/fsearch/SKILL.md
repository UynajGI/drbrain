---
name: fsearch
description: >
  Search local evidence and external sources (arXiv) in one command, with ingested annotation.
  Use when the user wants to search their library plus external sources, check if arXiv papers
  are already ingested, or discover new papers. Trigger on "federated search", "search arXiv
  and my library", "find papers everywhere", "cross-search", "check if paper is in my library".
---

# Federated search

`drbrain search --source` retrieves local evidence rows and/or external rows in one result set,
with an automatic "already ingested" annotation for external hits.

## Quick start

```bash
drbrain search "attention mechanism"                 # local evidence only (default)
drbrain search "graph neural network" --source all   # local evidence + arXiv
drbrain search "transformer" --source arxiv          # arXiv only
```

## How it works

- **Local** (`--source local`, default): the same retrieval chain as `ask` (bm25/vector/tree),
  returning evidence rows with sources, text locators, route and index generation.
- **arXiv** (`--source arxiv|all`): Atom API search with automatic dedup — results already in
  your library are annotated with `ingested: true`. External rows are marked `source="arxiv"`
  and carry `url` / `doi` / `arxiv_id`, but deliberately no local locator.
- **Cross-reference**: matches by DOI and normalized arXiv ID.

## CLI reference

| Command | What it does |
|---------|--------------|
| `drbrain search "<query>"` | Local evidence retrieval |
| `drbrain search "<query>" --source all` | Local evidence + arXiv rows |
| `drbrain search "<query>" --source arxiv` | arXiv rows only |
| `drbrain search "<query>" --limit <n> --json` | JSON payload (`route`, `generations`, `legs`, `evidence`) |
| `drbrain fsearch "<query>" --arxiv` | Historical federated command (hidden compatibility alias) |
