---
name: fsearch
description: >
  Federated search across the local DrBrain library and arXiv with ingested annotation.
  Use when the user wants to search their library plus external sources in a single command,
  check if arXiv papers are already ingested, or discover new papers. Trigger on
  "federated search", "search arXiv and my library", "find papers everywhere",
  "cross-search", "check if paper is in my library".
---

# Federated Search

Search local library + arXiv in one command, with automatic "already ingested" annotation.

## Quick Start

```bash
drbrain fsearch "attention mechanism"           # local library only
drbrain fsearch "graph neural network" --arxiv  # local + arXiv
drbrain fsearch "transformer" --arxiv-only      # arXiv only
```

## How it works

- **Local**: full-text search over papers, concepts, and arguments
- **arXiv**: Atom API search with automatic dedup — results already in your library
  are annotated with `[ingested]`
- **Cross-reference**: matches by DOI and normalized arXiv ID

## CLI Reference

| Command | What it does |
|---------|--------------|
| `drbrain fsearch <query>` | Search local library |
| `drbrain fsearch <query> --arxiv` | Local + arXiv with ingested tags |
| `drbrain fsearch <query> --arxiv-only` | arXiv only |
| `drbrain fsearch <query> --limit <n>` | Limit results per source |
| `drbrain fsearch <query> --json` | JSON output |
