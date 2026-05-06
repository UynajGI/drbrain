---
name: index
description: >
  Rebuild the BM25 search index over all concepts and arguments. Use this skill whenever the user
  says "rebuild the index", "update the search index", "search isn't finding papers", "reindex my
  library", "fix search", or when papers have been recently ingested or concepts modified and search
  results seem stale or incomplete. Also use when the user notices that `drbrain query` returns empty
  or unexpected results for terms they know should match, after running `drbrain build`, after
  running `drbrain repair`, or when search suddenly stops working. Trigger proactively whenever the
  user reports search problems or has just completed operations that change the concept database.
---

# Rebuild Search Index

Rebuild the BM25 full-text search index over all concepts and arguments in the library. The index
powers `drbrain query` — if papers are not appearing in search results after ingest/build, the
index likely needs rebuilding.

## Quick Start

```bash
drbrain index --rebuild
```

## What It Does

- Reads all concept labels, types, sections, arguments, and paper metadata from the database
- Tokenizes text and builds a BM25 inverted index with TF-IDF-like weighting
- Stores the index for fast retrieval by `drbrain query`
- Without `--rebuild`, loads the existing index (if available)
- `--json` outputs document count for verification

## When to rebuild

- After `drbrain ingest` or `drbrain build` — new concepts need indexing
- After `drbrain repair` — updated metadata changes searchable fields
- When `drbrain query` returns empty or irrelevant results for known terms
- When `drbrain query` doesn't surface recently added papers

## Examples

**Standard post-ingest workflow:**
```bash
drbrain ingest ~/Downloads/new-papers/
drbrain build
drbrain index --rebuild
drbrain query "attention mechanism"   # verify new papers appear
```

**Diagnose search failures:**
```bash
drbrain list                          # confirm papers exist
drbrain index --rebuild --json        # {"documents": 1234, "indexed": true}
drbrain query "known term"            # re-test
```

**Verify rebuild succeeded:**
```bash
drbrain index --rebuild --json | jq '.documents'
```

## CLI Reference

| Command | What it does |
|---------|--------------|
| `drbrain index` | Load existing index (no rebuild) |
| `drbrain index --rebuild` | Force full rebuild from database |
| `drbrain index --rebuild --json` | JSON output with document count |
