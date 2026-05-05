---
name: index
description: >
  Rebuild the BM25 search index. Use this skill whenever the user says "rebuild index", "update search
  index", "search not finding papers", "reindex", or when papers have been ingested or concepts modified
  and search results seem stale.
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
- Without `--rebuild`, loads existing index (if available)
- `--json` for machine-readable output with document count

## CLI Reference

| Command | What it does |
|---------|--------------|
| `drbrain index` | Load existing index (no rebuild) |
| `drbrain index --rebuild` | Force full rebuild from database |
| `drbrain index --rebuild --json` | JSON output with document count |

## Common Patterns

**After ingesting new papers:**
```bash
drbrain ingest ~/Downloads/new-paper.pdf
drbrain build
drbrain index --rebuild
drbrain query "my search terms"
```

**When search returns unexpected results:**
```bash
# Rebuild to pick up new/updated concepts
drbrain index --rebuild

# Verify
drbrain index --rebuild --json
# {"documents": 1234, "indexed": true}
```

**When search returns nothing but papers exist:**
```bash
drbrain list          # confirm papers are in the library
drbrain index --rebuild --json   # rebuild and check doc count
drbrain query "known term"       # re-test search
```
