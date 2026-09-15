---
name: index
description: >
  Prepare, inspect and verify the searchable index. Use this skill whenever the user
  says "build the index", "rebuild the index", "update the search index", "search isn't finding
  papers", "reindex my library", "fix search", "is the index ready", or when papers have been
  recently ingested and search results seem stale or incomplete. Also use when the user notices
  that `drbrain search` returns empty or unexpected results for terms they know should match
  (after running `drbrain ingest`, `drbrain graph build` or `drbrain repair`), or asks why
  `ask`/`search` cannot read the index. Trigger proactively whenever the user reports search
  problems or has just completed operations that change the corpus.
---

# Build, inspect and verify the search index

`drbrain index build` prepares every enabled retrieval leg from the canonical store and
publishes one queryable generation; `drbrain index status` reports what is ready and what is
still pending; `drbrain index verify` re-checks exactly what `search`/`ask` read.

## Quick start

```bash
drbrain index build                 # lexical BM25 + FTS + vectors + tree, publish a generation
drbrain index status                # ingested / indexed / retrievable, per leg
drbrain index verify                # what search/ask read: FTS, vectors, generation, profile
drbrain search "attention mechanism"   # verify retrieval end to end
```

## What it does

- Lexical stage: rebuilds the BM25 index over concepts and arguments (incremental by default).
- Unified stage: fills canonical FTS, the shared leaf/region vectors and the tree hierarchy
  incrementally, then publishes a tree generation only when something changed.
- `--force`/`-f` forces a full rebuild of every stage; `--tree-storage PATH` overrides the
  generation root; `--db PATH` targets a shard database.
- Exit code 1 when any stage failed (`failed_stages` in `--json`) — a failed stage is never
  reported as ready.

## When to build

- After `drbrain ingest` — new canonical content needs to be indexed
- After `drbrain graph build` / `drbrain repair` — new concepts or metadata change the index
- When `drbrain search`/`ask` reports that no index is prepared
- When `drbrain index status` shows pending vectors, missing parents or a stale generation

## Examples

**Standard post-ingest workflow:**
```bash
drbrain ingest ~/Downloads/new-papers/
drbrain index build                   # prepare + publish in one incremental run
drbrain search "attention mechanism"  # evidence rows with locators
drbrain index verify                  # confirm what search will read
```

**Diagnose search failures:**
```bash
drbrain list                          # confirm papers exist
drbrain index status --json           # per-leg ready + reasons + pending backlog
drbrain index build                   # fix whatever is not ready
drbrain search "known term"           # re-test
```

**Inspect a published generation:**
```bash
drbrain index status --json | jq '.generation, .legs.tree'
drbrain index verify --json | jq '.checks[] | select(.ok == false)'
```

## CLI reference

| Command | What it does |
|---------|--------------|
| `drbrain index build` | Incremental prepare (lexical + FTS + vectors + hierarchy) and publish |
| `drbrain index build --force` | Full rebuild of every stage |
| `drbrain index build --json` | Stage payload: `{"ok","changed","published","failed_stages","fts","vectors","hierarchy","publication","duration_ms","lexical"}` |
| `drbrain index status` | Ready/pending/reasons per leg; exit 0 |
| `drbrain index verify` | Retrieval-readiness checks; exit 1 on any error |
| `drbrain index --rebuild --json` | Legacy lexical-only rebuild (hidden compatibility alias; still supported) |
