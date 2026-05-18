---
name: enrich
description: >
  Enrich paper metadata by backfilling missing fields from CrossRef and detecting
  records that need cleanup. Use when the user wants to fix incomplete metadata,
  check paper record quality, find scrub-worthy records, or fill missing DOIs,
  years, authors, or journal names. Trigger on "enrich metadata", "fix metadata",
  "complete paper info", "find dirty records", "clean up metadata".
---

# Metadata Enrichment

Backfill missing metadata from CrossRef and detect scrub-worthy records.

## Quick Start

```bash
drbrain enrich <paper_id>            # check and backfill one paper
drbrain enrich <paper_id> --dry-run  # check only, no backfill
drbrain enrich --all                 # check all papers
drbrain enrich --all --dry-run       # audit all papers
```

## What it does

- **Completeness check**: title, year, authors, journal
- **Scrub detection**: empty titles, suspicious filenames, far-future years, missing authors
- **CrossRef backfill**: if DOI is present and fields are missing, fetches from CrossRef API

## CLI Reference

| Command | What it does |
|---------|--------------|
| `drbrain enrich <id>` | Check and backfill one paper |
| `drbrain enrich <id> --dry-run` | Check without backfilling |
| `drbrain enrich --all` | Check all papers |
| `drbrain enrich --all --dry-run` | Audit all papers |
| `drbrain enrich <id> --json` | JSON output |

## See also

- `drbrain repair` — comprehensive metadata repair via OpenAlex
- `drbrain audit` — 15 quality rules for the knowledge graph
