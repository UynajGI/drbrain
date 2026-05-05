---
name: audit
description: >
  Scan the entire library for data quality issues. Use this skill whenever the user asks to "check my
  library quality", "audit papers", "find data issues", "verify data integrity", "are there problems
  with my papers", or wants to diagnose why certain papers are missing metadata, concepts, or edges.
---

# Data Quality Audit

Run a 15-rule scan across all papers in the library, organized into three severity levels: error
(must-fix), warning (should-fix), and info (nice-to-know). Produces a structured report showing
which papers have which issues.

## Quick Start

```bash
drbrain audit
```

## What It Does

Runs 15 rules against every paper:

**Error rules (2):**
- `missing_title` — paper has no title or an empty title
- `missing_md` — no raw.md file exists (ingest likely failed)

**Warning rules (8):**
- `missing_doi` — no DOI, arXiv ID, or Semantic Scholar ID
- `missing_abstract` — abstract field is empty
- `missing_year` — year is NULL
- `missing_journal` — journal field is empty
- `missing_authors` — no Actor-type concepts (no author extracted)
- `short_md` — raw.md exists but is under 200 bytes (likely failed parse)
- `empty_tree` — tree.json missing or empty
- `low_concept_count` — fewer than 3 concepts (shallow extraction)
- `unresolved_env` — title contains `${}` (environment variable not resolved)

**Info rules (3):**
- `no_edges` — has concepts but zero knowledge graph edges
- `placeholder_status` — paper has status "placeholder" (imported but not ingested)
- `old_placeholder` — placeholder older than 30 days
- `duplicate_title` — normalized title matches another paper

## CLI Reference

| Command | What it does |
|---------|--------------|
| `drbrain audit` | Full audit at warning level (warnings + errors) |
| `drbrain audit --severity error` | Only errors |
| `drbrain audit --severity info` | All issues including info-level |
| `drbrain audit --json` | Machine-readable JSON output |
| `drbrain audit --workspace <name>` | Audit only papers in a workspace |

## Common Patterns

**Quick health check:**
```bash
drbrain audit --severity error
```
If this returns issues, something is fundamentally broken (missing titles, failed parses).

**Before running analysis:**
```bash
drbrain audit
```
Fix warnings before running `drbrain analyze` or `drbrain seed`. Papers with missing abstracts,
years, or journals produce noisy or incomplete analysis results.

**Find stale placeholders:**
```bash
drbrain audit --severity info | grep placeholder
```
Placeholders older than 30 days likely need attention — either ingest the PDF or delete the paper.

**Fix common issues:**
- `missing_md` / `empty_tree`: re-run `drbrain ingest <id>`
- `missing_doi` / `missing_authors` / `missing_abstract`: run `drbrain repair --all`
- `low_concept_count` / `no_edges`: re-run `drbrain build <id>`
