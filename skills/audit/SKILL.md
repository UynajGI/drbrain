---
name: audit
description: >
  Scan the entire library for data quality issues across 15 severity-graded rules. Use this skill
  whenever the user asks to "check my library quality", "audit my papers", "find data issues",
  "verify data integrity", "are there problems with my papers?", "diagnose missing metadata",
  "why are some papers incomplete?", or wants a health report on their knowledge graph. Also use
  when the user notices search results are incomplete, analysis output seems noisy, or before
  running large-scale analysis (seed, closure) to ensure the underlying data is clean. Trigger
  proactively when the user expresses concern about paper quality, asks why certain papers lack
  concepts or edges, or wants to clean up the library.
---

# Data Quality Audit

Run a 15-rule scan across all papers, organized into three severity levels: error (must-fix),
warning (should-fix), and info (nice-to-know). Produces a structured report showing which papers
have which issues, with fix guidance for each rule.

## Prerequisites

Papers must be in the library (`paper-ingest` skill). For rules checking concept/edge counts, the KG should be built (`kg-build` skill).

## Quick Start

```bash
drbrain audit
```

## What It Checks

**Error rules (2):** `missing_title`, `missing_md`
**Warning rules (8):** `missing_doi`, `missing_abstract`, `missing_year`, `missing_journal`,
  `missing_authors`, `short_md`, `empty_tree`, `low_concept_count`, `unresolved_env`
**Info rules (4):** `no_edges`, `placeholder_status`, `old_placeholder`, `duplicate_title`

## Common Patterns

**Quick health check before analysis:**
```bash
drbrain audit --severity error
```
If this returns issues, something is fundamentally broken — fix before running analysis.

**Full audit before `drbrain analyze` or `drbrain seed`:**
```bash
drbrain audit
```
Fix warnings first. Papers with missing abstracts, years, or journals produce noisy results.

**Find stale placeholders:**
```bash
drbrain audit --severity info --json | jq '.[] | select(.rule == "old_placeholder")'
```

**Workspace-specific audit:**
```bash
drbrain audit --workspace attention-methods
```

**Fix common issues:**
- `missing_md` / `empty_tree`: re-run `drbrain ingest`
- `missing_doi` / `missing_authors` / `missing_abstract`: run `drbrain repair --all`
- `low_concept_count` / `no_edges`: re-run `drbrain build`

## Examples

**Full audit with JSON for scripting:**
```bash
drbrain audit --severity info --json | jq 'group_by(.severity) | map({severity: .[0].severity, count: length})'
```

**Check a workspace for issues before submission:**
```bash
drbrain audit --workspace paper-draft --severity warning
```

## CLI Reference

| Command | What it does |
|---------|--------------|
| `drbrain audit` | Full audit at warning level (warnings + errors) |
| `drbrain audit --severity error` | Only errors |
| `drbrain audit --severity info` | All issues including info-level |
| `drbrain audit --json` | Machine-readable JSON output |
| `drbrain audit --workspace <name>` | Audit only papers in a workspace |
| `drbrain repair --all` | Auto-fix missing metadata from CrossRef/arXiv |
