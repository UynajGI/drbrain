---
name: proceedings
description: >
  Manage conference proceedings — create, list, show, and associate papers with
  proceedings. Use when the user wants to organize papers by conference, track
  which papers belong to a specific conference, or manage proceedings collections.
  Trigger on "conference proceedings", "organize by conference", "ICML papers",
  "NeurIPS proceedings", "add to proceedings".
---

# Proceedings

Organize papers into conference proceedings collections.

## Quick Start

```bash
drbrain proceedings --create "NeurIPS 2024"
drbrain proceedings --create "ICML 2023"
drbrain proceedings --list
```

## Adding papers

```bash
drbrain proceedings --add <proceeding_id> <paper_id>
drbrain proceedings --show <proceeding_id>
```

## CLI Reference

| Command | What it does |
|---------|--------------|
| `drbrain proceedings --list` | List all proceedings |
| `drbrain proceedings --create "<Name Year>"` | Create new proceeding |
| `drbrain proceedings --show <id>` | Show proceeding details |
| `drbrain proceedings --add <proc_id> <paper_id>` | Add paper to proceeding |
| `drbrain proceedings --json` | JSON output |
