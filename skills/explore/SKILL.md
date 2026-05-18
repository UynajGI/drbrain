---
name: explore
description: >
  Manage explore silos — lightweight literature discovery collections separate from
  the main library. Use when the user wants to collect papers for literature review,
  build a topic-specific reading list, or explore a new research area without polluting
  the main library. Trigger on "explore a topic", "create a reading list", "discovery
  collection", "literature survey", "topic exploration".
---

# Explore Silos

Lightweight collections for literature discovery — separate from the main library and
workspaces. Each silo stores papers in JSONL format with full-text keyword search.

## Quick Start

```bash
drbrain explore --create transformers    # create a silo
drbrain explore --list                   # list silos
drbrain explore --name transformers --show  # view papers
drbrain explore --name transformers --search "attention"  # search within silo
drbrain explore --delete transformers    # delete a silo
```

## Programmatic use

```python
from drbrain.storage.explore import create_explore_silo, add_paper_to_silo

create_explore_silo(Path("data/explore"), "nlp")
add_paper_to_silo(Path("data/explore"), "nlp", {
    "title": "Attention Is All You Need",
    "authors": ["Vaswani, Ashish"],
    "year": 2017,
})
```

## CLI Reference

| Command | What it does |
|---------|--------------|
| `drbrain explore --list` | List all silos |
| `drbrain explore --create <name>` | Create a silo |
| `drbrain explore --name <n> --show` | Show silo papers |
| `drbrain explore --name <n> --search <q>` | Search within silo |
| `drbrain explore --delete <name>` | Delete a silo |
| `drbrain explore --json` | JSON output |
