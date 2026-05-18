---
name: metrics
description: >
  View user behavior analytics — top search keywords, most-read papers, weekly trends.
  Use when the user wants to see their research activity, check what they've been
  reading, analyze search patterns, or get an overview of their library usage.
  Trigger on "metrics", "analytics", "reading stats", "search history",
  "what have I been reading", "usage statistics".
---

# Metrics & Analytics

User behavior metrics tracking search keywords, paper reads, and weekly trends.

Metrics are stored in `data/metrics.db` (separate from the main database).

## Quick Start

```bash
drbrain metrics                                 # show dashboard
drbrain metrics --json                          # JSON output
```

## What's tracked

- **Search events**: keywords searched via `drbrain query` and `drbrain fsearch`
- **Read events**: papers viewed via `drbrain show`
- **Weekly trends**: 7-day rolling counts of searches and reads
- **Top keywords**: most frequently searched terms
- **Most-read papers**: most frequently viewed papers

## CLI Reference

| Command | What it does |
|---------|--------------|
| `drbrain metrics` | Show analytics dashboard |
| `drbrain metrics --json` | JSON output |

## See also

- `drbrain insights` — OpenAlex-based research analytics (coming soon)
