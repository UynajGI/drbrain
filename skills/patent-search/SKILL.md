---
name: patent-search
description: >
  Search USPTO patents via PPUBS (free, no key) or ODP (API key required).
  Use when the user wants to search for patents, look up a patent by application number,
  find prior art, or explore patent literature. Trigger on "search patents", "look up patent",
  "USPTO", "patent search", "find prior art".
---

# Patent Search

Search USPTO patents via two sources: PPUBS (free, web-based) and ODP (API key, richer metadata).

## Quick Start (PPUBS — no API key)

```bash
drbrain patent-search "machine learning transformer"
drbrain patent-search "graph neural networks" --limit 5
```

## ODP search (API key required)

Register at https://data.uspto.gov/apis/getting-started, then:

```bash
export USPTO_ODP_API_KEY=your-key
drbrain patent-search "quantum computing" --source odp
```

## Application number lookup (ODP only)

```bash
drbrain patent-search --application 17123456 --source odp --api-key your-key
```

## JSON output

```bash
drbrain patent-search "neural network" --json
```

## CLI Reference

| Command | What it does |
|---------|--------------|
| `drbrain patent-search <query>` | Search patents (PPUBS) |
| `drbrain patent-search <query> --source odp` | Search with ODP |
| `drbrain patent-search --application <num> --source odp` | Lookup by app number |
| `drbrain patent-search <query> --limit <n>` | Limit results |
| `drbrain patent-search <query> --json` | JSON output |
