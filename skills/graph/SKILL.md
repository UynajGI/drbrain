---
name: graph
description: >
  Direct knowledge graph operations — traverse, find paths, cross-paper concept analysis. Use this
  skill whenever the user asks about "graph neighbors", "connections between concepts", "find path
  from X to Y", "related papers", "how are these papers connected", "traverse the graph", or wants
  to explore the knowledge graph without BM25 text search.
---

# Knowledge Graph

Query the knowledge graph directly — traverse from a node, find shortest paths between concepts,
analyze shared concepts across papers, and generate LLM-powered subgraph descriptions. All operations
work on the directed graph of concepts and relations extracted from the library.

## Quick Start

```bash
# See what a concept connects to
drbrain graph neighbors "Attention Mechanism" --hops 2

# Find how two concepts are connected
drbrain graph path "Transformer" "BERT"

# See shared concepts across papers
drbrain graph related p3f8a2 p7b1c4
```

## What It Does

### `neighbors` — Graph traversal
Traverse the graph from a starting node (concept label or paper ID), showing all reachable nodes
with their paths, types, and distances.

- `--hops` / `-n`: traversal depth (default 1)
- `--relation` / `-R`: filter by relation type (comma-separated: `addresses,extends,solves`)
- `--direction` / `-D`: forward, backward, or both
- `--json`: machine-readable output with full path data
- `--workspace` / `-w`: limit to workspace papers

Valid relation types: `addresses`, `leaves_open`, `points_to`, `proposes`, `extends`, `replaces`,
`solves`, `supports`, `challenges`, `limits`, `constrains`, `affiliated_with`.

### `path` — Shortest path
Find the shortest path between two nodes using BFS on an undirected copy of the graph, then
recover edge direction and relation type from the original directed graph.

- `--max-length`: maximum path length (default 6)
- `--json`: machine-readable output with path steps

### `related` — Cross-paper analysis
Analyze shared concepts and connections across two or more papers.

- `--mode concepts`: SQL intersection of concept labels across papers
- `--mode graph`: 1-hop graph traversal from each paper's concepts, then intersect
- `--mode edges`: shared (relation, target) edge patterns
- `--min-shared`: minimum papers a concept/edge must appear in (default 2)

### `describe` — LLM subgraph summary
Generate a natural language description of a subgraph centered on a node.

- `--depth` / `-n`: traversal depth (default 1)
- Requires LLM models configured

## CLI Reference

| Command | What it does |
|---------|--------------|
| `drbrain graph neighbors <node>` | 1-hop traversal from a node |
| `drbrain graph neighbors <node> -n 3` | 3-hop traversal |
| `drbrain graph neighbors <node> -R solves,extends` | Filtered by relation type |
| `drbrain graph neighbors <node> -D backward` | Only incoming edges |
| `drbrain graph path <src> <dst>` | Shortest path between two nodes |
| `drbrain graph path <src> <dst> --max-length 4` | BFS cutoff at 4 hops |
| `drbrain graph related p1 p2 p3` | Shared concepts across 3 papers |
| `drbrain graph related p1 p2 -m graph` | Graph-traversal mode |
| `drbrain graph related p1 p2 -m edges` | Shared edge patterns |
| `drbrain graph describe <node>` | LLM description of subgraph |
| `drbrain graph traverse-from <section>` | Tree+graph hybrid traversal |

## Common Patterns

**Explore around a concept:**
```bash
drbrain graph neighbors "Backpropagation" --hops 2 --direction both --json | jq '.'
```

**Check if two concepts are connected:**
```bash
drbrain graph path "Attention" "Gradient Descent"
# If no path: these concepts live in separate subgraphs (different research areas)
```

**Find what two papers have in common:**
```bash
drbrain graph related p3f8a2 p7b1c4
# Shared concepts suggest overlapping research domains
```

**Debug extraction quality:**
```bash
drbrain graph related p3f8a2 p7b1c4 -m edges
# Zero shared edges may indicate shallow extraction or truly disjoint papers
```
