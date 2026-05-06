---
name: graph
description: >
  Direct knowledge graph operations — traverse neighbors, find shortest paths, and analyze shared
  concepts across papers. Use this skill whenever the user asks about "graph neighbors", "connections
  between concepts", "find a path from X to Y", "how are these papers connected?", "are these
  concepts related?", "traverse the knowledge graph", "show me the subgraph around...", "related
  papers by concept", or wants to explore concept-to-concept relationships directly without BM25 text
  search. Also use when the user asks "what does this concept connect to?", "is there a path between
  A and B?", "what do these papers have in common?", or wants an LLM-generated natural language
  description of a subgraph. Trigger proactively when the user discusses concept relationships or
  paper-to-paper concept overlap.
---

# Knowledge Graph

Query the knowledge graph directly — traverse from a node, find shortest paths between concepts,
analyze shared concepts across papers, and generate LLM-powered subgraph descriptions. All operations
work on the directed graph of concepts and relations extracted from the library.

## Operations

### neighbors — Graph traversal

Traverse from a starting node (concept label or paper ID):

```bash
drbrain graph neighbors "Attention Mechanism" --hops 2
drbrain graph neighbors "Backpropagation" --direction both --json
drbrain graph neighbors "Transformer" --relation solves,extends
```

Options: `--hops` / `-n` (depth), `--relation` / `-R` (filter by type), `--direction` / `-D`
(forward/backward/both), `--json`, `--workspace` / `-w`.

Valid relation types: `addresses`, `leaves_open`, `points_to`, `proposes`, `extends`, `replaces`,
`solves`, `supports`, `challenges`, `limits`, `constrains`, `affiliated_with`.

### path — Shortest path

Find the shortest path between two nodes (BFS on undirected copy, recovers edge direction):

```bash
drbrain graph path "Transformer" "BERT"
drbrain graph path "Attention" "Gradient Descent" --max-length 4
```

### related — Cross-paper concept analysis

Analyze shared concepts across two or more papers:

```bash
drbrain graph related p3f8a2 p7b1c4                       # label intersection
drbrain graph related p3f8a2 p7b1c4 --mode graph           # 1-hop traversal intersection
drbrain graph related p3f8a2 p7b1c4 p9d2e5 --mode edges    # shared edge patterns
```

### describe — LLM subgraph summary

Generate a natural language description of a subgraph centered on a node:

```bash
drbrain graph describe "Attention Mechanism" --depth 2
```

## Examples

**Explore around a concept:**
```bash
drbrain graph neighbors "Backpropagation" --hops 2 --direction both --json | jq '.'
```

**Check if two papers share concepts:**
```bash
drbrain graph related p3f8a2 p7b1c4
# Zero shared concepts suggests disjoint research domains
```

**Find the connection path between two concepts:**
```bash
drbrain graph path "Self-Attention" "Cross-Entropy Loss"
# No path means these concepts live in separate subgraphs
```

## CLI Reference

| Command | What it does |
|---------|--------------|
| `drbrain graph neighbors <node>` | 1-hop traversal from a node |
| `drbrain graph neighbors <node> -n 3` | Multi-hop traversal |
| `drbrain graph neighbors <node> -R solves,extends` | Filtered by relation type |
| `drbrain graph path <src> <dst>` | Shortest path between two nodes |
| `drbrain graph related <id...>` | Shared concept analysis across papers |
| `drbrain graph related <id...> -m graph` | Graph-traversal shared concepts |
| `drbrain graph related <id...> -m edges` | Shared edge patterns |
| `drbrain graph describe <node>` | LLM subgraph description |
