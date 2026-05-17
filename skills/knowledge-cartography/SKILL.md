---
name: knowledge-cartography
description: >
  Map the knowledge landscape of your library — detect research seeds, trace concept evolution,
  find paradigm shifts, discover cross-domain transfers, and generate composite frontier reports.
  Use this skill whenever the user asks "what's emerging?", "how did this concept evolve?",
  "where are the gaps?", "is this a paradigm shift?", "find structurally similar concepts",
  "what's the knowledge frontier?", "how hard is this gap?", "trace academic descendants",
  or wants to understand research dynamics, trends, and opportunities across their collection.
  Trigger proactively when the user discusses research strategy, asks about the state of a field,
  or wants to identify high-impact open problems.
---

# Knowledge Cartography

Analyze research dynamics across the knowledge graph. Each command reveals a different dimension:
temporal evolution, structural patterns, difficulty assessment, and composite frontier mapping.

## Prerequisites

Knowledge graph must be built with embeddings and closure for best results (see `kg-build` skill):

```bash
drbrain build && drbrain embed --tree && drbrain closure --mode hybrid
```

Some commands require closure: `evolve --stats`, `paradigm`, `transfers`, `isomorphism`.

## Operations

### seed — Research dynamics scan

Detect research signals across all concepts:

```bash
drbrain seed
drbrain seed --json
drbrain seed -w my-workspace
```

Signal types: `contested` (active debate, low confidence), `emerging` (rapid growth), `established`
(stable), `declining` (dormant 3+ years), `resurging` (revived after dormancy).

### evolve — Concept lineage

Trace a concept's ancestors and descendants in the knowledge graph:

```bash
drbrain evolve "Self-Attention"
drbrain evolve "Backpropagation" -d ancestors -n 5
drbrain evolve "Transformer" --mermaid
drbrain evolve "Graph Neural Network" --stats --json
```

Options: `-d ancestors/descendants/both` (default both), `-n` max depth (default 3),
`--mermaid` (diagram), `--json`, `--stats` (temporal signal classification +
year-by-year counts with trend indicators: growing/declining/stable).

### descendants — Academic offspring

Trace who cited, extended, or refined a paper:

```bash
drbrain descendants p3f8a2
drbrain descendants p3f8a2 -g 5 --mermaid
drbrain descendants p3f8a2 --sections --json
```

Options: `-g N` generations (default 3), `--mermaid`, `--json`, `--sections` (provenance).

### landscape — Domain timeline

Map a domain's evolution: timeline, persistent gaps, debates, technology cliffs:

```bash
drbrain landscape
drbrain landscape my-workspace --top-n 10
drbrain landscape --json
```

### paradigm — Paradigm shift detection

Detect paradigm shifts: replacement (new method replaces old), explosion (rapid new subfield),
cross-domain invasion:

```bash
drbrain paradigm
drbrain paradigm "Transformer" --json
drbrain paradigm -w my-workspace
```

### transfers — Cross-domain method migration

Discover methods that could transfer from one domain to another:

```bash
drbrain transfers --from source-ws --to target-ws
drbrain transfers --auto --min-confidence 0.5
drbrain transfers --history --sections --json
```

Options: `--from` / `--to` (workspace names), `--auto` (auto-detect domains),
`--min-confidence` (default 0.3), `--json`, `--history` (historical timeline),
`--sections` (provenance).

### isomorphism — Structural pattern matching

Find concepts with similar relation patterns across domains (Jaccard + label similarity):

```bash
drbrain isomorphism
drbrain isomorphism "Attention Mechanism" --min-confidence 0.7 --json
```

Options: `[CONCEPT]` (anchor concept), `--min-confidence` (default 0.5), `--json`.

### difficulty — Gap difficulty assessment

Classify knowledge gaps by source section type with composite difficulty scoring:

```bash
drbrain difficulty
drbrain difficulty --json
```

Difficulty factors: section semantics (introduction/methods/results/discussion), confidence
collapse patterns, concept density in the gap region.

### frontier — Composite report

Combined knowledge frontier: active gaps + debates + paradigm shifts + difficulty scores +
confidence collapse patterns:

```bash
drbrain frontier
drbrain frontier --json
```

## Examples

**Quick library landscape:**
```bash
drbrain seed --json | jq '.[] | {type, label, signal}'
```

**Trace a concept's rise and influence:**
```bash
drbrain evolve "Self-Attention" --stats --mermaid
```

**Find method transfer opportunities:**
```bash
drbrain transfers --auto --min-confidence 0.4 --json | jq '.'
```

**Full frontier scan for research planning:**
```bash
drbrain frontier --json | jq '{gaps: .gaps[:3], debates: .debates[:3], shifts: .paradigm_shifts[:3]}'
```

## CLI Reference

| Command | What it does |
|---------|--------------|
| `drbrain seed` | Research signal detection across all concepts |
| `drbrain seed -w <ws>` | Workspace-scoped signals |
| `drbrain evolve <concept>` | Concept lineage tree |
| `drbrain evolve <c> --stats --mermaid` | Temporal evolution + diagram |
| `drbrain descendants <id>` | Academic offspring (3 generations) |
| `drbrain descendants <id> -g 5 --mermaid` | Deeper trace with diagram |
| `drbrain landscape` | Domain timeline: gaps, debates, cliffs |
| `drbrain landscape <ws> --top-n 10` | Workspace landscape |
| `drbrain paradigm` | Paradigm shift detection |
| `drbrain paradigm <concept>` | Check specific concept for shifts |
| `drbrain paradigm -w <ws>` | Workspace shift scan |
| `drbrain transfers --auto` | Auto-detect cross-domain transfers |
| `drbrain transfers --from A --to B` | Directed transfer analysis |
| `drbrain isomorphism` | Find structurally similar subgraphs |
| `drbrain isomorphism <concept>` | Patterns matching a concept |
| `drbrain difficulty` | Gap difficulty by section type |
| `drbrain frontier` | Composite knowledge frontier report |
