---
name: research-analysis
description: >
  Analyze the knowledge frontier of your paper collection — find research gaps, causal chains,
  emerging/declining trends, cross-domain method transfer opportunities, and testable hypotheses.
  Use this skill whenever the user asks "what should I research next?", "where are the gaps?",
  "what's the knowledge frontier?", "find research opportunities", "what are the emerging trends?",
  "how are these methods connected across domains?", "generate hypotheses from my papers", or wants
  to understand research dynamics across their entire library. Also use when the user mentions
  literature review synthesis, wants to discover underexplored areas, or asks about causal
  relationships between problems and solutions. Trigger proactively whenever the user discusses
  research strategy, asks about the state of a field based on their papers, or wants to identify
  high-impact open problems.
---

# Research Analysis

Orchestrate DrBrain's symbolic reasoning modules to produce a unified knowledge frontier report
covering graph closure inference, research seed detection, causal chains, and (optionally)
counterfactual analysis, hypothesis generation, and cross-domain isomorphism detection.

## Workflow

### Step 1: Assess the library

```bash
drbrain list
drbrain stats
```

If the library is empty, guide the user to ingest papers first (`paper-ingest` skill).

### Step 2: Run a landscape scan

Scan the entire library for research dynamics:

```bash
drbrain seed --json
```

Key signal types in the output:

- **contested**: active debate with low average confidence. Richest targets for new research.
- **emerging**: rapidly growing over the last 2 years. Early-adopter advantage.
- **resurging**: dormant for 3+ years then active again. A technical barrier was likely removed.
- **declining**: last activity > 3 years ago. Could be dead or ripe for revival with modern methods.

### Step 3: Deep dive on a specific paper

```bash
drbrain analyze p3f8a2 --full --json
```

Interpret the output sections:

- **causal_chains**: `source → target (via: mechanism)`. Multiple chains converging on one target
  suggest strong evidence. Long fragile chains collapse if any link is wrong.
- **critical_nodes**: concepts whose removal would collapse the most inferred edges. If a critical
  node is also contested, resolving that debate affects everything downstream.
- **hypotheses**: `gap_filling`, `debate_resolution`, or `technology_revival`, each with confidence.
- **isomorphisms**: structurally similar subgraphs from different domains. High similarity means
  method transfer across domains is promising.

### Step 4: Workspace-scoped analysis

For focused literature reviews:

```bash
drbrain ws list
drbrain analyze --workspace gnn-drugs --json
```

Compare workspace results to full-library results. Critical nodes appearing in both are genuinely
field-central. Hypotheses scoped to the workspace are more directly actionable.

## Examples

**Quick landscape of the entire library:**
```bash
drbrain seed --json | jq '.[] | {type, label, signal}'
```

**Full analysis of a specific paper with hypotheses:**
```bash
drbrain analyze p3f8a2 --full --json | jq '{causal_chains, critical_nodes, hypotheses}'
```

**Workspace-level frontier scan:**
```bash
drbrain analyze --workspace attention-methods --full --json
```

## CLI Reference

| Command | What it tells you |
|---------|-------------------|
| `drbrain seed` | Research dynamics across all concepts |
| `drbrain analyze <id>` | Paper-level analysis with seeds, chains, edges |
| `drbrain analyze <id> --full` | Adds counterfactual, hypotheses, isomorphism |
| `drbrain analyze -w <ws>` | Workspace-scoped analysis |
| `drbrain citations <id> --type shared-refs` | Shared-reference frontier signals |
| `drbrain closure` | Run graph inference rules, see new edges |

## Related Skills

- **knowledge-cartography** — `evolve`, `descendants`, `landscape`, `paradigm`, `transfers`, `isomorphism`, `difficulty`, `frontier`, `seed` for deeper analysis of each dimension
- **kg-reason** — `reason` for LLM agent reasoning over analysis results
- **kg-build** — `build`, `embed`, `closure` to regenerate the graph before analysis
- **graph** — `graph neighbors/path/related` for direct graph exploration
