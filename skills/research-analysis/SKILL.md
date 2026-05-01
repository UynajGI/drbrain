---
name: research-analysis
description: >
  Analyze the knowledge frontier of your paper collection. Use this skill whenever the user wants to
  find research gaps and opportunities, understand causal relationships between methods and problems,
  discover cross-domain method transfers, generate testable hypotheses, identify critical (high-impact)
  concepts, or get a comprehensive overview of what their paper library reveals about the state of
  research. Also use when the user asks "what should I research next?", "where are the gaps?", "what's
  the knowledge frontier?", or wants to understand how different research directions connect.
---

# Research Analysis

This skill orchestrates DrBrain's reasoning modules to produce a unified knowledge frontier report. It
runs graph closure inference, research seed detection, causal chain discovery, and (in full mode)
counterfactual analysis, hypothesis generation, and cross-domain isomorphism detection.

## Prerequisites

Before starting, verify the environment is ready:

```bash
drbrain check
```

If papers are not yet ingested, guide the user through `paper-ingest` first.

## Workflow

### Step 1: Assess what the user has

Start by showing the user what's in their library:

```bash
drbrain list
drbrain stats
```

If the library is empty, suggest they ingest some papers first (see `paper-ingest` skill).

### Step 2: Quick landscape scan

For a high-level overview of research dynamics across the entire library:

```bash
drbrain seed --json
```

The `type` and `signal` fields tell you what's happening. These patterns are worth calling out:

- **contested**: active debate with low average confidence. These are the richest targets for new
  research — either the evidence is mixed or the methodologies disagree. Look deeper with `drbrain
  citations <id> --type shared-refs` to see if the disputing papers share references (they might be
  talking past each other).
- **emerging**: rapidly growing over the last 2 years. Early-adopter advantage if you act now. Check
  `drbrain timeline <concept>` to see the growth curve.
- **resurging**: dormant for 3+ years then active again. Usually means a technical barrier was removed
  or a new method revived the field. The `causal_chains` in `drbrain analyze` will often reveal the
  mechanism.
- **declining**: last activity > 3 years ago. Could be a dead end, or could be ripe for revival with
  modern methods (technology cliff).

### Step 3: Deep dive on specific papers

For any paper that looks interesting, run the full analysis:

```bash
drbrain analyze <local_id> --full --json
```

The output has several sections — interpret them as follows:

**causal_chains**: Shows `source → target (via: mechanism)`. These are extracted from the paper's
arguments. Multiple chains converging on the same target suggest strong evidence. A long chain
(A→B→C→D) with only weak evidence at each hop is fragile — the conclusion collapses if any link is
wrong.

**critical_nodes**: These are concepts whose removal would cause the most inferred edges to disappear.
They're the load-bearing pillars of the graph. If a critical node is `contested`, that's a high-impact
research opportunity — resolving the debate affects everything downstream.

**hypotheses**: Generated from graph patterns. Each has a type and confidence:
- `gap_filling`: a known method could address an unaddressed gap
- `debate_resolution`: conflicting evidence needs new data to resolve
- `technology_revival`: a stalled method could work under new conditions

**isomorphisms**: Structurally similar subgraphs from different domains. A `high` similarity score
means the same pattern of relationships appears in two different contexts — this is where method
transfer across domains is most promising.

### Step 4: Workspace-level analysis

For multi-paper projects, workspaces let you focus the analysis:

```bash
drbrain ws list
drbrain analyze --workspace <name> --json
```

This runs the same analysis but scoped to the workspace's paper subset. Use this for literature reviews
or when exploring a specific research direction.

## CLI Reference

| Command | What it tells you |
|---------|-------------------|
| `drbrain seed` | Research dynamics across all concepts |
| `drbrain analyze <id>` | Paper-level analysis with seeds, chains, and inferred edges |
| `drbrain analyze <id> --full` | Adds counterfactual, hypotheses, and isomorphism detection |
| `drbrain analyze -w <ws>` | Workspace-scoped boundary scan |
| `drbrain timeline <concept>` | Year-by-year evolution of a concept |
| `drbrain citations <id> --type shared-refs` | Papers that share references but don't cite each other |
| `drbrain closure` | Run graph inference rules and see new edges |

## Output interpretation guide

When presenting results to the user, don't just dump JSON. Connect the dots:

1. Start with the research seeds — what's the landscape?
2. Point out contested concepts backed by critical nodes — these are the highest-leverage targets
3. For each interesting hypothesis, check whether it's supported by real evidence (paper edges) or
   only inferred by graph rules. Real evidence > inferred relationships.
4. If the user has a workspace set up, compare workspace results to full-library results to see what's
   unique about their subset.
