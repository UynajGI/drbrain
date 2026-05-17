---
name: workspace-analysis
description: >
  Manage and analyze paper subsets (workspaces) for focused research projects. Use this skill
  whenever the user wants to organize papers into projects, create a reading list for a specific
  topic, curate a literature review collection, analyze a subset of their library, or compare
  research dynamics within a workspace against the full library. Also use when the user mentions
  "literature review", "reading list", "project papers", "organize my research", "group these
  papers", "create a curated collection", "scope analysis to this domain", or wants to run analysis
  commands on a specific subset of papers rather than the whole library. Trigger proactively when
  the user talks about focusing on a research question, preparing for a paper submission, or
  organizing papers by topic.
---

# Workspace Analysis

Workspaces are paper subsets that scope DrBrain's reasoning to a specific topic. Each workspace
stores a `papers.json` reference list — papers are not copied, only referenced.

## Prerequisites

Papers must be ingested and built (`paper-ingest` + `kg-build` skills). Workspace-scoped analysis commands (`analyze -w`, `seed -w`, `closure -w`) need the KG built.

## Workflow

### Step 1: Create and populate a workspace

```bash
drbrain ws create gnn-drugs -d "Graph neural networks for drug discovery"
drbrain ws add gnn-drugs p3f8a2 p7b1c4 p9d2e5
drbrain ws show gnn-drugs
```

### Step 2: Manage workspace contents

```bash
drbrain ws remove gnn-drugs p3f8a2
drbrain ws list
drbrain ws delete gnn-drugs         # removes workspace, not the papers
```

### Step 3: Run scoped analysis

All analysis commands accept `--workspace` / `-w`:

```bash
drbrain analyze --workspace gnn-drugs --full --json
drbrain seed --workspace gnn-drugs
drbrain query "binding affinity" --workspace gnn-drugs
drbrain stats --workspace gnn-drugs
drbrain closure --workspace gnn-drugs
drbrain export --workspace gnn-drugs --format bib
```

### Step 4: Interpret workspace results

When running analysis on a well-curated workspace:
- **Seeds** reflect dynamics within this subfield, not all of science
- **Critical nodes** appearing in both workspace and full-library results are genuinely field-central
- **Hypotheses** are scoped to the workspace's domain, making them more actionable
- **Isomorphisms** within a workspace may reveal method transfer between sub-topics

## Examples

**Create a focused reading list and analyze it:**
```bash
drbrain ws create attention-survey -d "Attention mechanisms in transformers"
drbrain ws add attention-survey p3f8a2 p7b1c4 p9d2e5 p1a4b6
drbrain analyze --workspace attention-survey --full --json
```

**Export a workspace for Overleaf:**
```bash
drbrain export --workspace gnn-drugs --format bib --output gnn-refs.bib
```

## CLI Reference

| Command | What it does |
|---------|--------------|
| `drbrain ws create <name> -d "<desc>"` | Create a new workspace |
| `drbrain ws add <name> <id...>` | Add papers to a workspace |
| `drbrain ws remove <name> <id>` | Remove a paper from a workspace |
| `drbrain ws list` | List all workspaces |
| `drbrain ws show <name>` | Show workspace contents |
| `drbrain ws delete <name>` | Delete a workspace |
| `drbrain analyze -w <name>` | Workspace-scoped analysis |
| `drbrain backup` | Backup workspaces along with library |
