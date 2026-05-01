---
name: workspace-analysis
description: >
  Manage and analyze paper subsets (workspaces) for focused research projects. Use this skill whenever
  the user wants to organize papers into projects, create a reading list for a specific topic, analyze
  a subset of their library, or compare research dynamics within a workspace against the full library.
  Also use when the user mentions "literature review", "reading list", "project papers", "organize my
  research", or wants to scope analysis to a specific domain or research question.
---

# Workspace Analysis

Workspaces are paper subsets that let you focus DrBrain's reasoning on a specific topic. Each
workspace has its own `papers.json` reference list — papers are not copied, just referenced.

## Workspace management

```bash
drbrain ws create my-project -d "Graph neural networks for drug discovery"
drbrain ws add my-project p1a2b3c4 p5d6e7f8
drbrain ws remove my-project p1a2b3c4
drbrain ws list
drbrain ws show my-project
drbrain ws delete my-project
```

## Focused analysis

All analysis commands accept `--workspace` (or `-w`) to scope results:

```bash
drbrain analyze --workspace my-project --full --json
drbrain seed --workspace my-project
drbrain query "binding affinity" --workspace my-project
drbrain stats --workspace my-project
drbrain closure --workspace my-project
drbrain export --workspace my-project --format bib
```

## Interpreting workspace results

When you run `drbrain analyze --workspace` on a well-curated workspace, the results are more focused
than full-library analysis. The graph only contains edges from workspace papers, so:

- **Seeds** reflect dynamics within this specific subfield, not all of science
- **Critical nodes** that appear in both workspace and full-library analysis are genuinely
  field-central — they're important at multiple scales
- **Hypotheses** are scoped to the workspace's domain, making them more actionable
- **Isomorphisms** found within a workspace might reveal method transfer opportunities between
  sub-topics you're tracking

## Backup

Workspaces are stored as files — back them up along with the rest of the library:

```bash
drbrain backup
```
