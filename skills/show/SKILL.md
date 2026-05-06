---
name: show
description: >
  View paper contents at any depth — metadata, concepts by type, arguments with evidence, and
  knowledge graph edges. Use this skill whenever the user mentions a specific paper ID (like p3f8a2),
  wants to inspect a paper's contents, asks "what's in paper X?", "show me paper X", "details of
  paper X", "tell me about this paper", "what concepts did we extract from X?", or needs to check
  if a paper was ingested correctly. Also use before running analysis on a paper to verify its
  contents are complete (check concept count, edge count, abstract presence), or when the user wants
  to find a paper's DOI, title, journal, or citation metadata. Trigger proactively whenever a paper
  ID appears in conversation.
---

# Show Paper

Display a single paper's full contents: bibliographic metadata, extracted concepts (grouped by TBox
type), arguments (claims with mechanisms and targets), and outgoing/incoming knowledge graph edges.

## Quick Start

```bash
drbrain show p3f8a2
```

## What It Shows

- Bibliographic metadata: title, year, paper type, status, journal, DOI, abstract, citation count
- Concepts grouped by type: Problem, Method, Conclusion, Gap, Debate, Actor
- Arguments: claim type, claim text, and target concept
- Outgoing edges: what this paper asserts about other concepts
- Incoming edges: what other papers assert about this paper's concepts

## Common Patterns

**Check if a paper ingested correctly:**
```bash
drbrain show p3f8a2
```
Look for concept count, argument count, and edge count. Zero concepts usually means the build step
was skipped or failed. Zero edges may mean the LLM extraction didn't find clear relationships.

**Find a paper's DOI for external lookup:**
```bash
drbrain show p3f8a2 --json | jq '.paper.doi'
```

**Quick concept inventory:**
```bash
drbrain show p3f8a2 --json | jq '.concepts | group_by(.type) | map({type: .[0].type, count: length})'
```

## Examples

**Verify paper completeness before analysis:**
```bash
drbrain show p3f8a2 --json | jq '{concepts: (.concepts | length), edges: (.edges | length), has_abstract: (.paper.abstract != null)}'
```

**View concepts of a specific type:**
```bash
drbrain show p3f8a2 --json | jq '.concepts[] | select(.type == "Method") | .label'
```

## CLI Reference

| Command | What it shows |
|---------|---------------|
| `drbrain show <id>` | Full paper view with concepts, arguments, edges |
| `drbrain show <id> --json` | Machine-readable JSON output |
