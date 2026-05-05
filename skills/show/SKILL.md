---
name: show
description: >
  View a paper at any depth — metadata, concepts, arguments, and graph edges. Use this skill whenever
  the user asks to "show me paper X", "what's in paper X", "details of paper X", "tell me about this
  paper", or wants to inspect the contents of a specific paper in their library.
---

# Show Paper

Display a single paper's full contents: bibliographic metadata, extracted concepts (grouped by TBox
type), arguments (claims + mechanisms + targets), and outgoing/incoming knowledge graph edges.

## Quick Start

```bash
drbrain show p3f8a2
```

## What It Does

- Shows title, year, paper type, status, journal, DOI, abstract, and citation count
- Lists concepts grouped by type (Problem, Method, Conclusion, Gap, Debate, Actor)
- Lists arguments with claim type, claim text, and target concept
- Shows outgoing edges (what this paper asserts about other concepts)
- Shows incoming edges (what other papers assert about this paper's concepts)
- `--json` flag for machine-readable output

## CLI Reference

| Command | What it shows |
|---------|---------------|
| `drbrain show <id>` | Full paper view with concepts, arguments, edges |
| `drbrain show <id> --json` | Machine-readable JSON output |

## Common Patterns

**Check if a paper ingested correctly:**
```bash
drbrain show p3f8a2
```
Look for concept count, argument count, and edge count. Zero concepts usually means the build step
was skipped or failed. Zero edges may mean the LLM extraction didn't find clear relationships.

**Find paper ID to use in other commands:**
```bash
drbrain list          # see all paper IDs
drbrain show p<id>    # inspect the one you want
```

**Pipe to jq for quick checks:**
```bash
drbrain show p3f8a2 --json | jq '.concepts | length'
drbrain show p3f8a2 --json | jq '.paper.doi'
```
