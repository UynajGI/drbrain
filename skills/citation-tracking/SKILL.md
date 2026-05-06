---
name: citation-tracking
description: >
  Track and analyze citation relationships between papers. Use this skill whenever the user asks
  "who cites this?", "what does this paper reference?", "find related work", "are these papers
  connected?", "show me the citation graph", "check my citations", or wants to discover papers that
  share references but don't cite each other (knowledge frontier signal). Also use when the user
  mentions "shared references", "citation neighborhood", "verify citations", wants to check if their
  writing properly cites papers in the library, or needs to explore the intellectual lineage of a
  research area. Trigger proactively whenever the user discusses citation relationships, asks about
  paper-to-paper connections, or wants to validate reference lists.
---

# Citation Tracking

Analyze citation relationships to map the knowledge frontier. The citation graph reveals hidden
connections, missing links between research communities, and shared intellectual heritage.

## Workflow

### Step 1: Basic citation queries

```bash
drbrain citations p3f8a2                    # both refs and citing papers
drbrain citations p3f8a2 --type refs          # only what this paper references
drbrain citations p3f8a2 --type citing         # only papers that cite this one
drbrain citations p3f8a2 --type all --json     # full structured output
```

### Step 2: Shared-reference analysis (frontier signal)

```bash
drbrain citations p3f8a2 --type shared-refs
```

Papers marked `unlinked` share references but don't cite each other. This is a knowledge frontier
signal: two groups working on the same problem, reading the same literature, but unaware of each
other. High `shared_count` with `unlinked` status indicates:
- Parallel discovery worth investigating
- Literature review opportunities that bridge communities
- Potential duplicated effort

### Step 3: Citation verification

Check if in-text citations in writing match papers in the library:

```bash
drbrain check-citations "Smith (2023) proposed a method that Jones et al. (2022) extended."
drbrain check-citations --file draft.txt
```

Output shows matched and unmatched citations. Unmatched ones may exist under a different name
variant or need ingest.

### Step 4: Workspace-scoped analysis

```bash
drbrain citations p3f8a2 --type shared-refs --workspace attention-methods
```

## Examples

**Find unlinked papers that share references (frontier detection):**
```bash
drbrain citations p3f8a2 --type shared-refs --json | jq '.[] | select(.status == "unlinked")'
```

**Verify a draft's citations against the library:**
```bash
drbrain check-citations --file intro-draft.txt
```

## CLI Reference

| Command | What it does |
|---------|--------------|
| `drbrain citations <id>` | Both refs and citing papers |
| `drbrain citations <id> --type refs` | References only |
| `drbrain citations <id> --type citing` | Papers citing this one |
| `drbrain citations <id> --type shared-refs` | Shared-reference frontier signals |
| `drbrain citations <id> -w <ws>` | Workspace-scoped citation analysis |
| `drbrain check-citations "<text>"` | Verify in-text citations |
| `drbrain check-citations --file <path>` | Verify citations in a file |
