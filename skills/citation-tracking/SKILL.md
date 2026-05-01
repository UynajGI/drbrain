---
name: citation-tracking
description: >
  Track and analyze citation relationships between papers. Use this skill whenever the user wants to
  explore who cites whom, find shared references between papers, verify that in-text citations match
  their library, discover papers that should know about each other but don't (unlinked via shared
  refs), or understand the citation neighborhood of a paper. Also use when the user asks about
  "citation graph", "who cites this?", "find related work", "are these papers connected?", or wants
  to check if their writing properly cites papers in the library.
---

# Citation Tracking

Analyze citation relationships to map the knowledge frontier. The citation graph reveals hidden
connections, missing links, and shared intellectual heritage.

## Core queries

### References and citations

```bash
drbrain citations <local_id>                    # Both refs and citing papers
drbrain citations <local_id> --type refs          # Only what this paper references
drbrain citations <local_id> --type citing         # Only papers that cite this one
drbrain citations <local_id> --type all --json     # Full structured output
```

### Shared references (frontier signal)

The most valuable citation analysis for knowledge frontier work:

```bash
drbrain citations <local_id> --type shared-refs
```

This finds papers that cite the same references as the target paper. Papers marked `unlinked` share
references but don't cite each other — this is a knowledge frontier signal. Two groups working on
the same problem, reading the same literature, but not aware of each other.

High `shared_count` with `unlinked` status is a strong indicator of a research gap or parallel
discovery. These paper pairs are candidates for:
- Literature reviews that bridge communities
- Collaborative opportunities
- Identifying duplicated effort

### Citation verification

Check if in-text citations in your writing match papers in the library:

```bash
drbrain check-citations "Smith (2023) proposed a method that Jones et al. (2022) extended."
drbrain check-citations --file draft.txt
```

Output shows `✓` for matched citations and `✗` for unmatched ones. Unmatched citations might be in
the library under a different name variant (check aliases) or genuinely missing (needs ingest).

## Workspace-scoped citation analysis

Combine with workspaces for focused analysis:

```bash
drbrain citations <local_id> --type shared-refs --workspace <name>
```

## Export

Export citation data for use in papers:

```bash
drbrain export <local_id> --format bib
drbrain export <local_id> --format ris
drbrain export --workspace <name> --format bib
```
