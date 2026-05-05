---
name: export
description: >
  Export papers to BibTeX, RIS, or Markdown. Use this skill whenever the user asks to "export references",
  "generate bibtex", "get bibliography", "export my library", "create a .bib file", "format citations",
  or needs to move paper metadata to a reference manager or document.
---

# Export

Export paper metadata from the DrBrain library to standard reference formats: BibTeX (`.bib`), RIS
(`.ris`), or Markdown (`list`). Supports single paper, all papers, or workspace-scoped export.

## Quick Start

```bash
# Export a single paper as BibTeX
drbrain export p3f8a2 --format bib

# Export all papers to a .bib file
drbrain export --all --format bib --output references.bib
```

## What It Does

- Converts paper metadata (title, authors, year, DOI, journal, abstract, etc.) to the chosen format
- Authors are drawn from Actor-type concepts with their canonical aliases
- `--all` flag exports every paper in the library
- `--output` flag writes to a file instead of stdout
- `--json` flag wraps the result in JSON
- Works with workspace-scoped exports when combined with `--workspace` (for `--all` exports, filter
  first via `drbrain ws`)

## CLI Reference

| Command | What it does |
|---------|--------------|
| `drbrain export <id> --format bib` | Single paper as BibTeX |
| `drbrain export <id> --format ris` | Single paper as RIS |
| `drbrain export <id> --format md` | Single paper as Markdown |
| `drbrain export --all -f bib -o out.bib` | All papers to BibTeX file |
| `drbrain export --all -f ris -o out.ris` | All papers to RIS file |
| `drbrain export --all -f md -o out.md` | All papers to Markdown file |

## Common Patterns

**Export for Overleaf or LaTeX:**
```bash
drbrain export --all --format bib --output references.bib
```

**Export for Zotero/Mendeley import:**
```bash
drbrain export --all --format ris --output library.ris
```

**Export a reading list as Markdown:**
```bash
drbrain export --all --format md --output reading-list.md
```

**Export workspace papers:**
```bash
drbrain ws list                          # confirm workspace
drbrain export --all --format bib        # (if current workspace is active)
```
