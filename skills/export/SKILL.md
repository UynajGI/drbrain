---
name: export
description: >
  Export papers to BibTeX, RIS, or Markdown format. Use this skill whenever the user asks to
  "export references", "generate bibtex", "get a bibliography", "export my library", "create a .bib
  file", "format citations for LaTeX", "export for Zotero", "make a reading list", or needs to move
  paper metadata to a reference manager (Zotero, Mendeley, Endnote) or document (Overleaf, Word).
  Also use when the user mentions "references.bib", "bibliography export", "RIS export", wants to
  export a workspace's papers for a paper submission, or needs formatted citations for any external
  tool. Trigger proactively when the user talks about getting papers out of DrBrain into another
  system.
---

# Export Papers

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

- Converts paper metadata (title, authors, year, DOI, journal, abstract) to the chosen format
- Authors are drawn from Actor-type concepts with their canonical aliases
- `--all` exports every paper in the library
- `--output` writes to a file instead of stdout
- `--json` wraps the result in JSON
- Workspace-scoped export via `--workspace`

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

**Export workspace papers for a paper submission:**
```bash
drbrain export --workspace attention-methods --format bib --output attention-refs.bib
```

## Examples

**Single paper BibTeX for pasting into a document:**
```bash
drbrain export p3f8a2 --format bib
```

**Full library RIS export:**
```bash
drbrain export --all --format ris --output my-library.ris
```

## CLI Reference

| Command | What it does |
|---------|--------------|
| `drbrain export <id> --format bib` | Single paper as BibTeX |
| `drbrain export <id> --format ris` | Single paper as RIS |
| `drbrain export <id> --format md` | Single paper as Markdown |
| `drbrain export --all -f bib -o out.bib` | All papers to BibTeX file |
| `drbrain export --all -f ris -o out.ris` | All papers to RIS file |
| `drbrain export --workspace <ws> -f bib` | Workspace-scoped export |
