---
name: import
description: >
  Import papers from external reference managers into the DrBrain library. Use this skill whenever
  the user wants to "import from Zotero", "add my BibTeX library", "migrate references from
  Endnote", "import my existing collection", "bring in papers from my reference manager", or
  transition from another tool to DrBrain. Also use when the user mentions having papers in Zotero,
  a .bib file, an Endnote XML export, or a RIS file, and wants to add them to the knowledge graph.
  Trigger proactively whenever the user discusses moving from another reference manager, combining
  multiple paper collections, or bootstrapping their DrBrain library from existing sources.
---

# Import Papers

Import paper metadata from Zotero (Web API or local SQLite database), BibTeX `.bib` files, or
Endnote (`.xml` or `.ris`) exports. Imported papers are created as placeholders — run `drbrain
ingest` afterward if you have the PDFs, or `drbrain repair --all` to enrich metadata from online
sources.

## Quick Start

```bash
# Import from Zotero local database
drbrain import zotero ~/Zotero/zotero.sqlite

# Import from BibTeX file
drbrain import bibtex references.bib

# Import from Endnote export
drbrain import endnote library.xml
```

## Source-specific details

### Zotero

Two modes: local SQLite (auto-detects adjacent PDFs) and Web API (`--api-key`, `--library-id`).

```bash
drbrain import zotero ~/Zotero/zotero.sqlite --list-collections    # preview
drbrain import zotero ~/Zotero/zotero.sqlite --collection <key>    # filter
drbrain import zotero ~/Zotero/zotero.sqlite --no-pdf              # metadata only
drbrain import zotero ~/Zotero/zotero.sqlite --import-collections   # workspaces per collection
drbrain import zotero ~/Zotero/zotero.sqlite --dry-run             # preview only
```

### BibTeX

Parses `.bib` files via `bibtexparser`. Extracts title, authors, year, DOI, journal, abstract.
Supports standard entry types (article, inproceedings, book, etc.).

### Endnote

Supports `.xml` (Endnote XML export) and `.ris` (common export format).

## What to do after import

Imported papers have status `placeholder` (metadata only):

1. If PDFs were detected (Zotero local mode): `drbrain ingest`
2. If no PDFs: `drbrain repair --all` to fill missing metadata from CrossRef/arXiv

## Examples

**Full Zotero migration with workspaces:**
```bash
drbrain import zotero ~/Zotero/zotero.sqlite --list-collections
drbrain import zotero ~/Zotero/zotero.sqlite --import-collections
drbrain ingest
drbrain repair --all
```

**Import a shared BibTeX file with preview:**
```bash
drbrain import bibtex ~/Downloads/iclr2024.bib --dry-run
drbrain import bibtex ~/Downloads/iclr2024.bib
drbrain repair --all
```

## CLI Reference

| Command | What it does |
|---------|--------------|
| `drbrain import zotero <path>` | Import from Zotero local DB |
| `drbrain import zotero <path> --list-collections` | List Zotero collections |
| `drbrain import zotero <path> --dry-run` | Preview without importing |
| `drbrain import zotero <key> --api-key X --library-id Y` | Zotero Web API |
| `drbrain import bibtex <file>` | Import from .bib file |
| `drbrain import endnote <file>` | Import from .xml or .ris file |
| `drbrain import <src> <path> --json` | JSON output |
