---
name: import
description: >
  Import papers from reference managers into the DrBrain library. Use this skill whenever the user
  wants to "import from Zotero", "add my bibtex library", "migrate references from Endnote", "import
  my existing collection", or bring papers in from an external reference manager.
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

## What It Does

### Zotero
Two modes:
- **Local SQLite**: point to `zotero.sqlite` (typically in `~/Zotero/`). Auto-detects PDFs in the
  adjacent `storage/` directory.
- **Web API**: requires `--api-key` and `--library-id`. Fetches papers from Zotero's cloud sync.

Options:
- `--collection <key>`: filter by collection
- `--list-collections`: list all collections and exit
- `--no-pdf`: skip PDF detection
- `--import-collections`: create workspaces per collection after import
- `--library-type user|group`: for Web API mode
- `--dry-run`: preview only, no database writes

### BibTeX
Parse `.bib` files using Python's `bibtexparser`. Extracts title, authors, year, DOI, journal, and
abstract. Supports standard BibTeX entry types (article, inproceedings, book, etc.).

### Endnote
Two formats:
- `.xml`: Endnote XML export
- `.ris`: RIS format (common export format)

## CLI Reference

| Command | What it does |
|---------|--------------|
| `drbrain import zotero <path>` | Import from Zotero local DB |
| `drbrain import zotero <path> --list-collections` | List Zotero collections |
| `drbrain import zotero <path> --collection <key>` | Import one collection |
| `drbrain import zotero <path> --no-pdf` | Skip PDF detection |
| `drbrain import zotero <path> --dry-run` | Preview without importing |
| `drbrain import zotero <key> --api-key X --library-id Y` | Zotero Web API |
| `drbrain import bibtex <file>` | Import from .bib |
| `drbrain import endnote <file>` | Import from .xml or .ris |
| `drbrain import <src> <path> --json` | JSON output |

## Common Patterns

**Full Zotero migration:**
```bash
# 1. List collections to see what's there
drbrain import zotero ~/Zotero/zotero.sqlite --list-collections

# 2. Import everything (with PDFs)
drbrain import zotero ~/Zotero/zotero.sqlite

# 3. Import as workspaces
drbrain import zotero ~/Zotero/zotero.sqlite --import-collections

# 4. Process the papers
drbrain ingest
drbrain build
drbrain repair --all
```

**Import from a shared BibTeX file:**
```bash
drbrain import bibtex ~/Downloads/iclr2024.bib --dry-run   # preview first
drbrain import bibtex ~/Downloads/iclr2024.bib              # then import
drbrain repair --all                                         # enrich metadata
```

**After import — what to do:**
- Imported papers have status `placeholder`. They exist as metadata only.
- If PDFs were detected (Zotero local mode), `drbrain ingest` will process them.
- If no PDFs, run `drbrain repair --all` to fill in missing metadata from CrossRef/arXiv.
