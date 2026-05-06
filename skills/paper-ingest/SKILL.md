---
name: paper-ingest
description: >
  Ingest PDFs into the DrBrain knowledge graph — parse, identify, and extract structured concepts
  and arguments from academic papers. Use this skill whenever the user wants to add papers to their
  library, import PDFs, process academic papers, build up their research collection, or has new PDFs
  to work with. Also use when the user mentions downloading papers, "add this to my library",
  "process these PDFs", "ingest my arXiv downloads", or needs papers parsed before they can be
  queried or analyzed. This is the mandatory first step for any research workflow — papers must be
  ingested before they can be searched, analyzed, or cited. Trigger proactively whenever the user
  has PDF files they want to work with in DrBrain.
---

# Paper Ingest

Add papers to the DrBrain knowledge graph via a 9-stage pipeline: PDF parsing (MinerU CLI with
PyMuPDF fallback), identity resolution (DOI/arXiv via 5-source cross-validation), document tree
structuring, LLM concept/argument extraction, citation expansion, and graph closure inference.

## Prerequisites

```bash
drbrain check
```

Papers go into `data/spool/inbox/`. The ingest command scans this directory by default.

## Workflow

### Step 1: Add papers to the inbox

Place PDFs in `data/spool/inbox/`, then run:

```bash
drbrain ingest
```

Or target specific files directly:

```bash
drbrain ingest paper1.pdf paper2.pdf
drbrain ingest /path/to/papers/
```

### Step 2: Verify results

```bash
drbrain list                      # see all papers
drbrain show p3f8a2               # inspect a specific paper
```

### Step 3: Handle failures

Failed papers move to `data/spool/pending/`. Diagnose with:

```bash
cat data/spool/pending/pending.jsonl
```

Common failures:
- **PDF parse error**: the PDF may be scanned or corrupted. PyMuPDF fallback should handle most cases.
- **LLM extraction failed**: all configured models exhausted. Check API keys with `drbrain check`.
- **No DOI found**: paper couldn't be identified externally. Concepts are still extracted.

## Examples

**Ingest a single paper directly:**
```bash
drbrain ingest ~/Downloads/attention-is-all-you-need.pdf
drbrain show p3f8a2
```

**Batch ingest from arXiv downloads:**
```bash
drbrain ingest ~/Downloads/arxiv-papers/
drbrain list
```

## CLI Reference

| Command | What it does |
|---------|--------------|
| `drbrain ingest` | Process all PDFs in `data/spool/inbox/` |
| `drbrain ingest <file>` | Ingest specific PDF files |
| `drbrain ingest <dir>` | Ingest all PDFs in a directory |
| `drbrain list` | List all papers in the library |
| `drbrain show <id>` | Inspect a paper's metadata and concepts |
