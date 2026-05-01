---
name: paper-ingest
description: >
  Ingest PDFs into the DrBrain knowledge graph. Use this skill whenever the user wants to add papers
  to their library, import PDFs, process academic papers, or build up their research collection. Also
  use when the user mentions downloading papers, adding to their library, or needs to parse and
  extract concepts from academic PDFs. This is the first step for any research workflow — papers must
  be ingested before they can be analyzed.
---

# Paper Ingest

Add papers to the DrBrain knowledge graph. This pipeline parses the PDF (MinerU CLI → PyMuPDF
fallback), extracts structured concepts and arguments via LLM, resolves paper identity (DOI/arXiv),
fetches citations, and runs graph inference.

## Prerequisites

```bash
drbrain check
```

Papers go into `data/spool/inbox/`. The ingest command scans this directory by default.

## Workflow

### Basic ingest

```bash
# Put PDFs in the inbox and run:
drbrain ingest

# Or ingest specific files:
drbrain ingest paper1.pdf paper2.pdf

# Ingest a directory:
drbrain ingest /path/to/papers/
```

### Check results

After ingest, verify everything worked:

```bash
drbrain list                    # See all papers
drbrain report <local_id>       # Detailed per-paper report
```

### Handle failures

If a paper fails to process, it moves to `data/spool/pending/`. Check why:

```bash
cat data/spool/pending/pending.jsonl
```

Common failures and fixes:
- **PDF parse error**: The PDF may be scanned or corrupted. Check with `drbrain check` that PyMuPDF is
  available as a fallback.
- **LLM extraction failed**: All configured LLM models were exhausted. Check API keys with `drbrain
  check`.
- **No DOI found**: The paper couldn't be identified. This is usually fine — concepts are still
  extracted, just without external enrichment.

### What happens during ingest

The pipeline runs 9 stages automatically:
1. Parse PDF to markdown (MinerU CLI → PyMuPDF)
2. Identify paper (DOI/arXiv/title matching)
3. Build document tree structure
4. Extract concepts + arguments via LLM
5. Validate against schema rules
6. Queue low-confidence items for review
7. Align concepts with canonical IDs
8. Expand citations (Semantic Scholar / CrossRef / OpenAlex)
9. Run graph closure inference
