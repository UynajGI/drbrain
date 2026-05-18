---
name: document
description: >
  Inspect Office documents (DOCX, PPTX, XLSX) and output structured text summaries.
  Use when the user wants to check document structure, verify content, detect layout
  issues, or preview Office files without opening a GUI. Trigger on "inspect this document",
  "check this Word file", "what's in this PowerPoint", "examine this Excel spreadsheet",
  "verify document layout".
---

# Document Inspection

Inspect Office documents without a GUI. Returns structured summaries of structure, content,
and potential issues (overflow, missing content).

## Quick Start

```bash
drbrain document inspect --file presentation.pptx
drbrain document inspect --file report.docx
drbrain document inspect --file data.xlsx
```

## Installation

```bash
uv sync --extra office
# or
pip install drbrain[office]
```

## CLI Reference

| Command | What it does |
|---------|--------------|
| `drbrain document inspect --file <path>` | Inspect Office document |
