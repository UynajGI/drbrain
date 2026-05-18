---
name: ingest-link
description: >
  Ingest web URLs and online PDFs by extracting rendered content via an external
  qt-web-extractor service. Use when the user wants to save web pages, articles,
  or online PDFs into their DrBrain library. Trigger on "save this webpage",
  "ingest this URL", "add this article to my library", "download and save this page".
---

# Ingest Web Links

Extract rendered content from web URLs via an external qt-web-extractor service
and save as paper records in DrBrain.

## Prerequisites

An external `qt-web-extractor` service must be running (default: `http://127.0.0.1:8766`).
Set `WEBEXTRACT_URL` to configure a custom endpoint.

## Quick Start

```bash
drbrain ingest-link https://example.com/page
drbrain ingest-link https://example.com/report.pdf --pdf
drbrain ingest-link https://a.com https://b.com
drbrain ingest-link https://example.com --dry-run
```

## How it works

1. POSTs the URL to the external extractor service
2. Receives rendered text (markdown), title, and metadata
3. Saves as `raw.md` in a paper directory under `data/papers/`
4. Registers the paper in the database with status `uploaded`

## CLI Reference

| Command | What it does |
|---------|--------------|
| `drbrain ingest-link <url>` | Ingest a single URL |
| `drbrain ingest-link <url1> <url2>` | Batch ingest |
| `drbrain ingest-link <url> --pdf` | Force PDF extraction |
| `drbrain ingest-link <url> --dry-run` | Preview only |
| `drbrain ingest-link <url> --json` | JSON output |
