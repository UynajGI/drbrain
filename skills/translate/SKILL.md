---
name: translate
description: >
  Translate a paper's markdown via LLM. Use this skill whenever the user asks to "translate this paper",
  "translate to Chinese", "translate to English", "convert this paper to Japanese", or needs a
  machine-translated version of an ingested paper.
---

# Paper Translation

Translate an ingested paper's `raw.md` to another language using configured LLM models. Uses
placeholder-protected chunking to preserve code blocks, math notation, and image references.
Supports resume from interruption and concurrent chunk translation.

## Prerequisites

The paper must be ingested first (`drbrain ingest`). Translation works on the `raw.md` file, so
the paper needs a successful parse phase.

```bash
drbrain list                    # verify paper is in library
drbrain show <id>               # confirm raw.md exists
```

## Quick Start

```bash
drbrain translate p3f8a2 --lang zh
```

## What It Does

- Reads the paper's `raw.md` and splits it into chunks at natural boundaries
- Protects placeholders (code blocks, math, images, URLs) from translation
- Translates each chunk concurrently via configured LLM models with exponential backoff retry
- Reassembles chunks with placeholders restored, writing output to `data/papers/<id>/translated_<lang>.md`
- On interruption, saves progress — re-running resumes from where it left off
- `--force` re-translates even if output already exists

## CLI Reference

| Command | What it does |
|---------|--------------|
| `drbrain translate <id> --lang zh` | Translate to Chinese (default) |
| `drbrain translate <id> --lang en` | Translate to English |
| `drbrain translate <id> --lang ja` | Translate to Japanese |
| `drbrain translate <id> --force` | Re-translate, overwriting existing output |
| `drbrain translate <id> --json` | JSON output with progress info |

## Common Patterns

**First-time translation:**
```bash
drbrain translate p3f8a2 --lang zh
```

**Resume after interruption:**
```bash
# Just re-run the same command — it picks up where it left off
drbrain translate p3f8a2 --lang zh
```

**Re-translate with better model or target language:**
```bash
drbrain translate p3f8a2 --lang en --force
```

**Check translation output:**
```bash
ls data/papers/p3f8a2/translated_*.md
```
