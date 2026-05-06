---
name: translate
description: >
  Translate a paper's markdown to another language via LLM. Use this skill whenever the user asks to
  "translate this paper", "translate to Chinese", "translate to English", "convert this paper to
  Japanese", "machine translate this PDF", or needs a translated version of an ingested paper for
  reading or sharing. Also use when the user mentions language barriers with a paper, wants to read
  a non-English paper in their native language, or has a paper they need in a different language for
  collaboration. Trigger proactively when the user discusses non-English content or expresses
  difficulty reading a paper due to language.
---

# Paper Translation

Translate an ingested paper's `raw.md` to another language using configured LLM models. Uses
placeholder-protected chunking to preserve code blocks, math notation, and image references.
Supports resume from interruption and concurrent chunk translation.

## Prerequisites

The paper must be ingested first (parse phase must succeed). Verify:

```bash
drbrain list                          # confirm paper is in library
drbrain show p3f8a2                   # confirm raw.md exists
```

## Quick Start

```bash
drbrain translate p3f8a2 --lang zh
```

## What It Does

- Reads the paper's `raw.md` and splits it into chunks at natural boundaries
- Protects placeholders (code blocks, math, images, URLs) from translation
- Translates each chunk concurrently via configured LLM models with exponential backoff retry
- Reassembles chunks with placeholders restored to `data/papers/<id>/translated_<lang>.md`
- On interruption, saves progress — re-running resumes from where it left off
- `--force` re-translates even if output already exists

## Common Patterns

**First-time translation:**
```bash
drbrain translate p3f8a2 --lang zh
```

**Resume after interruption (same command):**
```bash
drbrain translate p3f8a2 --lang zh
```

**Re-translate with a different target language:**
```bash
drbrain translate p3f8a2 --lang en --force
```

**Check translation output:**
```bash
ls data/papers/p3f8a2/translated_*.md
```

## Examples

**Translate a Chinese paper to English:**
```bash
drbrain translate p7b1c4 --lang en
```

**Force re-translate an existing translation:**
```bash
drbrain translate p3f8a2 --lang ja --force
```

## CLI Reference

| Command | What it does |
|---------|--------------|
| `drbrain translate <id> --lang zh` | Translate to Chinese |
| `drbrain translate <id> --lang en` | Translate to English |
| `drbrain translate <id> --lang ja` | Translate to Japanese |
| `drbrain translate <id> --force` | Re-translate, overwriting existing output |
| `drbrain translate <id> --json` | JSON output with progress info |
