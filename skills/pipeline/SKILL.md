---
name: pipeline
description: >
  Chain multiple processing steps in sequence. Use when the user wants to run the full
  end-to-end pipeline (ingest → build → embed → closure) or a subset of steps, or needs
  to automate DrBrain workflows. Trigger on "run full pipeline", "process everything",
  "batch process", "chain steps", "ingest and build".
---

# Pipeline

Run multiple DrBrain steps in sequence via presets or custom step lists.

## Quick Start

```bash
drbrain pipeline --preset full       # ingest → build → embed → closure
drbrain pipeline --preset quick      # build → embed → closure
drbrain pipeline --preset embed      # embed → closure
drbrain pipeline --steps build,embed # custom steps
```

## Available Steps

```bash
drbrain pipeline --list
```

## Dry Run

```bash
drbrain pipeline --preset full --dry-run
```

## Presets

| Preset | Steps |
|--------|-------|
| `full` | ingest, build, embed, closure |
| `quick` | build, embed, closure |
| `embed` | embed, closure |

## CLI Reference

| Command | What it does |
|---------|--------------|
| `drbrain pipeline --preset <name>` | Run a preset |
| `drbrain pipeline --steps <names>` | Run custom steps |
| `drbrain pipeline --list` | List steps and presets |
| `drbrain pipeline --preset full --dry-run` | Preview without executing |
