# Layer 2 — Knowledge Genealogy

**Branch**: `dev/layer2-genealogy`

## Goal

Show how knowledge develops — concept lineage trees, paper descendants, domain landscapes.

## ✅ Feature 1: `drbrain evolve <concept>` — DONE

## ✅ Feature 2: `drbrain descendants <paper_id>` — DONE

## Feature 3: `drbrain landscape <workspace>` (TDD) — DONE

## Feature 4: `drbrain paradigm [concept]` (TDD)

Three detection modes:
1. **替代型**: Old method declines 50%+ over 3yr, new method grows fast, `challenges` edge
2. **引爆型**: Concept explodes 0→8+ papers in 2yr, spawns 3+ `extends`/`refines` descendants
3. **跨域入侵**: `applies` edge crosses domain boundary, cascading in new domain

Configurable thresholds in `config.yaml` under `paradigm:` section.
`--workspace` flag scans entire workspace.

领域全景：ASCII 时间线 + 关键论文 + 未解决 gap 持久性 + 争议区。

```
Landscape: nlp-transformers
═══════════════════════════════════════
2017  "Attention Is All You Need"          — origin
2018  ├─ "BERT"                             — extends
      ├─ "GPT"                              — extends
2019  │   ├─ "GPT-2"                         — refines
      └── "DistilBERT"                       — refines

Persistent gaps (>3 years unresolved):
  • "Transformer interpretability" (2018–)
  • "Efficient attention for long sequences" (2019–)

Debates:
  • "Encoder-only vs decoder-only" (2019–) — 4 papers disputing
```

### TDD Approach
1. Write failing tests first in `tests/test_genealogy.py`
2. Implement `landscape_workspace()` in `genealogy.py`
3. Add `landscape_cmd` to `commands.py` + register in `main.py`
4. Tests go green

### Implementation
- Query workspace papers → get concepts grouped by year
- From seed detection: find persistent gaps (gap type >3 years without solution)
- From seed detection: find debate zones
- Render as ASCII timeline with Rich
- Flags: `--top-n 5`

### Acceptance
- `drbrain landscape <workspace>` shows timeline + gaps + debates
- Empty workspace → "no papers found" message
- TDD: tests written first, then implementation
