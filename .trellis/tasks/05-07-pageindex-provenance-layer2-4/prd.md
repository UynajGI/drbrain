# Inject PageIndex Provenance into Landscape Gap/Debate Map

## Goal

`landscape` command already detects gaps and debates via `detect_research_seeds()`. But output is flat — you see "GNN scalability is an unaddressed gap" without knowing which paper, which section, or what context.

Inject PageIndex provenance (`node_id` → `section` → `paper_id`) into landscape's gap and debate output. No new extraction, no classification — just surface what's already in the DB.

## Requirements

1. **Shared query helper**: `_get_concept_provenance(db, label, ctype)` → `(section, node_id, paper_id)`. Queries `concepts` table for highest-confidence match.
2. **landscape gap output**: each gap shows `[source: <section> of <paper_id>]`
3. **landscape debate output**: each debate shows source sections of both supporting and challenging concepts
4. **Fallback**: `node_id` empty → `[source: unknown]`; `section` empty but `node_id` present → resolve from tree.json
5. **Existing tests unbroken**: additive change, no signature changes to public functions

## MVP Scope (this task)

```
landscape --gaps   →  each gap annotated with section provenance
landscape --debates →  each debate annotated with bilateral section provenance
+ shared _get_concept_provenance() in genealogy.py
```

### Follow-up (explicitly NOT this task)

- `evolve` — filter by section type
- `paradigm` — ontology structure diff
- `transfers` — section-type hints
- `descendants` — inheritance source annotation

## Decision (ADR-lite)

**Context**: Section-type classification has ambiguous edge cases; LLM classifier is premature.

**Decision**: Pass section title as-is. Format: `"label [source: <section> of <paper_id>]"`. No heuristic or LLM classification.

**Consequences**: Light implementation, no misclassification. Programmatic consumers see raw section titles. Classification layer addable later.

## Acceptance Criteria

- [ ] `_get_concept_provenance(db, label, ctype)` returns `(section, node_id, paper_id)` for existing concepts
- [ ] `_get_concept_provenance()` returns `("", "", "")` for non-existent concept
- [ ] `landscape` output includes section provenance on gaps
- [ ] `landscape` output includes section provenance on debates
- [ ] Gap/debate with empty `node_id` shows `[source: unknown]`
- [ ] `uv run pytest tests/test_genealogy.py` — 27 existing tests pass
- [ ] New tests for `_get_concept_provenance`
- [ ] `uv run ruff check .` clean

## Definition of Done

- Tests pass: `uv run pytest -m "not integration"`
- Lint clean: `uv run ruff check .`
- No new deps
- No signature changes to public Layer 2 functions

## Out of Scope

- RAPTOR integration in reasoning
- New LLM extraction
- New CLI commands / flags
- evolve/paradigm/transfers/descendants provenance
- Section classification (heuristic or LLM)

## Technical Notes

- Key files: `src/drbrain/graph/genealogy.py` (query helper + landscape), `src/drbrain/graph/engine.py` (seed detection), `src/drbrain/cli/commands.py` (landscape_cmd)
- DB: `concepts` table — `node_id`, `section`, `local_id` columns
- Tests: `tests/test_genealogy.py`
- Spec: `.trellis/spec/backend/pipeline-architecture.md`
