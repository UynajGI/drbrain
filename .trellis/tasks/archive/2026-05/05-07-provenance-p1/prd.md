# Extend PageIndex Provenance to evolve/paradigm/transfers/descendants

## Goal

Landscape gap/debate output already carries section provenance. Extend the same pattern to the remaining 4 Layer 2 features. Shared `_get_concept_provenance()` already exists in genealogy.py.

**Pattern**: annotation only (no filtering). Each feature's output gains `section`, `node_id`, `paper_id`, `provenance` fields. Filtering is follow-up.

## Requirements

1. **evolve**: tree nodes show section provenance for each concept
2. **paradigm**: shift detection output includes section context for shifted concepts
3. **transfers**: transfer candidates show source section of the method being transferred
4. **descendants**: child papers show which section of the parent they inherit from

## Acceptance Criteria

- [ ] `evolve` output includes provenance per tree node
- [ ] `paradigm` output includes provenance
- [ ] `transfers` output includes provenance
- [ ] `descendants` output includes provenance
- [ ] All existing genealogy tests pass (36 tests)
- [ ] New provenance tests per feature
- [ ] `uv run ruff check .` clean

## Out of Scope

- Section-based filtering (`--no-related-work` flag)
- New CLI flags
- RAPTOR integration

## Technical Notes

- Shared: `_get_concept_provenance(db, label, ctype)` in genealogy.py
- Key file: `src/drbrain/graph/genealogy.py`
- Tests: `tests/test_genealogy.py`
