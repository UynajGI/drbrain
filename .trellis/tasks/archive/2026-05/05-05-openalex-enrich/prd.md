# Fix OpenAlex Enrichment in Repair

## Problem

`_enrich_via_openalex` in `services/repair.py` has dead code paths and missing functionality:

1. **DOI path broken**: `get_work_by_doi()` returns `{doi, title, year, openalex_id, referenced_works}` — no `abstract` or `cited_by_count`. The code checks `data.get("abstract")` / `data.get("cited_by_count")` but they're never present. Entire DOI enrichment path is no-op.

2. **Title path partial**: `_fetch_openalex_metadata()` returns `(title, year, openalex_id, journal, cited_by_count)` — has cited_by_count but NO abstract. `data["abstract"]` always missing.

3. **Authors never fetched**: `search_authors_by_work()` exists in `extractor/openalex.py` and works — but `_enrich_via_openalex` never calls it.

4. **repair_paper apply section** doesn't handle `authors` field — only title/year/doi/journal/abstract/citation_count.

## Fix

### 1. Fix `_enrich_via_openalex` to actually call the right APIs

- Use `search_authors_by_work(doi=doi, title=title)` to get authors → format as "Given Family and Given Family"
- For abstract/cited_count: create a proper OpenAlex work fetch with the right `select` fields (add `abstract_inverted_index`, `cited_by_count`) in the repair module, or add a helper to openalex.py

### 2. Add authors apply in `repair_paper`

- Check `paper_ids.authors` column exists; if not, add via migration
- Apply `authors` repairs to DB

### 3. Add volume/pages enrichment from OpenAlex

- OpenAlex returns `biblio.volume` and `biblio.first_page`/`biblio.last_page`
- Fetch and apply if paper is missing them

## Files

- `src/drbrain/services/repair.py` — main changes
- `src/drbrain/extractor/openalex.py` — may need a new helper for enriched work fetch
- `tests/test_repair.py` — add tests for new enrichment paths

## Acceptance

- `_enrich_via_openalex` returns authors when available
- `_enrich_via_openalex` returns abstract when available (both DOI and title paths)
- `repair_paper` applies authors to DB
- Existing repair tests pass
