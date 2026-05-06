# Boost CLI + Query Module Test Coverage

## A. CLI Commands (commands.py)

Add tests for commands with zero or low coverage. Focus on error paths and
non-destructive operations (no real PDFs, no real API calls).

### New tests in `tests/test_cli_commands.py`:

- `test_show_cmd_nonexistent_paper` — `drbrain show` exits cleanly for missing paper
- `test_index_cmd_basic` — `drbrain index` runs without error
- `test_fetch_cmd_doi_parsing` — DOI identifier detection
- `test_fetch_cmd_arxiv_parsing` — arXiv flag detection
- `test_clean_cmd_empty_dirs` (existing) — verify it works
- `test_clean_cmd_force_without_password` — force works when no password set
- `test_repair_paper_nonexistent` — repair handles missing paper
- `test_export_cmd_unsupported_format` — bad format error
- `test_backup_cmd_no_data` — backup with empty data

### New tests for argument parsing:
- `test_identifier_parsing` — DOI vs title vs arXiv ID detection in fetch

## B. BM25 Search (query/bm25.py)

### New tests in `tests/test_bm25.py`:
- `test_build_index_empty` — empty concept list produces empty index
- `test_search_empty_index` — search on empty index returns empty
- `test_search_no_results` — query with no matching terms
- `test_build_index_preserves_metadata` — local_id, type, label in results
- `test_search_limit` — limit parameter respects count

## C. setup.py

### New tests in `tests/test_setup.py`:
- `test_detect_platforms_returns_dict` — _detect_platforms returns expected structure
- `test_injection_map_has_all_platforms` — covers all 7 verified platforms
- `test_injection_map_templates_exist` — all templates referenced exist
- `test_generate_local_config_writes_file` — verify config generation

## Acceptance
- 15+ new tests
- All new tests pass
- ruff clean
