# Boost Query + Report Module Test Coverage

## A. query/bm25.py (3 targets)

- `tokenize()` — edge cases: empty, None, mixed case, punctuation
- `BM25Search` class — `search()`, `__init__` params (k1, b), empty corpus
- `build_bm25_index()` — with real DB data, filters by status='extracted'

## B. query/tree_retrieval.py (5 targets)

- `_get_node_title()` — missing node, empty structure
- `_collect_all_leaf_ids()` — nested tree, single node
- `_build_remaining_structure()` — partial reads
- `_build_top_level_structure()` — depth limiting
- `_expand_branch()` — branch selection

## C. report/analyzer.py (2 targets)

- `analyze_paper()` — empty concepts, full=False, full=True, nonexistent paper
- `add_cross_paper_insights()` — single paper, multi-paper, empty reports

## D. report/generator.py (3 targets)

- `RefEntry` — construction
- `PaperReport` — construction, to_dict
- `total_refs_and_citations()` — various counts

## Acceptance
- 15+ new tests
- All pass
- ruff clean
