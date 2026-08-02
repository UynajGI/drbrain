# RRF Parameter Tuning

The fusion layer (`src/drbrain/query/fusion.py`) supports two knobs:

1. **`k`** — the RRF damping constant (default 60). Smaller `k` gives top-ranked
   items a sharper advantage; larger `k` flattens differences.
2. **`weights`** — per-source multipliers (default: all 1.0). Lets you upweight a
   retriever that empirically delivers higher relevance, e.g.
   `{"bm25": 0.4, "embedding": 0.6}`.

This document describes **how to tune them honestly**. A crucial caveat first:

## The hard requirement: labeled relevance judgments

RRF tuning is meaningless without gold answers. To know whether `k=30` beats
`k=60`, you need a labeled set of `(query, paper_id, relevance)` triples —
papers a human judged relevant to each query. **Without this, any "optimal k"
number is fabrication.** As of this writing DrBrain has no such labeled set, so
no specific values are recommended here. The infrastructure below lets you tune
once you collect one.

## Step 1 — collect a labeled evaluation set

For N representative queries (10-20 is a usable start), list the paper_ids a
domain expert considers relevant. Store as:

```json
[
  {"query": "turbulent drag reduction", "relevant": ["p1", "p7", "p12"]},
  {"query": "transformer attention",     "relevant": ["p3", "p8"]}
]
```

## Step 2 — capture retriever outputs

For each query, save the raw BM25 and embedding ranked lists (before fusion).
This is the input to the sweep — re-running it should not hit the LLM or DB.

```python
from drbrain.query.hybrid_retrieval import _run_bm25, _run_embedding
# (these are the per-leg runners; call them once and pickle the output)
```

## Step 3 — sweep parameters and measure recall@k

```python
from drbrain.query.fusion import compare_fusion_params

# ranked_lists = [bm25_output, embed_output] for one query
report = compare_fusion_params(
    [bm25_hits, embed_hits],
    k_values=[10, 30, 60, 100],
    weights_options={
        "equal":        {"bm25": 1.0, "embedding": 1.0},
        "embed_heavy":  {"bm25": 0.3, "embedding": 0.7},
        "bm25_heavy":   {"bm25": 0.7, "embedding": 0.3},
    },
)
# report["k=30"]["kendall_tau"] tells you how much k=30 reshapes the order
# vs the baseline (k=60, equal weight). Low tau = big change → investigate.
```

`compare_fusion_params` reports **ranking perturbation** (kendall_tau vs the
baseline), not relevance. To measure relevance, compute recall@10 / nDCG of
each setting's `top10` against your labeled `relevant` set, then pick the
setting with the best mean metric across queries.

```python
def recall_at_k(fused_top_ids, relevant_ids, k=10):
    hits = set(fused_top_ids[:k]) & set(relevant_ids)
    return len(hits) / max(len(relevant_ids), 1)
```

## Step 4 — apply the winning config

```python
from drbrain.query.hybrid_retrieval import hybrid_search

hits = hybrid_search(
    query, db, db_path, embed_cfg=cfg.embed,
    rrf_k=30,                      # tuned
    rrf_weights={"bm25": 0.3, "embedding": 0.7},  # tuned
)
```

Or via CLI:

```bash
drbrain hybrid "query" --rrf-k 30
# (per-source weights are API-only for now; add a CLI flag if tuning settles)
```

## Sanity bounds (no labels needed)

Even without labels, two invariants must hold — the tests in
`tests/test_rrf_tuning.py` enforce them:

- `weights=None` reproduces canonical equal-weight RRF exactly.
- An unknown source defaults to weight 1.0 (never raises `KeyError`).
- `kendall_tau` is 1.0 for identical orderings, -1.0 for reversed.

## Current status (as of this writing)

- Weighted RRF + sweep tool: **implemented and tested** (7 tests in
  `test_rrf_tuning.py`).
- Labeled evaluation set: **does not exist yet** — collecting one is the
  prerequisite to actually tuning. No `k` or weight values are recommended
  until then; the defaults (k=60, equal) are the well-justified literature
  baseline.
