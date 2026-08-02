# Reranker Validation

The rerank layer (`src/drbrain/query/rerank.py`) is designed to **degrade
gracefully** — if `sentence-transformers` (and `torch`) are not installed, or
the cross-encoder model cannot be loaded, it falls back to returning hits in
their original retrieval order. The pipeline never crashes.

This document describes two levels of validation:

1. **Code-path validation** (always available, no heavy deps) — covered by
   `tests/test_rerank_validation.py` using a fake cross-encoder.
2. **Real-model validation** (requires installing the dependency) — the
   procedure below lets you verify actual relevance quality with
   `BAAI/bge-reranker-base`.

## 1. Code-path validation (already in CI)

```bash
uv run pytest tests/test_rerank_validation.py -v
```

These tests inject a fake `CrossEncoder` via `sys.modules` and verify:

- `rerank()` reorders hits by the model's predicted scores
- results are truncated to `top_n`, keeping the highest-scoring
- hits with no extractable text are preserved at the tail in input order
- a mid-flight `predict()` failure degrades to input order (no exception)
- the fallback path (`ImportError`) returns input order (see
  `test_crossencoder_fallback_on_missing_deps` in
  `tests/test_fusion_rerank_hybrid.py`)

This proves the rerank **logic** is correct independent of the real model.

## 2. Real-model validation (requires optional deps)

Install the optional dependency:

```bash
uv pip install sentence-transformers torch
```

Then run this one-off script to confirm the real cross-encoder loads and
produces a different ordering than the no-op baseline on a sample query:

```python
# scripts/validate_bge_reranker.py
from drbrain.query.hybrid_retrieval import hybrid_search
from drbrain.storage.database import Database
from pathlib import Path

db = Database("data/drbrain.db")
hits_no_rerank = hybrid_search("turbulent drag reduction", db, Path("data/drbrain.db"),
                               top_k=10, rerank=False)
hits_reranked   = hybrid_search("turbulent drag reduction", db, Path("data/drbrain.db"),
                               top_k=10, rerank=True)

print("without rerank:", [h.paper_id for h in hits_no_rerank])
print("with rerank:   ", [h.paper_id for h in hits_reranked])
# Expect: ordering changes; reranker promotes query-relevant papers.
```

Run it:

```bash
uv run python scripts/validate_bge_reranker.py
```

**Pass criteria:** the reranked ordering differs from the no-rerank baseline,
and visual inspection confirms the top results are more topically relevant to
the query. The cross-encoder model is cached after the first run
(`~/.cache/huggingface/hub/models--BAAI--bge-reranker-base`).

## Configuration

The reranker is invoked via `hybrid_search(..., rerank=True)`. Model selection:

```python
from drbrain.query.rerank import get_reranker
reranker = get_reranker("auto", model_name="BAAI/bge-reranker-base")
```

Any HuggingFace cross-encoder works. Alternatives worth benchmarking:
`BAAI/bge-reranker-large`, `cross-encoder/ms-marco-MiniLM-L-6-v2`.

## Current status (as of this writing)

- `sentence-transformers` / `torch`: **not installed** in the default env.
- `BAAI/bge-reranker-base`: **not cached** locally.
- Code-path tests: **passing** (8 tests in `test_rerank_validation.py` +
  fallback test in `test_fusion_rerank_hybrid.py`).
- Real-model relevance: **not yet measured** — pending the dependency install
  above. This is intentionally documented rather than asserted.
