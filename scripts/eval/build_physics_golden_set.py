#!/usr/bin/env python3
"""Build the physics retrieval golden set from library data (review R-I5).

The built-in golden set (~50 materials/nanotech DOIs) is domain-misaligned, so
physics retrieval quality is currently unmeasurable. This tool auto-constructs
a golden set from in-library citation graphs: for every resolved citation
(``paper_cite_keys.cited_local_id`` non-empty), the citing paper's title plus
TF keywords from its abstract form the query, and the cited paper is the
expected hit:

    (query, expected_paper_id) pairs, each annotated ``source=arxiv-citation``.

Output is **JSONL in exactly the schema ``drbrain.rag.eval.load_golden``
consumes** (``--out``, default ``data/physics_golden_set.jsonl``)::

    {"query", "relevant_papers": [cited], "relevant_nodes": [],
     "split": "dev|val|test", "reference_answer": <cited abstract>,
     "source", "created_at"}

``reference_answer`` carries the cited paper's abstract — the same cheap
non-LLM ground truth the built-in set uses. Splits are deterministic
60/20/20 after the (optional) shuffle, so dev never leaks into test.
``relevant_nodes`` is left empty: node-level metrics stay neutral and only
paper-level HitRate/MRR are scored (RAGAS-style metrics use
``reference_answer``).

To evaluate against it, point the config at the file and run::

    # config.yaml → llamaindex.eval.golden_set: data/physics_golden_set.jsonl
    drbrain rag eval --split dev

``--limit`` bounds the size, ``--seed`` makes the (shuffled) sample
reproducible. An empty library yields an empty file with a hint — not an
error.

Usage:
  .venv/bin/python scripts/eval/build_physics_golden_set.py
  .venv/bin/python scripts/eval/build_physics_golden_set.py --limit 500 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

DEFAULT_DB = REPO / "data" / "drbrain.db"
DEFAULT_OUT = REPO / "data" / "physics_golden_set.jsonl"
SOURCE = "arxiv-citation"
QUERY_KEYWORDS = 6
# 与 drbrain.rag.eval 的 dev/val/test 约定一致（60/20/20）。
_SPLIT_RATIOS = (("dev", 0.6), ("val", 0.2), ("test", 0.2))

_STOPWORDS = frozenset(
    """a an the and or of in on for to with by from as is are was were be been being
    am do does did doing have has had having
    this that these those it its we our us you your they their he she his her hers
    not no nor but if then than so such can could may might must shall should will would
    at into onto over under between among during after before above below up down out off
    how what when where which who whom whose why whether although because however therefore
    thus moreover furthermore also more most other others some any all both each few many
    much very quite rather only just even still yet ever never always often sometimes
    using used use uses based paper study studies result results show shows shown
    suggest suggests suggested proposed propose new novel first second third
    within without upon about along across toward towards due give given gives
    via per etc ie eg""".split()
)

_WORD_RE = re.compile(r"[a-z][a-z0-9\-]{2,}")

_CITE_PAIRS_SQL = """
    SELECT ck.citing_local_id, ck.cited_local_id,
           p.title, p.abstract AS citing_abstract,
           c.abstract AS cited_abstract
    FROM paper_cite_keys ck
    JOIN papers p ON p.local_id = ck.citing_local_id
    JOIN papers c ON c.local_id = ck.cited_local_id
    WHERE ck.cited_local_id IS NOT NULL
      AND ck.cited_local_id != ck.citing_local_id
    ORDER BY ck.citing_local_id, ck.cited_local_id
"""


def _keywords(text: str, k: int = QUERY_KEYWORDS) -> str:
    """Top-k TF terms (stopword-filtered) as a query keyword string."""
    counts: Counter[str] = Counter(m.group(0) for m in _WORD_RE.finditer(text.lower()))
    for w in _STOPWORDS:
        counts.pop(w, None)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return " ".join(w for w, _ in ranked[:k])


def _split_for(index: int, total: int) -> str:
    """Deterministic 60/20/20 split assignment for one case."""
    for name, ratio in _SPLIT_RATIOS:
        if index < round(total * ratio):
            return name
        index -= round(total * ratio)
    return _SPLIT_RATIOS[-1][0]


def build_cases(
    conn, limit: int = 0, seed: int | None = None, now: str | None = None
) -> list[dict]:
    """Construct golden cases from resolved in-library citations.

    Cases come out in the exact schema ``drbrain.rag.eval.load_golden``
    consumes (JSONL lines), so ``drbrain rag eval`` can score against this
    file without any translation layer. Deterministic: rows are ordered by
    (citing, cited); when ``seed`` is given the list is shuffled with
    ``random.Random(seed)`` before ``limit`` is applied, and the 60/20/20
    dev/val/test split is assigned after the shuffle.
    """
    created_at = now or datetime.now(UTC).isoformat(timespec="seconds")
    cases: list[dict] = []
    for citing, cited, title, citing_abstract, cited_abstract in conn.execute(
        _CITE_PAIRS_SQL
    ).fetchall():
        title = str(title or "").strip()
        reference = str(cited_abstract or "").strip()
        kws = _keywords(str(citing_abstract or title))
        query = re.sub(r"\s+", " ", f"{title} {kws}").strip()
        cases.append(
            {
                "query": query,
                "relevant_papers": [cited],
                "relevant_nodes": [],  # paper-level eval only; node hits stay neutral
                "reference_answer": reference,
                "source": SOURCE,
                "created_at": created_at,
            }
        )
    if seed is not None:
        random.Random(seed).shuffle(cases)
    if limit > 0:
        cases = cases[:limit]
    for i, case in enumerate(cases):
        case["split"] = _split_for(i, len(cases))
    return cases


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB, help="main library DB")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output JSON path")
    ap.add_argument("--limit", type=int, default=0, help="max cases (0 = all)")
    ap.add_argument("--seed", type=int, default=None, help="shuffle seed for a reproducible sample")
    args = ap.parse_args(argv)

    if not args.db.exists():
        print(f"[golden] library DB missing: {args.db} — writing empty golden set", flush=True)
        cases: list[dict] = []
    else:
        conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
        try:
            cases = build_cases(conn, limit=args.limit, seed=args.seed)
        finally:
            conn.close()

    if not cases:
        print(
            "[golden] no resolved in-library citations found (empty library, or run "
            "`kg_lazy_build`/`ingest_arxiv_latex.py --resolve-citations` first) — "
            "writing empty golden set",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case, ensure_ascii=False) + "\n")
    splits = Counter(case["split"] for case in cases)
    breakdown = ", ".join(f"{name}={splits.get(name, 0)}" for name, _ in _SPLIT_RATIOS)
    print(f"[golden] wrote {len(cases)} cases ({breakdown}) → {args.out}", flush=True)
    print(
        "[golden] evaluate with: config llamaindex.eval.golden_set → this file, then "
        "`drbrain rag eval --split dev`",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
