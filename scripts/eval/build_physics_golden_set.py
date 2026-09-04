#!/usr/bin/env python3
"""Build the physics retrieval golden set from library data (review R-I5).

The built-in golden set (~50 materials/nanotech DOIs) is domain-misaligned, so
physics retrieval quality is currently unmeasurable. This tool auto-constructs
a golden set from in-library citation graphs: for every resolved citation
(``paper_cite_keys.cited_local_id`` non-empty), the citing paper's title plus
TF keywords from its abstract form the query, and the cited paper is the
expected hit:

    (query, expected_paper_id) pairs, each annotated ``source=arxiv-citation``.

Output JSON (``--out``, default ``data/physics_golden_set.json``)::

    {"cases": [{"query", "expected_paper_id", "source", "created_at"}]}

``--limit`` bounds the size, ``--seed`` makes the (shuffled) sample
reproducible. An empty library yields ``{"cases": []}`` with a hint — not an
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
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

DEFAULT_DB = REPO / "data" / "drbrain.db"
DEFAULT_OUT = REPO / "data" / "physics_golden_set.json"
SOURCE = "arxiv-citation"
QUERY_KEYWORDS = 6

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
    SELECT ck.citing_local_id, ck.cited_local_id, p.title, p.abstract
    FROM paper_cite_keys ck
    JOIN papers p ON p.local_id = ck.citing_local_id
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


def build_cases(conn, limit: int = 0, seed: int | None = None, now: str | None = None) -> list[dict]:
    """Construct golden cases from resolved in-library citations.

    ``conn`` is any object with ``execute(sql)`` returning a cursor (a sqlite3
    connection or ``drbrain.storage.database.Database``). Deterministic: rows
    are ordered by (citing, cited); when ``seed`` is given the list is shuffled
    with ``random.Random(seed)`` before ``limit`` is applied.
    """
    created_at = now or datetime.now(timezone.utc).isoformat(timespec="seconds")
    cases: list[dict] = []
    for citing, cited, title, abstract in conn.execute(_CITE_PAIRS_SQL).fetchall():
        title = str(title or "").strip()
        kws = _keywords(str(abstract or title))
        query = re.sub(r"\s+", " ", f"{title} {kws}").strip()
        cases.append(
            {
                "query": query,
                "expected_paper_id": cited,
                "source": SOURCE,
                "created_at": created_at,
            }
        )
    if seed is not None:
        random.Random(seed).shuffle(cases)
    if limit > 0:
        cases = cases[:limit]
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
    args.out.write_text(
        json.dumps({"cases": cases}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[golden] wrote {len(cases)} cases → {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
