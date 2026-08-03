"""Batch ingest materials-science corpus from Sciverse, newest-first.

Drives ``ingest_corpus`` over strongly-correlated materials sub-domain
queries, iterating years from newest to oldest, until the target total
library size is reached. Idempotent (unique_id/DOI dedup), safe to re-run.

Usage:
    uv run python scripts/cg_ingest_materials.py [--target 22000]
    uv run python scripts/cg_ingest_materials.py --broad --target 220000
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time

from drbrain.cli._common import open_db
from drbrain.concept_graph.ingest import ingest_corpus
from drbrain.concept_graph.sources.sciverse import SciverseSource
from drbrain.config import load_config

# Strongly-correlated materials sub-domain queries, in priority order.
QUERIES = [
    "battery materials",
    "alloy",
    "perovskite",
    "polymer materials",
    "ceramic materials",
    "catalyst materials",
    "nanomaterials",
    "semiconductor materials",
    "composite materials",
    "thin film materials",
]

# Broad materials-science sweep (full corpus mining).
BROAD_QUERIES = [
    "materials",
    "materials science",
    "materials engineering",
    "advanced materials",
    "functional materials",
    "structural materials",
    "surface coating",
    "corrosion",
    "metallurgy",
    "crystal growth",
    "biomaterials",
    "optical materials",
    "magnetic materials",
    "energy storage",
    "photovoltaic",
    "hydrogen storage",
    "additive manufacturing",
    "wear resistance",
    "mechanical properties",
    "microstructure",
]

YEAR_NEWEST = 2026
YEAR_OLDEST = 2000
PER_YEAR_LIMIT = 300


def _library_size(db_path: str) -> int:
    with sqlite3.connect(db_path) as conn:
        return conn.execute("SELECT count(*) FROM papers").fetchone()[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=22000, help="Stop at this total paper count")
    ap.add_argument("--per-year", type=int, default=PER_YEAR_LIMIT)
    ap.add_argument("--broad", action="store_true", help="Use broad materials-science queries")
    args = ap.parse_args()

    queries = BROAD_QUERIES if args.broad else QUERIES

    cfg = load_config()
    src = SciverseSource(
        cfg.api.sciverse_token, cfg.api.sciverse_base_url, rate_limit=cfg.api.sciverse_rate_limit
    )

    start = _library_size(cfg.db.path)
    print(f"[batch] library start={start} target={args.target}", flush=True)

    total_inserted = 0
    with open_db(cfg) as db:
        for query in queries:
            if _library_size(cfg.db.path) >= args.target:
                break
            for year in range(YEAR_NEWEST, YEAR_OLDEST - 1, -1):
                if _library_size(cfg.db.path) >= args.target:
                    break
                t0 = time.time()
                try:
                    stats = ingest_corpus(
                        db,
                        src,
                        query,
                        year_from=year,
                        year_to=year,
                        limit=args.per_year,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[batch] {query!r} {year}: ERROR {exc}", flush=True)
                    continue
                total_inserted += stats.inserted
                if stats.fetched:
                    print(
                        f"[batch] {query!r} {year}: fetched={stats.fetched} "
                        f"inserted={stats.inserted} ({time.time() - t0:.0f}s)",
                        flush=True,
                    )

    end = _library_size(cfg.db.path)
    print(f"[batch] done: inserted={total_inserted} library={end} (start={start})", flush=True)


if __name__ == "__main__":
    sys.exit(main())
