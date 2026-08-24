#!/usr/bin/env python
"""分片 db 合并进主库（INSERT OR REPLACE，幂等可重跑）。

用法:
    uv run python scripts/pipeline/merge_shards.py --shards data/shards/shard*.db \
        --main data/drbrain.db
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

TABLES = ["papers", "paper_ids", "concepts", "edges", "tree_vectors", "tree_summaries"]


def merge_one(shard: Path, main: sqlite3.Connection) -> dict:
    s = sqlite3.connect(str(shard))
    s.row_factory = sqlite3.Row
    counts = {}
    for table in TABLES:
        try:
            cols = [r[1] for r in s.execute(f"PRAGMA table_info({table})").fetchall()]
        except sqlite3.Error:
            continue
        if not cols:
            continue
        rows = s.execute(f"SELECT * FROM {table}").fetchall()
        if not rows:
            counts[table] = 0
            continue
        placeholders = ",".join("?" * len(cols))
        colnames = ",".join(cols)
        main.executemany(
            f"INSERT OR REPLACE INTO {table} ({colnames}) VALUES ({placeholders})",
            [tuple(r[c] for c in cols) for r in rows],
        )
        counts[table] = len(rows)
    s.close()
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--shards", type=str, required=True, help="glob pattern, e.g. data/shards/shard*.db"
    )
    ap.add_argument("--main", type=str, default="data/drbrain.db")
    args = ap.parse_args()

    shards = sorted(Path(".").glob(args.shards))
    if not shards:
        print(f"no shards matched: {args.shards}")
        sys.exit(1)
    main = sqlite3.connect(args.main)
    main.execute("PRAGMA journal_mode=WAL")
    total = {}
    for shard in shards:
        counts = merge_one(shard, main)
        main.commit()
        print(f"{shard.name}: {counts}")
        for k, v in counts.items():
            total[k] = total.get(k, 0) + v
    main.close()
    print(f"\n合并完成: {total}")


if __name__ == "__main__":
    main()
