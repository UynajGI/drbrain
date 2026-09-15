#!/usr/bin/env python
"""分片 db 合并进主库（INSERT OR REPLACE，幂等可重跑）。

用法:
    uv run python scripts/pipeline/merge_shards.py --shards data/shards/shard*.db \
        --main data/drbrain.db

统一存储迁移（2026-09 起）: 除旧 KG/向量表外，分片里的规范正文
(``document_revisions``/``content_blocks``) 与已发布叶节点 (``tree_nodes``
``kind='leaf'``) 一并合并；分片的 region 属于分片本地层次，不进入主库。
FTS 由 ``content_blocks`` 触发器随合并自动同步；ANN 向量存放在文件系统、
区域层次是语料级派生结果，二者都在主库合并完成后由 ``drbrain index build``
统一重建——分片阶段不再需要 ``embed --tree``。
分片 ingest 写入规范正文尚待迁移（当前分片库仍以旧式内容为主，若表不存在
则自动跳过，行为与迁移前一致）。
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

#: 旧式 KG / 文本向量表（保持既有合并语义）。
TABLES = ["papers", "paper_ids", "concepts", "edges", "tree_vectors", "tree_summaries"]

#: 统一存储表：规范正文 + 已发布叶节点。``tree_nodes`` 只取叶子——
#: 分片 region 属于分片本地层次，混入主库会得到一个跨语料的假层次。
UNIFIED_TABLES = [
    ("document_revisions", ""),
    ("content_blocks", ""),
    ("tree_nodes", "WHERE kind = 'leaf'"),
]

#: ``(table, where)`` 合并清单，顺序执行、表不存在自动跳过。
MERGE_TABLES: list[tuple[str, str]] = [(name, "") for name in TABLES] + UNIFIED_TABLES


def merge_one(shard: Path, main: sqlite3.Connection) -> dict:
    s = sqlite3.connect(str(shard))
    s.row_factory = sqlite3.Row
    counts = {}
    for table, where in MERGE_TABLES:
        try:
            cols = [r[1] for r in s.execute(f"PRAGMA table_info({table})").fetchall()]
        except sqlite3.Error:
            continue
        if not cols:
            continue
        rows = s.execute(f"SELECT * FROM {table} {where}").fetchall()
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
