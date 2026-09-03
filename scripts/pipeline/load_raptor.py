#!/usr/bin/env python
"""RAPTOR 缓存 jsonl → 主库批量入库(tree_summaries + tree_vectors)。

embed_batch.py --raptor-out 只算不写(先缓存后入库),全部跑完后本脚本
executemany 批量写入。幂等:INSERT OR REPLACE(node_id PK),重跑安全。

用法:
    uv run python scripts/pipeline/load_raptor.py \
        --inputs '/tmp/raptor_out_*.jsonl' [--dry-run]
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import sys
import time
from pathlib import Path

ROOT = Path("/home/jiangyuan/drbrain")
sys.path.insert(0, str(ROOT))

from drbrain.storage.connection import connect_wal  # noqa: E402

COMMIT_EVERY = 5000  # 每 N 条 commit 一次


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=str, default="/tmp/raptor_out_*.jsonl")
    ap.add_argument("--db", type=str, default="data/drbrain.db")
    ap.add_argument("--dry-run", action="store_true", help="只统计不写入")
    args = ap.parse_args()

    files = sorted(glob.glob(str(ROOT / args.inputs)))
    if not files:
        print(f"无匹配文件: {args.inputs}")
        return

    n_summary = n_vector = 0
    for mf in files:
        for line in open(mf, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("type") == "summary":
                n_summary += 1
            elif r.get("type") == "vector":
                n_vector += 1
    print(f"缓存文件: {len(files)}, summaries={n_summary}, vectors={n_vector}")
    if args.dry_run:
        return

    t0 = time.monotonic()
    conn = connect_wal(args.db)
    # 提速:关同步写 + 暂删 raptor 相关索引(插入后重建)
    conn.execute("PRAGMA synchronous=OFF")
    index_ddl = [
        r
        for r in conn.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type='index' AND sql IS NOT NULL "
            "AND tbl_name IN ('tree_summaries', 'tree_vectors') AND name LIKE 'idx_%'"
        )
    ]
    for name, _sql in index_ddl:
        conn.execute(f"DROP INDEX {name}")
    conn.commit()

    s_batch: list[tuple] = []
    v_batch: list[tuple] = []
    ins_s = ins_v = 0

    def flush() -> None:
        nonlocal ins_s, ins_v
        if s_batch:
            conn.executemany(
                "INSERT OR REPLACE INTO tree_summaries "
                "(node_id, paper_id, summary_text, source_node_ids, tree_layer) "
                "VALUES (?, ?, ?, ?, ?)",
                s_batch,
            )
            ins_s += len(s_batch)
            s_batch.clear()
        if v_batch:
            conn.executemany(
                "INSERT OR REPLACE INTO tree_vectors "
                "(node_id, paper_id, embedding, content_hash, tree_layer) "
                "VALUES (?, ?, ?, ?, ?)",
                v_batch,
            )
            ins_v += len(v_batch)
            v_batch.clear()

    total = n_summary + n_vector
    i = 0
    for mf in files:
        for line in open(mf, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            i += 1
            if r.get("type") == "summary":
                s_batch.append(
                    (
                        r["node_id"],
                        r["paper_id"],
                        r.get("summary_text", ""),
                        json.dumps(r.get("source_node_ids") or []),
                        r.get("tree_layer", 1),
                    )
                )
            elif r.get("type") == "vector":
                blob = base64.b64decode(r["embedding_blob_b64"])
                v_batch.append(
                    (
                        r["node_id"],
                        r["paper_id"],
                        blob,
                        r.get("content_hash", ""),
                        r.get("tree_layer", "raptor_L1"),
                    )
                )
            if i % COMMIT_EVERY == 0:
                flush()
                conn.commit()
                print(
                    f"[{i}/{total}] summaries={ins_s} vectors={ins_v} "
                    f"elapsed={time.monotonic() - t0:.0f}s",
                    flush=True,
                )

    flush()
    conn.commit()
    # 重建索引
    for name, sql in index_ddl:
        conn.execute(sql)
    conn.commit()
    print(f"\n入库完成: summaries={ins_s} vectors={ins_v} ({time.monotonic() - t0:.0f}s)")
    conn.close()


if __name__ == "__main__":
    main()
