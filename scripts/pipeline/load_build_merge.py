#!/usr/bin/env python
"""build jsonl（jsonl-out 分片产物）→ 主库批量入库（concepts/edges + status=extracted）。

读全部 build 分片 manifest 的 ok 记录，executemany 批量写主库。
幂等：edges ON CONFLICT 更新 updated_at；concepts 重跑会重复插入
（与 build_cmd 行为一致，靠下游 dedup_concepts_by_label 收敛），
已 extracted 的篇跳过避免重复。

用法:
    uv run python scripts/pipeline/load_build_merge.py \
        --manifests 'data/spool/oa_build_s*.jsonl' [--dry-run]
"""

from __future__ import annotations

import argparse
import glob
import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path("/home/jiangyuan/drbrain")
sys.path.insert(0, str(ROOT))

from drbrain.storage.database import Database  # noqa: E402

VALID_TYPES = {"Problem", "Method", "Conclusion", "Debate", "Gap", "Actor"}
COMMIT_EVERY = 500  # 每 N 篇 commit 一次


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifests", type=str, default="data/spool/oa_build_s*.jsonl")
    ap.add_argument("--db", type=str, default="data/drbrain.db")
    ap.add_argument("--dry-run", action="store_true", help="只统计不写入")
    args = ap.parse_args()

    # 读全部分片 ok 记录
    records: dict[str, dict] = {}
    files = sorted(glob.glob(str(ROOT / args.manifests)))
    for mf in files:
        for line in open(mf, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("ok") and r.get("local_id"):
                records[r["local_id"]] = r
    total_concepts = sum(len(r.get("concepts", [])) for r in records.values())
    total_relations = sum(len(r.get("relations", [])) for r in records.values())
    print(f"分片文件: {len(files)}, ok 记录: {len(records)}")
    print(f"待入库 concepts≈{total_concepts} relations≈{total_relations}")

    if args.dry_run:
        return

    db = Database(args.db)
    conn: sqlite3.Connection = db.conn
    t0 = time.monotonic()
    done_papers = 0
    ins_c = ins_e = skip_c = skip_e = 0
    c_batch: list[tuple] = []
    e_batch: list[tuple] = []

    def flush() -> None:
        nonlocal ins_c, ins_e
        if c_batch:
            conn.executemany(
                "INSERT INTO concepts (local_id, type, label, confidence, section, node_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                c_batch,
            )
            ins_c += len(c_batch)
            c_batch.clear()
        if e_batch:
            conn.executemany(
                "INSERT INTO edges (src_id, dst_id, relation, source_paper, weight, node_id, "
                "section, updated_at) VALUES (?, ?, ?, ?, 1.0, ?, ?, CURRENT_TIMESTAMP) "
                "ON CONFLICT(src_id, dst_id, relation, source_paper) "
                "DO UPDATE SET updated_at = CURRENT_TIMESTAMP",
                e_batch,
            )
            ins_e += len(e_batch)
            e_batch.clear()

    for i, (lid, rec) in enumerate(records.items(), 1):
        # 已抽取的篇跳过（幂等重跑）
        row = conn.execute("SELECT status FROM papers WHERE local_id = ?", (lid,)).fetchone()
        if row and row[0] == "extracted":
            done_papers += 1
            continue

        for c in rec.get("concepts", []):
            ctype = c.get("type", "")
            label = c.get("label", "")
            if ctype not in VALID_TYPES or not label:
                skip_c += 1
                continue
            try:
                conf = float(c.get("confidence", 0.5))
            except (TypeError, ValueError):
                conf = 0.5
            c_batch.append((lid, ctype, label, conf, c.get("section", ""), c.get("node_id", "")))

        for rel in rec.get("relations", []):
            head = rel.get("head", "")
            tail = rel.get("tail", "")
            r = rel.get("rel", "")
            if not head or not tail or not r:
                skip_e += 1
                continue
            e_batch.append((head, tail, r, lid, rel.get("node_id", ""), rel.get("section", "")))

        conn.execute(
            "UPDATE papers SET status='extracted', updated_at=CURRENT_TIMESTAMP WHERE local_id=?",
            (lid,),
        )
        done_papers += 1

        if i % COMMIT_EVERY == 0:
            flush()
            conn.commit()
            print(
                f"[{i}/{len(records)}] papers={done_papers} concepts={ins_c}(skip {skip_c}) "
                f"edges={ins_e}(skip {skip_e}) elapsed={time.monotonic() - t0:.0f}s",
                flush=True,
            )

    flush()
    conn.commit()
    print(
        f"\n入库完成: papers={done_papers} concepts={ins_c}(无效 {skip_c}) "
        f"edges={ins_e}(无效 {skip_e}) ({time.monotonic() - t0:.0f}s)"
    )
    db.close()


if __name__ == "__main__":
    main()
