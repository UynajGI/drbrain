#!/usr/bin/env python
"""统一入库：把 --no-db 缓存的 tree.json + manifest 批量写入主库。

读 8 个 oa ingest manifest 的 ok 记录（local_id 是 DOI 哈希），
校验 tree.json/raw.md 存在后写主库 papers/paper_ids 表。
幂等：INSERT OR REPLACE，重跑安全。

用法:
    uv run python scripts/pipeline/load_ingest_cache.py [--batch-size 500]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path("/home/jiangyuan/drbrain")
sys.path.insert(0, str(ROOT))

from drbrain.storage.database import Database  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifests", type=str, default="data/shards/oa_shard*.ingest.jsonl")
    ap.add_argument("--db", type=str, default="data/drbrain.db")
    ap.add_argument("--papers-dir", type=str, default=None)
    args = ap.parse_args()

    cfg_papers_dir = ROOT / "data/papers"
    papers_dir = Path(args.papers_dir) if args.papers_dir else cfg_papers_dir

    # 收集全部 manifest 记录（DOI → local_id/title/year）
    records: dict[str, dict] = {}
    import glob

    for mf in sorted(glob.glob(str(ROOT / args.manifests))):
        for line in open(mf, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not r.get("ok") or not r.get("local_id"):
                continue
            doi = (r.get("doi") or "").strip().lower()
            if not doi and str(r.get("file", "")).endswith(".json"):
                doi = str(r["file"])[:-5].replace("_", "/").lower()
            if doi:
                records[doi] = {"local_id": r["local_id"], "doi": doi}
    print(f"manifest 待入库记录: {len(records)}")

    db = Database(args.db)
    t0 = time.monotonic()
    inserted = skipped_no_tree = skipped_exists = 0
    batch: list[tuple] = []

    def flush() -> None:
        nonlocal inserted
        if not batch:
            return
        for local_id, title, year, doi in batch:
            # papers
            db.conn.execute(
                "INSERT OR IGNORE INTO papers (local_id, title, year, status) VALUES (?, ?, ?, 'uploaded')",
                (local_id, title, year),
            )
            # paper_ids
            db.conn.execute(
                "INSERT OR REPLACE INTO paper_ids (local_id, doi, arxiv, s2_id, openalex_id) VALUES (?, ?, NULL, NULL, NULL)",
                (local_id, doi),
            )
        db.commit()
        inserted += len(batch)
        batch.clear()

    for i, (doi, rec) in enumerate(records.items(), 1):
        lid = rec["local_id"]
        tree_path = papers_dir / lid / "tree.json"
        md_path = papers_dir / lid / "raw.md"
        if not tree_path.exists() or not md_path.exists():
            skipped_no_tree += 1
            continue
        # 已存在则跳过（幂等）
        row = db.conn.execute("SELECT 1 FROM paper_ids WHERE doi = ?", (doi,)).fetchone()
        if row:
            skipped_exists += 1
            continue
        # title/year 从 raw.md 首行或 meta 拿不到（--no-db 模式没存），用 DOI 占位
        batch.append((lid, doi, None, doi))
        if len(batch) >= 500:
            flush()
        if i % 5000 == 0:
            print(
                f"[{i}/{len(records)}] 入库={inserted} 已存在={skipped_exists} "
                f"缺文件={skipped_no_tree} elapsed={time.monotonic() - t0:.0f}s",
                flush=True,
            )
    flush()
    print(
        f"\n完成: 入库={inserted} 已存在跳过={skipped_exists} 缺文件={skipped_no_tree} "
        f"({time.monotonic() - t0:.0f}s)"
    )
    db.close()


if __name__ == "__main__":
    main()
