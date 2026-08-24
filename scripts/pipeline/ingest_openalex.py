#!/usr/bin/env python
"""主库 fulltext 表全文 → 分片 db 增强入库（db 驱动版 ingest_scibase）。

按 DOI 哈希分片，从主库 fulltext 表读全文，复用 ingest_from_md 做
identify→tree→paper，写入独立分片 db。跳过 scibase 分片已增强的 DOI。

用法:
    uv run python scripts/pipeline/ingest_openalex.py --shard-id 0 --shard-total 8 \
        --db data/shards/oa_shard0.db --manifest data/shards/oa_shard0.ingest.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path("/home/jiangyuan/drbrain")
sys.path.insert(0, str(ROOT))

from drbrain.dedup.resolver import DedupEngine  # noqa: E402
from drbrain.storage.database import Database  # noqa: E402
from scripts.pipeline.common import load_cfg  # noqa: E402
from scripts.pipeline.ingest_scibase import _identify, _worker, _write_db  # noqa: E402

MAIN_DB = ROOT / "data/drbrain.db"


def _shard_of(doi: str, total: int) -> int:
    return int(hashlib.md5(doi.encode()).hexdigest(), 16) % total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard-id", type=int, required=True)
    ap.add_argument("--shard-total", type=int, default=8)
    ap.add_argument("--db", type=str, required=True)
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--no-db",
        action="store_true",
        help="只缓存文件（raw.md + tree.json），不写 db；local_id 用 DOI 哈希",
    )
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    manifest = Path(args.manifest)

    # 从主库 fulltext 表读全文
    main = sqlite3.connect(str(MAIN_DB))
    rows = main.execute(
        "SELECT doi, text, meta FROM fulltext WHERE text IS NOT NULL AND length(text) > 500"
    ).fetchall()
    main.close()
    print(f"fulltext rows: {len(rows)}")

    # 跳过 scibase 分片已增强的 DOI（从 scibase ingest manifest 收集）
    enhanced_dois: set[str] = set()
    for mf in sorted((ROOT / "data/shards").glob("shard*.ingest.jsonl")):
        for line in mf.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                if rec.get("ok") and rec.get("local_id"):
                    fname = rec.get("file", "")
                    if fname.endswith(".json"):
                        enhanced_dois.add(fname[:-5].replace("_", "/"))
            except json.JSONDecodeError:
                continue
    print(f"scibase 已增强 DOI: {len(enhanced_dois)}")

    done: set[str] = set()
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                if rec.get("ok"):
                    # 兼容旧 manifest：doi 或 file（DOI 文件名）字段
                    doi = rec.get("doi") or ""
                    if not doi and rec.get("file", "").endswith(".json"):
                        doi = rec["file"][:-5].replace("_", "/")
                    if doi:
                        done.add(doi)
            except json.JSONDecodeError:
                continue
        print(f"[resume] 已跳过 {len(done)} 篇")

    concurrency = int(os.environ.get("INGEST_CONCURRENCY", "1"))
    stats = {"ok": 0, "fail": 0}
    bad: list[dict] = []
    t0 = time.monotonic()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_f = open(manifest, "a", encoding="utf-8")
    db = Database(args.db)
    dedup = DedupEngine(db)
    processed = 0
    try:
        # 主进程 identify（写 papers 表）→ worker 抽 LLM（不碰 db）→ 主进程写 db
        # --no-db 模式：local_id 用 DOI 哈希（确定性去重），完全不写 db，只缓存文件
        pending: list[tuple] = []
        for doi, text, meta in rows:
            if _shard_of(doi, args.shard_total) != args.shard_id:
                continue
            if doi in done or doi in enhanced_dois:
                continue
            if args.limit and processed >= args.limit:
                break
            processed += 1
            try:
                meta_d = json.loads(meta) if meta else {}
            except (json.JSONDecodeError, TypeError):
                meta_d = {}
            cleaned = {
                "markdown": text,
                "doi": doi,
                "title": meta_d.get("title") or "",
                "year": meta_d.get("year"),
                "_file": doi.replace("/", "_") + ".json",
            }
            try:
                if args.no_db:
                    import hashlib as _hl

                    local_id = f"p{_hl.md5(doi.strip().lower().encode()).hexdigest()[:8]}"
                else:
                    local_id = _identify(cleaned, db, dedup)
            except Exception as e:  # noqa: BLE001
                rec = {
                    "doi": doi,
                    "ok": False,
                    "local_id": None,
                    "error": f"identify: {type(e).__name__}: {e}",
                }
                bad.append(rec)
                manifest_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            pending.append((cleaned, cfg, local_id))

        if concurrency <= 1:
            for i, task in enumerate(pending, 1):
                rec = _worker(task)
                if not args.no_db:
                    _write_db(rec, db)
                stats["ok" if rec["ok"] else "fail"] += 1
                manifest_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                manifest_f.flush()
                if not rec["ok"]:
                    bad.append(rec)
                if i % 10 == 0:
                    print(
                        f"[{i}] ok={stats['ok']} fail={stats['fail']} "
                        f"elapsed={time.monotonic() - t0:.0f}s",
                        flush=True,
                    )
        else:
            import concurrent.futures

            with concurrent.futures.ProcessPoolExecutor(max_workers=concurrency) as ex:
                for i, rec in enumerate(ex.map(_worker, pending), 1):
                    if not args.no_db:
                        _write_db(rec, db)
                    stats["ok" if rec["ok"] else "fail"] += 1
                    manifest_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    manifest_f.flush()
                    if not rec["ok"]:
                        bad.append(rec)
                    if i % 10 == 0:
                        print(
                            f"[{i}] ok={stats['ok']} fail={stats['fail']} "
                            f"elapsed={time.monotonic() - t0:.0f}s",
                            flush=True,
                        )
    finally:
        manifest_f.close()
        db.close()

    print(f"\n完成: ok={stats['ok']} fail={stats['fail']} ({time.monotonic() - t0:.0f}s)")
    for r in bad[:10]:
        print(f"  FAIL {r['doi']}: {str(r.get('error'))[:120]}")


if __name__ == "__main__":
    main()
