#!/usr/bin/env python
"""批量 build：对已 ingest 的论文跑 5-stage 因果推理抽取。

jsonl-out 模式：只抽 LLM 结果写 jsonl（不写 db，并发无锁），入库单独跑
（load_build.py）。并发：BUILD_CONCURRENCY 个线程 × 每篇内部叶子并发。

用法:
    uv run python scripts/pipeline/build.py --from-manifest data/shards/shard0.ingest.jsonl \
        --db data/shards/shard0.db --manifest data/shards/shard0.build.jsonl \
        --jsonl-out data/shards/shard0.build.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from collections import Counter
from pathlib import Path

from drbrain.storage.database import Database

ROOT = Path("/home/jiangyuan/drbrain")
sys.path.insert(0, str(ROOT))

from scripts.pipeline.common import load_cfg  # noqa: E402

VALID_TYPES = {"Problem", "Method", "Conclusion", "Debate", "Gap", "Actor"}


def build_one(
    local_id: str, cfg: dict, db: Database, skip_refine: bool, jsonl_only: bool = False
) -> dict:
    """复刻 build_cmd 核心：tree → 5-stage 抽取 → 插入 concepts/edges。

    jsonl_only=True 时只返回抽取结果不写 db（并发 build 无锁，入库单独跑）。
    """
    import json as _json

    from loguru import logger

    from drbrain.extractor.cache import ApiCache
    from drbrain.extractor.concept import build_graph_from_tree

    papers_dir = Path(cfg.get("dirs", {}).get("papers", "data/papers"))
    tree_path = papers_dir / local_id / "tree.json"
    md_path = papers_dir / local_id / "raw.md"
    if not tree_path.exists() or not md_path.exists():
        return {"ok": False, "local_id": local_id, "error": "tree/raw.md 缺失"}
    tree = _json.loads(tree_path.read_text(encoding="utf-8"))
    structure = tree.get("structure", [])
    if not structure:
        return {"ok": False, "local_id": local_id, "error": "empty tree"}

    llm_models = cfg.get("llm", {}).get("models", [])
    cache = ApiCache(str(ROOT / "data/spool/llm_cache"))
    t0 = time.monotonic()
    try:
        result = asyncio.run(
            build_graph_from_tree(
                md_path, structure, llm_models, skip_refine=skip_refine, cache=cache
            )
        )
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "local_id": local_id, "error": f"{type(e).__name__}: {e}"}

    concepts = result.get("concepts", [])
    relations = result.get("relations", [])
    merges = result.get("merges", [])
    corrections = result.get("corrections", [])

    if jsonl_only:
        return {
            "ok": True,
            "local_id": local_id,
            "concepts": concepts,
            "relations": relations,
            "merges": merges,
            "corrections": corrections,
            "report": {
                "concepts": len(concepts),
                "relations": len(relations),
                "merges": len(merges),
                "corrections": len(corrections),
                "secs": time.monotonic() - t0,
            },
        }

    valid_count = 0
    rejected = 0
    for c in concepts:
        ctype = c.get("type", "")
        label = c.get("label", "")
        conf = c.get("confidence", 0.5)
        if ctype not in VALID_TYPES or not label:
            rejected += 1
            continue
        db.insert_concept(
            local_id, ctype, label, conf, section=c.get("section", ""), node_id=c.get("node_id", "")
        )
        valid_count += 1

    for r in relations:
        head, rel, tail = r.get("head", ""), r.get("rel", ""), r.get("tail", "")
        if head and rel and tail:
            try:
                db.insert_edge(
                    head,
                    tail,
                    rel,
                    local_id,
                    node_id=r.get("node_id", ""),
                    section=r.get("section", ""),
                )
            except Exception:  # noqa: BLE001
                pass

    db.set_paper_status(local_id, "extracted")
    db.commit()
    logger.info(
        f"[build] {local_id} done in {time.monotonic() - t0:.0f}s — "
        f"concepts={valid_count} relations={len(relations)} merges={len(merges)} "
        f"corrections={len(corrections)} rejected={rejected}"
    )
    return {
        "ok": True,
        "local_id": local_id,
        "report": {
            "concepts": valid_count,
            "relations": len(relations),
            "merges": len(merges),
            "corrections": len(corrections),
            "secs": time.monotonic() - t0,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--from-manifest",
        type=str,
        required=True,
        help="ingest manifest（成功入库的 local_id 列表）",
    )
    ap.add_argument("--db", type=str, required=True)
    ap.add_argument(
        "--manifest", type=str, required=True, help="build manifest 输出路径（断点续传）"
    )
    ap.add_argument(
        "--jsonl-out", type=str, default=None, help="只抽 LLM 结果写 jsonl（不写 db），入库单独跑"
    )
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-refine", action="store_true", default=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    db_path = args.db
    manifest_path = Path(args.manifest)

    # 从 ingest manifest 收集成功入库的 local_id
    ids: set[str] = set()
    src = Path(args.from_manifest)
    if src.exists():
        for line in src.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                if rec.get("ok") and rec.get("local_id"):
                    ids.add(rec["local_id"])
            except json.JSONDecodeError:
                continue
    papers = [{"local_id": lid} for lid in sorted(ids)]
    print(f"[build] from-manifest: {len(papers)} 篇")

    done: set[str] = set()
    if args.resume and manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                if rec.get("ok"):
                    done.add(rec["local_id"])
            except json.JSONDecodeError:
                continue
        print(f"[resume] 已跳过 {len(done)} 篇")

    stats = Counter()
    bad: list[dict] = []
    t0 = time.monotonic()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_f = open(manifest_path, "a", encoding="utf-8")
    manifest_lock = threading.Lock()
    try:
        import concurrent.futures

        concurrency = int(os.environ.get("BUILD_CONCURRENCY", "4"))

        def _build_one(lid: str) -> dict:
            # jsonl-only 模式不碰 db（避免 20 线程 schema 写锁与 embed 冲突），入库单独跑
            tdb = Database(db_path) if not args.jsonl_out else None
            try:
                r = build_one(
                    lid, cfg, tdb, skip_refine=args.skip_refine, jsonl_only=bool(args.jsonl_out)
                )
                return {
                    "local_id": lid,
                    "ok": r["ok"],
                    "error": r.get("error"),
                    "concepts": r.get("concepts", []),
                    "relations": r.get("relations", []),
                    "report": r.get("report"),
                }
            except Exception as e:  # noqa: BLE001
                if tdb:
                    tdb.rollback()
                return {"local_id": lid, "ok": False, "error": f"{type(e).__name__}: {e}"}
            finally:
                if tdb:
                    tdb.close()

        pending = [p["local_id"] for p in papers if p["local_id"] not in done]
        # 单篇超时保护：超长文档会让全部模型 exhausted 且不结束，若不加超时
        # 整片 worker 会卡死（shard3 曾因此停摆 14h）。
        per_paper_timeout = int(os.environ.get("BUILD_PAPER_TIMEOUT", "900"))
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {ex.submit(_build_one, lid): lid for lid in pending}
            for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
                lid = futures[fut]
                try:
                    rec = fut.result(timeout=per_paper_timeout)
                except concurrent.futures.TimeoutError:
                    rec = {
                        "local_id": lid,
                        "ok": False,
                        "error": f"timeout>{per_paper_timeout}s",
                    }
                except Exception as e:  # noqa: BLE001
                    rec = {
                        "local_id": lid,
                        "ok": False,
                        "error": f"{type(e).__name__}: {e}",
                    }
                stats["ok" if rec["ok"] else "fail"] += 1
                with manifest_lock:
                    manifest_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    manifest_f.flush()
                if not rec["ok"]:
                    bad.append(rec)
                if i % 5 == 0 or i == len(pending):
                    print(
                        f"[{i}/{len(pending)}] ok={stats['ok']} fail={stats['fail']} "
                        f"elapsed={time.monotonic() - t0:.0f}s",
                        flush=True,
                    )
    finally:
        manifest_f.close()

    print(f"\n完成: ok={stats['ok']} fail={stats['fail']} ({time.monotonic() - t0:.0f}s)")
    for r in bad[:15]:
        print(f"  FAIL {r['local_id']}: {str(r.get('error'))[:160]}")


if __name__ == "__main__":
    main()
