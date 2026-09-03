#!/usr/bin/env python
"""批量 tree 向量嵌入：从文件读 local_id 列表，逐篇生成 vectors+summaries。

用法:
    uv run python scripts/pipeline/embed_batch.py --ids-file /tmp/embed_q1.txt \
        --config config.embed1.yaml --db data/drbrain.db
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import threading
from pathlib import Path

ROOT = Path("/home/jiangyuan/drbrain")
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from drbrain.config import merge_dicts  # noqa: E402
from drbrain.storage.database import Database  # noqa: E402


def _load_cfg(config_name: str) -> dict:
    base = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")) or {}
    local = yaml.safe_load((ROOT / "config.local.yaml").read_text(encoding="utf-8")) or {}
    extra = yaml.safe_load((ROOT / config_name).read_text(encoding="utf-8")) or {}
    return merge_dicts(merge_dicts(base, local), extra)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ids-file", type=str, required=True)
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--db", type=str, default="data/drbrain.db")
    ap.add_argument(
        "--skip-raptor",
        action="store_true",
        help="只生成 PageIndex 向量，跳过 RAPTOR（RAPTOR 需要 LLM 摘要，与 build 抢 key）",
    )
    ap.add_argument(
        "--raptor-out",
        type=str,
        default=None,
        help="RAPTOR 结果缓存到 jsonl（先缓存后入库，不写 db），入库用 load_raptor.py",
    )
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    ids = [x for x in Path(args.ids_file).read_text().strip().split(",") if x]
    print(f"待嵌入: {len(ids)} 篇")

    db = Database(args.db)
    papers_dir = Path(cfg["dirs"]["papers"])
    llm_models = cfg.get("llm", {}).get("models", [])
    embed_cfg = cfg.get("embed", {})
    from drbrain.config import EmbedConfig

    if isinstance(embed_cfg, dict):
        embed_cfg = EmbedConfig(**embed_cfg)

    import concurrent.futures
    import json as _json
    import os

    # append 模式:重启/续跑不覆盖已缓存结果(分片清单负责排除已完成篇)
    raptor_f = open(args.raptor_out, "a", encoding="utf-8") if args.raptor_out else None
    _json_lock = threading.Lock()
    # RAPTOR 摘要 ApiCache:中断/续跑命中缓存零 LLM 成本
    from drbrain.extractor.cache import ApiCache

    _api_cache = ApiCache(str(ROOT / "data/spool/llm_cache"))

    bridge = __import__("drbrain.services.embedding", fromlist=["build_paper_tree_vectors"])
    # 并发:每 worker 线程独立 asyncio.run + 独立 SQLite 连接(WAL+busy_timeout),
    # 单篇超时保护防卡死(与 build.py 同模式)。
    workers = int(os.environ.get("EMBED_WORKERS", "8"))
    per_timeout = int(os.environ.get("EMBED_PAPER_TIMEOUT", "900"))

    def _embed_one(lid: str) -> tuple[str, int, str | None]:
        pdir = papers_dir / lid
        if not (pdir / "tree.json").exists():
            return lid, 0, "no tree.json"
        sink: list[dict] = []
        try:
            if args.skip_raptor:
                from drbrain.services.embedding import build_tree_vectors

                n = build_tree_vectors(db.path, pdir, embed_cfg)
            else:
                n = asyncio.run(
                    bridge.build_paper_tree_vectors(
                        pdir, db.path, embed_cfg, llm_models, sink=sink, cache=_api_cache
                    )
                )
            if sink and raptor_f is not None:
                with _json_lock:
                    for rec in sink:
                        raptor_f.write(_json.dumps(rec, ensure_ascii=False) + "\n")
                    raptor_f.flush()
            return lid, n, None
        except Exception as e:  # noqa: BLE001
            return lid, 0, f"{type(e).__name__}: {e}"

    total_vec = done = 0
    fails: list[tuple[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_embed_one, lid): lid for lid in ids}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            lid = futures[fut]
            try:
                _lid, n, err = fut.result(timeout=per_timeout)
            except concurrent.futures.TimeoutError:
                fails.append((lid, f"timeout>{per_timeout}s"))
                n = 0
                err = None
            except Exception as e:  # noqa: BLE001
                fails.append((lid, f"{type(e).__name__}: {e}"))
                n = 0
                err = None
            if err:
                fails.append((lid, err))
                n = 0
            total_vec += n
            done += 1
            if i % 500 == 0 or i == len(ids):
                print(f"[{i}/{len(ids)}] done={done} vec={total_vec}", flush=True)
    print(f"\n完成: {done} 篇, {total_vec} vectors, fail={len(fails)}")
    for lid, e in fails[:10]:
        print(f"  FAIL {lid}: {e}")
    if raptor_f:
        raptor_f.close()


if __name__ == "__main__":
    main()
