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

    bridge = __import__("drbrain.services.embedding", fromlist=["build_paper_tree_vectors"])
    total_vec = done = 0
    for i, lid in enumerate(ids, 1):
        pdir = papers_dir / lid
        if not (pdir / "tree.json").exists():
            continue
        try:
            if args.skip_raptor:
                from drbrain.services.embedding import build_tree_vectors

                n = build_tree_vectors(db.path, pdir, embed_cfg)
            else:
                n = asyncio.run(
                    bridge.build_paper_tree_vectors(pdir, db.path, embed_cfg, llm_models)
                )
            total_vec += n
            done += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {lid}: {str(e)[:100]}", flush=True)
        if i % 500 == 0 or i == len(ids):
            print(f"[{i}/{len(ids)}] done={done} vec={total_vec}", flush=True)
    print(f"\n完成: {done} 篇, {total_vec} vectors")


if __name__ == "__main__":
    main()
