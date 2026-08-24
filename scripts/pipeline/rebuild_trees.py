#!/usr/bin/env python
"""空树重建：对 tree.json 结构为空的论文重跑 md_to_tree + doc description。

输入: /tmp/empty_trees.txt（每行一个 local_id）
用法: uv run python scripts/pipeline/rebuild_trees.py [--workers 8]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

ROOT = Path("/home/jiangyuan/drbrain")
sys.path.insert(0, str(ROOT))

from drbrain.parser.pageindex_parser import TreeConfig, md_to_tree  # noqa: E402
from scripts.pipeline.common import load_cfg  # noqa: E402


def rebuild_one(args: tuple) -> dict:
    lid, cfg, build_cfg = args
    from loguru import logger

    pdir = ROOT / "data/papers" / lid
    md_path = pdir / "raw.md"
    if not md_path.exists():
        return {"lid": lid, "ok": False, "error": "no raw.md"}
    llm_models = cfg.get("llm", {}).get("models", [])
    try:
        pageindex_cfg = TreeConfig(
            if_thinning=False,
            if_add_node_summary=True,
            if_add_doc_description=False,  # doc description 单独用 hy3（ox 返回空）
            if_add_node_text=False,
            if_add_node_id=True,
            max_node_tokens=10000,
            summary_token_threshold=2000,
        )
        doc_tree = asyncio.run(md_to_tree(md_path, config=pageindex_cfg, models=llm_models))
        # 无 markdown 标题的纯文本片段（书摘/表格等）切不出章节——
        # 合成单节点全文树，保证可进向量检索
        if not doc_tree.structure:
            text = md_path.read_text(encoding="utf-8")
            doc_tree.structure = [
                {
                    "title": "Full Text",
                    "node_id": "0000",
                    "summary": text[:8000],
                }
            ]
        try:
            from drbrain.parser.pageindex.retrieval import (
                _create_clean_structure_for_description,
            )
            from drbrain.parser.pageindex.summary import _generate_doc_description

            hy3_models = build_cfg.get("llm", {}).get("models", [])
            clean = _create_clean_structure_for_description(doc_tree.structure)
            if isinstance(clean, list):
                clean = {"structure": clean}
            desc = asyncio.run(_generate_doc_description(clean, hy3_models))
            if desc:
                doc_tree.doc_description = desc
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[rebuild] doc-desc failed {lid}: {e}")
        (pdir / "tree.json").write_text(doc_tree.to_json(), encoding="utf-8")
        return {"lid": lid, "ok": True, "sections": len(doc_tree.structure)}
    except Exception as e:  # noqa: BLE001
        return {"lid": lid, "ok": False, "error": f"{type(e).__name__}: {e}"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--list", type=str, default="/tmp/empty_trees.txt")
    args = ap.parse_args()

    ids = [x for x in Path(args.list).read_text().split("\n") if x.strip()]
    print(f"待重建树: {len(ids)} 篇")
    cfg = load_cfg(None)
    build_cfg = load_cfg("config.build.yaml")

    tasks = [(lid, cfg, build_cfg) for lid in ids]
    ok = fail = 0
    t0 = time.monotonic()
    import concurrent.futures

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, rec in enumerate(ex.map(rebuild_one, tasks), 1):
            ok += rec["ok"]
            fail += not rec["ok"]
            if i % 50 == 0 or i == len(ids):
                print(
                    f"[{i}/{len(ids)}] ok={ok} fail={fail} elapsed={time.monotonic() - t0:.0f}s",
                    flush=True,
                )
    print(f"\n重建完成: ok={ok} fail={fail} ({time.monotonic() - t0:.0f}s)")


if __name__ == "__main__":
    main()
