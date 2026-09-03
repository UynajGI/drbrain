#!/usr/bin/env python
"""Autoresearch monitor driver — run the research loop end-to-end on real data.

This is the integration-test entry point: it wires the three layers (RAG graph
tools + external plugins + the 12-node loop) against the real ``data/drbrain.db``
and the primary model (OpenCode Zen ``deepseek-v4-flash``), then runs one full
research task and prints the report.

Usage:
    uv run python tests/integration/run_flatband.py [--task "..."] \
        [--db data/drbrain.db] [--plugins PLUGINS_DIR] [--no-db]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_TASK = (
    "从文献网络和概念网络中提取拓扑平带（topological flat band）的基本特征与拓扑化机制，"
    "基于高质量一般平带材料候选，做排列组合式推理，提出哪些材料可能具有拓扑平带并解释原因；"
    "对尚无拓扑平带的材料，判断能否拓扑化；最后选出最可能的材料，给出 DFT 计算建议以验证是否真的存在拓扑平带。"
)


def _build_cfg():
    from drbrain.config import load_config

    return load_config(str(ROOT / "config.yaml"), str(ROOT / "config.local.yaml"))


def _build_db(db_path: str | None):
    from drbrain.storage.database import Database

    return Database(db_path) if db_path else None


def _build_graph():
    from drbrain.graph.engine import GraphEngine

    return GraphEngine()


async def _run(wf, task: str) -> str:
    handler = wf.run(task=task)
    return await handler


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--db", default=str(ROOT / "data" / "drbrain.db"))
    ap.add_argument("--no-db", action="store_true")
    ap.add_argument("--plugins", default=os.environ.get("DRBRAIN_PLUGINS_DIR", ""))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Let the local KG plugin find the real db (read-only).
    os.environ.setdefault("DRBRAIN_DB", args.db)

    cfg = _build_cfg()
    db = None if args.no_db else _build_db(args.db)
    graph = _build_graph()

    from drbrain.loop import ResearchLoopWorkflow

    wf = ResearchLoopWorkflow(
        cfg=cfg,
        db=db,
        graph=graph,
        plugins_dir=args.plugins,
    )
    try:
        report = asyncio.run(_run(wf, args.task))
    finally:
        if db is not None:
            db.close()

    print("\n" + "=" * 72)
    print("RESEARCH LOOP REPORT")
    print("=" * 72)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
