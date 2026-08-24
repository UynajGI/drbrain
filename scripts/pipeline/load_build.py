#!/usr/bin/env python
"""把 build jsonl（jsonl-out 模式产物）批量写入分片 db。

build 用 --jsonl-out 只抽 LLM 结果不写 db（并发无锁），本脚本把 jsonl 里的
concepts/relations 批量 INSERT 进分片 db（单进程串行写，无锁冲突）。

用法:
    uv run python scripts/pipeline/load_build.py --jsonl data/shards/shard0.build.jsonl \
        --db data/shards/shard0.db
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from drbrain.storage.database import Database

ROOT = Path("/home/jiangyuan/drbrain")
sys.path.insert(0, str(ROOT))

from scripts.pipeline.build import VALID_TYPES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", type=str, required=True)
    ap.add_argument("--db", type=str, required=True)
    args = ap.parse_args()

    db = Database(args.db)
    t0 = time.monotonic()
    ok = fail = 0
    concepts = relations = 0
    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            lid = rec.get("local_id", "")
            if not rec.get("ok"):
                fail += 1
                continue
            for c in rec.get("concepts", []):
                ctype = c.get("type", "")
                label = c.get("label", "")
                conf = c.get("confidence", 0.5)
                if ctype not in VALID_TYPES or not label:
                    continue
                db.insert_concept(
                    lid,
                    ctype,
                    label,
                    conf,
                    section=c.get("section", ""),
                    node_id=c.get("node_id", ""),
                )
                concepts += 1
            for r in rec.get("relations", []):
                head, rel, tail = r.get("head", ""), r.get("rel", ""), r.get("tail", "")
                if head and rel and tail:
                    try:
                        db.insert_edge(
                            head,
                            tail,
                            rel,
                            lid,
                            node_id=r.get("node_id", ""),
                            section=r.get("section", ""),
                        )
                        relations += 1
                    except Exception:  # noqa: BLE001
                        pass
            db.set_paper_status(lid, "extracted")
            db.commit()
            ok += 1
    db.close()
    print(
        f"入库完成: ok={ok} fail={fail} concepts={concepts} relations={relations} "
        f"({time.monotonic() - t0:.0f}s)"
    )


if __name__ == "__main__":
    main()
