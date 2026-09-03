#!/usr/bin/env python
"""主库 → RAG 工作副本增量同步（tree_vectors + tree_summaries）。

drbrain_rag.db 是 drbrain.db 的热备副本 + RAG 层（node_texts/FTS5）。
主库每次批量入库（如 load_raptor）后，跑本脚本把新增行同步过来，
两库即同为最新。幂等：INSERT OR REPLACE(node_id PK)，主库为准，重跑安全。

用法:
    uv run python scripts/pipeline/ragdb_sync.py [--main data/drbrain.db] [--rag data/drbrain_rag.db]
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

ROOT = Path("/home/jiangyuan/drbrain")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--main", type=str, default="data/drbrain.db")
    ap.add_argument("--rag", type=str, default="data/drbrain_rag.db")
    args = ap.parse_args()

    main_db = ROOT / args.main
    rag_db = ROOT / args.rag
    if not main_db.exists() or not rag_db.exists():
        print(f"库不存在: {main_db} / {rag_db}")
        return

    t0 = time.time()
    # uri=True so the ATTACHed main db can use a file:...?mode=ro URI
    conn = sqlite3.connect(f"file:{rag_db}", uri=True)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute(f"ATTACH DATABASE 'file:{main_db}?mode=ro' AS main_db")
    before_v = conn.execute("SELECT COUNT(*) FROM tree_vectors").fetchone()[0]
    before_s = conn.execute("SELECT COUNT(*) FROM tree_summaries").fetchone()[0]

    cur = conn.execute(
        """
        INSERT OR REPLACE INTO tree_vectors (node_id, paper_id, embedding, content_hash, tree_layer)
        SELECT node_id, paper_id, embedding, content_hash, tree_layer
        FROM main_db.tree_vectors
        """
    )
    n_v = cur.rowcount
    conn.commit()
    cur = conn.execute(
        """
        INSERT OR REPLACE INTO tree_summaries (node_id, paper_id, summary_text, source_node_ids, tree_layer)
        SELECT node_id, paper_id, summary_text, source_node_ids, tree_layer
        FROM main_db.tree_summaries
        """
    )
    n_s = cur.rowcount
    conn.commit()
    conn.execute("DETACH DATABASE main_db")
    conn.close()

    print(
        f"同步完成: tree_vectors +{n_v} ({before_v} → {before_v + n_v}), "
        f"tree_summaries +{n_s} ({before_s} → {before_s + n_s}), "
        f"耗时 {time.time() - t0:.0f}s"
    )


if __name__ == "__main__":
    main()
