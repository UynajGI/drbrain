#!/usr/bin/env python
"""存量 tree_vectors → tree_vectors_vec 回填（幂等，可断点续传）。

对比 base 与 vec 表的 node_id 集合，只同步缺失/不一致的行。
用法:
    uv run python scripts/pipeline/vec_backfill.py [--db data/drbrain.db] [--batch 2000]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path("/home/jiangyuan/drbrain")
sys.path.insert(0, str(ROOT))

from drbrain.storage import vector_index as vi  # noqa: E402
from drbrain.storage.connection import connect_wal  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=str, default="data/drbrain.db")
    ap.add_argument("--batch", type=int, default=2000)
    args = ap.parse_args()

    conn = connect_wal(args.db)
    if not vi.load_vec(conn):
        print("sqlite-vec 不可用，退出")
        sys.exit(1)

    t0 = time.monotonic()
    n_base = conn.execute("SELECT COUNT(*) FROM tree_vectors").fetchone()[0]
    print(f"base 表: {n_base} 行")

    # 建表（以首个向量的实际维度为准）
    row = conn.execute("SELECT embedding FROM tree_vectors LIMIT 1").fetchone()
    dim = len(row[0]) // 4 if row else 1024
    vi.ensure_vec_table(conn, dim)
    n_vec = conn.execute(f"SELECT COUNT(*) FROM {vi.VEC_TABLE}").fetchone()[0]
    print(f"vec 表: {n_vec} 行（dim={dim}）")

    # 找缺失的 node_id（base 有 vec 无）
    missing = [
        r[0]
        for r in conn.execute(
            f"SELECT node_id FROM tree_vectors "
            f"WHERE node_id NOT IN (SELECT node_id FROM {vi.VEC_TABLE})"
        )
    ]
    print(f"待回填: {len(missing)}")

    done = fail = 0
    for i in range(0, len(missing), args.batch):
        chunk = missing[i : i + args.batch]
        ph = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"SELECT node_id, embedding FROM tree_vectors WHERE node_id IN ({ph})", chunk
        ).fetchall()
        for nid, blob in rows:
            try:
                vi.vec_upsert(conn, nid, blob)
                done += 1
            except Exception:  # noqa: BLE001 — 维度异常等跳过
                fail += 1
        conn.commit()
        vi.mark_vec_synced(conn)  # R-I2: 完整回填后打同步水位
        print(
            f"[{min(i + args.batch, len(missing))}/{len(missing)}] "
            f"ok={done} fail={fail} elapsed={time.monotonic() - t0:.0f}s",
            flush=True,
        )

    n_vec = conn.execute(f"SELECT COUNT(*) FROM {vi.VEC_TABLE}").fetchone()[0]
    print(
        f"\n回填完成: ok={done} fail={fail}，vec 表现在 {n_vec} 行 ({time.monotonic() - t0:.0f}s)"
    )
    conn.close()


if __name__ == "__main__":
    main()
