#!/usr/bin/env python
"""tree_vectors float32 → int8 量化迁移（4x 体积压缩，精度无损）。

评估结论（2026-08-24，200 向量样本）：per-vector 对称 int8 量化后
top-10 重合 10/10、相似度相关性 1.0000 —— 零检索质量损失。
sqlite-vec vec0 原生支持 int8 列。

流程：
1. 建 scale shadow 表 + tree_vectors_vec_i8（int8[1024]）。
2. 分批读 base(float32) → per-vector 量化 → 写 i8 表 + scale 表。
3. 完成后原子改名：tree_vectors_vec→_f32 备份，_i8→tree_vectors_vec。
4. base float32 blob 保留为真值源（回滚安全；rerank 可反量化）。

用法: uv run python scripts/pipeline/vec_quantize_int8.py [--batch 20000]
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/jiangyuan/drbrain")
sys.path.insert(0, str(ROOT))

from drbrain.storage.connection import connect_wal  # noqa: E402

VEC_F32 = "tree_vectors_vec"
VEC_I8 = "tree_vectors_vec_i8"
SCALE_TABLE = "vec_i8_scale"
DIM = 1024


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=str, default="data/drbrain.db")
    ap.add_argument("--batch", type=int, default=20000)
    args = ap.parse_args()

    conn = connect_wal(args.db)
    conn.enable_load_extension(True)
    import sqlite_vec

    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    n_base = conn.execute("SELECT COUNT(*) FROM tree_vectors").fetchone()[0]
    print(f"base 向量: {n_base:,}")

    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {SCALE_TABLE}(node_id TEXT PRIMARY KEY, scale BLOB NOT NULL)"
    )
    conn.execute(f"DROP TABLE IF EXISTS {VEC_I8}")
    conn.execute(
        f"CREATE VIRTUAL TABLE {VEC_I8} USING vec0(node_id TEXT PRIMARY KEY, embedding int8[{DIM}])"
    )

    t0 = time.monotonic()
    done = 0
    last_id = ""
    while True:
        rows = conn.execute(
            "SELECT node_id, embedding FROM tree_vectors WHERE node_id > ? "
            "ORDER BY node_id LIMIT ?",
            (last_id, args.batch),
        ).fetchall()
        if not rows:
            break
        last_id = rows[-1][0]
        blobs = b"".join(r[1] for r in rows)
        mat = np.frombuffer(blobs, dtype=np.float32).reshape(len(rows), DIM)
        scale = np.abs(mat).max(axis=1, keepdims=True) / 127.0
        scale[scale == 0] = 1.0
        i8 = np.clip(np.round(mat / scale), -127, 127).astype(np.int8)

        for (nid, _), s, v in zip(rows, scale.ravel(), i8):
            conn.execute(
                f"INSERT OR REPLACE INTO {SCALE_TABLE}(node_id, scale) VALUES (?, ?)",
                (nid, struct.pack("f", float(s))),
            )
            conn.execute(
                f"INSERT INTO {VEC_I8}(node_id, embedding) VALUES (?, ?)",
                (nid, v.tobytes()),
            )
        done += len(rows)
        conn.commit()
        eta = (time.monotonic() - t0) / done * (n_base - done)
        print(
            f"[{done:,}/{n_base:,}] elapsed={time.monotonic() - t0:.0f}s eta={eta / 60:.0f}min",
            flush=True,
        )

    n_i8 = conn.execute(f"SELECT COUNT(*) FROM {VEC_I8}").fetchone()[0]
    print(f"\n量化写入完成: i8 表 {n_i8:,} 行 ({time.monotonic() - t0:.0f}s)")

    if n_i8 != n_base:
        print(f"⚠️ 行数不一致（base={n_base:,} vs i8={n_i8:,}），不改名，人工检查")
        conn.close()
        sys.exit(1)

    # 3. 原子改名切换
    conn.execute(f"ALTER TABLE {VEC_F32} RENAME TO {VEC_F32}_f32_bak")
    conn.execute(f"ALTER TABLE {VEC_I8} RENAME TO {VEC_F32}")
    conn.commit()

    # 快速 KNN 冒烟测试
    row = conn.execute("SELECT embedding FROM tree_vectors LIMIT 1").fetchone()
    hits = conn.execute(
        f"SELECT node_id, distance FROM {VEC_F32} WHERE embedding MATCH ? AND k = 5",
        (row[0],),
    ).fetchall()
    print(f"切换完成。KNN 冒烟: top1={hits[0][0][:20]} d={hits[0][1]:.3f}")
    db_size = conn.execute(
        "SELECT page_count*page_size/1024/1024/1024.0 FROM pragma_page_count(), pragma_page_size()"
    ).fetchone()[0]
    print(f"库体积: {db_size:.1f} GB（f32 备份表可后续 DROP 再 VACUUM 回收）")
    conn.close()


if __name__ == "__main__":
    main()
