#!/usr/bin/env python
"""双库一致性校验 + RAG 层质量审计。

校验 drbrain.db（主库）与 drbrain_rag.db（RAG 工作副本）：
1. 数据量一致：papers/tree_vectors/tree_summaries 计数逐表对比
2. 内容一致：tree_vectors 按 content_hash 聚合指纹对比（快，不逐行比 blob）
3. RAG 层健康：node_texts/FTS5 覆盖、向量-文本哈希对齐率

用法:
    uv run python scripts/pipeline/ragdb_verify.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

ROOT = Path("/home/jiangyuan/drbrain")
MAIN = ROOT / "data" / "drbrain.db"
RAG = ROOT / "data" / "drbrain_rag.db"


def _fp(conn: sqlite3.Connection) -> str:
    """聚合指纹：层分布 + 哈希异或/计数，百万行级别秒级完成。"""
    rows = conn.execute(
        """
        SELECT tree_layer, COUNT(*), SUM(CAST(substr(content_hash, 1, 8) AS INTEGER))
        FROM tree_vectors GROUP BY tree_layer ORDER BY tree_layer
        """
    ).fetchall()
    return " | ".join(f"{layer}: n={n}, sum={s}" for layer, n, s in rows)


def main() -> None:
    main = sqlite3.connect(f"file:{MAIN}?mode=ro", uri=True)
    rag = sqlite3.connect(f"file:{RAG}?mode=ro", uri=True)

    print("== 1. 计数对比（主库 → rag 库）==")
    for table in ("tree_vectors", "tree_summaries"):
        m = main.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        r = rag.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        mark = "✅" if m == r else "❌"
        print(f"  {table}: {m} → {r} {mark}")
    m = main.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    r = rag.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    mark = "✅" if m == r else "❌"
    print(f"  papers: {m} → {r} {mark}")

    print("== 2. tree_vectors 内容指纹 ==")
    fm, fr = _fp(main), _fp(rag)
    print(f"  主库: {fm}")
    print(f"  rag : {fr}")
    print("  一致" if fm == fr else "  ❌ 不一致")

    print("== 3. RAG 层健康 ==")
    nt = rag.execute("SELECT COUNT(*) FROM node_texts").fetchone()[0]
    fts = rag.execute("SELECT COUNT(*) FROM node_texts_fts").fetchone()[0]
    pi = rag.execute("SELECT COUNT(*) FROM tree_vectors WHERE tree_layer='pageindex'").fetchone()[0]
    print(f"  node_texts: {nt} | FTS5: {fts} | pageindex 向量: {pi}")
    match = rag.execute(
        """
        SELECT COUNT(*) FROM tree_vectors tv
        JOIN node_texts x ON x.node_key = tv.node_id
        WHERE tv.tree_layer = 'pageindex' AND x.content_hash = tv.content_hash
        """
    ).fetchone()[0]
    aligned = rag.execute(
        """
        SELECT COUNT(*) FROM tree_vectors tv
        JOIN node_texts x ON x.node_key = tv.node_id
        WHERE tv.tree_layer = 'pageindex'
        """
    ).fetchone()[0]
    print(
        f"  向量-文本哈希对齐: {match}/{aligned} "
        f"({100.0 * match / max(aligned, 1):.1f}%) —— 未对齐者向量与全文不同源，检索会错位"
    )
    orphan = rag.execute(
        "SELECT COUNT(*) FROM tree_vectors tv LEFT JOIN papers p ON tv.paper_id=p.local_id WHERE p.local_id IS NULL"
    ).fetchone()[0]
    print(f"  孤儿向量: {orphan}")
    main.close()
    rag.close()


if __name__ == "__main__":
    main()
