"""sqlite-vec 向量索引层 — tree_vectors 的 ANN 加速。

存储：tree_vectors_vec(vec0 虚拟表)，node_id TEXT PK + float[1024]。
写入：build_tree_vectors 双写（base 表 + vec 表）。
检索：search_tree 全库查询走 vec KNN（L2，归一化向量下与余弦序一致）；
      分数换算 cos = 1 - d²/2 保持 fusion 端 cosine ∈ [-1,1] 语义。
回退：扩展不可用 / 表未同步（count 不等）/ 维度不匹配 → 原暴力路径。
"""

from __future__ import annotations

import sqlite3
import struct

VEC_TABLE = "tree_vectors_vec"
BASE_TABLE = "tree_vectors"


def load_vec(conn: sqlite3.Connection) -> bool:
    """Load sqlite-vec extension into *conn*. Returns False if unavailable."""
    try:
        import sqlite_vec

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return True
    except Exception:  # noqa: BLE001 — extension optional everywhere
        return False


def ensure_vec_table(conn: sqlite3.Connection, dim: int = 1024) -> bool:
    """Create the vec0 virtual table if missing. Returns True when usable."""
    if not load_vec(conn):
        return False
    try:
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {VEC_TABLE} USING vec0("
            f"node_id TEXT PRIMARY KEY, embedding float[{dim}])"
        )
        return True
    except sqlite3.Error:
        return False


META_TABLE = "vector_meta"


def _ensure_meta_table(conn: sqlite3.Connection) -> None:
    conn.execute(f"CREATE TABLE IF NOT EXISTS {META_TABLE} (key TEXT PRIMARY KEY, value TEXT)")


def mark_vec_synced(conn: sqlite3.Connection) -> None:
    """Writer hook (R-I2): stamp the watermark after a COMPLETED vector sync.

    Readers then check one row per query instead of two O(N) COUNT(*) scans.
    Any incremental ``vec_upsert`` marks the store dirty until the next full
    sync completes.
    """
    _ensure_meta_table(conn)
    conn.execute(f"INSERT OR REPLACE INTO {META_TABLE} (key, value) VALUES ('synced', '1')")
    conn.commit()


def mark_vec_dirty(conn: sqlite3.Connection) -> None:
    """Invalidate the watermark — vectors changed since the last full sync."""
    _ensure_meta_table(conn)
    conn.execute(f"INSERT OR REPLACE INTO {META_TABLE} (key, value) VALUES ('synced', '0')")
    conn.commit()


def vec_synced(conn: sqlite3.Connection) -> bool:
    """True when the vector store carries a completed-sync watermark (R-I2).

    The old per-query ``COUNT(base) == COUNT(vec)`` check was O(N) on every
    retrieval and degraded to a full scan on any mismatch. The watermark is
    written by the sync writers (embedding build / vec_backfill) and cleared
    by every incremental write, so a missing or cleared watermark means
    "not synced" without paying for a scan. Hot path (OCR r6): a plain
    SELECT with no schema DDL — the CREATE runs only on the cold
    table-missing error path, never on the per-query read.
    """
    try:
        row = conn.execute(f"SELECT value FROM {META_TABLE} WHERE key = 'synced'").fetchone()
    except sqlite3.OperationalError:
        # 表尚不存在（冷库）：建表一次后重读，之后每查询都是纯 SELECT。
        _ensure_meta_table(conn)
        try:
            row = conn.execute(f"SELECT value FROM {META_TABLE} WHERE key = 'synced'").fetchone()
        except sqlite3.Error:
            return False
        return bool(row and row[0] == "1")
    except sqlite3.Error:
        return False
    return bool(row and row[0] == "1")


def embedding_byte_len(conn: sqlite3.Connection) -> int:
    """Expected float32 byte length of one embedding, read from the data.

    Callers must stop hardcoding 4096 (R-I2): the corpus decides its own dim.
    Falls back to the historical 1024-float layout when the corpus is empty.
    """
    try:
        row = conn.execute(f"SELECT length(embedding) FROM {BASE_TABLE} LIMIT 1").fetchone()
        return int(row[0]) if row and row[0] else 4096
    except sqlite3.Error:
        return 4096


def vec_stored_dim(conn: sqlite3.Connection) -> int | None:
    """Declared embedding dim of the vec table (0 if empty/unknown)."""
    try:
        row = conn.execute(f"SELECT length(embedding)/4 FROM {VEC_TABLE} LIMIT 1").fetchone()
        return row[0] if row else None
    except sqlite3.Error:
        return None


def pack_query(vec: list[float]) -> bytes:
    """Pack a python float list into sqlite-vec's expected blob format."""
    return struct.pack(f"{len(vec)}f", *vec)


def vec_knn(
    conn: sqlite3.Connection,
    query_blob: bytes,
    k: int,
) -> list[tuple[str, float]]:
    """KNN search. Returns [(node_id, l2_distance)] ordered nearest-first."""
    rows = conn.execute(
        f"SELECT node_id, distance FROM {VEC_TABLE} WHERE embedding MATCH ? AND k = ?",
        (query_blob, k),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def l2_to_cosine(l2: float) -> float:
    """Convert squared-free L2 distance to cosine for unit vectors: cos = 1 - d²/2."""
    return 1.0 - (l2 * l2) / 2.0


def vec_upsert(conn: sqlite3.Connection, node_id: str, blob: bytes) -> None:
    """Insert/replace one vector blob (must match declared dim).

    vec0 virtual tables reject INSERT OR REPLACE (UNIQUE pk error via shadow
    tables), so emulate upsert with DELETE + INSERT.
    """
    conn.execute(f"DELETE FROM {VEC_TABLE} WHERE node_id = ?", (node_id,))
    conn.execute(
        f"INSERT INTO {VEC_TABLE}(node_id, embedding) VALUES (?, ?)",
        (node_id, blob),
    )
    # R-I2: incremental writes invalidate the completed-sync watermark.
    mark_vec_dirty(conn)
