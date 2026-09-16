from __future__ import annotations

import array
import sqlite3

from drbrain.rag.zvec_index import build_zvec_index, query_zvec_index


def _blob(values: list[float]) -> bytes:
    out = array.array("f", values)
    return out.tobytes()


def test_zvec_build_and_query_round_trip(tmp_path):
    db_path = tmp_path / "rag.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE tree_vectors("
        "node_id TEXT, paper_id TEXT, embedding BLOB, content_hash TEXT, tree_layer TEXT)"
    )
    conn.executemany(
        "INSERT INTO tree_vectors VALUES (?, ?, ?, ?, ?)",
        [
            ("p1:1", "p1", _blob([1.0, 0.0, 0.0]), "h1", "pageindex"),
            ("p2:1", "p2", _blob([0.0, 1.0, 0.0]), "h2", "pageindex"),
            ("p3:r", "p3", _blob([0.0, 0.0, 1.0]), "h3", "raptor_L1"),
        ],
    )
    conn.commit()
    conn.close()

    index_path = tmp_path / "zvec"
    stats = build_zvec_index(db_path, index_path)
    assert stats == {"backend": "zvec", "count": 2, "dimension": 3}

    results = query_zvec_index(index_path, [1.0, 0.0, 0.0], 2)
    assert results[0][0] == "p1:1"
    assert results[0][2] == "p1"
    assert results[0][1] > results[1][1]
