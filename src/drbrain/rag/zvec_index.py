"""Zvec-backed vector index for immutable SQL RAG generations.

The authoritative corpus remains ``corpus.sqlite3``.  This module only builds
and queries a rebuildable ANN directory that is copied into the same published
generation as the SQLite snapshot, so text, metadata, and vectors cannot drift.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from drbrain.rag.status import RetrievalUnavailableError

INDEX_DIR_NAME = "zvec"
_VECTOR_FIELD = "embedding"
_COLLECTION_NAME = "drbrain_rag"


class ZvecUnavailableError(RetrievalUnavailableError):
    """Raised when a configured Zvec index cannot be loaded or queried."""


def configured_vector_backend(cfg: Any) -> str:
    """Return the configured vector backend and validate its vocabulary."""

    raw = cfg.get("retrieval", {}) if isinstance(cfg, dict) else getattr(cfg, "retrieval", {})
    value = (
        raw.get("vector_backend", "sqlite")
        if isinstance(raw, dict)
        else getattr(raw, "vector_backend", "sqlite")
    )
    backend = str(value or "sqlite").strip().lower()
    if backend not in {"sqlite", "zvec"}:
        raise ValueError(f"unsupported retrieval.vector_backend: {backend!r}")
    return backend


def configured_vector_top_k(cfg: Any, default: int = 100) -> int:
    raw = cfg.get("retrieval", {}) if isinstance(cfg, dict) else getattr(cfg, "retrieval", {})
    value = (
        raw.get("vector_top_k", default)
        if isinstance(raw, dict)
        else getattr(raw, "vector_top_k", default)
    )
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _zvec():
    try:
        import zvec
    except ImportError as exc:  # pragma: no cover - exercised on minimal installs
        raise ZvecUnavailableError(
            "retrieval.vector_backend=zvec requires the 'zvec' package"
        ) from exc
    return zvec


def _dimension(blob: bytes) -> int:
    if not blob or len(blob) % 4:
        return 0
    return len(blob) // 4


def build_zvec_index(sqlite_path: str | Path, index_path: str | Path) -> dict[str, int | str]:
    """Build a Zvec HNSW index from PageIndex rows in a derived SQLite DB.

    ``index_path`` must not already exist; callers should pass a generation
    staging path so publication remains atomic.
    """

    zvec = _zvec()
    source = Path(sqlite_path)
    target = Path(index_path)
    if not source.is_file():
        raise ZvecUnavailableError(f"vector source database is missing: {source}")
    if target.exists():
        raise ZvecUnavailableError(f"refusing to overwrite existing Zvec index: {target}")

    conn = sqlite3.connect(source)
    try:
        rows = conn.execute(
            "SELECT node_id, paper_id, embedding, content_hash, tree_layer "
            "FROM tree_vectors WHERE tree_layer = 'pageindex' "
            "AND embedding IS NOT NULL AND length(embedding) > 0"
        ).fetchall()
    except sqlite3.Error as exc:
        raise ZvecUnavailableError("tree_vectors table is unavailable") from exc
    finally:
        conn.close()

    if not rows:
        return {"backend": "zvec", "count": 0, "dimension": 0}

    dimension = _dimension(bytes(rows[0][2]))
    if dimension <= 0:
        raise ZvecUnavailableError("tree_vectors contains an invalid embedding blob")
    for row in rows:
        if _dimension(bytes(row[2])) != dimension:
            raise ZvecUnavailableError("tree_vectors contains mixed embedding dimensions")

    target.parent.mkdir(parents=True, exist_ok=True)
    schema = zvec.CollectionSchema(
        _COLLECTION_NAME,
        fields=[
            zvec.FieldSchema("paper_id", zvec.DataType.STRING),
            zvec.FieldSchema("node_id", zvec.DataType.STRING),
            zvec.FieldSchema("tree_layer", zvec.DataType.STRING),
            zvec.FieldSchema("content_hash", zvec.DataType.STRING),
        ],
        vectors=[
            zvec.VectorSchema(
                _VECTOR_FIELD,
                zvec.DataType.VECTOR_FP32,
                dimension=dimension,
                index_param=zvec.HnswIndexParam(zvec.MetricType.COSINE),
            )
        ],
    )
    collection = zvec.create_and_open(str(target), schema)
    try:
        docs = []
        for node_id, paper_id, blob, content_hash, tree_layer in rows:
            docs.append(
                zvec.Doc(
                    # Zvec IDs use a stricter alphabet than PageIndex node
                    # keys (which commonly contain ``:``). Keep the original
                    # key in a scalar field and use a deterministic safe ID.
                    id="n_" + hashlib.sha256(str(node_id).encode("utf-8")).hexdigest()[:61],
                    vectors={_VECTOR_FIELD: _decode_embedding(bytes(blob))},
                    fields={
                        "paper_id": str(paper_id),
                        "node_id": str(node_id),
                        "tree_layer": str(tree_layer),
                        "content_hash": str(content_hash or ""),
                    },
                )
            )
            if len(docs) >= 1000:
                collection.insert(docs)
                docs = []
        if docs:
            collection.insert(docs)
        collection.optimize()
    finally:
        collection.close()

    metadata: dict[str, int | str] = {"backend": "zvec", "count": len(rows), "dimension": dimension}
    (target / "metadata.json").write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    return metadata


def _decode_embedding(blob: bytes) -> list[float]:
    import array

    values = array.array("f")
    values.frombytes(blob)
    return values.tolist()


def query_zvec_index(
    index_path: str | Path,
    query_vector: Iterable[float],
    top_k: int,
) -> list[tuple[str, float, str]]:
    """Query Zvec, returning ``(node_id, cosine_similarity, paper_id)``."""
    return [
        (node_id, score, paper_id)
        for node_id, score, paper_id, _ in query_zvec_evidence(index_path, query_vector, top_k)
    ]


def query_zvec_evidence(
    index_path: str | Path,
    query_vector: Iterable[float],
    top_k: int,
) -> list[tuple[str, float, str, str]]:
    """Query Zvec, returning ``(node_id, cosine_similarity, paper_id, content_hash)``.

    The content hash lets a caller verify that an ANN hit still matches the
    evidence row it claims to be (same node id, same revision) before use.
    """

    zvec = _zvec()
    path = Path(index_path)
    if not path.is_dir() or not (path / "metadata.json").is_file():
        raise ZvecUnavailableError(f"Zvec index is missing or incomplete: {path}")
    try:
        collection = zvec.open(str(path))
    except Exception as exc:  # noqa: BLE001 - normalize native errors
        raise ZvecUnavailableError(f"failed to open Zvec index: {path}") from exc
    try:
        query = zvec.Query(field_name=_VECTOR_FIELD, vector=[float(v) for v in query_vector])
        docs = collection.query(
            query,
            topk=max(1, int(top_k)),
            output_fields=["paper_id", "node_id", "content_hash"],
        )
        out: list[tuple[str, float, str, str]] = []
        for doc in docs:
            # Zvec returns cosine distance (0 = identical), while DrBrain's
            # fusion legs use a higher-is-better similarity score.
            distance = float(doc.score or 0.0)
            fields = doc.fields or {}
            out.append(
                (
                    str(fields.get("node_id") or doc.id),
                    1.0 - distance,
                    str(fields.get("paper_id", "")),
                    str(fields.get("content_hash") or ""),
                )
            )
        return out
    except Exception as exc:  # noqa: BLE001 - normalize native errors
        raise ZvecUnavailableError("Zvec query failed") from exc
    finally:
        collection.close()


__all__ = [
    "INDEX_DIR_NAME",
    "ZvecUnavailableError",
    "build_zvec_index",
    "configured_vector_backend",
    "configured_vector_top_k",
    "query_zvec_evidence",
    "query_zvec_index",
]
