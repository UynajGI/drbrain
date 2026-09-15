"""Unified vector store: one shared Zvec index for leaves and regions (T24/T25).

Design rules:

* ``node kind`` is a payload field, never a second index: the external vector
  leg reads only ``kind = "leaf"`` entries while the tree search may read all
  layers of the same collection.
* The main database keeps *metadata only* (profile id, revision, content hash,
  dimension, state) — never another float copy of a vector.
* Scalar doc ids are a safe hash of the canonical node id (Zvec's id alphabet
  rejects ``:`` and other characters the pipeline uses).
* A write is only "ready" once both the vector and its metadata agree; a
  staging row is retried instead of being treated as valid.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

VECTOR_FIELD = "embedding"
_COLLECTION_NAME = "drbrain_tree"


class VectorStoreError(RuntimeError):
    """The shared vector index is unavailable or inconsistent."""


def _zvec():
    try:
        import zvec
    except ImportError as exc:  # pragma: no cover - minimal installs
        raise VectorStoreError(
            "the unified vector store requires the 'zvec' package (rag profile)"
        ) from exc
    return zvec


def _doc_id(node_id: str) -> str:
    return "n_" + hashlib.sha256(str(node_id).encode("utf-8")).hexdigest()[:61]


def _escape(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


@dataclass(frozen=True)
class VectorEntry:
    node_id: str
    node_revision: int
    kind: str
    local_id: str
    layer: int
    content_hash: str
    profile_id: str
    vector: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.kind not in {"leaf", "region"}:
            raise ValueError(f"unsupported node kind {self.kind!r}")
        if self.node_revision < 1:
            raise ValueError("node_revision must be >= 1")
        if not self.vector:
            raise ValueError("vector must be non-empty")
        if not self.profile_id:
            raise ValueError("profile_id is required")


@dataclass(frozen=True)
class VectorHit:
    node_id: str
    node_revision: int
    kind: str
    local_id: str
    layer: int
    content_hash: str
    profile_id: str
    score: float  # cosine similarity, higher is closer


def _filter_expression(
    *,
    kind: str | None = None,
    local_ids: Sequence[str] | None = None,
    node_revision: int | None = None,
    profile_id: str | None = None,
) -> str | None:
    clauses: list[str] = []
    if kind is not None:
        if kind not in {"leaf", "region"}:
            raise ValueError(f"unsupported kind filter {kind!r}")
        clauses.append(f'kind = "{_escape(kind)}"')
    if local_ids:
        values = ", ".join(f'"{_escape(item)}"' for item in dict.fromkeys(local_ids))
        clauses.append(f"local_id IN ({values})")
    if node_revision is not None:
        clauses.append(f"node_revision = {int(node_revision)}")
    if profile_id is not None:
        clauses.append(f'profile_id = "{_escape(profile_id)}"')
    return " AND ".join(clauses) if clauses else None


class UnifiedVectorStore:
    """One Zvec collection shared by the vector leg and the tree search."""

    def __init__(
        self,
        index_dir: str | Path,
        *,
        create: bool = False,
        dimension: int | None = None,
    ) -> None:
        self.path = Path(index_dir)
        self._collection: Any | None = None
        self._create = bool(create)
        if dimension is not None and int(dimension) <= 0:
            raise ValueError("dimension must be positive when provided")
        self.dimension = int(dimension) if dimension is not None else None

    # ── lifecycle ────────────────────────────────────────────────
    def open(self) -> UnifiedVectorStore:
        zvec = _zvec()
        if self._collection is not None:
            return self
        if not self._create and not self.path.is_dir():
            raise VectorStoreError(f"vector index directory is missing: {self.path}")
        if self.dimension is None:
            raise VectorStoreError(
                "opening the shared vector index requires the embedding dimension "
                "(from the embedding profile)"
            )
        schema = zvec.CollectionSchema(
            _COLLECTION_NAME,
            fields=[
                zvec.FieldSchema("node_id", zvec.DataType.STRING),
                zvec.FieldSchema("kind", zvec.DataType.STRING),
                zvec.FieldSchema("local_id", zvec.DataType.STRING),
                zvec.FieldSchema("content_hash", zvec.DataType.STRING),
                zvec.FieldSchema("profile_id", zvec.DataType.STRING),
                zvec.FieldSchema("node_revision", zvec.DataType.INT64),
                zvec.FieldSchema("layer", zvec.DataType.INT64),
            ],
            vectors=[
                zvec.VectorSchema(
                    VECTOR_FIELD,
                    zvec.DataType.VECTOR_FP32,
                    dimension=self.dimension,
                    index_param=zvec.HnswIndexParam(zvec.MetricType.COSINE),
                )
            ],
        )
        try:
            if self.path.exists():
                self._collection = zvec.open(str(self.path))
            elif self._create:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._collection = zvec.create_and_open(str(self.path), schema)
            else:
                raise VectorStoreError(f"vector index directory is missing: {self.path}")
        except VectorStoreError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize native errors
            raise VectorStoreError(f"cannot open vector index at {self.path}: {exc}") from exc
        return self

    def close(self) -> None:
        if self._collection is not None:
            try:
                self._collection.close()
            finally:
                self._collection = None

    def __enter__(self) -> UnifiedVectorStore:
        return self.open()

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ── writes ───────────────────────────────────────────────────
    def upsert(self, entries: Iterable[VectorEntry]) -> int:
        """Write entries idempotently; returns the number of docs written.

        Re-writing the same node id with the same revision/content hash is a
        no-op at the API level (upsert overwrites), and a changed revision is
        reported so callers can update metadata.
        """
        zvec = _zvec()
        if self._collection is None:
            self.open()
        collection = self._collection
        assert collection is not None
        entries = list(entries)
        if not entries:
            return 0
        dimensions = {len(entry.vector) for entry in entries}
        if len(dimensions) > 1:
            raise VectorStoreError(
                f"refusing to mix embedding dimensions in one write: {sorted(dimensions)}"
            )
        if self.dimension is not None and dimensions != {self.dimension}:
            raise VectorStoreError(
                f"embedding dimension {sorted(dimensions)} does not match the "
                f"index dimension {self.dimension}"
            )
        docs = [
            zvec.Doc(
                id=_doc_id(entry.node_id),
                vectors={VECTOR_FIELD: [float(value) for value in entry.vector]},
                fields={
                    "node_id": entry.node_id,
                    "kind": entry.kind,
                    "local_id": entry.local_id,
                    "content_hash": entry.content_hash,
                    "profile_id": entry.profile_id,
                    "node_revision": int(entry.node_revision),
                    "layer": int(entry.layer),
                },
            )
            for entry in entries
        ]
        try:
            for start in range(0, len(docs), 1000):
                collection.upsert(docs[start : start + 1000])
            collection.flush()
        except Exception as exc:  # noqa: BLE001 - normalize native errors
            raise VectorStoreError(f"vector write failed: {exc}") from exc
        return len(docs)

    def delete(self, node_ids: Sequence[str]) -> None:
        if self._collection is None:
            self.open()
        collection = self._collection
        assert collection is not None
        ids = {_doc_id(node_id) for node_id in node_ids}
        collection.delete(list(ids))

    # ── reads ────────────────────────────────────────────────────
    def query(
        self,
        vector: Sequence[float],
        *,
        top_k: int = 100,
        view: str = "all",
        kind: str | None = None,
        local_ids: Sequence[str] | None = None,
        node_revision: int | None = None,
        profile_id: str | None = None,
    ) -> list[VectorHit]:
        """ANN read; ``view="leaf"`` is the external vector leg, "all" the tree."""
        zvec = _zvec()
        if self._collection is None:
            self.open()
        collection = self._collection
        assert collection is not None
        if view not in {"leaf", "all"}:
            raise ValueError("view must be 'leaf' or 'all'")
        effective_kind = "leaf" if view == "leaf" else kind
        filter_expression = _filter_expression(
            kind=effective_kind,
            local_ids=local_ids,
            node_revision=node_revision,
            profile_id=profile_id,
        )
        query = zvec.Query(field_name=VECTOR_FIELD, vector=[float(v) for v in vector])
        try:
            docs = collection.query(
                query,
                topk=max(1, int(top_k)),
                filter=filter_expression,
                output_fields=[
                    "node_id",
                    "kind",
                    "local_id",
                    "content_hash",
                    "profile_id",
                    "node_revision",
                    "layer",
                ],
            )
        except Exception as exc:  # noqa: BLE001 - normalize native errors
            raise VectorStoreError(f"vector query failed: {exc}") from exc
        hits: list[VectorHit] = []
        for doc in docs:
            fields = doc.fields or {}
            node_id = str(fields.get("node_id") or doc.id)
            hits.append(
                VectorHit(
                    node_id=node_id,
                    node_revision=int(fields.get("node_revision") or 0),
                    kind=str(fields.get("kind") or ""),
                    local_id=str(fields.get("local_id") or ""),
                    layer=int(fields.get("layer") or 0),
                    content_hash=str(fields.get("content_hash") or ""),
                    profile_id=str(fields.get("profile_id") or ""),
                    # Zvec reports cosine distance (0 = identical).
                    score=1.0 - float(doc.score or 0.0),
                )
            )
        return hits

    def count(self) -> int:
        if self._collection is None:
            self.open()
        collection = self._collection
        assert collection is not None
        try:
            stats = getattr(collection, "stats", None)
            if callable(stats):
                try:
                    stats = stats()
                except TypeError:
                    pass
            for attribute in ("doc_count", "count", "num_docs"):
                if stats is not None and hasattr(stats, attribute):
                    return int(getattr(stats, attribute))
        except Exception:  # noqa: BLE001 - stats are advisory
            logger.debug("[vector] stats unavailable for {}", self.path)
        return -1


def needs_write(existing_meta: dict | None, entry: VectorEntry) -> bool:
    """Whether one node's vector must be (re)written for this entry."""
    return needs_write_meta(
        existing_meta,
        node_revision=entry.node_revision,
        content_hash=entry.content_hash,
        profile_id=entry.profile_id,
    )


def needs_write_meta(
    existing_meta: dict | None,
    *,
    node_revision: int,
    content_hash: str,
    profile_id: str,
) -> bool:
    """Whether a node's vector must be (re)written for these identifiers.

    The staging entry form of :func:`needs_write`, usable before the vector
    itself exists (T45 preparation walks nodes that need embedding).
    """
    if existing_meta is None:
        return True
    if str(existing_meta.get("state")) != "ready":
        return True
    if int(existing_meta.get("node_revision") or 0) != int(node_revision):
        return True
    if str(existing_meta.get("content_hash") or "") != str(content_hash):
        return True
    return str(existing_meta.get("profile_id") or "") != str(profile_id)
