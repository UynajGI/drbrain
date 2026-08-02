"""Semantic concept embeddings for the concept graph layer.

Computes one semantic vector per concept node using the project's configured
embedding provider (reusing :func:`drbrain.services.embedding._embed_batch`).
Optionally enriches each concept vector by averaging the embeddings of the
concept label together with the titles of papers containing it — a lightweight
analogue of the paper's "average token embeddings across abstracts".
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from loguru import logger

from drbrain.storage.database import Database

EmbedFn = Callable[[list[str]], list[list[float]]]


def _default_embed_fn(cfg) -> EmbedFn:
    from drbrain.services.embedding import _embed_batch

    def _fn(texts: list[str]) -> list[list[float]]:
        return _embed_batch(texts, cfg)

    return _fn


def aggregate_vectors(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    """Average a set of vectors and L2-normalize the result.

    Args:
        vectors: Non-empty sequence of equal-length vectors.

    Returns:
        A 1-D float32 unit vector (zero vector if input is empty).
    """
    if len(vectors) == 0:
        return np.zeros(0, dtype=np.float32)
    arr = np.asarray(vectors, dtype=np.float32)
    mean = arr.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm > 0:
        mean = mean / norm
    return mean


def compute_concept_embeddings(
    db: Database,
    cfg=None,
    *,
    embed_fn: EmbedFn | None = None,
    context: bool = False,
    model_name: str = "",
) -> int:
    """Compute and store a semantic embedding for every concept node.

    Args:
        db: Database handle.
        cfg: Optional config (used to resolve the default embedding provider).
        embed_fn: Injectable embedding function (texts -> vectors) for tests.
        context: When True, average the label embedding with the titles of papers
            containing the concept for a richer vector.
        model_name: Label recorded in the ``model`` column.

    Returns:
        The number of concept embeddings written (0 if the provider is disabled).
    """
    fn = embed_fn or _default_embed_fn(cfg)
    labels = [r[0] for r in db.conn.execute("SELECT label FROM concept_nodes").fetchall()]
    if not labels:
        return 0

    if not context:
        vectors = fn(labels)
        if not vectors:
            logger.warning("[cg.embed] embedding provider returned nothing (disabled?)")
            return 0
        for label, vec in zip(labels, vectors):
            _store(db, label, vec, model_name)
        db.conn.commit()
        logger.info("[cg.embed] stored {} concept embeddings", len(labels))
        return len(labels)

    # Context mode: per-concept average of label + containing-paper titles.
    written = 0
    for label in labels:
        titles = _paper_titles_for_concept(db, label)
        texts = [label] + titles
        vectors = fn(texts)
        if not vectors:
            continue
        avg = aggregate_vectors(vectors)
        _store(db, label, avg.tolist(), model_name)
        written += 1
    db.conn.commit()
    logger.info("[cg.embed] stored {} context concept embeddings", written)
    return written


def _paper_titles_for_concept(db: Database, label: str, limit: int = 16) -> list[str]:
    rows = db.conn.execute(
        "SELECT DISTINCT p.title FROM concept_cooccurrence c "
        "JOIN papers p ON p.local_id = c.paper_id "
        "WHERE (c.src_label = ? OR c.dst_label = ?) AND p.title != '' LIMIT ?",
        (label, label, limit),
    ).fetchall()
    return [r[0] for r in rows]


def _store(db: Database, label: str, vec: list[float], model_name: str) -> None:
    arr = np.asarray(vec, dtype=np.float32)
    db.insert_concept_embedding(label, arr.tobytes(), int(arr.shape[0]), model_name)


def load_concept_embeddings(db: Database) -> dict[str, np.ndarray]:
    """Load all concept embeddings as ``{label: vector}``."""
    out: dict[str, np.ndarray] = {}
    for label, blob, dim in db.conn.execute("SELECT label, vec, dim FROM concept_embeddings"):
        out[label] = np.frombuffer(blob, dtype=np.float32).copy()
    return out


def nearest_neighbors(db: Database, label: str, k: int = 10) -> list[tuple[str, float]]:
    """Return the ``k`` most similar concepts to ``label`` by cosine similarity.

    Args:
        db: Database handle.
        label: Query concept label (must have an embedding).
        k: Number of neighbours.

    Returns:
        Sorted list of ``(label, similarity)`` excluding the query itself.
    """
    embeddings = load_concept_embeddings(db)
    target = embeddings.get(label)
    if target is None:
        return []
    target_norm = np.linalg.norm(target)
    scored: list[tuple[str, float]] = []
    for other, vec in embeddings.items():
        if other == label:
            continue
        denom = target_norm * np.linalg.norm(vec)
        sim = float(np.dot(target, vec) / denom) if denom > 0 else 0.0
        scored.append((other, sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
