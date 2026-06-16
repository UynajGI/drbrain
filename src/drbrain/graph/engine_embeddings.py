"""TransE embedding mixin extracted from GraphEngine.

Provides persistent embedding training, link prediction, and entity
similarity. Mixed into ``GraphEngine`` so callers see a single object.
"""

from __future__ import annotations

import numpy as np


class EmbeddingsMixin:
    """TransE embedding operations.

    Depends on the host class providing ``self.graph`` (NetworkX) and
    ``self._transE`` (cached TransE instance or None).
    """

    def learn_embeddings(
        self, dim: int = 128, epochs: int = 100, lr: float = 0.01, db=None
    ) -> None:
        """Train TransE embeddings on current graph edges.

        Loads existing embeddings from *db* as warm-start initialization
        (incremental training). If *db* is provided, persists trained
        vectors to the ``embeddings`` table.  Relations are stored with
        the ``__rel__`` prefix to distinguish them from entities.

        Args:
            dim: Embedding dimension.
            epochs: Training epochs.
            lr: Learning rate.
            db: Optional Database instance for persistence and warm-start.
        """
        from drbrain.graph.embedding import TransE
        from drbrain.graph.query_embeddings import RELATION_PREFIX

        init_entities = None
        init_relations = None
        if db:
            raw = db.load_embeddings()
            if raw:
                init_entities = {}
                init_relations = {}
                for key, vec in raw.items():
                    if key.startswith(RELATION_PREFIX):
                        init_relations[key[len(RELATION_PREFIX) :]] = vec
                    else:
                        init_entities[key] = vec

        t = TransE(dim=dim, epochs=epochs, lr=lr)
        t.train(self.graph, init_entities=init_entities, init_relations=init_relations)

        if db:
            for entity, vec in t.entities.items():
                db.save_embedding(entity, vec, dim)
            for rel_name, vec in t.relations.items():
                db.save_embedding(RELATION_PREFIX + rel_name, vec, dim)
            db.commit()

        self._transE = t

    def entity_embedding(self, label: str, db=None) -> np.ndarray | None:
        """Return the TransE embedding vector for *label*.

        Checks the in-memory cache first, then falls back to the
        ``embeddings`` table when *db* is provided.

        Args:
            label: Entity label.
            db: Optional Database instance.

        Returns:
            Float32 numpy array of shape ``(dim,)``, or ``None`` if the
            entity is unknown.
        """
        if self._transE:
            emb = self._transE.entity_embedding(label)
            if emb is not None:
                return emb
        if db:
            row = db.conn.execute(
                "SELECT vec FROM embeddings WHERE entity = ?", (label,)
            ).fetchone()
            if row:
                return np.frombuffer(row[0], dtype=np.float32)
        return None

    def predict_link(
        self, head: str, relation: str, top_k: int = 10, db=None
    ) -> list[tuple[str, float]]:
        """Predict tail entities for *(head, relation)* via TransE scoring.

        Returns the *top_k* entities ranked by ascending TransE distance
        ``||h + r - t||``.  Requires embeddings to be loaded (via
        ``learn_embeddings`` or from *db*).

        Args:
            head: Head entity label.
            relation: Relation name.
            top_k: Number of predictions to return.
            db: Optional Database to load embeddings from.

        Returns:
            List of ``(label, score)`` tuples, or empty list if no
            embeddings are available.
        """
        self._ensure_embeddings(db)
        if self._transE:
            return self._transE.predict_link(head, relation, top_k)
        return []

    def similar_entities(self, label: str, top_k: int = 10, db=None) -> list[tuple[str, float]]:
        """Find entities with similar embedding vectors via cosine similarity.

        Requires embeddings to be loaded (via ``learn_embeddings`` or
        from *db*).

        Args:
            label: Entity label.
            top_k: Number of results to return.
            db: Optional Database to load embeddings from.

        Returns:
            List of ``(label, similarity)`` tuples (higher is more
            similar), or empty list if no embeddings are available.
        """
        self._ensure_embeddings(db)
        if self._transE:
            return self._transE.similar_entities(label, top_k)
        return []

    def _ensure_embeddings(self, db=None) -> None:
        """Load embeddings from *db* into the cache when it is empty."""
        if self._transE is not None:
            return
        if db:
            from drbrain.graph.embedding import TransE
            from drbrain.graph.query_embeddings import RELATION_PREFIX

            raw = db.load_embeddings()
            if raw:
                t = TransE()
                for key, vec in raw.items():
                    if key.startswith(RELATION_PREFIX):
                        t.relations[key[len(RELATION_PREFIX) :]] = vec
                    else:
                        t.entities[key] = vec
                self._transE = t

    def invalidate_embeddings(self) -> None:
        """Clear the in-memory embedding cache.

        Call after modifying the graph or embeddings table so that
        subsequent operations reload fresh data.
        """
        self._transE = None
