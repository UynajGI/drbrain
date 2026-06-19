"""TransE knowledge graph embeddings."""

from __future__ import annotations

import numpy as np


class TransE:
    """TransE: h + r ≈ t in vector space."""

    def __init__(self, dim: int = 128, epochs: int = 100, lr: float = 0.01, margin: float = 1.0):
        """Initialize TransE model.

        Args:
            dim: Embedding dimension (default 128).
            epochs: Training epochs (default 100).
            lr: Learning rate for SGD (default 0.01).
            margin: Hinge loss margin (default 1.0).
        """
        self.dim = dim
        self.epochs = epochs
        self.lr = lr
        self.margin = margin
        self.entities: dict[str, np.ndarray] = {}
        self.relations: dict[str, np.ndarray] = {}
        self._entity_list: list[str] = []

    def train(self, graph, init_entities=None, init_relations=None) -> None:
        """Train TransE embeddings on a NetworkX graph via SGD.

        Args:
            graph: NetworkX graph with edges carrying ``relation`` attribute.
            init_entities: Optional pre-trained entity vectors keyed by label.
            init_relations: Optional pre-trained relation vectors keyed by name.
        """
        edges = []
        entities_set: set[str] = set()
        relations_set: set[str] = set()
        for u, v, data in graph.edges(data=True):
            edges.append((u, data["relation"], v))
            entities_set.add(u)
            entities_set.add(v)
            relations_set.add(data["relation"])
        self._train_on_edges(
            edges,
            entities_set,
            relations_set,
            init_entities=init_entities,
            init_relations=init_relations,
        )

    def train_incremental(
        self,
        graph,
        new_edges: list[tuple[str, str, str]],
        epochs_multiplier: float = 0.3,
        init_entities: dict | None = None,
        init_relations: dict | None = None,
    ) -> None:
        """Continue training TransE only on a subset of (new/changed) edges.

        Unlike train(), this does NOT reinitialize all entities from scratch.
        Existing entity/relation vectors (loaded from disk or passed via
        init_entities/init_relations) are preserved and only nudged by training
        on ``new_edges``. Brand-new entities (not in init) get random init.

        Args:
            graph: NetworkX graph (used only for entity discovery / neighbor lookup).
            new_edges: list of (head, relation, tail) triples to train on.
            epochs_multiplier: scale self.epochs by this factor for incremental
                training (fewer epochs needed since most vectors are already good).
            init_entities: pre-trained entity vectors keyed by label.
            init_relations: pre-trained relation vectors keyed by name.
        """
        if not new_edges:
            return
        # Seed self.entities / self.relations with the prior vectors so that
        # _train_on_edges preserves them (its guard skips already-loaded keys)
        # and so they get persisted by the caller's save loop.
        if init_entities:
            for k, v in init_entities.items():
                self.entities.setdefault(k, np.array(v, dtype=np.float32))
        if init_relations:
            for k, v in init_relations.items():
                self.relations.setdefault(k, np.array(v, dtype=np.float32))
        entities_set: set[str] = set()
        relations_set: set[str] = set()
        edge_list = []
        for h, r, t in new_edges:
            edge_list.append((h, r, t))
            entities_set.add(h)
            entities_set.add(t)
            relations_set.add(r)
        # Use a shortened epoch budget for the incremental pass
        original_epochs = self.epochs
        self.epochs = max(1, int(original_epochs * epochs_multiplier))
        try:
            self._train_on_edges(
                edge_list,
                entities_set,
                relations_set,
                init_entities=init_entities,
                init_relations=init_relations,
            )
        finally:
            self.epochs = original_epochs  # restore for subsequent calls

    def _train_on_edges(
        self,
        edges: list[tuple[str, str, str]],
        entities_set: set[str],
        relations_set: set[str],
        init_entities: dict | None = None,
        init_relations: dict | None = None,
    ) -> None:
        """Shared SGD training loop over a list of (head, relation, tail) edges.

        Seeds known entities/relations from init_* if provided; random-inits
        the rest. The negative-sampling entity pool is the full entity set seen
        in ``edges`` (for incremental training this is a subset, which is fine
        because we are nudging, not learning from scratch).
        """
        if not edges:
            return
        # Keep entities already loaded (e.g. from a previous train_incremental
        # call or via train()) and add any new ones.
        rng = np.random.default_rng(42)
        scale = np.sqrt(6.0 / self.dim)
        for e in entities_set:
            if e in self.entities:
                continue
            if init_entities and e in init_entities:
                self.entities[e] = np.array(init_entities[e], dtype=np.float32)
            else:
                self.entities[e] = rng.uniform(-scale, scale, self.dim).astype(np.float32)
        for r in relations_set:
            if r in self.relations:
                continue
            if init_relations and r in init_relations:
                self.relations[r] = np.array(init_relations[r], dtype=np.float32)
            else:
                self.relations[r] = rng.uniform(-scale, scale, self.dim).astype(np.float32)
                self.relations[r] /= np.linalg.norm(self.relations[r])

        # Entity pool for negative sampling: all known entities so far
        self._entity_list = list(self.entities.keys())
        if len(self._entity_list) < 2:
            return

        for epoch in range(self.epochs):
            total_loss = 0.0
            for h, r, t in edges:
                if h not in self.entities or t not in self.entities or r not in self.relations:
                    continue
                # Negative sampling: pick a random entity != h,t. Guard against
                # infinite loops when the entity pool is tiny.
                if len(self._entity_list) <= 2:
                    candidates = [e for e in self._entity_list if e != t and e != h]
                    neg_t = candidates[0] if candidates else t
                else:
                    neg_t = self._entity_list[rng.integers(len(self._entity_list))]
                    _tries = 0
                    while (neg_t == t or neg_t == h) and _tries < 10:
                        neg_t = self._entity_list[rng.integers(len(self._entity_list))]
                        _tries += 1
                    if neg_t == t or neg_t == h:
                        continue  # give up on this edge, can't sample a clean negative

                h_vec = self.entities[h]
                r_vec = self.relations[r]
                t_vec = self.entities[t]
                n_vec = self.entities[neg_t]

                pos = np.linalg.norm(h_vec + r_vec - t_vec)
                neg = np.linalg.norm(h_vec + r_vec - n_vec)
                loss = float(max(0.0, self.margin + pos - neg))
                if loss <= 0:
                    continue
                total_loss += loss

                grad_h = 2 * (h_vec + r_vec - t_vec)
                grad_t = -2 * (h_vec + r_vec - t_vec)
                grad_n = 2 * (h_vec + r_vec - n_vec)
                self.entities[h] -= self.lr * grad_h
                self.entities[t] -= self.lr * grad_t
                self.entities[neg_t] -= self.lr * grad_n
                self.relations[r] -= self.lr * (grad_h + grad_n) * 0.5

            for e in self.entities:
                n = np.linalg.norm(self.entities[e])
                if n > 0:
                    self.entities[e] /= n

    def entity_embedding(self, label: str) -> np.ndarray | None:
        """Return the embedding vector for an entity label, or None if not found."""
        return self.entities.get(label)

    def relation_embedding(self, rel: str) -> np.ndarray | None:
        """Return the embedding vector for a relation name, or None if not found."""
        return self.relations.get(rel)

    def score(self, head: str, relation: str, tail: str) -> float:
        """Compute TransE score: ‖head + relation - tail‖. Lower is better."""
        h = self.entities.get(head)
        r = self.relations.get(relation)
        t = self.entities.get(tail)
        if h is None or r is None or t is None:
            return float("inf")
        return float(np.linalg.norm(h + r - t))

    def predict_link(self, head: str, relation: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Predict most likely tail entities given a head and relation.

        Returns:
            List of (entity_label, score) sorted by score ascending.
        """
        h = self.entities.get(head)
        r = self.relations.get(relation)
        if h is None or r is None:
            return []
        scores = [
            (e, float(np.linalg.norm(h + r - v))) for e, v in self.entities.items() if e != head
        ]
        scores.sort(key=lambda x: x[1])
        return scores[:top_k]

    def similar_entities(self, label: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Find entities with the most similar embedding vectors via cosine similarity.

        Returns:
            List of (entity_label, cosine_similarity) sorted by similarity descending.
        """
        vec = self.entities.get(label)
        if vec is None:
            return []
        scores = []
        for e, v in self.entities.items():
            if e != label:
                sim = float(np.dot(vec, v) / (np.linalg.norm(vec) * np.linalg.norm(v) + 1e-8))
                scores.append((e, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
