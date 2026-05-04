"""TransE knowledge graph embeddings."""
from __future__ import annotations

import numpy as np


class TransE:
    """TransE: h + r ≈ t in vector space."""

    def __init__(self, dim: int = 128, epochs: int = 100, lr: float = 0.01, margin: float = 1.0):
        self.dim = dim
        self.epochs = epochs
        self.lr = lr
        self.margin = margin
        self.entities: dict[str, np.ndarray] = {}
        self.relations: dict[str, np.ndarray] = {}
        self._entity_list: list[str] = []

    def train(self, graph) -> None:
        edges = []
        entities_set: set[str] = set()
        relations_set: set[str] = set()
        for u, v, data in graph.edges(data=True):
            edges.append((u, data["relation"], v))
            entities_set.add(u)
            entities_set.add(v)
            relations_set.add(data["relation"])
        if not edges:
            return

        self._entity_list = list(entities_set)
        rng = np.random.default_rng(42)
        scale = np.sqrt(6.0 / self.dim)
        for e in entities_set:
            self.entities[e] = rng.uniform(-scale, scale, self.dim).astype(np.float32)
        for r in relations_set:
            self.relations[r] = rng.uniform(-scale, scale, self.dim).astype(np.float32)
            self.relations[r] /= np.linalg.norm(self.relations[r])

        for epoch in range(self.epochs):
            total_loss = 0.0
            for h, r, t in edges:
                if h not in self.entities or t not in self.entities or r not in self.relations:
                    continue
                neg_t = self._entity_list[rng.integers(len(self._entity_list))]
                while neg_t == t:
                    neg_t = self._entity_list[rng.integers(len(self._entity_list))]

                h_vec = self.entities[h]
                r_vec = self.relations[r]
                t_vec = self.entities[t]
                n_vec = self.entities[neg_t]

                pos = np.linalg.norm(h_vec + r_vec - t_vec)
                neg = np.linalg.norm(h_vec + r_vec - n_vec)
                loss = max(0.0, self.margin + pos - neg)
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

    def entity_embedding(self, label: str):
        return self.entities.get(label)

    def relation_embedding(self, rel: str):
        return self.relations.get(rel)

    def score(self, head: str, relation: str, tail: str) -> float:
        h = self.entities.get(head)
        r = self.relations.get(relation)
        t = self.entities.get(tail)
        if h is None or r is None or t is None:
            return float("inf")
        return float(np.linalg.norm(h + r - t))

    def predict_link(self, head: str, relation: str, top_k: int = 10) -> list[tuple[str, float]]:
        h = self.entities.get(head)
        r = self.relations.get(relation)
        if h is None or r is None:
            return []
        scores = [(e, float(np.linalg.norm(h + r - v))) for e, v in self.entities.items() if e != head]
        scores.sort(key=lambda x: x[1])
        return scores[:top_k]

    def similar_entities(self, label: str, top_k: int = 10) -> list[tuple[str, float]]:
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
