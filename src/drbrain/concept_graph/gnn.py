"""Optional GNN link prediction (GraphSAGE-style) for the concept graph.

Implements the paper's GNN model: a 2-layer mean-aggregation GraphSAGE encoder
whose node representations are initialized with topological vectors (degree +
2-path counts over the feature years), followed by an MLP decoder that scores
concept pairs. This is the ``--model gnn`` option for ``drbrain cg predict``.

PyTorch is an OPTIONAL dependency (``pip install drbrain[gnn]``). All torch
imports are lazy so the rest of the concept-graph layer works without it; call
:func:`is_available` / :func:`require_torch` to gate usage.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import TYPE_CHECKING

import numpy as np

from drbrain.concept_graph.dataset import oversample_indices
from drbrain.concept_graph.features import two_path_count, yearly_subgraph

if TYPE_CHECKING:
    from drbrain.storage.database import Database


def is_available() -> bool:
    """Return True when PyTorch is importable."""
    return importlib.util.find_spec("torch") is not None


def require_torch():
    """Import and return torch, raising a helpful error if it is missing."""
    if not is_available():
        raise RuntimeError(
            "GNN link prediction requires PyTorch. Install with `pip install drbrain[gnn]` "
            "or `pip install torch`."
        )
    return importlib.import_module("torch")


def build_node_topo_features(
    db: Database, labels: list[str], years: list[int], *, whitelist: set[str] | None = None
) -> np.ndarray:
    """Per-node topological features: ``[degree, two_path]`` for each year.

    Args:
        db: Database handle.
        labels: Ordered node labels (row order of the returned matrix).
        years: Ascending slice years.
        whitelist: Optional node whitelist to restrict the yearly subgraphs.

    Returns:
        Float matrix of shape ``(len(labels), 2 * len(years))``.
    """
    n = len(labels)
    feat = np.zeros((n, 2 * len(years)), dtype=np.float32)
    idx = {lab: i for i, lab in enumerate(labels)}
    for yi, t in enumerate(years):
        g = yearly_subgraph(db, t, labels=whitelist)
        for lab, i in idx.items():
            if lab in g:
                feat[i, 2 * yi] = float(g.degree(lab, weight="weight"))
                feat[i, 2 * yi + 1] = two_path_count(g, lab)
    # Standardize columns for stable training.
    std = feat.std(axis=0)
    std[std == 0] = 1.0
    return (feat - feat.mean(axis=0)) / std


def build_normalized_adjacency(
    db: Database, labels: list[str], cutoff: int, *, whitelist: set[str] | None = None
):
    """Row-normalized adjacency tensor (with self-loops) over ``labels`` at G_cutoff."""
    torch = require_torch()
    n = len(labels)
    idx = {lab: i for i, lab in enumerate(labels)}
    adj = np.zeros((n, n), dtype=np.float32)
    g = yearly_subgraph(db, cutoff, labels=whitelist)
    for u, v in g.edges():
        if u in idx and v in idx:
            adj[idx[u], idx[v]] = 1.0
            adj[idx[v], idx[u]] = 1.0
    np.fill_diagonal(adj, 1.0)  # self-loop
    row_sum = adj.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    adj_norm = adj / row_sum
    return torch.tensor(adj_norm, dtype=torch.float32)


class GNNLinkClassifier:
    """GraphSAGE encoder + MLP decoder for temporal link prediction.

    Args:
        hidden_dim: GraphSAGE hidden layer width.
        embed_dim: Node embedding dimensionality (encoder output).
        lr: Learning rate.
        epochs: Training epochs.
        pos_fraction: Positive fraction per epoch (oversampling).
        seed: RNG seed.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        embed_dim: int = 32,
        lr: float = 0.01,
        epochs: int = 100,
        pos_fraction: float = 0.3,
        seed: int = 42,
        device: str | None = None,
    ):
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.lr = lr
        self.epochs = epochs
        self.pos_fraction = pos_fraction
        self.seed = seed
        self.device = device
        self._trained = False

    def _build_model(self, torch, in_dim: int):
        encoder = _GraphSAGEEncoder(torch, in_dim, self.hidden_dim, self.embed_dim)
        decoder = torch.nn.Sequential(
            torch.nn.Linear(2 * self.embed_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, 1),
        )
        return encoder, decoder

    def fit(
        self,
        db: Database,
        pairs: list[tuple[str, str]],
        labels: np.ndarray,
        years: list[int],
        cutoff: int,
        *,
        node_labels: set[str] | None = None,
    ) -> GNNLinkClassifier:
        """Train the encoder + decoder on labelled concept pairs.

        Args:
            node_labels: Optional node whitelist. When given, the GNN vocabulary
                is restricted to the whitelist (e.g. a subfield), keeping the
                dense adjacency matrix tractable.
        """
        torch = require_torch()
        torch.manual_seed(self.seed)
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Vocabulary = ALL nodes in the cutoff graph (so message passing sees the
        # full neighbourhood and test pairs over valid nodes get real scores),
        # unioned with any training-pair endpoints. With a subfield whitelist the
        # graph is restricted to those nodes instead.
        cutoff_nodes = set(yearly_subgraph(db, cutoff, labels=node_labels).nodes())
        nodes = sorted(cutoff_nodes | {n for pair in pairs for n in pair})
        self._nodes = nodes
        self._label2idx = {lab: i for i, lab in enumerate(nodes)}
        x_np = build_node_topo_features(db, nodes, years, whitelist=node_labels)
        self._x_np = x_np
        x = torch.tensor(x_np, dtype=torch.float32)
        self._adj = build_normalized_adjacency(db, nodes, cutoff, whitelist=node_labels)

        encoder, decoder = self._build_model(torch, x.shape[1])
        self._encoder, self._decoder = encoder.to(self.device), decoder.to(self.device)
        params = list(encoder.parameters()) + list(decoder.parameters())
        opt = torch.optim.Adam(params, lr=self.lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        y = np.asarray(labels, dtype=np.float32)
        pair_idx = np.array(
            [[self._label2idx[u], self._label2idx[v]] for u, v in pairs], dtype=np.int64
        )
        pair_idx_t = torch.tensor(pair_idx, dtype=torch.long, device=self.device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)

        for epoch in range(self.epochs):
            order = oversample_indices(y, pos_fraction=self.pos_fraction, seed=self.seed + epoch)
            emb = encoder(x, self._adj)
            sel = pair_idx_t[order]
            pair_emb = torch.cat([emb[sel[:, 0]], emb[sel[:, 1]]], dim=1)
            logits = decoder(pair_emb).squeeze(-1)
            loss = loss_fn(logits, y_t[order])
            opt.zero_grad()
            loss.backward()
            opt.step()

        self._trained = True
        return self

    def predict_proba(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Return P(edge) for each pair using the trained encoder + decoder."""
        torch = require_torch()
        if not self._trained:
            raise RuntimeError("GNNLinkClassifier must be fitted before predict_proba.")
        with torch.no_grad():
            x = torch.tensor(self._x_np, dtype=torch.float32, device=self.device)
            emb = self._encoder(x, self._adj)
            idx = [[self._label2idx.get(u, -1), self._label2idx.get(v, -1)] for u, v in pairs]
            out = np.zeros(len(pairs), dtype=np.float32)
            for i, (a, b) in enumerate(idx):
                if a < 0 or b < 0:
                    out[i] = 0.0
                    continue
                pair_emb = torch.cat([emb[a], emb[b]], dim=0)
                out[i] = float(torch.sigmoid(self._decoder(pair_emb).squeeze(-1)))
        return out


class _GraphSAGEEncoder:
    """Factory wrapper so the encoder class is built with the lazy torch module."""

    def __new__(cls, torch, in_dim: int, hidden_dim: int, embed_dim: int):
        class _Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w1 = torch.nn.Linear(2 * in_dim, hidden_dim)
                self.w2 = torch.nn.Linear(2 * hidden_dim, embed_dim)

            def forward(self, x, adj_norm):
                agg1 = adj_norm @ x
                h1 = torch.relu(self.w1(torch.cat([x, agg1], dim=1)))
                agg2 = adj_norm @ h1
                h2 = torch.relu(self.w2(torch.cat([h1, agg2], dim=1)))
                return h2

        return _Encoder()
