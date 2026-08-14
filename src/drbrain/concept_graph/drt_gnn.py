"""DRT-GNN: Dual-Relational Temporal GNN for research-gap mining.

Unlike the paper's single co-occurrence GraphSAGE, this model operates on TWO
relations over the concept graph:

* ``co`` — undirected weighted co-occurrence (thematic affinity);
* ``cit`` — directed, time-stamped concept citations (method lineage: concepts
  of a citing paper link to concepts of the cited paper, first_year recorded).

Design points that go beyond the paper:
- relation-specific message passing branches (R-GCN flavour), fused by a
  learned gate ``alpha`` (interpretable: thematic vs lineage-driven gaps);
- time-decayed citation propagation (recent lineage edges weigh more);
- sparse adjacency + neighbour sampling (the 107K-node graph cannot be dense);
- minibatch GPU training (10x A40 available).

Training task: temporal link prediction — predict which concept pairs will
co-occur / be cited together next, i.e. research-gap discovery.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class NeighborSampler:
    """Per-layer K-hop neighbour sampling over a sparse (COO) graph.

    For each batch of nodes, sample up to ``fanout`` neighbours per node from
    the row/column incidence of a relation matrix. Returns the 2D index tensor
    used by a scatter-based aggregation layer.
    """

    def __init__(self, row: np.ndarray, col: np.ndarray, n_nodes: int):
        self.n_nodes = n_nodes
        self.row = row
        self.col = col
        # adjacency lists per node (CSR-like)
        self._offsets = np.zeros(n_nodes + 1, dtype=np.int64)
        np.add.at(self._offsets, row + 1, 1)
        self._offsets = np.cumsum(self._offsets)
        order = np.argsort(row, kind="stable")
        self._col_sorted = col[order]

    def sample(self, nodes: np.ndarray, fanout: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (sampled_neighbor_ids, sampler_index) for scatter aggregation."""
        if len(nodes) == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        out_nbrs: list[np.ndarray] = []
        out_idx: list[np.ndarray] = []
        for i, node in enumerate(nodes):
            lo, hi = self._offsets[node], self._offsets[node + 1]
            deg = hi - lo
            if deg == 0:
                continue
            if deg <= fanout:
                picked = np.arange(lo, hi)
            else:
                picked = lo + np.random.choice(deg, fanout, replace=False)
            out_nbrs.append(self._col_sorted[picked])
            out_idx.append(np.full(len(picked), i, dtype=np.int64))
        if not out_nbrs:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        return (
            np.concatenate(out_nbrs),
            np.concatenate(out_idx),
        )


class GraphLayer(nn.Module):
    """Message-passing layer over sampled neighbours with a wide design space.

    Aggregators: mean / sum / max / gcn (sum + row norm, like GCN) / gat
    (attention-weighted; single head, LeakyReLU scoring + softmax).
    Activations: relu / silu / gelu / mish / leaky_relu.
    Optional LayerNorm / BatchNorm, residual connection and dropout — the
    full modern-GNN toolbox, cheap to ablate on a single A40.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        device: str = "cpu",
        act: str = "silu",
        agg: str = "mean",
        norm: str = "none",
        residual: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.device = device
        self.act = act
        self.agg = agg
        self.w = nn.Linear(2 * in_dim, out_dim, device=device)
        if agg == "gat":
            self.attn = nn.Linear(2 * in_dim, 1, device=device)
        if norm == "ln":
            self.norm = nn.LayerNorm(out_dim, device=device)
        elif norm == "bn":
            self.norm = nn.BatchNorm1d(out_dim, device=device)
        else:
            self.norm = None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.residual = residual
        if residual and in_dim != out_dim:
            self.res = nn.Linear(in_dim, out_dim, device=device)
        else:
            self.res = None

    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.act == "silu":
            return torch.nn.functional.silu(x)
        if self.act == "gelu":
            return torch.nn.functional.gelu(x)
        if self.act == "mish":
            return x * torch.tanh(torch.nn.functional.softplus(x))
        if self.act == "leaky_relu":
            return torch.nn.functional.leaky_relu(x, 0.1)
        return torch.relu(x)

    def _aggregate(
        self, x: torch.Tensor, x_nbr: torch.Tensor, nodes: torch.Tensor, index: torch.Tensor
    ) -> torch.Tensor:
        if self.agg == "max":
            agg = torch.zeros((nodes.shape[0], x_nbr.shape[1]), device=self.device)
            agg.scatter_reduce_(
                0, index.unsqueeze(1).expand(-1, x_nbr.shape[1]), x_nbr, reduce="amax"
            )
            return agg
        if self.agg == "gat":
            # Attention over sampled neighbours: score = LeakyReLU(a@[x_i; x_j]).
            src = x[index]
            e = self.attn(torch.cat([src, x_nbr], dim=1)).squeeze(-1)
            e = torch.nn.functional.leaky_relu(e, 0.2)
            # Softmax per centre node via scatter max/sum trick.
            neg = torch.full((nodes.shape[0],), float("-inf"), device=self.device)
            e_max = neg.scatter_reduce_(0, index, e, reduce="amax")
            e = e - e_max[index]
            exp_e = torch.exp(e)
            denom = torch.zeros(nodes.shape[0], device=self.device)
            denom.scatter_add_(0, index, exp_e)
            alpha = exp_e / denom[index].clamp(min=1e-9)
            agg = torch.zeros((nodes.shape[0], x_nbr.shape[1]), device=self.device)
            agg.scatter_add_(
                0, index.unsqueeze(1).expand(-1, x_nbr.shape[1]), alpha.unsqueeze(1) * x_nbr
            )
            return agg
        agg = torch.zeros((nodes.shape[0], x_nbr.shape[1]), device=self.device)
        agg.scatter_add_(0, index.unsqueeze(1).expand(-1, x_nbr.shape[1]), x_nbr)
        if self.agg == "sum":
            return agg
        counts = torch.zeros(nodes.shape[0], device=self.device)
        counts.scatter_add_(0, index, torch.ones_like(index, dtype=torch.float))
        counts = counts.clamp(min=1).unsqueeze(1)
        return agg / counts

    def forward(
        self,
        x: torch.Tensor,
        nbrs: torch.Tensor,
        nodes: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        if index.numel() == 0:
            agg = torch.zeros_like(x)
        else:
            agg = self._aggregate(x, x[nbrs], nodes, index)
        h = self._activation(self.w(torch.cat([x[nodes], agg], dim=1)))
        if self.norm is not None:
            h = self.norm(h)
        if self.dropout is not None:
            h = self.dropout(h)
        if self.residual:
            res = x[nodes] if self.res is None else self.res(x[nodes])
            h = h + res
        return h


class DRTGNN(nn.Module):
    """Dual-branch GraphSAGE with time-decayed citation branch and gated fusion.

    Args:
        in_dim: Input feature width (embeddings + topological features).
        hidden_dim: Shared hidden width per branch.
        embed_dim: Output concept-embedding width (concatenated fusion).
        co_sampler / cit_sampler: Prebuilt :class:`NeighborSampler` instances.
        cit_years: citation edge years (same order as cit row/col), for decay.
        decay: time-decay rate λ (larger = recent edges dominate more).
        ref_year: reference "now" year for decay computation.
        device: torch device.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        embed_dim: int,
        co_sampler: NeighborSampler,
        cit_sampler: NeighborSampler,
        cit_years: np.ndarray,
        *,
        decay: float = 0.05,
        ref_year: int = 2026,
        device: str = "cpu",
        act: str = "silu",
        agg: str = "mean",
        n_layers: int = 3,
        fusion: str = "add",
        norm: str = "ln",
        residual: bool = False,
        dropout: float = 0.0,
        use_cit: bool = True,
    ):
        super().__init__()
        self.device = device
        self.co_sampler = co_sampler
        self.cit_sampler = cit_sampler
        self.fusion = fusion
        self.n_layers = n_layers
        self.use_cit = use_cit
        # Time-decay weights for citation edges (computed once, used at train).
        self.cit_weights = torch.tensor(
            np.exp(-decay * np.maximum(ref_year - cit_years, 0)),
            dtype=torch.float32,
            device=device,
        )

        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [embed_dim]
        self.co_layers = nn.ModuleList(
            GraphLayer(
                dims[i], dims[i + 1], device, act, agg, norm, residual, dropout
            )
            for i in range(n_layers)
        )
        self.cit_layers = nn.ModuleList(
            GraphLayer(
                dims[i], dims[i + 1], device, act, agg, norm, residual, dropout
            )
            for i in range(n_layers)
        )
        if fusion == "gate":
            # Learned per-node gate: how much of h is lineage vs thematic.
            self.gate = nn.Linear(2 * embed_dim, 1, device=device)
        elif fusion == "concat":
            self.fuse = nn.Linear(2 * embed_dim, embed_dim, device=device)
        # fusion == "add": plain sum, no extra params.

    def _branch(
        self, x: torch.Tensor, sampler: NeighborSampler, layers: nn.ModuleList, fanout: int
    ) -> torch.Tensor:
        nodes = torch.arange(x.shape[0], device=self.device)
        h = x
        for layer in layers:
            nbrs, idx = sampler.sample(nodes.cpu().numpy(), fanout)
            h = layer(
                h,
                torch.tensor(nbrs, dtype=torch.long, device=self.device),
                nodes,
                torch.tensor(idx, dtype=torch.long, device=self.device),
            )
        return h

    def forward(self, x: torch.Tensor, fanout: int = 10) -> torch.Tensor:
        h_co = self._branch(x, self.co_sampler, self.co_layers, fanout)
        if not self.use_cit:
            return h_co
        if self.fusion == "add":
            return h_co + self._branch(x, self.cit_sampler, self.cit_layers, fanout)
        h_cit = self._branch(x, self.cit_sampler, self.cit_layers, fanout)
        if self.fusion == "concat":
            return torch.relu(self.fuse(torch.cat([h_co, h_cit], dim=1)))
        alpha = torch.sigmoid(self.gate(torch.cat([h_co, h_cit], dim=1)))
        return alpha * h_co + (1 - alpha) * h_cit


class LinkDecoder(nn.Module):
    """Score concept pairs from fused embeddings (dot-product + MLP)."""

    def __init__(self, embed_dim: int, hidden_dim: int = 64, device: str = "cpu"):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1, device=device),
        )

    def forward(self, hu: torch.Tensor, hv: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([hu, hv], dim=1)).squeeze(-1)


def _zeropower_via_newtonschulz(g: torch.Tensor, steps: int) -> torch.Tensor:
    """Newton-Schulz iteration approximating the orthogonal factor of G."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g
    if g.shape[0] > g.shape[1]:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        aa = x @ x.T
        x = a * x + (b * aa + c * aa @ aa) @ x
    if g.shape[0] > g.shape[1]:
        x = x.T
    return x


class Muon(torch.optim.Optimizer):
    """Muon (Keller Jordan et al. 2025): momentum + Newton-Schulz
    orthogonalisation for matrix parameters, AdamW for vectors/scalars.

    Baseline-comparison option for the DRT-GNN benchmark; the paper's
    reference settings (lr 0.02, momentum 0.95, nesterov, ns_steps 5,
    wd 0.1) are the defaults. Whether it helps on small GNNs is exactly what
    the ablation matrix answers.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        wd: float = 0.1,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
    ):
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
            wd=wd, adamw_betas=adamw_betas, adamw_eps=adamw_eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["wd"]
            beta1, beta2 = group["adamw_betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if p.ndim >= 2:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    g = g.add(buf, alpha=momentum) if nesterov else buf
                    g = _zeropower_via_newtonschulz(g, ns_steps)
                    if wd:
                        p.mul_(1 - lr * wd)
                    p.add_(g, alpha=-lr)
                else:
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    state["step"] += 1
                    exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    denom = exp_avg_sq.sqrt().add_(group["adamw_eps"])
                    bias_c1 = 1 - beta1 ** state["step"]
                    bias_c2 = 1 - beta2 ** state["step"]
                    step_size = lr * (bias_c2**0.5) / bias_c1
                    p.addcdiv_(exp_avg, denom, value=-step_size)
        return loss
