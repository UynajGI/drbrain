"""Feature engineering for temporal link prediction on the concept graph.

Builds time-sliced subgraphs and per-pair feature vectors:

* topological features (node degree + 2-path counts over a window of years),
  mirroring the paper's 20-dim Baseline vector;
* semantic features (concatenated concept embeddings).
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from drbrain.storage.database import Database


def yearly_subgraph(db: Database, t: int, *, labels: set[str] | None = None) -> nx.Graph:
    """Build the undirected concept graph accumulated up to year ``t`` (G_t).

    Args:
        db: Database handle.
        t: Include all co-occurrence edges with ``year <= t``.
        labels: Optional whitelist of node labels (e.g. filtered concept_nodes).

    Returns:
        A weighted undirected NetworkX graph keyed by concept label.
    """
    if labels is not None:
        # Push the whitelist into SQL (uses idx_cooccurrence_src) so only the
        # subfield's rows are scanned instead of the full 22M-edge table.
        ph = ",".join("?" * len(labels))
        rows = db.conn.execute(
            "SELECT src_label, dst_label, SUM(weight) FROM concept_cooccurrence "
            f"WHERE year <= ? AND (src_label IN ({ph}) OR dst_label IN ({ph})) "
            "GROUP BY src_label, dst_label",
            (t, *labels, *labels),
        ).fetchall()
    else:
        rows = db.conn.execute(
            "SELECT src_label, dst_label, SUM(weight) FROM concept_cooccurrence "
            "WHERE year <= ? GROUP BY src_label, dst_label",
            (t,),
        ).fetchall()
    g = nx.Graph()
    for src, dst, weight in rows:
        if labels is not None and (src not in labels or dst not in labels):
            continue
        if g.has_edge(src, dst):
            g[src][dst]["weight"] += weight or 1.0
        else:
            g.add_edge(src, dst, weight=weight or 1.0)
    return g


def two_path_count(g: nx.Graph, node: str) -> float:
    """Weighted count of length-2 walks starting at ``node`` (row-sum of A^2).

    For weighted edges this is ``sum_k A[node, k] * weighted_degree(k)`` — the
    incoming edge weight ``A[node, k]`` must be included so a multi-paper
    co-occurrence contributes proportionally.
    """
    if node not in g:
        return 0.0
    total = 0.0
    for neighbor in g.neighbors(node):
        w = g[node][neighbor].get("weight", 1.0)
        total += w * g.degree(neighbor, weight="weight")
    return float(total)


def precompute_yearly_subgraphs(
    db: Database, years: list[int], *, labels: set[str] | None = None
) -> dict[int, nx.Graph]:
    """Build each year's accumulated subgraph once, for reuse across many pairs."""
    return {t: yearly_subgraph(db, t, labels=labels) for t in years}


def topo_features(
    db: Database,
    u: str,
    v: str,
    years: list[int],
    graphs: dict[int, nx.Graph] | None = None,
) -> np.ndarray:
    """Topological feature vector for a node pair across ``years``.

    For each year, computes ``[degree(u), degree(v), two_path(u), two_path(v)]``
    on the accumulated subgraph G_t. With 5 years this yields the paper's 20-dim
    Baseline vector. Pass prebuilt ``graphs`` (see
    :func:`precompute_yearly_subgraphs`) to avoid rebuilding G_t per pair.

    Args:
        db: Database handle.
        u: First concept label.
        v: Second concept label.
        years: Ascending list of slice years.
        graphs: Optional prebuilt ``{year: subgraph}`` cache.

    Returns:
        A 1-D float vector of length ``4 * len(years)``.
    """
    feats: list[float] = []
    for t in years:
        g = graphs[t] if graphs is not None else yearly_subgraph(db, t)
        feats.append(float(g.degree(u, weight="weight")) if u in g else 0.0)
        feats.append(float(g.degree(v, weight="weight")) if v in g else 0.0)
        feats.append(two_path_count(g, u))
        feats.append(two_path_count(g, v))
    return np.asarray(feats, dtype=np.float32)


def semantic_features(
    u: str, v: str, embeddings: dict[str, np.ndarray], dim: int | None = None
) -> np.ndarray:
    """Concatenate the embeddings of two concepts (zero-fill if missing).

    Args:
        u: First concept label.
        v: Second concept label.
        embeddings: ``{label: vector}`` mapping.
        dim: Embedding dimensionality (inferred when omitted).

    Returns:
        Concatenated vector of length ``2 * dim``.
    """
    if dim is None:
        dim = next(iter(embeddings.values())).shape[0] if embeddings else 0
    zero = np.zeros(dim, dtype=np.float32)
    eu = embeddings.get(u, zero)
    ev = embeddings.get(v, zero)
    return np.concatenate([eu, ev]).astype(np.float32)
