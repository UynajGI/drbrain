"""Deterministic retrieval metrics, independent of model judges and dataset generation."""
from __future__ import annotations
from collections.abc import Sequence
from typing import Any


def _node_identity(nws: Any) -> tuple[str, str]:
    """Return ``(paper_id, node_id)`` from a ``NodeWithScore`` metadata."""
    node = getattr(nws, "node", None)
    meta = dict(getattr(node, "metadata", None) or {}) if node is not None else {}
    pid = str(meta.get("paper_id") or "")
    nid = str(meta.get("parent_node_id") or meta.get("node_id") or getattr(node, "node_id", None) or "")
    return pid, nid

def _rank_metrics(nodes: Sequence[Any], item: dict[str, Any], ks: Sequence[int]) -> dict[str, Any]:
    """Paper-level + node-level hit/mrr ranks for one golden query.

    Returns the first relevant rank per level (``None`` when nothing matched)
    plus per-``k`` hit/MRR values:

    * ``hit_rate@k`` (paper) = first relevant *paper* appears within top-k;
    * ``hit_rate@k`` (node) = first relevant *(paper_id, node_id)* within top-k;
    * ``mrr@k`` = 1/first-relevant-rank when the rank is within top-k, else 0.
    """
    rel_papers = {str(p) for p in item.get("relevant_papers") or []}
    rel_nodes = {
        (str(r.get("paper_id") or ""), str(r.get("node_id") or ""))
        for r in item.get("relevant_nodes") or []
    }
    paper_rank: int | None = None
    node_rank: int | None = None
    for i, nws in enumerate(nodes, start=1):
        pid, nid = _node_identity(nws)
        if paper_rank is None and pid in rel_papers:
            paper_rank = i
        if node_rank is None and rel_nodes and (pid, nid) in rel_nodes:
            node_rank = i
        if paper_rank is not None and (node_rank is not None or not rel_nodes):
            break
    ks_sorted = sorted(int(k) for k in ks)

    def _levels(first_rank: int | None) -> dict[str, Any]:
        hits: dict[str, bool] = {}
        mrrs: dict[str, float] = {}
        for k in ks_sorted:
            hits[str(k)] = first_rank is not None and first_rank <= k
            mrrs[str(k)] = (
                round(1.0 / first_rank, 6) if first_rank is not None and first_rank <= k else 0.0
            )
        return {"hit_rate": hits, "mrr": mrrs, "first_rank": first_rank}

    return {
        "query": item.get("query", ""),
        "paper": _levels(paper_rank),
        "node": _levels(node_rank),
    }

def _aggregate_rank(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean hit_rate/MRR over per-query rows, for paper and node levels."""
    ks = sorted({int(k) for row in rows for k in row["paper"]["hit_rate"]})
    out: dict[str, Any] = {"queries": len(rows), "ks": ks, "hit_rate": {}, "mrr": {}}
    for level in ("paper", "node"):
        out["hit_rate"][level] = {
            str(k): round(sum(1 for r in rows if r[level]["hit_rate"].get(str(k))) / len(rows), 4)
            for k in ks
        }
        out["mrr"][level] = {
            str(k): round(sum(r[level]["mrr"].get(str(k), 0.0) for r in rows) / len(rows), 4)
            for k in ks
        }
    return out
