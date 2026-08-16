#!/usr/bin/env python3
"""A/B comparison of rerank on/off over a query set (Ticket T8).

Runs the same queries through the T4 fusion retriever with and without the
T8 rerank postprocessor and reports how the ranking changes.

Two modes:

* ``perturb`` (default when no golden relevance is provided): ranking
  perturbation statistics — top-1 change, top-5 overlap (Jaccard), Kendall
  tau, mean absolute rank displacement, per query plus a final mean row.
* ``mrr`` (auto when the query file carries relevance — T7 golden set entries
  use ``relevant_nodes`` [{paper_id, node_id}] and/or ``relevant_papers``):
  MRR@k and HitRate@k for coarse (rerank off) vs reranked (rerank on).

The reranker defaults to ``llamaindex.rerank_model`` (production:
``Qwen/Qwen3-Reranker-0.6B``, ~1.2 GB, downloaded on first use). Pass
``--rerank-model mock`` to use a built-in deterministic lexical-overlap
scorer that runs fully offline — useful to exercise the pipeline without a
model; it is *not* a real reranker and its numbers are only illustrative.
When the configured model cannot be loaded the tool exits 3 with a clear
message (the query chain itself degrades to Noop, but an A/B needs both
rankings to compare).

Usage::

    uv run python scripts/rerank_ab.py \\  # needs an index at cfg storage_dir
    uv run python scripts/rerank_ab.py --storage-dir test-run/data/llamaindex \\
        --rerank-model mock --queries "perovskite|grain boundary|band gap"
    uv run python scripts/rerank_ab.py --query-file data/llamaindex/golden.jsonl \\
        --rerank-model mock   # MRR/HitRate table if golden entries have relevance

Exit codes: 0 = ok, 2 = no index/legs, 3 = reranker unavailable.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from drbrain.config import Config, load_config
from drbrain.rag.config import get_llamaindex_config
from drbrain.rag.rerank import (
    CrossEncoderReranker,
    RerankPostprocessor,
    build_reranker,
    kendall_tau,
    mean_rank_displacement,
    top_k_overlap,
)

log = logging.getLogger("rerank_ab")

#: Sample queries used when neither --query-file nor --queries is given.
DEFAULT_QUERIES = [
    "synthesis of perovskite solar cells",
    "grain boundary effects on mechanical properties",
    "band gap engineering in transition metal oxides",
    "electrochemical energy storage mechanisms",
    "thermal conductivity of polymer composites",
]


class _LexicalReranker:
    """Deterministic offline stand-in: scores a passage by query-token overlap.

    ``--rerank-model mock`` selects this. Purely illustrative for exercising
    the A/B pipeline without downloading a model — never a production reranker.
    """

    name = "mock-lexical"

    @property
    def available(self) -> bool:
        return True

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        q_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        out = []
        for p in passages:
            p_tokens = set(re.findall(r"[a-z0-9]+", p.lower()))
            out.append(float(len(q_tokens & p_tokens)))
        return out


# ── metric helpers ───────────────────────────────────────────────────────────


def _mrr_hitrate(ranked_ids: list[str], relevant, k: int) -> tuple[float, float]:
    """(MRR@k, HitRate@k) for one ranking against a :func:`_relevant_set` result."""
    hits = [_is_relevant(nid, relevant) for nid in ranked_ids[:k]]
    rr = 0.0
    for i, h in enumerate(hits, start=1):
        if h:
            rr = 1.0 / i
            break
    return rr, (1.0 if any(hits) else 0.0)


def _relevant_set(entry: dict) -> tuple[set[str], set[str]]:
    """Relevance from a golden entry → (node composites, paper ids).

    ``relevant_nodes`` entries ``{paper_id, node_id}`` map to the fusion
    layer's composite ``paper_id:node_id`` node ids; ``relevant_papers`` are
    matched at paper granularity. A node counts as relevant when either its
    composite id or its ``paper_id:`` prefix is in the sets.
    """
    nodes = set()
    for rn in entry.get("relevant_nodes") or []:
        pid = str(rn.get("paper_id") or "")
        nid = str(rn.get("node_id") or "")
        if pid and nid:
            nodes.add(f"{pid}:{nid}")
    papers = {str(pid) for pid in entry.get("relevant_papers") or []}
    return nodes, papers


def _is_relevant(node_id: str, relevant: tuple[set[str], set[str]]) -> bool:
    nodes, papers = relevant
    if node_id in nodes:
        return True
    return node_id.split(":", 1)[0] in papers


# ── main ─────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rerank on/off A/B comparison (T8)")
    p.add_argument("--config", help="config.yaml path (default: load_config from CWD)")
    p.add_argument("--storage-dir", help="override llamaindex.storage_dir (index location)")
    p.add_argument("--db", help="optional db path for tree/graph legs")
    p.add_argument(
        "--query-file", help="JSONL: one {query, [relevant_ids], [relevant_papers]} per line"
    )
    p.add_argument("--queries", help="semicolon-separated inline queries")
    p.add_argument("--rerank-model", help="reranker model id; 'mock' = offline lexical scorer")
    p.add_argument("--rerank-top-k", type=int, default=None, help="candidates fed to the reranker")
    p.add_argument(
        "--top-k", type=int, default=10, help="head size for metrics (MRR/HitRate@k, top-k overlap)"
    )
    p.add_argument("--mode", choices=["auto", "perturb", "mrr"], default="auto")
    p.add_argument("--split", default=None, help="filter golden queries by split (dev/val/test)")
    p.add_argument("--limit", type=int, default=None, help="max queries to process")
    p.add_argument("--json", action="store_true", help="emit JSON instead of the text table")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    cfg: Config = (
        Config.from_yaml(args.config, local_path=Path(args.config).parent / "config.local.yaml")
        if args.config
        else load_config()
    )
    li = get_llamaindex_config(cfg)
    if args.storage_dir:
        li.storage_dir = args.storage_dir
    if li.rerank_top_k and not args.rerank_top_k:
        args.rerank_top_k = li.rerank_top_k
    rerank_top_k = int(args.rerank_top_k or 20)
    if rerank_top_k < 2:
        print("--rerank-top-k must be >= 2", file=sys.stderr)
        return 2

    try:
        from llama_index.core.schema import QueryBundle

        from drbrain.rag.fusion import build_fusion_retriever, get_retrievers
    except ImportError as exc:  # pragma: no cover
        print(f"llama-index missing ({exc}); cannot run A/B", file=sys.stderr)
        return 2

    db = None
    if args.db:
        from drbrain.storage.database import Database

        db = Database(Path(args.db))
    legs = get_retrievers(cfg, db)
    if not legs:
        print(
            "no retrieval legs found — build an index first "
            f"(storage_dir={li.storage_dir!r}; see 'drbrain rag index')",
            file=sys.stderr,
        )
        return 2
    vector = legs.pop("vector", None)
    bm25 = legs.pop("bm25", None)
    fusion = build_fusion_retriever(
        cfg, vector_index=vector, bm25_retriever=bm25, custom_retrievers=legs, top_k=rerank_top_k
    )
    if fusion is None:
        print("fusion retriever could not be built (no legs)", file=sys.stderr)
        return 2

    model = (args.rerank_model or li.rerank_model or "").strip()
    if model == "mock":
        reranker = _LexicalReranker()
        print("reranker: mock-lexical (offline heuristic, illustrative only)")
    else:
        reranker = build_reranker(cfg) if not args.rerank_model else CrossEncoderReranker(model)
        if reranker is None:
            print("reranker model unset in config; pass --rerank-model", file=sys.stderr)
            return 3
    if not getattr(reranker, "available", False):
        print(
            f"reranker {getattr(reranker, 'model_name', model)!r} unavailable — "
            "cannot compare rerank on/off (query chain would degrade to Noop)",
            file=sys.stderr,
        )
        return 3
    print(
        f"reranker: {getattr(reranker, 'name', getattr(reranker, 'model_name', model))} | "
        f"candidates/query: {rerank_top_k}"
    )

    pp = RerankPostprocessor(top_k=rerank_top_k, reranker=reranker)
    queries = _load_queries(args)
    if args.split:
        queries = [e for e in queries if str(e.get("split", "")) == args.split]
        if not queries:
            print(f"--split {args.split!r} matched no queries", file=sys.stderr)
            return 2
    if args.limit:
        queries = queries[: args.limit]
    if not queries:
        print("no queries (give --query-file or --queries)", file=sys.stderr)
        return 2

    top_k = int(args.top_k or 10)
    mode = args.mode
    if mode == "auto":
        mode = "mrr" if any(any(x) for x in (_relevant_set(e) for e in queries)) else "perturb"

    records = []
    for i, entry in enumerate(queries, start=1):
        query = str(entry["query"])
        qb = QueryBundle(query)
        coarse = fusion.retrieve(qb)
        coarse_ids = [nws.node.node_id for nws in coarse]
        reranked = pp.postprocess_nodes(coarse, query_bundle=qb)
        reranked_ids = [nws.node.node_id for nws in reranked]
        relevant = _relevant_set(entry)
        records.append(
            {
                "query": query,
                "n_candidates": len(coarse_ids),
                "coarse_ids": coarse_ids,
                "reranked_ids": reranked_ids,
                "relevant": sorted(relevant[0] | relevant[1]),
                "top1_changed": (coarse_ids[:1] != reranked_ids[:1]),
                "top_k_overlap": top_k_overlap(coarse_ids, reranked_ids, top_k),
                "tau": kendall_tau(coarse_ids, reranked_ids),
                "mean_disp": mean_rank_displacement(coarse_ids, reranked_ids),
                "coarse_mrr": None,
                "coarse_hit": None,
                "rerank_mrr": None,
                "rerank_hit": None,
            }
        )
        if mode == "mrr" and (relevant[0] or relevant[1]):
            records[-1]["coarse_mrr"], records[-1]["coarse_hit"] = _mrr_hitrate(
                coarse_ids, relevant, top_k
            )
            records[-1]["rerank_mrr"], records[-1]["rerank_hit"] = _mrr_hitrate(
                reranked_ids, relevant, top_k
            )

    if args.json:
        print(
            json.dumps(
                {"mode": mode, "top_k": top_k, "queries": records}, ensure_ascii=False, indent=2
            )
        )
    else:
        _print_table(records, mode, top_k)
    return 0


def _load_queries(args) -> list[dict]:
    """Query list from --query-file JSONL, --queries, or the built-in sample."""
    if args.query_file:
        path = Path(args.query_file)
        if not path.exists():
            print(f"--query-file {path} not found", file=sys.stderr)
            sys.exit(2)
        entries = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries
    if args.queries:
        return [{"query": q.strip()} for q in args.queries.split("|") if q.strip()]
    return [{"query": q} for q in DEFAULT_QUERIES]


def _short(nid: str, width: int = 32) -> str:
    return nid if len(nid) <= width else nid[: width - 1] + "…"


def _print_table(records: list[dict], mode: str, top_k: int) -> None:
    hdr = (
        f"{'#':>3} {'query':<28} {'n':>3} {'top1Δ':>6} "
        f"{'top-k o' if mode == 'perturb' else f'HR@{top_k}':>7} "
        f"{'τ' if mode == 'perturb' else f'MRR@{top_k}':>7} "
        f"{'mean|Δr|' if mode == 'perturb' else 'gain':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(records, start=1):
        q = r["query"]
        if len(q) > 28:
            q = q[:27] + "…"
        if mode == "perturb":
            right = f"{r['top_k_overlap']:.2f} {r['tau']:+.2f} {r['mean_disp']:.2f}"
        else:
            coarse = r["coarse_mrr"] or 0.0
            fine = r["rerank_mrr"] or 0.0
            right = f"{r['rerank_hit']:.2f} {fine:.3f} {fine - coarse:+.3f}"
        print(
            f"{i:>3} {q:<28} {r['n_candidates']:>3} "
            f"{'YES' if r['top1_changed'] else 'no':>6} {right:>7}"
        )
    print("-" * len(hdr))
    n = len(records)
    if mode == "perturb":
        agg = {
            "top1 changed": sum(1 for r in records if r["top1_changed"]) / n,
            f"top-{top_k} overlap": sum(r["top_k_overlap"] for r in records) / n,
            "kendall tau": sum(r["tau"] for r in records) / n,
            "mean |Δrank|": sum(r["mean_disp"] for r in records) / n,
        }
    else:
        with_rel = [r for r in records if r["coarse_mrr"] is not None]
        agg = {
            f"MRR@{top_k} rerank off": sum(r["coarse_mrr"] or 0.0 for r in with_rel)
            / max(1, len(with_rel)),
            f"MRR@{top_k} rerank on": sum(r["rerank_mrr"] or 0.0 for r in with_rel)
            / max(1, len(with_rel)),
            f"HR@{top_k} rerank off": sum(r["coarse_hit"] or 0.0 for r in with_rel)
            / max(1, len(with_rel)),
            f"HR@{top_k} rerank on": sum(r["rerank_hit"] or 0.0 for r in with_rel)
            / max(1, len(with_rel)),
        }
    for k, v in agg.items():
        print(f"{k:<24} {v:.4f}")
    print(f"\n(n={n} queries, mode={mode})")


if __name__ == "__main__":
    raise SystemExit(main())
