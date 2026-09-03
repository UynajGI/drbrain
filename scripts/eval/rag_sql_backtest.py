#!/usr/bin/env python3
"""SQL-native RAG backtest over the golden set (production-engine metrics).

The legacy ``drbrain rag eval`` runs the LlamaIndex parallel index; this script
evaluates the active SQL engine (``retrieve_documents_sql`` over
``drbrain_rag.db``) instead. Pipeline:

1. load ``data/llamaindex/golden.jsonl`` (split filter, default ``dev``);
2. map DOI labels to library local ids via ``paper_ids`` (slash/underscore
   dual forms), keep queries whose labels exist in ``node_texts``;
3. run ``retrieve_documents_sql`` per query under each ablation config
   (retriever legs x rerank on/off) on a mutable Config instance;
4. compute paper- and node-level HitRate@k / MRR@k plus per-query
   diagnostics, and write JSON + markdown artifacts.

Usage:
    .venv/bin/python scripts/eval/rag_sql_backtest.py [--split dev] [--k 10]
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

GOLDEN = ROOT / "data/llamaindex/golden.jsonl"
OUT_DIR = ROOT / "research/semifinal/evidence/rag-backtest"

# (name, retrievers, rerank) — production = four legs + rerank
CONFIGS = [
    ("bm25-only", ["bm25"], False),
    ("bm25+vector", ["bm25", "vector"], False),
    ("bm25+vector+raptor", ["bm25", "vector", "raptor"], False),
    ("bm25+vector+graph", ["bm25", "vector", "graph"], False),
    ("four-leg", ["bm25", "vector", "raptor", "graph"], False),
    ("four-leg+rerank (production)", ["bm25", "vector", "raptor", "graph"], True),
]


def load_golden(split: str) -> list[dict]:
    rows = []
    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("split") == split:
            rows.append(rec)
    return rows


def map_dois(con_main, dois: list[str]) -> set[str]:
    """Map DOI labels to library local ids (slash/underscore dual forms)."""
    ids: set[str] = set()
    for doi in dois or []:
        v = str(doi).strip()
        if not v:
            continue
        cands = {v, v.replace("/", "_"), v.replace("_", "/")}
        ph = ",".join("?" * len(cands))
        for (local_id,) in con_main.execute(
            f"SELECT local_id FROM paper_ids WHERE doi IN ({ph})", tuple(cands)
        ):
            if local_id:
                ids.add(local_id)
    return ids


def metrics(ranked_papers: list[str], relevant: set[str], k: int) -> tuple[float, float]:
    hit = 0.0
    mrr = 0.0
    seen: set[str] = set()
    for rank, pid in enumerate(ranked_papers[:k], start=1):
        if pid in seen:
            continue
        seen.add(pid)
        if pid in relevant:
            hit = 1.0
            mrr = 1.0 / rank
            break
    return hit, mrr


def node_metrics(rows: list[dict], relevant_nodes: list[dict], mapped: set[str], k: int) -> float:
    """Node-level hit@k: any retrieved (paper, node) in the mapped label set."""
    want: set[tuple[str, str]] = set()
    for n in relevant_nodes or []:
        for m in mapped:
            want.add((m, str(n.get("node_id"))))
    if not want:
        return 0.0
    for r in rows[:k]:
        if (str(r.get("paper_id")), str(r.get("node_id"))) in want:
            return 1.0
    return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    import sqlite3

    from drbrain.config import load_config
    from drbrain.graph.engine import GraphEngine
    from drbrain.rag.sql_retrie import _generation_id, _open, retrieve_documents_sql
    from drbrain.storage.database import Database

    cfg = load_config()
    db = Database(cfg.db.path)
    graph = GraphEngine()
    con_main = sqlite3.connect(f"file:{Path(cfg.db.path)}?mode=ro", uri=True)
    conn_rag = _open(cfg)

    n_nodes = conn_rag.execute("SELECT COUNT(*) FROM node_texts").fetchone()[0]
    n_papers = conn_rag.execute("SELECT COUNT(DISTINCT paper_id) FROM node_texts").fetchone()[0]
    generation = _generation_id(conn_rag)

    golden = load_golden(args.split)
    cases = []
    for rec in golden:
        mapped = map_dois(con_main, rec.get("relevant_papers") or [])
        in_store = set()
        for mid in mapped:
            if conn_rag.execute(
                "SELECT 1 FROM node_texts WHERE paper_id=? LIMIT 1", (mid,)
            ).fetchone():
                in_store.add(mid)
        if in_store:
            cases.append({"rec": rec, "mapped": in_store})

    print(
        f"golden {args.split}: {len(golden)} queries, evaluable {len(cases)} "
        f"(node_texts {n_nodes} rows / {n_papers} papers, generation {generation})",
        flush=True,
    )

    results: dict[str, list] = {}
    summary: list[dict] = []
    for name, retrievers, rerank in CONFIGS:
        cfg.llamaindex.retrievers = list(retrievers)
        cfg.llamaindex.rerank = rerank
        per_query = []
        t0 = time.perf_counter()
        for case in cases:
            q = case["rec"]["query"]
            tq = time.perf_counter()
            rows = retrieve_documents_sql(cfg, db, q, top_k=args.k, graph=graph) or []
            dt = time.perf_counter() - tq
            papers = [str(r.get("paper_id")) for r in rows]
            hp5, mr5 = metrics(papers, case["mapped"], 5)
            hp10, mr10 = metrics(papers, case["mapped"], args.k)
            nh10 = node_metrics(rows, case["rec"].get("relevant_nodes"), case["mapped"], args.k)
            per_query.append(
                {
                    "query": q[:80],
                    "mapped_labels": sorted(case["mapped"]),
                    "hit@5": hp5,
                    "mrr@5": mr5,
                    f"hit@{args.k}": hp10,
                    f"mrr@{args.k}": mr10,
                    f"node_hit@{args.k}": nh10,
                    "latency_s": round(dt, 2),
                    "top_papers": papers[: args.k],
                    "legs": sorted({leg for r in rows for leg in (r.get("legs") or [])}),
                }
            )
            print(f"  [{name}] {q[:40]!r} hit@10={hp10} mrr@10={mr10:.3f} {dt:.1f}s", flush=True)
        dt_total = time.perf_counter() - t0
        n = len(per_query)
        agg = {
            "config": name,
            "retrievers": retrievers,
            "rerank": rerank,
            "n": n,
            "hit@5": round(sum(q["hit@5"] for q in per_query) / n, 4),
            "mrr@5": round(sum(q["mrr@5"] for q in per_query) / n, 4),
            f"hit@{args.k}": round(sum(q[f"hit@{args.k}"] for q in per_query) / n, 4),
            f"mrr@{args.k}": round(sum(q[f"mrr@{args.k}"] for q in per_query) / n, 4),
            f"node_hit@{args.k}": round(sum(q[f"node_hit@{args.k}"] for q in per_query) / n, 4),
            "latency_s_avg": round(dt_total / n, 2),
        }
        summary.append(agg)
        results[name] = per_query
        print(f"[{name}] {agg}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "split": args.split,
        "k": args.k,
        "index": {"node_texts": n_nodes, "papers": n_papers, "generation": generation},
        "golden_total": len(golden),
        "evaluable": len(cases),
        "summary": summary,
        "per_query": results,
    }
    (OUT_DIR / f"rag-sql-backtest-{stamp}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        f"# SQL 引擎检索回测（{stamp}）",
        "",
        f"- 口径：golden {args.split} {len(golden)} 条，可评测 {len(cases)} 条（DOI→local_id 映射后在 node_texts 全文库内）",
        f"- 索引：node_texts {n_nodes:,} 行 / {n_papers:,} 篇，generation `{generation}`",
        "- 引擎：retrieve_documents_sql（两阶段：BM25 召回池内重排；RRF k=60；rerank_top_k=20）",
        "",
        "| 配置 | HR@5 | MRR@5 | HR@10 | MRR@10 | node-HR@10 | 均延迟/s |",
        "|---|---|---|---|---|---|---|",
    ]
    for a in summary:
        lines.append(
            f"| {a['config']} | {a['hit@5']} | {a['mrr@5']} | {a[f'hit@{args.k}']} | "
            f"{a[f'mrr@{args.k}']} | {a[f'node_hit@{args.k}']} | {a['latency_s_avg']} |"
        )
    (OUT_DIR / f"rag-sql-backtest-{stamp}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
