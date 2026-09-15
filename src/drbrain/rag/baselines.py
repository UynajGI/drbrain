"""Evaluation-only baseline adapters (plan T56).

Each baseline is a standalone comparison algorithm registered *only* for the
evaluation entry: BM25+vector fusion, the real PageIndex iterative tree
search (model-guided over ``tree.json``; never SQL LIKE), RAPTOR's collapsed
tree (flat all-layer ANN, no expansion) and a simple concatenation arm.
Baselines share the unified model/context caps, record the algorithm they
actually ran plus cost counters, and never become production routes.

Dependencies are injectable so tests exercise the algorithms without a
database, a vector store or a model.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

#: Registered evaluation baselines → the real algorithm they run.
BASELINES: dict[str, str] = {
    "bm25_vector": "BM25 + vector (RRF fusion of both legs)",
    "pageindex": "PageIndex iterative tree search (model-guided over tree.json; no SQL LIKE)",
    "raptor_collapsed": "RAPTOR collapsed tree (flat all-layer ANN, no expansion)",
    "concat": "simple concatenation (BM25 top-k then vector top-k, deduplicated)",
}

_SQL_SEARCH = Callable[..., list[dict[str, Any]]]
_TREE_SEARCH = Callable[[str, int], list[dict[str, Any]]]
_PAGEINDEX_SEARCH = Callable[[str, Sequence[str], int], list[dict[str, Any]]]


def baseline_algorithm(name: str) -> str:
    """The documented algorithm of a registered baseline."""
    key = str(name).strip().lower()
    if key not in BASELINES:
        raise ValueError(f"unknown baseline {name!r}; expected one of {sorted(BASELINES)}")
    return BASELINES[key]


@dataclass
class BaselineOutcome:
    name: str
    algorithm: str
    status: str = "ok"  # "ok" | "empty" | "unavailable"
    keys: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "baseline": self.name,
            "algorithm": self.algorithm,
            "status": self.status,
            "keys": list(self.keys),
            "details": self.details,
            "notes": list(self.notes),
        }


def _key_of(row: dict[str, Any]) -> str:
    paper = str(row.get("paper_id") or "")
    node = str(row.get("node_id") or "")
    if not node and row.get("evidence_id"):
        return str(row["evidence_id"])
    return f"{paper}:{node}" if paper and node else str(row.get("key") or "")


def _dedup(keys: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for key in keys:
        text = str(key)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


@dataclass
class BaselineRunner:
    """Runs one registered baseline with injectable algorithm dependencies."""

    cfg: Any = None
    db: Any = None
    sql_search: _SQL_SEARCH | None = None
    tree_search: _TREE_SEARCH | None = None
    pageindex_search: _PAGEINDEX_SEARCH | None = None

    # ── algorithms ──────────────────────────────────────────────────────

    def _run_sql(self, legs: Sequence[str], query: str, top_k: int) -> tuple[list[dict], int]:
        search = self.sql_search or _default_sql_search(self.cfg, self.db)
        started = time.perf_counter()
        rows = list(search(list(legs), query, top_k) or [])
        return rows, int((time.perf_counter() - started) * 1000)

    def _run_sql_soft(
        self, legs: Sequence[str], query: str, top_k: int
    ) -> tuple[list[dict], str | None, int]:
        """One leg of a concatenation arm; a failure degrades, not aborts."""
        try:
            rows, elapsed = self._run_sql(legs, query, top_k)
            return rows, None, elapsed
        except Exception as exc:  # noqa: BLE001 - the arm reports the failure
            logger.info("[rag] concat leg {} unavailable: {}", "+".join(legs), exc)
            return [], f"{type(exc).__name__}: {exc}", 0

    def run(
        self,
        name: str,
        query: str,
        *,
        top_k: int = 10,
        papers: Sequence[str] | None = None,
    ) -> BaselineOutcome:
        key = str(name).strip().lower()
        algorithm = baseline_algorithm(key)
        try:
            if key == "bm25_vector":
                return self._bm25_vector(query, top_k, algorithm)
            if key == "concat":
                return self._concat(query, top_k, algorithm)
            if key == "raptor_collapsed":
                return self._raptor_collapsed(query, top_k, algorithm)
            return self._pageindex(query, top_k, papers, algorithm)
        except Exception as exc:  # noqa: BLE001 - a baseline failure is a reported state
            logger.warning("[rag] baseline {} unavailable: {}", key, exc)
            return BaselineOutcome(
                name=key,
                algorithm=algorithm,
                status="unavailable",
                details={"error": f"{type(exc).__name__}: {exc}"},
            )

    def _bm25_vector(self, query: str, top_k: int, algorithm: str) -> BaselineOutcome:
        rows, elapsed = self._run_sql(["bm25", "vector"], query, top_k)
        keys = _dedup([_key_of(row) for row in rows])[:top_k]
        return BaselineOutcome(
            name="bm25_vector",
            algorithm=algorithm,
            status="ok" if keys else "empty",
            keys=tuple(keys),
            details={"top_k": top_k, "candidates": len(rows), "elapsed_ms": elapsed},
        )

    def _concat(self, query: str, top_k: int, algorithm: str) -> BaselineOutcome:
        bm25_rows, bm25_error, ms_bm25 = self._run_sql_soft(["bm25"], query, top_k)
        vector_rows, vector_error, ms_vector = self._run_sql_soft(["vector"], query, top_k)
        keys = _dedup([_key_of(row) for row in bm25_rows] + [_key_of(row) for row in vector_rows])
        notes = tuple(
            note
            for note in (
                f"bm25 unavailable: {bm25_error}" if bm25_error else "",
                f"vector unavailable: {vector_error}" if vector_error else "",
            )
            if note
        )
        if keys:
            status = "ok"
        elif bm25_error and vector_error:
            status = "unavailable"
        else:
            status = "empty"
        return BaselineOutcome(
            name="concat",
            algorithm=algorithm,
            status=status,
            keys=tuple(keys[: top_k * 2]),
            details={
                "top_k": top_k,
                "bm25": len(bm25_rows),
                "vector": len(vector_rows),
                "elapsed_ms": ms_bm25 + ms_vector,
            },
            notes=notes,
        )

    def _raptor_collapsed(self, query: str, top_k: int, algorithm: str) -> BaselineOutcome:
        search = self.tree_search or _default_tree_search(self.cfg, self.db)
        started = time.perf_counter()
        candidates = list(search(query, top_k) or [])
        elapsed = int((time.perf_counter() - started) * 1000)
        keys = _dedup(
            [
                f"{item.get('local_id', '')}:{item.get('node_id', '')}"
                if item.get("local_id")
                else str(item.get("node_id") or "")
                for item in candidates
            ]
        )[:top_k]
        return BaselineOutcome(
            name="raptor_collapsed",
            algorithm=algorithm,
            status="ok" if keys else "empty",
            keys=tuple(keys),
            details={
                "top_k": top_k,
                "candidates": len(candidates),
                "view": "all",
                "elapsed_ms": elapsed,
            },
        )

    def _pageindex(
        self,
        query: str,
        top_k: int,
        papers: Sequence[str] | None,
        algorithm: str,
    ) -> BaselineOutcome:
        search = self.pageindex_search or _default_pageindex_search(self.cfg, self.db)
        started = time.perf_counter()
        results = list(search(query, list(papers or []), top_k) or [])
        elapsed = int((time.perf_counter() - started) * 1000)
        keys = _dedup(
            [
                f"{item.get('paper_id', '')}:{item.get('node_id', '')}"
                for item in results
                if item.get("node_id")
            ]
        )[:top_k]
        return BaselineOutcome(
            name="pageindex",
            algorithm=algorithm,
            status="ok" if keys else "empty",
            keys=tuple(keys),
            details={
                "top_k": top_k,
                "papers": len(papers or []),
                "candidates": len(results),
                "elapsed_ms": elapsed,
                "uses_sql_like": False,
            },
        )


# ── default wiring ───────────────────────────────────────────────────────────


def _default_sql_search(cfg: Any, db: Any) -> _SQL_SEARCH:
    from drbrain.rag.config import get_llamaindex_config
    from drbrain.rag.sql_retrie import retrieve_documents_sql

    def search(legs: Sequence[str], query: str, top_k: int) -> list[dict[str, Any]]:
        li = get_llamaindex_config(cfg)
        route_cfg = cfg
        if list(getattr(li, "retrievers", []) or []) != list(legs):
            route_cfg = dataclasses.replace(
                cfg, llamaindex=dataclasses.replace(li, retrievers=list(legs))
            )
        rows = retrieve_documents_sql(route_cfg, db, query, top_k=top_k)
        return [dict(row) for row in rows]

    return search


def _default_tree_search(cfg: Any, db: Any) -> _TREE_SEARCH:
    def search(query: str, top_k: int) -> list[dict[str, Any]]:
        from drbrain.rag.config import get_llamaindex_config
        from drbrain.services.embedding import _embed_batch
        from drbrain.tree.embedding_identity import profile_from_config
        from drbrain.tree.publish import get_active_tree_generation, resolve_tree_generation
        from drbrain.tree.search import TreeSearch
        from drbrain.tree.vector_store import UnifiedVectorStore

        li = get_llamaindex_config(cfg)
        root = Path(li.tree_storage or "data/tree")
        generation = get_active_tree_generation(root)
        if not generation:
            raise RuntimeError(f"no active tree generation under {root}")
        resolved = resolve_tree_generation(root, generation)
        profile = profile_from_config(cfg.embed)
        with UnifiedVectorStore(Path(resolved["vectors"]), dimension=profile.dimension) as store:
            searcher = TreeSearch(store, profile_id=profile.profile_id(), top_k=top_k)
            candidates = searcher.search_from_text(
                lambda texts: _embed_batch(list(texts), cfg.embed),
                query,
                top_k=top_k,
                view="all",
            )
        return [candidate.to_json() for candidate in candidates]

    return search


def _default_pageindex_search(cfg: Any, db: Any) -> _PAGEINDEX_SEARCH:
    def search(query: str, papers: Sequence[str], top_k: int) -> list[dict[str, Any]]:
        from drbrain.query.tree_retrieval import query_by_structure
        from drbrain.storage.paths import paper_dir, tree_json_path

        models = list(getattr(cfg.llm, "models", None) or [])
        if not models:
            raise RuntimeError("PageIndex baseline needs llm.models")
        papers_root = Path(getattr(getattr(cfg, "dirs", None), "papers", "data/papers"))
        selected = list(papers) or [
            str(row["local_id"]) for row in (db.get_all_papers() if db is not None else [])
        ]
        out: list[dict[str, Any]] = []
        for local_id in selected:
            directory = paper_dir(papers_root, local_id)
            if not tree_json_path(directory).is_file():
                continue
            results = asyncio.run(query_by_structure(query, directory, models))
            for item in results or []:
                if not isinstance(item, dict):
                    continue
                out.append(
                    {
                        "paper_id": local_id,
                        "node_id": str(item.get("node_id") or ""),
                        "score": float(item.get("score") or 0.0),
                    }
                )
        out.sort(key=lambda row: row["score"], reverse=True)
        return out[:top_k]

    return search


# ── scoring (evaluation entry only) ─────────────────────────────────────────


@dataclass
class BaselineEval:
    name: str
    algorithm: str
    queries: int = 0
    k: int = 10
    hit_rate_paper: float = 0.0
    hit_rate_node: float = 0.0
    mrr_paper: float = 0.0
    mrr_node: float = 0.0
    cost: dict[str, Any] = field(default_factory=dict)
    per_query: list[dict[str, Any]] = field(default_factory=list)
    notes: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "baseline": self.name,
            "algorithm": self.algorithm,
            "queries": self.queries,
            "k": self.k,
            "hit_rate_paper": round(self.hit_rate_paper, 4),
            "hit_rate_node": round(self.hit_rate_node, 4),
            "mrr_paper": round(self.mrr_paper, 4),
            "mrr_node": round(self.mrr_node, 4),
            "cost": self.cost,
            "per_query": self.per_query,
            "notes": list(self.notes),
        }


def _rank_metrics(keys: Sequence[str], entry: dict[str, Any], k: int) -> dict[str, Any]:
    papers = {str(p) for p in entry.get("relevant_papers") or []}
    nodes = {str(n) for n in entry.get("relevant_nodes") or []}
    paper_rank: int | None = None
    node_rank: int | None = None
    for rank, key in enumerate(keys[:k], start=1):
        paper_id, _, node_id = str(key).partition(":")
        if paper_rank is None and paper_id in papers:
            paper_rank = rank
        if node_rank is None and node_id in nodes:
            node_rank = rank
        if paper_rank is not None and node_rank is not None:
            break
    return {"paper_rank": paper_rank, "node_rank": node_rank}


def evaluate_baseline(
    name: str,
    cfg: Any,
    db: Any,
    entries: Sequence[dict[str, Any]],
    *,
    k: int = 10,
    runner: BaselineRunner | None = None,
    papers_for: Callable[[dict[str, Any]], Sequence[str] | None] | None = None,
) -> BaselineEval:
    """Run one baseline over a golden split and score it (evaluation only)."""
    runner = runner or BaselineRunner(cfg=cfg, db=db)
    result = BaselineEval(name=name, algorithm=baseline_algorithm(name), k=k)
    if not entries:
        result.notes = ("no golden entries",)
        return result
    total_paper = total_node = 0.0
    elapsed_ms = 0
    for entry in entries:
        papers = papers_for(entry) if papers_for else entry.get("relevant_papers")
        outcome = runner.run(name, str(entry.get("query") or ""), top_k=k, papers=papers)
        metrics = _rank_metrics(outcome.keys, entry, k)
        elapsed_ms += int(outcome.details.get("elapsed_ms") or 0)
        result.queries += 1
        if metrics["paper_rank"]:
            total_paper += 1.0 / metrics["paper_rank"]
        if metrics["node_rank"]:
            total_node += 1.0 / metrics["node_rank"]
        result.per_query.append({"id": entry.get("id"), "status": outcome.status, **metrics})
    result.hit_rate_paper = sum(1 for row in result.per_query if row["paper_rank"]) / result.queries
    result.hit_rate_node = sum(1 for row in result.per_query if row["node_rank"]) / result.queries
    result.mrr_paper = total_paper / result.queries
    result.mrr_node = total_node / result.queries
    result.cost = {
        "elapsed_ms": elapsed_ms,
        "queries": result.queries,
        "k": k,
        "production_route": False,
    }
    return result


__all__ = [
    "BASELINES",
    "BaselineEval",
    "BaselineOutcome",
    "BaselineRunner",
    "baseline_algorithm",
    "evaluate_baseline",
]
