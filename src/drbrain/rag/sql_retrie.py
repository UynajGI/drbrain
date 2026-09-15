"""SQL-native RAG retrieval over ``drbrain_rag.db`` (target architecture).

Full text lives IN the library database (``node_texts`` + FTS5) and vectors
are the pipeline's ``tree_vectors`` / ``tree_vectors_vec`` — retrieval becomes
a database feature instead of a parallel LlamaIndex store.  Legs fused via
reciprocal-rank fusion (each gated by ``llamaindex.retrievers``):

* BM25:   FTS5 ``MATCH`` with ``bm25()`` ranking (recall stage)
* vector: Zvec HNSW ANN over PageIndex vectors (or SQLite cosine rerank in
  compatibility mode)
* tree:   pinned ANN recall verified against the ``node_texts`` evidence row
  (same node id and content hash — same revision).  The retired SQL LIKE
  heading scan and the legacy RAPTOR pool are no longer legs (T43): the
  legacy names fold into this one tree leg.
* graph:  KG concept seeds + 1-hop neighbours (live extra)

Rows match the shape of :func:`drbrain.rag.agent._retrieval_rows` so the loop's
evidence machinery (``build_evidence_record``) works unchanged.
"""

from __future__ import annotations

import math
import os
import re
import sqlite3
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from loguru import logger as log

from drbrain.rag.contracts import (
    LegResult,
    RetrievalRequest,
    RetrievalRows,
    failure_leg,
    finish_retrieval,
    matches_scope,
    normalize_filters,
)
from drbrain.rag.evidence import build_evidence_record
from drbrain.rag.legs import normalize_legs
from drbrain.rag.status import RetrievalStatus, RetrievalUnavailableError
from drbrain.utils.rrf import DEFAULT_K as _RRF_K
from drbrain.utils.rrf import rrf_fuse_scores

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-.]*")
_MAX_TERMS = 16
_KNN_POOL = 100


def _default_rag_db(cfg: Any) -> Path:
    from drbrain.config import Config
    from drbrain.runtime import RuntimeContext, runtime_root

    runtime_selected = "DRBRAIN_ROOT" in os.environ or "DRBRAIN_RUNTIME_ROOT" in os.environ
    # Let RuntimeContext inspect the selector itself so an explicitly empty
    # value fails closed instead of being treated as legacy/no-runtime mode.
    runtime = RuntimeContext.create() if runtime_selected else None

    if isinstance(cfg, Config):
        if cfg.db.path in ("", ":memory:"):
            root = runtime_root() / "data"
        else:
            db_path = Path(cfg.db.path).expanduser()
            if not db_path.is_absolute():
                db_path = runtime_root() / db_path
            if runtime is not None:
                db_path = runtime.assert_within_root(db_path, label="RAG source database")
            root = db_path.resolve().parent
    else:
        db_cfg = cfg.get("db", {}) if isinstance(cfg, dict) else {}
        db_value = db_cfg.get("path") if isinstance(db_cfg, dict) else None
        if db_value and db_value != ":memory:":
            db_path = Path(db_value).expanduser()
            if not db_path.is_absolute():
                db_path = runtime_root() / db_path
            if runtime is not None:
                db_path = runtime.assert_within_root(db_path, label="RAG source database")
            root = db_path.resolve().parent
        else:
            root = runtime_root() / "data"
    rag_path = root / "drbrain_rag.db"
    if runtime is not None:
        return runtime.assert_within_root(rag_path, label="RAG database")
    return rag_path


def _open(cfg: Any, generation: str | None = None) -> sqlite3.Connection:
    from drbrain.rag.sql_snapshot import resolve_sql_snapshot

    path = resolve_sql_snapshot(cfg, generation) if generation is not None else _default_rag_db(cfg)
    if not path.exists():
        raise RetrievalUnavailableError("SQL corpus is unavailable")
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    # All legs see one read transaction, including the mutable working-copy mode.
    try:
        conn.execute("BEGIN")
    except Exception:
        conn.close()
        raise
    return conn


def _generation_id(conn: sqlite3.Connection) -> str:
    """Complete fingerprint for publishing/auditing, never a query-time scan."""
    from drbrain.storage.rag_snapshot import content_fingerprint

    return content_fingerprint(conn)


def _fts_query(query: str) -> str | None:
    words = _WORD_RE.findall(query)[:_MAX_TERMS]
    if not words:
        return None
    # Quoted terms: safe literals, OR semantics (porter tokenizer handles stems).
    return " OR ".join(f'"{w}"' for w in words)


def _categories_filter(
    conn: sqlite3.Connection, categories: list[str] | tuple[str, ...] | str | None
) -> tuple[str, list[str]]:
    """Build a paper_id subquery restricting candidates to wanted categories.

    ``categories`` comes from ``filters["categories"]`` (a string, or list of
    strings). Matching is arXiv token-prefix aware — ``cond-mat`` hits
    ``cond-mat.mes-hall`` but never ``xcond-mat``. An absent filter adds no
    constraint; an empty allow-list matches nothing. A requested category
    filter requires metadata and raises when its table is missing.
    """
    if isinstance(categories, str):
        raw: list[str] = [categories]
    else:
        raw = list(categories or [])
    wanted = [str(c).strip().lower() for c in raw if str(c).strip()]
    if categories is None:
        return "", []
    if not wanted:
        return " AND 0", []
    has_table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='paper_categories'"
    ).fetchone()
    if not has_table:
        raise ValueError("categories filter requires paper_categories metadata")
    clauses: list[str] = []
    params: list[str] = []
    for cat in wanted:
        # 用户输入先转义 LIKE 通配符：math.G_ 里的 "_" 不该匹配任意字符。
        escaped = cat.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        clauses.append("' ' || categories || ' ' LIKE ? ESCAPE '\\'")
        params.append(f"% {escaped}%")
    subquery = (
        " AND nt.paper_id IN "
        "(SELECT paper_id FROM paper_categories WHERE (" + " OR ".join(clauses) + "))"
    )
    return subquery, params


def _bm25_leg(
    conn: sqlite3.Connection,
    query: str,
    k: int,
    *,
    categories_filter: tuple[str, list[str]] = ("", []),
) -> list[tuple[str, float]]:
    fts_q = _fts_query(query)
    if not fts_q:
        return []
    clause, params = categories_filter
    try:
        rows = conn.execute(
            """SELECT nt.node_key, bm25(node_texts_fts) AS s
               FROM node_texts_fts
               JOIN node_texts nt ON nt.rowid = node_texts_fts.rowid
               WHERE node_texts_fts MATCH ?"""
            + clause
            + """
               ORDER BY s LIMIT ?""",
            (fts_q, *params, k),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        log.warning("[rag-sql] FTS5 query failed: {}", exc)
        raise
    return [(r[0], float(r[1])) for r in rows]


def _leg_cap(cfg: Any, name: str, default: int) -> int:
    """Configured per-leg candidate cap (T44), bounded to >= 1."""
    from drbrain.rag.config import get_llamaindex_config

    li = get_llamaindex_config(cfg)
    try:
        return max(1, int(getattr(li, f"{name}_candidates", default) or default))
    except (TypeError, ValueError):
        return default


def _same_revision(left: str, right: str) -> bool:
    """True when two content hashes identify the same text revision."""
    a = str(left or "").strip().lower()
    b = str(right or "").strip().lower()
    if not a or not b:
        return False
    # Legacy stores keep sha256[:16]; canonical stores keep the full hash.
    return a[:16] == b[:16]


def _tree_leg(
    cfg: Any,
    conn: sqlite3.Connection,
    query: str,
    generation: str | None,
    k: int,
    *,
    allowed_papers: set[str] | None = None,
):
    """Unified tree recall over the published generation (T43/T45/T46).

    The leg resolves the *active unified tree generation* published by
    ``drbrain index build``, searches every published layer through the
    shared ANN, walks the tree with the stateful navigator, and keeps only
    leaf text whose receipt validates against the SQL node projection the
    BM25/vector legs read (same node id, same content revision).  The retired
    PageIndex ANN (``tree_vectors.tree_layer='pageindex'``) is never read and
    there is no legacy fallback: a missing generation is fail-closed.

    Returns ``(entries, outcome)``: the key/score pairs for fusion and the
    auditable navigation outcome (entry layer, reads, planner, unresolved).
    """
    from drbrain.tree.leg import TreeLegUnavailableError, run_tree_leg

    def verify(hit) -> bool:
        if allowed_papers is not None and hit.local_id not in allowed_papers:
            return False
        row = conn.execute(
            "SELECT node_key, content_hash FROM node_texts WHERE paper_id = ? AND node_id = ?",
            (hit.local_id, hit.node_id),
        ).fetchone()
        if row is None:
            return False
        return _same_revision(str(row[1] or ""), hit.content_hash)

    try:
        outcome = run_tree_leg(
            cfg,
            query=query,
            top_k=max(int(k), 1),
            verify=verify,
            local_ids=sorted(allowed_papers) if allowed_papers is not None else None,
        )
    except TreeLegUnavailableError as exc:
        raise RetrievalUnavailableError(str(exc)) from exc
    entries = [(hit.key, hit.score) for hit in outcome.hits]
    return entries, outcome


def _rerank_with_vectors(
    cfg: Any,
    conn: sqlite3.Connection,
    query: str,
    keys: list[str],
    k: int,
) -> list[tuple[str, float]]:
    """Vector rerank inside the BM25 candidate pool.

    A whole-library vec0 KNN is a brute-force scan over ~4M vectors (~17s per
    query) — unusable online. Instead: fetch the candidate pool's vectors by
    node_key (indexed point reads) and cosine-rank in numpy. Classic two-stage
    retrieval, milliseconds instead of seconds.
    """
    if not keys:
        return []
    try:
        from drbrain.services.embedding import _embed_batch

        qvec = _embed_batch([query], cfg.embed)[0]
    except Exception as exc:  # noqa: BLE001 - embedding must not raise here
        log.warning("[rag-sql] query embedding failed: {}", exc)
        raise
    import numpy as np

    from drbrain.storage import vector_index as vi

    q = np.asarray(qvec, dtype=np.float32)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    scored: list[tuple[str, float]] = []
    batch = 500
    for s in range(0, len(keys), batch):
        chunk = keys[s : s + batch]
        ph = ",".join("?" * len(chunk))
        rows = conn.execute(
            "SELECT node_id, embedding FROM tree_vectors "
            f"WHERE tree_layer = 'pageindex' AND node_id IN ({ph}) "
            "AND length(embedding) = ?",
            (*chunk, vi.embedding_byte_len(conn)),
        ).fetchall()
        for node_key, blob in rows:
            v = np.frombuffer(blob, dtype=np.float32).copy()
            v /= max(float(np.linalg.norm(v)), 1e-12)
            scored.append((node_key, float(q @ v)))
    scored.sort(key=lambda kv: kv[1], reverse=True)
    return scored[:k]


def _zvec_leg(
    cfg: Any,
    query: str,
    generation: str | None,
    k: int,
    *,
    allowed_papers: set[str] | None = None,
) -> list[tuple[str, float]]:
    """Independent ANN recall from the Zvec index pinned to ``generation``."""

    if generation is None:
        raise RetrievalUnavailableError("Zvec retrieval requires a pinned SQL generation")
    from drbrain.rag.sql_snapshot import resolve_sql_vector_index
    from drbrain.rag.zvec_index import configured_vector_top_k, query_zvec_index
    from drbrain.services.embedding import _embed_batch

    qvec = _embed_batch([query], cfg.embed)[0]
    # Oversample before applying category/ACL filters, which are authoritative
    # in SQLite and intentionally not duplicated in the ANN payload.
    requested = max(int(k), configured_vector_top_k(cfg), 100)
    raw = query_zvec_index(resolve_sql_vector_index(cfg, generation), qvec, requested)
    out = [
        (node_id, score)
        for node_id, score, paper_id in raw
        if allowed_papers is None or paper_id in allowed_papers
    ]
    return out[:k]


def _fuse(legs: list[list[tuple[str, float]]]) -> list[tuple[str, float]]:
    # RRF 收敛（R-I7）：实现与常量统一来自 drbrain.utils.rrf，不再自留一份。
    return rrf_fuse_scores(legs, k=_RRF_K)


# Module-level reranker cache: CrossEncoderReranker's model load is lazy but
# per-instance, so rebuilding it per query would reload the model every time.
_RERANKER_CACHE: dict[str, Any] = {}


def _get_reranker(cfg: Any) -> Any:
    """Process-wide reranker for the SQL path (``None`` = rerank disabled)."""
    from drbrain.rag.fusion import get_llamaindex_config

    li = get_llamaindex_config(cfg)
    if not getattr(li, "rerank", False):
        return None
    model = str(getattr(li, "rerank_model", "") or "").strip()
    if not model:
        return None
    device = getattr(getattr(cfg, "embed", None), "device", None)
    cache_key = f"{model}@{device}"
    if cache_key not in _RERANKER_CACHE:
        try:
            from drbrain.rag.rerank import build_reranker

            _RERANKER_CACHE[cache_key] = build_reranker(cfg)
        except Exception as exc:  # noqa: BLE001 - rerank stays optional
            log.warning("[rag-sql] reranker init failed ({}); rerank disabled", exc)
            _RERANKER_CACHE[cache_key] = None
    return _RERANKER_CACHE[cache_key]


def _graph_neighbors(
    graph: Any, label: str, seed_score: float, max_neighbors: int = 8
) -> list[dict[str, Any]]:
    """1-hop neighbour expansion of one seed concept (score-decayed)."""
    if graph is None:
        return []
    from drbrain.extractor.agent_tools import get_neighbors

    try:
        neighbors = get_neighbors(graph, label, hops=1, direction="both") or []
    except Exception as exc:  # noqa: BLE001 - graph outage must not break the leg
        log.warning("[rag-sql] graph neighbor expansion failed for {}: {}", label, exc)
        raise
    out: list[dict[str, Any]] = []
    for nb in neighbors[:max_neighbors]:
        target = str(nb.get("target") or "").strip()
        if not target or target == label:
            continue
        path = nb.get("path") or []
        relation = str(path[0].get("relation") or "") if path else ""
        distance = int(nb.get("distance") or 1) or 1
        text = f"{label} --[{relation}]--> {target}" if relation else f"{label} → {target}"
        out.append(
            {
                "key": f"concept:{target}",
                "score": seed_score * 0.5 / distance,
                "paper_id": "",
                "node_id": f"concept:{target}",
                "title": target,
                "text": text[:500],
                "source": "graph",
            }
        )
    return out


def _graph_leg(db: Any, graph: Any, query: str, k: int) -> list[dict[str, Any]]:
    """Knowledge-graph traversal leg: seed concepts + 1-hop neighbours.

    Seeds come from the concept BM25 (``search_concepts`` over the runtime
    library DB); each seed is enriched from the ``concepts`` table and expanded
    one hop through the graph. Entries carry their own row metadata because
    concept nodes have no ``node_texts`` counterpart.
    """
    if db is None:
        return []
    from drbrain.extractor.agent_tools import search_concepts

    try:
        concepts = search_concepts(db, query, limit=k) or []
    except Exception as exc:  # noqa: BLE001
        log.warning("[rag-sql] graph concept search failed: {}", exc)
        raise
    best: dict[str, dict] = {}
    for c in concepts:
        label = str(c.get("label") or "").strip()
        if not label:
            continue
        score = float(c.get("score") or 0.0)
        if label not in best or score > float(best[label].get("score") or 0.0):
            best[label] = {**c, "score": score}
    entries: list[dict[str, Any]] = []
    for c in best.values():
        label = c["label"]
        score = float(c.get("score") or 0.0)
        ctype = str(c.get("type") or "")
        local_id = ""
        section = ""
        try:
            row = db.conn.execute(
                "SELECT local_id, type, label, confidence, section FROM concepts "
                "WHERE label = ? ORDER BY confidence DESC LIMIT 1",
                (label,),
            ).fetchone()
        except Exception:  # noqa: BLE001 - schema drift must not break the leg
            row = None
        if row:
            local_id = str(row[0] or "")
            ctype = str(row[1] or "") or ctype
            label = str(row[2] or "") or label
            section = str(row[4] or "")
        text = " ".join(
            part
            for part in (label, f"({ctype})" if ctype else "", f"— {section}" if section else "")
            if part
        ).strip()
        entries.append(
            {
                "key": f"concept:{label}",
                "score": score,
                "paper_id": local_id,
                "node_id": f"concept:{label}",
                "title": label,
                "text": text[:500],
                "source": "graph",
            }
        )
        entries.extend(_graph_neighbors(graph, label, score))
    return entries


def _claims_leg(db: Any, query: str, k: int) -> list[dict[str, Any]]:
    """Settled-claims leg (review §7.4): the loop's own conclusions are knowledge.

    Reads the main DB ``claims`` table (verified/falsified/predicted assertions
    written at settle time) so the next cycle's retrieve step can see — and
    cite — what earlier cycles concluded. Entries carry their own row metadata
    because claims have no ``node_texts`` counterpart. Keywords score by
    hit count × claim confidence; read failures propagate to the leg trace.
    """
    if db is None:
        return []
    words = _WORD_RE.findall(query)[:_MAX_TERMS]
    if not words:
        return []
    try:
        rows = db.execute(
            "SELECT claim_id, claim_text, claim_type, confidence FROM claims "
            "ORDER BY created_at DESC LIMIT 500"
        ).fetchall()
    except Exception as exc:  # noqa: BLE001 — classify schema drift at the fusion boundary
        log.warning("[rag-sql] claims leg read failed: {}", exc)
        raise
    lowered = [w.lower() for w in words]
    scored: list[tuple[float, tuple]] = []
    for row in rows:
        text_l = str(row[1]).lower()
        hits = sum(1 for w in lowered if w in text_l)
        if hits:
            scored.append((hits * max(float(row[3] or 0.0), 0.05), row))
    scored.sort(key=lambda t: t[0], reverse=True)
    entries: list[dict[str, Any]] = []
    for score, (claim_id, claim_text, claim_type, confidence) in scored[:k]:
        entries.append(
            {
                "key": f"claim:{claim_id}",
                "score": max(0.05, min(score, 1.0)),
                "paper_id": "",
                "node_id": f"claim:{claim_id}",
                "title": f"[{claim_type or 'claim'}]",
                "text": str(claim_text)[:500],
                "source": "claims",
            }
        )
    return entries


def retrieve_documents_sql(
    cfg: Any,
    db: Any,
    query: str,
    *,
    filters: dict[str, Any] | None = None,
    top_k: int = 5,
    graph: Any = None,
    generation: str | None = None,
    acl_filter: dict[str, str] | None = None,
) -> RetrievalRows:
    """Retrieve from a pinned SQL snapshot or an explicitly live working copy.

    Zvec vector and PageIndex legs are independent recall sources.  The legacy
    SQLite vector backend and RAPTOR remain BM25-pool rerankers.  Diagnostics
    remain available as rows.result, including empty results.
    """
    from drbrain.rag.config import get_llamaindex_config
    from drbrain.rag.zvec_index import configured_vector_backend

    li = get_llamaindex_config(cfg)
    vector_backend = configured_vector_backend(cfg)
    normalized = normalize_legs(li.retrievers)
    for note in normalized.notes:
        log.info("[rag-sql] retriever config: {}", note)
    wanted = normalized.as_list()
    captured_generation = False
    if generation is None and vector_backend == "zvec" and {"vector", "tree"} & set(wanted):
        # Direct callers (the research loop and low-level API) may omit the
        # generation. Resolve the active immutable snapshot before opening the
        # database so the ANN sidecar and SQLite text always share an epoch.
        from drbrain.rag.index_generations import capture_index_generation

        generation = capture_index_generation(cfg)
        captured_generation = True
    if generation is None and not _default_rag_db(cfg).is_file():
        # The default ``drbrain index build`` no longer copies the text/vector
        # projection: serve the request from the published unified generation
        # (canonical FTS + shared leaf ANN + tree) and report only the legs the
        # main store cannot answer (fail-closed per leg).  Checked after the
        # snapshot capture so a published SQL generation outside the working
        # copy still wins; it catches the case where that capture finds none.
        return _unified_corpus_retrieval(
            cfg, db, wanted, query, top_k, filters=filters, acl_filter=acl_filter
        )
    if generation is None and captured_generation:
        raise RetrievalUnavailableError("no active SQL generation for Zvec retrieval")
    request = RetrievalRequest(query, top_k, generation, filters or {}, acl_filter or {})
    if generation is not None and set(wanted).intersection({"graph", "claims"}):
        raise ValueError("pinned SQL retrieval cannot include live graph/claims sources")
    if set(request.acl_filter) - {"paper_id"}:
        raise ValueError(
            "SQL access filters only support paper_id; other scope metadata is unavailable"
        )
    conn = _open(cfg, generation) if generation is not None else _open(cfg)
    resolved = generation or ("working-" + uuid.uuid4().hex)
    traces: list[LegResult] = []
    capabilities = {
        "backend": "sql",
        "snapshot": generation is not None,
        "vector_recall": "zvec_ann" if vector_backend == "zvec" else "bm25_pool",
    }
    try:
        scope_sql, scope_params = _categories_filter(conn, request.filters.get("categories"))
        if "paper_ids" in request.filters:
            papers = request.filters["paper_ids"]
            if not papers:
                scope_sql += " AND 0"
            else:
                scope_sql += " AND nt.paper_id IN (" + ",".join("?" for _ in papers) + ")"
                scope_params += papers
        if request.acl_filter.get("paper_id") not in (None, "*"):
            scope_sql += " AND nt.paper_id = ?"
            scope_params += [request.acl_filter["paper_id"]]
        allowed_papers: set[str] | None = None
        if scope_sql:
            allowed_papers = {
                str(row[0])
                for row in conn.execute(
                    "SELECT DISTINCT nt.paper_id FROM node_texts nt WHERE 1=1" + scope_sql,
                    tuple(scope_params),
                ).fetchall()
            }
        if top_k == 0:
            return finish_retrieval([], generation=resolved, legs=[], capabilities=capabilities)
        pool_error: Exception | None = None
        pool_started = time.perf_counter()
        try:
            bm25 = (
                _bm25_leg(
                    conn,
                    query,
                    _leg_cap(cfg, "bm25", 1000),
                    categories_filter=(scope_sql, scope_params),
                )
                if set(wanted).intersection(
                    {"bm25"} | ({"vector"} if vector_backend == "sqlite" else set())
                )
                else []
            )
        except Exception as exc:
            pool_error, bm25 = exc, []
        pool_ms = (time.perf_counter() - pool_started) * 1000
        pool = [key for key, _ in bm25]
        legs: list[tuple[str, list[dict[str, Any]]]] = []
        entries: list[dict[str, Any]]
        for name in wanted:
            started = time.perf_counter()
            leg_status: str | None = None
            leg_reason = ""
            try:
                if (
                    name == "bm25" or (name == "vector" and vector_backend == "sqlite")
                ) and pool_error is not None:
                    raise pool_error
                if name == "bm25":
                    entries = [{"key": key, "score": score} for key, score in bm25]
                elif name == "vector":
                    if vector_backend == "zvec":
                        vector_entries = _zvec_leg(
                            cfg,
                            query,
                            generation,
                            _leg_cap(cfg, "vector", _KNN_POOL),
                            allowed_papers=allowed_papers,
                        )
                    else:
                        vector_entries = _rerank_with_vectors(
                            cfg, conn, query, pool, _leg_cap(cfg, "vector", _KNN_POOL)
                        )
                    entries = [{"key": key, "score": score} for key, score in vector_entries]
                elif name == "tree":
                    tree_entries, tree_outcome = _tree_leg(
                        cfg,
                        conn,
                        query,
                        generation,
                        _leg_cap(cfg, "tree", _KNN_POOL),
                        allowed_papers=allowed_papers,
                    )
                    entries = [{"key": key, "score": score} for key, score in tree_entries]
                    capabilities["tree"] = tree_outcome.to_json()
                    leg_status = "ok" if entries else tree_outcome.status
                    if tree_outcome.status != "ok":
                        # A leg's failure reason must be a RetrievalStatus value;
                        # the detailed navigation reason stays in capabilities.
                        leg_reason = (
                            RetrievalStatus.INSUFFICIENT_EVIDENCE.value
                            if tree_outcome.status == "unavailable"
                            else ""
                        )
                elif name == "graph":
                    entries = _graph_leg(db, graph, query, max(top_k, 20))
                else:
                    entries = _claims_leg(db, query, max(top_k, 20))
                traces.append(
                    LegResult(
                        name,
                        leg_status or ("ok" if entries else "empty"),
                        len(entries),
                        pool_ms if name == "bm25" else (time.perf_counter() - started) * 1000,
                        leg_reason,
                    )
                )
                legs.append((name, entries))
            except Exception as exc:
                traces.append(failure_leg(name, exc, (time.perf_counter() - started) * 1000))
        rich = {
            entry["key"]: entry
            for name, entries in legs
            if name in {"graph", "claims"}
            for entry in entries
        }
        membership: dict[str, list[str]] = {}
        for name, entries in legs:
            for entry in entries:
                membership.setdefault(entry["key"], []).append(name)
        fused = _fuse(
            [[(entry["key"], entry["score"]) for entry in entries] for _, entries in legs]
        )
        candidates = _materialize(conn, fused, rich, membership)
        candidates = [
            row for row in candidates if matches_scope(row, request.filters, request.acl_filter)
        ]
        rerank_status = "disabled"
        reranker = _get_reranker(cfg)
        if reranker is not None and candidates:
            count = max(int(li.rerank_top_k or 20), top_k)
            head = candidates[:count]
            try:
                scores = reranker.rerank(query, [row["text"][:2000] for row in head])
                if len(scores) != len(head) or any(
                    score is None or not math.isfinite(float(score)) for score in scores
                ):
                    raise ValueError("invalid rerank scores")
                for row, score in zip(head, scores):
                    row["score"] = float(score)
                    row["score_kind"] = "rerank"
                candidates = (
                    sorted(head, key=lambda row: row["score"], reverse=True) + candidates[count:]
                )
                rerank_status = "ok"
            except Exception:
                rerank_status = "degraded"
        elif candidates and reranker is None and li.rerank:
            rerank_status = "unavailable"
        capabilities["rerank_status"] = rerank_status
        primary = _diverse_head(candidates, top_k)
        rows = []
        for rank, row in enumerate(primary, 1):
            full_text = row["text"]
            row = {key: value for key, value in row.items() if key != "key"}
            row["score"] = round(float(row["score"]), 6)
            row["text"] = full_text[:500]
            row.update(
                build_evidence_record(
                    generation=resolved,
                    query=query,
                    retriever="sql-fusion",
                    rank=rank,
                    score=row["score"],
                    source={**row, "text": full_text},
                    filters=request.filters,
                    excerpt=row["text"],
                )
            )
            rows.append(row)
        result = finish_retrieval(rows, generation=resolved, legs=traces, capabilities=capabilities)
        if rerank_status in {"degraded", "unavailable"}:
            result.result.status = "degraded"
        return result
    finally:
        conn.close()


def _unified_embedder(cfg: Any):
    """One query-embedding callable shared by the vector and tree legs.

    Cached by text: the navigator's in-walk re-searches still embed their own
    text, while the original query is embedded exactly once per request.
    """
    from drbrain.services.embedding import _embed_batch

    cache: dict[str, list[float]] = {}

    def embed_texts(texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            value = str(text)
            if value not in cache:
                cache[value] = _embed_batch([value], getattr(cfg, "embed", None))[0]
            out.append(cache[value])
        return out

    return embed_texts


def _unified_bm25_entries(
    db: Any, query: str, cap: int, scope: set[str] | None
) -> list[tuple[str, float]]:
    """``(paper_id:node_id, score)`` pairs from the canonical FTS, best-first.

    Blocks without a ready leaf are skipped: the evidence identity is the
    published leaf node, never the bare block row.
    """
    fts = _fts_query(query)
    if not fts:
        return []
    hits = db.search_content(fts, limit=max(1, int(cap)))
    if scope is not None:
        hits = [hit for hit in hits if str(hit.get("local_id") or "") in scope]
    if not hits:
        return []
    leaves: dict[str, tuple[str, str]] = {}
    block_ids = list(dict.fromkeys(str(hit["block_id"]) for hit in hits))
    for offset in range(0, len(block_ids), 500):
        batch = block_ids[offset : offset + 500]
        placeholders = ",".join("?" for _ in batch)
        for node_id, local_id, block_id in db.conn.execute(
            "SELECT node_id, local_id, block_id FROM tree_nodes "
            f"WHERE kind = 'leaf' AND state = 'ready' AND block_id IN ({placeholders})",
            batch,
        ):
            leaves[str(block_id)] = (str(node_id), str(local_id))
    entries: list[tuple[str, float]] = []
    seen: set[str] = set()
    for hit in hits:
        mapped = leaves.get(str(hit["block_id"]))
        if mapped is None:
            continue
        key = f"{mapped[1]}:{mapped[0]}"
        if key in seen:
            continue
        seen.add(key)
        # FTS5 bm25 is "lower is better"; negate so every leg presents a
        # higher-is-better score (RRF itself ranks by list order).
        entries.append((key, -float(hit["score"])))
    return entries[: max(1, int(cap))]


def _unified_vector_entries(
    cfg: Any, query_vector: Sequence[float], cap: int, scope: Sequence[str] | None
) -> list[tuple[str, float]]:
    """``(paper_id:node_id, score)`` leaf hits from the shared ANN (view=leaf)."""
    from pathlib import Path

    from drbrain.tree.embedding_identity import profile_from_config
    from drbrain.tree.leg import tree_storage_root
    from drbrain.tree.publish import get_active_tree_generation, resolve_tree_generation
    from drbrain.tree.search import TreeSearch
    from drbrain.tree.vector_store import UnifiedVectorStore

    root = tree_storage_root(cfg)
    generation = get_active_tree_generation(root)
    if not generation:
        raise RetrievalUnavailableError(f"no active unified generation under {root}")
    resolved = resolve_tree_generation(root, generation)
    profile = profile_from_config(getattr(cfg, "embed", None))
    if profile.dimension is None:
        raise RetrievalUnavailableError("embedding profile has no dimension; cannot read the ANN")
    limit = max(1, int(cap))
    with UnifiedVectorStore(Path(resolved["vectors"]), dimension=int(profile.dimension)) as store:
        searcher = TreeSearch(store, profile_id=profile.profile_id(), top_k=limit)
        candidates = searcher.search(query_vector, top_k=limit, view="leaf", local_ids=scope)
    return [
        (f"{candidate.local_id}:{candidate.node_id}", float(candidate.score))
        for candidate in candidates
    ]


def _unified_materialize(
    db: Any,
    fused: list[tuple[str, float]],
    membership: dict[str, list[str]],
    request_filters: dict[str, Any],
    acl_filter: dict[str, str] | None,
) -> list[dict[str, Any]]:
    """Resolve fused keys against the main store (ready leaf row + canonical text)."""
    keys = [key for key, _score in fused]
    if not keys:
        return []
    meta: dict[str, dict[str, Any]] = {}
    for offset in range(0, len(keys), 500):
        batch = keys[offset : offset + 500]
        node_ids = [key.split(":", 1)[-1] for key in batch]
        placeholders = ",".join("?" for _ in node_ids)
        for row in db.conn.execute(
            "SELECT n.node_id, n.local_id, n.title, n.block_id, n.char_start, n.char_end, "
            "       n.content_hash, b.text "
            "FROM tree_nodes n LEFT JOIN content_blocks b ON b.block_id = n.block_id "
            f"WHERE n.kind = 'leaf' AND n.state = 'ready' AND n.node_id IN ({placeholders})",
            node_ids,
        ):
            node_id, local_id, title, block_id, char_start, char_end, content_hash, text = row
            meta[f"{local_id}:{node_id}"] = {
                "node_id": str(node_id),
                "paper_id": str(local_id or ""),
                "title": str(title or ""),
                "block_id": str(block_id or ""),
                "char_start": char_start,
                "char_end": char_end,
                "content_hash": str(content_hash or ""),
                "text": str(text or ""),
            }
    categories: dict[str, str] = {}
    papers = sorted({str(item["paper_id"]) for item in meta.values() if item["paper_id"]})
    try:
        for offset in range(0, len(papers), 500):
            batch = papers[offset : offset + 500]
            placeholders = ",".join("?" for _ in batch)
            for paper_id, value in db.conn.execute(
                "SELECT paper_id, categories FROM paper_categories "
                f"WHERE paper_id IN ({placeholders})",
                batch,
            ):
                categories[str(paper_id)] = str(value or "")
    except sqlite3.Error:
        categories = {}  # metadata table absent on old corpora
    rows: list[dict[str, Any]] = []
    for key, score in fused:
        item = meta.get(key)
        if item is None:
            continue
        text = str(item["text"])
        row: dict[str, Any] = {
            "key": key,
            "paper_id": item["paper_id"],
            "node_id": item["node_id"],
            "title": str(item["title"]) or text.split("\n", 1)[0][:120],
            "text": text,
            "source": "unified-fusion",
            "score": float(score),
            "score_kind": "rrf",
            "categories": categories.get(item["paper_id"], ""),
            "legs": membership.get(key, []),
            "block_id": item["block_id"],
            "char_start": item["char_start"],
            "char_end": item["char_end"],
            "content_hash": item["content_hash"],
        }
        rows.append(row)
    return [row for row in rows if matches_scope(row, request_filters, acl_filter)]


def _unified_corpus_retrieval(
    cfg: Any,
    db: Any,
    wanted: list[str],
    query: str,
    top_k: int,
    *,
    filters: dict[str, Any] | None,
    acl_filter: dict[str, str] | None,
) -> RetrievalRows:
    """Three-leg retrieval over the unified store when no SQL corpus exists.

    ``drbrain index build`` publishes one main-store generation (canonical FTS
    + shared vectors + tree).  This path reads it directly: bm25 searches the
    canonical FTS, vector reads the shared leaf ANN, and tree walks the same
    generation — one store, one publication, one evidence identity
    (``paper_id:node_id`` with the main-store revision and content hash), with
    no legacy ``drbrain_rag.db`` projection involved.  ``filters["paper_ids"]``
    scopes all three legs; graph/claims stay ``source_unavailable`` (they need
    the live corpus); a missing generation is still fail-closed so the
    missing-index remedy keeps pointing at ``drbrain index build``.
    """
    from drbrain.rag.config import get_llamaindex_config
    from drbrain.rag.status import RetrievalStatus
    from drbrain.tree.leg import active_tree_generation, run_tree_leg

    li = get_llamaindex_config(cfg)
    request_filters = normalize_filters(filters)
    scope = list(request_filters.get("paper_ids") or []) or None
    scope_set = set(scope) if scope else None

    generation = active_tree_generation(cfg)
    if not generation:
        raise RetrievalUnavailableError(
            "no active unified tree generation; run `drbrain index build` to publish one"
        )
    capabilities: dict[str, Any] = {
        "backend": "unified",
        "snapshot": True,
        "vector_recall": "shared_ann_leaf",
        "sql_corpus": False,
    }
    if int(top_k) <= 0:
        return finish_retrieval([], generation=generation, legs=[], capabilities=capabilities)

    embed_texts = _unified_embedder(cfg)
    traces: list[LegResult] = []
    legs: list[tuple[str, list[tuple[str, float]]]] = []
    for name in wanted:
        started = time.perf_counter()
        try:
            if name == "bm25":
                entries = _unified_bm25_entries(db, query, _leg_cap(cfg, "bm25", 1000), scope_set)
            elif name == "vector":
                query_vector = embed_texts([query])[0]
                entries = _unified_vector_entries(
                    cfg, query_vector, _leg_cap(cfg, "vector", _KNN_POOL), scope
                )
            elif name == "tree":
                outcome = run_tree_leg(
                    cfg,
                    query=query,
                    top_k=max(_leg_cap(cfg, "tree", _KNN_POOL), int(top_k), 1),
                    local_ids=scope,
                    embed=embed_texts,
                )
                entries = [(hit.key, float(hit.score)) for hit in outcome.hits]
                capabilities["tree"] = outcome.to_json()
                traces.append(
                    LegResult(
                        name,
                        "ok"
                        if entries
                        else ("empty" if outcome.status == "empty" else "unavailable"),
                        len(entries),
                        (time.perf_counter() - started) * 1000,
                        ""
                        if outcome.status in {"ok", "empty"}
                        else RetrievalStatus.INSUFFICIENT_EVIDENCE.value,
                    )
                )
                legs.append((name, entries))
                continue
            else:
                # graph/claims need the live corpus: fail-closed per leg, not
                # a failed whole request.
                traces.append(
                    LegResult(name, "unavailable", reason=RetrievalStatus.SOURCE_UNAVAILABLE.value)
                )
                continue
        except Exception as exc:  # noqa: BLE001 - one leg failing degrades the fusion
            traces.append(failure_leg(name, exc, (time.perf_counter() - started) * 1000))
            continue
        traces.append(
            LegResult(
                name,
                "ok" if entries else "empty",
                len(entries),
                (time.perf_counter() - started) * 1000,
            )
        )
        legs.append((name, entries))

    membership: dict[str, list[str]] = {}
    for name, entries in legs:
        for key, _score in entries:
            membership.setdefault(key, []).append(name)
    fused = _fuse([[(key, score) for key, score in entries] for _name, entries in legs])
    candidates = _unified_materialize(db, fused, membership, request_filters, acl_filter)
    rerank_status = "disabled"
    reranker = _get_reranker(cfg)
    if reranker is not None and candidates:
        count = max(int(li.rerank_top_k or 20), top_k)
        head = candidates[:count]
        try:
            scores = reranker.rerank(query, [row["text"][:2000] for row in head])
            if len(scores) != len(head) or any(
                score is None or not math.isfinite(float(score)) for score in scores
            ):
                raise ValueError("invalid rerank scores")
            for row, score in zip(head, scores):
                row["score"] = float(score)
                row["score_kind"] = "rerank"
            candidates = (
                sorted(head, key=lambda row: row["score"], reverse=True) + candidates[count:]
            )
            rerank_status = "ok"
        except Exception:
            rerank_status = "degraded"
    elif candidates and reranker is None and li.rerank:
        rerank_status = "unavailable"
    capabilities["rerank_status"] = rerank_status
    primary = _diverse_head(candidates, top_k)
    rows: list[dict[str, Any]] = []
    for rank, row in enumerate(primary, 1):
        full_text = row["text"]
        row = {key: value for key, value in row.items() if key != "key"}
        row["score"] = round(float(row["score"]), 6)
        row["text"] = full_text[:500]
        row.update(
            build_evidence_record(
                generation=generation,
                query=query,
                retriever="unified-fusion",
                rank=rank,
                score=row["score"],
                source={**row, "text": full_text},
                filters=filters,
                excerpt=row["text"],
            )
        )
        rows.append(row)
    result = finish_retrieval(rows, generation=generation, legs=traces, capabilities=capabilities)
    if rerank_status in {"degraded", "unavailable"}:
        result.result.status = "degraded"
    return result


def _materialize(
    conn: sqlite3.Connection,
    fused: list[tuple[str, float]],
    rich: dict[str, dict[str, Any]],
    membership: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """Resolve candidate metadata before scope checks and truncation."""
    meta: dict[str, tuple[str, str]] = {}
    keys = [key for key, _ in fused if key not in rich]
    for offset in range(0, len(keys), 500):
        batch = keys[offset : offset + 500]
        ph = ",".join("?" for _ in batch)
        meta.update(
            (row[0], (row[1], row[2]))
            for row in conn.execute(
                f"SELECT node_key, paper_id, text FROM node_texts WHERE node_key IN ({ph})", batch
            )
        )
        missing = [key for key in batch if key not in meta]
        if missing:
            ph = ",".join("?" for _ in missing)
            meta.update(
                (row[0], (row[1], row[2]))
                for row in conn.execute(
                    f"SELECT node_id, paper_id, summary_text FROM tree_summaries WHERE node_id IN ({ph})",
                    missing,
                )
            )
    categories: dict[str, str] = {}
    papers = list(
        {str(row.get("paper_id") or "") for row in rich.values()}
        | {value[0] for value in meta.values()}
    )
    has_categories = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='paper_categories'"
    ).fetchone()
    if has_categories:
        for offset in range(0, len(papers), 500):
            batch = papers[offset : offset + 500]
            ph = ",".join("?" for _ in batch)
            categories.update(
                conn.execute(
                    f"SELECT paper_id, categories FROM paper_categories WHERE paper_id IN ({ph})",
                    batch,
                )
            )
    rows = []
    for key, score in fused:
        if not math.isfinite(float(score)):
            continue
        if key in rich:
            item = rich[key]
            paper, text = str(item.get("paper_id") or ""), str(item.get("text") or "")
            node_id, title = item.get("node_id", key), item.get("title", "")
        else:
            if key not in meta:
                continue
            paper, text = meta[key]
            node_id, title = key.split(":", 1)[-1], text.split("\n", 1)[0][:120]
        rows.append(
            {
                "key": key,
                "paper_id": paper,
                "node_id": node_id,
                "title": title,
                "text": text,
                "source": "sql-fusion",
                "score": score,
                "score_kind": "rrf",
                "categories": categories.get(paper, ""),
                "legs": membership.get(key, []),
            }
        )
    return rows


def _diverse_head(candidates, top_k):
    """Reserve specialised hits within the cap, keeping the best overall hit."""
    if top_k <= 0:
        return []
    selected = list(candidates[:top_k])
    protected = {selected[0]["key"]} if selected else set()
    for name in ("tree", "graph", "claims"):
        best = next((row for row in candidates if name in row["legs"]), None)
        if best is None:
            continue
        if best not in selected:
            replace = next(
                (
                    i
                    for i in range(len(selected) - 1, -1, -1)
                    if selected[i]["key"] not in protected
                ),
                None,
            )
            if replace is None:
                continue
            selected[replace] = best
        protected.add(best["key"])
    return selected
