"""SQL-native RAG retrieval over ``drbrain_rag.db`` (target architecture).

Full text lives IN the library database (``node_texts`` + FTS5) and vectors
are the pipeline's ``tree_vectors`` / ``tree_vectors_vec`` — retrieval becomes
a database feature instead of a parallel LlamaIndex store.  Legs fused via
reciprocal-rank fusion (each gated by ``llamaindex.retrievers``):

* BM25:   FTS5 ``MATCH`` with ``bm25()`` ranking (recall stage)
* vector: cosine rerank of the BM25 pool over ``tree_vectors`` (pageindex)
* raptor: paper-scoped KNN over hierarchical-summary vectors
* graph:  KG concept seeds + 1-hop neighbours

``tree`` is inherent rather than a separate leg: the PageIndex tree nodes ARE
the ``node_texts`` retrieval units (section navigation is the
``get_section_content`` tool).

Rows match the shape of :func:`drbrain.rag.agent._retrieval_rows` so the loop's
evidence machinery (``build_evidence_record``) works unchanged.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from loguru import logger as log

from drbrain.rag.evidence import build_evidence_record
from drbrain.utils.rrf import DEFAULT_K as _RRF_K
from drbrain.utils.rrf import rrf_fuse_scores

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-.]*")
_MAX_TERMS = 16
_KNN_POOL = 100


def _default_rag_db(cfg: Any) -> Path:
    from drbrain.config import Config

    root = Path(cfg.db.path).parent if isinstance(cfg, Config) else Path("data")
    return root / "drbrain_rag.db"


def _open(cfg: Any) -> sqlite3.Connection | None:
    path = _default_rag_db(cfg)
    if not path.exists():
        log.warning("[rag-sql] {} not found", path)
        return None
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        import sqlite_vec

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except Exception as exc:  # noqa: BLE001 - vector leg becomes unavailable
        log.warning("[rag-sql] sqlite-vec unavailable: {}", exc)
    return conn


def _generation_id(conn: sqlite3.Connection) -> str:
    """Content fingerprint of the SQL snapshot (evidence-pinning anchor)."""
    n = conn.execute("SELECT COUNT(*) FROM node_texts").fetchone()[0]
    sample = conn.execute("SELECT content_hash FROM node_texts ORDER BY rowid LIMIT 1").fetchone()
    seed = f"{n}:{sample[0] if sample else '-'}"
    return "sql-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


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
    ``cond-mat.mes-hall`` but never ``xcond-mat``. Returns ``("", [])`` when
    the filter is absent or the corpus predates category metadata (no
    ``paper_categories`` table): missing metadata must silently widen, never
    error.
    """
    if isinstance(categories, str):
        raw: list[str] = [categories]
    else:
        raw = list(categories or [])
    wanted = [str(c).strip().lower() for c in raw if str(c).strip()]
    if not wanted:
        return "", []
    has_table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='paper_categories'"
    ).fetchone()
    if not has_table:
        return "", []
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
        return []
    return [(r[0], float(r[1])) for r in rows]


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
        return []
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


def _raptor_leg(
    cfg: Any,
    conn: sqlite3.Connection,
    query: str,
    papers: list[str],
    k: int,
) -> list[tuple[str, float]]:
    """RAPTOR hierarchical-summary leg, scoped to the BM25 candidate papers.

    Same two-stage logic as :func:`_rerank_with_vectors`: point reads via the
    ``(tree_layer, paper_id)`` index, cosine rank in numpy — milliseconds, not
    a whole-library vec0 scan.
    """
    if not papers:
        return []
    try:
        from drbrain.services.embedding import _embed_batch

        qvec = _embed_batch([query], cfg.embed)[0]
    except Exception as exc:  # noqa: BLE001 - embedding must not raise here
        log.warning("[rag-sql] raptor leg embedding failed: {}", exc)
        return []
    import numpy as np

    from drbrain.storage import vector_index as vi

    q = np.asarray(qvec, dtype=np.float32)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    scored: list[tuple[str, float]] = []
    batch = 200
    for s in range(0, len(papers), batch):
        chunk = papers[s : s + batch]
        ph = ",".join("?" * len(chunk))
        rows = conn.execute(
            "SELECT node_id, embedding FROM tree_vectors "
            f"WHERE tree_layer = 'raptor_L1' AND paper_id IN ({ph}) "
            "AND length(embedding) = ?",
            (*chunk, vi.embedding_byte_len(conn)),
        ).fetchall()
        for node_id, blob in rows:
            v = np.frombuffer(blob, dtype=np.float32).copy()
            norm = float(np.linalg.norm(v))
            if norm <= 0.0:
                continue
            scored.append((node_id, float(q @ (v / norm))))
    scored.sort(key=lambda kv: kv[1], reverse=True)
    return scored[:k]


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
        return []
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
        return []
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
    hit count × claim confidence; failures degrade to an empty leg.
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
    except Exception as exc:  # noqa: BLE001 — schema drift must not break the leg
        log.warning("[rag-sql] claims leg read failed: {}", exc)
        return []
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
) -> list[dict[str, Any]]:
    """SQL-native replacement for the LlamaIndex fused retriever.

    Legs are gated by ``llamaindex.retrievers`` (default ``["bm25", "vector"]``;
    add ``"raptor"`` / ``"graph"`` to enable the hierarchical-summary and
    knowledge-graph legs). The vector and raptor legs share the BM25 candidate
    pool (two-stage retrieval), so ``bm25`` recall always runs when either is
    wanted. Each output row carries an additive ``legs`` field naming the legs
    that surfaced it.
    """
    conn = _open(cfg)
    if conn is None:
        return []
    try:
        from drbrain.rag.fusion import get_llamaindex_config

        wanted = [
            str(x).strip() for x in (get_llamaindex_config(cfg).retrievers or ["bm25", "vector"])
        ]
        generation = _generation_id(conn)
        started = time.perf_counter()

        need_pool = any(w in wanted for w in ("bm25", "vector", "raptor"))
        categories = (filters or {}).get("categories")
        cats_filter = _categories_filter(conn, categories)
        bm25 = _bm25_leg(conn, query, 1000, categories_filter=cats_filter) if need_pool else []
        pool = [k for k, _ in bm25]
        pool_papers = list(dict.fromkeys(k.split(":", 1)[0] for k in pool))

        legs: list[tuple[str, list[dict[str, Any]]]] = []
        leg_ms: list[str] = []
        if "bm25" in wanted:
            t0 = time.perf_counter()
            legs.append(("bm25", [{"key": k, "score": s} for k, s in bm25]))
            leg_ms.append(f"bm25={len(pool)}@{(time.perf_counter() - t0) * 1000:.0f}ms")
        if "vector" in wanted:
            t0 = time.perf_counter()
            legs.append(
                (
                    "vector",
                    [
                        {"key": k, "score": s}
                        for k, s in _rerank_with_vectors(cfg, conn, query, pool, _KNN_POOL)
                    ],
                )
            )
            leg_ms.append(f"vector@{(time.perf_counter() - t0) * 1000:.0f}ms")
        if "raptor" in wanted:
            t0 = time.perf_counter()
            legs.append(
                (
                    "raptor",
                    [
                        {"key": k, "score": s}
                        for k, s in _raptor_leg(cfg, conn, query, pool_papers, _KNN_POOL)
                    ],
                )
            )
            leg_ms.append(f"raptor@{(time.perf_counter() - t0) * 1000:.0f}ms")
        if "graph" in wanted:
            t0 = time.perf_counter()
            legs.append(("graph", _graph_leg(db, graph, query, top_k)))
            leg_ms.append(f"graph={len(legs[-1][1])}@{(time.perf_counter() - t0) * 1000:.0f}ms")
        if "claims" in wanted:
            t0 = time.perf_counter()
            legs.append(("claims", _claims_leg(db, query, top_k)))
            leg_ms.append(f"claims={len(legs[-1][1])}@{(time.perf_counter() - t0) * 1000:.0f}ms")
        if not legs:
            return []

        rich = {
            e["key"]: e
            for name, entries in legs
            if name in ("graph", "claims")  # self-describing legs
            for e in entries
        }
        tuple_legs = [[(e["key"], e["score"]) for e in entries] for _, entries in legs]
        fused = _fuse(tuple_legs)
        membership: dict[str, list[str]] = {}
        for (name, _entries), pairs in zip(legs, tuple_legs):
            for key, _s in pairs:
                names = membership.setdefault(key, [])
                if name not in names:
                    names.append(name)

        # Cross-encoder rerank (same "粗排截断 → 精排" contract as the legacy
        # RerankPostprocessor): only the first ``rerank_top_k`` fused candidates
        # are re-scored; a missing/failed model degrades to the RRF order.
        li = get_llamaindex_config(cfg)
        rerank_top_k = int(getattr(li, "rerank_top_k", None) or 20)
        reranker = _get_reranker(cfg)
        cand = fused[: max(rerank_top_k, top_k)] if reranker is not None else fused[:top_k]

        # Meta resolution: graph rows carry their own metadata; node rows come
        # from node_texts (pageindex units) with tree_summaries as the raptor
        # fallback.
        text_keys = [k for k, _ in cand if k not in rich]
        meta: dict[str, tuple[str, str]] = {}
        if text_keys:
            ph = ",".join("?" * len(text_keys))
            meta.update(
                (r[0], (r[1], r[2]))
                for r in conn.execute(
                    f"SELECT node_key, paper_id, text FROM node_texts WHERE node_key IN ({ph})",
                    text_keys,
                )
            )
            missing = [k for k in text_keys if k not in meta]
            if missing:
                ph2 = ",".join("?" * len(missing))
                meta.update(
                    (r[0], (r[1], r[2]))
                    for r in conn.execute(
                        "SELECT node_id, paper_id, summary_text FROM tree_summaries "
                        f"WHERE node_id IN ({ph2})",
                        missing,
                    )
                )

        def _text_of(key: str) -> str:
            if key in rich:
                return str(rich[key].get("text") or "")
            return meta.get(key, ("", ""))[1]

        primary: list[tuple[str, float]] = cand[:top_k]
        if reranker is not None and cand:
            try:
                passages = [_text_of(k)[:2000] for k, _ in cand]
                scores = reranker.rerank(query, passages)
                if scores and len(scores) == len(cand):
                    primary = sorted(
                        ((k, float(s)) for (k, _s), s in zip(cand, scores) if s is not None),
                        key=lambda kv: kv[1],
                        reverse=True,
                    )[:top_k]
            except Exception as exc:  # noqa: BLE001 - degrade to coarse order
                log.warning("[rag-sql] rerank failed ({}); falling back to RRF order", exc)
                primary = cand[:top_k]

        # Leg diversity guarantee: bm25+vector share the same retrieval units, so
        # their double-hits mathematically crowd single-leg raptor / graph
        # entries out of the fused ranking. The (possibly reranked) head ordering
        # is untouched; when a specialised leg's best entry still missed the cut,
        # it is appended (marked) so downstream consumers see summary-level and
        # graph-level evidence.
        primary_keys = {k for k, _ in primary}
        # claims 与 graph/raptor 一样参与保底：融合头部挤不掉已沉淀结论的
        # 最佳命中（claims 条目自描述，rich 里已带元数据，渲染无需回查）。
        for name in ("raptor", "graph", "claims"):
            leg_entries = next((entries for lname, entries in legs if lname == name), [])
            best = max(leg_entries, key=lambda e: e["score"], default=None)
            if best is not None and best["key"] not in primary_keys:
                key = best["key"]
                primary.append((key, best["score"]))
                if key not in rich and key not in meta:
                    if ":" not in key:
                        row_ = conn.execute(
                            "SELECT paper_id, summary_text FROM tree_summaries WHERE node_id = ?",
                            (key,),
                        ).fetchone()
                    else:
                        row_ = conn.execute(
                            "SELECT paper_id, text FROM node_texts WHERE node_key = ?",
                            (key,),
                        ).fetchone()
                    if row_:
                        meta[key] = (row_[0], row_[1])
        rows: list[dict[str, Any]] = []
        for rank, (key, score) in enumerate(primary, start=1):
            node_id = key
            if key in rich:
                paper_id = str(rich[key].get("paper_id") or "")
                full_text = str(rich[key].get("text") or "")
                title = str(rich[key].get("title") or "")
                node_id = str(rich[key].get("node_id") or key)
            else:
                paper_id, full_text = meta.get(key, ("", ""))
                if ":" in key:
                    node_id = key.split(":", 1)[1]
                title = full_text.split("\n", 1)[0].strip()[:120]
            row: dict[str, Any] = {
                "paper_id": paper_id,
                "node_id": node_id,
                "title": title,
                "source": "sql-fusion",
                "score": round(float(score), 6),
                "text": full_text[:500],
                "legs": membership.get(key, []),
            }
            row.update(
                build_evidence_record(
                    generation=generation,
                    query=query,
                    retriever="sql-fusion",
                    rank=rank,
                    score=row["score"],
                    source={**row, "text": full_text},
                    filters=filters,
                    excerpt=str(row["text"]),
                )
            )
            rows.append(row)
        log.debug(
            "[rag-sql] {} | total={:.0f}ms",
            " ".join(leg_ms),
            (time.perf_counter() - started) * 1000,
        )
        return rows
    finally:
        conn.close()
