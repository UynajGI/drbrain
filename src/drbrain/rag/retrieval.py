"""Retrieval service shared by agents and CLI adapters; no loop/session dependency."""
from __future__ import annotations
import logging
from collections.abc import Sequence
from typing import Any
from drbrain.config import Config
from drbrain.rag.evidence import build_evidence_record
from drbrain.rag.status import RetrievalUnavailableError
log = logging.getLogger(__name__)


def retrieve(cfg, db, graph, request):
    """Structured API; the historical list API remains available below."""
    from drbrain.rag.config import coerce_config

    rows = retrieve_documents(
        coerce_config(cfg), db, graph, request.query, generation=request.generation,
        filters=request.filters, top_k=request.top_k, acl_filter=request.acl_filter,
    )
    return rows.result


def _retrieval_rows(
    nodes: Sequence[Any],
    *,
    generation: str,
    query: str,
    filters: dict[str, Any] | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Map fused nodes to the historic payload plus additive evidence fields.

    R-I3: when the node metadata carries ``line_start``/``line_end`` (the
    raw.md offsets of the section whose text is shown — the parent section for
    tree/raptor leaves expanded to their parent), they travel on the row so
    settle/verify can re-locate the checksummed text. Rows without offsets in
    the metadata gain no new keys (additive-only contract).
    """
    rows: list[dict[str, Any]] = []
    for rank, nws in enumerate(nodes[:top_k], start=1):
        md = dict(nws.node.metadata or {})
        score = round(float(nws.score), 4) if nws.score is not None else None
        full_text = nws.node.get_content() or ""
        row: dict[str, Any] = {
            "paper_id": md.get("paper_id", ""),
            "node_id": md.get("node_id", ""),
            "title": md.get("title", ""),
            "source": md.get("source", ""),
            "score": score,
            "text": full_text[:500],
        }
        for key in (
            "parent_node_id",
            "parent_document_id",
            "parent_checksum",
            "char_start",
            "char_end",
            "offset_basis",
            "parent_line_start",
            "parent_line_end",
        ):
            if key in md:
                row[key] = md[key]
        line_start = md.get("line_start")
        line_end = md.get("line_end")
        if line_start is not None and line_end is not None:
            try:
                row["line_start"] = int(line_start)
                row["line_end"] = int(line_end)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                pass
        row.update(
            build_evidence_record(
                generation=generation,
                query=query,
                retriever="fusion",
                rank=rank,
                score=score,
                source={**row, "text": full_text},
                filters=filters,
                excerpt=str(row["text"]),
            )
        )
        rows.append(row)
    return rows

def retrieve_documents(
    cfg: Config,
    db: Any,
    graph: Any,
    query: str,
    *,
    generation: str | None = None,
    filters: dict[str, Any] | None = None,
    top_k: int = 5,
    acl_filter: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve RAG records, optionally constrained to one published snapshot.

    A supplied generation intentionally enables only persisted BM25/vector legs:
    the other retrievers read mutable filesystem or SQLite state and would make
    a supposedly pinned result mix index epochs.
    """
    from drbrain.rag.config import get_llamaindex_config
    from drbrain.rag.contracts import (
        RetrievalRequest,
        LegResult,
        finish_retrieval,
        matches_scope,
    )

    request = RetrievalRequest(query, top_k, generation, filters or {}, acl_filter or {})

    if getattr(get_llamaindex_config(cfg), "rag_engine", "llamaindex") == "sql":
        from drbrain.rag.sql_retrie import retrieve_documents_sql

        return retrieve_documents_sql(
            cfg,
            db,
            query,
            filters=request.filters,
            top_k=top_k,
            graph=graph,
            generation=generation,
            acl_filter=acl_filter,
        )
    try:
        from llama_index.core.schema import QueryBundle

        from drbrain.rag.fusion import build_fusion_retriever, get_retrievers
        from drbrain.rag.indexer import capture_index_generation
    except Exception as exc:
        raise RetrievalUnavailableError(f"llama-index stack unavailable: {exc}") from exc
    resolved_generation = generation or capture_index_generation(cfg)
    if resolved_generation is None:
        log.warning("[rag] retrieval unavailable (invalid active index pointer)")
        raise RetrievalUnavailableError("invalid active index pointer")
    try:
        legs = get_retrievers(
            cfg,
            db,
            graph,
            generation=resolved_generation,
            generation_backed_only=True,
        )
        if not legs:
            raise RetrievalUnavailableError("no persisted retrieval legs resolved")
        fused = build_fusion_retriever(
            cfg,
            vector_index=legs.get("vector"),
            bm25_retriever=legs.get("bm25"),
            custom_retrievers={k: v for k, v in legs.items() if k not in ("bm25", "vector")},
            top_k=max(top_k, 1000) if request.filters else top_k,
            acl_filter=acl_filter,
        )
    except RetrievalUnavailableError:
        raise
    except Exception as exc:  # pragma: no cover - depends on on-disk index state
        log.warning("[rag] retrieval unavailable (%s)", exc)
        raise RetrievalUnavailableError(str(exc)) from exc
    if fused is None:
        raise RetrievalUnavailableError("fusion retriever could not be built")
    nodes = fused.retrieve(QueryBundle(query_str=query))
    nodes = [
        node
        for node in nodes
        if matches_scope(dict(node.node.metadata or {}), request.filters, acl_filter)
    ]
    rows = _retrieval_rows(
        nodes,
        generation=resolved_generation,
        query=query,
        filters=request.filters,
        top_k=top_k,
    )
    trace = fused.get_last_trace() if hasattr(fused, "get_last_trace") else {}
    legs = [
        LegResult(
            str(item["source"]),
            "unavailable"
            if item.get("status") not in ("ok", "no_results", "empty")
            else "ok"
            if item.get("returned", 0)
            else "empty",
            int(item.get("returned", 0)),
            float(item.get("duration_ms", 0)),
            str(item.get("status", "retrieval_failure")),
        )
        for item in trace.get("legs", [])
    ]
    return finish_retrieval(
        rows,
        generation=resolved_generation,
        legs=legs,
        capabilities={"backend": "llamaindex", "snapshot": True, "vector_recall": "independent"},
    )
