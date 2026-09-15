"""LlamaIndex query-engine adapter over the shared SQL retrieval contract."""

from __future__ import annotations

from dataclasses import asdict

from drbrain.rag.index_generations import capture_index_generation
from drbrain.rag.sql_retrie import retrieve_documents_sql


def build_sql_retriever(cfg, db, *, top_k, acl_filter=None):
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, TextNode
    from pydantic import PrivateAttr

    generation = capture_index_generation(cfg)
    if generation is None:
        # The default ``rag prepare`` publishes the unified tree generation
        # without copying the SQL working database.  A tree request can still
        # be served from that generation; anything else stays not-prepared.
        from drbrain.rag.legs import normalize_legs
        from drbrain.tree.leg import active_tree_generation

        li = getattr(cfg, "llamaindex", None)
        wanted = normalize_legs(getattr(li, "retrievers", None) if li is not None else None)
        if "tree" not in wanted.legs or active_tree_generation(cfg) is None:
            return None

    class SQLRetriever(BaseRetriever):
        _trace: dict = PrivateAttr(default_factory=dict)

        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            # Pydantic leaves an unassigned ``PrivateAttr`` as the class
            # descriptor on this subclass, so a failed retrieval used to hand
            # the descriptor to the telemetry code and crash the abstain path.
            # Initialize explicitly: an error path must still expose a trace.
            self._trace = {}

        def _retrieve(self, query_bundle):
            rows = retrieve_documents_sql(
                cfg,
                db,
                query_bundle.query_str,
                generation=generation,
                top_k=top_k,
                acl_filter=acl_filter,
            )
            result = rows.result
            self._trace = {
                "legs": [{**asdict(leg), "returned": leg.count} for leg in result.legs],
                "fusion": {"status": result.status, "returned": len(rows)},
                "capabilities": result.capabilities,
            }
            return [
                NodeWithScore(
                    node=TextNode(text=row["text"], id_=row["evidence_id"], metadata=dict(row)),
                    score=row["score"],
                )
                for row in rows
            ]

        def get_last_trace(self):
            return self._trace

    return SQLRetriever()
