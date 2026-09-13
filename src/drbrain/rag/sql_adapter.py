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
        return None

    class SQLRetriever(BaseRetriever):
        _trace: dict = PrivateAttr(default_factory=dict)

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
