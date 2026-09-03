"""DrBrain RAG layer built on LlamaIndex (work-in-progress).

Replaces homogeneous retrieval/synthesis/agent/eval implementations with
LlamaIndex while preserving drbrain-only assets (PageIndex tree, RAPTOR,
knowledge graph, SQLite). See ``docs/llamaindex-integration-design.md``.

Ticket ownership:
    T1 (this) — deps, config, Settings init, package skeleton
    T2 — rag/llm.py LLM bridge          T3 — rag/indexer.py
    T4 — rag/retrievers.py + fusion     T5 — rag/engine.py
    T6 — rag/agent.py                   T7 — rag/eval.py
    T8 — rag/rerank.py
"""

from drbrain.rag.agent import build_agent, reason_llamaindex
from drbrain.rag.llm import init_llamaindex_settings

__all__ = ["build_agent", "init_llamaindex_settings", "reason_llamaindex"]
