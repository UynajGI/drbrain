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

__all__ = ["build_agent", "init_llamaindex_settings", "reason_llamaindex"]


def __getattr__(name):
    """Keep the core and SQL backend importable without loading agent SDKs."""
    from importlib import import_module

    if name not in __all__:
        raise AttributeError(name)
    module = "llm" if name == "init_llamaindex_settings" else "agent"
    value = getattr(import_module(f"drbrain.rag.{module}"), name)
    globals()[name] = value
    return value
