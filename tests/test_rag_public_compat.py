"""Compatibility guard for the supported RAG API surface.

The project policy is additive-only: existing imports, parameter names, default
values, CLI commands, status values, and serialized model fields may not be
removed or renamed.  New optional parameters and new fields remain allowed.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import fields
from typing import Any

_REQUIRED = object()


PUBLIC_SYMBOLS: dict[str, tuple[str, ...]] = {
    "drbrain.rag": ("build_agent", "init_llamaindex_settings", "reason_llamaindex"),
    "drbrain.rag.agent": (
        "AgentFunctionLLM",
        "build_agent",
        "load_session_history",
        "reason_llamaindex",
    ),
    "drbrain.rag.authority": ("ResolvedClaim", "authority_rank", "is_stale", "resolve_claims"),
    "drbrain.rag.config": ("get_llamaindex_config",),
    "drbrain.rag.engine": (
        "SimilarityCutoffPostprocessor",
        "ask_llamaindex",
        "build_hybrid_retriever",
        "build_query_engine",
        "extract_sources",
        "nodes_to_paper_results",
        "resolve_engine",
    ),
    "drbrain.rag.eval": (
        "build_golden_set",
        "format_eval_report",
        "load_golden",
        "run_qagen",
        "run_ragas_eval",
        "run_retriever_eval",
        "run_semantic_eval",
    ),
    "drbrain.rag.fusion": ("FusionRetriever", "build_fusion_retriever", "get_retrievers"),
    "drbrain.rag.indexer": ("build_index", "collect_tree_nodes", "get_index_health", "load_index"),
    "drbrain.rag.llm": ("DrbrainEmbedding", "DrbrainLLM", "init_llamaindex_settings"),
    "drbrain.rag.mcp_tools": ("call_mcp_tool", "discover_mcp_tools", "load_mcp_tools"),
    "drbrain.rag.rerank": (
        "CrossEncoderReranker",
        "DeduplicatePostprocessor",
        "RerankPostprocessor",
        "build_reranker",
        "kendall_tau",
        "mean_rank_displacement",
        "top_k_overlap",
    ),
    "drbrain.rag.retrievers": (
        "DrbrainGraphRetriever",
        "DrbrainRAPTORRetriever",
        "DrbrainTreeRetriever",
    ),
    "drbrain.rag.status": ("RetrievalError", "RetrievalStatus", "classify_failure"),
    "drbrain.cli.rag_commands": ("rag_app", "rag_eval_cmd", "rag_health_cmd", "rag_index_cmd"),
}


# (parameter name, inspect.Parameter.kind.name, default or _REQUIRED).  Each
# historical parameter is frozen; later additions must remain optional.
SIGNATURES: dict[str, tuple[tuple[str, str, Any], ...]] = {
    "drbrain.rag.agent.build_agent": (
        ("cfg", "POSITIONAL_OR_KEYWORD", _REQUIRED),
        ("db", "POSITIONAL_OR_KEYWORD", None),
        ("session_id", "POSITIONAL_OR_KEYWORD", None),
        ("graph", "KEYWORD_ONLY", None),
        ("closure_context", "KEYWORD_ONLY", ""),
        ("temperature", "KEYWORD_ONLY", 0.3),
        ("max_tokens", "KEYWORD_ONLY", 1024),
        ("include_retrieval", "KEYWORD_ONLY", True),
        ("plugins_dir", "KEYWORD_ONLY", None),
        ("mcp_servers", "KEYWORD_ONLY", None),
    ),
    "drbrain.rag.agent.load_session_history": (
        ("db", "POSITIONAL_OR_KEYWORD", _REQUIRED),
        ("session_id", "POSITIONAL_OR_KEYWORD", _REQUIRED),
        ("token_budget", "POSITIONAL_OR_KEYWORD", 8000),
    ),
    "drbrain.rag.agent.reason_llamaindex": (
        ("cfg", "POSITIONAL_OR_KEYWORD", _REQUIRED),
        ("db", "POSITIONAL_OR_KEYWORD", None),
        ("question", "POSITIONAL_OR_KEYWORD", ""),
        ("max_turns", "POSITIONAL_OR_KEYWORD", 5),
        ("session_id", "POSITIONAL_OR_KEYWORD", None),
        ("graph", "KEYWORD_ONLY", None),
        ("closure_context", "KEYWORD_ONLY", ""),
    ),
    "drbrain.rag.authority.authority_rank": (("authority", "POSITIONAL_OR_KEYWORD", _REQUIRED),),
    "drbrain.rag.authority.is_stale": (
        ("claim", "POSITIONAL_OR_KEYWORD", _REQUIRED),
        ("now", "POSITIONAL_OR_KEYWORD", None),
    ),
    "drbrain.rag.authority.resolve_claims": (
        ("claims", "POSITIONAL_OR_KEYWORD", _REQUIRED),
        ("now", "KEYWORD_ONLY", None),
    ),
    "drbrain.rag.config.get_llamaindex_config": (("cfg", "POSITIONAL_OR_KEYWORD", None),),
    "drbrain.rag.engine.resolve_engine": (
        ("cfg", "POSITIONAL_OR_KEYWORD", _REQUIRED),
        ("requested", "POSITIONAL_OR_KEYWORD", "llamaindex"),
    ),
    "drbrain.rag.engine.build_query_engine": (
        ("cfg", "POSITIONAL_OR_KEYWORD", _REQUIRED),
        ("db", "POSITIONAL_OR_KEYWORD", _REQUIRED),
        ("streaming", "POSITIONAL_OR_KEYWORD", None),
        ("top_k", "POSITIONAL_OR_KEYWORD", None),
        ("acl_filter", "POSITIONAL_OR_KEYWORD", None),
    ),
    "drbrain.rag.engine.build_hybrid_retriever": (
        ("cfg", "POSITIONAL_OR_KEYWORD", _REQUIRED),
        ("db", "POSITIONAL_OR_KEYWORD", _REQUIRED),
        ("top_k", "POSITIONAL_OR_KEYWORD", None),
        ("acl_filter", "POSITIONAL_OR_KEYWORD", None),
    ),
    "drbrain.rag.engine.extract_sources": (("nodes", "POSITIONAL_OR_KEYWORD", _REQUIRED),),
    "drbrain.rag.engine.nodes_to_paper_results": (
        ("nodes", "POSITIONAL_OR_KEYWORD", _REQUIRED),
        ("top_k", "POSITIONAL_OR_KEYWORD", None),
    ),
    "drbrain.rag.engine.ask_llamaindex": (
        ("cfg", "POSITIONAL_OR_KEYWORD", _REQUIRED),
        ("db", "POSITIONAL_OR_KEYWORD", _REQUIRED),
        ("question", "POSITIONAL_OR_KEYWORD", _REQUIRED),
        ("top_k", "POSITIONAL_OR_KEYWORD", 5),
        ("streaming", "POSITIONAL_OR_KEYWORD", True),
        ("acl_filter", "POSITIONAL_OR_KEYWORD", None),
    ),
    "drbrain.rag.eval.load_golden": (
        ("cfg", "POSITIONAL_OR_KEYWORD", None),
        ("split", "POSITIONAL_OR_KEYWORD", None),
    ),
    "drbrain.rag.eval.build_golden_set": (
        ("cfg", "POSITIONAL_OR_KEYWORD", None),
        ("papers_dir", "POSITIONAL_OR_KEYWORD", None),
        ("force", "POSITIONAL_OR_KEYWORD", False),
        ("out_path", "POSITIONAL_OR_KEYWORD", None),
        ("query_ids", "POSITIONAL_OR_KEYWORD", None),
    ),
    "drbrain.rag.eval.run_retriever_eval": (
        ("cfg", "POSITIONAL_OR_KEYWORD", _REQUIRED),
        ("db", "POSITIONAL_OR_KEYWORD", _REQUIRED),
        ("split", "POSITIONAL_OR_KEYWORD", "dev"),
        ("ks", "POSITIONAL_OR_KEYWORD", (5, 10)),
        ("top_k", "POSITIONAL_OR_KEYWORD", None),
        ("max_queries", "POSITIONAL_OR_KEYWORD", None),
    ),
    "drbrain.rag.eval.run_ragas_eval": (
        ("cfg", "POSITIONAL_OR_KEYWORD", _REQUIRED),
        ("db", "POSITIONAL_OR_KEYWORD", _REQUIRED),
        ("split", "POSITIONAL_OR_KEYWORD", "val"),
        ("n", "POSITIONAL_OR_KEYWORD", 10),
        ("top_k", "POSITIONAL_OR_KEYWORD", 5),
        ("max_queries", "POSITIONAL_OR_KEYWORD", None),
    ),
    "drbrain.rag.eval.format_eval_report": (
        ("cfg", "POSITIONAL_OR_KEYWORD", _REQUIRED),
        ("retriever", "POSITIONAL_OR_KEYWORD", None),
        ("ragas", "POSITIONAL_OR_KEYWORD", None),
    ),
    "drbrain.rag.eval.run_semantic_eval": (
        ("cfg", "POSITIONAL_OR_KEYWORD", _REQUIRED),
        ("db", "POSITIONAL_OR_KEYWORD", _REQUIRED),
        ("split", "POSITIONAL_OR_KEYWORD", "val"),
        ("n", "POSITIONAL_OR_KEYWORD", 30),
        ("top_k", "POSITIONAL_OR_KEYWORD", 5),
    ),
    "drbrain.rag.eval.run_qagen": (
        ("cfg", "POSITIONAL_OR_KEYWORD", _REQUIRED),
        ("n_nodes", "POSITIONAL_OR_KEYWORD", 25),
        ("num_questions_per_chunk", "POSITIONAL_OR_KEYWORD", 2),
        ("out_path", "POSITIONAL_OR_KEYWORD", None),
    ),
    "drbrain.rag.indexer.collect_tree_nodes": (
        ("paper_dir", "POSITIONAL_OR_KEYWORD", _REQUIRED),
        ("tree_json", "POSITIONAL_OR_KEYWORD", None),
        ("max_node_tokens", "POSITIONAL_OR_KEYWORD", None),
    ),
    "drbrain.rag.indexer.build_index": (
        ("cfg", "POSITIONAL_OR_KEYWORD", _REQUIRED),
        ("db", "POSITIONAL_OR_KEYWORD", _REQUIRED),
        ("paper_ids", "POSITIONAL_OR_KEYWORD", None),
        ("force", "POSITIONAL_OR_KEYWORD", False),
        ("embed_model", "POSITIONAL_OR_KEYWORD", None),
        ("max_node_tokens", "POSITIONAL_OR_KEYWORD", None),
    ),
    "drbrain.rag.indexer.load_index": (
        ("cfg", "POSITIONAL_OR_KEYWORD", _REQUIRED),
        ("embed_model", "POSITIONAL_OR_KEYWORD", None),
    ),
    "drbrain.rag.indexer.get_index_health": (("cfg", "POSITIONAL_OR_KEYWORD", _REQUIRED),),
    "drbrain.rag.llm.init_llamaindex_settings": (("cfg", "POSITIONAL_OR_KEYWORD", _REQUIRED),),
    "drbrain.rag.mcp_tools.discover_mcp_tools": (("server", "POSITIONAL_OR_KEYWORD", _REQUIRED),),
    "drbrain.rag.mcp_tools.call_mcp_tool": (
        ("server", "POSITIONAL_OR_KEYWORD", _REQUIRED),
        ("tool_name", "POSITIONAL_OR_KEYWORD", _REQUIRED),
        ("arguments", "POSITIONAL_OR_KEYWORD", _REQUIRED),
    ),
    "drbrain.rag.mcp_tools.load_mcp_tools": (("servers", "POSITIONAL_OR_KEYWORD", _REQUIRED),),
}


def _resolve(path: str) -> Any:
    module_name, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), name)


def test_rag_public_symbols_remain_importable():
    for module_name, names in PUBLIC_SYMBOLS.items():
        module = importlib.import_module(module_name)
        for name in names:
            assert hasattr(module, name), f"removed or renamed public RAG API: {module_name}.{name}"


def test_rag_function_signatures_are_additive_only():
    for path, expected in SIGNATURES.items():
        actual = list(inspect.signature(_resolve(path)).parameters.values())
        assert len(actual) >= len(expected), f"removed parameters from public RAG API: {path}"
        for parameter, (name, kind, default) in zip(actual[: len(expected)], expected, strict=True):
            assert (parameter.name, parameter.kind.name) == (name, kind), path
            if default is _REQUIRED:
                assert parameter.default is inspect.Parameter.empty, path
            else:
                assert parameter.default == default, path
        for parameter in actual[len(expected) :]:
            assert parameter.default is not inspect.Parameter.empty, (
                f"new public parameter must be optional: {path}.{parameter.name}"
            )


def test_rag_data_contracts_remain_additive_only():
    from drbrain.config import LlamaIndexConfig, LlamaIndexEvalConfig
    from drbrain.rag.authority import ResolvedClaim
    from drbrain.rag.status import RetrievalStatus

    assert [member.value for member in RetrievalStatus] == [
        "ok",
        "no_results",
        "retrieval_failure",
        "permission_denied",
        "timeout",
        "source_unavailable",
        "insufficient_evidence",
    ]
    assert [field.name for field in fields(ResolvedClaim)][:9] == [
        "label",
        "value",
        "authority",
        "provenance",
        "confidence",
        "valid_from",
        "valid_to",
        "resolution",
        "reason",
    ]
    config = LlamaIndexConfig()
    assert (
        config.enabled,
        config.llm,
        config.vector_store,
        config.storage_dir,
        config.retrievers,
        config.fusion_mode,
        config.rerank,
        config.rerank_model,
        config.rerank_top_k,
        config.similarity_cutoff,
        config.streaming,
        config.max_node_tokens,
    ) == (
        False,
        "litellm",
        "memory",
        "data/llamaindex",
        ["bm25", "vector"],
        "reciprocal_rank",
        True,
        "Qwen/Qwen3-Reranker-0.6B",
        20,
        0.7,
        True,
        4000,
    )
    assert isinstance(config.eval, LlamaIndexEvalConfig)
    assert (config.eval.golden_set, config.eval.split) == (
        "data/llamaindex/golden.jsonl",
        ["dev", "val", "test"],
    )


def test_rag_cli_commands_remain_registered():
    from drbrain.cli.rag_commands import rag_app

    names = {command.name for command in rag_app.registered_commands}
    assert {"index", "health", "eval"} <= names
