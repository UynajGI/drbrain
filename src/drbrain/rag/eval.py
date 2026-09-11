"""Evaluation layer: golden set, retriever metrics, RAGAS-style metrics.

Ticket: T7 (评估体系). Depends on T3 (index layer) / T4 (fusion) / T5 (query
engine). Implements the design-doc §4.5 evaluation loop:

* :func:`build_golden_set` — semi-automated golden set construction from the
  ``test-run/papers`` corpus (query = title/abstract-derived question, relevant
  papers = source paper + same-topic papers, relevant nodes derived from
  ``tree.json`` structure). Idempotent: an existing file is left untouched
  unless ``force=True``.
* :func:`load_golden` — read the JSONL golden set, filtered by split.
* :func:`run_retriever_eval` — HitRate@K / MRR@K over the T4 fusion retriever
  (paper-level and node-level relevance), aggregated per split.
* :func:`run_ragas_eval` — self-written 4-metric prompt evaluation of the T5
  ``ask_llamaindex`` output (faithfulness / answer_relevancy /
  context_precision / answer_correctness), scored through the DrbrainLLM
  bridge (the drbrain fallback chain).
* :func:`format_eval_report` — markdown baseline report for
  ``docs/llamaindex-eval-baseline.md``.

Design decisions (llama-index-core 0.14.23):

* ``RetrieverEvaluator`` *is* importable in 0.14.23, but its ``evaluate``
  expects a single flat list of ``expected_ids`` per query (node-level only).
  Our golden set carries both paper-level and node-level relevance, and the
  framework's hit_rate/mrr semantics are awkward to bend for a fused
  multi-leg retriever — so the ticket's fallback clause is used: hit_rate/mrr
  are computed by hand (the math is a one-liner; the framework adds no value
  here).
* RAGAS is not installed (heavy dependency, optional extra per design §5) —
  the 4 metrics are self-written prompt evaluations through
  ``DrbrainLLM.complete`` (``call_text_with_fallback``, drbrain fallback chain
  intact). See :mod:`drbrain.rag.llm`.
* ``answer_correctness`` compares the generated answer against a
  ``reference_answer`` stored in the golden set — the abstract of the primary
  relevant paper (cheap, non-LLM ground truth; no golden answers were
  hand-written, keeping annotation cost low per ticket guidance).
"""

from __future__ import annotations

from drbrain.rag.eval_report import (
    format_eval_report as format_eval_report,
)

from drbrain.rag.eval_judges import (
    _prompt_faithfulness as _prompt_faithfulness,
    _prompt_answer_relevancy as _prompt_answer_relevancy,
    _prompt_context_precision as _prompt_context_precision,
    _prompt_answer_correctness as _prompt_answer_correctness,
    _parse_score as _parse_score,
    _score_metric as _score_metric,
    _context_for as _context_for,
)

from drbrain.rag.eval_metrics import (
    _node_identity as _node_identity,
    _rank_metrics as _rank_metrics,
    _aggregate_rank as _aggregate_rank,
)

from drbrain.rag.eval_data import (
    _runtime_selected as _runtime_selected,
    _ensure_eval_parent as _ensure_eval_parent,
    _safe_eval_output as _safe_eval_output,
    _write_text_atomically as _write_text_atomically,
    _append_text_atomically as _append_text_atomically,
    load_golden as load_golden,
    _paper_nodes as _paper_nodes,
    _is_authorish as _is_authorish,
    _reference_paragraph as _reference_paragraph,
    _is_content_title as _is_content_title,
    _relevant_nodes_for as _relevant_nodes_for,
    _assign_splits as _assign_splits,
    build_golden_set as build_golden_set,
)

import json
import logging
import os
import random
import re
import tempfile
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from drbrain.config import Config
from drbrain.rag.config import get_llamaindex_config
from drbrain.security import redact_sensitive_text
from drbrain.storage.paths import (
    raw_md_path,
    resolve_paper_dir,
    tree_json_path,
    writable_artifact_path,
)

try:
    from llama_index.core.schema import NodeWithScore

    _LLAMA_INDEX_AVAILABLE = True
except ImportError:  # pragma: no cover - envs without llama-index
    NodeWithScore = None  # type: ignore[assignment,misc]
    _LLAMA_INDEX_AVAILABLE = False

log = logging.getLogger(__name__)

__all__ = [
    "_LLAMA_INDEX_AVAILABLE",
    "build_golden_set",
    "format_eval_report",
    "load_golden",
    "run_ragas_eval",
    "run_retriever_eval",
]

#: Default dev/val/test split ratio for the golden set (design §4.5, 60/20/20).
DEFAULT_SPLIT_RATIO = (0.6, 0.2, 0.2)
#: Fixed shuffle seed so split assignment is deterministic across runs
#: (idempotent regeneration and reproducible baselines).
_SPLIT_SEED = 20260812
#: Cap on reference answers (abstracts) — enough text for a correctness check.
_REFERENCE_MAX_CHARS = 800
#: Cap on a single context chunk handed to the scoring LLM.
_CONTEXT_CHUNK_MAX_CHARS = 1500
#: Titles treated as "content" nodes when deriving relevant_nodes.
_CONTENT_TITLE_PREFIXES = (
    "abstract",
    "summary",
    "overview",
    "introduction",
    "results",
    "discussion",
    "conclusion",
    "experimental",
    "methods",
    "materials",
    "section ",
)
#: Titles preferred as the ``reference_answer`` source (abstract first).
_ABSTRACT_TITLE_PREFIXES = ("abstract", "summary")


































def _coerce_cfg(cfg: Config | dict[str, Any]) -> Config:
    """Compatibility alias for the shared boundary conversion."""
    from drbrain.rag.config import coerce_config

    return coerce_config(cfg)


def run_retriever_eval(
    cfg: Config | dict[str, Any],
    db: Any,
    split: str = "dev",
    ks: Sequence[int] = (5, 10),
    top_k: int | None = None,
    max_queries: int | None = None,
) -> dict[str, Any]:
    """HitRate@K / MRR@K of the T4 fusion retriever over the golden ``split``.

    Retrieves each golden query once through ``build_hybrid_retriever``
    (BM25 + vector + configured custom legs) with ``top_k=max(ks)`` and scores
    paper-level and node-level relevance. Returns per-query rows plus the
    aggregated means. Status ``unavailable`` when no fusion retriever can be
    built (no index / llamaindex disabled); ``empty`` when the split has no
    golden queries.
    """
    cfg = _coerce_cfg(cfg)
    golden = load_golden(cfg, split=split)
    if not golden:
        return {"status": "empty", "split": split, "queries": 0}
    if max_queries:
        golden = golden[: int(max_queries)]

    from drbrain.rag.engine import build_hybrid_retriever

    k_max = max(int(k) for k in ks)
    fusion = build_hybrid_retriever(cfg, db, top_k=k_max)
    if fusion is None:
        return {
            "status": "unavailable",
            "split": split,
            "reason": "no fusion retriever (no index built or llamaindex disabled)",
        }

    rows: list[dict[str, Any]] = []
    for item in golden:
        try:
            nodes = fusion.retrieve(item["query"])[:k_max]
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("[rag] retriever eval failed for query %.60r: %s", item["query"], exc)
            nodes = []
        rows.append(_rank_metrics(nodes, item, ks))

    agg = _aggregate_rank(rows)
    agg.update({"status": "ok", "split": split, "per_query": rows})
    return agg


# ── RAGAS-style generation metrics (self-written prompts) ───────────────────
















def _coerce_llm_cfg(cfg: Config | dict[str, Any]) -> Any:
    """Minimal object bearing ``llm.models`` for ``DrbrainLLM`` from a dict.

    Mirrors T6's dict/Config dual-form support: CLI tests pass plain dicts,
    while the CLI itself always passes a real :class:`Config`.
    """
    if not isinstance(cfg, dict):
        return cfg
    from types import SimpleNamespace

    return SimpleNamespace(
        llm=SimpleNamespace(models=list((cfg.get("llm") or {}).get("models") or [])),
        api=SimpleNamespace(cache_ttl=(cfg.get("api") or {}).get("cache_ttl") or 0),
        dirs=SimpleNamespace(cache=(cfg.get("dirs") or {}).get("cache", "data/cache")),
    )


def run_ragas_eval(
    cfg: Config | dict[str, Any],
    db: Any,
    split: str = "val",
    n: int = 10,
    top_k: int = 5,
    max_queries: int | None = None,
) -> dict[str, Any]:
    """RAGAS-style 4-metric evaluation of the T5 ``ask_llamaindex`` output.

    For each of the first ``n`` golden queries of ``split`` (deterministic
    order), synthesizes an answer with
    :func:`~drbrain.rag.engine.ask_llamaindex` and scores it with four
    self-written prompt metrics through the ``DrbrainLLM`` bridge (drbrain
    fallback chain intact):

    * ``faithfulness`` — answer claims vs retrieved context;
    * ``answer_relevancy`` — answer vs question;
    * ``context_precision`` — retrieved context vs question;
    * ``answer_correctness`` — answer vs golden ``reference_answer`` (omitted
      when the golden entry carries none).

    Status ``unavailable`` when llamaindex cannot be used; ``empty`` when the
    split has no golden queries.
    """
    cfg = _coerce_cfg(cfg)
    golden = load_golden(cfg, split=split)
    if not golden:
        return {"status": "empty", "split": split, "queries": 0}
    if not _LLAMA_INDEX_AVAILABLE:
        return {
            "status": "unavailable",
            "split": split,
            "reason": "llama-index not installed",
        }
    if max_queries:
        golden = golden[: int(max_queries)]
    sample = golden[: int(n)]

    from drbrain.rag.engine import ask_llamaindex, build_hybrid_retriever
    from drbrain.rag.llm import DrbrainLLM

    llm = DrbrainLLM(_coerce_llm_cfg(cfg))
    fusion = build_hybrid_retriever(cfg, db, top_k=top_k)
    if fusion is None:
        return {
            "status": "unavailable",
            "split": split,
            "reason": "no fusion retriever (no index built or llamaindex disabled)",
        }

    rows: list[dict[str, Any]] = []
    for item in sample:
        question = item.get("query", "")
        try:
            result = ask_llamaindex(cfg, db, question, top_k=top_k, streaming=False)
            answer = str(result.get("answer") or "")
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("[rag] ask_llamaindex failed for %.60r: %s", question, exc)
            answer = ""
        try:
            nodes = fusion.retrieve(question)[:top_k]
        except Exception:  # pragma: no cover - defensive
            nodes = []
        context = _context_for(nodes)
        reference = item.get("reference_answer") or ""
        row: dict[str, Any] = {
            "query": question,
            "answer_len": len(answer),
            "context_nodes": len(nodes),
            "faithfulness": _score_metric(llm, _prompt_faithfulness(question, answer, context)),
            "answer_relevancy": _score_metric(llm, _prompt_answer_relevancy(question, answer)),
            "context_precision": _score_metric(llm, _prompt_context_precision(question, context)),
            "answer_correctness": (
                _score_metric(llm, _prompt_answer_correctness(question, answer, reference))
                if reference
                else None
            ),
        }
        rows.append(row)

    metrics: dict[str, dict[str, Any]] = {}
    for key in ("faithfulness", "answer_relevancy", "context_precision", "answer_correctness"):
        values = [r[key] for r in rows if r[key] is not None]
        metrics[key] = {
            "mean": round(sum(values) / len(values), 4) if values else None,
            "missing": sum(1 for r in rows if r[key] is None),
        }
    return {
        "status": "ok",
        "split": split,
        "queries": len(rows),
        "metrics": metrics,
        "per_query": rows,
    }


# ── baseline report ──────────────────────────────────────────────────────────




def _semantic_answer_text(result: Any) -> tuple[str, str | None]:
    """Extract a scorable answer from the stable ``ask_llamaindex`` result.

    ``ask_llamaindex(..., streaming=False)`` has always returned a result
    dictionary.  Retrieval abstentions deliberately keep explanatory prose in
    ``answer``, so evaluators must consult their machine-readable ``status``
    before embedding that text.  A bare string remains accepted for legacy
    adapters used by downstream callers.
    """
    if isinstance(result, str):
        return result.strip(), None
    if not isinstance(result, dict):
        return "", "invalid_answer"
    status = result.get("status")
    if status and str(status) != "ok":
        return "", str(status)
    answer = result.get("answer")
    return (str(answer).strip() if answer is not None else ""), None


def run_semantic_eval(
    cfg: Config | dict[str, Any],
    db: Any,
    split: str = "val",
    n: int = 30,
    top_k: int = 5,
) -> dict[str, Any]:
    """Embedding-cosine similarity of synthesized answers vs golden references.

    Zero-LLM (per LlamaIndex ``SemanticSimilarityEvaluator`` semantics): the
    answer and the golden ``reference_answer`` are embedded through the
    configured drbrain embed provider (persistent 0.6B service when
    ``embed.provider=openai-compat``) and scored by cosine similarity. Cheap
    enough to run as a regression gate after every library merge.

    Requires golden entries with a ``reference_answer``; entries without one
    are skipped (counted as ``missing``).
    """
    cfg = _coerce_cfg(cfg)
    golden = load_golden(cfg, split=split)
    if not golden:
        return {"status": "empty", "split": split, "queries": 0}
    golden = [g for g in golden if g.get("reference_answer")][: int(n)]
    if not golden:
        return {"status": "empty", "split": split, "reason": "no reference_answer in golden"}

    from drbrain.rag.engine import ask_llamaindex
    from drbrain.services.embedding import _embed_batch

    embed_cfg = getattr(cfg, "embed", None)
    scores: list[float] = []
    missing = 0
    failed_queries = 0
    failure_counts: dict[str, int] = {}
    for item in golden:
        try:
            result = ask_llamaindex(cfg, db, item["query"], top_k=top_k, streaming=False)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("[rag] semantic eval ask failed for %.60r: %s", item["query"], exc)
            missing += 1
            failed_queries += 1
            reason = f"ask:{type(exc).__name__}"
            failure_counts[reason] = failure_counts.get(reason, 0) + 1
            continue
        text, failure = _semantic_answer_text(result)
        if failure:
            missing += 1
            failed_queries += 1
            failure_counts[failure] = failure_counts.get(failure, 0) + 1
            continue
        if not text:
            missing += 1
            continue
        try:
            vecs = _embed_batch([text, str(item["reference_answer"])], embed_cfg)
            if not vecs or len(vecs) != 2:
                missing += 1
                failed_queries += 1
                failure_counts["embedding_invalid"] = failure_counts.get("embedding_invalid", 0) + 1
                continue
            import numpy as np

            a, b = np.asarray(vecs[0], dtype="float32"), np.asarray(vecs[1], dtype="float32")
            cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
            scores.append(cos)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("[rag] semantic eval embed failed: %s", exc)
            missing += 1
            failed_queries += 1
            reason = f"embedding:{type(exc).__name__}"
            failure_counts[reason] = failure_counts.get(reason, 0) + 1

    if not scores:
        return {
            "status": "empty",
            "split": split,
            "reason": "no scorable answers",
            "missing": missing,
            "failed_queries": failed_queries,
            "failure_counts": failure_counts,
        }
    mean = sum(scores) / len(scores)
    return {
        "status": "ok",
        "split": split,
        "queries": len(golden),
        "scored": len(scores),
        "missing": missing,
        "failed_queries": failed_queries,
        "failure_counts": failure_counts,
        "mean_similarity": round(mean, 4),
        "pass_rate": round(sum(1 for s in scores if s >= 0.8) / len(scores), 4),
        "threshold": 0.8,
    }


# ── QA pair generation (one-off LLM cost, reusable golden expansion) ─────────


def run_qagen(
    cfg: Config | dict[str, Any],
    n_nodes: int = 25,
    num_questions_per_chunk: int = 2,
    out_path: str | None = None,
) -> dict[str, Any]:
    """Generate retrieval QA pairs from indexed nodes via LlamaIndex
    ``generate_question_context_pairs`` and merge them into the golden set.

    One-off LLM cost (tokenrouter/fallback chain); the generated pairs are
    appended to the golden JSONL as split ``generated`` so later retriever
    evals can use ``--split generated`` for a statistically thicker test.
    """
    cfg = _coerce_cfg(cfg)
    if not _LLAMA_INDEX_AVAILABLE:
        return {"status": "unavailable", "reason": "llama-index not installed"}

    from drbrain.rag.indexer import load_index

    index, _bm25 = load_index(cfg)
    if index is None:
        return {"status": "unavailable", "reason": "no vector index (run: drbrain rag index)"}

    nodes = index.docstore.docs.values() if hasattr(index.docstore, "docs") else []
    nodes = list(nodes)[: int(n_nodes)]
    if not nodes:
        return {"status": "empty", "reason": "no nodes in index docstore"}

    try:
        from llama_index.core.evaluation import DatasetGenerator
    except ImportError:
        return {
            "status": "unavailable",
            "reason": "qagen dependency missing: install llama-index-core",
        }

    models = list(getattr(cfg.llm, "models", []) or [])
    llm = None
    if models:
        try:
            from llama_index.llms.openai import OpenAI as LIOpenAI
        except ImportError:
            return {
                "status": "unavailable",
                "reason": "qagen dependency missing: install llama-index-llms-openai",
            }

        m = models[0]
        # Keep YAML-key compatibility for qagen, but never invent a fake key:
        # a missing key should be resolved by the SDK's environment/provider
        # configuration (or fail with a useful, redacted error).  ``api_keys``
        # is reduced to one ephemeral value only for this live client; it is
        # not included in any generated record or returned payload.
        api_key = m.get("api_key")
        if not api_key and isinstance(m.get("api_keys"), list):
            api_key = next((key for key in m["api_keys"] if key), None)
        llm_kwargs: dict[str, Any] = {
            "model": str(m.get("model", "gpt-4o-mini")),
            "api_base": str(m.get("base_url") or "https://api.openai.com/v1").replace("/v1", ""),
            "temperature": 0.1,
        }
        if api_key:
            llm_kwargs["api_key"] = str(api_key)
        try:
            llm = LIOpenAI(**llm_kwargs)
        except Exception as exc:
            reason = redact_sensitive_text(str(exc)) or "LLM initialization failed"
            return {"status": "error", "reason": f"LLM initialization failed: {reason}"}
    try:
        # llama-index-core >=0.14 removed ``generate_question_context_pairs``;
        # DatasetGenerator is the replacement. Generate per node so each
        # question keeps its source node id (relevant_docs semantics).
        queries: dict[str, str] = {}
        relevant_docs: dict[str, list[str]] = {}
        for node in nodes:
            gen = DatasetGenerator(
                [node], llm=llm, num_questions_per_chunk=int(num_questions_per_chunk)
            )
            for q in gen.generate_questions_from_nodes():
                qid = f"{node.node_id}:{len(queries)}"
                queries[qid] = str(q)
                relevant_docs[qid] = [str(node.node_id)]
    except Exception as exc:
        reason = redact_sensitive_text(str(exc)) or "generation failed"
        return {"status": "error", "reason": f"generation failed: {reason}"}

    li = get_llamaindex_config(cfg)
    golden_path = Path(out_path) if out_path else Path(li.eval.golden_set)
    records: list[str] = []
    for q, ctx_ids in zip(queries.values(), relevant_docs.values()):
        if not str(q).strip():
            continue
        records.append(
            json.dumps(
                {
                    "query": str(q).strip(),
                    "relevant_papers": [],
                    "relevant_nodes": [str(i) for i in ctx_ids],
                    "reference_answer": "",
                    "split": "generated",
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    if records:
        _append_text_atomically(golden_path, "".join(records))
    appended = len(records)
    return {
        "status": "ok",
        "generated": appended,
        "nodes_used": len(nodes),
        "golden_set": str(golden_path),
        "note": "eval with: drbrain rag eval --metrics retriever --split generated",
    }
