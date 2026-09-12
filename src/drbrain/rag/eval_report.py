"""Human-readable evaluation reports from explicit measured results."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from drbrain.config import Config
from drbrain.rag.config import get_llamaindex_config


def format_eval_report(
    cfg: Config | dict[str, Any],
    retriever: dict[str, Any] | None = None,
    ragas: dict[str, Any] | None = None,
) -> str:
    """Render a markdown baseline report (timestamped) for the eval doc."""
    li = get_llamaindex_config(cfg)
    lines = [
        f"## LlamaIndex RAG 评估基线 — {datetime.now().isoformat(timespec='seconds')}",
        "",
        "### 配置",
        f"- golden_set: `{li.eval.golden_set}`;split 选项: {li.eval.split}",
        f"- enabled={li.enabled} · retrievers={li.retrievers} · fusion_mode={li.fusion_mode}"
        f" · rerank={li.rerank} · similarity_cutoff={li.similarity_cutoff}",
        f"- embed_model: `{getattr(getattr(cfg, 'embed', None), 'model', 'n/a')}`",
        "",
    ]

    if retriever is not None:
        lines.append("### Retriever eval(HitRate@K / MRR@K)")
        lines.append("")
        lines.append(
            f"- status: `{retriever.get('status')}`;split: `{retriever.get('split')}`"
            f";queries: {retriever.get('queries', 0)}"
        )
        if retriever.get("status") == "ok":
            lines.append("")
            lines.append(
                "| level | metric | " + " | ".join(f"K={k}" for k in retriever["ks"]) + " |"
            )
            lines.append("| --- | --- | " + " | ".join("---" for _ in retriever["ks"]) + " |")
            for level in ("paper", "node"):
                for metric in ("hit_rate", "mrr"):
                    vals = retriever[metric][level]
                    lines.append(
                        f"| {level} | {metric} | "
                        + " | ".join(str(vals[str(k)]) for k in retriever["ks"])
                        + " |"
                    )
        if retriever.get("reason"):
            lines.append("")
            lines.append(f"- reason: {retriever['reason']}")
        lines.append("")

    if ragas is not None:
        lines.append("### RAGAS-style eval(自写 4 指标 prompt)")
        lines.append("")
        lines.append(
            f"- status: `{ragas.get('status')}`;split: `{ragas.get('split')}`"
            f";queries: {ragas.get('queries', 0)}"
        )
        if ragas.get("status") == "ok":
            lines.append("")
            lines.append("| metric | mean | missing |")
            lines.append("| --- | --- | --- |")
            for key, info in ragas["metrics"].items():
                mean = info["mean"] if info["mean"] is not None else "n/a"
                lines.append(f"| {key} | {mean} | {info['missing']} |")
        if ragas.get("reason"):
            lines.append("")
            lines.append(f"- reason: {ragas['reason']}")
        lines.append("")

    return "\n".join(lines) + "\n"


# ── Semantic similarity eval (zero-LLM, embedding cosine) ────────────────────
