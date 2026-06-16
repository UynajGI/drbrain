"""Causal explanation workflow — answers "why does X cause/affect Y?".

Pipeline:
1. Extract entity labels from the question (BM25 search)
2. Find causal chain between source and target concepts
3. Extract mechanism text from the chain's arguments
4. Counterfactual check: remove key node, measure impact
5. LLM synthesizes a natural-language explanation from all evidence
"""

from __future__ import annotations

from typing import Any

from drbrain.reasoning.base import (
    ReasoningWorkflow,
    WorkflowContext,
    WorkflowStep,
    register_workflow,
)


class _ExtractEntitiesStep(WorkflowStep):
    """Extract source and target concept labels from the question via BM25."""

    name = "extract_entities"
    requires_llm = False

    def run(self, ctx: WorkflowContext) -> dict[str, Any]:
        from drbrain.extractor.agent_tools import search_concepts

        # Search for concepts mentioned in the question
        hits = search_concepts(ctx.db, ctx.question, limit=10)
        if not hits:
            return {"source": None, "target": None, "candidates": []}

        # Heuristic: first hit is source, look for a different one as target
        labels = [h["label"] for h in hits if h.get("label")]
        source = labels[0] if labels else None
        target = labels[1] if len(labels) > 1 else None

        return {"source": source, "target": target, "candidates": labels}


class _FindCausalChainStep(WorkflowStep):
    """Find a causal chain between the extracted source and target."""

    name = "find_causal_chain"
    requires_llm = False

    def run(self, ctx: WorkflowContext) -> dict[str, Any]:
        from drbrain.extractor.causal_chain import build_causal_chains, find_path

        entities = ctx.get("extract_entities", {})
        source = entities.get("source")
        target = entities.get("target")

        # Load all arguments from DB
        rows = ctx.db.conn.execute(
            "SELECT claim, claim_type, target_label, target_type, evidence_type, "
            "evidence_detail, mechanism, section, confidence FROM arguments"
        ).fetchall()
        from drbrain.extractor.argument import ExtractedArgument

        args = [
            ExtractedArgument(
                claim=r[0],
                claim_type=r[1],
                target=r[2],
                target_type=r[3],
                evidence_type=r[4],
                evidence_detail=r[5],
                mechanism=r[6] or "",
                section=r[7] or "",
                confidence=r[8] or 1.0,
            )
            for r in rows
        ]

        if not args:
            return {"chain": None, "all_chains": [], "args_count": 0}

        all_chains = build_causal_chains(args)

        if source and target:
            chain = find_path(args, source, target)
        elif source:
            chains = [c for c in all_chains if c.links and c.links[0].target == source]
            chain = chains[0] if chains else None
        else:
            chain = all_chains[0] if all_chains else None

        return {
            "chain": chain.summary() if chain else None,
            "chain_links": [a.claim for a in chain.links] if chain else [],
            "all_chains_count": len(all_chains),
            "args_count": len(args),
        }


class _ExtractMechanismsStep(WorkflowStep):
    """Extract mechanism descriptions from the causal chain."""

    name = "extract_mechanisms"
    requires_llm = False

    def run(self, ctx: WorkflowContext) -> list[str]:
        chain_data = ctx.get("find_causal_chain", {})
        chain = chain_data.get("chain_links", [])
        # The chain_links are claim strings; mechanisms are embedded in the
        # original arguments. For now, use the claims as mechanism descriptions.
        return chain if chain else []


class _CounterfactualCheckStep(WorkflowStep):
    """Run counterfactual analysis on the source concept."""

    name = "counterfactual_check"
    requires_llm = False

    def run(self, ctx: WorkflowContext) -> dict[str, Any]:
        from drbrain.extractor.counterfactual import run_counterfactual

        entities = ctx.get("extract_entities", {})
        source = entities.get("source")
        if not source or source not in ctx.graph.graph:
            return {"impact": None, "summary": "Source concept not in graph"}

        impact = run_counterfactual(ctx.graph, source)
        return {
            "removed_edges": impact.removed_edges,
            "affected_concepts": impact.affected_concepts,
            "lost_inferences": list(impact.lost_inferences)[:10],
            "summary": impact.summary(),
        }


class _SynthesizeExplanationStep(WorkflowStep):
    """Use LLM to synthesize a natural-language causal explanation."""

    name = "synthesize_explanation"
    requires_llm = True

    def run(self, ctx: WorkflowContext) -> str:
        import asyncio

        from drbrain.extractor.llm_client import acall_text_with_fallback

        entities = ctx.get("extract_entities", {})
        chain = ctx.get("find_causal_chain", {})
        mechanisms = ctx.get("extract_mechanisms", [])
        cf = ctx.get("counterfactual_check", {})

        prompt_parts = [
            f"Question: {ctx.question}",
            f"\nSource concept: {entities.get('source', 'unknown')}",
            f"Target concept: {entities.get('target', 'unknown')}",
        ]

        if chain.get("chain"):
            prompt_parts.append(f"\nCausal chain: {chain['chain']}")
        if mechanisms:
            prompt_parts.append("\nMechanisms / evidence:")
            for i, m in enumerate(mechanisms, 1):
                prompt_parts.append(f"  {i}. {m}")
        if cf.get("summary"):
            prompt_parts.append(f"\nCounterfactual analysis: {cf['summary']}")
            if cf.get("lost_inferences"):
                prompt_parts.append(
                    f"  Lost inferences if source removed: {', '.join(cf['lost_inferences'][:5])}"
                )

        prompt_parts.append(
            "\nBased on the above evidence from the knowledge graph, "
            "provide a clear causal explanation answering the question. "
            "Explain the mechanism, the chain of causation, and the "
            "counterfactual evidence."
        )

        prompt = "\n".join(prompt_parts)
        result = asyncio.run(acall_text_with_fallback(prompt, ctx.models, max_tokens=512))
        return result or "Unable to generate explanation."


@register_workflow("causal")
class CausalWorkflow(ReasoningWorkflow):
    """Explains causal relationships between concepts in the knowledge graph."""

    name = "causal"
    description = "Explain why X causes/affects Y using causal chains + counterfactual analysis"
    steps = [
        _ExtractEntitiesStep(),
        _FindCausalChainStep(),
        _ExtractMechanismsStep(),
        _CounterfactualCheckStep(),
        _SynthesizeExplanationStep(),
    ]
