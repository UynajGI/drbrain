"""Hypothesis generation workflow — generates cross-domain research hypotheses.

Pipeline:
1. Find cross-domain isomorphism patterns (disconnected subgraphs sharing problems)
2. Find transfer candidates (methods that could address problems in other domains)
3. LLM generates analogy-based hypothesis descriptions
4. KG validates hypothesis consistency (TBox/RBox + pattern detection)
5. Score with confidence propagation
"""

from __future__ import annotations

from typing import Any

from drbrain.reasoning.base import (
    ReasoningWorkflow,
    WorkflowContext,
    WorkflowStep,
    register_workflow,
)


class _FindCrossDomainStep(WorkflowStep):
    """Detect cross-domain isomorphism patterns in the graph."""

    name = "find_cross_domain"
    requires_llm = False

    def run(self, ctx: WorkflowContext) -> list[dict[str, Any]]:
        seeds = ctx.graph.detect_research_seeds(ctx.db)
        cross_domain = [s for s in seeds if s.get("type") == "cross_domain_isomorphism"]

        # Also check for unaddressed gaps that could inspire hypotheses
        gaps = [s for s in seeds if s.get("type") == "unaddressed_gap"]

        return {
            "cross_domain_patterns": cross_domain[:10],
            "unaddressed_gaps": gaps[:10],
            "total_seeds": len(seeds),
        }


class _FindTransferCandidatesStep(WorkflowStep):
    """Find method→problem transfer opportunities between domains."""

    name = "find_transfer_candidates"
    requires_llm = False

    def run(self, ctx: WorkflowContext) -> list[dict[str, Any]]:
        from drbrain.graph.genealogy import find_transfer_opportunities

        transfers = find_transfer_opportunities(ctx.db, ctx.graph)
        return transfers[:20] if transfers else []


class _AnalogizeStep(WorkflowStep):
    """LLM generates analogy-based hypothesis descriptions from transfer candidates."""

    name = "analogize"
    requires_llm = True

    def run(self, ctx: WorkflowContext) -> list[dict[str, Any]]:

        from drbrain.extractor.llm_client import call_with_fallback

        transfers = ctx.get("find_transfer_candidates", [])
        cross_domain = ctx.get("find_cross_domain", {})
        gaps = cross_domain.get("unaddressed_gaps", [])

        if not transfers and not gaps:
            return []

        # Build prompt for hypothesis generation
        prompt_parts = [
            "Based on the knowledge graph analysis below, generate novel research hypotheses.",
            "Each hypothesis should be specific, testable, and grounded in the evidence.\n",
        ]

        if transfers:
            prompt_parts.append("Transfer opportunities (method → problem):")
            for t in transfers[:5]:
                prompt_parts.append(
                    f"  - {t.get('source_concept', '?')} could address "
                    f"{t.get('target_concept', '?')} "
                    f"(confidence: {t.get('confidence', '?')})"
                )

        if gaps:
            prompt_parts.append("\nUnaddressed gaps:")
            for g in gaps[:5]:
                prompt_parts.append(f"  - {g.get('description', g.get('concept', 'unknown'))}")

        prompt_parts.append(
            '\nReturn JSON: {"hypotheses": [{"description": "...", '
            '"type": "transfer|gap_filling|cross_domain", '
            '"source_evidence": "brief justification"}]}'
        )

        data = call_with_fallback(
            "\n".join(prompt_parts),
            ctx.models,
            system_prompt="You are a creative but rigorous research hypothesis generator.",
            max_tokens=1024,
        )

        if data and "hypotheses" in data:
            return data["hypotheses"][:10]
        return []


class _ValidateStep(WorkflowStep):
    """KG validates each hypothesis for consistency."""

    name = "validate"
    requires_llm = False

    def run(self, ctx: WorkflowContext) -> list[dict[str, Any]]:
        from drbrain.extractor.agent_tools import kg_validate

        hypotheses = ctx.get("analogize", [])
        validated = []

        for hyp in hypotheses:
            description = hyp.get("description", "")
            result = kg_validate(description, db=ctx.db, graph=ctx.graph)
            validated.append(
                {
                    **hyp,
                    "consistent": result.get("consistent", True),
                    "violations": result.get("violations", []),
                    "patterns": result.get("patterns", []),
                }
            )

        return validated


class _ScoreStep(WorkflowStep):
    """Score hypotheses with confidence propagation."""

    name = "score"
    requires_llm = False

    def run(self, ctx: WorkflowContext) -> list[dict[str, Any]]:
        from drbrain.extractor.hypothesis import Hypothesis, score_hypothesis

        validated = ctx.get("validate", [])
        scored = []

        for hyp in validated:
            # Build a Hypothesis object for scoring
            evidence = []
            if hyp.get("source_evidence"):
                evidence.append(hyp["source_evidence"])
            if hyp.get("consistent"):
                evidence.append("KG validation passed")
            for v in hyp.get("violations", []):
                evidence.append(f"Violation: {v.get('reason', '')}")

            h = Hypothesis(
                description=hyp.get("description", ""),
                type=hyp.get("type", "unknown"),
                base_confidence=0.5 if hyp.get("consistent", True) else 0.3,
                evidence=evidence,
            )
            hyp["score"] = score_hypothesis(h)
            scored.append(hyp)

        scored.sort(key=lambda h: h.get("score", 0), reverse=True)
        return scored


@register_workflow("hypothesis")
class HypothesisWorkflow(ReasoningWorkflow):
    """Generates and validates novel cross-domain research hypotheses."""

    name = "hypothesis"
    description = "Generate cross-domain hypotheses with KG validation and confidence scoring"
    steps = [
        _FindCrossDomainStep(),
        _FindTransferCandidatesStep(),
        _AnalogizeStep(),
        _ValidateStep(),
        _ScoreStep(),
    ]
