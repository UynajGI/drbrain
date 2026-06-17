"""Research gap analysis workflow — identifies and prioritizes open problems.

Pipeline:
1. Detect all unaddressed gaps and stale problems
2. Classify gap difficulty (limitation vs future_work vs uncategorized)
3. Score gaps by impact (paper_count + debate intensity + staleness)
4. LLM generates actionable research agenda from gap analysis
"""

from __future__ import annotations

from typing import Any

from drbrain.reasoning.base import (
    ReasoningWorkflow,
    WorkflowContext,
    WorkflowStep,
    register_workflow,
)


class _DetectGapsStep(WorkflowStep):
    """Find all unaddressed gaps and stale problems in the KG."""

    name = "detect_gaps"
    requires_llm = False

    def run(self, ctx: WorkflowContext) -> dict[str, Any]:
        seeds = ctx.graph.detect_research_seeds(ctx.db)

        unaddressed = [s for s in seeds if s.get("type") == "unaddressed_gap"]
        stale = [s for s in seeds if s.get("type") == "stale_problem"]
        debates = [s for s in seeds if s.get("type") == "debate_zone"]
        cliffs = [s for s in seeds if s.get("type") == "technology_cliff"]

        return {
            "unaddressed_gaps": unaddressed,
            "stale_problems": stale,
            "debate_zones": debates,
            "technology_cliffs": cliffs,
            "total_signals": len(seeds),
        }


class _ClassifyDifficultyStep(WorkflowStep):
    """Classify gap difficulty using analyze_difficulty()."""

    name = "classify_difficulty"
    requires_llm = False

    def run(self, ctx: WorkflowContext) -> dict[str, Any]:
        from drbrain.graph.genealogy import analyze_difficulty

        result = analyze_difficulty(ctx.db)
        return result if result else {}


class _ScoreGapsStep(WorkflowStep):
    """Score gaps by research impact potential."""

    name = "score_gaps"
    requires_llm = False

    def run(self, ctx: WorkflowContext) -> list[dict[str, Any]]:
        gaps_data = ctx.get("detect_gaps", {})
        all_gaps = (
            gaps_data.get("unaddressed_gaps", [])
            + gaps_data.get("stale_problems", [])
            + gaps_data.get("technology_cliffs", [])
        )

        scored = []
        for gap in all_gaps:
            concept = gap.get("concept") or gap.get("description", "")
            # Score: more debates around a gap = higher impact
            debate_count = sum(
                1 for d in gaps_data.get("debate_zones", []) if d.get("concept") == concept
            )
            score = min(0.3 + 0.15 * debate_count, 1.0)
            scored.append(
                {
                    "concept": concept,
                    "type": gap.get("type", "unknown"),
                    "description": gap.get("description", ""),
                    "score": round(score, 2),
                    "nearby_debates": debate_count,
                }
            )

        scored.sort(key=lambda g: g["score"], reverse=True)
        return scored[:15]


class _GenerateAgendaStep(WorkflowStep):
    """LLM generates an actionable research agenda from gap analysis."""

    name = "generate_agenda"
    requires_llm = True

    def run(self, ctx: WorkflowContext) -> str:

        from drbrain.extractor.llm_client import call_text_with_fallback

        gaps_data = ctx.get("detect_gaps", {})
        difficulty = ctx.get("classify_difficulty", {})
        scored = ctx.get("score_gaps", [])

        prompt_parts = [
            f"Question/Topic: {ctx.question}",
            "\nGap analysis summary:",
            f"  Unaddressed gaps: {len(gaps_data.get('unaddressed_gaps', []))}",
            f"  Stale problems: {len(gaps_data.get('stale_problems', []))}",
            f"  Debate zones: {len(gaps_data.get('debate_zones', []))}",
            f"  Technology cliffs: {len(gaps_data.get('technology_cliffs', []))}",
        ]

        if scored:
            prompt_parts.append("\nTop research opportunities (by impact score):")
            for g in scored[:5]:
                prompt_parts.append(
                    f"  [{g['score']:.2f}] {g['concept']} ({g['type']}): {g['description']}"
                )

        if difficulty:
            limitation_count = len(difficulty.get("limitations", []))
            future_count = len(difficulty.get("future_work", []))
            prompt_parts.append(
                f"\nDifficulty classification: {limitation_count} limitations, {future_count} future work items"
            )

        prompt_parts.append(
            "\nBased on this gap analysis, generate a prioritized research agenda:\n"
            "1. The 3 most impactful research questions to pursue\n"
            "2. Why each matters (evidence from the KG)\n"
            "3. What data/methods would be needed\n"
            "4. Estimated difficulty (low/medium/high) for each\n"
            "Be specific and grounded in the evidence."
        )

        result = call_text_with_fallback("\n".join(prompt_parts), ctx.models, max_tokens=3072)
        return result or "Unable to generate research agenda."


@register_workflow("gap-analysis")
class GapAnalysisWorkflow(ReasoningWorkflow):
    """Identifies and prioritizes open research problems from the knowledge graph."""

    name = "gap-analysis"
    description = "Analyze research gaps, classify difficulty, generate prioritized agenda"
    steps = [
        _DetectGapsStep(),
        _ClassifyDifficultyStep(),
        _ScoreGapsStep(),
        _GenerateAgendaStep(),
    ]
