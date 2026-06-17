"""Impact assessment workflow — evaluates concept/paper influence on the KG.

Pipeline:
1. Identify critical nodes via counterfactual analysis
2. Measure each node's downstream influence (affected concepts + lost inferences)
3. Rank by weighted impact (section-aware importance)
4. LLM generates an impact narrative with rankings and explanations
"""

from __future__ import annotations

from typing import Any

from drbrain.reasoning.base import (
    ReasoningWorkflow,
    WorkflowContext,
    WorkflowStep,
    register_workflow,
)


class _FindCriticalNodesStep(WorkflowStep):
    """Rank nodes by counterfactual impact (what happens if removed)."""

    name = "find_critical_nodes"
    requires_llm = False

    def run(self, ctx: WorkflowContext) -> list[dict[str, Any]]:
        from drbrain.extractor.counterfactual import find_critical_nodes

        if ctx.graph.graph.number_of_nodes() == 0:
            return []

        nodes = find_critical_nodes(ctx.graph, top_n=15)
        return nodes


class _MeasureInfluenceStep(WorkflowStep):
    """Deep-dive counterfactual on top nodes to measure downstream impact."""

    name = "measure_influence"
    requires_llm = False

    def run(self, ctx: WorkflowContext) -> list[dict[str, Any]]:
        from drbrain.extractor.counterfactual import run_counterfactual

        critical = ctx.get("find_critical_nodes", [])
        if not critical:
            return []

        detailed = []
        for node_info in critical[:5]:
            node = node_info.get("node", "")
            if not node or node not in ctx.graph.graph:
                continue

            impact = run_counterfactual(ctx.graph, node)
            # Get concept type for context
            type_row = ctx.db.conn.execute(
                "SELECT type FROM concepts WHERE label = ? LIMIT 1", (node,)
            ).fetchone()
            concept_type = type_row[0] if type_row else "unknown"

            detailed.append(
                {
                    "node": node,
                    "type": concept_type,
                    "removed_edges": impact.removed_edges,
                    "affected_concepts": impact.affected_concepts,
                    "lost_inferences": list(impact.lost_inferences)[:5],
                    "impact_score": node_info.get("impact", 0),
                    "summary": impact.summary(),
                }
            )

        return detailed


class _DetectEvolutionStep(WorkflowStep):
    """Check for paradigm shifts and confidence trends around critical nodes."""

    name = "detect_evolution"
    requires_llm = False

    def run(self, ctx: WorkflowContext) -> dict[str, Any]:
        from drbrain.graph.genealogy import detect_paradigm_shifts

        if ctx.graph.graph.number_of_edges() == 0:
            return {"shifts": [], "signals": []}

        shifts = detect_paradigm_shifts(ctx.graph, ctx.db)

        # Get evolution signals for top critical nodes
        critical = ctx.get("find_critical_nodes", [])
        signals = []
        for node_info in critical[:5]:
            node = node_info.get("node", "")
            if not node:
                continue
            sig = ctx.db.get_concept_signal(node)
            if sig:
                signals.append(
                    {
                        "concept": node,
                        "signal": sig.get("signal", "unknown"),
                        "first_seen": sig.get("first_seen"),
                        "last_seen": sig.get("last_seen"),
                        "paper_count": sig.get("paper_count", 0),
                    }
                )

        return {"shifts": shifts[:5], "signals": signals}


class _GenerateImpactReportStep(WorkflowStep):
    """LLM generates a structured impact assessment report."""

    name = "generate_impact_report"
    requires_llm = True

    def run(self, ctx: WorkflowContext) -> str:

        from drbrain.extractor.llm_client import call_text_with_fallback

        critical = ctx.get("measure_influence", [])
        evolution = ctx.get("detect_evolution", {})

        prompt_parts = [
            f"Question: {ctx.question}",
            f"\nKnowledge graph: {ctx.graph.graph.number_of_nodes()} concepts, "
            f"{ctx.graph.graph.number_of_edges()} relations",
        ]

        if critical:
            prompt_parts.append("\nMost influential concepts (by counterfactual impact):")
            for c in critical[:5]:
                prompt_parts.append(
                    f"  [{c['impact_score']}] {c['node']} ({c['type']}): "
                    f"{c['removed_edges']} edges, {c['affected_concepts']} affected concepts, "
                    f"{len(c['lost_inferences'])} lost inferences"
                )
                if c.get("lost_inferences"):
                    prompt_parts.append(f"    Lost: {', '.join(c['lost_inferences'][:3])}")

        if evolution.get("shifts"):
            prompt_parts.append(f"\nParadigm shifts detected: {len(evolution['shifts'])}")
            for s in evolution["shifts"][:3]:
                prompt_parts.append(
                    f"  - {s.get('shift_type', '?')}: {s.get('description', '')[:80]}"
                )

        if evolution.get("signals"):
            prompt_parts.append("\nEvolution signals:")
            for sig in evolution["signals"][:3]:
                prompt_parts.append(
                    f"  - {sig['concept']}: {sig['signal']} "
                    f"({sig['first_seen']}–{sig['last_seen']}, {sig['paper_count']} papers)"
                )

        prompt_parts.append(
            "\nGenerate an impact assessment report:\n"
            "1. Rank the most influential concepts/methods\n"
            "2. Explain why each is critical (what breaks without it)\n"
            "3. Identify rising vs declining concepts\n"
            "4. Highlight concepts at paradigm shift points\n"
            "Ground all claims in the data provided."
        )

        result = call_text_with_fallback("\n".join(prompt_parts), ctx.models, max_tokens=2048)
        return result or "Unable to generate impact report."


@register_workflow("impact")
class ImpactWorkflow(ReasoningWorkflow):
    """Evaluates concept and method influence using counterfactual analysis."""

    name = "impact"
    description = "Assess concept/method impact via counterfactual analysis and evolution trends"
    steps = [
        _FindCriticalNodesStep(),
        _MeasureInfluenceStep(),
        _DetectEvolutionStep(),
        _GenerateImpactReportStep(),
    ]
