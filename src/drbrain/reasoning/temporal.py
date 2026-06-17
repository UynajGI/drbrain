"""Temporal narrative workflow — answers "how did concept X evolve over time?".

Pipeline:
1. Build year-by-year timeline (frequency + confidence + papers)
2. Detect turning points (paradigm shifts, confidence collapse)
3. Trace concept lineage (ancestors and descendants)
4. LLM synthesizes a coherent narrative from temporal data
"""

from __future__ import annotations

from typing import Any

from drbrain.reasoning.base import (
    ReasoningWorkflow,
    WorkflowContext,
    WorkflowStep,
    register_workflow,
)


class _BuildTimelineStep(WorkflowStep):
    """Aggregate concept mentions by year with frequency and confidence."""

    name = "build_timeline"
    requires_llm = False

    def run(self, ctx: WorkflowContext) -> dict[str, Any]:
        from drbrain.extractor.agent_tools import search_concepts

        # Find the concept from the question
        hits = search_concepts(ctx.db, ctx.question, limit=1)
        if not hits:
            return {"concept": None, "timeline": [], "evolution": []}

        concept = hits[0]["label"]
        evolution = ctx.db.get_concept_evolution(concept)

        timeline = []
        for entry in evolution:
            timeline.append(
                {
                    "year": entry["year"],
                    "count": entry["count"],
                    "avg_confidence": round(entry.get("avg_conf", 0), 3),
                    "trend": entry.get("trend", "stable"),
                }
            )

        return {"concept": concept, "timeline": timeline, "evolution": evolution}


class _DetectTurningPointsStep(WorkflowStep):
    """Detect paradigm shifts and confidence collapse for the concept."""

    name = "detect_turning_points"
    requires_llm = False

    def run(self, ctx: WorkflowContext) -> dict[str, Any]:
        from drbrain.graph.genealogy import detect_paradigm_shifts

        timeline_data = ctx.get("build_timeline", {})
        concept = timeline_data.get("concept")

        if not concept:
            return {"shifts": [], "signal": None}

        # Get the concept's evolution signal
        signal = ctx.db.get_concept_signal(concept)

        # Detect paradigm shifts involving this concept
        all_shifts = detect_paradigm_shifts(ctx.graph, ctx.db)
        relevant = [s for s in all_shifts if s.get("concept", "").lower() == concept.lower()]

        return {
            "shifts": relevant,
            "signal": signal.get("signal") if signal else None,
            "first_seen": signal.get("first_seen") if signal else None,
            "last_seen": signal.get("last_seen") if signal else None,
        }


class _TraceLineageStep(WorkflowStep):
    """Build ancestor/descendant tree for the concept."""

    name = "trace_lineage"
    requires_llm = False

    def run(self, ctx: WorkflowContext) -> dict[str, Any]:
        from drbrain.graph.genealogy import evolve_concept

        timeline_data = ctx.get("build_timeline", {})
        concept = timeline_data.get("concept")

        if not concept:
            return {"ancestors": [], "descendants": []}

        ancestors = evolve_concept(ctx.graph, ctx.db, concept, direction="ancestors")
        descendants = evolve_concept(ctx.graph, ctx.db, concept, direction="descendants")

        def _extract_labels(tree: list[dict]) -> list[str]:
            labels = []
            for node in tree:
                labels.append(node.get("label", ""))
                labels.extend(_extract_labels(node.get("children", [])))
            return [lbl for lbl in labels if lbl]

        return {
            "ancestors": _extract_labels(ancestors),
            "descendants": _extract_labels(descendants),
        }


class _GenerateNarrativeStep(WorkflowStep):
    """LLM synthesizes temporal data into a coherent narrative."""

    name = "generate_narrative"
    requires_llm = True

    def run(self, ctx: WorkflowContext) -> str:

        from drbrain.extractor.llm_client import call_text_with_fallback

        timeline_data = ctx.get("build_timeline", {})
        turning = ctx.get("detect_turning_points", {})
        lineage = ctx.get("trace_lineage", {})

        concept = timeline_data.get("concept")
        if not concept:
            return "Concept not found in the knowledge graph."

        timeline = timeline_data.get("timeline", [])
        prompt_parts = [
            f"Question: {ctx.question}",
            f"\nConcept: {concept}",
            f"Evolution signal: {turning.get('signal', 'unknown')}",
        ]

        if timeline:
            prompt_parts.append("\nYear-by-year data:")
            for t in timeline:
                prompt_parts.append(
                    f"  {t['year']}: {t['count']} papers, "
                    f"confidence={t['avg_confidence']}, trend={t['trend']}"
                )

        if turning.get("shifts"):
            prompt_parts.append(f"\nParadigm shifts detected: {len(turning['shifts'])}")
            for s in turning["shifts"][:3]:
                prompt_parts.append(
                    f"  - {s.get('shift_type', 'unknown')}: {s.get('description', '')}"
                )

        if lineage.get("ancestors"):
            prompt_parts.append(f"\nAncestral concepts: {', '.join(lineage['ancestors'][:5])}")
        if lineage.get("descendants"):
            prompt_parts.append(f"Descendant concepts: {', '.join(lineage['descendants'][:5])}")

        prompt_parts.append(
            "\nWrite a coherent narrative explaining how this concept evolved over time. "
            "Identify key turning points, explain what caused shifts, and describe the "
            "current trajectory. Write in an academic but accessible style."
        )

        result = call_text_with_fallback("\n".join(prompt_parts), ctx.models, max_tokens=3072)
        return result or "Unable to generate narrative."


@register_workflow("temporal")
class TemporalWorkflow(ReasoningWorkflow):
    """Generates a temporal narrative of how a concept evolved over time."""

    name = "temporal"
    description = "Trace concept evolution over time with turning points and narrative"
    steps = [
        _BuildTimelineStep(),
        _DetectTurningPointsStep(),
        _TraceLineageStep(),
        _GenerateNarrativeStep(),
    ]
