"""Literature review workflow — generates a structured survey from the KG.

Pipeline:
1. Collect papers and build concept statistics
2. Identify key themes via research seed detection
3. Extract causal chains as evidence backbone
4. LLM synthesizes a structured literature review
"""

from __future__ import annotations

from typing import Any

from drbrain.reasoning.base import (
    ReasoningWorkflow,
    WorkflowContext,
    WorkflowStep,
    register_workflow,
)


class _CollectPapersStep(WorkflowStep):
    """Gather paper metadata and concept distribution."""

    name = "collect_papers"
    requires_llm = False

    def run(self, ctx: WorkflowContext) -> dict[str, Any]:
        papers = ctx.db.get_all_papers()
        # Concept type distribution
        type_rows = ctx.db.conn.execute(
            "SELECT type, COUNT(*) FROM concepts GROUP BY type ORDER BY COUNT(*) DESC"
        ).fetchall()
        type_dist = {r[0]: r[1] for r in type_rows}

        # Year range
        year_rows = ctx.db.conn.execute(
            "SELECT MIN(year), MAX(year) FROM papers WHERE year IS NOT NULL"
        ).fetchone()

        return {
            "paper_count": len(papers),
            "paper_titles": [p.get("title", "")[:80] for p in papers[:20]],
            "concept_types": type_dist,
            "year_range": (year_rows[0], year_rows[1]) if year_rows and year_rows[0] else None,
            "edge_count": ctx.graph.graph.number_of_edges(),
            "node_count": ctx.graph.graph.number_of_nodes(),
        }


class _IdentifyThemesStep(WorkflowStep):
    """Detect key research themes via seed detection."""

    name = "identify_themes"
    requires_llm = False

    def run(self, ctx: WorkflowContext) -> list[dict[str, Any]]:
        seeds = ctx.graph.detect_research_seeds(ctx.db)
        # Group by type for structured output
        by_type: dict[str, list[dict]] = {}
        for seed in seeds:
            stype = seed.get("type", "unknown")
            by_type.setdefault(stype, []).append(seed)

        return [
            {"type": stype, "count": len(items), "top_items": items[:3]}
            for stype, items in sorted(by_type.items(), key=lambda x: -len(x[1]))
        ]


class _ExtractEvidenceStep(WorkflowStep):
    """Build causal chains as the evidence backbone for the review."""

    name = "extract_evidence"
    requires_llm = False

    def run(self, ctx: WorkflowContext) -> dict[str, Any]:
        from drbrain.extractor.argument import ExtractedArgument
        from drbrain.extractor.causal_chain import build_causal_chains

        rows = ctx.db.conn.execute(
            "SELECT claim, claim_type, target_label, target_type, "
            "evidence_type, evidence_detail, mechanism, section, confidence "
            "FROM arguments WHERE mechanism IS NOT NULL AND mechanism != ''"
        ).fetchall()

        args = [
            ExtractedArgument(
                claim=r[0],
                claim_type=r[1],
                target=r[2],
                target_type=r[3],
                evidence_type=r[4],
                evidence_detail=r[5],
                mechanism=r[6],
                section=r[7] or "",
                confidence=r[8] or 1.0,
            )
            for r in rows
        ]

        chains = build_causal_chains(args)
        return {
            "argument_count": len(args),
            "chain_count": len(chains),
            "top_chains": [c.summary() for c in chains[:5]],
        }


class _GenerateReviewStep(WorkflowStep):
    """LLM synthesizes a structured literature review."""

    name = "generate_review"
    requires_llm = True

    def run(self, ctx: WorkflowContext) -> str:

        from drbrain.extractor.llm_client import call_text_with_fallback

        papers = ctx.get("collect_papers", {})
        themes = ctx.get("identify_themes", [])
        evidence = ctx.get("extract_evidence", {})

        prompt_parts = [
            f"Topic/Question: {ctx.question}",
            f"\nCorpus: {papers.get('paper_count', 0)} papers, "
            f"{papers.get('node_count', 0)} concepts, "
            f"{papers.get('edge_count', 0)} relations",
        ]

        if papers.get("year_range"):
            yr = papers["year_range"]
            prompt_parts.append(f"Year range: {yr[0]}–{yr[1]}")

        if papers.get("concept_types"):
            types = ", ".join(f"{k}({v})" for k, v in papers["concept_types"].items())
            prompt_parts.append(f"Concept distribution: {types}")

        if themes:
            prompt_parts.append("\nResearch landscape signals:")
            for theme in themes[:5]:
                prompt_parts.append(f"  - {theme['type']}: {theme['count']} found")

        if evidence.get("top_chains"):
            prompt_parts.append("\nKey causal evidence chains:")
            for chain in evidence["top_chains"]:
                prompt_parts.append(f"  - {chain}")

        prompt_parts.append(
            "\nWrite a structured literature review covering:\n"
            "1. Overview of the research landscape\n"
            "2. Key themes and debates\n"
            "3. Methodological evolution\n"
            "4. Open problems and gaps\n"
            "5. Future directions\n"
            "Base claims strictly on the evidence provided."
        )

        result = call_text_with_fallback("\n".join(prompt_parts), ctx.models, max_tokens=2048)
        return result or "Unable to generate review."


@register_workflow("review")
class ReviewWorkflow(ReasoningWorkflow):
    """Generates a structured literature review from the knowledge graph."""

    name = "review"
    description = "Generate a structured survey/review from KG papers and evidence"
    steps = [
        _CollectPapersStep(),
        _IdentifyThemesStep(),
        _ExtractEvidenceStep(),
        _GenerateReviewStep(),
    ]
