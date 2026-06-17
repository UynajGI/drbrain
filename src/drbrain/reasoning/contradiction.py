"""Contradiction discovery workflow — finds cross-paper conflicting claims.

Pipeline:
1. Scan for debate concepts (concepts with both supports and challenges)
2. Build argument map: for each debate, pair supporting vs challenging arguments
3. LLM classifies whether each pair is a true contradiction or perspective difference
4. LLM summarizes the overall contradiction landscape
"""

from __future__ import annotations

from typing import Any

from drbrain.reasoning.base import (
    ReasoningWorkflow,
    WorkflowContext,
    WorkflowStep,
    register_workflow,
)


class _ScanDebatesStep(WorkflowStep):
    """Find all concepts that have both supporting and challenging arguments."""

    name = "scan_debates"
    requires_llm = False

    def run(self, ctx: WorkflowContext) -> list[dict[str, Any]]:
        from collections import defaultdict

        # Query all arguments grouped by target
        rows = ctx.db.conn.execute(
            "SELECT target_label, claim_type, claim, source_paper, confidence "
            "FROM arguments WHERE target_label IS NOT NULL AND target_label != ''"
        ).fetchall()

        by_target: dict[str, list[dict]] = defaultdict(list)
        for target_label, claim_type, claim, paper, conf in rows:
            by_target[target_label].append(
                {
                    "claim": claim,
                    "claim_type": (claim_type or "").lower().strip(),
                    "paper": paper,
                    "confidence": conf or 1.0,
                }
            )

        # Find targets with both supports and challenges
        debates = []
        for target, args in by_target.items():
            supports = [a for a in args if "support" in a["claim_type"]]
            challenges = [
                a for a in args if "challenge" in a["claim_type"] or "limit" in a["claim_type"]
            ]
            if supports and challenges:
                debates.append(
                    {
                        "concept": target,
                        "supports": supports,
                        "challenges": challenges,
                        "support_count": len(supports),
                        "challenge_count": len(challenges),
                    }
                )

        debates.sort(key=lambda d: d["support_count"] + d["challenge_count"], reverse=True)
        return debates


class _BuildArgumentMapStep(WorkflowStep):
    """Pair supporting and challenging arguments for each debate concept."""

    name = "build_argument_map"
    requires_llm = False

    def run(self, ctx: WorkflowContext) -> list[dict[str, Any]]:
        debates = ctx.get("scan_debates", [])
        if not debates:
            return []

        pairs = []
        for debate in debates:
            concept = debate["concept"]
            for supp in debate["supports"]:
                for chall in debate["challenges"]:
                    if supp["paper"] != chall["paper"]:
                        pairs.append(
                            {
                                "concept": concept,
                                "support_claim": supp["claim"],
                                "support_paper": supp["paper"],
                                "challenge_claim": chall["claim"],
                                "challenge_paper": chall["paper"],
                            }
                        )
        return pairs[:50]  # cap to avoid explosion


class _ClassifyContradictionsStep(WorkflowStep):
    """LLM classifies each pair as true contradiction or perspective difference."""

    name = "classify_contradictions"
    requires_llm = True

    def run(self, ctx: WorkflowContext) -> list[dict[str, Any]]:

        from drbrain.extractor.llm_client import call_with_fallback

        pairs = ctx.get("build_argument_map", [])
        if not pairs:
            return []

        # Batch classify up to 10 pairs in one LLM call
        batch = pairs[:10]
        pair_descriptions = []
        for i, p in enumerate(batch):
            pair_descriptions.append(
                f"Pair {i + 1} — Concept: {p['concept']}\n"
                f"  Support (paper {p['support_paper']}): {p['support_claim']}\n"
                f"  Challenge (paper {p['challenge_paper']}): {p['challenge_claim']}"
            )

        prompt = (
            "For each pair of claims below, classify whether they represent:\n"
            '- "contradiction": the claims directly conflict on the same aspect\n'
            '- "perspective": the claims address different aspects or conditions\n\n'
            'Return a JSON object: {"results": [{"pair": 1, "type": "contradiction"|"perspective", "reason": "..."}]}\n\n'
            + "\n\n".join(pair_descriptions)
        )

        data = call_with_fallback(
            prompt,
            ctx.models,
            system_prompt="You are a research contradiction analyst.",
            max_tokens=1024,
        )

        results = []
        if data and "results" in data:
            for item in data["results"]:
                idx = item.get("pair", 0) - 1
                if 0 <= idx < len(batch):
                    batch[idx]["classification"] = item.get("type", "unknown")
                    batch[idx]["reason"] = item.get("reason", "")
                    results.append(batch[idx])

        return results


class _SummarizeStep(WorkflowStep):
    """LLM generates a summary of the contradiction landscape."""

    name = "summarize"
    requires_llm = True

    def run(self, ctx: WorkflowContext) -> str:

        from drbrain.extractor.llm_client import call_text_with_fallback

        debates = ctx.get("scan_debates", [])
        classified = ctx.get("classify_contradictions", [])
        contradictions = [c for c in classified if c.get("classification") == "contradiction"]

        if not debates:
            return "No conflicting claims found in the knowledge graph."

        prompt_parts = [
            f"Found {len(debates)} concepts with conflicting evidence.",
            f"Classified {len(classified)} claim pairs: {len(contradictions)} true contradictions.",
            "\nKey contradictions:",
        ]
        for c in contradictions[:5]:
            prompt_parts.append(
                f"  - {c['concept']}: {c['support_claim'][:80]}... vs {c['challenge_claim'][:80]}..."
            )

        prompt_parts.append(
            "\nSummarize the state of conflicting evidence in this knowledge graph. "
            "Highlight the most important contradictions and what would be needed to resolve them."
        )

        result = call_text_with_fallback("\n".join(prompt_parts), ctx.models, max_tokens=2048)
        return result or "Unable to generate summary."


@register_workflow("contradiction")
class ContradictionWorkflow(ReasoningWorkflow):
    """Discovers and classifies contradictions across papers in the knowledge graph."""

    name = "contradiction"
    description = "Find cross-paper contradictions, classify true vs perspective differences"
    steps = [
        _ScanDebatesStep(),
        _BuildArgumentMapStep(),
        _ClassifyContradictionsStep(),
        _SummarizeStep(),
    ]
