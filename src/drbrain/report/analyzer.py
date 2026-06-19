"""Knowledge frontier analysis — orchestrates reasoning modules into unified report."""

from __future__ import annotations

from drbrain.graph.engine import GraphEngine
from drbrain.storage.database import Database


def analyze_paper(
    db: Database,
    graph: GraphEngine,
    local_id: str,
    *,
    full: bool = False,
    models: list[dict] | None = None,
) -> dict:
    """Run all reasoning modules on a single paper. Returns structured report."""
    paper = db.get_paper(local_id)
    if not paper:
        return {"error": f"Paper not found: {local_id}"}

    paper_concepts = {c["label"] for c in db.get_concepts_by_paper(local_id)}

    report: dict = {
        "paper": {
            "local_id": local_id,
            "title": paper.get("title", ""),
            "year": paper.get("year"),
        },
    }

    # Seeds (always)
    seeds = graph.detect_research_seeds(db)
    relevant_seeds = [s for s in seeds if s.get("concept")]
    report["seeds"] = relevant_seeds[:10]

    # Causal chains — per-concept lookup
    from drbrain.extractor.argument import ExtractedArgument

    raw_args = db.get_arguments_by_paper(local_id)
    parsed_args = [
        ExtractedArgument(
            claim=r["claim"],
            claim_type=r["claim_type"],
            target=r["target_label"],
            target_type=r["target_type"],
            mechanism=r.get("mechanism", ""),
            section=r.get("section", ""),
            confidence=r.get("confidence", 1.0),
            evidence_type=r.get("evidence_type") or "",
            evidence_detail=r.get("evidence_detail") or "",
        )
        for r in raw_args
    ]

    from drbrain.extractor.causal_chain import find_chains_from

    chains: list[dict] = []
    for concept in list(paper_concepts)[:5]:
        for c in find_chains_from(parsed_args, concept):
            chains.append(
                {
                    "source": c.links[0].target if c.links else "",
                    "target": c.links[-1].target if c.links else "",
                    "via": getattr(c, "mechanism", ""),
                }
            )
    report["causal_chains"] = chains[:10]

    if full:
        # Counterfactual
        from drbrain.extractor.counterfactual import find_critical_nodes

        critical = find_critical_nodes(graph, top_n=5)
        # Filter to nodes that are in this paper's concepts
        report["critical_nodes"] = [n for n in critical if n.get("node") in paper_concepts][:5]

        # Hypotheses
        from drbrain.extractor.hypothesis import generate_hypotheses

        hypotheses = generate_hypotheses(graph)
        report["hypotheses"] = [
            {"description": h.description, "type": h.type, "confidence": h.base_confidence}
            for h in hypotheses[:10]
        ]

        # Isomorphism
        from drbrain.extractor.isomorphism import find_isomorphic_patterns

        isomorphisms = find_isomorphic_patterns(graph)
        report["isomorphisms"] = [
            {
                "source": getattr(iso, "source", ""),
                "target": getattr(iso, "target", ""),
                "similarity": getattr(iso, "similarity", 0.0),
            }
            for iso in isomorphisms[:5]
        ]

    # Counts
    closure_edges = graph.closure()
    report["summary"] = {
        "seeds": len(report["seeds"]),
        "causal_chains": len(report["causal_chains"]),
        "inferred_edges": len(closure_edges),
        "critical_nodes": len(report.get("critical_nodes", [])),
        "hypotheses": len(report.get("hypotheses", [])),
        "isomorphisms": len(report.get("isomorphisms", [])),
    }

    if models:
        import asyncio

        report["executive_summary"] = asyncio.run(_generate_executive_summary(report, models))

        # Enhance seeds with LLM-suggested solution directions
        if report["seeds"]:
            report["seeds"] = asyncio.run(
                _enhance_seeds(report["seeds"], paper.get("title", ""), models)
            )

        # Graph Summary — LLM description of top concepts' subgraphs
        from drbrain.services.graph_to_text import describe_subgraph

        all_concepts = db.get_concepts_by_paper(local_id)
        top_concepts = sorted(all_concepts, key=lambda c: c.get("confidence", 0), reverse=True)[:3]
        graph_descriptions: list[str] = []
        for concept in top_concepts:
            desc = asyncio.run(describe_subgraph(graph, db, concept["label"], models, depth=1))
            if desc:
                graph_descriptions.append(f"{concept['label']}: {desc}")
        report["graph_summary"] = "\n\n".join(graph_descriptions) if graph_descriptions else ""

    return report


def add_cross_paper_insights(reports: list[dict], db=None) -> list[dict]:
    """Find method migration opportunities across analyzed papers.

    Detects where a Method from one paper could address a Problem
    in another paper, using concept label similarity via embedding.
    """
    if len(reports) < 2 or not db:
        return reports

    # Collect concepts by type across all papers
    methods: list[dict] = []  # {label, paper_id, paper_title}
    problems: list[dict] = []

    for r in reports:
        pid = r["paper"]["local_id"]
        ptitle = r["paper"]["title"]
        for c in db.get_concepts_by_paper(pid):
            if c["type"] == "Method":
                methods.append({"label": c["label"], "paper_id": pid, "paper_title": ptitle[:60]})
            elif c["type"] == "Problem":
                problems.append({"label": c["label"], "paper_id": pid, "paper_title": ptitle[:60]})

    # Find cross-paper method→problem pairs
    insights: list[dict] = []
    from drbrain.extractor.concept import _label_similarity

    for m in methods:
        for p in problems:
            if m["paper_id"] == p["paper_id"]:
                continue  # same paper, skip
            sim = _label_similarity(m["label"], p["label"])
            if sim > 0.3:  # weak similarity threshold
                insights.append(
                    {
                        "method": m["label"],
                        "method_paper": m["paper_title"],
                        "problem": p["label"],
                        "problem_paper": p["paper_title"],
                        "similarity": round(sim, 3),
                    }
                )

    insights.sort(key=lambda x: x["similarity"], reverse=True)
    if insights:
        reports[0]["cross_paper_insights"] = insights[:10]
    return reports


async def _generate_executive_summary(report: dict, models: list[dict]) -> str:
    """Generate a 3-5 sentence executive summary of the analysis report."""
    from drbrain.extractor.llm_client import acall_text_with_fallback

    # Build a compact summary of what was found
    summary = report.get("summary", {})
    paper = report.get("paper", {})
    seeds = report.get("seeds", [])
    chains = report.get("causal_chains", [])
    hypotheses = report.get("hypotheses", [])

    seed_text = ", ".join(s.get("description", "")[:80] for s in seeds[:3]) or "none"
    chain_text = ", ".join(f"{c['source']} -> {c['target']}" for c in chains[:3]) or "none"
    hypo_text = ", ".join(h.get("description", "")[:80] for h in hypotheses[:3]) or "none"

    prompt = (
        f"Paper: {paper.get('title', 'Unknown')} ({paper.get('year', '?')})\n"
        f"Research seeds found: {summary.get('seeds', 0)} ({seed_text})\n"
        f"Causal chains: {summary.get('causal_chains', 0)} ({chain_text})\n"
        f"Hypotheses: {summary.get('hypotheses', 0)} ({hypo_text})\n"
        f"Inferred edges: {summary.get('inferred_edges', 0)}\n\n"
        "Write a 3-5 sentence executive summary of this knowledge frontier analysis. "
        "Focus on the most important findings and actionable insights. "
        "Be concise and direct. Return plain text, no markdown."
    )

    result = await acall_text_with_fallback(prompt, models, max_tokens=300)
    return result.strip() if result else ""


async def _enhance_seeds(seeds: list[dict], paper_title: str, models: list[dict]) -> list[dict]:
    """Add LLM-suggested solution directions to each research seed."""
    from drbrain.extractor.llm_client import acall_text_with_fallback

    for seed in seeds[:5]:  # only top 5 get enhanced
        prompt = (
            f"Paper: {paper_title}\n"
            f"Research gap: {seed.get('description', '')}\n\n"
            "Suggest 1-2 concrete research directions or methods that could address this gap. "
            "Be specific. One sentence per direction. Return plain text, no markdown."
        )
        result = await acall_text_with_fallback(prompt, models, max_tokens=150)
        if result:
            seed["suggested_solutions"] = result.strip()

    return seeds
