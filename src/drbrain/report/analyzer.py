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
    relevant_seeds = [s for s in seeds if s.get("node") and s["node"] in paper_concepts]
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
                    "source": c.source,
                    "target": c.target,
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

    return report
