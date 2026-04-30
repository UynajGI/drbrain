"""Hypothesis generation from graph patterns.

Generates research hypotheses based on:
- Unaddressed gaps: "Method M could address Gap G"
- Debate zones: "New evidence needed to resolve Conclusion C"
- Technology cliffs: "Revived method M under new conditions"
- Cross-domain isomorphisms: "Method from domain A may work in domain B"
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from drbrain.graph.engine import GraphEngine


@dataclass
class Hypothesis:
    """A generated research hypothesis."""

    description: str
    type: str
    base_confidence: float
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "type": self.type,
            "base_confidence": self.base_confidence,
            "evidence": self.evidence,
            "score": score_hypothesis(self),
        }


def score_hypothesis(hyp: Hypothesis) -> float:
    """Score a hypothesis: base_confidence + evidence bonus.

    Evidence bonus: 0.05 per evidence item, capped at 0.15.
    """
    bonus = min(len(hyp.evidence) * 0.05, 0.15)
    return round(min(hyp.base_confidence + bonus, 1.0), 3)


def detect_section_contradictions(
    graph: GraphEngine,
    section_map: dict[str, str],
) -> list[dict]:
    """Find conclusions supported in one section but challenged in another.

    Returns list of dicts with conclusion, supporting_sections, challenging_sections.
    """
    # Build: conclusion → set of (section, relation_type)
    conclusion_evidence: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for u, v, data in graph.graph.edges(data=True):
        rel = data.get("relation", "")
        if rel in ("supports", "challenges"):
            section = section_map.get(u, "")
            if section:
                conclusion_evidence[v].append((section, rel))

    contradictions = []
    for conclusion, entries in conclusion_evidence.items():
        supporting = {s for s, r in entries if r == "supports"}
        challenging = {s for s, r in entries if r == "challenges"}

        # Only report if supports and challenges come from DIFFERENT sections
        if supporting and challenging and supporting != challenging:
            contradictions.append(
                {
                    "conclusion": conclusion,
                    "supporting_sections": sorted(supporting),
                    "challenging_sections": sorted(challenging),
                }
            )

    return contradictions


def generate_hypotheses(
    graph: GraphEngine,
    section_map: dict[str, str] | None = None,
) -> list[Hypothesis]:
    """Generate research hypotheses from graph patterns.

    Args:
        graph: The knowledge graph engine.
        section_map: Optional mapping of node label → section title.
            When provided, evidence strings include section provenance.
    """
    hyps: list[Hypothesis] = []
    section_map = section_map or {}

    def _section_suffix(node: str) -> str:
        section = section_map.get(node, "")
        return f" (found in: {section} section)" if section else ""

    # Build relation indices
    edges_by_rel: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for u, v, data in graph.graph.edges(data=True):
        edges_by_rel[data["relation"]].append((u, v))

    # Pattern 1: Unaddressed gaps -> "X could address Gap Y"
    gap_nodes: set[str] = set()
    for _, dst in edges_by_rel.get("leaves_open", []):
        gap_nodes.add(dst)
    addressed: set[str] = set()
    for rel_name in ("solves", "addresses"):
        for _, dst in edges_by_rel.get(rel_name, []):
            addressed.add(dst)
    unaddressed = gap_nodes - addressed
    for gap in unaddressed:
        # Find methods that could address this gap
        methods: set[str] = set()
        for u, v in edges_by_rel.get("addresses", []):
            if v != gap:
                methods.add(u)
        for u, v in edges_by_rel.get("extends", []):
            methods.add(u)

        top_methods = list(methods)[:3]
        if top_methods:
            evidence = [
                f"Method {m} addresses related concepts{_section_suffix(m)}" for m in top_methods
            ]
            hyps.append(
                Hypothesis(
                    description=f"One of [{', '.join(top_methods)}] could address Gap '{gap}'",
                    type="gap_filling",
                    base_confidence=0.5,
                    evidence=evidence,
                )
            )
        else:
            hyps.append(
                Hypothesis(
                    description=f"Gap '{gap}' is unaddressed — new method needed",
                    type="gap_filling",
                    base_confidence=0.4,
                    evidence=[
                        f"Identified via {len(edges_by_rel.get('leaves_open', []))} leaves_open edges"
                    ],
                )
            )

    # Pattern 2: Debate zones -> "Resolution needed"
    supports_targets = {v for _, v in edges_by_rel.get("supports", [])}
    challenges_targets = {v for _, v in edges_by_rel.get("challenges", [])}
    debate_targets = supports_targets & challenges_targets
    for target in debate_targets:
        n_support = len([v for _, v in edges_by_rel["supports"] if v == target])
        n_challenge = len([v for _, v in edges_by_rel["challenges"] if v == target])
        # Collect section info from supporting/challenging papers
        support_sections = [
            section_map.get(u, "") for u, v in edges_by_rel["supports"] if v == target
        ]
        challenge_sections = [
            section_map.get(u, "") for u, v in edges_by_rel["challenges"] if v == target
        ]
        all_sections = [s for s in support_sections + challenge_sections if s]
        section_info = f" (sections: {', '.join(set(all_sections))})" if all_sections else ""
        hyps.append(
            Hypothesis(
                description=f"'{target}' has conflicting evidence ({n_support} support, {n_challenge} challenge) — resolution needed",
                type="debate_resolution",
                base_confidence=0.6,
                evidence=[f"{n_support + n_challenge} papers engaged in debate{section_info}"],
            )
        )

    # Pattern 3: Technology cliffs -> "Revival possible"
    extends_methods: set[str] = set()
    for u, v in edges_by_rel.get("extends", []):
        extends_methods.add(u)
        extends_methods.add(v)

    constraining: dict[str, str] = {}
    for u, v in edges_by_rel.get("constrains", []):
        constraining[v] = u

    for method in extends_methods:
        if method in constraining:
            gap = constraining[method]
            hyps.append(
                Hypothesis(
                    description=f"Method '{method}' may be revivable if constraint '{gap}' is relaxed",
                    type="technology_revival",
                    base_confidence=0.4,
                    evidence=[
                        f"Method '{method}' was actively extended before stalling{_section_suffix(method)}"
                    ],
                )
            )

    return hyps
