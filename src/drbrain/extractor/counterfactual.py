"""Counterfactual queries: 'what if X didn't exist?'

For a given node, compute the impact of its removal on the graph:
- Edges directly removed
- Inferred relationships lost (comparing full vs reduced closure)
- Affected downstream nodes
"""

from __future__ import annotations

from dataclasses import dataclass, field

from drbrain.graph.engine import GraphEngine


@dataclass
class CounterfactualImpact:
    """Result of removing a node from the graph."""

    removed_node: str
    removed_edges: int = 0
    affected_concepts: int = 0
    lost_inferences: set[str] = field(default_factory=set)
    affected_nodes: set[str] = field(default_factory=set)

    def summary(self) -> str:
        return (
            f"Removing '{self.removed_node}': "
            f"{self.removed_edges} edges removed, "
            f"{self.affected_concepts} concepts affected, "
            f"lost inferences: {', '.join(sorted(self.lost_inferences)) or 'none'}"
        )


def run_counterfactual(graph: GraphEngine, node: str) -> CounterfactualImpact:
    """Simulate removal of a node and measure downstream impact.

    Compares closure inferences with and without the node.
    """
    impact = CounterfactualImpact(removed_node=node)

    if node not in graph.graph:
        return impact

    # Count direct edges involving this node
    direct_edges = [(u, v, d) for u, v, d in graph.graph.edges(data=True) if u == node or v == node]
    impact.removed_edges = len(direct_edges)

    if not direct_edges:
        return impact

    # Get full closure inferences
    full_inferred = graph.closure()
    full_rel_set = {(e["src"], e["dst"], e["relation"]) for e in full_inferred}

    # Build reduced graph without the node
    reduced = GraphEngine()
    for u, v, data in graph.graph.edges(data=True):
        if u == node or v == node:
            continue
        reduced.add_edge(u, v, data["relation"], data["source"], data.get("weight", 1.0))

    # Get reduced closure inferences
    reduced_inferred = reduced.closure()
    reduced_rel_set = {(e["src"], e["dst"], e["relation"]) for e in reduced_inferred}

    # Find lost inferences
    lost = full_rel_set - reduced_rel_set
    impact.lost_inferences = {rel for _, _, rel in lost}

    # Find affected nodes (nodes that appear in lost inferences)
    for src, dst, rel in lost:
        impact.affected_nodes.add(src)
        impact.affected_nodes.add(dst)

    impact.affected_concepts = len(impact.affected_nodes)

    return impact


def find_critical_nodes(graph: GraphEngine, top_n: int = 10) -> list[dict]:
    """Rank nodes by counterfactual impact.

    Returns list of {node, impact} dicts sorted by impact descending.
    """
    if graph.graph.number_of_nodes() == 0:
        return []

    scores: dict[str, int] = {}
    for node in graph.graph.nodes():
        impact = run_counterfactual(graph, node)
        # Score: edges removed + concepts affected + lost inferences
        scores[node] = impact.removed_edges + impact.affected_concepts + len(impact.lost_inferences)

    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"node": n, "impact": s} for n, s in sorted_nodes[:top_n]]
