"""Cross-domain isomorphism detection.

Finds structurally similar subgraphs across disconnected domains:
- Problems with similar incoming relation patterns
- Methods with similar outgoing relation patterns
- Suggests knowledge transfer opportunities
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from drbrain.graph.engine import GraphEngine


@dataclass
class IsomorphicMapping:
    """A discovered structural isomorphism between two domains."""

    source_domain: str
    target_domain: str
    shared_structure: str
    confidence: float = 0.0


def _relation_signature(
    graph: GraphEngine,
    node: str,
    section_map: dict[str, str] | None = None,
) -> dict[str, int]:
    """Build a relation signature for a node: {relation_type: count}.

    Args:
        graph: The knowledge graph engine.
        node: Node label to build signature for.
        section_map: Optional mapping of node label → section title.
            If provided, signature keys include section info
            (e.g. "in:supports@Methods").
    """
    sig: dict[str, int] = defaultdict(int)
    for u, v, data in graph.graph.edges(data=True):
        if v == node:
            key = f"in:{data['relation']}"
            if section_map:
                section = section_map.get(u, "")
                if section:
                    key = f"{key}@{section}"
            sig[key] += 1
        elif u == node:
            key = f"out:{data['relation']}"
            if section_map:
                section = section_map.get(v, "")
                if section:
                    key = f"{key}@{section}"
            sig[key] += 1
    return dict(sig)


def find_similar_problems(
    graph: GraphEngine, problem: str, min_similarity: float = 0.8
) -> list[str]:
    """Find Problems with similar relation signatures.

    Uses Jaccard similarity on relation signatures.
    """
    if problem not in graph.graph:
        return []

    target_sig = _relation_signature(graph, problem)
    if not target_sig:
        return []

    # Find all nodes that are targets of 'addresses' or 'solves'
    problem_nodes: set[str] = set()
    for u, v, data in graph.graph.edges(data=True):
        if v != problem and data["relation"] in ("addresses", "solves"):
            problem_nodes.add(v)

    similar_list: list[str] = []

    for candidate in problem_nodes:
        cand_sig = _relation_signature(graph, candidate)
        if not cand_sig:
            continue

        # Jaccard similarity on (relation_type, count) pairs
        target_set = set(target_sig.items())
        cand_set = set(cand_sig.items())
        if not target_set or not cand_set:
            continue

        intersection = target_set & cand_set
        union = target_set | cand_set
        similarity = len(intersection) / len(union)

        if similarity >= min_similarity:
            similar_list.append(candidate)

    return similar_list


def find_isomorphic_patterns(graph: GraphEngine) -> list[IsomorphicMapping]:
    """Find all pairs of nodes with isomorphic relation signatures.

    Groups nodes by their relation signature and finds pairs within groups.
    """
    if graph.graph.number_of_nodes() == 0:
        return []

    # Group nodes by their signature (as a frozenset for hashing)
    sig_groups: dict[frozenset, list[str]] = defaultdict(list)
    for node in graph.graph.nodes():
        sig = _relation_signature(graph, node)
        if sig:
            sig_key = frozenset(sig.items())
            sig_groups[sig_key].append(node)

    mappings: list[IsomorphicMapping] = []
    for sig, nodes in sig_groups.items():
        if len(nodes) < 2:
            continue

        # Create pairwise mappings
        structure_desc = ", ".join(f"{rel}: {count}" for rel, count in sorted(sig))

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                mappings.append(
                    IsomorphicMapping(
                        source_domain=nodes[i],
                        target_domain=nodes[j],
                        shared_structure=structure_desc,
                        confidence=0.6,
                    )
                )

    return mappings
