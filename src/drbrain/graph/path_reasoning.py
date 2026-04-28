"""Multi-hop path reasoning rules for DrBrain graph."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class PathRule:
    """A multi-hop path pattern that infers a new edge.

    pattern: list of (relation, direction) tuples forming a chain.
        direction "forward" means src→dst, "backward" means dst→src.
    consequence: the relation to infer between the chain endpoints.
    name: unique identifier for the rule.
    """

    name: str
    pattern: list[tuple[str, str]]
    consequence: str


def get_builtin_rules() -> list[PathRule]:
    """Return the standard path reasoning rules."""
    return [
        PathRule(
            name="method_supersedes_problem",
            # Method_B replaces Method_A, Method_A addresses Problem_X
            # => Method_B addresses Problem_X
            pattern=[("replaces", "forward"), ("addresses", "backward")],
            consequence="supersedes_address",
        ),
        PathRule(
            name="challenge_chain",
            # Method_B extends Method_A, Method_A challenges Conclusion_X
            # => Method_B challenges Conclusion_X
            pattern=[("extends", "forward"), ("challenges", "backward")],
            consequence="challenge_chain",
        ),
        PathRule(
            name="gap_inheritance",
            # Paper_B extends Paper_A, Paper_A leaves_open Gap_X
            # => Gap_X relevant to Paper_B
            pattern=[("extends", "forward"), ("leaves_open", "backward")],
            consequence="gap_inheritance",
        ),
        PathRule(
            name="indirect_support",
            # Method_B extends Method_A, Method_A solves Problem_X
            # => Method_B solves Problem_X
            pattern=[("extends", "forward"), ("solves", "backward")],
            consequence="indirect_support",
        ),
    ]


def _apply_path_rules_subgraph(subgraph) -> list[dict]:
    """Apply path rules to a NetworkX subgraph directly.

    Used by closure_incremental which builds a subgraph, not a GraphEngine.
    """
    inferred: list[dict] = []
    rules = get_builtin_rules()

    for rule in rules:
        matches = _match_pattern_subgraph(subgraph, rule.pattern)
        for src, dst in matches:
            inferred.append(
                {
                    "src": src,
                    "dst": dst,
                    "relation": rule.consequence,
                    "via": rule.name,
                }
            )

    return inferred


def _match_pattern_subgraph(graph, pattern: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Find all chains matching the pattern on a raw NetworkX graph."""
    if len(pattern) < 2:
        return []

    rel_indices = []
    for rel, direction in pattern:
        idx: dict[str, set[str]] = defaultdict(set)
        for u, v, data in graph.edges(data=True):
            if data["relation"] == rel:
                if direction == "forward":
                    idx[v].add(u)
                else:
                    idx[u].add(v)
        rel_indices.append(idx)

    first_idx = rel_indices[0]
    results: list[tuple[str, str]] = []
    visited_edges: set[tuple[str, str, str]] = set()

    for middle_node, prev_nodes in first_idx.items():
        for prev in prev_nodes:
            end_nodes = _extend_chain_subgraph(graph, rel_indices[1:], middle_node)
            for end in end_nodes:
                edge_key = (prev, end, pattern[0][0])
                if edge_key not in visited_edges:
                    visited_edges.add(edge_key)
                    results.append((prev, end))

    return results


def _extend_chain_subgraph(
    graph, remaining_indices: list[dict[str, set[str]]], current: str
) -> set[str]:
    """Recursively extend a chain through remaining relation indices."""
    if not remaining_indices:
        return {current}

    idx = remaining_indices[0]
    next_nodes = idx.get(current, set())
    if not remaining_indices[1:]:
        return next_nodes

    result: set[str] = set()
    for node in next_nodes:
        result |= _extend_chain_subgraph(graph, remaining_indices[1:], node)
    return result


def apply_path_rules(graph) -> list[dict]:
    """Apply all path rules to the graph, return inferred edges."""
    from drbrain.graph.engine import GraphEngine

    if not isinstance(graph, GraphEngine):
        return []

    rules = get_builtin_rules()
    inferred: list[dict] = []

    for rule in rules:
        matches = _match_pattern(graph, rule.pattern)
        for src, dst in matches:
            inferred.append(
                {
                    "src": src,
                    "dst": dst,
                    "relation": rule.consequence,
                    "via": rule.name,
                }
            )

    return inferred


def _match_pattern(graph, pattern: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Find all chains matching the pattern.

    Pattern is a list of (relation, direction) steps.
    Returns list of (start_node, end_node) pairs that match the full chain.
    """
    if len(pattern) < 2:
        return []

    # Build adjacency indices for each relation in the pattern
    rel_indices = []
    for rel, direction in pattern:
        idx: dict[str, set[str]] = defaultdict(set)
        for u, v, data in graph.graph.edges(data=True):
            if data["relation"] == rel:
                if direction == "forward":
                    # u → v, we want: given v, find u
                    idx[v].add(u)
                else:
                    # u → v, we want: given u, find v (backward traversal)
                    idx[u].add(v)
        rel_indices.append(idx)

    # Start with all possible first-step sources
    first_idx = rel_indices[0]
    results: list[tuple[str, str]] = []
    visited_edges: set[tuple[str, str, str]] = set()

    for middle_node, prev_nodes in first_idx.items():
        for prev in prev_nodes:
            # prev is the start of a 2-step chain
            # Try to extend through remaining steps
            end_nodes = _extend_chain(graph, rel_indices[1:], middle_node)
            for end in end_nodes:
                edge_key = (prev, end, pattern[0][0])
                if edge_key not in visited_edges:
                    visited_edges.add(edge_key)
                    results.append((prev, end))

    return results


def _extend_chain(graph, remaining_indices: list[dict[str, set[str]]], current: str) -> set[str]:
    """Recursively extend a chain through remaining relation indices."""
    if not remaining_indices:
        return {current}

    idx = remaining_indices[0]
    next_nodes = idx.get(current, set())
    if not remaining_indices[1:]:
        return next_nodes

    result: set[str] = set()
    for node in next_nodes:
        result |= _extend_chain(graph, remaining_indices[1:], node)
    return result
