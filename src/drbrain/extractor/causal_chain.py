"""Causal chain extraction from argument mechanisms.

Builds X→Y(via Z) chains from ExtractedArgument objects that have
non-empty mechanism fields. A causal chain links concepts where each
argument's target becomes the next argument's source concept.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from drbrain.extractor.argument import ExtractedArgument


@dataclass
class CausalChain:
    """A sequence of arguments forming a causal chain."""

    links: list[ExtractedArgument]

    def __len__(self) -> int:
        return len(self.links)

    def summary(self) -> str:
        """Human-readable summary: origin → end (via mechanisms...).

        The origin is the first argument's target (the starting concept).
        The end is the last argument's target (the final concept reached).
        """
        if not self.links:
            return "(empty chain)"
        origin = self.links[0].target
        end = self.links[-1].target
        mechanisms = ", ".join(a.mechanism for a in self.links if a.mechanism)
        return f"{origin} → {end} (via {mechanisms})"


def build_causal_chains(args: list[ExtractedArgument]) -> list[CausalChain]:
    """Build all maximal causal chains from arguments with mechanisms.

    Chain rule: arg_A targets concept X, arg_B targets concept Y where
    Y appears in arg_A's target set. This forms X→Y linkage.

    For simplicity, we chain by target matching: if arg1's target ==
    arg2's claim subject (approximated by checking if arg2's target
    is referenced by arg1), they form a chain.

    Simplified: build a directed graph of targets, edge from arg.target
    to the concept the arg introduces (approximated as the other args'
    targets that match). Chain by following these edges.
    """
    mech_args = [a for a in args if a.mechanism]
    if not mech_args:
        return []

    # Build adjacency: for each argument, find which other arguments
    # have a target that this argument's claim is "about".
    # Simplified heuristic: arg_i -> arg_j if arg_j.target == arg_i.target
    # (they're about the same concept, forming a chain of perspectives)
    # OR arg_i targets X and arg_j targets arg_i's implied subject.

    # Even simpler: treat each unique target as a node. An argument
    # creates a directed edge. Chain = path through this graph.
    # Build: target -> list of args targeting it

    # For chain building: group args by target, then connect groups
    # where one group's args introduce concepts that another group targets.

    # Practical approach: build a graph where nodes are concept labels
    # (targets), and each argument is an edge from its target to...
    # itself for now (self-referential). Chain = sequences of args
    # targeting related concepts.

    # Most practical: find chains where consecutive args share targets
    # or where one arg's target is another's subject.
    # Since we lack a "claim_subject" field, use target as the key.

    # Final approach: build chains by target equivalence.
    # Args targeting the same concept form a "hub". Chains traverse hubs.

    # Actually, the cleanest model: each arg IS a node in the chain graph.
    # Edge from arg_i to arg_j exists if arg_j.target == arg_i.target.
    # This means arg_j builds on the same concept arg_i addresses.

    adj: dict[int, list[int]] = {i: [] for i in range(len(mech_args))}
    for i, arg_i in enumerate(mech_args):
        for j, arg_j in enumerate(mech_args):
            if i == j:
                continue
            if arg_j.target == arg_i.target:
                adj[i].append(j)

    # Find maximal chains via DFS from nodes with no incoming edges
    has_incoming = set()
    for neighbors in adj.values():
        has_incoming.update(neighbors)

    starts = [i for i in adj if i not in has_incoming]
    if not starts:
        starts = list(adj.keys())

    chains: list[CausalChain] = []
    global_visited: set[int] = set()

    def _dfs(node: int, path: list[int]) -> None:
        next_nodes = [n for n in adj[node] if n not in path]
        if not next_nodes:
            chain = CausalChain(links=[mech_args[i] for i in path])
            chains.append(chain)
            global_visited.update(path)
            return
        for n in next_nodes:
            _dfs(n, path + [n])

    for s in starts:
        if s not in global_visited:
            _dfs(s, [s])

    return chains


def find_chains_from(args: list[ExtractedArgument], concept: str) -> list[CausalChain]:
    """Find all causal chains starting from a given concept.

    'Starting from' means the chain's first argument targets this concept.
    """
    mech_args = [a for a in args if a.mechanism]
    starters = [a for a in mech_args if a.target == concept]
    if not starters:
        return []

    # Build adjacency among all mech args
    adj: dict[int, list[int]] = {}
    for i, arg_i in enumerate(mech_args):
        adj[i] = []
        for j, arg_j in enumerate(mech_args):
            if i != j and arg_j.target == arg_i.target:
                adj[i].append(j)

    chains: list[CausalChain] = []

    def _dfs(node: int, path: list[int]) -> None:
        next_nodes = [n for n in adj[node] if n not in path]
        if not next_nodes:
            chains.append(CausalChain(links=[mech_args[i] for i in path]))
            return
        for n in next_nodes:
            _dfs(n, path + [n])

    start_indices = [i for i, a in enumerate(mech_args) if a in starters]
    for s in start_indices:
        _dfs(s, [s])

    return chains


def find_path(args: list[ExtractedArgument], source: str, target: str) -> CausalChain | None:
    """Find shortest causal path from source concept to target concept.

    Uses BFS over the argument chain graph.
    """
    mech_args = [a for a in args if a.mechanism]
    if not mech_args:
        return None

    # Build adjacency: arg_i -> arg_j if arg_j.target == arg_i.target
    adj: dict[int, list[int]] = {}
    for i, arg_i in enumerate(mech_args):
        adj[i] = []
        for j, arg_j in enumerate(mech_args):
            if i != j and arg_j.target == arg_i.target:
                adj[i].append(j)

    # BFS: start from args targeting source, find path to args targeting target
    start_indices = [i for i, a in enumerate(mech_args) if a.target == source]
    target_indices = {i for i, a in enumerate(mech_args) if a.target == target}

    if not start_indices:
        return None

    # BFS queue: (current_index, path_indices)
    queue: deque[tuple[int, list[int]]] = deque()
    for s in start_indices:
        queue.append((s, [s]))

    visited: set[int] = set()

    while queue:
        current, path = queue.popleft()

        if current in target_indices:
            return CausalChain(links=[mech_args[i] for i in path])

        if current in visited:
            continue
        visited.add(current)

        for neighbor in adj[current]:
            if neighbor not in visited and neighbor not in path:
                queue.append((neighbor, path + [neighbor]))

    return None
