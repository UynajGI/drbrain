"""Embedding-driven rule mining from TransE relation vectors.

Mines new path rules by discovering relation compositions in embedding space
and by walking the graph to find recurring path patterns.

Inspired by NeuralLP / DRUM: uses TransE vector addition (h + r1 + r2 ≈ t)
as a proxy for rule confidence — a path composes to a relation r iff
cos_sim(r_vec, r1_vec + r2_vec) is high.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def compose_path(rel_vecs: list[np.ndarray]) -> np.ndarray:
    """Compose relation vectors along a path: r1 + r2 + ... + rn.

    Follows the TransE scoring principle where path composition is vector addition.
    """
    if not rel_vecs:
        return np.zeros(1, dtype=np.float32)
    return np.add.reduce(rel_vecs)


def mine_path_rules(
    graph,
    db,
    models=None,
    min_confidence: float = 0.6,
    top_k: int = 20,
) -> list[dict]:
    """Mine path rules from relation embeddings.

    For each relation r, checks if composing two other relations (r_i, r_j)
    approximates r in embedding space. Uses TransE vector addition as the
    composition operator.

    Args:
        graph: GraphEngine instance (for counting support from existing edges).
        db: Database instance with load_embeddings() method.
        models: Reserved for future model-based scoring.
        min_confidence: Minimum cosine similarity threshold for rule acceptance.
        top_k: Maximum number of rules to return.

    Returns:
        List of dicts with keys: head, body_path, confidence, support.
    """
    from drbrain.graph.query_embeddings import _load_embeddings_for_query

    entities, relations = _load_embeddings_for_query(db)
    if len(relations) < 3:
        return []

    rel_names = list(relations.keys())
    rel_vecs = {name: relations[name] for name in rel_names}

    # Pre-build path support index: for each relation pair (r_i, r_j),
    # count how many times the pattern appears in the graph
    path_support: dict[tuple[str, str], int] = _count_path_patterns(graph, rel_names)

    rules: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    for head_rel in rel_names:
        head_vec = rel_vecs[head_rel]
        for r_i in rel_names:
            if r_i == head_rel:
                continue
            for r_j in rel_names:
                if r_j == head_rel or r_i == r_j:
                    continue

                key = (head_rel, r_i, r_j)
                if key in seen:
                    continue
                seen.add(key)

                composed = compose_path([rel_vecs[r_i], rel_vecs[r_j]])
                confidence = _cosine_sim(head_vec, composed)

                if confidence >= min_confidence:
                    support = path_support.get((r_i, r_j), 0)
                    rules.append(
                        {
                            "head": head_rel,
                            "body_path": [r_i, r_j],
                            "confidence": round(float(confidence), 4),
                            "support": support,
                        }
                    )

    # Sort by confidence * support bonus, keep top_k
    rules.sort(
        key=lambda r: r["confidence"] * (1.0 + min(r["support"], 10) / 10.0),
        reverse=True,
    )
    return rules[:top_k]


def _count_path_patterns(graph, rel_names: list[str]) -> dict[tuple[str, str], int]:
    """Count occurrences of each 2-hop relation path pattern in the graph.

    For each node that has both incoming and outgoing edges, record the
    relation pair (incoming_rel, outgoing_rel) and increment the count.
    """
    # Build: for each node, incoming edges by relation and outgoing edges by relation
    incoming: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    outgoing: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    rel_set = set(rel_names)
    for u, v, data in graph.graph.edges(data=True):
        rel = data["relation"]
        if rel not in rel_set:
            continue
        outgoing[u][rel].add(v)
        incoming[v][rel].add(u)

    counts: dict[tuple[str, str], int] = defaultdict(int)
    for node in set(incoming.keys()) | set(outgoing.keys()):
        in_rels = incoming.get(node, {})
        out_rels = outgoing.get(node, {})
        for in_rel, srcs in in_rels.items():
            for out_rel, dsts in out_rels.items():
                counts[(in_rel, out_rel)] += len(srcs) * len(dsts)

    return counts


def mine_from_graph_walks(
    graph,
    max_length: int = 3,
    min_support: int = 3,
    top_k: int = 20,
    relation_vecs: dict[str, np.ndarray] | None = None,
) -> list[dict]:
    """Walk the graph to find recurring path patterns as candidate rules.

    Enumerates all paths up to max_length hops, counts relation-sequence
    frequencies, and optionally maps each frequent sequence to its closest
    embedding relation as the rule head.

    Args:
        graph: GraphEngine instance.
        max_length: Maximum path length (number of edges) to consider.
        min_support: Minimum number of occurrences for a pattern to be kept.
        top_k: Maximum number of rules to return.
        relation_vecs: Optional dict of relation name → vector for head matching.
            If provided, the composed path vector is compared against all
            relation vectors, and the closest one (above min_confidence)
            becomes the head. If None, head is derived only from the pattern.

    Returns:
        List of dicts with keys: head, body_path, confidence, support.
    """
    # Collect all relation sequences from paths in the graph
    pattern_counts: dict[tuple[str, ...], int] = defaultdict(int)

    # 1. Sequential walks: A → B → C (relation paths through intermediate nodes)
    for node in graph.graph.nodes():
        _walk_from_node(graph, node, max_length, pattern_counts)

    # 2. Parallel edge patterns: same (src, dst) with multiple relation types
    _count_parallel_patterns(graph, pattern_counts)

    rules: list[dict] = []
    for pattern, support in pattern_counts.items():
        if support < min_support:
            continue

        rule = {
            "body_path": list(pattern),
            "support": support,
            "confidence": 1.0,
        }

        if relation_vecs and len(pattern) >= 2:
            head, confidence = _best_head_for_path(pattern, relation_vecs)
            if head is not None:
                rule["head"] = head
                rule["confidence"] = round(float(confidence), 4)
            else:
                rule["head"] = _default_head_name(pattern)
        else:
            rule["head"] = _default_head_name(pattern)

        rules.append(rule)

    rules.sort(key=lambda r: r["support"], reverse=True)
    return rules[:top_k]


def _count_parallel_patterns(graph, counts: dict[tuple[str, ...], int]) -> None:
    """Count co-occurring relation pairs on the same (src, dst) node pair.

    When the same source and target have multiple edges with different
    relations (e.g., M1 -proposes-> P1 and M1 -addresses-> P1), that
    forms a 2-relation pattern (proposes, addresses) that may indicate
    a meaningful composition.
    """
    # For each pair of nodes, collect all relation types between them
    pair_rels: dict[tuple[str, str], set[str]] = defaultdict(set)
    for u, v, data in graph.graph.edges(data=True):
        rel = data.get("relation", "")
        if rel:
            pair_rels[(u, v)].add(rel)

    # For each pair with multiple relations, add all ordered pairs as patterns
    for rels in pair_rels.values():
        if len(rels) < 2:
            continue
        rel_list = sorted(rels)
        for i, r1 in enumerate(rel_list):
            for r2 in rel_list[i + 1 :]:
                counts[(r1, r2)] += 1
                counts[(r2, r1)] += 1


def _walk_from_node(
    graph,
    start_node: str,
    max_depth: int,
    counts: dict[tuple[str, ...], int],
) -> None:
    """BFS from start_node, collecting relation sequences along paths."""
    from collections import deque

    # Queue entries: (current_node, relation_path_so_far)
    queue: deque[tuple[str, tuple[str, ...]]] = deque()
    queue.append((start_node, ()))

    while queue:
        node, path = queue.popleft()
        if len(path) >= max_depth:
            continue

        for _, neighbor, data in graph.graph.edges(node, data=True):
            rel = data.get("relation", "")
            if not rel:
                continue
            new_path = path + (rel,)
            if len(new_path) >= 2:
                counts[new_path] += 1
            if len(new_path) < max_depth:
                queue.append((neighbor, new_path))


def _best_head_for_path(
    pattern: tuple[str, ...],
    relation_vecs: dict[str, np.ndarray],
    min_confidence: float = 0.5,
) -> tuple[str | None, float]:
    """Find the relation whose vector is closest to the composed path vector.

    Returns (relation_name, confidence) or (None, 0.0) if none found.
    """
    # Compose from pattern's relation vectors (skip relations not in vecs)
    vecs = [relation_vecs[r] for r in pattern if r in relation_vecs]
    if len(vecs) < 2:
        return None, 0.0  # need at least 2 vectors to compose

    composed = compose_path(vecs)
    best_rel: str | None = None
    best_sim: float = -1.0

    for rel_name, rel_vec in relation_vecs.items():
        # Skip relations that are already in the pattern
        if rel_name in pattern:
            continue
        sim = _cosine_sim(composed, rel_vec)
        if sim > best_sim:
            best_sim = sim
            best_rel = rel_name

    if best_rel is None or best_sim < min_confidence:
        return None, 0.0
    return best_rel, best_sim


def _default_head_name(pattern: tuple[str, ...]) -> str:
    """Generate a default rule head name from a relation path pattern."""
    return "_".join(pattern) + "_path"
