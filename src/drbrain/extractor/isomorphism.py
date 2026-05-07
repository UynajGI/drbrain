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
    raptor_source_context: list[dict] | None = None
    raptor_target_context: list[dict] | None = None

    def __post_init__(self):
        if self.raptor_source_context is None:
            self.raptor_source_context = []
        if self.raptor_target_context is None:
            self.raptor_target_context = []


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
    Confidence combines Jaccard similarity (0.7 weight) and label
    similarity (0.3 weight).
    """
    if graph.graph.number_of_nodes() == 0:
        return []

    # Build raw signatures for each node (as dicts, not frozenset)
    node_sigs: dict[str, dict] = {}
    for node in graph.graph.nodes():
        sig = _relation_signature(graph, node)
        if sig:
            node_sigs[node] = sig

    # Group nodes by frozenset key (identical signatures)
    sig_groups: dict[frozenset, list[str]] = defaultdict(list)
    for node, sig in node_sigs.items():
        sig_key = frozenset(sig.items())
        sig_groups[sig_key].append(node)

    from drbrain.extractor.concept import _label_similarity

    mappings: list[IsomorphicMapping] = []
    for sig_items, nodes in sig_groups.items():
        if len(nodes) < 2:
            continue

        raw_sig = dict(sig_items)
        structure_desc = ", ".join(f"{rel}: {count}" for rel, count in sorted(raw_sig.items()))

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                # Jaccard from raw signatures
                sig_i = node_sigs[nodes[i]]
                sig_j = node_sigs[nodes[j]]
                set_i = set(sig_i.items())
                set_j = set(sig_j.items())
                intersection = set_i & set_j
                union = set_i | set_j
                jaccard = len(intersection) / len(union) if union else 0.0

                # Label similarity
                label_sim = _label_similarity(nodes[i], nodes[j])

                # Combined confidence
                confidence = jaccard * 0.7 + label_sim * 0.3

                mappings.append(
                    IsomorphicMapping(
                        source_domain=nodes[i],
                        target_domain=nodes[j],
                        shared_structure=structure_desc,
                        confidence=round(confidence, 4),
                    )
                )

    return mappings


def enrich_isomorphisms_with_raptor(
    mappings: list[IsomorphicMapping],
    db,  # Database
) -> list[IsomorphicMapping]:
    """Enrich isomorphic mappings with RAPTOR cross-section summaries.

    For each mapping, looks up the papers containing source/target concepts
    and fetches RAPTOR summaries for semantic context.

    Args:
        mappings: List of IsomorphicMapping from find_isomorphic_patterns.
        db: Database instance.

    Returns:
        Same mappings with raptor_source_context and raptor_target_context
        populated (empty lists if no RAPTOR data exists).
    """
    if not db or not mappings:
        return mappings

    # Collect all unique concept labels
    all_labels: set[str] = set()
    for m in mappings:
        all_labels.add(m.source_domain)
        all_labels.add(m.target_domain)

    if not all_labels:
        return mappings

    # Find which papers contain which concepts
    placeholders = ",".join("?" for _ in all_labels)
    rows = db.conn.execute(
        f"SELECT DISTINCT label, local_id FROM concepts WHERE label IN ({placeholders})",
        tuple(all_labels),
    ).fetchall()

    concept_papers: dict[str, list[str]] = defaultdict(list)
    for label, local_id in rows:
        concept_papers[label].append(local_id)

    # Fetch RAPTOR summaries for all relevant papers
    all_paper_ids: set[str] = set()
    for papers in concept_papers.values():
        all_paper_ids.update(papers)

    raptor_cache: dict[str, list[dict]] = {}
    if all_paper_ids:
        import json

        p_placeholders = ",".join("?" for _ in all_paper_ids)
        raptor_rows = db.conn.execute(
            f"SELECT paper_id, node_id, summary_text, source_node_ids, tree_layer "
            f"FROM tree_summaries WHERE paper_id IN ({p_placeholders}) "
            f"ORDER BY tree_layer",
            tuple(all_paper_ids),
        ).fetchall()

        for row in raptor_rows:
            pid = row[0]
            if pid not in raptor_cache:
                raptor_cache[pid] = []
            raptor_cache[pid].append(
                {
                    "node_id": row[1],
                    "summary_text": row[2],
                    "source_node_ids": json.loads(row[3]) if row[3] else [],
                    "tree_layer": row[4],
                }
            )

    # Enrich each mapping
    for m in mappings:
        source_papers = concept_papers.get(m.source_domain, [])
        target_papers = concept_papers.get(m.target_domain, [])

        m.raptor_source_context = []
        for pid in source_papers:
            m.raptor_source_context.extend(raptor_cache.get(pid, []))

        m.raptor_target_context = []
        for pid in target_papers:
            m.raptor_target_context.extend(raptor_cache.get(pid, []))

    return mappings
