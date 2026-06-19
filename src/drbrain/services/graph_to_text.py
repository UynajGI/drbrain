"""KG subgraph -> natural language description via LLM."""

from __future__ import annotations


def describe_path(path: list[dict]) -> str:
    """Describe a single graph path in natural language using templates.

    e.g. "Attention Is All You Need proposes Transformer, which addresses
    the Sequence Modeling problem."

    Args:
        path: List of steps, each with ``src``, ``relation``, ``dst`` keys
            (TraverseStep objects also accepted via __dict__ fallback).

    Returns:
        Human-readable sentence describing the path.
    """
    if not path:
        return ""

    parts: list[str] = []
    for step in path:
        src = getattr(step, "src", None) or (step.get("src", "") if isinstance(step, dict) else "")
        rel = getattr(step, "relation", None) or (
            step.get("relation", "") if isinstance(step, dict) else ""
        )
        dst = getattr(step, "dst", None) or (step.get("dst", "") if isinstance(step, dict) else "")

        _rel_text = _relation_to_text(rel)
        parts.append(f"{src} {_rel_text} {dst}")

    if len(parts) == 1:
        return parts[0]

    # Chain with ", which "
    result = parts[0]
    for i in range(1, len(parts)):
        prev_step = path[i - 1]
        curr_step = path[i]
        prev_dst = getattr(prev_step, "dst", None) or (
            prev_step.get("dst", "") if isinstance(prev_step, dict) else ""
        )
        curr_src = getattr(curr_step, "src", None) or (
            curr_step.get("src", "") if isinstance(curr_step, dict) else ""
        )
        if prev_dst == curr_src:
            # Natural chaining: "A proposes B, which extends C"
            result += f", which\n{parts[i]}"
        else:
            result += f"\n{parts[i]}"

    return result


def _relation_to_text(relation: str) -> str:
    """Map a relation name to its natural-language verb/gerund form."""
    _map = {
        "proposes": "proposes",
        "extends": "extends",
        "replaces": "replaces",
        "addresses": "addresses",
        "solves": "solves",
        "supports": "supports",
        "challenges": "challenges",
        "limits": "limits",
        "constrains": "constrains",
        "leaves_open": "leaves open",
        "points_to": "points to",
        "affiliated_with": "is affiliated with",
    }
    return _map.get(relation, relation.replace("_", " "))


async def describe_subgraph(
    graph,
    db,
    center_entity: str,
    models: list[dict],
    depth: int = 1,
) -> str:
    """Generate a natural language description of a subgraph centered on an entity.

    1. Traverse neighbors up to ``depth`` hops from ``center_entity``
    2. Collect entities, relations, and edge patterns
    3. Build a structured prompt describing the subgraph
    4. LLM generates a concise paragraph summary

    Args:
        graph: ``GraphEngine`` instance with loaded graph.
        db: ``Database`` instance (unused directly but kept for future use).
        center_entity: Node label to center the subgraph on.
        models: List of LLM model configs (provider + model).
        depth: Number of hops to traverse.

    Returns:
        Natural language paragraph describing the subgraph, or empty string
        if all LLM backends are exhausted.
    """
    from drbrain.extractor.llm_client import acall_text_with_fallback

    # 1. Traverse neighbors
    tr_results = graph.traverse(
        start_nodes={center_entity},
        hops=depth,
        direction="both",
    )

    # 2. Collect unique entities and relations
    seen_targets: set[str] = set()
    entities: list[tuple[str, str, int]] = []  # (label, relation, distance)
    relations: set[str] = set()

    for tr in tr_results:
        relations.add(tr.path[-1].relation if tr.path else "connected")
        if tr.target not in seen_targets:
            seen_targets.add(tr.target)
            # Use the first relation on the path for the edge label
            edge_rel = tr.path[0].relation if tr.path else "connected"
            entities.append((tr.target, edge_rel, tr.distance))

    # 3. Build structured prompt
    entity_list = "\n".join(
        f"  - {label} (via {rel}, distance={dist})" for label, rel, dist in entities[:20]
    )
    relation_list = ", ".join(sorted(relations)) if relations else "none"

    if entities:
        prompt = (
            f"Center entity: {center_entity}\n"
            f"Connected entities ({len(entities)} total, showing up to 20):\n"
            f"{entity_list}\n"
            f"Relation types: {relation_list}\n\n"
            "Describe this subgraph in 2-4 concise sentences. "
            "Mention how the center entity relates to its neighbors. "
            "Use natural language (not bullet points). "
            "Return plain text, no markdown."
        )
    else:
        prompt = (
            f"Center entity: {center_entity}\n"
            f"No connected entities found within {depth} hops.\n\n"
            "Write 1 sentence noting that no connections were found. "
            "Return plain text, no markdown."
        )

    # 4. LLM generates summary
    result = await acall_text_with_fallback(prompt, models, max_tokens=250)
    return result.strip() if result else ""
