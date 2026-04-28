"""Schema-first validation: TBox type constraints + RBox relation restrictions."""
from __future__ import annotations

from dataclasses import dataclass

TBOX = {
    "Problem":   {"addresses", "leaves_open", "points_to"},
    "Method":    {"addresses", "proposes", "extends", "replaces", "solves"},
    "Conclusion":{"supports", "challenges", "limits"},
    "Debate":    {"supports", "challenges"},
    "Gap":       {"leaves_open", "points_to", "constrains"},
    "Actor":     {"affiliated_with", "proposes"},
}

RBOX = {
    "transitive": {"extends"},
    "asymmetric": {"extends", "replaces", "challenges", "supports"},
    "irreflexive": {"extends", "replaces", "challenges", "supports", "limits"},
}


@dataclass
class ValidationResult:
    valid: bool
    reason: str = ""

    def to_dict(self) -> dict:
        return {"valid": self.valid, "reason": self.reason}


def validate_tbox(concept_type: str, relation: str) -> ValidationResult:
    """Check if a relation is valid for a given concept type."""
    allowed = TBOX.get(concept_type)
    if allowed is None:
        return ValidationResult(False, f"Unknown concept type: {concept_type}")
    if relation not in allowed:
        return ValidationResult(
            False,
            f"TBox violation: {concept_type} cannot use relation '{relation}'. "
            f"Allowed: {sorted(allowed)}",
        )
    return ValidationResult(True)


def validate_rbox(src_label: str, relation: str, dst_label: str) -> ValidationResult:
    """Check RBox constraints for a single edge."""
    if relation in RBOX["irreflexive"] and src_label == dst_label:
        return ValidationResult(
            False,
            f"RBox violation: '{relation}' is irreflexive, cannot relate "
            f"'{src_label}' to itself.",
        )
    return ValidationResult(True)


def validate_relation(concept_type: str, relation: str,
                      src_label: str, dst_label: str) -> ValidationResult:
    """Full validation: TBox + RBox."""
    tbox = validate_tbox(concept_type, relation)
    if not tbox.valid:
        return tbox
    return validate_rbox(src_label, relation, dst_label)


def validate_extraction(concepts: dict, relations: list[dict]) -> dict:
    """Validate all concepts and relations from LLM extraction.

    Returns: {"valid": [...], "rejected": [...]}
    """
    valid = []
    rejected = []

    for rel in relations:
        head = rel.get("head", "")
        rel_type = rel.get("rel", "")
        tail = rel.get("tail", "")

        head_type = _find_concept_type(head, concepts)
        if head_type:
            result = validate_relation(head_type, rel_type, head, tail)
            if result.valid:
                valid.append({"type": "relation", "detail": rel})
            else:
                rejected.append({"type": "relation", "detail": rel, "reason": result.reason})
        else:
            valid.append({"type": "relation", "detail": rel})

    return {"valid": valid, "rejected": rejected}


def _find_concept_type(label: str, concepts: dict) -> str | None:
    """Find the concept type for a label in the extraction result."""
    type_map = {
        "Problem": concepts.get("problems", []),
        "Method": concepts.get("methods", []),
        "Conclusion": concepts.get("conclusions", []),
        "Debate": concepts.get("debates", []),
        "Gap": concepts.get("gaps", []),
        "Actor": concepts.get("actors", []),
    }
    for ctype, items in type_map.items():
        for item in items:
            if item.get("label", "") == label:
                return ctype
    return None
