"""Argument unit extraction and validation."""
from __future__ import annotations

from dataclasses import dataclass

from brbrain.validator.schema import ValidationResult

VALID_CLAIM_TYPES = {"supports", "challenges", "extends", "limits", "solves", "proposes"}
VALID_TARGET_TYPES = {"Method", "Problem", "Conclusion", "Gap", "Debate", "Argument"}


@dataclass
class ExtractedArgument:
    """A single argument unit from LLM extraction."""
    claim: str
    claim_type: str
    target: str
    target_type: str
    evidence_type: str | None = None
    evidence_detail: str | None = None
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "claim_type": self.claim_type,
            "target": self.target,
            "target_type": self.target_type,
            "evidence_type": self.evidence_type,
            "evidence_detail": self.evidence_detail,
            "confidence": self.confidence,
        }


def parse_arguments(raw: list[dict]) -> list[ExtractedArgument]:
    """Parse raw LLM argument dicts into ExtractedArgument objects."""
    args = []
    for item in raw:
        args.append(ExtractedArgument(
            claim=item.get("claim", ""),
            claim_type=item.get("claim_type", ""),
            target=item.get("target", ""),
            target_type=item.get("target_type", ""),
            evidence_type=item.get("evidence_type"),
            evidence_detail=item.get("evidence_detail"),
            confidence=item.get("confidence", 1.0),
        ))
    return args


def validate_argument(arg: ExtractedArgument) -> ValidationResult:
    """Validate an argument's claim_type and target_type against allowed values."""
    if arg.claim_type not in VALID_CLAIM_TYPES:
        return ValidationResult(
            False,
            f"Invalid claim_type '{arg.claim_type}'. Allowed: {sorted(VALID_CLAIM_TYPES)}",
        )
    if arg.target_type not in VALID_TARGET_TYPES:
        return ValidationResult(
            False,
            f"Invalid target_type '{arg.target_type}'. Allowed: {sorted(VALID_TARGET_TYPES)}",
        )
    return ValidationResult(True)


def validate_arguments(args: list[ExtractedArgument]) -> tuple[list[ExtractedArgument], list[dict]]:
    """Validate all arguments. Returns (valid, rejected)."""
    valid = []
    rejected = []
    for arg in args:
        result = validate_argument(arg)
        if result.valid:
            valid.append(arg)
        else:
            rejected.append({"argument": arg.to_dict(), "reason": result.reason})
    return valid, rejected
