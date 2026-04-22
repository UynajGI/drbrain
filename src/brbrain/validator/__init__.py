"""Schema validation package."""
from brbrain.validator.schema import (
    TBOX, RBOX, validate_tbox, validate_rbox, validate_relation,
    validate_extraction, ValidationResult,
)

__all__ = [
    "TBOX", "RBOX", "validate_tbox", "validate_rbox",
    "validate_relation", "validate_extraction", "ValidationResult",
]
