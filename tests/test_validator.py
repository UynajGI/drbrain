"""Tests for TBox/RBox schema validation engine."""
from drbrain.validator.schema import (
    TBOX, RBOX, validate_relation, validate_tbox, validate_rbox, ValidationResult
)

def test_tbox_valid_relation():
    """validate_tbox accepts Method --proposes--> anything."""
    result = validate_tbox("Method", "proposes")
    assert result.valid is True

def test_tbox_invalid_relation():
    """validate_tbox rejects Problem --proposes--> (Problems cannot propose)."""
    result = validate_tbox("Problem", "proposes")
    assert result.valid is False
    assert "Problem" in result.reason

def test_tbox_all_types():
    """Each concept type has valid relation whitelist."""
    assert validate_tbox("Problem", "addresses").valid
    assert validate_tbox("Problem", "leaves_open").valid
    assert validate_tbox("Problem", "proposes").valid is False

    assert validate_tbox("Method", "proposes").valid
    assert validate_tbox("Method", "extends").valid
    assert validate_tbox("Method", "replaces").valid

    assert validate_tbox("Conclusion", "supports").valid
    assert validate_tbox("Conclusion", "challenges").valid
    assert validate_tbox("Conclusion", "proposes").valid is False

def test_rbox_irreflexive():
    """validate_rbox rejects self-relations for irreflexive relations."""
    result = validate_rbox("A", "extends", "A")
    assert result.valid is False
    assert "irreflexive" in result.reason

def test_rbox_valid_cross_relation():
    """validate_rbox accepts cross-node relations."""
    assert validate_rbox("A", "extends", "B").valid
    assert validate_rbox("X", "supports", "Y").valid

def test_validate_relation_full():
    """validate_relation checks both TBox and RBox."""
    result = validate_relation("Method", "proposes", "Transformer", "Method")
    assert result.valid is True

    result = validate_relation("Problem", "proposes", "Solution", "Method")
    assert result.valid is False

def test_validation_result_to_dict():
    """ValidationResult can be serialized."""
    result = validate_tbox("Gap", "proposes")
    d = result.to_dict()
    assert "valid" in d
    assert "reason" in d
    assert d["valid"] is False
