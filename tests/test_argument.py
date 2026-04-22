"""Tests for argument extraction from LLM output."""
from brbrain.extractor.argument import ExtractedArgument, parse_arguments, validate_argument, validate_arguments

def test_parse_arguments():
    """parse_arguments converts raw LLM dicts to ExtractedArgument objects."""
    raw = [
        {
            "claim": "Self-attention replaces RNN",
            "claim_type": "proposes",
            "target": "Transformer",
            "target_type": "Method",
            "evidence_type": "empirical",
            "evidence_detail": "WMT14 BLEU +2.0",
            "confidence": 0.95,
        }
    ]
    args = parse_arguments(raw)
    assert len(args) == 1
    assert args[0].claim == "Self-attention replaces RNN"
    assert args[0].claim_type == "proposes"
    assert args[0].target == "Transformer"

def test_validate_argument_valid():
    """validate_argument accepts valid claim_type and target_type."""
    arg = ExtractedArgument(
        claim="X solves Y", claim_type="solves",
        target="Problem Y", target_type="Problem",
        evidence_type="empirical", confidence=0.9,
    )
    result = validate_argument(arg)
    assert result.valid is True

def test_validate_argument_invalid_claim_type():
    """validate_argument rejects unknown claim_type."""
    arg = ExtractedArgument(
        claim="X does Y", claim_type="magical",
        target="Z", target_type="Method",
        evidence_type="empirical", confidence=0.9,
    )
    result = validate_argument(arg)
    assert result.valid is False

def test_validate_argument_invalid_target_type():
    """validate_argument rejects unknown target_type."""
    arg = ExtractedArgument(
        claim="X proposes Y", claim_type="proposes",
        target="Z", target_type="UnknownType",
        evidence_type="empirical", confidence=0.9,
    )
    result = validate_argument(arg)
    assert result.valid is False

def test_extracted_argument_to_dict():
    """ExtractedArgument serializes to dict."""
    arg = ExtractedArgument(
        claim="Test claim", claim_type="proposes",
        target="Target", target_type="Method",
        evidence_type="empirical", evidence_detail="details",
        confidence=0.85,
    )
    d = arg.to_dict()
    assert d["claim"] == "Test claim"
    assert d["claim_type"] == "proposes"
    assert d["target"] == "Target"
    assert d["evidence_type"] == "empirical"

def test_validate_arguments_batch():
    """validate_arguments separates valid from rejected."""
    args = [
        ExtractedArgument("Valid claim", "proposes", "Method X", "Method", "empirical", "details", 0.9),
        ExtractedArgument("Invalid claim", "magic", "Target Y", "Method", "empirical", "", 0.8),
    ]
    valid, rejected = validate_arguments(args)
    assert len(valid) == 1
    assert len(rejected) == 1
    assert "magic" in rejected[0]["reason"]

def test_parse_empty_arguments():
    """parse_arguments handles empty input."""
    args = parse_arguments([])
    assert args == []

def test_parse_arguments_defaults():
    """parse_arguments uses defaults for optional fields."""
    raw = [{"claim": "Simple", "claim_type": "proposes", "target": "X", "target_type": "Method"}]
    args = parse_arguments(raw)
    assert args[0].evidence_type is None
    assert args[0].confidence == 1.0
