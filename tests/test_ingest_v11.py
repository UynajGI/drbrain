"""Integration test for v1.1 features: validation, queue, arguments, temporal."""
import tempfile
from pathlib import Path
from drbrain.storage.database import Database
from drbrain.validator.schema import validate_extraction
from drbrain.extractor.queue import route_item
from drbrain.extractor.argument import ExtractedArgument, validate_arguments

def test_validation_rejects_invalid_relations():
    """validate_extraction rejects Problem --proposes--> relations."""
    concepts = {
        "problems": [{"label": "X problem", "confidence": 0.9}],
        "methods": [{"label": "Y method", "confidence": 0.9}],
        "conclusions": [], "debates": [], "gaps": [], "actors": [],
    }
    relations = [
        {"head": "X problem", "rel": "proposes", "tail": "Y method"},
        {"head": "Y method", "rel": "addresses", "tail": "X problem"},
    ]
    result = validate_extraction(concepts, relations)
    assert len(result["rejected"]) == 1
    assert "proposes" in result["rejected"][0]["reason"]
    assert len(result["valid"]) == 1

def test_argument_validation():
    """validate_arguments filters invalid arguments."""
    args = [
        ExtractedArgument("Valid claim", "proposes", "Method X", "Method", "empirical", "details", 0.9),
        ExtractedArgument("Invalid claim", "magic", "Target Y", "Method", "empirical", "", 0.8),
    ]
    valid, rejected = validate_arguments(args)
    assert len(valid) == 1
    assert len(rejected) == 1
    assert "magic" in rejected[0]["reason"]

def test_queue_routing():
    """route_item correctly routes low-confidence items to queue."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        result = route_item(db, "p1", "concept", {"label": "low conf item"}, 0.4)
        assert result["action"] == "queued"
        pending = db.get_queue_pending()
        assert len(pending) == 1
        db.close()

def test_db_insert_argument_with_paper():
    """Full flow: insert paper, insert argument, query back."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        db.insert_paper("p1", "Test Paper", 2024, "uploaded")
        db.insert_concept("p1", "Method", "Transformer", 0.95, year=2024)
        db.commit()
        arg_id = db.insert_argument(
            "p1", "Self-attention replaces RNN", "proposes",
            "Transformer", "Method", "empirical", "WMT14 BLEU +2.0", 0.95,
        )
        args = db.get_arguments_by_paper("p1")
        assert len(args) == 1
        assert args[0]["claim"] == "Self-attention replaces RNN"
        db.close()

def test_report_includes_validation_and_arguments():
    """PaperReport.to_dict includes arguments and validation sections."""
    from drbrain.report.generator import PaperReport
    report = PaperReport(
        local_id="p1", title="Test", year=2024,
        concepts={"problems": [], "methods": [], "conclusions": [], "debates": [], "gaps": [], "actors": []},
        arguments=[{"claim": "X", "claim_type": "proposes", "target": "Y", "target_type": "Method"}],
        validation={"items_rejected": 1, "items_queued": 0, "tbox_violations": ["test"]},
    )
    d = report.to_dict()
    assert "arguments" in d
    assert "validation" in d
    assert d["arguments"][0]["claim"] == "X"
    assert d["validation"]["items_rejected"] == 1

def test_extracted_concepts_has_arguments():
    """ExtractedConcepts includes arguments field."""
    from drbrain.extractor.concept import ExtractedConcepts
    data = {
        "problems": [], "methods": [], "conclusions": [],
        "debates": [], "gaps": [], "actors": [], "relations": [],
        "arguments": [
            {"claim": "Test", "claim_type": "proposes", "target": "X", "target_type": "Method", "confidence": 0.9}
        ],
    }
    ec = ExtractedConcepts(data)
    assert len(ec.arguments) == 1
    assert ec.arguments[0].claim == "Test"
    assert "arguments" in ec.to_dict()
