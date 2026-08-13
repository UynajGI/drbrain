"""Tests for the :class:`~drbrain.rag.state.RAGState` dialogue-state model."""

from drbrain.rag.state import RAGState


def test_default_construction_and_roundtrip():
    state = RAGState()
    assert state.entity_ids == []
    assert state.intent == ""
    assert state.extra == {}

    d = state.to_dict()
    assert set(d) == {
        "entity_ids",
        "intent",
        "attribute",
        "period",
        "comparison_period",
        "tenant_id",
        "user_id",
        "snapshot_id",
        "extra",
    }
    assert RAGState.from_dict(d) == state


def test_from_dict_tolerates_missing_fields():
    assert RAGState.from_dict({}) == RAGState()

    partial = RAGState.from_dict({"intent": "负责人", "entity_ids": ["p1"]})
    assert partial.intent == "负责人"
    assert partial.entity_ids == ["p1"]
    assert partial.attribute == ""
    assert partial.period == ""
    assert partial.extra == {}


def test_from_dict_absorbs_unknown_keys_into_extra():
    state = RAGState.from_dict({"entity_ids": ["p1"], "confidence": 0.9, "locale": "zh"})
    assert state.entity_ids == ["p1"]
    assert state.extra == {"confidence": 0.9, "locale": "zh"}


def test_update_returns_new_object_and_does_not_mutate():
    original = RAGState(entity_ids=["p1"], intent="budget")
    updated = original.update(attribute="owner", period="2026")

    assert updated is not original
    assert original.entity_ids == ["p1"]
    assert original.intent == "budget"
    assert original.attribute == ""
    assert original.period == ""

    assert updated.entity_ids == ["p1"]
    assert updated.intent == "budget"
    assert updated.attribute == "owner"
    assert updated.period == "2026"


def test_update_unknown_keys_land_in_extra():
    state = RAGState().update(confidence=0.9)
    assert state.extra == {"confidence": 0.9}


def test_update_copies_mutable_fields():
    original = RAGState(entity_ids=["p1"])
    updated = original.update(attribute="owner")
    updated.entity_ids.append("p2")
    assert original.entity_ids == ["p1"]


def test_resolve_reference_rewrites_anaphoric_prompt():
    state = RAGState(entity_ids=["proj-1"], attribute="owner")
    rewritten = state.resolve_reference("那负责人呢?")
    assert rewritten != "那负责人呢?"
    assert "proj-1" in rewritten
    assert "owner" in rewritten
    assert "那负责人呢?" in rewritten


def test_resolve_reference_passes_through_non_reference():
    state = RAGState(entity_ids=["proj-1"], attribute="owner")
    assert state.resolve_reference("这个项目预算多少?") == "这个项目预算多少?"


def test_resolve_reference_without_context_is_identity():
    assert RAGState().resolve_reference("那负责人呢?") == "那负责人呢?"


def test_extra_roundtrips():
    state = RAGState(extra={"confidence": 0.9, "nested": {"a": 1}})
    assert state.to_dict()["extra"] == {"confidence": 0.9, "nested": {"a": 1}}
    assert RAGState.from_dict(state.to_dict()) == state


def test_to_dict_returns_copies_of_mutable_fields():
    state = RAGState(entity_ids=["p1"], extra={"x": 1})
    d = state.to_dict()
    d["entity_ids"].append("p2")
    d["extra"]["y"] = 2
    assert state.entity_ids == ["p1"]
    assert state.extra == {"x": 1}
