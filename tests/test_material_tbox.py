"""Tests for the pluggable material-science domain TBox schema.

The material schema is additive: it defines five material-domain concept
types (Composition/Structure/Property/SimulationMethod/SynthesisCondition)
with their own TBox relation whitelist, coexisting with the generic academic
six-category TBox (Problem/Method/Conclusion/Gap/Debate/Actor).
"""

from drbrain.extractor.concept.types import (
    MATERIAL_DOMAIN,
    MATERIAL_TYPE_FIELDS,
    ExtractedConcepts,
)
from drbrain.validator.schema import (
    COMPOSITION,
    MATERIAL_TBOX,
    MATERIAL_TYPES,
    PROPERTY,
    SIMULATION_METHOD,
    STRUCTURE,
    SYNTHESIS_CONDITION,
    validate_tbox,
)


def test_material_type_constants_defined():
    """The five material-domain type names are defined as constants."""
    assert COMPOSITION == "Composition"
    assert STRUCTURE == "Structure"
    assert PROPERTY == "Property"
    assert SIMULATION_METHOD == "SimulationMethod"
    assert SYNTHESIS_CONDITION == "SynthesisCondition"


def test_material_types_set_is_exactly_five():
    """MATERIAL_TYPES enumerates exactly the five material types."""
    assert MATERIAL_TYPES == frozenset(
        {COMPOSITION, STRUCTURE, PROPERTY, SIMULATION_METHOD, SYNTHESIS_CONDITION}
    )
    assert len(MATERIAL_TYPES) == 5


def test_material_tbox_covers_every_type():
    """Every material type has a non-empty allowed-relation whitelist."""
    assert set(MATERIAL_TBOX.keys()) == set(MATERIAL_TYPES)
    for mtype in MATERIAL_TYPES:
        assert MATERIAL_TBOX[mtype], f"{mtype} has no allowed relations"


def test_material_tbox_relations_reasonable():
    """Spot-check the key material-science relations are allowed."""
    assert "has_structure" in MATERIAL_TBOX[COMPOSITION]
    assert "measured_by" in MATERIAL_TBOX[COMPOSITION]

    assert "has_composition" in MATERIAL_TBOX[STRUCTURE]
    assert "determines" in MATERIAL_TBOX[STRUCTURE]

    assert "measured_by" in MATERIAL_TBOX[PROPERTY]
    assert "predicted_by" in MATERIAL_TBOX[PROPERTY]

    assert "predicts" in MATERIAL_TBOX[SIMULATION_METHOD]
    assert "simulates" in MATERIAL_TBOX[SIMULATION_METHOD]

    assert "produces" in MATERIAL_TBOX[SYNTHESIS_CONDITION]
    assert "controls" in MATERIAL_TBOX[SYNTHESIS_CONDITION]


def test_validate_tbox_accepts_material_relations():
    """validate_tbox falls back to MATERIAL_TBOX for material types."""
    assert validate_tbox(PROPERTY, "measured_by").valid
    assert validate_tbox(PROPERTY, "produces").valid is False
    assert validate_tbox(COMPOSITION, "has_structure").valid
    assert validate_tbox(COMPOSITION, "predicts").valid is False


def test_material_type_field_map():
    """Type -> field mapping exposes the five plural material fields."""
    assert MATERIAL_TYPE_FIELDS[COMPOSITION] == "compositions"
    assert MATERIAL_TYPE_FIELDS[STRUCTURE] == "structures"
    assert MATERIAL_TYPE_FIELDS[PROPERTY] == "properties"
    assert MATERIAL_TYPE_FIELDS[SIMULATION_METHOD] == "simulation_methods"
    assert MATERIAL_TYPE_FIELDS[SYNTHESIS_CONDITION] == "synthesis_conditions"


def test_default_domain_omits_material_fields():
    """Without the material domain, to_dict keeps the generic six categories."""
    concepts = ExtractedConcepts(
        {
            "problems": [{"label": "P", "confidence": 0.9}],
            "compositions": [{"label": "CrF3", "confidence": 0.9}],
        }
    )
    assert concepts.domain is None
    d = concepts.to_dict()
    assert "compositions" not in d
    assert d["problems"][0]["label"] == "P"


def test_material_domain_carries_material_fields():
    """With domain='material', the extraction result carries material fields."""
    data = {
        "compositions": [{"label": "CrF3", "confidence": 0.95}],
        "structures": [{"label": "R-3c space group", "confidence": 0.9}],
        "properties": [{"label": "band gap 3.1 eV", "confidence": 0.9}],
        "simulation_methods": [{"label": "DFT+U", "confidence": 0.85}],
        "synthesis_conditions": [{"label": "700 K", "confidence": 0.8}],
    }
    concepts = ExtractedConcepts(data, domain=MATERIAL_DOMAIN)
    assert concepts.domain == MATERIAL_DOMAIN
    assert concepts.compositions[0]["label"] == "CrF3"

    fields = concepts.material_fields()
    assert fields["compositions"][0]["label"] == "CrF3"
    assert fields["structures"][0]["label"] == "R-3c space group"

    d = concepts.to_dict()
    assert d["compositions"] == data["compositions"]
    assert d["properties"] == data["properties"]
    assert d["simulation_methods"] == data["simulation_methods"]
    assert d["synthesis_conditions"] == data["synthesis_conditions"]


def test_merge_concepts_preserves_material_fields():
    """_merge_concepts deduplicates material fields across sections (no drop)."""
    from drbrain.extractor.concept.merge import _merge_concepts

    a = ExtractedConcepts(
        {
            "compositions": [{"label": "CrF3", "confidence": 0.9}],
            "properties": [{"label": "band gap", "confidence": 0.8}],
        },
        domain=MATERIAL_DOMAIN,
    )
    b = ExtractedConcepts(
        {
            "compositions": [{"label": "CrF3", "confidence": 0.7}],
            "properties": [{"label": "formation energy", "confidence": 0.85}],
        },
        domain=MATERIAL_DOMAIN,
    )

    merged = _merge_concepts([a, b], sections=["Intro", "Results"])

    assert merged.domain == MATERIAL_DOMAIN
    # CrF3 deduplicated — highest confidence kept, section annotated.
    assert len(merged.compositions) == 1
    assert merged.compositions[0]["label"] == "CrF3"
    assert merged.compositions[0]["confidence"] == 0.9
    assert merged.compositions[0]["section"] == "Intro"
    # Distinct properties all preserved.
    assert {p["label"] for p in merged.properties} == {"band gap", "formation energy"}
