import json
from types import SimpleNamespace
from unittest.mock import Mock

from drbrain.loop.events import Evidence, EvidenceBundle, Hypothesis, ResearchState, Verification
from drbrain.loop.workflow import (
    ResearchLoopWorkflow,
    _has_required_evidence,
    _referenced_evidence_ids,
)
from drbrain.rag import agent as rag_agent
from drbrain.rag import indexer
from drbrain.rag.evidence import build_evidence_record


def test_evidence_record_binds_generation_locator_and_content():
    record = build_evidence_record(
        generation="g-20260826-a",
        query="perovskite stability",
        retriever="fusion",
        rank=1,
        score=0.875,
        source={
            "paper_id": "paper-1",
            "node_id": "section-2#0",
            "title": "Stability",
            "text": "The passivation layer improves stability.",
        },
    )

    assert record["evidence_id"].startswith("ev-")
    assert record["generation"] == "g-20260826-a"
    assert record["document_locator"] == {"paper_id": "paper-1", "title": "Stability"}
    assert record["chunk_locator"] == {"node_id": "section-2#0"}
    assert len(record["content_checksum"]) == 64
    assert record["query"] == "perovskite stability"
    assert record["retriever"] == "fusion"
    assert record["rank"] == 1
    assert record["score"] == 0.875


def test_evidence_record_id_changes_when_generation_or_content_changes():
    source = {"paper_id": "paper-1", "node_id": "section-2", "text": "first"}
    first = build_evidence_record(
        generation="g-one", query="q", retriever="vector", rank=1, score=1.0, source=source
    )
    next_generation = build_evidence_record(
        generation="g-two", query="q", retriever="vector", rank=1, score=1.0, source=source
    )
    changed_content = build_evidence_record(
        generation="g-one",
        query="q",
        retriever="vector",
        rank=1,
        score=1.0,
        source={**source, "text": "second"},
    )

    assert first["evidence_id"] != next_generation["evidence_id"]
    assert first["evidence_id"] != changed_content["evidence_id"]


def test_capture_index_generation_makes_legacy_snapshot_explicit(monkeypatch):
    monkeypatch.setattr(
        indexer, "get_llamaindex_config", lambda _cfg: SimpleNamespace(storage_dir="unused")
    )
    monkeypatch.setattr(indexer, "_active_storage_root", lambda _path: (object(), None))

    assert indexer.capture_index_generation(object()) == indexer.LEGACY_INDEX_GENERATION


def test_capture_index_generation_refuses_an_invalid_active_pointer(monkeypatch):
    monkeypatch.setattr(
        indexer, "get_llamaindex_config", lambda _cfg: SimpleNamespace(storage_dir="unused")
    )
    monkeypatch.setattr(indexer, "_active_storage_root", lambda _path: None)

    assert indexer.capture_index_generation(object()) is None


def test_retrieve_documents_refuses_an_invalid_active_pointer(monkeypatch):
    get_retrievers = Mock()
    monkeypatch.setattr(indexer, "capture_index_generation", lambda _cfg: None)
    monkeypatch.setattr("drbrain.rag.fusion.get_retrievers", get_retrievers)

    assert rag_agent.retrieve_documents(object(), object(), object(), "perovskite") == []
    get_retrievers.assert_not_called()


def test_pinned_generation_lookup_does_not_follow_a_new_active_pointer(tmp_path):
    root = tmp_path / "index"
    first = root / indexer.GENERATIONS_DIR_NAME / "g-first"
    second = root / indexer.GENERATIONS_DIR_NAME / "g-second"
    first.mkdir(parents=True)
    second.mkdir()
    (root / indexer.ACTIVE_POINTER_NAME).write_text(
        json.dumps({"generation": "g-second"}), encoding="utf-8"
    )

    active_root = indexer._storage_root_for_generation(root, None)
    pinned_root = indexer._storage_root_for_generation(root, "g-first")

    assert active_root == second
    assert pinned_root == first


def test_report_keeps_claim_to_evidence_links_and_rejects_unknown_ids():
    evidence = Evidence(
        evidence_id="ev-1",
        generation="g-1",
        document_locator={"paper_id": "paper-1"},
        chunk_locator={"node_id": "section-2"},
        content_checksum="a" * 64,
        rank=1,
        score=0.9,
    )
    state = ResearchState(
        hypotheses=[Hypothesis(claim_id="cl-1", statement="A supports B")],
        evidence=[evidence],
        evidence_bundles=[
            EvidenceBundle(bundle_id="eb-1", evidence_ids=["ev-1"], records=[evidence])
        ],
        verifications=[
            Verification(
                claim_id="cl-1",
                statement="A supports B",
                evidence_ids=["ev-1"],
                status="verified",
            )
        ],
    )

    report = ResearchLoopWorkflow._build_template_report(state)

    assert "cl-1" in report
    assert "ev-1" in report
    assert "generation=g-1" in report
    assert _referenced_evidence_ids(["ev-1", "unknown", "ev-1"], {"ev-1"}) == ["ev-1"]
    assert _has_required_evidence(state, ["ev-1"])
    assert not _has_required_evidence(state, [])
