import asyncio
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

from drbrain.loop.events import (
    Computed,
    Evidence,
    EvidenceBundle,
    Hypothesis,
    ResearchState,
    Verification,
)
from drbrain.loop.workflow import (
    ResearchLoopWorkflow,
    _has_required_evidence,
    _referenced_evidence_ids,
)
from drbrain.rag import agent as rag_agent
from drbrain.rag import indexer
from drbrain.rag.evidence import build_evidence_record
from drbrain.storage.database import Database


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


def test_resolved_generation_only_uses_persisted_retrievers(monkeypatch):
    observed: dict[str, object] = {}
    monkeypatch.setattr(indexer, "capture_index_generation", lambda _cfg: "g-pinned")

    def get_retrievers(*_args, **kwargs):
        observed.update(kwargs)
        return {}

    monkeypatch.setattr("drbrain.rag.fusion.get_retrievers", get_retrievers)

    assert rag_agent.retrieve_documents(object(), object(), object(), "perovskite") == []
    assert observed == {"generation": "g-pinned", "generation_backed_only": True}


def test_retrieval_tool_uses_the_resolved_generation_for_its_legs(monkeypatch):
    observed: dict[str, object] = {}
    monkeypatch.setattr(indexer, "capture_index_generation", lambda _cfg: "g-pinned")

    def get_retrievers(*_args, **kwargs):
        observed.update(kwargs)
        return {}

    monkeypatch.setattr("drbrain.rag.fusion.get_retrievers", get_retrievers)

    assert rag_agent._build_retrieval_tool(object(), object(), object()) is None
    assert observed == {"generation": "g-pinned", "generation_backed_only": True}


def test_retrieval_rows_bind_the_visible_excerpt_separately_from_the_full_chunk():
    full_text = "a" * 501
    node = SimpleNamespace(
        metadata={"paper_id": "paper-1", "node_id": "section-2", "title": "Stability"},
        get_content=lambda: full_text,
    )

    row = rag_agent._retrieval_rows(
        [SimpleNamespace(node=node, score=0.9)], generation="g-1", query="stability"
    )[0]

    assert row["text"] == full_text[:500]
    assert row["content_checksum"] == hashlib.sha256(full_text.encode("utf-8")).hexdigest()
    assert row["excerpt_checksum"] == hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
    assert row["content_length"] == len(full_text)
    assert row["excerpt_length"] == len(row["text"])

    unscored = rag_agent._retrieval_rows(
        [SimpleNamespace(node=node, score=None)], generation="g-1", query="stability"
    )[0]
    assert unscored["score"] is None


def test_explicit_unavailable_generation_does_not_recapture_the_active_pointer(monkeypatch):
    capture = Mock(return_value="g-new-active")
    monkeypatch.setattr(indexer, "capture_index_generation", capture)

    workflow = ResearchLoopWorkflow(cfg=object(), rag_generation=None)

    assert workflow._rag_generation is None
    capture.assert_not_called()


def _pinned_record() -> dict[str, object]:
    return {
        "evidence_id": "ev-1",
        "generation": "g-1",
        "document_locator": {"paper_id": "paper-1", "title": "Stability"},
        "chunk_locator": {"node_id": "section-2"},
        "content_checksum": "a" * 64,
        "query": "stability",
        "retriever": "fusion",
        "rank": 1,
        "score": 0.9,
        "text": "Stable passage",
    }


def test_brokerless_rag_evidence_is_durably_recorded(monkeypatch):
    monkeypatch.setattr(
        rag_agent, "retrieve_documents", lambda *_args, **_kwargs: [_pinned_record()]
    )

    async def immediate_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("drbrain.loop.workflow.asyncio.to_thread", immediate_to_thread)
    recorded: list[dict[str, object]] = []
    workflow = ResearchLoopWorkflow(
        cfg=object(),
        db=object(),
        graph=object(),
        rag_generation="g-1",
        evidence_recorder=recorded.append,
    )

    titles, bundle = asyncio.run(workflow._retrieve_rag_evidence("stability"))

    assert titles == ["Stability"]
    assert bundle is not None
    assert bundle.tool_call_id.startswith("rag-")
    assert recorded == [bundle.model_dump(mode="json")]


def test_brokered_rag_retrieval_runs_the_executor_off_loop(monkeypatch):
    monkeypatch.setattr(
        rag_agent, "retrieve_documents", lambda *_args, **_kwargs: [_pinned_record()]
    )

    class Broker:
        def __init__(self):
            self.recorded: list[dict[str, object]] = []
            self.policy = None

        async def execute(self, **kwargs):
            return SimpleNamespace(
                ok=True,
                output=await kwargs["executor"](),
                tool_call_id="rag-brokered",
            )

        def record_evidence_bundle(self, bundle):
            self.recorded.append(bundle)

    broker = Broker()
    workflow = ResearchLoopWorkflow(
        cfg=object(),
        db=object(),
        graph=object(),
        rag_generation="g-1",
        tool_broker=broker,
    )

    titles, bundle = asyncio.run(workflow._retrieve_rag_evidence("stability"))

    assert titles == ["Stability"]
    assert bundle is not None
    assert broker.recorded == [bundle.model_dump(mode="json")]


def test_brokerless_evidence_write_failure_is_fail_closed(monkeypatch):
    monkeypatch.setattr(
        rag_agent, "retrieve_documents", lambda *_args, **_kwargs: [_pinned_record()]
    )

    def fail_record(_bundle):
        raise OSError("ledger unavailable")

    workflow = ResearchLoopWorkflow(
        cfg=object(),
        db=object(),
        graph=object(),
        rag_generation="g-1",
        evidence_recorder=fail_record,
    )

    assert asyncio.run(workflow._retrieve_rag_evidence("stability")) == ([], None)


def test_brokerless_rag_ignores_unciteable_records(monkeypatch):
    monkeypatch.setattr(
        rag_agent, "retrieve_documents", lambda *_args, **_kwargs: [{"title": "No ID"}]
    )
    recorded: list[dict[str, object]] = []
    workflow = ResearchLoopWorkflow(
        cfg=object(),
        db=object(),
        graph=object(),
        rag_generation="g-1",
        evidence_recorder=recorded.append,
    )

    assert asyncio.run(workflow._retrieve_rag_evidence("stability")) == ([], None)
    assert recorded == []


def test_brokerless_rag_skips_a_malformed_record_without_dropping_valid_evidence(monkeypatch):
    malformed = {**_pinned_record(), "rank": "not-an-integer"}
    monkeypatch.setattr(
        rag_agent,
        "retrieve_documents",
        lambda *_args, **_kwargs: [_pinned_record(), malformed],
    )
    recorded: list[dict[str, object]] = []
    workflow = ResearchLoopWorkflow(
        cfg=object(),
        db=object(),
        graph=object(),
        rag_generation="g-1",
        evidence_recorder=recorded.append,
    )

    titles, bundle = asyncio.run(workflow._retrieve_rag_evidence("stability"))

    assert titles == ["Stability"]
    assert bundle is not None
    assert len(bundle.records) == 1
    assert recorded == [bundle.model_dump(mode="json")]


def test_verifier_accepts_a_claim_id_when_the_model_omits_the_statement(monkeypatch):
    hypothesis = Hypothesis(claim_id="cl-1", statement="Canonical claim", status="critiqued")
    evidence = Evidence(evidence_id="ev-1", generation="g-1")
    state = ResearchState(
        evidence=[evidence],
        evidence_bundles=[
            EvidenceBundle(bundle_id="eb-1", records=[evidence], evidence_ids=["ev-1"])
        ],
    )

    class Store:
        async def get(self, key, default=None):
            return state if key == "research_state" else default

        async def set(self, _key, _value):
            return None

    workflow = ResearchLoopWorkflow()
    monkeypatch.setattr(workflow, "build_node_agent", lambda **_kwargs: object())
    monkeypatch.setattr(workflow, "_has_compute_tools", lambda _agent: False)

    async def verifier_result(*_args, **_kwargs):
        return {
            "verifications": [
                {"claim_id": "cl-1", "evidence_ids": ["ev-1"], "supports": 1, "refutes": 0}
            ]
        }

    monkeypatch.setattr(workflow, "run_agent_json", verifier_result)
    result = asyncio.run(
        workflow.verify(SimpleNamespace(store=Store()), Computed(hypotheses=[hypothesis]))
    )

    assert result.verifications[0].claim_id == "cl-1"
    assert result.verifications[0].statement == "Canonical claim"


def test_verifier_ignores_an_ambiguous_duplicate_hypothesis_statement(monkeypatch):
    evidence = Evidence(evidence_id="ev-1", generation="g-1")
    state = ResearchState(
        evidence=[evidence],
        evidence_bundles=[
            EvidenceBundle(bundle_id="eb-1", records=[evidence], evidence_ids=["ev-1"])
        ],
    )

    class Store:
        async def get(self, key, default=None):
            return state if key == "research_state" else default

        async def set(self, _key, _value):
            return None

    workflow = ResearchLoopWorkflow()
    monkeypatch.setattr(workflow, "build_node_agent", lambda **_kwargs: object())
    monkeypatch.setattr(workflow, "_has_compute_tools", lambda _agent: False)

    async def verifier_result(*_args, **_kwargs):
        return {"verifications": [{"statement": "Duplicate", "evidence_ids": ["ev-1"]}]}

    monkeypatch.setattr(workflow, "run_agent_json", verifier_result)
    result = asyncio.run(
        workflow.verify(
            SimpleNamespace(store=Store()),
            Computed(
                hypotheses=[
                    Hypothesis(claim_id="cl-1", statement="Duplicate", status="critiqued"),
                    Hypothesis(claim_id="cl-2", statement="Duplicate", status="critiqued"),
                ]
            ),
        )
    )

    assert result.verifications == []


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
    assert "- [cl-1] A supports B" in report
    assert "ev-1" in report
    assert "generation=g-1" in report
    assert _referenced_evidence_ids(["ev-1", "unknown", "ev-1"], {"ev-1"}) == ["ev-1"]
    assert _has_required_evidence(state, ["ev-1"])
    assert not _has_required_evidence(state, [])
    assert not _has_required_evidence(ResearchState(), [], evidence_required=True)
    assert ResearchLoopWorkflow(
        rag_generation="g-1", require_rag_evidence=True
    )._requires_evidence_ids(ResearchState())


def test_loop_persists_verified_claim_with_its_real_retrieval_evidence(tmp_path):
    db = Database(tmp_path / "knowledge.sqlite3")
    evidence = Evidence(
        evidence_id="ev-real",
        generation="g-20260826-a",
        document_locator={"paper_id": "paper-1", "title": "Evidence paper"},
        chunk_locator={"node_id": "section-2", "chunk_id": "chunk-2"},
        content_checksum="a" * 64,
        excerpt_checksum="b" * 64,
        query="Does A support B?",
        retriever="fusion",
        rank=1,
        score=0.9,
        paper_id="paper-1",
        page=3,
        snippet="A supports B under controlled conditions.",
        value=0.9,
        unit="score",
        conditions={"temperature": "300K"},
        provenance="retrieval",
        authority="source",
    )
    state = ResearchState(
        task="autoresearch",
        evidence=[evidence],
        evidence_bundles=[EvidenceBundle(bundle_id="eb-1", evidence_ids=["ev-real"])],
        verifications=[
            Verification(
                claim_id="cl-1",
                statement="A supports B",
                evidence_ids=["ev-real"],
                status="verified",
            )
        ],
        verified=["A supports B"],
    )

    ResearchLoopWorkflow(db=db)._persist_claims(state)

    claim_id, evidence_id = db.conn.execute(
        "SELECT claim_id, evidence_id FROM claim_evidence"
    ).fetchone()
    assert evidence_id == "ev-real"
    evidence_row = db.conn.execute(
        "SELECT paper_id, node_id, page, snippet, conditions FROM evidence WHERE evidence_id = ?",
        (evidence_id,),
    ).fetchone()
    assert evidence_row[:4] == (
        "paper-1",
        "section-2",
        "3",
        "A supports B under controlled conditions.",
    )
    persisted = json.loads(evidence_row[4])
    assert persisted["generation"] == "g-20260826-a"
    assert persisted["content_checksum"] == "a" * 64
    assert persisted["chunk_locator"]["chunk_id"] == "chunk-2"
    assert db.conn.execute(
        "SELECT claim_text FROM claims WHERE claim_id = ?", (claim_id,)
    ).fetchone() == ("A supports B",)
    db.close()
