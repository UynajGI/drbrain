"""RAG failure-semantics + ACL-enforcement tests.

Covers :mod:`drbrain.rag.status` and the ACL/failure hooks in
:mod:`drbrain.rag.fusion` (plus the engine's abstain path):

* ``RetrievalStatus`` / ``classify_failure`` — failure taxonomy
* ``RetrievalError`` — the "every leg failed" signal
* ``FusionRetriever._apply_acl`` — default-deny post-filter (match kept,
  missing key excluded, explicit wildcard)
* ``FusionRetriever`` — all-legs-fail raises ``RetrievalError``, single-leg
  failure still degrades, ``acl_filter`` enforced on the fused result
* ``ask_llamaindex`` — abstains (no LLM call) on retrieval failure vs no
  results

The status/RetrievalError/engine-abstain tests run without llama-index; the
ACL and fusion integration tests skip when it is missing.
"""

from types import SimpleNamespace

import pytest

from drbrain.rag.status import (
    RetrievalError,
    RetrievalStatus,
    classify_failure,
)

# ── RetrievalStatus / classify_failure ──────────────────────────────────────


def test_retrieval_status_values():
    assert RetrievalStatus.OK.value == "ok"
    assert RetrievalStatus.NO_RESULTS.value == "no_results"
    assert RetrievalStatus.RETRIEVAL_FAILURE.value == "retrieval_failure"
    assert RetrievalStatus.PERMISSION_DENIED.value == "permission_denied"
    assert RetrievalStatus.TIMEOUT.value == "timeout"
    assert RetrievalStatus.SOURCE_UNAVAILABLE.value == "source_unavailable"


def test_classify_failure_timeout_vs_generic():
    assert classify_failure(TimeoutError("vector store timed out")) is RetrievalStatus.TIMEOUT
    assert classify_failure(RuntimeError("db down")) is RetrievalStatus.RETRIEVAL_FAILURE
    assert classify_failure(ValueError("bad query")) is RetrievalStatus.RETRIEVAL_FAILURE


# ── RetrievalError ──────────────────────────────────────────────────────────


def test_retrieval_error_defaults():
    err = RetrievalError()
    assert isinstance(err, Exception)
    assert err.failures == []
    assert "retrieval failed" in str(err)


def test_retrieval_error_carries_failures():
    failures = [("vector", RetrievalStatus.TIMEOUT), ("bm25", RetrievalStatus.RETRIEVAL_FAILURE)]
    err = RetrievalError("all legs failed", failures=failures)
    assert err.failures == failures
    assert "all legs failed" in str(err)


# ── FusionRetriever failure distinction + ACL (llama-index required) ───────


def _make_retriever(nodes=(), error=None):
    """Minimal ``BaseRetriever`` returning fixed nodes, or raising ``error``."""
    from llama_index.core.retrievers import BaseRetriever

    class _Static(BaseRetriever):
        def __init__(self) -> None:
            super().__init__()
            self._nodes = list(nodes)
            self._error = error

        def _retrieve(self, query_bundle):  # noqa: ANN001
            if self._error is not None:
                raise self._error
            return list(self._nodes)

    return _Static()


def _mk(nid: str, meta: dict | None = None):
    from llama_index.core.schema import NodeWithScore, TextNode

    return NodeWithScore(node=TextNode(text="body", id_=nid, metadata=meta or {}), score=0.5)


def test_fusion_all_legs_fail_raises_retrieval_error():
    pytest.importorskip("llama_index")
    from drbrain.rag.fusion import FusionRetriever

    fused = FusionRetriever(
        [
            _make_retriever(error=RuntimeError("db down")),
            _make_retriever(error=TimeoutError("vector timeout")),
        ],
        sources=["bm25", "vector"],
    )
    with pytest.raises(RetrievalError) as exc_info:
        fused.retrieve("q")
    # per-leg status is captured: generic → RETRIEVAL_FAILURE, timeout → TIMEOUT.
    assert exc_info.value.failures == [
        ("bm25", RetrievalStatus.RETRIEVAL_FAILURE),
        ("vector", RetrievalStatus.TIMEOUT),
    ]


def test_fusion_single_leg_failure_still_degrades():
    pytest.importorskip("llama_index")
    from drbrain.rag.fusion import FusionRetriever

    fused = FusionRetriever(
        [_make_retriever(error=RuntimeError("boom")), _make_retriever(nodes=[_mk("z")])],
        sources=["bad", "good"],
    )
    out = fused.retrieve("q")
    assert [n.node.node_id for n in out] == ["z"]


def test_fusion_acl_filter_enforced_on_fused_result():
    pytest.importorskip("llama_index")
    from drbrain.rag.fusion import FusionRetriever

    nodes = [
        _mk("a", meta={"tenant_id": "company_A"}),
        _mk("b", meta={"tenant_id": "company_B"}),
        _mk("c", meta={}),  # missing tenant → default-deny
    ]
    fused = FusionRetriever(
        [_make_retriever(nodes=nodes)], sources=["x"], acl_filter={"tenant_id": "company_A"}
    )
    out = fused.retrieve("q")
    assert [n.node.node_id for n in out] == ["a"]


# ── FusionRetriever._apply_acl (default-deny post-filter) ───────────────────


def _nws(**meta):
    """Duck-typed ``NodeWithScore`` stand-in: a node with ``.metadata``."""
    return SimpleNamespace(node=SimpleNamespace(metadata=dict(meta)))


def _apply_acl(acl_filter, nodes):
    """Run ``FusionRetriever._apply_acl`` with the given filter on *nodes*."""
    from drbrain.rag.fusion import FusionRetriever

    return FusionRetriever([_make_retriever()], acl_filter=acl_filter)._apply_acl(nodes)


def test_apply_acl_matching_key_kept():
    pytest.importorskip("llama_index")
    nodes = [_nws(tenant_id="company_A"), _nws(tenant_id="company_B")]
    kept = _apply_acl({"tenant_id": "company_A"}, nodes)
    assert len(kept) == 1
    assert kept[0].node.metadata["tenant_id"] == "company_A"


def test_apply_acl_missing_key_excluded():
    pytest.importorskip("llama_index")
    # A node without the key is denied even though it "could be public" —
    # default-deny is safer than assuming an unclassified node is readable.
    nodes = [_nws(tenant_id="company_A"), _nws(title="no tenant metadata")]
    kept = _apply_acl({"tenant_id": "company_A"}, nodes)
    assert len(kept) == 1
    assert kept[0].node.metadata["tenant_id"] == "company_A"


def test_apply_acl_explicit_wildcard_matches_any_value_but_requires_key():
    pytest.importorskip("llama_index")
    nodes = [_nws(tenant_id="company_B"), _nws(title="no tenant metadata")]
    kept = _apply_acl({"tenant_id": "*"}, nodes)
    # wildcard relaxes the value, not the presence: missing key still denied.
    assert len(kept) == 1
    assert kept[0].node.metadata["tenant_id"] == "company_B"


def test_apply_acl_empty_filter_is_passthrough():
    pytest.importorskip("llama_index")
    nodes = [_nws(tenant_id="company_A")]
    assert _apply_acl(None, nodes) == nodes
    assert _apply_acl({}, nodes) == nodes


def test_apply_acl_multiple_keys_all_must_match():
    pytest.importorskip("llama_index")
    nodes = [
        _nws(tenant_id="company_A", user_id="u1"),
        _nws(tenant_id="company_A"),  # missing user_id → denied
        _nws(tenant_id="company_B", user_id="u1"),
    ]
    kept = _apply_acl({"tenant_id": "company_A", "user_id": "u1"}, nodes)
    assert len(kept) == 1
    assert kept[0].node.metadata == {"tenant_id": "company_A", "user_id": "u1"}


# ── engine abstain behavior (no llama-index needed: engine is mocked) ───────


def _patch_engine(monkeypatch, engine):
    monkeypatch.setattr(
        "drbrain.rag.engine.build_query_engine",
        lambda cfg, db, streaming=False, top_k=None, acl_filter=None: engine,
    )


def test_ask_llamaindex_abstains_on_retrieval_failure(monkeypatch):
    from drbrain.rag.engine import ask_llamaindex

    class _BoomEngine:
        def query(self, question):
            raise RetrievalError("all legs failed")

    _patch_engine(monkeypatch, _BoomEngine())
    result = ask_llamaindex({}, db=None, question="q?", streaming=False)
    assert result["status"] == "retrieval_failure"
    assert result["message"] == "检索失败,无法回答"
    assert result["answer"] == "检索失败,无法回答"  # abstention text, not an answer
    assert result["sources"] == []
    assert result["engine"] == "llamaindex"


def test_ask_llamaindex_abstains_on_no_results(monkeypatch):
    from drbrain.rag.engine import ask_llamaindex

    class _EmptyEngine:
        def query(self, question):
            return SimpleNamespace(source_nodes=[])

    _patch_engine(monkeypatch, _EmptyEngine())
    result = ask_llamaindex({}, db=None, question="q?", streaming=False)
    assert result["status"] == "no_results"
    assert result["message"] == "当前知识库中没有找到相关信息"
    assert result["answer"] == "当前知识库中没有找到相关信息"
    assert result["sources"] == []
    assert result["engine"] == "llamaindex"


def test_ask_llamaindex_streaming_abstains_on_retrieval_failure(monkeypatch):
    from drbrain.rag.engine import ask_llamaindex

    class _BoomEngine:
        def query(self, question):
            raise RetrievalError("all legs failed")

    _patch_engine(monkeypatch, _BoomEngine())
    items = list(ask_llamaindex({}, db=None, question="q?", streaming=True))
    assert len(items) == 1
    assert items[0]["status"] == "retrieval_failure"
    assert "chunk" not in items[0]
