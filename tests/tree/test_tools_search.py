"""T38-T40: node tools, the tree search entry and the stateful navigator."""

from __future__ import annotations

import hashlib

import pytest

from drbrain.storage.database import Database
from drbrain.tree.blocks import build_content_blocks
from drbrain.tree.contracts import ChildRef, LeafRef, NodeRecord, leaf_node_id, region_node_id
from drbrain.tree.navigator import TreeNavigator
from drbrain.tree.search import TreeSearch
from drbrain.tree.tools import ToolBudget, ToolError, ToolState, TreeTools
from drbrain.tree.vector_store import UnifiedVectorStore, VectorEntry

DOC_A = "# Methods\n\nalpha beta gamma delta epsilon zeta\n\n# Results\n\neta theta iota kappa\n"
DOC_B = "# Methods\n\nlambda mu nu xi omicron pi\n"


def _doc(db: Database, local_id: str, text: str):
    db.insert_paper(local_id, "T", 2024, "uploaded")
    blocks = build_content_blocks(
        text, local_id=local_id, revision=1, media_type="md", parser="test"
    )
    db.upsert_document_revision(
        local_id,
        1,
        source_hash=f"s-{local_id}",
        canonical_hash=hashlib.sha256(text.encode()).hexdigest(),
        media_type="md",
    )
    db.insert_content_blocks(blocks)
    return blocks


def _leaf(block, local_id: str) -> NodeRecord:
    ref = LeafRef(
        local_id=local_id,
        revision=1,
        block_id=block.block_id,
        char_start=0,
        char_end=len(block.text),
    )
    return NodeRecord(
        node_id=leaf_node_id(ref),
        revision=1,
        kind="leaf",
        state="ready",
        layer=0,
        content_hash=block.text_hash,
        leaf=ref,
        heading_path=block.heading_path,
    )


def _region(children, *, layer=1, summary="joint summary", contract=None) -> NodeRecord:
    contract = contract or {"prompt": "p"}
    refs = tuple(
        ChildRef(child_id=child.node_id, child_revision=1, ordinal=index)
        for index, child in enumerate(children)
    )
    return NodeRecord(
        node_id=region_node_id(refs, contract),
        revision=1,
        kind="region",
        state="ready",
        layer=layer,
        content_hash=hashlib.sha256(summary.encode()).hexdigest(),
        summary=summary,
        children=refs,
        contract=contract,
    )


@pytest.fixture()
def corpus(tmp_path):
    db = Database(tmp_path / "db.sqlite")
    blocks_a = _doc(db, "p1", DOC_A)
    blocks_b = _doc(db, "p2", DOC_B)
    leaves = [
        _leaf(blocks_a[1], "p1"),
        _leaf(blocks_a[3], "p1"),
        _leaf(blocks_b[1], "p2"),
    ]
    for leaf in leaves:
        db.insert_tree_node(leaf, publish=True)
    top = _region(leaves[:2], summary="p1 summary")
    db.insert_tree_node(top, publish=True)
    return db, leaves, top


class TestTools:
    def test_visibility_rules(self, corpus):
        db, leaves, top = corpus
        tools = TreeTools(db)
        assert tools.visible(leaves[0].node_id)
        from drbrain.tree.contracts import ChildRef, region_node_id

        refs = (ChildRef(child_id=leaves[0].node_id, child_revision=1, ordinal=0),)
        staging = NodeRecord(
            node_id=region_node_id(refs, {"p": "q"}),
            revision=1,
            kind="region",
            state="staging",
            layer=1,
            content_hash=hashlib.sha256(b"x").hexdigest(),
            summary="x",
            children=refs,
            contract={"p": "q"},
        )
        db.insert_tree_node(staging)
        assert not tools.visible(staging.node_id)
        with pytest.raises(ToolError, match="only published"):
            tools.read(staging.node_id, ToolState())

    def test_read_returns_exact_text_and_receipt(self, corpus):
        db, leaves, top = corpus
        tools = TreeTools(db)
        state = ToolState()
        text, receipt = tools.read(leaves[0].node_id, state, request_id="q1")
        assert text.startswith("alpha beta")
        assert receipt.tool == "read" and receipt.tokens > 0
        assert receipt.char_end > receipt.char_start
        # Same read twice -> identical text and identical span.
        again, receipt2 = tools.read(leaves[0].node_id, ToolState())
        assert again == text and receipt2.span_key == receipt.span_key

    def test_subrange_read_stays_inside_the_leaf(self, corpus):
        db, leaves, top = corpus
        tools = TreeTools(db)
        text, receipt = tools.read(leaves[0].node_id, ToolState(), char_start=0, char_end=5)
        assert text == "alpha"
        assert receipt.char_start == int(db.get_tree_node(leaves[0].node_id)["char_start"])
        with pytest.raises(ToolError, match="outside the leaf"):
            tools.read(leaves[0].node_id, ToolState(), char_start=0, char_end=10_000)

    def test_expand_and_parents_are_reverse_consistent(self, corpus):
        db, leaves, top = corpus
        tools = TreeTools(db)
        state = ToolState()
        children = tools.expand(top.node_id, state)
        assert [child["node_id"] for child in children] == [leaf.node_id for leaf in leaves[:2]]
        parents = tools.parents(leaves[0].node_id, ToolState())
        assert [parent["node_id"] for parent in parents] == [top.node_id]

    def test_read_scope_covers_unique_spans_once(self, corpus):
        db, leaves, top = corpus
        tools = TreeTools(db)
        state = ToolState()
        spans = tools.read_scope(top.node_id, state)
        assert len(spans) == 2
        assert {receipt.span_key for _text, receipt in spans} == {
            (receipt.local_id, receipt.block_id, receipt.char_start, receipt.char_end)
            for receipt in (span[1] for span in spans)
        }
        # Re-reading through the parent does not duplicate the same span.
        assert len({span[1].span_key for span in spans}) == len(spans)

    def test_budget_truncation_is_reported(self, corpus):
        db, leaves, top = corpus
        tools = TreeTools(db, budget=ToolBudget(max_tokens=5, max_calls=100))
        state = ToolState()
        with pytest.raises(ToolError, match="token budget"):
            tools.read(leaves[0].node_id, state)
        assert state.truncated

    def test_region_summary_reads_whole(self, corpus):
        db, leaves, top = corpus
        tools = TreeTools(db)
        text, receipt = tools.read(top.node_id, ToolState())
        assert text == "p1 summary"
        with pytest.raises(ToolError, match="whole"):
            tools.read(top.node_id, ToolState(), char_start=0, char_end=3)


class TestSearchEntry:
    def _store(self, tmp_path, entries):
        store = UnifiedVectorStore(tmp_path / "zvec", create=True, dimension=3)
        store.upsert(entries)
        return store

    def test_search_returns_all_layers(self, tmp_path, corpus):
        db, leaves, top = corpus
        store = self._store(
            tmp_path,
            [
                VectorEntry(
                    node_id=leaves[0].node_id,
                    node_revision=1,
                    kind="leaf",
                    local_id="p1",
                    layer=0,
                    content_hash="h0",
                    profile_id="emb-test",
                    vector=(1.0, 0.0, 0.0),
                ),
                VectorEntry(
                    node_id=top.node_id,
                    node_revision=1,
                    kind="region",
                    local_id="",
                    layer=1,
                    content_hash="h1",
                    profile_id="emb-test",
                    vector=(0.9, 0.1, 0.0),
                ),
            ],
        )
        search = TreeSearch(store, profile_id="emb-test")
        found = search.search((1.0, 0.0, 0.0), top_k=5)
        assert {candidate.node_id for candidate in found} == {leaves[0].node_id, top.node_id}
        assert [candidate.node_id for candidate in found][0] == leaves[0].node_id

    def test_search_does_not_depend_on_bm25(self, tmp_path, corpus):
        """No text index is consulted: an empty FTS still yields candidates."""
        db, leaves, top = corpus
        # Make the text index useless without touching the ANN index.
        db.conn.execute("INSERT INTO content_fts(content_fts) VALUES('delete-all')")
        db.conn.commit()
        store = self._store(
            tmp_path,
            [
                VectorEntry(
                    node_id=leaves[0].node_id,
                    node_revision=1,
                    kind="leaf",
                    local_id="p1",
                    layer=0,
                    content_hash="h0",
                    profile_id="emb-test",
                    vector=(1.0, 0.0, 0.0),
                )
            ],
        )
        search = TreeSearch(store, profile_id="emb-test")
        assert search.search((1.0, 0.0, 0.0), top_k=3)

    def test_ready_only_filters_unpublished_nodes(self, tmp_path, corpus):
        db, leaves, top = corpus
        store = self._store(
            tmp_path,
            [
                VectorEntry(
                    node_id="nl-ghost",
                    node_revision=1,
                    kind="leaf",
                    local_id="p9",
                    layer=0,
                    content_hash="hx",
                    profile_id="emb-test",
                    vector=(1.0, 0.0, 0.0),
                )
            ],
        )
        search = TreeSearch(store, profile_id="emb-test")
        assert len(search.search((1.0, 0.0, 0.0), top_k=3)) == 1
        assert search.search((1.0, 0.0, 0.0), top_k=3, ready_only=[leaves[0].node_id]) == []

    def test_leaf_view_is_available_for_the_vector_leg(self, tmp_path, corpus):
        db, leaves, top = corpus
        store = self._store(
            tmp_path,
            [
                VectorEntry(
                    node_id=leaves[0].node_id,
                    node_revision=1,
                    kind="leaf",
                    local_id="p1",
                    layer=0,
                    content_hash="h0",
                    profile_id="emb-test",
                    vector=(1.0, 0.0, 0.0),
                ),
                VectorEntry(
                    node_id=top.node_id,
                    node_revision=1,
                    kind="region",
                    local_id="",
                    layer=1,
                    content_hash="h1",
                    profile_id="emb-test",
                    vector=(1.0, 0.0, 0.0),
                ),
            ],
        )
        search = TreeSearch(store, profile_id="emb-test")
        assert [c.node_id for c in search.search((1.0, 0.0, 0.0), top_k=5, view="leaf")] == [
            leaves[0].node_id
        ]


class TestNavigator:
    def _search(self, tmp_path, entries):
        store = UnifiedVectorStore(tmp_path / "zvec", create=True, dimension=3)
        store.upsert(entries)
        return TreeSearch(store, profile_id="emb-test")

    def test_leaf_hits_produce_evidence(self, tmp_path, corpus):
        db, leaves, top = corpus
        search = self._search(
            tmp_path,
            [
                VectorEntry(
                    node_id=leaves[0].node_id,
                    node_revision=1,
                    kind="leaf",
                    local_id="p1",
                    layer=0,
                    content_hash="h0",
                    profile_id="emb-test",
                    vector=(1.0, 0.0, 0.0),
                )
            ],
        )
        candidates = search.search((1.0, 0.0, 0.0), top_k=3)
        result = TreeNavigator(db).navigate("alpha", candidates)
        assert result.status == "ok"
        assert result.evidence[0]["source"] == "leaf"
        assert "alpha beta" in result.evidence[0]["text"]
        assert result.receipts and result.budget["nodes_read"] == 1

    def test_summary_hits_expand_to_backing_text(self, tmp_path, corpus):
        db, leaves, top = corpus
        search = self._search(
            tmp_path,
            [
                VectorEntry(
                    node_id=top.node_id,
                    node_revision=1,
                    kind="region",
                    local_id="",
                    layer=1,
                    content_hash="h1",
                    profile_id="emb-test",
                    vector=(1.0, 0.0, 0.0),
                )
            ],
        )
        candidates = search.search((1.0, 0.0, 0.0), top_k=3)
        result = TreeNavigator(db).navigate("summary query", candidates)
        sources = {item["source"] for item in result.evidence}
        assert sources == {"summary", "leaf"}
        leaf_evidence = [item for item in result.evidence if item["source"] == "leaf"]
        assert leaf_evidence and all(item["via"] == top.node_id for item in leaf_evidence)

    def test_same_span_is_never_counted_twice(self, tmp_path, corpus):
        db, leaves, top = corpus
        search = self._search(
            tmp_path,
            [
                VectorEntry(
                    node_id=top.node_id,
                    node_revision=1,
                    kind="region",
                    local_id="",
                    layer=1,
                    content_hash="h1",
                    profile_id="emb-test",
                    vector=(1.0, 0.0, 0.0),
                ),
                VectorEntry(
                    node_id=leaves[0].node_id,
                    node_revision=1,
                    kind="leaf",
                    local_id="p1",
                    layer=0,
                    content_hash="h0",
                    profile_id="emb-test",
                    vector=(1.0, 0.0, 0.0),
                ),
            ],
        )
        candidates = search.search((1.0, 0.0, 0.0), top_k=5)
        result = TreeNavigator(db).navigate("both", candidates)
        spans = [receipt.span_key for receipt in result.receipts]
        assert len(spans) == len(set(spans))

    def test_budget_exhaustion_reports_partial(self, tmp_path, corpus):
        db, leaves, top = corpus
        search = self._search(
            tmp_path,
            [
                VectorEntry(
                    node_id=leaves[0].node_id,
                    node_revision=1,
                    kind="leaf",
                    local_id="p1",
                    layer=0,
                    content_hash="h0",
                    profile_id="emb-test",
                    vector=(1.0, 0.0, 0.0),
                ),
                VectorEntry(
                    node_id=leaves[1].node_id,
                    node_revision=1,
                    kind="leaf",
                    local_id="p1",
                    layer=0,
                    content_hash="h1",
                    profile_id="emb-test",
                    vector=(1.0, 0.0, 0.0),
                ),
            ],
        )
        candidates = search.search((1.0, 0.0, 0.0), top_k=5)
        # Allow exactly one read, then let the second one exhaust the budget.
        from drbrain.services.tokens import count_tokens

        one_read = count_tokens(TreeTools(db).node_text(leaves[0].node_id))
        navigator = TreeNavigator(db, budget=ToolBudget(max_tokens=one_read + 1, max_calls=100))
        result = navigator.navigate("alpha", candidates)
        assert result.status == "partial"
        assert result.reason == "budget_exhausted"
        assert result.budget["truncated"] is True
        assert result.evidence, "the successful read must still be reported"

    def test_empty_candidates_are_explicit(self, tmp_path, corpus):
        db, leaves, top = corpus
        result = TreeNavigator(db).navigate("nothing", [])
        assert result.status == "empty" and result.reason == "no_candidates"

    def test_navigator_reports_unavailable_search(self, tmp_path, corpus):
        db, leaves, top = corpus
        search = self._search(tmp_path, [])
        candidates = search.search((1.0, 0.0, 0.0), top_k=3)
        result = TreeNavigator(db).navigate("q", candidates)
        assert result.status == "empty"
