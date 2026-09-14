"""T24/T25: shared Zvec vector store for leaves and regions."""

from __future__ import annotations

import pytest

from drbrain.storage.database import Database
from drbrain.tree.vector_store import (
    UnifiedVectorStore,
    VectorEntry,
    VectorStoreError,
    needs_write,
)

zvec = pytest.importorskip("zvec", reason="zvec package required for the ANN tests")


def _entry(
    node_id: str,
    *,
    kind="leaf",
    local_id="p1",
    layer=0,
    vec=(1.0, 0.0, 0.0),
    revision=1,
    profile="emb-test",
    content_hash=None,
) -> VectorEntry:
    return VectorEntry(
        node_id=node_id,
        node_revision=revision,
        kind=kind,
        local_id=local_id,
        layer=layer,
        content_hash=content_hash or ("h-" + node_id),
        profile_id=profile,
        vector=tuple(vec),
    )


@pytest.fixture()
def store(tmp_path):
    with UnifiedVectorStore(tmp_path / "zvec", create=True, dimension=3) as instance:
        yield instance


class TestWrites:
    def test_roundtrip_and_idempotent_upsert(self, store):
        entries = [
            _entry("nl-a"),
            _entry("nl-b", vec=(0.0, 1.0, 0.0)),
            _entry("nr-c", kind="region", layer=1, vec=(0.0, 0.0, 1.0)),
        ]
        assert store.upsert(entries) == 3
        hits = store.query((1.0, 0.0, 0.0), top_k=5)
        assert {hit.node_id for hit in hits} == {"nl-a", "nl-b", "nr-c"}
        # Re-writing the same entries must not duplicate docs.
        store.upsert(entries)
        assert len(store.query((1.0, 0.0, 0.0), top_k=10)) == 3

    def test_mixed_dimensions_rejected(self, store):
        with pytest.raises(VectorStoreError, match="dimensions"):
            store.upsert([_entry("nl-a", vec=(1.0, 0.0)), _entry("nl-b", vec=(1.0, 0.0, 0.0))])

    def test_revision_replacement_keeps_one_doc(self, store):
        store.upsert([_entry("nl-a", vec=(1.0, 0.0, 0.0))])
        store.upsert([_entry("nl-a", revision=2, vec=(0.0, 1.0, 0.0))])
        hits = store.query((0.0, 1.0, 0.0), top_k=5)
        assert len(hits) == 1 and hits[0].node_revision == 2

    def test_delete_removes_doc(self, store):
        store.upsert([_entry("nl-a")])
        store.delete(["nl-a"])
        assert store.query((1.0, 0.0, 0.0), top_k=5) == []


class TestViews:
    def test_leaf_view_excludes_regions_and_all_view_keeps_them(self, store):
        store.upsert(
            [
                _entry("nl-a", vec=(1.0, 0.0, 0.0)),
                _entry("nr-top", kind="region", layer=2, vec=(0.9, 0.1, 0.0)),
            ]
        )
        leaf_hits = store.query((1.0, 0.0, 0.0), top_k=5, view="leaf")
        assert [hit.node_id for hit in leaf_hits] == ["nl-a"]
        all_hits = store.query((1.0, 0.0, 0.0), top_k=5, view="all")
        assert {hit.node_id for hit in all_hits} == {"nl-a", "nr-top"}

    def test_scope_and_profile_filters(self, store):
        store.upsert(
            [
                _entry("nl-a", local_id="p1"),
                _entry("nl-b", local_id="p2", vec=(0.0, 1.0, 0.0)),
                _entry("nl-c", local_id="p2", profile="emb-other", vec=(0.0, 0.0, 1.0)),
            ]
        )
        scoped = store.query((1.0, 0.0, 0.0), top_k=5, local_ids=["p2"])
        assert {hit.node_id for hit in scoped} == {"nl-b", "nl-c"}
        profiled = store.query((1.0, 0.0, 0.0), top_k=5, profile_id="emb-other")
        assert [hit.node_id for hit in profiled] == ["nl-c"]

    def test_revision_filter_rejects_mixed_reads(self, store):
        store.upsert([_entry("nl-a", revision=1), _entry("nl-b", revision=2, vec=(0.0, 1.0, 0.0))])
        hits = store.query((1.0, 0.0, 0.0), top_k=5, node_revision=2)
        assert [hit.node_id for hit in hits] == ["nl-b"]

    def test_scores_are_similarity_ordered(self, store):
        store.upsert(
            [
                _entry("nl-a", vec=(1.0, 0.0, 0.0)),
                _entry("nl-b", vec=(0.5, 0.5, 0.0)),
                _entry("nl-c", vec=(0.0, 1.0, 0.0)),
            ]
        )
        hits = store.query((1.0, 0.0, 0.0), top_k=5)
        assert [hit.node_id for hit in hits][0] == "nl-a"
        assert hits[0].score > hits[-1].score

    def test_unknown_view_rejected(self, store):
        with pytest.raises(ValueError, match="view"):
            store.query((1.0, 0.0, 0.0), view="everything")


class TestMetadata:
    def test_metadata_has_no_float_copy(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        columns = {row[1] for row in db.conn.execute("PRAGMA table_info(node_vectors)").fetchall()}
        assert {"node_id", "node_revision", "profile_id", "content_hash", "state"} <= columns
        assert not {"embedding", "vector", "blob", "vec"} & columns

    def test_needs_write_decision(self):
        entry = _entry("nl-a", revision=2)
        assert needs_write(None, entry)
        assert needs_write(
            {
                "state": "staging",
                "node_revision": 2,
                "content_hash": entry.content_hash,
                "profile_id": entry.profile_id,
            },
            entry,
        )
        ready = {
            "state": "ready",
            "node_revision": 2,
            "content_hash": entry.content_hash,
            "profile_id": entry.profile_id,
        }
        assert not needs_write(ready, entry)
        for field, value in (
            ("state", "failed"),
            ("node_revision", 1),
            ("content_hash", "old"),
            ("profile_id", "emb-old"),
        ):
            broken = dict(ready)
            broken[field] = value
            assert needs_write(broken, entry), field

    def test_metadata_upsert_and_listing(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        db.upsert_node_vector(
            "nl-a",
            node_revision=1,
            kind="leaf",
            profile_id="emb-x",
            content_hash="h1",
            dimension=3,
            local_id="p1",
        )
        assert db.get_node_vector("nl-a")["state"] == "staging"
        db.upsert_node_vector(
            "nl-a",
            node_revision=1,
            kind="leaf",
            profile_id="emb-x",
            content_hash="h1",
            dimension=3,
            local_id="p1",
            state="ready",
        )
        assert db.get_node_vector("nl-a")["state"] == "ready"
        db.upsert_node_vector(
            "nr-b",
            node_revision=1,
            kind="region",
            profile_id="emb-x",
            content_hash="h2",
            dimension=3,
            state="ready",
        )
        assert db.count_node_vectors(state="ready") == 2
        assert [row["node_id"] for row in db.list_node_vectors(kind="region")] == ["nr-b"]
        db.delete_node_vector("nr-b")
        assert db.get_node_vector("nr-b") is None
        with pytest.raises(ValueError, match="kind"):
            db.upsert_node_vector(
                "x", node_revision=1, kind="leafy", profile_id="p", content_hash="h", dimension=1
            )
