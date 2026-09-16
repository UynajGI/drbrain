"""T36: job claims, checkpoints and build re-run idempotency."""

from __future__ import annotations

import hashlib

import pytest

from drbrain.storage.database import Database
from drbrain.tree.blocks import build_content_blocks
from drbrain.tree.builder import BuilderConfig, TreeBuilder
from drbrain.tree.clustering import ClusteringParams
from drbrain.tree.contracts import LeafRef, NodeRecord, leaf_node_id
from drbrain.tree.cost import CostParams
from drbrain.tree.jobs import JobError, TreeJobStore
from drbrain.tree.summary import SummaryContract, SummaryResponse, SummaryService


def _count(text: str) -> int:
    return max(1, len(text.split()))


def _seed(db: Database, papers: int = 3, sections: int = 4) -> list[str]:
    leaves: list[str] = []
    for paper in range(papers):
        text = "".join(
            f"# Section {section}\n\n" + " ".join([f"token{paper}x{section}"] * 25) + "\n\n"
            for section in range(sections)
        )
        db.insert_paper(f"p{paper}", "T", 2024, "uploaded")
        blocks = build_content_blocks(
            text, local_id=f"p{paper}", revision=1, media_type="md", parser="test"
        )
        db.upsert_document_revision(
            f"p{paper}",
            1,
            source_hash=f"src{paper}",
            canonical_hash=hashlib.sha256(text.encode()).hexdigest(),
            media_type="md",
        )
        db.insert_content_blocks(blocks)
        for block in blocks:
            if block.kind != "paragraph":
                continue
            ref = LeafRef(
                local_id=f"p{paper}",
                revision=1,
                block_id=block.block_id,
                char_start=0,
                char_end=len(block.text),
            )
            leaf = NodeRecord(
                node_id=leaf_node_id(ref),
                revision=1,
                kind="leaf",
                state="ready",
                layer=0,
                content_hash=block.text_hash,
                leaf=ref,
                heading_path=block.heading_path,
            )
            db.insert_tree_node(leaf, publish=True)
            leaves.append(leaf.node_id)
    return leaves


def _builder(db: Database, model) -> TreeBuilder:
    builder = TreeBuilder(
        db,
        config=BuilderConfig(
            clustering=ClusteringParams(dim=4, max_clusters=6),
            cost=CostParams(summary_output_budget=16, tool_overhead_tokens=1),
            contract=SummaryContract(model="fake", max_output_tokens=16, input_budget=4000),
            lam=4.0,
            max_layers=3,
        ),
        count_tokens=_count,
        embed=lambda texts: [[float(len(text) % 5), 0.5, 0.25, 0.125] for text in texts],
    )
    builder.model = model
    builder._model_impl = model
    builder.profile_id = "emb-test"
    return builder


class CountingModel:
    def __init__(self):
        self.calls = 0

    def complete(self, prompt: str, *, max_tokens: int) -> SummaryResponse:
        self.calls += 1
        return SummaryResponse(f"summary {self.calls}")


class TestJobLedger:
    def test_claim_is_exclusive_and_expiry_allows_reclaim(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        store = TreeJobStore(db)
        job_id = store.create("scope-1")
        first = store.claim(job_id, "worker-a", ttl_seconds=600)
        assert first.granted
        second = store.claim(job_id, "worker-b", ttl_seconds=600)
        assert not second.granted
        db.conn.execute(
            "UPDATE tree_build_jobs SET claim_expires_at = datetime('now', '-10 seconds') "
            "WHERE job_id = ?",
            (job_id,),
        )
        db.conn.commit()
        reclaimed = store.claim(job_id, "worker-b", ttl_seconds=600)
        assert reclaimed.granted
        assert db.get_tree_job(job_id)["owner"] == "worker-b"

    def test_checkpoint_roundtrip_and_owner_guard(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        store = TreeJobStore(db)
        job_id = store.create("scope-2")
        store.claim(job_id, "worker-a")
        store.save_checkpoint(
            job_id,
            owner="worker-a",
            checkpoint={"next_round": 3, "created": ["nr-1"]},
            metrics={"prompt_tokens": 42},
        )
        claim = store.claim(job_id, "worker-a")  # already running, same owner
        assert not claim.granted  # a running job is not re-claimable by design
        row = db.get_tree_job(job_id)
        assert "next_round" in row["checkpoint_json"]
        with pytest.raises(ValueError, match="not claimed"):
            store.save_checkpoint(job_id, owner="worker-b", checkpoint={"x": 1})

    def test_finish_states(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        store = TreeJobStore(db)
        done_id = store.create("s")
        store.claim(done_id, "w")
        store.finish(done_id, done=True)
        assert db.get_tree_job(done_id)["state"] == "done"
        failed_id = store.create("s")
        store.claim(failed_id, "w")
        store.finish(failed_id, done=False, reason="boom")
        row = db.get_tree_job(failed_id)
        assert row["state"] == "failed" and row["reason"] == "boom"
        paused_id = store.create("s")
        store.pause(paused_id)
        assert db.get_tree_job(paused_id)["state"] == "paused"
        with pytest.raises(JobError):
            store.claim("job-missing", "w")


class TestBuildIdempotency:
    def test_second_build_reuses_everything(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        leaves = _seed(db)
        model = CountingModel()
        builder = _builder(db, model)
        first = builder.build(leaves)
        assert first.created_nodes
        first_calls = model.calls
        second_model = CountingModel()
        second = _builder(db, second_model).build(leaves)
        assert second.created_nodes == [] or set(second.created_nodes) <= set(first.created_nodes)
        assert second_model.calls == 0, "cached summaries must not be recomputed"
        assert db.count_summary_cache("ready") >= len(first.created_nodes)
        del first_calls

    def test_node_revisions_are_preserved_across_reruns(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        leaves = _seed(db)
        builder = _builder(db, CountingModel())
        first = builder.build(leaves)
        revisions = {
            node_id: db.get_tree_node(node_id)["revision"] for node_id in first.created_nodes
        }
        _builder(db, CountingModel()).build(leaves)
        for node_id, revision in revisions.items():
            assert db.get_tree_node(node_id)["revision"] == revision

    def test_contract_change_publishes_new_identity(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        leaves = _seed(db)
        first = _builder(db, CountingModel()).build(leaves)
        other = _builder(db, CountingModel())
        other.config = BuilderConfig(
            clustering=other.config.clustering,
            cost=other.config.cost,
            contract=SummaryContract(model="other-model", max_output_tokens=16, input_budget=4000),
            lam=other.config.lam,
            max_layers=other.config.max_layers,
        )
        second = other.build(leaves)
        overlap = set(first.created_nodes) & set(second.created_nodes)
        assert not overlap, "a different contract must be a different node identity"
        for node_id in first.created_nodes:
            assert db.get_tree_node(node_id)["state"] == "ready"

    def test_builder_uses_shared_summary_service(self, tmp_path):
        db = Database(tmp_path / "db.sqlite")
        leaves = _seed(db, papers=2, sections=3)
        service = SummaryService(db, count_tokens=_count)
        builder = _builder(db, CountingModel())
        builder.summary = service
        builder.build(leaves)
        assert service.model_calls > 0
