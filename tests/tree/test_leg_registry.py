"""T43: the outer layer registers only bm25/vector/tree, each exactly once."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from drbrain.config import EmbedConfig, LlamaIndexConfig
from drbrain.rag import fusion
from drbrain.rag.legs import CANONICAL_LEGS, LegConfigError, normalize_legs


def _cfg(papers_dir: str = ".") -> SimpleNamespace:
    return SimpleNamespace(
        llm=SimpleNamespace(models=[]),
        dirs=SimpleNamespace(papers=papers_dir),
        embed=EmbedConfig(device="cpu", top_k=10),
        llamaindex=LlamaIndexConfig(
            fusion_mode="reciprocal_rank", retrievers=["tree"], rerank=False
        ),
    )


class TestNormalizeLegs:
    def test_the_outer_layer_is_the_three_production_legs(self):
        assert normalize_legs(["vector", "tree", "bm25"]).legs == CANONICAL_LEGS
        assert normalize_legs(None).legs == ("bm25", "vector")

    def test_legacy_names_merge_into_the_single_tree_leg(self):
        normalized = normalize_legs(["pageindex", "raptor"])
        assert normalized.legs == ("tree",)
        assert len(normalized.notes) == 1
        assert "pageindex" in normalized.notes[0] and "raptor" in normalized.notes[0]
        assert normalize_legs(["pageindex"]).legs == ("tree",)
        assert normalize_legs(["raptor"]).legs == ("tree",)

    def test_new_and_legacy_tree_names_conflict(self):
        for wanted in (["tree", "pageindex"], ["tree", "raptor"]):
            with pytest.raises(LegConfigError, match="tree"):
                normalize_legs(wanted)

    def test_unknown_names_are_refused(self):
        with pytest.raises(LegConfigError, match="unknown retriever"):
            normalize_legs(["dense"])

    def test_extras_are_not_part_of_the_three_legs(self):
        normalized = normalize_legs(["tree", "graph"])
        assert normalized.legs == ("tree",)
        assert normalized.extras == ("graph",)
        assert normalized.as_list() == ["tree", "graph"]


@pytest.mark.skipif(not fusion._LLAMA_INDEX_AVAILABLE, reason="llama-index required")
class TestFusionRegistration:
    def test_legacy_sources_fold_into_one_tree_leg(self):
        marker_a, marker_b = object(), object()
        legs = fusion._iter_custom_retrievers({"pageindex": marker_a})
        assert [name for name, _ in legs] == ["tree"]
        with pytest.raises(LegConfigError, match="duplicates the 'tree' leg"):
            fusion._iter_custom_retrievers({"tree": marker_a, "pageindex": marker_b})

    def test_multi_layer_tree_leg_contributes_one_source_label(self):
        from llama_index.core.schema import NodeWithScore, TextNode

        def scored(node_id: str, layer: str, score: float):
            node = TextNode(
                text=f"body of {node_id}",
                id_=f"pX:{node_id}",
                metadata={"paper_id": "pX", "node_id": node_id, "tree_layer": layer},
            )
            return NodeWithScore(node=node, score=score)

        class _Static:
            def __init__(self, nodes):
                self._nodes = nodes

            def retrieve(self, query_bundle):
                return list(self._nodes)

        tree = _Static([scored("nl-1", "leaf", 0.9), scored("nr-1", "region", 0.8)])
        fused = fusion.build_fusion_retriever(_cfg(), custom_retrievers={"tree": tree}, top_k=10)
        out = fused.retrieve("q")
        assert {item.node.node_id for item in out} == {"pX:nl-1", "pX:nr-1"}
        for item in out:
            assert item.node.metadata["source"] == "tree"
            assert item.node.metadata["sources"] == ["tree"]

    def test_get_retrievers_folds_legacy_names(self, tmp_path, monkeypatch):
        from drbrain.rag.retrievers import UnifiedTreeRetriever

        monkeypatch.setattr(
            "drbrain.tree.publish.get_active_tree_generation", lambda root: "gen-test"
        )
        cfg = _cfg(str(tmp_path))
        cfg.llamaindex.retrievers = ["pageindex"]
        retrievers = fusion.get_retrievers(cfg)
        assert set(retrievers) == {"tree"}
        assert isinstance(retrievers["tree"], UnifiedTreeRetriever)

    def test_missing_unified_generation_omits_the_tree_leg(self, tmp_path, monkeypatch):
        """Fail-closed: no generation means no tree leg, never the legacy walker."""
        monkeypatch.setattr("drbrain.tree.publish.get_active_tree_generation", lambda root: None)
        cfg = _cfg(str(tmp_path))
        cfg.llamaindex.retrievers = ["tree"]
        assert fusion.get_retrievers(cfg) == {}
