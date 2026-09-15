"""T56: evaluation-only baselines with injectable algorithms and cost."""

from __future__ import annotations

import json
from unittest import mock

import pytest
import typer

from drbrain.rag.baselines import (
    BASELINES,
    BaselineRunner,
    RaptorBuild,
    baseline_algorithm,
    evaluate_baseline,
)


def _rows(*keys: str) -> list[dict]:
    out = []
    for key in keys:
        paper, _, node = key.partition(":")
        out.append({"paper_id": paper, "node_id": node, "evidence_id": key, "score": 0.5})
    return out


def _provenance() -> dict:
    return {
        "algorithm": "raptor",
        "algorithm_version": "raptor@0.0.1+vendored-pin",
        "build_params": {"umap_dim": 10, "threshold": 0.1},
        "corpus": {"name": "mixed-9", "digest": "sha256:aaa"},
        "generated_at": "2026-09-15T00:00:00Z",
        "generation": "gen-raptor-1",
        "manifest_fingerprint": "fp-raptor-1",
        "content_hash": "sha256:bbb",
        "members_hash": "sha256:ccc",
    }


def _raptor_build(*, provenance: dict | None = None) -> RaptorBuild:
    return RaptorBuild(
        storage_root="/nonexistent/raptor",
        generation="gen-raptor-1",
        manifest={"generation": "gen-raptor-1", "fingerprint": "fp-raptor-1"},
        provenance=_provenance() if provenance is None else provenance,
    )


class TestRegistry:
    def test_registered_algorithms_are_documented(self):
        assert set(BASELINES) == {
            "bm25_vector",
            "pageindex",
            "unified_tree_flat",
            "raptor_collapsed",
            "concat",
        }
        assert "no SQL LIKE" in BASELINES["pageindex"]
        assert "all-layer" in BASELINES["unified_tree_flat"]
        # finding 8: the unified-tree ablation may never carry the RAPTOR label
        assert "RAPTOR" not in BASELINES["unified_tree_flat"]
        assert "unified tree" in BASELINES["unified_tree_flat"]
        assert "independent RAPTOR build" in BASELINES["raptor_collapsed"]
        assert "provenance" in BASELINES["raptor_collapsed"]
        with pytest.raises(ValueError, match="unknown baseline"):
            baseline_algorithm("dense_only")


class TestAlgorithms:
    def test_concat_is_bm25_then_vector_deduplicated(self):
        calls: list[list[str]] = []

        def sql(legs, query, top_k):
            calls.append(legs)
            if legs == ["bm25"]:
                return _rows("p1:a", "p2:b")
            return _rows("p2:b", "p3:c")

        outcome = BaselineRunner(sql_search=sql).run("concat", "q", top_k=5)
        assert calls == [["bm25"], ["vector"]]
        assert outcome.keys == ("p1:a", "p2:b", "p3:c")
        assert outcome.status == "ok"
        assert outcome.details["bm25"] == 2 and outcome.details["vector"] == 2

    def test_concat_degrades_when_one_leg_is_unavailable(self):
        def sql(legs, query, top_k):
            if legs == ["vector"]:
                raise RuntimeError("no vector index")
            return _rows("p1:a")

        outcome = BaselineRunner(sql_search=sql).run("concat", "q", top_k=5)
        assert outcome.status == "ok" and outcome.keys == ("p1:a",)
        assert any("vector unavailable" in note for note in outcome.notes)

    def test_bm25_vector_fuses_both_legs_only(self):
        calls: list[list[str]] = []

        def sql(legs, query, top_k):
            calls.append(legs)
            return _rows("p1:a", "p1:a", "p2:b")

        outcome = BaselineRunner(sql_search=sql).run("bm25_vector", "q", top_k=2)
        assert calls == [["bm25", "vector"]]
        assert outcome.keys == ("p1:a", "p2:b")  # deduplicated, capped at top_k
        assert outcome.details["candidates"] == 3

    def test_unified_tree_flat_reads_all_layers_without_expansion(self):
        seen: dict = {}

        def tree(query, top_k):
            seen["top_k"] = top_k
            return [
                {"node_id": "nr-region", "local_id": "p1", "kind": "region", "score": 0.9},
                {"node_id": "nl-leaf", "local_id": "p2", "kind": "leaf", "score": 0.7},
            ]

        outcome = BaselineRunner(tree_search=tree).run("unified_tree_flat", "q", top_k=4)
        assert outcome.keys == ("p1:nr-region", "p2:nl-leaf")  # both layers kept
        assert outcome.details["view"] == "all"
        assert seen["top_k"] == 4
        assert outcome.algorithm == BASELINES["unified_tree_flat"]
        # finding 8: the ablation must say what it reads and what it is not
        assert "not an independent RAPTOR baseline" in " ".join(outcome.notes)

    def test_raptor_collapsed_never_reads_the_unified_tree(self):
        unified_calls: list[str] = []

        def tree(query, top_k):
            unified_calls.append(query)
            return [{"node_id": "nl-leaf", "local_id": "p1", "kind": "leaf", "score": 0.9}]

        outcome = BaselineRunner(tree_search=tree).run("raptor_collapsed", "q", top_k=3)
        assert unified_calls == []  # the unified-tree arm is not what the RAPTOR label scores
        assert outcome.status == "unavailable"
        assert outcome.keys == ()
        assert outcome.details["fail_closed"] is True
        assert "BaselineProvenanceError" in outcome.details["error"]

    def test_raptor_collapsed_runs_only_on_a_verified_build(self):
        calls: list[tuple[str, int]] = []

        def raptor(query, top_k):
            calls.append((query, top_k))
            return [{"node_id": "nl-1", "local_id": "p1", "kind": "leaf", "score": 0.9}]

        outcome = BaselineRunner(raptor_build=_raptor_build(), raptor_search=raptor).run(
            "raptor_collapsed", "q", top_k=5
        )
        assert calls == [("q", 5)]
        assert outcome.status == "ok" and outcome.keys == ("p1:nl-1",)
        assert outcome.algorithm == BASELINES["raptor_collapsed"]
        recorded = outcome.details["build"]
        assert outcome.details["generation"] == "gen-raptor-1"
        assert recorded["manifest_fingerprint"] == "fp-raptor-1"
        assert recorded["algorithm_version"] == "raptor@0.0.1+vendored-pin"
        assert recorded["provenance_fingerprint"]

    def test_a_build_without_provenance_is_refused_before_searching(self):
        calls: list[str] = []

        def raptor(query, top_k):
            calls.append(query)
            return [{"node_id": "nl-1", "local_id": "p1", "kind": "leaf", "score": 0.9}]

        outcome = BaselineRunner(
            raptor_build=_raptor_build(provenance={}), raptor_search=raptor
        ).run("raptor_collapsed", "q", top_k=3)
        assert calls == []  # fail-closed: no search happens without a verified build
        assert outcome.status == "unavailable" and outcome.details["fail_closed"] is True

    def test_pageindex_uses_the_tree_search_not_like(self):
        seen: dict = {}

        def pageindex(query, papers, top_k):
            seen["papers"] = list(papers)
            return [
                {"paper_id": "p1", "node_id": "0001", "score": 0.8},
                {"paper_id": "p2", "node_id": "0002", "score": 0.6},
            ]

        outcome = BaselineRunner(pageindex_search=pageindex).run(
            "pageindex", "q", top_k=3, papers=["p1", "p2"]
        )
        assert seen["papers"] == ["p1", "p2"]
        assert outcome.keys == ("p1:0001", "p2:0002")
        assert outcome.details["uses_sql_like"] is False

    def test_failures_are_reported_not_raised(self):
        def boom(legs, query, top_k):
            raise RuntimeError("index unavailable")

        outcome = BaselineRunner(sql_search=boom).run("bm25_vector", "q", top_k=3)
        assert outcome.status == "unavailable"
        assert "index unavailable" in outcome.details["error"]

    def test_empty_results_are_marked_empty(self):
        outcome = BaselineRunner(sql_search=lambda legs, q, k: []).run("bm25_vector", "q", top_k=3)
        assert outcome.status == "empty" and outcome.keys == ()


class _FakeRunner:
    def __init__(self, ranked: dict[str, list[str]]):
        self.ranked = ranked
        self.papers_seen: list = []

    def run(self, name, query, *, top_k=10, papers=None):
        from drbrain.rag.baselines import BaselineOutcome

        self.papers_seen.append(papers)
        keys = self.ranked.get(query, [])
        return BaselineOutcome(
            name=name,
            algorithm=baseline_algorithm(name),
            status="ok" if keys else "empty",
            keys=tuple(keys),
            details={"elapsed_ms": 7},
        )


class TestScoring:
    ENTRIES = [
        {
            "id": "q1",
            "query": "first",
            "relevant_papers": ["p1"],
            "relevant_nodes": ["nl-1"],
        },
        {
            "id": "q2",
            "query": "second",
            "relevant_papers": ["p2"],
            "relevant_nodes": ["nl-2"],
        },
    ]

    def test_hit_rate_and_mrr_at_paper_and_node_level(self):
        runner = _FakeRunner({"first": ["p1:nl-1", "p9:x"], "second": ["p9:x"]})
        result = evaluate_baseline("bm25_vector", None, None, self.ENTRIES, k=5, runner=runner)
        assert result.queries == 2
        assert result.hit_rate_paper == 0.5 and result.hit_rate_node == 0.5
        assert result.mrr_paper == 0.5 and result.mrr_node == 0.5  # rank 1 → 1.0, miss → 0
        assert result.per_query[0]["node_rank"] == 1
        assert result.cost["elapsed_ms"] == 14 and result.cost["production_route"] is False
        assert result.cost["usable"] is True and result.cost["unavailable_queries"] == 0

    def test_papers_for_is_forwarded(self):
        runner = _FakeRunner({"first": ["p1:nl-1"], "second": ["p2:nl-2"]})
        result = evaluate_baseline(
            "pageindex",
            None,
            None,
            self.ENTRIES,
            k=5,
            runner=runner,
            papers_for=lambda entry: entry["relevant_papers"],
        )
        assert runner.papers_seen == [["p1"], ["p2"]]
        assert result.hit_rate_node == 1.0

    def test_empty_split_is_noted(self):
        result = evaluate_baseline("concat", None, None, [], k=5)
        assert result.queries == 0 and result.notes == ("no golden entries",)

    def test_an_all_unavailable_run_is_not_a_usable_comparison(self):
        from drbrain.rag.baselines import BaselineOutcome

        class _Down:
            def run(self, name, query, *, top_k=10, papers=None):
                return BaselineOutcome(
                    name=name, algorithm=baseline_algorithm(name), status="unavailable", details={}
                )

        result = evaluate_baseline("concat", None, None, self.ENTRIES, k=5, runner=_Down())
        assert result.cost["usable"] is False
        assert result.cost["unavailable_queries"] == 2
        assert any("unavailable" in note for note in result.notes)


class TestCli:
    def _ctx(self, tmp_path):
        ctx = mock.MagicMock(spec=typer.Context)
        ctx.obj = {"config": {"db": {"path": str(tmp_path / "db.sqlite")}}}
        return ctx

    def test_cli_runs_the_named_baseline(self, tmp_path):
        from drbrain.cli import rag_commands

        payload = {
            "baseline": "concat",
            "algorithm": BASELINES["concat"],
            "queries": 3,
            "k": 10,
            "hit_rate_paper": 1.0,
            "hit_rate_node": 0.5,
            "mrr_paper": 0.8,
            "mrr_node": 0.4,
            "cost": {"elapsed_ms": 12},
            "per_query": [],
            "notes": [],
        }
        captured: list[str] = []
        with (
            mock.patch.object(rag_commands, "open_db", mock.MagicMock()),
            mock.patch("drbrain.rag.eval_data.load_golden", return_value=[{"query": "q"}]),
            mock.patch(
                "drbrain.rag.baselines.evaluate_baseline",
                return_value=mock.MagicMock(to_json=lambda: payload),
            ),
            mock.patch("typer.echo", side_effect=lambda m="", *a, **k: captured.append(str(m))),
        ):
            rag_commands.rag_baselines_cmd(
                self._ctx(tmp_path),
                name="concat",
                split="dev",
                k=10,
                out="",
                json_output=True,
            )
        report = json.loads(captured[0])
        assert report["baselines"]["concat"]["hit_rate_paper"] == 1.0

    def test_cli_rejects_unknown_names(self, tmp_path):
        from drbrain.cli import rag_commands

        with mock.patch.object(rag_commands, "open_db", mock.MagicMock()):
            with pytest.raises(typer.BadParameter):
                rag_commands.rag_baselines_cmd(
                    self._ctx(tmp_path),
                    name="dense_only",
                    split="dev",
                    k=10,
                    out="",
                    json_output=False,
                )
