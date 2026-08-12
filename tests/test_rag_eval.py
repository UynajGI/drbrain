"""T7 tests: evaluation layer (golden set, retriever metrics, RAGAS-style).

Covers :mod:`drbrain.rag.eval`:

* :func:`load_golden` — JSONL loading + split filtering
* :func:`build_golden_set` — idempotent generation, node/reference derivation
* :func:`_rank_metrics` / :func:`_aggregate_rank` — hand-computable hit/MRR
* :func:`run_retriever_eval` — aggregation over a mocked fusion retriever
* :func:`run_ragas_eval` — 4-metric pipeline with mocked LLM/fusion
* :func:`format_eval_report` — baseline markdown shape
* ``drbrain rag eval`` CLI — output structure + baseline file writing

All unit tests run offline (mocked retriever/LLM, no network, no GPU). One
live end-to-end test (real test-run papers + index + LLM) is marked
``integration`` and skipped by default.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from drbrain.config import Config, LlamaIndexConfig

_HAS_LLAMA_INDEX = importlib.util.find_spec("llama_index") is not None

if _HAS_LLAMA_INDEX:
    from llama_index.core.schema import NodeWithScore, TextNode

pytestmark = pytest.mark.skipif(not _HAS_LLAMA_INDEX, reason="llama_index not installed")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _nws(paper_id: str, node_id: str, text: str = "some section body") -> NodeWithScore:
    """A vector-leg style fused node (paper_id/node_id metadata present)."""
    node = TextNode(
        text=text,
        id_=f"{paper_id}:{node_id}",
        metadata={"paper_id": paper_id, "node_id": node_id, "title": "Section"},
    )
    return NodeWithScore(node=node, score=0.5)


def _golden_cfg(tmp_path: Path, enabled: bool = True) -> dict:
    return {
        "db": {"path": str(tmp_path / "t.db")},
        "llm": {"models": [{"provider": "openai", "model": "gpt-4o", "api_key": "x"}]},
        "dirs": {"papers": str(tmp_path), "cache": str(tmp_path), "reports": str(tmp_path)},
        "api": {"cache_ttl": 0},
        "llamaindex": {
            "enabled": enabled,
            "eval": {
                "golden_set": str(tmp_path / "golden.jsonl"),
                "split": ["dev", "val", "test"],
            },
        },
    }


def _write_golden(tmp_path: Path, items: list[dict]) -> Path:
    path = tmp_path / "golden.jsonl"
    path.write_text(
        "\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n", encoding="utf-8"
    )
    return path


class _FakeFusion:
    """Minimal retriever stand-in: query → fixed node list."""

    def __init__(self, mapping: dict) -> None:
        self._mapping = mapping

    def retrieve(self, query: str):
        return list(self._mapping.get(query, []))


# ── load_golden ──────────────────────────────────────────────────────────────


def test_load_golden_filters_by_split(tmp_path):
    golden = [
        {"query": "q1", "relevant_papers": ["p1"], "relevant_nodes": [], "split": "dev"},
        {"query": "q2", "relevant_papers": ["p2"], "relevant_nodes": [], "split": "val"},
        {"query": "q3", "relevant_papers": ["p3"], "relevant_nodes": [], "split": "test"},
    ]
    _write_golden(tmp_path, golden)
    cfg = _golden_cfg(tmp_path)

    from drbrain.rag.eval import load_golden

    assert [g["query"] for g in load_golden(cfg, split="dev")] == ["q1"]
    assert [g["query"] for g in load_golden(cfg, split="val")] == ["q2"]
    assert [g["query"] for g in load_golden(cfg, split=None)] == ["q1", "q2", "q3"]
    assert load_golden(cfg, split="nope") == []


def test_load_golden_missing_file_returns_empty(tmp_path):
    from drbrain.rag.eval import load_golden

    assert load_golden(_golden_cfg(tmp_path), split="dev") == []


def test_load_golden_skips_malformed_lines(tmp_path):
    path = _write_golden(
        tmp_path,
        [{"query": "q1", "relevant_papers": ["p1"], "relevant_nodes": [], "split": "dev"}],
    )
    path.write_text('{"query": "ok", "split": "dev"}\nnot-json\n\n', encoding="utf-8")

    from drbrain.rag.eval import load_golden

    items = load_golden(_golden_cfg(tmp_path), split="dev")
    assert [g["query"] for g in items] == ["ok"]


# ── rank metrics (hand-computable) ───────────────────────────────────────────


def test_rank_metrics_hand_computed():
    from drbrain.rag.eval import _rank_metrics

    item = {
        "query": "q",
        "relevant_papers": ["p1", "p2"],
        "relevant_nodes": [
            {"paper_id": "p1", "node_id": "n1"},
            {"paper_id": "p2", "node_id": "n9"},
        ],
    }
    # rank 1 = p3 (irrelevant), rank 2 = p1, rank 3 = p2:n9
    nodes = [_nws("p3", "n1"), _nws("p1", "n2"), _nws("p2", "n9")]
    res = _rank_metrics(nodes, item, ks=[1, 5, 10])

    paper = res["paper"]
    assert paper["first_rank"] == 2
    assert paper["hit_rate"] == {"1": False, "5": True, "10": True}
    assert paper["mrr"] == {"1": 0.0, "5": 0.5, "10": 0.5}

    node = res["node"]
    assert node["first_rank"] == 3
    assert node["hit_rate"] == {"1": False, "5": True, "10": True}
    assert node["mrr"] == {"1": 0.0, "5": round(1 / 3, 6), "10": round(1 / 3, 6)}


def test_rank_metrics_no_relevant_node_keeps_paper_level(tmp_path):
    from drbrain.rag.eval import _rank_metrics

    item = {
        "query": "q",
        "relevant_papers": ["p1"],
        "relevant_nodes": [],  # node-level annotation absent → paper-level only
    }
    res = _rank_metrics([_nws("p1", "x")], item, ks=[5])
    assert res["paper"]["hit_rate"] == {"5": True}
    assert res["node"]["first_rank"] is None
    assert res["node"]["hit_rate"] == {"5": False}


def test_rank_metrics_no_hit():
    from drbrain.rag.eval import _rank_metrics

    item = {"query": "q", "relevant_papers": ["p9"], "relevant_nodes": []}
    res = _rank_metrics([_nws("p1", "n1"), _nws("p2", "n2")], item, ks=[5, 10])
    assert res["paper"]["first_rank"] is None
    assert res["paper"]["hit_rate"] == {"5": False, "10": False}
    assert res["paper"]["mrr"] == {"5": 0.0, "10": 0.0}


def test_aggregate_rank_means():
    from drbrain.rag.eval import _aggregate_rank

    rows = [
        {
            "query": "a",
            "paper": {"hit_rate": {"5": True}, "mrr": {"5": 1.0}, "first_rank": 1},
            "node": {"hit_rate": {"5": True}, "mrr": {"5": 1.0}, "first_rank": 1},
        },
        {
            "query": "b",
            "paper": {"hit_rate": {"5": False}, "mrr": {"5": 0.0}, "first_rank": None},
            "node": {"hit_rate": {"5": False}, "mrr": {"5": 0.0}, "first_rank": None},
        },
    ]
    agg = _aggregate_rank(rows)
    assert agg["queries"] == 2
    assert agg["hit_rate"]["paper"] == {"5": 0.5}
    assert agg["mrr"]["paper"] == {"5": 0.5}
    assert agg["hit_rate"]["node"] == {"5": 0.5}


# ── run_retriever_eval ───────────────────────────────────────────────────────


def test_run_retriever_eval_aggregates(monkeypatch, tmp_path):
    from drbrain.rag.eval import run_retriever_eval

    golden = [
        {
            "query": "q1",
            "relevant_papers": ["p1"],
            "relevant_nodes": [{"paper_id": "p1", "node_id": "n1"}],
            "split": "dev",
        },
        {"query": "q2", "relevant_papers": ["p9"], "relevant_nodes": [], "split": "dev"},
    ]
    _write_golden(tmp_path, golden)
    fusion = _FakeFusion({"q1": [_nws("p1", "n1")], "q2": [_nws("p2", "n1"), _nws("p3", "n2")]})
    monkeypatch.setattr(
        "drbrain.rag.engine.build_hybrid_retriever", lambda cfg, db, top_k=None: fusion
    )

    res = run_retriever_eval(_golden_cfg(tmp_path), db=None, split="dev", ks=[5, 10])
    assert res["status"] == "ok"
    assert res["split"] == "dev"
    assert res["queries"] == 2
    # q1 hits at rank 1; q2 misses entirely → paper means = 0.5
    assert res["hit_rate"]["paper"] == {"5": 0.5, "10": 0.5}
    assert res["mrr"]["paper"] == {"5": 0.5, "10": 0.5}
    assert len(res["per_query"]) == 2
    assert res["per_query"][0]["paper"]["first_rank"] == 1


def test_run_retriever_eval_empty_split(monkeypatch, tmp_path):
    from drbrain.rag.eval import run_retriever_eval

    _write_golden(
        tmp_path, [{"query": "q1", "relevant_papers": [], "relevant_nodes": [], "split": "test"}]
    )
    res = run_retriever_eval(_golden_cfg(tmp_path), db=None, split="dev", ks=[5])
    assert res["status"] == "empty"
    assert res["queries"] == 0


def test_run_retriever_eval_unavailable_no_index(monkeypatch, tmp_path):
    from drbrain.rag.eval import run_retriever_eval

    _write_golden(
        tmp_path, [{"query": "q1", "relevant_papers": ["p1"], "relevant_nodes": [], "split": "dev"}]
    )
    monkeypatch.setattr(
        "drbrain.rag.engine.build_hybrid_retriever", lambda cfg, db, top_k=None: None
    )
    res = run_retriever_eval(_golden_cfg(tmp_path), db=None, split="dev", ks=[5])
    assert res["status"] == "unavailable"
    assert "reason" in res


def test_run_retriever_eval_max_queries(monkeypatch, tmp_path):
    from drbrain.rag.eval import run_retriever_eval

    golden = [
        {"query": "q1", "relevant_papers": ["p1"], "relevant_nodes": [], "split": "dev"},
        {"query": "q2", "relevant_papers": ["p2"], "relevant_nodes": [], "split": "dev"},
    ]
    _write_golden(tmp_path, golden)
    fusion = _FakeFusion({"q1": [_nws("p1", "n1")], "q2": [_nws("p2", "n1")]})
    monkeypatch.setattr(
        "drbrain.rag.engine.build_hybrid_retriever", lambda cfg, db, top_k=None: fusion
    )
    res = run_retriever_eval(_golden_cfg(tmp_path), db=None, split="dev", ks=[5], max_queries=1)
    assert res["queries"] == 1


# ── build_golden_set ─────────────────────────────────────────────────────────


def _fake_papers(tmp_path: Path) -> Path:
    """Two mini papers: p1 with an ABSTRACT+Introduction tree, p2 back-matter only."""
    root = tmp_path / "papers"
    for pid in ("p1", "p2"):
        (root / pid).mkdir(parents=True, exist_ok=True)
    (root / "p1" / "tree.json").write_text(
        json.dumps(
            {
                "structure": [
                    {"node_id": "abs", "title": "ABSTRACT", "line_num": 1},
                    {"node_id": "intro", "title": "Introduction", "line_num": 3},
                    {"node_id": "back", "title": "Acknowledgements", "line_num": 8},
                ]
            }
        ),
        encoding="utf-8",
    )
    (root / "p1" / "raw.md").write_text(
        "Perovskite stability abstract paragraph with enough words to be meaningful.\n\n"
        "Introduction body text.\n\n"
        "Thanks.\n",
        encoding="utf-8",
    )
    (root / "p2" / "tree.json").write_text(
        json.dumps(
            {"structure": [{"node_id": "0000", "title": "Acknowledgements", "line_num": 1}]}
        ),
        encoding="utf-8",
    )
    (root / "p2" / "raw.md").write_text(
        "Zinc anode abstract text of sufficient length here.\n", encoding="utf-8"
    )
    return root


def test_build_golden_set_ok_and_idempotent(monkeypatch, tmp_path):
    from drbrain.rag.eval import build_golden_set

    monkeypatch.setattr(
        "drbrain.rag.eval._GOLDEN_QUERIES",
        [
            {"id": "q1", "q": "Question about perovskite stability?", "papers": ["p1"]},
            {"id": "q2", "q": "Question about zinc anodes?", "papers": ["p2"]},
        ],
    )
    papers = _fake_papers(tmp_path)
    cfg = _golden_cfg(tmp_path)

    stats = build_golden_set(cfg, papers_dir=papers, out_path=tmp_path / "golden.jsonl")
    assert stats["status"] == "ok"
    assert stats["queries"] == 2
    assert stats["papers"] == ["p1", "p2"]
    assert stats["missing_papers"] == []

    # Second call is idempotent — file untouched.
    before = (tmp_path / "golden.jsonl").read_text(encoding="utf-8")
    stats2 = build_golden_set(cfg, papers_dir=papers, out_path=tmp_path / "golden.jsonl")
    assert stats2["status"] == "exists"
    assert (tmp_path / "golden.jsonl").read_text(encoding="utf-8") == before

    # Content: node annotation (content nodes only for p1) + reference answer.
    item = json.loads((tmp_path / "golden.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert item["relevant_papers"] == ["p1"]
    node_ids = {(n["paper_id"], n["node_id"]) for n in item["relevant_nodes"]}
    assert ("p1", "abs") in node_ids
    assert ("p1", "intro") in node_ids
    assert ("p1", "back") not in node_ids  # back-matter excluded when content exists
    assert "perovskite" in item["reference_answer"].lower()
    assert item["split"] in ("dev", "val", "test")


def test_build_golden_set_missing_papers_skipped(monkeypatch, tmp_path):
    from drbrain.rag.eval import build_golden_set

    monkeypatch.setattr(
        "drbrain.rag.eval._GOLDEN_QUERIES",
        [{"id": "q1", "q": "Q?", "papers": ["p1", "p-ghost"]}],
    )
    papers = _fake_papers(tmp_path)
    stats = build_golden_set(
        _golden_cfg(tmp_path), papers_dir=papers, out_path=tmp_path / "golden.jsonl"
    )
    assert stats["status"] == "ok"
    assert stats["missing_papers"] == ["p-ghost"]
    item = json.loads((tmp_path / "golden.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert item["relevant_papers"] == ["p1"]


def test_build_golden_set_query_ids_subset(monkeypatch, tmp_path):
    from drbrain.rag.eval import build_golden_set

    monkeypatch.setattr(
        "drbrain.rag.eval._GOLDEN_QUERIES",
        [
            {"id": "q1", "q": "Q1?", "papers": ["p1"]},
            {"id": "q2", "q": "Q2?", "papers": ["p2"]},
        ],
    )
    papers = _fake_papers(tmp_path)
    stats = build_golden_set(
        _golden_cfg(tmp_path),
        papers_dir=papers,
        out_path=tmp_path / "golden.jsonl",
        query_ids=["q2"],
    )
    assert stats["status"] == "ok"
    assert stats["queries"] == 1
    content = (tmp_path / "golden.jsonl").read_text(encoding="utf-8")
    assert "Q2?" in content  # the q2 entry was built…
    assert "Q1?" not in content  # …and q1 was excluded


def test_split_assignment_covers_all_splits_and_is_deterministic():
    from drbrain.rag.eval import _GOLDEN_QUERIES, _assign_splits

    assert 30 <= len(_GOLDEN_QUERIES) <= 50, "golden set must stay within the 30-50 ticket range"
    split_of = _assign_splits(_GOLDEN_QUERIES)
    counts = {}
    for entry in _GOLDEN_QUERIES:
        counts[split_of[entry["id"]]] = counts.get(split_of[entry["id"]], 0) + 1
    assert set(counts) == {"dev", "val", "test"}
    # 60/20/20 within tolerance (rounding): dev ≥ 50%, val/test ≥ 10% each.
    n = len(_GOLDEN_QUERIES)
    assert counts["dev"] / n >= 0.5
    assert counts["val"] / n >= 0.1
    assert counts["test"] / n >= 0.1
    # Deterministic across calls.
    assert _assign_splits(_GOLDEN_QUERIES) == split_of


# ── run_ragas_eval (self-written 4 metrics) ──────────────────────────────────


def test_run_ragas_eval_pipeline(monkeypatch, tmp_path):
    from drbrain.rag.eval import run_ragas_eval

    golden = [
        {
            "query": "q1",
            "relevant_papers": ["p1"],
            "relevant_nodes": [],
            "split": "val",
            "reference_answer": "ref one",
        },
        {
            "query": "q2",
            "relevant_papers": ["p2"],
            "relevant_nodes": [],
            "split": "val",
            "reference_answer": "ref two",
        },
    ]
    _write_golden(tmp_path, golden)
    fusion = _FakeFusion({"q1": [_nws("p1", "n1", "context text one")], "q2": []})

    monkeypatch.setattr(
        "drbrain.rag.engine.ask_llamaindex",
        lambda cfg, db, question, top_k=5, streaming=False: {
            "answer": f"answer for {question}",
            "sources": [],
            "engine": "llamaindex",
        },
    )
    monkeypatch.setattr(
        "drbrain.rag.engine.build_hybrid_retriever", lambda cfg, db, top_k=None: fusion
    )
    monkeypatch.setattr(
        "drbrain.rag.llm.DrbrainLLM",
        lambda cfg: type(
            "_FakeLLM",
            (),
            {
                "complete": lambda self, prompt, max_tokens=None: type(
                    "R", (), {"text": "SCORE: 0.8"}
                )()
            },
        )(),
    )

    res = run_ragas_eval(_golden_cfg(tmp_path), db=None, split="val", n=10)
    assert res["status"] == "ok"
    assert res["queries"] == 2
    for key in ("faithfulness", "answer_relevancy", "context_precision", "answer_correctness"):
        assert key in res["metrics"]
        assert res["metrics"][key]["mean"] == 0.8
        assert res["metrics"][key]["missing"] == 0
    assert res["per_query"][0]["answer_len"] > 0
    assert res["per_query"][0]["context_nodes"] == 1


def test_run_ragas_eval_correctness_missing_without_reference(monkeypatch, tmp_path):
    from drbrain.rag.eval import run_ragas_eval

    _write_golden(
        tmp_path,
        [{"query": "q1", "relevant_papers": ["p1"], "relevant_nodes": [], "split": "val"}],
    )
    fusion = _FakeFusion({"q1": [_nws("p1", "n1")]})
    monkeypatch.setattr(
        "drbrain.rag.engine.ask_llamaindex",
        lambda cfg, db, question, top_k=5, streaming=False: {
            "answer": "a",
            "sources": [],
            "engine": "llamaindex",
        },
    )
    monkeypatch.setattr(
        "drbrain.rag.engine.build_hybrid_retriever", lambda cfg, db, top_k=None: fusion
    )
    monkeypatch.setattr(
        "drbrain.rag.llm.DrbrainLLM",
        lambda cfg: type(
            "_FakeLLM",
            (),
            {
                "complete": lambda self, prompt, max_tokens=None: type(
                    "R", (), {"text": "SCORE: 0.9"}
                )()
            },
        )(),
    )
    res = run_ragas_eval(_golden_cfg(tmp_path), db=None, split="val", n=10)
    assert res["metrics"]["answer_correctness"]["mean"] is None
    assert res["metrics"]["answer_correctness"]["missing"] == 1


def test_run_ragas_eval_empty_split(tmp_path):
    from drbrain.rag.eval import run_ragas_eval

    _write_golden(
        tmp_path, [{"query": "q1", "relevant_papers": [], "relevant_nodes": [], "split": "dev"}]
    )
    res = run_ragas_eval(_golden_cfg(tmp_path), db=None, split="val", n=10)
    assert res["status"] == "empty"


def test_run_ragas_eval_unavailable_without_llama_index(monkeypatch, tmp_path):
    from drbrain.rag.eval import run_ragas_eval

    _write_golden(
        tmp_path, [{"query": "q1", "relevant_papers": ["p1"], "relevant_nodes": [], "split": "val"}]
    )
    monkeypatch.setattr("drbrain.rag.eval._LLAMA_INDEX_AVAILABLE", False)
    res = run_ragas_eval(_golden_cfg(tmp_path), db=None, split="val", n=10)
    assert res["status"] == "unavailable"
    assert "llama-index" in res["reason"]


# ── score parsing / context assembly ─────────────────────────────────────────


def test_reference_paragraph_skips_author_lines():
    """The raw.md heuristic must pick the abstract, not the author line.

    Layout without an ABSTRACT node (e.g. 10.1002_adma.202308655): title,
    author line, abstract paragraph, then ``1. Introduction``.
    """
    from drbrain.rag.eval import _reference_paragraph

    raw = (
        "High-Stable Lead-Free Solar Cells\n\n"
        "Feng Yang, Rui Zhu,* Zuhong Zhang, Weiwei Zuo, Bingchen He, Mahmoud Hussein Aldamasy\n\n"
        "Tin halide perovskites are an appealing alternative to lead perovskites. "
        "However, owing to the lower redox potential of Sn(II)/Sn(IV), particularly under "
        "the presence of oxygen and water, the accumulation of Sn(IV) at the surface layer "
        "will negatively impact the device's performance and stability. To this end, this "
        "work introduced a novel multifunctional molecule to form a protective layer on the "
        "surface of Sn-based perovskite films.\n\n"
        "1. Introduction\n\nPerovskite solar cells (PSCs) certified power conversion "
        "efficiency of 26.1%."
    )
    ref = _reference_paragraph(raw)
    assert "Tin halide perovskites" in ref
    assert "Feng Yang" not in ref
    assert "Introduction" not in ref


def test_reference_paragraph_abstract_heading_layout():
    """``ABSTRACT``-headed layout: the text follows the heading."""
    from drbrain.rag.eval import _reference_paragraph

    raw = (
        "Zinc anodes for batteries\n\n"
        "AUTHOR LINE ONE, AUTHOR LINE TWO\n\n"
        "ABSTRACT\n\n"
        "Three-dimensional Zn metal anodes are promising for battery applications "
        "because of their high theoretical capacity and low cost, yet dendrite growth "
        "remains a challenge that recent developments in electrode architecture and "
        "interfacial engineering aim to solve.\n\n"
        "1. Introduction\n\nBody text here."
    )
    ref = _reference_paragraph(raw)
    assert "Zn metal anodes" in ref
    assert "ABSTRACT" not in ref


def test_reference_paragraph_roman_numeral_heading():
    """arXiv layout: ``I. INTRODUCTION`` must bound the abstract, too."""
    from drbrain.rag.eval import _reference_paragraph

    raw = (
        "Sputtered NbN films for single-photon detectors\n\n"
        "Ilya A. Stepanov,* Aleksandr S. Baburin,* and Ilya A. Rodionov\n\n"
        "Nowadays ultrahigh performance superconducting nanowire single-photon "
        "detectors are key elements in a variety of devices from biological research "
        "to quantum communications. Accurate tuning of superconducting material "
        "properties is a powerful resource for fabricating detectors with the desired "
        "properties.\n\n"
        "I. INTRODUCTION\n\n"
        "Superconducting nanowire single-photon detectors (SNSPDs) have become the "
        "technology of choice for single-photon counting in the near infrared."
    )
    ref = _reference_paragraph(raw)
    assert "superconducting nanowire single-photon detectors" in ref
    assert "Ilya" not in ref
    assert "I. INTRODUCTION" not in ref


def test_parse_score_variants():
    from drbrain.rag.eval import _parse_score

    assert _parse_score("SCORE: 0.75") == 0.75
    assert _parse_score("The answer is good. SCORE = 1.0") == 1.0
    assert _parse_score("reason... SCORE: 0.42") == 0.42
    assert _parse_score("SCORE: 0.75\nmore text") == 0.75
    assert _parse_score("0.9") == 0.9
    assert _parse_score("nonsense") is None
    assert _parse_score(None) is None
    # Out-of-range values are clamped.
    assert _parse_score("SCORE: 1.7") == 1.0


def test_context_for_truncates_and_skips_empty(tmp_path):
    from drbrain.rag.eval import _CONTEXT_CHUNK_MAX_CHARS, _context_for

    long_text = "x" * (_CONTEXT_CHUNK_MAX_CHARS + 100)
    nodes = [_nws("p1", "n1", long_text), _nws("p2", "n2", ""), _nws("p3", "n3", "short")]
    ctx = _context_for(nodes, limit=3)
    assert len(ctx) <= _CONTEXT_CHUNK_MAX_CHARS + 100  # long chunk truncated + separators
    assert "short" in ctx
    assert _context_for([], limit=3) == ""


# ── format_eval_report / CLI ─────────────────────────────────────────────────


def test_format_eval_report_shapes():
    from drbrain.rag.eval import format_eval_report

    retriever = {
        "status": "ok",
        "split": "dev",
        "queries": 2,
        "ks": [5, 10],
        "hit_rate": {"paper": {"5": 0.5, "10": 1.0}, "node": {"5": 0.0, "10": 0.5}},
        "mrr": {"paper": {"5": 0.25, "10": 0.3}, "node": {"5": 0.0, "10": 0.1}},
    }
    ragas = {
        "status": "ok",
        "split": "val",
        "queries": 1,
        "metrics": {"faithfulness": {"mean": 0.8, "missing": 0}},
    }
    report = format_eval_report({"llamaindex": {"enabled": True}}, retriever=retriever, ragas=ragas)
    assert "## LlamaIndex RAG 评估基线" in report
    assert "| paper | hit_rate | 0.5 | 1.0 |" in report
    assert "| faithfulness | 0.8 | 0 |" in report
    assert report.count("## LlamaIndex RAG 评估基线") == 1


def test_eval_cmd_structure_and_baseline_file(monkeypatch, tmp_path, capsys):
    from unittest import mock

    from drbrain.cli.rag_commands import rag_eval_cmd

    retriever_result = {
        "status": "ok",
        "split": "dev",
        "queries": 2,
        "ks": [5, 10],
        "hit_rate": {"paper": {"5": 0.5, "10": 1.0}, "node": {"5": 0.0, "10": 0.5}},
        "mrr": {"paper": {"5": 0.25, "10": 0.3}, "node": {"5": 0.0, "10": 0.1}},
        "per_query": [],
    }
    ragas_result = {
        "status": "ok",
        "split": "dev",
        "queries": 1,
        "metrics": {"faithfulness": {"mean": 0.8, "missing": 0}},
        "per_query": [],
    }
    monkeypatch.setattr("drbrain.rag.eval.run_retriever_eval", lambda *a, **k: retriever_result)
    monkeypatch.setattr("drbrain.rag.eval.run_ragas_eval", lambda *a, **k: ragas_result)

    ctx = mock.MagicMock(spec=__import__("typer").Context)
    ctx.obj = {"config": _golden_cfg(tmp_path)}
    out_file = tmp_path / "baseline.md"

    rag_eval_cmd(
        ctx, split="dev", metrics="all", k="5,10", n=2, json_output=True, out=str(out_file)
    )
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["split"] == "dev"
    assert payload["metrics"] == "all"
    assert payload["retriever"]["status"] == "ok"
    assert payload["ragas"]["queries"] == 1
    assert payload["retriever"]["hit_rate"]["paper"]["5"] == 0.5

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "## LlamaIndex RAG 评估基线" in content
    assert "Retriever eval" in content
    assert "RAGAS-style eval" in content


def test_eval_cmd_validation_errors(tmp_path, capsys):
    from unittest import mock

    import typer

    from drbrain.cli.rag_commands import rag_eval_cmd

    ctx = mock.MagicMock(spec=__import__("typer").Context)
    ctx.obj = {"config": _golden_cfg(tmp_path)}
    out_file = tmp_path / "baseline.md"

    with pytest.raises(typer.Exit) as exc:
        rag_eval_cmd(
            ctx,
            split="nope",
            metrics="retriever",
            k="5,10",
            n=2,
            json_output=True,
            out=str(out_file),
        )
    assert exc.value.exit_code == 2
    assert "split" in capsys.readouterr().err

    with pytest.raises(typer.Exit) as exc:
        rag_eval_cmd(
            ctx, split="dev", metrics="bogus", k="5,10", n=2, json_output=True, out=str(out_file)
        )
    assert exc.value.exit_code == 2
    assert "metrics" in capsys.readouterr().err

    with pytest.raises(typer.Exit) as exc:
        rag_eval_cmd(
            ctx, split="dev", metrics="retriever", k="abc", n=2, json_output=True, out=str(out_file)
        )
    assert exc.value.exit_code == 2
    assert "k" in capsys.readouterr().err


# ── integration: real test-run corpus + index + real LLM ────────────────────


@pytest.mark.integration
def test_integration_eval_real_corpus(tmp_path, monkeypatch):
    """End-to-end: small golden subset → real index → retriever + ragas evals.

    Uses the opencode.ai ``deepseek-v4-flash`` test key from ``test-run/``
    (never hardcoded) and the real Qwen embedding model. Skipped unless the
    test-run config exists.
    """
    test_cfg_path = Path(__file__).resolve().parents[1] / "test-run" / "config.yaml"
    if not test_cfg_path.exists():
        pytest.skip("test-run/config.yaml (opencode test key) not present")
    cfg = Config.from_yaml(
        str(test_cfg_path), local_path=test_cfg_path.parent / "config.local.yaml"
    )
    assert cfg.llm.models, "test-run config must define llm.models"
    base = test_cfg_path.parent
    cfg.dirs.papers = str(base / cfg.dirs.papers)
    cfg.dirs.cache = str(tmp_path)
    cfg.llamaindex = LlamaIndexConfig(
        enabled=True,
        streaming=False,
        storage_dir=str(tmp_path / "llamaindex"),
        similarity_cutoff=None,
        retrievers=["bm25", "vector"],
    )
    # Big nodes (up to ~39 KB in this corpus) OOM a 16 GB V100 in fp32 —
    # CPU embedding is the T3-sanctioned mitigation (slower, but safe).
    cfg.embed.device = "cpu"

    from drbrain.rag.eval import build_golden_set, run_ragas_eval, run_retriever_eval

    # 6 curated queries across 6 distinct papers (keeps embedding cost low).
    query_ids = [
        "pv-tin-surface-reconstruction",
        "pv-ws2-invisible-solar-cell",
        "cat-methylene-blue-papaya",
        "2d-graphene-edge-pinning",
        "metal-fecral-annealing",
        "pol-flame-retardant-pva",
    ]
    stats = build_golden_set(
        cfg, papers_dir=cfg.dirs.papers, out_path=tmp_path / "golden.jsonl", query_ids=query_ids
    )
    assert stats["status"] == "ok", stats
    assert stats["missing_papers"] == []
    cfg.llamaindex.eval.golden_set = str(tmp_path / "golden.jsonl")

    # Build the real index for exactly the relevant papers.
    from drbrain.rag.indexer import build_index

    class _PaperDB:
        def get_all_papers(self):
            return [{"local_id": p} for p in stats["papers"]]

    index_stats = build_index(cfg, _PaperDB(), paper_ids=stats["papers"])
    assert index_stats["nodes"] >= 1, index_stats

    class _EvalDB:
        """No-op db — the bm25/vector fusion legs never touch it."""

    retriever = run_retriever_eval(cfg, _EvalDB(), split="dev", ks=[5, 10])
    assert retriever["status"] == "ok", retriever
    assert retriever["queries"] >= 1
    assert set(retriever["hit_rate"]) == {"paper", "node"}
    assert set(retriever["mrr"]) == {"paper", "node"}

    ragas = run_ragas_eval(cfg, _EvalDB(), split="val", n=1)
    assert ragas["status"] == "ok", ragas
    assert ragas["queries"] >= 1
    for key in ("faithfulness", "answer_relevancy", "context_precision", "answer_correctness"):
        assert key in ragas["metrics"], ragas
