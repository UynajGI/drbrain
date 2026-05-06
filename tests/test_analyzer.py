"""Tests for report/analyzer.py — knowledge frontier analysis orchestrator."""

from unittest.mock import MagicMock, patch

from drbrain.report.analyzer import analyze_paper

# -- helpers ----------------------------------------------------------------


def _fake_paper(local_id="paper-1", title="Test Paper", year=2024):
    return {"local_id": local_id, "title": title, "year": year}


def _fake_concept(label, type_="Method", confidence=0.9):
    return {"label": label, "type": type_, "confidence": confidence}


def _fake_arg(claim, claim_type, target_label, target_type, mechanism="", section=""):
    return {
        "claim": claim,
        "claim_type": claim_type,
        "target_label": target_label,
        "target_type": target_type,
        "mechanism": mechanism,
        "section": section,
        "confidence": 0.85,
        "evidence_type": "",
        "evidence_detail": "",
    }


def _seed(node):
    return {"concept": node, "type": "stale_problem"}


# -- 1. basic paper with concepts -------------------------------------------


def test_analyze_paper_basic():
    """Paper with concepts → paper/title/year in result, seeds filtered."""
    db = MagicMock()
    db.get_paper.return_value = _fake_paper("p1", "Gradient Descent Revisited", 2023)
    db.get_concepts_by_paper.return_value = [
        _fake_concept("Gradient Descent", type_="Method"),
        _fake_concept("SGD", type_="Method"),
        _fake_concept("Overfitting", type_="Problem"),
    ]
    db.get_arguments_by_paper.return_value = []

    graph = MagicMock()
    graph.detect_research_seeds.return_value = [
        _seed("Gradient Descent"),
        _seed("Overfitting"),
        _seed("Unrelated Concept"),
    ]
    graph.closure.return_value = []

    result = analyze_paper(db, graph, "p1")

    assert "error" not in result
    assert result["paper"] == {
        "local_id": "p1",
        "title": "Gradient Descent Revisited",
        "year": 2023,
    }
    # Seeds from detect_research_seeds are included directly (no paper-concept filter)
    assert len(result["seeds"]) == 3
    assert result["seeds"][0]["concept"] == "Gradient Descent"
    assert result["seeds"][1]["concept"] == "Overfitting"
    # Chains empty (no arguments)
    assert result["causal_chains"] == []
    # Summary counts match actual lengths
    assert result["summary"]["seeds"] == 3
    assert result["summary"]["causal_chains"] == 0
    assert result["summary"]["inferred_edges"] == 0


# -- 2. nonexistent paper ---------------------------------------------------


def test_analyze_paper_nonexistent():
    """Nonexistent paper returns error dict."""
    db = MagicMock()
    db.get_paper.return_value = None
    graph = MagicMock()

    result = analyze_paper(db, graph, "nonexistent-id")

    assert "error" in result
    assert result["error"] == "Paper not found: nonexistent-id"


# -- 3. full=False — no deep sections ---------------------------------------


def test_analyze_paper_full_false():
    """full=False omits critical_nodes, hypotheses, and isomorphisms."""
    db = MagicMock()
    db.get_paper.return_value = _fake_paper("p2", "Deep Learning Basics", 2022)
    db.get_concepts_by_paper.return_value = [_fake_concept("Deep Learning")]
    db.get_arguments_by_paper.return_value = []

    graph = MagicMock()
    graph.detect_research_seeds.return_value = [_seed("Deep Learning")]
    graph.closure.return_value = [{"src": "X", "dst": "Y", "relation": "gap_addressed"}]

    result = analyze_paper(db, graph, "p2", full=False)

    assert "critical_nodes" not in result
    assert "hypotheses" not in result
    assert "isomorphisms" not in result
    assert result["summary"]["inferred_edges"] == 1


# -- 4. full=True — all sections present ------------------------------------


def test_analyze_paper_full_true():
    """full=True populates critical_nodes, hypotheses, and isomorphisms."""
    db = MagicMock()
    db.get_paper.return_value = _fake_paper("p3", "Full Analysis Paper", 2025)
    db.get_concepts_by_paper.return_value = [
        _fake_concept("Transformer"),
        _fake_concept("Attention"),
    ]
    db.get_arguments_by_paper.return_value = []

    graph = MagicMock()
    graph.detect_research_seeds.return_value = []
    graph.closure.return_value = []

    # Mock objects for the full-mode modules.  CausalChain lacks .source / .target
    # and IsomorphicMapping uses source_domain / target_domain, so we supply
    # simple namespace mocks that provide the attributes the analyzer expects.

    mock_chain = MagicMock()
    mock_chain.source = "Transformer"
    mock_chain.target = "Attention"
    mock_chain.mechanism = "scaled dot-product"

    mock_hypothesis = MagicMock()
    mock_hypothesis.description = "H1: new architecture"
    mock_hypothesis.type = "gap_filling"
    mock_hypothesis.base_confidence = 0.88

    mock_iso = MagicMock()
    mock_iso.source = "NLP"
    mock_iso.target = "CV"
    mock_iso.similarity = 0.72

    with (
        patch("drbrain.extractor.causal_chain.find_chains_from", return_value=[mock_chain]),
        patch(
            "drbrain.extractor.counterfactual.find_critical_nodes",
            return_value=[{"node": "Transformer", "impact": 0.5}],
        ),
        patch("drbrain.extractor.hypothesis.generate_hypotheses", return_value=[mock_hypothesis]),
        patch("drbrain.extractor.isomorphism.find_isomorphic_patterns", return_value=[mock_iso]),
    ):
        result = analyze_paper(db, graph, "p3", full=True)

    assert "critical_nodes" in result
    assert "hypotheses" in result
    assert "isomorphisms" in result

    assert result["critical_nodes"] == [{"node": "Transformer", "impact": 0.5}]
    assert len(result["hypotheses"]) == 1
    assert result["hypotheses"][0]["description"] == "H1: new architecture"
    assert result["hypotheses"][0]["type"] == "gap_filling"
    assert result["hypotheses"][0]["confidence"] == 0.88
    assert len(result["isomorphisms"]) == 1
    assert result["isomorphisms"][0]["source"] == "NLP"
    assert result["isomorphisms"][0]["target"] == "CV"
    assert result["isomorphisms"][0]["similarity"] == 0.72

    # Summary counts
    assert result["summary"]["critical_nodes"] == 1
    assert result["summary"]["hypotheses"] == 1
    assert result["summary"]["isomorphisms"] == 1
    # find_chains_from is called per concept (2 concepts), 1 chain each → 2 total
    assert result["summary"]["causal_chains"] == 2


# -- 5. empty concepts ------------------------------------------------------


def test_analyze_paper_empty_concepts():
    """Paper with zero concepts → seeds and chains are empty lists."""
    db = MagicMock()
    db.get_paper.return_value = _fake_paper("p4", "Empty Paper", 2021)
    db.get_concepts_by_paper.return_value = []
    db.get_arguments_by_paper.return_value = []

    graph = MagicMock()
    graph.detect_research_seeds.return_value = [_seed("Some Concept")]
    graph.closure.return_value = []

    result = analyze_paper(db, graph, "p4")

    assert len(result["seeds"]) == 1
    assert result["causal_chains"] == []
    assert result["summary"]["seeds"] == 1
    assert result["summary"]["causal_chains"] == 0


# -- 6. summary fields match actual array lengths ---------------------------


def test_summary_counts_match_arrays():
    """Every summary count matches the length of the corresponding array."""
    db = MagicMock()
    db.get_paper.return_value = _fake_paper("p5", "Count Check Paper", 2022)
    # Single concept so find_chains_from is called once, returning exactly 2 chains
    db.get_concepts_by_paper.return_value = [
        _fake_concept("C1"),
    ]
    db.get_arguments_by_paper.return_value = [
        _fake_arg("A solves C1", "solves", "C1", "Problem", mechanism="gradient descent"),
        _fake_arg("B extends C2", "extends", "C2", "Method", mechanism="attention"),
    ]

    graph = MagicMock()
    graph.detect_research_seeds.return_value = [
        _seed("C1"),
        _seed("Outside"),
    ]
    graph.closure.return_value = [
        {"src": "X", "dst": "Y", "relation": "creates_debate"},
        {"src": "A", "dst": "B", "relation": "gap_addressed"},
    ]

    # Mock chains: two chains, each with source/target/mechanism
    chain1 = MagicMock()
    chain1.source = "C1"
    chain1.target = "C2"
    chain1.mechanism = "via1"
    chain2 = MagicMock()
    chain2.source = "C2"
    chain2.target = "C3"
    chain2.mechanism = "via2"

    with patch("drbrain.extractor.causal_chain.find_chains_from", return_value=[chain1, chain2]):
        result = analyze_paper(db, graph, "p5")

    s = result["summary"]
    assert s["seeds"] == len(result["seeds"]) == 2
    assert s["causal_chains"] == len(result["causal_chains"]) == 2
    assert s["inferred_edges"] == len(graph.closure()) == 2
    assert s["critical_nodes"] == len(result.get("critical_nodes", [])) == 0
    assert s["hypotheses"] == len(result.get("hypotheses", [])) == 0
    assert s["isomorphisms"] == len(result.get("isomorphisms", [])) == 0


# -- 7. chains limited to 5 concepts ----------------------------------------


def test_chains_only_for_first_5_concepts():
    """find_chains_from is called at most 5 times, once per concept in order."""
    db = MagicMock()
    db.get_paper.return_value = _fake_paper("p6", "Many Concepts", 2023)
    # 7 concepts; only first 5 should be used for chain lookup
    concepts = [_fake_concept(f"C{i}") for i in range(7)]
    db.get_concepts_by_paper.return_value = concepts
    db.get_arguments_by_paper.return_value = [
        _fake_arg("arg", "proposes", "C0", "Method", mechanism="m"),
    ]

    graph = MagicMock()
    graph.detect_research_seeds.return_value = []
    graph.closure.return_value = []

    call_log = []

    def _fake_find_chains_from(args, concept):
        call_log.append(concept)
        chain = MagicMock()
        chain.source = concept
        chain.target = "Dst"
        chain.mechanism = "m"
        return [chain]

    with patch(
        "drbrain.extractor.causal_chain.find_chains_from", side_effect=_fake_find_chains_from
    ):
        analyze_paper(db, graph, "p6")

    # Only first 5 concepts queried
    assert len(call_log) == 5


# -- 8. seeds limited to 10 -------------------------------------------------


def test_seeds_limited_to_10():
    """At most 10 relevant seeds are included."""
    db = MagicMock()
    db.get_paper.return_value = _fake_paper("p7", "Seed Limit Paper", 2020)
    db.get_concepts_by_paper.return_value = [_fake_concept("Concept")]
    db.get_arguments_by_paper.return_value = []

    # 15 seeds all matching the paper concept
    graph = MagicMock()
    graph.detect_research_seeds.return_value = [_seed("Concept") for _ in range(15)]
    graph.closure.return_value = []

    result = analyze_paper(db, graph, "p7")

    assert len(result["seeds"]) == 10
    assert result["summary"]["seeds"] == 10


# -- 9. chains limited to 10 ------------------------------------------------


def test_chains_limited_to_10():
    """At most 10 causal chains are included."""
    db = MagicMock()
    db.get_paper.return_value = _fake_paper("p8", "Chain Limit Paper", 2021)
    db.get_concepts_by_paper.return_value = [_fake_concept("C")]
    db.get_arguments_by_paper.return_value = [
        _fake_arg("arg", "proposes", "C", "Method", mechanism="m"),
    ]

    graph = MagicMock()
    graph.detect_research_seeds.return_value = []
    graph.closure.return_value = []

    chains = []
    for i in range(15):
        c = MagicMock()
        c.source = f"S{i}"
        c.target = f"T{i}"
        c.mechanism = f"m{i}"
        chains.append(c)

    with patch("drbrain.extractor.causal_chain.find_chains_from", return_value=chains):
        result = analyze_paper(db, graph, "p8")

    assert len(result["causal_chains"]) == 10
    assert result["summary"]["causal_chains"] == 10


# -- 10. db and graph calls are made -----------------------------------------


def test_db_calls():
    """Verify db methods are called with correct local_id."""
    db = MagicMock()
    db.get_paper.return_value = _fake_paper()
    db.get_concepts_by_paper.return_value = []
    db.get_arguments_by_paper.return_value = []

    graph = MagicMock()
    graph.detect_research_seeds.return_value = []
    graph.closure.return_value = []

    analyze_paper(db, graph, "paper-xyz")

    db.get_paper.assert_called_once_with("paper-xyz")
    db.get_concepts_by_paper.assert_called_once_with("paper-xyz")
    db.get_arguments_by_paper.assert_called_once_with("paper-xyz")
    graph.detect_research_seeds.assert_called_once_with(db)
    graph.closure.assert_called_once()


# -- 11. paper with null year preserved -------------------------------------


def test_paper_null_year():
    """Paper with None year is preserved as-is."""
    db = MagicMock()
    db.get_paper.return_value = {"local_id": "p-nullyear", "title": "No Year", "year": None}
    db.get_concepts_by_paper.return_value = []
    db.get_arguments_by_paper.return_value = []

    graph = MagicMock()
    graph.detect_research_seeds.return_value = []
    graph.closure.return_value = []

    result = analyze_paper(db, graph, "p-nullyear")

    assert result["paper"]["year"] is None
    assert result["paper"]["title"] == "No Year"


# -- Real Database tests (no mocking of DB layer) --


def test_analyze_paper_nonexistent_real_db():
    """analyze_paper returns error dict for missing paper using real Database."""
    from drbrain.graph.engine import GraphEngine
    from drbrain.storage.database import Database

    db = Database(":memory:")
    graph = GraphEngine()
    result = analyze_paper(db, graph, "nonexistent")
    assert "error" in result
    db.close()


def test_analyze_paper_empty_concepts_real_db():
    """analyze_paper handles paper with no concepts using real Database."""
    from drbrain.graph.engine import GraphEngine
    from drbrain.storage.database import Database

    db = Database(":memory:")
    db.insert_paper("p1", "Empty Concepts Paper", 2026, "extracted")
    db.commit()
    graph = GraphEngine()
    result = analyze_paper(db, graph, "p1")
    assert result["paper"]["local_id"] == "p1"
    assert result["paper"]["title"] == "Empty Concepts Paper"
    assert len(result.get("seeds", [])) >= 0
    assert result["causal_chains"] == []
    db.close()


def test_analyze_paper_full_false_real_db():
    """analyze_paper with full=False skips counterfactual/hypotheses/isomorphism."""
    from drbrain.graph.engine import GraphEngine
    from drbrain.storage.database import Database

    db = Database(":memory:")
    db.insert_paper("p1", "Test Paper", 2026, "extracted")
    db.commit()
    graph = GraphEngine()
    result = analyze_paper(db, graph, "p1", full=False)
    assert "seeds" in result
    assert "causal_chains" in result
    assert "critical_nodes" not in result
    assert "hypotheses" not in result
    assert "isomorphisms" not in result
    db.close()


# -- LLM path tests (models=) and cross-paper insights --


def _async_return(value):
    """Return an async function that yields a fixed value."""

    async def _inner(*a, **kw):
        return value

    return _inner


# -- add_cross_paper_insights tests --


def test_add_cross_paper_insights_no_db():
    """No db returns reports unchanged."""
    from drbrain.report.analyzer import add_cross_paper_insights

    reports = [{"paper": {"local_id": "p1", "title": "Paper 1"}}]
    result = add_cross_paper_insights(reports, db=None)
    assert result is reports
    assert "cross_paper_insights" not in result[0]


def test_add_cross_paper_insights_single_report():
    """Single report returns unchanged (fewer than 2 reports)."""
    from drbrain.report.analyzer import add_cross_paper_insights

    db = MagicMock()
    reports = [{"paper": {"local_id": "p1", "title": "Paper 1"}}]
    result = add_cross_paper_insights(reports, db=db)
    assert result is reports
    assert "cross_paper_insights" not in result[0]


def test_add_cross_paper_insights_same_paper_skip():
    """Same paper Method and Problem does not produce cross-paper insights."""
    from drbrain.report.analyzer import add_cross_paper_insights
    from drbrain.storage.database import Database

    db = Database(":memory:")
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) "
        "VALUES ('p1', 'Paper 1', 2026, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) "
        "VALUES ('p1', 'Method', 'CNN', 0.9, 'method')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) "
        "VALUES ('p1', 'Problem', 'CNN problem', 0.9, 'intro')"
    )
    db.commit()

    reports = [{"paper": {"local_id": "p1", "title": "Paper 1"}}]
    result = add_cross_paper_insights(reports, db=db)
    # Single report, so early return
    assert result is reports
    assert "cross_paper_insights" not in result[0]
    db.close()


def test_add_cross_paper_insights_basic():
    """Two papers with Method and Problem concepts produce cross-paper insights."""
    from drbrain.report.analyzer import add_cross_paper_insights
    from drbrain.storage.database import Database

    db = Database(":memory:")
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) "
        "VALUES ('p1', 'Graph Neural Networks', 2026, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) "
        "VALUES ('p2', 'Graph Classification', 2026, 'extracted')"
    )
    # Method from p1 shares words with Problem from p2 -> Jaccard > 0.3
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) "
        "VALUES ('p1', 'Method', 'graph neural network', 0.9, 'method')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) "
        "VALUES ('p2', 'Problem', 'neural network classification', 0.9, 'intro')"
    )
    db.commit()

    reports = [
        {"paper": {"local_id": "p1", "title": "Graph Neural Networks"}},
        {"paper": {"local_id": "p2", "title": "Graph Classification"}},
    ]
    result = add_cross_paper_insights(reports, db=db)
    assert isinstance(result, list)
    # cross_paper_insights added to first report
    assert "cross_paper_insights" in result[0]
    insights = result[0]["cross_paper_insights"]
    assert len(insights) >= 1
    # Each insight has the expected keys
    insight = insights[0]
    assert "method" in insight
    assert "method_paper" in insight
    assert "problem" in insight
    assert "problem_paper" in insight
    assert "similarity" in insight
    db.close()


def test_add_cross_paper_insights_no_similarity():
    """Two papers with completely unrelated Method/Problem produce no insights."""
    from drbrain.report.analyzer import add_cross_paper_insights
    from drbrain.storage.database import Database

    db = Database(":memory:")
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) "
        "VALUES ('p1', 'Paper 1', 2026, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) "
        "VALUES ('p2', 'Paper 2', 2026, 'extracted')"
    )
    # Completely unrelated labels — no shared words
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) "
        "VALUES ('p1', 'Method', 'xyzqwerty', 0.9, 'method')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) "
        "VALUES ('p2', 'Problem', 'foobarbaz', 0.9, 'intro')"
    )
    db.commit()

    reports = [
        {"paper": {"local_id": "p1", "title": "Paper 1"}},
        {"paper": {"local_id": "p2", "title": "Paper 2"}},
    ]
    result = add_cross_paper_insights(reports, db=db)
    assert "cross_paper_insights" not in result[0]
    db.close()


def test_add_cross_paper_insights_same_paper_skip_in_loop():
    """Two papers where one has both Method and Problem — same-pair skipped in loop."""
    from drbrain.report.analyzer import add_cross_paper_insights
    from drbrain.storage.database import Database

    db = Database(":memory:")
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) "
        "VALUES ('p1', 'Paper A', 2026, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) "
        "VALUES ('p2', 'Paper B', 2026, 'extracted')"
    )
    # p1 has both Method and Problem with overlapping words
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) "
        "VALUES ('p1', 'Method', 'deep learning model', 0.9, 'method')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) "
        "VALUES ('p1', 'Problem', 'learning model task', 0.9, 'intro')"
    )
    # p2 has a Problem that overlaps with p1's Method
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) "
        "VALUES ('p2', 'Problem', 'deep learning task', 0.9, 'intro')"
    )
    db.commit()

    reports = [
        {"paper": {"local_id": "p1", "title": "Paper A"}},
        {"paper": {"local_id": "p2", "title": "Paper B"}},
    ]
    result = add_cross_paper_insights(reports, db=db)
    assert "cross_paper_insights" in result[0]
    insights = result[0]["cross_paper_insights"]
    # Should have cross-paper insights (p1 Method -> p2 Problem)
    # but NOT same-paper (p1 Method -> p1 Problem)
    for insight in insights:
        assert insight["method_paper"] != insight["problem_paper"]
    db.close()


# -- analyze_paper with models= tests (mock LLM) --


def test_analyze_paper_with_models_executive_summary():
    """models param triggers executive_summary via LLM mock."""
    from drbrain.graph.engine import GraphEngine
    from drbrain.storage.database import Database

    db = Database(":memory:")
    db.insert_paper("p1", "Test Paper", 2026, "extracted")
    db.commit()
    graph = GraphEngine()

    models = [{"provider": "openai", "model": "gpt-4o", "api_key": "test"}]

    with patch(
        "drbrain.extractor.llm_client.acall_text_with_fallback",
        new=_async_return("This is a mock executive summary."),
    ):
        result = analyze_paper(db, graph, "p1", models=models)

    assert "executive_summary" in result
    assert result["executive_summary"] == "This is a mock executive summary."
    db.close()


def test_analyze_paper_with_models_enhances_seeds():
    """models param with seeds triggers seed enhancement via LLM mock."""
    from drbrain.graph.engine import GraphEngine
    from drbrain.storage.database import Database

    db = Database(":memory:")
    db.insert_paper("p1", "Seed Test Paper", 2026, "extracted")
    db.commit()
    graph = GraphEngine()
    # Make sure detect_research_seeds returns seeds with 'concept' key
    with patch.object(
        graph,
        "detect_research_seeds",
        return_value=[
            {"concept": "test concept", "type": "stale_problem", "description": "Some gap"}
        ],
    ):
        models = [{"provider": "openai", "model": "gpt-4o", "api_key": "test"}]

        # First call: executive_summary, second: seed enhancement
        with patch(
            "drbrain.extractor.llm_client.acall_text_with_fallback",
            new=_async_return("Try transformer-based approach."),
        ):
            result = analyze_paper(db, graph, "p1", models=models)

        assert "executive_summary" in result
        # Seeds should have suggested_solutions added
        assert len(result["seeds"]) >= 1
        assert "suggested_solutions" in result["seeds"][0]
    db.close()


def test_analyze_paper_with_models_graph_summary():
    """models param triggers graph_summary via LLM + describe_subgraph mocks."""
    from drbrain.graph.engine import GraphEngine
    from drbrain.storage.database import Database

    db = Database(":memory:")
    db.insert_paper("p1", "Graph Summary Test", 2026, "extracted")
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) "
        "VALUES ('p1', 'Method', 'Transformer', 0.95, 'method')"
    )
    db.commit()
    graph = GraphEngine()
    graph.load_from_db(db)

    models = [{"provider": "openai", "model": "gpt-4o", "api_key": "test"}]

    with (
        patch(
            "drbrain.extractor.llm_client.acall_text_with_fallback",
            new=_async_return("Mock summary"),
        ),
        patch(
            "drbrain.services.graph_to_text.describe_subgraph",
            new=_async_return("Transformer is connected to attention mechanism."),
        ),
    ):
        result = analyze_paper(db, graph, "p1", models=models)

    assert "executive_summary" in result
    assert "graph_summary" in result
    assert "Transformer" in result["graph_summary"]
    db.close()


def test_analyze_paper_models_empty_seeds():
    """models param with no seeds skips seed enhancement."""
    from drbrain.graph.engine import GraphEngine
    from drbrain.storage.database import Database

    db = Database(":memory:")
    db.insert_paper("p1", "Empty Seeds Test", 2026, "extracted")
    db.commit()
    graph = GraphEngine()

    models = [{"provider": "openai", "model": "gpt-4o", "api_key": "test"}]

    # Only executive_summary call (no seed enhancement)
    with (
        patch.object(graph, "detect_research_seeds", return_value=[]),
        patch(
            "drbrain.extractor.llm_client.acall_text_with_fallback",
            new=_async_return("Mock summary for empty seeds."),
        ),
    ):
        result = analyze_paper(db, graph, "p1", models=models)

    assert "executive_summary" in result
    assert result["executive_summary"] == "Mock summary for empty seeds."
    assert result["seeds"] == []
    db.close()


def test_analyze_paper_full_true_real_graph():
    """full=True with real graph and concepts includes all analysis sections."""
    from drbrain.graph.engine import GraphEngine
    from drbrain.storage.database import Database

    db = Database(":memory:")
    db.insert_paper("p1", "Full Analysis", 2026, "extracted")
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, section) "
        "VALUES ('p1', 'Problem', 'test problem', 0.9, 'intro')"
    )
    db.commit()
    graph = GraphEngine()
    graph.load_from_db(db)

    result = analyze_paper(db, graph, "p1", full=True)

    assert "critical_nodes" in result
    assert "hypotheses" in result
    assert "isomorphisms" in result
    assert "summary" in result
    assert "seeds" in result
    # Summary counts match array lengths
    s = result["summary"]
    assert s["critical_nodes"] == len(result["critical_nodes"])
    assert s["hypotheses"] == len(result["hypotheses"])
    assert s["isomorphisms"] == len(result["isomorphisms"])
    db.close()


def test_analyze_paper_summary_counts():
    """Summary counts match actual array lengths."""
    from drbrain.graph.engine import GraphEngine
    from drbrain.storage.database import Database

    db = Database(":memory:")
    db.insert_paper("p1", "Count Test", 2026, "extracted")
    db.commit()
    graph = GraphEngine()

    result = analyze_paper(db, graph, "p1", full=False)
    s = result["summary"]
    assert s["seeds"] == len(result["seeds"])
    assert s["causal_chains"] == len(result["causal_chains"])
    assert s["critical_nodes"] == len(result.get("critical_nodes", []))
    assert s["hypotheses"] == len(result.get("hypotheses", []))
    assert s["isomorphisms"] == len(result.get("isomorphisms", []))
    db.close()
