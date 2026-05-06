# Boost analyzer.py Coverage from 40% → 80%+

## Uncovered lines: 111-230 (LLM paths + cross_paper_insights)

### What needs testing:
1. `add_cross_paper_insights()` — cross-paper method→Problem matching
2. `_generate_executive_summary()` — mock `acall_text_with_fallback`
3. `_enhance_seeds()` — mock LLM call
4. `analyze_paper` with `models=` — mock `acall_text_with_fallback` + `describe_subgraph`
5. `analyze_paper` full=True with real graph data

## Approach

Mock `acall_text_with_fallback` to return canned responses. Set up DB with:
- 2+ papers with Method and Problem concepts
- Graph with closure edges
- Arguments for causal chains

### Tests to add to `tests/test_analyzer.py`:

```python
@mock.patch("drbrain.extractor.llm_client.acall_text_with_fallback", new=mock_async_return("mock summary"))
def test_analyze_paper_with_models_generates_summary():
    """models param triggers executive_summary + graph_summary"""
    ...

@mock.patch("drbrain.extractor.llm_client.acall_text_with_fallback", new=mock_async_return("solution"))
def test_analyze_paper_models_enhances_seeds():
    """LLM adds suggested_solutions to seeds"""
    ...

def test_add_cross_paper_insights_basic():
    """Two papers with Method→Problem similarity"""
    from drbrain.report.analyzer import add_cross_paper_insights
    from drbrain.storage.database import Database
    
    db = Database(":memory:")
    # Setup 2 papers with Method and Problem concepts
    db.conn.execute("INSERT INTO papers (local_id, title, year, status) VALUES ('p1', 'Paper 1', 2026, 'extracted')")
    db.conn.execute("INSERT INTO papers (local_id, title, year, status) VALUES ('p2', 'Paper 2', 2026, 'extracted')")
    db.conn.execute("INSERT INTO concepts (paper_id, type, label, confidence, section) VALUES ('p1', 'Method', 'graph neural networks', 0.9, 'method')")
    db.conn.execute("INSERT INTO concepts (paper_id, type, label, confidence, section) VALUES ('p2', 'Problem', 'graph classification', 0.9, 'intro')")
    db.commit()
    
    reports = [{"paper": {"local_id": "p1", "title": "Paper 1"}}, {"paper": {"local_id": "p2", "title": "Paper 2"}}]
    result = add_cross_paper_insights(reports, db)
    assert isinstance(result, list)
    # Method→Problem with some similarity should produce insight
    ...
    
    db.close()

def test_add_cross_paper_insights_single_report():
    """Single report returns unchanged"""
    ...

def test_add_cross_paper_insights_no_db():
    """No db returns reports unchanged"""
    ...
```

## Acceptance
- analyzer.py coverage ≥ 80%
- All new tests pass
- ruff clean
