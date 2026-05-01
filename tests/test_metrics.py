"""Tests for LLM metrics tracking."""

from drbrain.metrics import LLMTimer, MetricsStore, get_metrics


def test_metrics_store_creates_db(tmp_path):
    """MetricsStore creates the database file and tables."""
    db = tmp_path / "test.db"
    store = MetricsStore(str(db))
    conn = store._ensure_conn()
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = {t[0] for t in tables}
    assert "llm_calls" in table_names
    assert "api_calls" in table_names
    store.close()


def test_record_llm_inserts_row(tmp_path):
    """record_llm inserts a row into llm_calls."""
    db = tmp_path / "test.db"
    store = MetricsStore(str(db))
    store.record_llm(
        model="gpt-4", provider="openai", tokens_in=100, tokens_out=50, duration_ms=2000
    )
    conn = store._ensure_conn()
    row = conn.execute(
        "SELECT model, provider, tokens_in, tokens_out, duration_ms FROM llm_calls"
    ).fetchone()
    assert row[0] == "gpt-4"
    assert row[1] == "openai"
    assert row[2] == 100
    assert row[3] == 50
    assert row[4] == 2000
    store.close()


def test_record_api_inserts_row(tmp_path):
    """record_api inserts a row into api_calls."""
    db = tmp_path / "test.db"
    store = MetricsStore(str(db))
    store.record_api(api="CrossRef", endpoint="works/doi", duration_ms=500, status="ok")
    conn = store._ensure_conn()
    row = conn.execute("SELECT api, endpoint, duration_ms, status FROM api_calls").fetchone()
    assert row[0] == "CrossRef"
    assert row[1] == "works/doi"
    assert row[2] == 500
    assert row[3] == "ok"
    store.close()


def test_get_llm_stats_empty(tmp_path):
    """Empty store returns zero stats."""
    db = tmp_path / "test.db"
    store = MetricsStore(str(db))
    stats = store.get_llm_stats()
    assert stats["total_calls"] == 0
    assert stats["total_tokens_in"] == 0
    store.close()


def test_get_llm_stats_with_data(tmp_path):
    """get_llm_stats returns aggregated counts."""
    db = tmp_path / "test.db"
    store = MetricsStore(str(db))
    store.record_llm(model="gpt-4", tokens_in=100, tokens_out=50, duration_ms=1000)
    store.record_llm(model="gpt-4", tokens_in=200, tokens_out=100, duration_ms=2000)
    stats = store.get_llm_stats()
    assert stats["total_calls"] == 2
    assert stats["total_tokens_in"] == 300
    assert stats["total_tokens_out"] == 150
    assert stats["total_duration_ms"] == 3000
    store.close()


def test_get_metrics_singleton():
    """get_metrics returns the same MetricsStore instance."""
    import drbrain.metrics as m

    m._store = None
    s1 = get_metrics()
    s2 = get_metrics()
    assert s1 is s2
    s1.close()
    m._store = None


def test_llm_timer_context():
    """LLMTimer works as context manager and records metrics."""
    with LLMTimer(model="gpt-4", provider="openai", source="test") as timer:
        pass  # fake work
    timer.record(tokens_in=50, tokens_out=25)
    stats = get_metrics().get_llm_stats()
    assert stats["total_calls"] >= 1
