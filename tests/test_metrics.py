"""Tests for metrics tracking."""

import threading
import time

import pytest

from drbrain.metrics import MetricsStore, get_metrics


class TestMetricsStoreCreation:
    """Database and table creation tests."""

    def test_creates_db_and_tables(self, tmp_path):
        """MetricsStore creates the database file, llm_calls + events tables."""
        db = tmp_path / "test.db"
        store = MetricsStore(str(db))
        conn = store._ensure_conn()
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = {t[0] for t in tables}
        assert "llm_calls" in table_names
        assert "events" in table_names
        store.close()

    def test_api_calls_table_not_created(self, tmp_path):
        """api_calls table is no longer created (removed dead code)."""
        db = tmp_path / "test.db"
        store = MetricsStore(str(db))
        conn = store._ensure_conn()
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = {t[0] for t in tables}
        assert "api_calls" not in table_names
        store.close()


class TestRecordLLM:
    """Backward-compat: record_llm still works."""

    def test_inserts_row(self, tmp_path):
        """record_llm inserts a row into llm_calls."""
        db = tmp_path / "test.db"
        store = MetricsStore(str(db))
        store.record_llm(
            model="gpt-4",
            provider="openai",
            tokens_in=100,
            tokens_out=50,
            duration_ms=2000,
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

    def test_full_backward_compat(self, tmp_path):
        """record_llm works exactly as before after refactor."""
        db = tmp_path / "test.db"
        store = MetricsStore(str(db))
        store.record_llm(
            model="claude-4",
            provider="anthropic",
            tokens_in=50,
            tokens_out=25,
            duration_ms=500,
        )
        conn = store._ensure_conn()
        row = conn.execute(
            "SELECT model, provider, tokens_in, tokens_out, duration_ms FROM llm_calls"
        ).fetchone()
        assert row == ("claude-4", "anthropic", 50, 25, 500)
        store.close()

    def test_session_id_column_exists(self, tmp_path):
        """llm_calls has session_id column after migration."""
        db = tmp_path / "test.db"
        store = MetricsStore(str(db))
        conn = store._ensure_conn()
        cols = {c[1] for c in conn.execute("PRAGMA table_info(llm_calls)").fetchall()}
        assert "session_id" in cols
        store.close()


class TestTimerContextManager:
    """New timer() context manager."""

    def test_records_on_success(self, tmp_path):
        """timer() records duration with status='ok'."""
        db = tmp_path / "test.db"
        store = MetricsStore(str(db))
        with store.timer("test-category", "test-name"):
            time.sleep(0.01)
        conn = store._ensure_conn()
        row = conn.execute(
            "SELECT category, name, status FROM events WHERE category='test-category'"
        ).fetchone()
        assert row is not None
        assert row[0] == "test-category"
        assert row[1] == "test-name"
        assert row[2] == "ok"
        store.close()

    def test_records_error_status(self, tmp_path):
        """timer() sets status='error' on exception, re-raises."""
        db = tmp_path / "test.db"
        store = MetricsStore(str(db))
        with pytest.raises(ValueError, match="boom"):
            with store.timer("test-category", "test-err"):
                raise ValueError("boom")
        conn = store._ensure_conn()
        row = conn.execute(
            "SELECT category, name, status FROM events WHERE category='test-category'"
        ).fetchone()
        assert row is not None
        assert row[2] == "error"
        store.close()

    def test_records_positive_duration(self, tmp_path):
        """timer() records positive duration_ms."""
        db = tmp_path / "test.db"
        store = MetricsStore(str(db))
        with store.timer("perf", "duration-test"):
            time.sleep(0.05)
        conn = store._ensure_conn()
        row = conn.execute("SELECT duration_ms FROM events WHERE category='perf'").fetchone()
        assert row[0] > 0
        store.close()

    def test_empty_name_default(self, tmp_path):
        """timer() with empty name stores empty string."""
        db = tmp_path / "test.db"
        store = MetricsStore(str(db))
        with store.timer("cat-only"):
            pass
        conn = store._ensure_conn()
        row = conn.execute("SELECT name FROM events WHERE category='cat-only'").fetchone()
        assert row[0] == ""
        store.close()


class TestTimedDecorator:
    """New timed() decorator."""

    def test_decorator_records_timing(self, tmp_path):
        """timed() decorator records timing using function name."""
        db = tmp_path / "test.db"
        store = MetricsStore(str(db))

        @store.timed("decorator-test")
        def my_func():
            time.sleep(0.01)
            return 42

        result = my_func()
        assert result == 42
        conn = store._ensure_conn()
        row = conn.execute(
            "SELECT category, name, status FROM events WHERE category='decorator-test'"
        ).fetchone()
        assert row is not None
        assert row[1] == "my_func"
        assert row[2] == "ok"
        store.close()

    def test_custom_name(self, tmp_path):
        """timed() uses custom name when provided."""
        db = tmp_path / "test.db"
        store = MetricsStore(str(db))

        @store.timed("decorator-test", "custom-func-name")
        def my_func():
            return 1

        my_func()
        conn = store._ensure_conn()
        row = conn.execute("SELECT name FROM events WHERE category='decorator-test'").fetchone()
        assert row[0] == "custom-func-name"
        store.close()

    def test_preserves_func_metadata(self, tmp_path):
        """timed() preserves __name__ and __doc__ via functools.wraps."""
        db = tmp_path / "test.db"
        store = MetricsStore(str(db))

        @store.timed("meta")
        def documented_func():
            """This is a docstring."""
            return "ok"

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ == "This is a docstring."
        assert documented_func() == "ok"
        store.close()

    def test_error_propagates_and_records(self, tmp_path):
        """timed() records error status and re-raises exception."""
        db = tmp_path / "test.db"
        store = MetricsStore(str(db))

        @store.timed("error-cat")
        def failing_func():
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError, match="fail"):
            failing_func()
        conn = store._ensure_conn()
        row = conn.execute("SELECT status FROM events WHERE category='error-cat'").fetchone()
        assert row[0] == "error"
        store.close()


class TestWALAndThreadSafety:
    """WAL mode and thread safety."""

    def test_wal_mode_enabled(self, tmp_path):
        """PRAGMA journal_mode=WAL is set on connection."""
        db = tmp_path / "test.db"
        store = MetricsStore(str(db))
        conn = store._ensure_conn()
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.upper() == "WAL"
        store.close()

    def test_concurrent_writes_no_crash(self, tmp_path):
        """Concurrent record_llm writes from multiple threads don't crash."""
        db = tmp_path / "test.db"
        store = MetricsStore(str(db))
        errors = []

        def write_record(i):
            try:
                store.record_llm(model=f"model-{i}", tokens_in=i, tokens_out=i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_record, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        store.close()

    def test_concurrent_timer_writes(self, tmp_path):
        """Concurrent timer() usage from multiple threads doesn't crash."""
        db = tmp_path / "test.db"
        store = MetricsStore(str(db))
        errors = []

        def timed_work(i):
            try:
                with store.timer("concurrent", f"work-{i}"):
                    time.sleep(0.005)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=timed_work, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        conn = store._ensure_conn()
        count = conn.execute("SELECT COUNT(*) FROM events WHERE category='concurrent'").fetchone()[
            0
        ]
        assert count == 10
        store.close()


class TestRecordEvent:
    """Generic _record_event method."""

    def test_writes_to_events_table(self, tmp_path):
        """_record_event writes a row to the events table."""
        db = tmp_path / "test.db"
        store = MetricsStore(str(db))
        store._record_event("http", "fetch_arxiv", 123.4, status="ok")
        conn = store._ensure_conn()
        row = conn.execute("SELECT category, name, duration_ms, status FROM events").fetchone()
        assert row == ("http", "fetch_arxiv", 123, "ok")
        store.close()

    def test_writes_optional_fields(self, tmp_path):
        """_record_event writes tokens_in/out, model, detail when provided."""
        db = tmp_path / "test.db"
        store = MetricsStore(str(db))
        store._record_event(
            "llm",
            "chat",
            500.0,
            status="ok",
            tokens_in=100,
            tokens_out=50,
            model="gpt-4",
            detail='{"prompt":"hi"}',
        )
        conn = store._ensure_conn()
        row = conn.execute("SELECT tokens_in, tokens_out, model, detail FROM events").fetchone()
        assert row[0] == 100
        assert row[1] == 50
        assert row[2] == "gpt-4"
        assert row[3] == '{"prompt":"hi"}'
        store.close()


class TestEventsTableSchema:
    """Schema verification for the events table."""

    def test_has_session_id_column(self, tmp_path):
        """events table has session_id column."""
        db = tmp_path / "test.db"
        store = MetricsStore(str(db))
        conn = store._ensure_conn()
        cols = {c[1] for c in conn.execute("PRAGMA table_info(events)").fetchall()}
        assert "session_id" in cols
        store.close()

    def test_has_expected_columns(self, tmp_path):
        """events table has all expected columns."""
        db = tmp_path / "test.db"
        store = MetricsStore(str(db))
        conn = store._ensure_conn()
        cols = {c[1] for c in conn.execute("PRAGMA table_info(events)").fetchall()}
        expected = {
            "id",
            "session_id",
            "category",
            "name",
            "duration_ms",
            "tokens_in",
            "tokens_out",
            "model",
            "status",
            "detail",
            "created_at",
        }
        assert expected.issubset(cols)
        store.close()


class TestDeadCodeRemoved:
    """Verify dead code was actually removed."""

    def test_llm_timer_removed(self):
        """LLMTimer class no longer exists."""
        import drbrain.metrics as m

        assert not hasattr(m, "LLMTimer")

    def test_timed_llm_removed(self):
        """timed_llm function no longer exists."""
        import drbrain.metrics as m

        assert not hasattr(m, "timed_llm")

    def test_record_api_removed(self):
        """record_api method no longer exists on MetricsStore."""
        import drbrain.metrics as m

        assert not hasattr(m.MetricsStore, "record_api")

    def test_get_llm_stats_removed(self):
        """get_llm_stats method no longer exists on MetricsStore."""
        import drbrain.metrics as m

        assert not hasattr(m.MetricsStore, "get_llm_stats")


class TestSingleton:
    """Module-level singleton behavior."""

    def test_returns_same_instance(self):
        """get_metrics returns the same MetricsStore instance."""
        import drbrain.metrics as m

        m._store = None
        s1 = get_metrics()
        s2 = get_metrics()
        assert s1 is s2
        s1.close()
        m._store = None

    def test_close_then_reopen(self):
        """After close(), next get_metrics() creates fresh instance."""
        import drbrain.metrics as m

        m._store = None
        s1 = get_metrics()
        s1.close()
        m._store = None
        s2 = get_metrics()
        assert s1 is not s2
        s2.close()
        m._store = None
