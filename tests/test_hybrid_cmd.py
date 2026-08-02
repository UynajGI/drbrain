"""Tests for the 'drbrain hybrid' command (RRF fusion CLI entry).

Mirrors the direct-call style of test_search_cmd.py: invoke hybrid_cmd with a
mocked typer.Context, capture typer.echo output, assert on the rendered text.
Pure-BM25 mode is exercised (no real embedding model required).
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from unittest import mock

import typer

from drbrain.storage.database import Database


def _make_config(db_path: str) -> dict:
    return {
        "db": {"path": db_path},
        "llm": {"models": []},
        "dirs": {
            "inbox": "data/spool/inbox",
            "papers": "data/papers",
            "reports": "data/reports",
            "cache": "data/cache",
            "logs": "data/logs",
        },
        "api": {},
        "mineru": {},
        "extract": {"max_concurrent": 1},
        "bm25": {"k1": 1.5, "b": 0.75},
        "queue": {"weak_threshold": 0.5, "auto_accept": False},
        # provider=none → pure BM25, avoids loading a real embedding model
        "embed": {"provider": "none"},
    }


def _make_ctx(cfg: dict):
    ctx = mock.MagicMock(spec=typer.Context)
    ctx.obj = {"config": cfg}
    return ctx


def _capture(func, *args, **kwargs) -> str:
    """Run func with typer.echo patched to a StringIO buffer."""
    buf = io.StringIO()
    with mock.patch("typer.echo", side_effect=lambda *a, **kw: buf.write(a[0] + "\n")):
        func(*args, **kwargs)
    return buf.getvalue()


class TestHybridCmd:
    def test_hybrid_returns_matching_papers(self):
        """hybrid_cmd finds papers matching the query (pure-BM25 mode)."""
        from drbrain.cli.query_commands import hybrid_cmd

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "test.db"
            db = Database(db_path)
            db.insert_paper("p1", "Transformer Paper", 2024, "uploaded")
            db.insert_concept("p1", "Method", "transformer architecture", 0.9, year=2024)
            db.commit()
            db.close()

            out = _capture(
                hybrid_cmd,
                _make_ctx(_make_config(str(db_path))),
                "transformer",
                limit=5,
                rerank=False,
                rerank_model=None,
                rrf_k=60,
                json_output=False,
            )
            assert "p1" in out
            assert "Hybrid search" in out

    def test_hybrid_json_output(self):
        """hybrid_cmd --json returns valid JSON with result objects."""
        from drbrain.cli.query_commands import hybrid_cmd

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "test.db"
            db = Database(db_path)
            db.insert_paper("p1", "Embedding Paper", 2024, "uploaded")
            db.insert_concept("p1", "Method", "word embedding", 0.9, year=2024)
            db.commit()
            db.close()

            out = _capture(
                hybrid_cmd,
                _make_ctx(_make_config(str(db_path))),
                "embedding",
                limit=5,
                rerank=False,
                rerank_model=None,
                rrf_k=60,
                json_output=True,
            )
            payload = json.loads(out)
            assert payload["query"] == "embedding"
            assert isinstance(payload["results"], list)
            assert len(payload["results"]) >= 1
            r = payload["results"][0]
            # Stable output fields from SearchHit.to_dict()
            for field in ("paper_id", "score", "rank", "source"):
                assert field in r
            assert r["paper_id"] == "p1"

    def test_hybrid_no_results_unmatched_query(self):
        """Non-matching query returns all docs at minimal score (BM25 behavior).

        BM25 returns every doc with score 0 when no query term matches (see
        test_search_cmd.py::test_search_no_matching_terms_score_zero). So
        hybrid_cmd does NOT print 'No results' here; it surfaces the paper at
        a low RRF score. The true no-results path is covered by
        test_hybrid_empty_db below.
        """
        from drbrain.cli.query_commands import hybrid_cmd

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "test.db"
            db = Database(db_path)
            db.insert_paper("p1", "Paper", 2024, "uploaded")
            db.insert_concept("p1", "Method", "transformer", 0.9, year=2024)
            db.commit()
            db.close()

            out = _capture(
                hybrid_cmd,
                _make_ctx(_make_config(str(db_path))),
                "zzz_nonexistent_xyz",
                limit=5,
                rerank=False,
                rerank_model=None,
                rrf_k=60,
                json_output=False,
            )
            # BM25 still returns p1 (score 0); hybrid surfaces it at low RRF.
            assert "p1" in out

    def test_hybrid_empty_db(self):
        """hybrid_cmd on an empty database does not crash."""
        from drbrain.cli.query_commands import hybrid_cmd

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "test.db"
            Database(db_path).close()

            out = _capture(
                hybrid_cmd,
                _make_ctx(_make_config(str(db_path))),
                "anything",
                limit=5,
                rerank=False,
                rerank_model=None,
                rrf_k=60,
                json_output=False,
            )
            assert "No results" in out
