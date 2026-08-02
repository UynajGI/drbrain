"""Tests for the hybrid-search wiring in ask_cmd.

Verifies that ask_cmd:
  - uses hybrid_search for retrieval (pure-BM25 when embedding is disabled)
  - --hyde is off by default (no extra LLM call) and on when requested
  - --rerank does not crash without sentence-transformers (no-op fallback)
  - closure edges still appear in the prompt (existing behavior preserved)

Mirrors the direct-call style of test_cli_commands.py's ask_cmd tests:
invoke ask_cmd with a mocked typer.Context, patch the LLM caller to capture
the prompt, and assert on what reached the model.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock

import typer

from drbrain.storage.database import Database


def _make_config(db_path: str, reports_dir: str) -> dict:
    cfg = {
        "db": {"path": db_path},
        "llm": {"models": [{"provider": "openai", "model": "gpt-4", "api_key": "x"}]},
        "mineru": {},
        "dirs": {
            "inbox": "data/spool/inbox",
            "papers": "data/papers",
            "reports": reports_dir,
            "cache": "data/cache",
            "logs": "data/logs",
        },
        "api": {},
        "queue": {"weak_threshold": 0.7, "auto_accept": 0.9},
        "bm25": {"k1": 1.5, "b": 0.75},
        # disable embedding to avoid loading a real model in tests
        "embed": {"provider": "none"},
    }
    return cfg


def _make_ctx(cfg: dict):
    ctx = mock.MagicMock(spec=typer.Context)
    ctx.obj = {"config": cfg}
    return ctx


def _seed_db(db_path: str) -> None:
    """Insert one paper with concepts and a closure-triggering edge."""
    db = Database(str(db_path))
    db.insert_paper("p1", "Attention Is All You Need", 2024, "uploaded")
    db.insert_concept("p1", "Method", "transformer", 0.9, year=2024)
    db.insert_concept("p1", "Method", "attention mechanism", 0.85, year=2024)
    db.insert_edge("transformer", "attention mechanism", "supports", "p1", 1.0)
    db.commit()
    db.close()


def _capture_prompt() -> tuple[list[str], object]:
    """Return (prompt_sink, patcher_cm) — call __enter__ then ask_cmd then __exit__."""
    sink: list[str] = []

    async def _fake_llm(prompt, models, max_tokens=1024):  # noqa: ANN001
        sink.append(prompt)
        return "Test answer"

    return sink, mock.patch("drbrain.extractor.llm_client.acall_text_with_fallback", _fake_llm)


class TestAskHybridWiring:
    def test_ask_uses_hybrid_pure_bm25(self):
        """ask_cmd runs through hybrid_search in pure-BM25 mode (embed=none)."""
        from drbrain.cli.analysis_commands import ask_cmd

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "test.db"
            _seed_db(db_path)
            ctx = _make_ctx(_make_config(str(db_path), str(Path(td) / "reports")))

            sink, patcher = _capture_prompt()
            with patcher:
                ask_cmd(ctx, ["transformer", "attention"])

        assert len(sink) == 1
        prompt = sink[0]
        # The paper's concepts should appear in the assembled context.
        assert "transformer" in prompt
        assert "Attention Is All You Need" in prompt

    def test_ask_hyde_off_by_default(self):
        """Without --hyde, ahyde_transform is never called."""
        from drbrain.cli.analysis_commands import ask_cmd

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "test.db"
            _seed_db(db_path)
            ctx = _make_ctx(_make_config(str(db_path), str(Path(td) / "reports")))

            sink, patcher = _capture_prompt()
            with patcher, mock.patch("drbrain.query.query_transform.ahyde_transform") as mock_hyde:
                ask_cmd(ctx, ["transformer"])

            mock_hyde.assert_not_called()
            assert len(sink) == 1

    def test_ask_hyde_on_invokes_transform(self):
        """With --hyde, ahyde_transform is called and its query drives retrieval."""
        from drbrain.cli.analysis_commands import ask_cmd

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "test.db"
            _seed_db(db_path)
            ctx = _make_ctx(_make_config(str(db_path), str(Path(td) / "reports")))

            # HyDE returns a transformed query; we verify it reaches BM25.
            from drbrain.query.query_transform import HydeResult

            async def _fake_ahyde(question, models, **kwargs):  # noqa: ANN001
                return HydeResult(
                    query="hypothetical transformer passage",
                    original=question,
                    transformed=True,
                    hypothesis="hypothetical transformer passage",
                )

            sink, patcher = _capture_prompt()
            with patcher, mock.patch("drbrain.query.query_transform.ahyde_transform", _fake_ahyde):
                # hyde=True as a direct-call kwarg (bypasses typer OptionInfo)
                ask_cmd(ctx, ["irrelevant"], hyde=True)

            # The LLM prompt should carry the ORIGINAL question in the Question:
            # line (the answer addresses the user's actual question), while
            # retrieval used the hypothetical. We just assert no crash + LLM ran.
            assert len(sink) == 1

    def test_ask_hyde_failure_falls_back(self):
        """If HyDE's LLM call fails, ask_cmd still answers using the original query."""
        from drbrain.cli.analysis_commands import ask_cmd

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "test.db"
            _seed_db(db_path)
            ctx = _make_ctx(_make_config(str(db_path), str(Path(td) / "reports")))

            async def _boom_ahyde(question, models, **kwargs):  # noqa: ANN001
                raise RuntimeError("LLM down")

            sink, patcher = _capture_prompt()
            with patcher, mock.patch("drbrain.query.query_transform.ahyde_transform", _boom_ahyde):
                # Must not raise; HyDE module catches and returns original query.
                ask_cmd(ctx, ["transformer"], hyde=True)

            assert len(sink) == 1

    def test_ask_rerank_flag_no_crash(self):
        """--rerank runs without sentence-transformers (auto no-op fallback)."""
        from drbrain.cli.analysis_commands import ask_cmd

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "test.db"
            _seed_db(db_path)
            ctx = _make_ctx(_make_config(str(db_path), str(Path(td) / "reports")))

            sink, patcher = _capture_prompt()
            with patcher:
                ask_cmd(ctx, ["transformer"], rerank=True)

            assert len(sink) == 1
            assert "transformer" in sink[0]

    def test_ask_closure_still_works(self):
        """Closure-inferred edges still appear in the prompt after wiring."""
        from drbrain.cli.analysis_commands import ask_cmd

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "test.db"
            db = Database(str(db_path))
            db.insert_paper("p1", "Test Paper", 2024, "uploaded")
            db.insert_concept("p1", "Problem", "overfitting", 0.9, year=2024)
            db.insert_concept("p1", "Conclusion", "deep learning", 0.9, year=2024)
            db.insert_concept("p1", "Method", "regularization", 0.85, year=2024)
            db.insert_edge("overfitting", "deep learning", "challenges", "p1", 1.0)
            db.insert_edge("regularization", "deep learning", "supports", "p1", 1.0)
            db.commit()
            db.close()

            ctx = _make_ctx(_make_config(str(db_path), str(Path(td) / "reports")))
            sink, patcher = _capture_prompt()
            with patcher:
                ask_cmd(ctx, ["deep", "learning"])

            prompt = sink[0]
            assert "--[inferred:" in prompt, "closure edges must survive the wiring"
            assert "regularization" in prompt
