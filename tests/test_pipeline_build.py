"""Failure-boundary tests for the standalone build worker."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO = Path(__file__).resolve().parents[1]


def _load_build_module():
    spec = importlib.util.spec_from_file_location(
        "pipeline_build_testee", REPO / "scripts" / "pipeline" / "build.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_one_relation_failure_rolls_back_without_extracting(tmp_path: Path):
    """A failed edge must not leave concepts or an extracted paper marker."""
    build = _load_build_module()
    paper_path = tmp_path / "data" / "papers" / "p1"
    paper_path.mkdir(parents=True)
    (paper_path / "raw.md").write_text("paper text", encoding="utf-8")
    (paper_path / "tree.json").write_text(
        json.dumps({"structure": [{"title": "Section", "node_id": "n1"}]}),
        encoding="utf-8",
    )

    db = MagicMock()
    db.insert_edge.side_effect = RuntimeError("foreign-key failure")

    async def fake_extract(*_args, **_kwargs):
        return {
            "concepts": [{"type": "Method", "label": "method", "confidence": 0.8}],
            "relations": [{"head": "method", "rel": "uses", "tail": "missing"}],
            "merges": [],
            "corrections": [],
        }

    cfg = {"dirs": {"papers": "data/papers"}, "llm": {"models": ["stub"]}}
    with patch("drbrain.extractor.concept.build_graph_from_tree", new=fake_extract):
        result = build.build_one("p1", cfg, db, skip_refine=True, root=tmp_path)

    assert result["ok"] is False
    assert "relation 0 failed" in result["error"]
    db.conn.rollback.assert_called_once()
    db.set_paper_status.assert_not_called()
    db.commit.assert_not_called()


def test_build_worker_never_receives_destination_database(tmp_path: Path, monkeypatch):
    """Timeout-safe workers must return a staging payload, never a DB handle."""
    build = _load_build_module()
    seen: dict[str, object] = {}

    def fake_build_one(local_id, cfg, db, **kwargs):
        seen["db"] = db
        assert kwargs["jsonl_only"] is True
        return {
            "ok": True,
            "local_id": local_id,
            "concepts": [],
            "relations": [],
            "merges": [],
            "corrections": [],
            "report": {},
        }

    monkeypatch.setattr(build, "build_one", fake_build_one)
    result = build._run_build_worker("p1", {"llm": {"models": []}}, True, tmp_path, 1)

    assert result["ok"] is True
    assert seen["db"] is None


def test_build_worker_redacts_unlabelled_provider_secret(tmp_path: Path, monkeypatch):
    """Worker errors persisted to a manifest must not expose configured keys."""
    build = _load_build_module()
    secret = "provider-secret-123"
    cfg = {"llm": {"models": [{"provider": "stub", "api_key": secret}]}}

    def fail_build_one(*_args, **_kwargs):
        raise RuntimeError(f"provider rejected credential {secret}")

    monkeypatch.setattr(build, "build_one", fail_build_one)
    result = build._run_build_worker("p1", cfg, True, tmp_path, 1)

    assert result["ok"] is False
    assert secret not in result["error"]
    assert "[REDACTED]" in result["error"]


def test_build_main_commits_staged_worker_result_in_parent(tmp_path: Path, monkeypatch):
    """Normal mode publishes a worker result only from the parent collector."""
    build = _load_build_module()
    root = tmp_path / "runtime"
    paper_dir = root / "data" / "papers" / "p1"
    paper_dir.mkdir(parents=True)
    (paper_dir / "raw.md").write_text("paper text", encoding="utf-8")
    (paper_dir / "tree.json").write_text(
        json.dumps({"structure": [{"title": "Section", "node_id": "n1"}]}),
        encoding="utf-8",
    )
    config = root / "config.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        "dirs:\n  papers: data/papers\nllm:\n  models:\n    - provider: stub\n      model: test\n",
        encoding="utf-8",
    )
    db_path = root / "data" / "drbrain.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    from drbrain.storage.database import Database

    seed = Database(db_path)
    seed.insert_paper("p1", "Paper", 2024, "uploaded")
    seed.commit()
    seed.close()
    source = root / "ingest.jsonl"
    source.write_text(json.dumps({"ok": True, "local_id": "p1"}) + "\n", encoding="utf-8")
    manifest = root / "build.jsonl"

    async def fake_extract(*_args, **_kwargs):
        return {
            "concepts": [{"type": "Method", "label": "method", "confidence": 0.8}],
            "relations": [],
            "merges": [],
            "corrections": [],
        }

    monkeypatch.setenv("DRBRAIN_ROOT", str(root))
    monkeypatch.setenv("BUILD_CONCURRENCY", "1")
    monkeypatch.setenv("BUILD_PAPER_TIMEOUT", "10")
    monkeypatch.setattr("drbrain.extractor.concept.build_graph_from_tree", fake_extract)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build.py",
            "--from-manifest",
            str(source),
            "--db",
            str(db_path),
            "--manifest",
            str(manifest),
        ],
    )

    assert build.main() == 0
    check = Database(db_path)
    try:
        assert (
            check.conn.execute("SELECT COUNT(*) FROM concepts WHERE local_id = 'p1'").fetchone()[0]
            == 1
        )
        assert (
            check.conn.execute("SELECT status FROM papers WHERE local_id = 'p1'").fetchone()[0]
            == "extracted"
        )
    finally:
        check.close()
    record = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
    assert record["ok"] is True
