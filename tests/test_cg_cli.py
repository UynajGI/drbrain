"""CLI tests for the ``drbrain cg`` sub-app (registration + ingest)."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from pathlib import Path

from typer.testing import CliRunner

from drbrain.cli.concept_graph_commands import cg_app
from drbrain.concept_graph.sources import registry
from drbrain.concept_graph.sources.base import PaperRecord

runner = CliRunner()


class FakeSource:
    name = "fake"

    def __init__(self, records: list[PaperRecord]):
        self._records = records

    def search(
        self, query=None, *, year_from=None, year_to=None, venues=None, sort=None, limit=100
    ) -> Iterator[PaperRecord]:
        yield from self._records[:limit]

    def fetch_relations(self, unique_id: str):
        return None

    def catalog(self) -> dict:
        return {}


def _records(n: int) -> list[PaperRecord]:
    return [
        PaperRecord(unique_id=f"u{i}", title=f"T{i}", doi=f"10.1/{i}", year=2020, source="fake")
        for i in range(n)
    ]


def test_cg_ingest_dry_run(monkeypatch) -> None:
    monkeypatch.setattr(registry, "get_source", lambda name: FakeSource(_records(3)))
    result = runner.invoke(
        cg_app, ["ingest", "--dry-run", "--json", "--source", "fake"], obj={"config": {}}
    )
    assert result.exit_code == 0
    assert json.loads(result.stdout)["would_fetch"] == 3


def test_cg_ingest_writes(monkeypatch) -> None:
    monkeypatch.setattr(registry, "get_source", lambda name: FakeSource(_records(2)))
    with tempfile.TemporaryDirectory() as td:
        cfg = {"db": {"path": str(Path(td) / "test.db")}}
        result = runner.invoke(
            cg_app, ["ingest", "--json", "--source", "fake"], obj={"config": cfg}
        )
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["inserted"] == 2
        assert payload["fetched"] == 2


def test_cg_ingest_unknown_source(monkeypatch) -> None:
    def _raise(name: str):
        raise KeyError(f"Unknown corpus source '{name}'")

    monkeypatch.setattr(registry, "get_source", _raise)
    result = runner.invoke(cg_app, ["ingest", "--source", "nope"], obj={"config": {}})
    assert result.exit_code == 1


def test_cg_predict_invalid_model() -> None:
    result = runner.invoke(
        cg_app,
        [
            "predict",
            "--feat-cutoff",
            "2016",
            "--train-end",
            "2019",
            "--test-end",
            "2022",
            "--model",
            "nope",
        ],
        obj={"config": {}},
    )
    assert result.exit_code == 1
    assert "Invalid model" in result.output


def test_cg_predict_gnn_mixture_requires_embeddings() -> None:
    from drbrain.storage.database import Database

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        db = Database(db_path)
        db.conn.execute("INSERT INTO papers (local_id, title, year) VALUES ('p', 'Paper', 2010)")
        # Pre-cutoff triangle + one edge emerging in the test window.
        for u, v, year in [("a", "b", 2010), ("a", "c", 2012), ("b", "c", 2021)]:
            db.conn.execute(
                "INSERT INTO concept_cooccurrence (src_label, dst_label, year, paper_id, weight) "
                "VALUES (?, ?, ?, 'p', 1)",
                (u, v, year),
            )
        db.conn.commit()
        db.close()
        cfg = {"db": {"path": str(db_path)}}
        result = runner.invoke(
            cg_app,
            [
                "predict",
                "--feat-cutoff",
                "2016",
                "--train-end",
                "2019",
                "--test-end",
                "2022",
                "--model",
                "gnn-mixture",
            ],
            obj={"config": cfg},
        )
        assert result.exit_code == 1
        assert "embeddings" in result.output
