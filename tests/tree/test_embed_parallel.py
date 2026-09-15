"""Parallel vectors stage: device parsing, spool/load round-trip, worker resume."""

from __future__ import annotations

import dataclasses
import json

import pytest

from drbrain.config import EmbedConfig
from drbrain.services.canonical_content import write_canonical_content
from drbrain.storage.database import Database
from drbrain.tree.embed_parallel import (
    _run_worker,
    parallel_vectors_stage,
    parse_embed_devices,
)
from drbrain.tree.embedding_identity import profile_from_config
from drbrain.tree.vector_store import UnifiedVectorStore


def _embed_cfg(**overrides) -> EmbedConfig:
    base = dict(
        provider="local",
        model="bge-small-en-v1.5",
        cache_dir="",
        device="cuda:0",
        extra_gpus=[1],
        cpu_workers=0,
        source="modelscope",
        batch_size=8,
        dim=3,
        max_seq_length=512,
    )
    base.update(overrides)
    return EmbedConfig(**base)


def _doc(db: Database, local_id: str, sections: int = 3, words: int = 12) -> None:
    text = "".join(
        f"# Section {index}\n\n" + " ".join([f"{local_id}body{index}"] * words) + "\n\n"
        for index in range(sections)
    )
    if db.get_paper(local_id) is None:
        db.insert_paper(local_id, "T", 2024, "uploaded")
        db.commit()
    result = write_canonical_content(db, local_id, text, media_type="md", parser="test")
    assert result["ok"]


class _FakeEmbed:
    def __init__(self) -> None:
        self.calls = 0
        self.texts = 0

    def __call__(self, texts, cfg=None):
        values = list(texts)
        self.calls += 1
        self.texts += len(values)
        return [[float(len(text) % 7), float(len(text) % 5), 0.5] for text in values]


@pytest.fixture()
def fake_embed(monkeypatch):
    fake = _FakeEmbed()
    import drbrain.services.embedding as embedding

    monkeypatch.setattr(embedding, "_embed_batch", fake)
    return fake


def _worker_spawn(*, devices, db_path, cfg_path, shard_root, part_size, embed_chunk, force=False):
    """In-process stand-in for the subprocess spawner (same worker code)."""
    results = []
    for index, device in enumerate(devices):
        argv = [
            "--db-path",
            str(db_path),
            "--embed-cfg-json",
            str(cfg_path),
            "--shard-dir",
            str(shard_root),
            "--worker-index",
            str(index),
            "--worker-count",
            str(len(devices)),
            "--device",
            device,
            "--part-size",
            str(part_size),
            "--embed-chunk",
            str(embed_chunk),
        ]
        if force:
            argv.append("--force")
        results.append((device, _run_worker(argv)))
    return results


def test_parse_embed_devices_matrix() -> None:
    assert parse_embed_devices(_embed_cfg(extra_gpus=[])) == []
    assert parse_embed_devices(_embed_cfg()) == ["cuda:0", "cuda:1"]
    assert parse_embed_devices(_embed_cfg(extra_gpus=[0, 1], cpu_workers=2)) == [
        "cuda:0",
        "cuda:1",
        "cpu",
        "cpu",
    ]
    assert parse_embed_devices(_embed_cfg(provider="openai-compat")) == []
    assert parse_embed_devices(_embed_cfg(device="cpu", extra_gpus=[])) == []
    assert parse_embed_devices(_embed_cfg(device="cpu", extra_gpus=[2, 3])) == [
        "cuda:2",
        "cuda:3",
    ]


def test_parallel_stage_round_trip(tmp_path, fake_embed) -> None:
    db = Database(tmp_path / "db.sqlite")
    _doc(db, "p1")
    _doc(db, "p2")
    cfg = _embed_cfg()
    profile = profile_from_config(cfg)
    store = UnifiedVectorStore(tmp_path / "tree" / "vectors", create=True, dimension=3)
    try:
        outcome = parallel_vectors_stage(
            db,
            store,
            profile,
            cfg,
            devices=parse_embed_devices(cfg),
            spool_dir=tmp_path / "tree" / "spool",
            part_size=2,
            load_batch=3,
            spawn=_worker_spawn,
            log=lambda message: None,
        )
    finally:
        store.close()

    assert outcome["status"] == "ok"
    assert outcome["embedded"] == outcome["nodes"] > 0
    assert outcome["empty"] == 0
    ready = db.list_node_vectors(state="ready")
    assert len(ready) == outcome["embedded"]
    stored = UnifiedVectorStore(tmp_path / "tree" / "vectors", create=True, dimension=3)
    try:
        fetched = stored.get([row["node_id"] for row in ready])
    finally:
        stored.close()
    assert len(fetched) == len(ready)
    assert not (tmp_path / "tree" / "spool" / "manifest.json").exists()
    assert not list((tmp_path / "tree" / "spool").glob("shard-*/*.npz"))


def test_worker_resumes_without_re_embedding(tmp_path, fake_embed) -> None:
    db = Database(tmp_path / "db.sqlite")
    _doc(db, "p1")
    cfg = _embed_cfg()
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(dataclasses.asdict(cfg)), encoding="utf-8")
    shard_root = tmp_path / "spool"

    def run() -> int:
        return _run_worker(
            [
                "--db-path",
                str(db.path),
                "--embed-cfg-json",
                str(cfg_path),
                "--shard-dir",
                str(shard_root),
                "--worker-index",
                "0",
                "--worker-count",
                "1",
                "--device",
                "cpu",
                "--part-size",
                "2",
                "--embed-chunk",
                "2",
            ]
        )

    assert run() == 0
    first = fake_embed.texts
    assert first > 0
    assert run() == 0
    assert fake_embed.texts == first


def test_stage_refuses_mismatched_spool_dir(tmp_path, fake_embed) -> None:
    db = Database(tmp_path / "db.sqlite")
    _doc(db, "p1")
    cfg = _embed_cfg()
    profile = profile_from_config(cfg)
    store = UnifiedVectorStore(tmp_path / "tree" / "vectors", create=True, dimension=3)
    spool = tmp_path / "spool"
    spool.mkdir()
    (spool / "manifest.json").write_text(
        json.dumps(
            {
                "worker_count": 9,
                "part_size": 1,
                "force": False,
                "profile_id": "other",
                "pending_digest": "stale",
                "pending_count": 1,
            }
        ),
        encoding="utf-8",
    )
    try:
        outcome = parallel_vectors_stage(
            db,
            store,
            profile,
            cfg,
            devices=["cuda:0", "cuda:1"],
            spool_dir=spool,
            spawn=_worker_spawn,
            log=lambda message: None,
        )
    finally:
        store.close()
    assert outcome["status"] == "failed"
    assert "different pending set" in outcome["error"]


def test_prepare_vectors_dispatches_to_parallel(tmp_path, monkeypatch) -> None:
    import drbrain.tree.embed_parallel as embed_parallel
    from drbrain.tree import prepare as prepare_mod

    db = Database(tmp_path / "db.sqlite")
    _doc(db, "p1")
    cfg = _embed_cfg()
    profile = profile_from_config(cfg)
    store = UnifiedVectorStore(tmp_path / "tree" / "vectors", create=True, dimension=3)
    seen: dict = {}

    def fake_stage(db_, store_, profile_, cfg_, *, devices, force=False):
        seen["devices"] = list(devices)
        seen["force"] = force
        return {"status": "ok", "nodes": 0, "embedded": 0, "pending": 0, "empty": 0}

    monkeypatch.setattr(embed_parallel, "parallel_vectors_stage", fake_stage)
    try:
        result = prepare_mod._prepare_vectors(
            db,
            store,
            profile,
            embed=lambda texts: [],
            force=False,
            batch_size=4,
            embed_cfg=cfg,
        )
    finally:
        store.close()
    assert seen["devices"] == ["cuda:0", "cuda:1"]
    assert result["status"] == "ok"
