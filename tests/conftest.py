"""Shared pytest fixtures for DrBrain tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from drbrain.storage.database import Database


@pytest.fixture
def tmp_db():
    """Temporary SQLite database for testing. Auto-closes on teardown."""
    with tempfile.TemporaryDirectory() as td:
        db = Database(Path(td) / "test.db")
        yield db
        db.close()


@pytest.fixture
def cfg_dict():
    """Minimal config dict for tests that don't need real config."""
    return {
        "db": {"path": ":memory:"},
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
    }
