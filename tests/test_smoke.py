"""Smoke tests — verify basic functionality without external dependencies."""

from drbrain.auth import hash_password, verify_password
from drbrain.config import Config
from drbrain.storage.database import Database


def test_auth_hash_verify_roundtrip():
    """Password hash and verify works end-to-end."""
    pw = "test-admin-password-123"
    h = hash_password(pw)
    assert verify_password(pw, h)
    assert not verify_password("wrong-password", h)


def test_database_create_tables():
    """Database creates schema tables without errors."""
    db = Database(":memory:")
    tables = db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    table_names = [t[0] for t in tables]
    assert "papers" in table_names
    assert "concepts" in table_names
    assert "edges" in table_names
    db.close()


def test_database_insert_and_get_paper():
    """Insert and retrieve a paper."""
    db = Database(":memory:")
    db.insert_paper("test-001", "Smoke Test Paper", 2026, "uploaded")
    paper = db.get_paper("test-001")
    assert paper is not None
    assert paper["title"] == "Smoke Test Paper"
    assert paper["year"] == 2026
    db.close()


def test_config_defaults():
    """Config loads with sensible defaults."""
    config = Config()
    assert config.db.path == "data/drbrain.db"
