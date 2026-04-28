"""Tests for API cache module."""
import json
import tempfile
import time
from pathlib import Path

from drbrain.extractor.cache import ApiCache


def _make_cache(cache_dir: str, ttl: int = 86400) -> ApiCache:
    return ApiCache(cache_dir=cache_dir, ttl=ttl)


def test_cache_get_miss():
    """Cache returns None for uncached key."""
    with tempfile.TemporaryDirectory() as td:
        cache = _make_cache(td)
        assert cache.get("nonexistent") is None


def test_cache_set_and_get():
    """Cache stores and retrieves data."""
    with tempfile.TemporaryDirectory() as td:
        cache = _make_cache(td)
        cache.set("key1", {"paperId": "abc", "title": "Test"})
        result = cache.get("key1")
        assert result == {"paperId": "abc", "title": "Test"}


def test_cache_ttl_expiry():
    """Cache returns None after TTL expires."""
    with tempfile.TemporaryDirectory() as td:
        cache = _make_cache(td, ttl=1)  # 1 second TTL
        cache.set("key1", {"data": "value"})
        assert cache.get("key1") == {"data": "value"}
        time.sleep(2)
        assert cache.get("key1") is None


def test_cache_ttl_fresh():
    """Cache returns data when still within TTL."""
    with tempfile.TemporaryDirectory() as td:
        cache = _make_cache(td, ttl=300)
        cache.set("key1", {"data": "value"})
        assert cache.get("key1") == {"data": "value"}


def test_cache_overwrites_stale():
    """Cache overwrites stale entry with new data."""
    with tempfile.TemporaryDirectory() as td:
        cache = _make_cache(td, ttl=1)
        cache.set("key1", {"old": True})
        time.sleep(2)
        cache.set("key1", {"new": True})
        result = cache.get("key1")
        assert result == {"new": True}


def test_cache_creates_directory():
    """Cache creates the cache directory if it doesn't exist."""
    with tempfile.TemporaryDirectory() as td:
        cache_dir = Path(td) / "subdir" / "cache"
        assert not cache_dir.exists()
        cache = _make_cache(str(cache_dir))
        cache.set("key1", {"data": "value"})
        assert cache_dir.exists()


def test_cache_persists_across_instances():
    """Cache data persists when loading a new instance."""
    with tempfile.TemporaryDirectory() as td:
        cache1 = _make_cache(td, ttl=300)
        cache1.set("key1", {"persisted": True})

        cache2 = _make_cache(td, ttl=300)
        result = cache2.get("key1")
        assert result == {"persisted": True}


def test_cache_key_uniqueness():
    """Different keys store separate entries."""
    with tempfile.TemporaryDirectory() as td:
        cache = _make_cache(td)
        cache.set("key_a", {"a": 1})
        cache.set("key_b", {"b": 2})
        assert cache.get("key_a") == {"a": 1}
        assert cache.get("key_b") == {"b": 2}


def test_cache_delete():
    """Cache delete removes entry."""
    with tempfile.TemporaryDirectory() as td:
        cache = _make_cache(td)
        cache.set("key1", {"data": "value"})
        assert cache.get("key1") is not None
        cache.delete("key1")
        assert cache.get("key1") is None


def test_cache_clear():
    """Cache clear removes all entries."""
    with tempfile.TemporaryDirectory() as td:
        cache = _make_cache(td)
        cache.set("key1", {"a": 1})
        cache.set("key2", {"b": 2})
        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None


def test_cache_handles_non_dict_data():
    """Cache handles list data correctly."""
    with tempfile.TemporaryDirectory() as td:
        cache = _make_cache(td)
        data = [{"paperId": "a"}, {"paperId": "b"}]
        cache.set("list_key", data)
        assert cache.get("list_key") == data
