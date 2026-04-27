"""File-based API response cache with TTL expiry."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)


class ApiCache:
    """Simple file-based JSON cache for API responses.

    Each entry is stored as a JSON file containing the data and a timestamp.
    Entries older than `ttl` seconds are considered expired and ignored.
    """

    def __init__(self, cache_dir: str, ttl: int = 86400) -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._ttl = ttl

    def get(self, key: str) -> dict | list | None:
        """Return cached data if present and not expired, else None."""
        path = self._path(key)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                entry = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"Cache read error for {key}: {e}")
            return None
        if time.time() - entry.get("cached_at", 0) > self._ttl:
            return None
        return entry.get("data")

    def set(self, key: str, data: dict | list) -> None:
        """Store data in cache with current timestamp."""
        path = self._path(key)
        try:
            with open(path, "w") as f:
                json.dump({"cached_at": time.time(), "data": data}, f)
        except OSError as e:
            log.warning(f"Cache write error for {key}: {e}")

    def delete(self, key: str) -> None:
        """Remove a single cached entry."""
        path = self._path(key)
        if path.exists():
            path.unlink(missing_ok=True)

    def clear(self) -> None:
        """Remove all cached entries."""
        for path in self._dir.glob("*.json"):
            path.unlink(missing_ok=True)

    def _path(self, key: str) -> Path:
        """Map a cache key to a file path using MD5 hash."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self._dir / f"{key_hash}.json"
