"""File-based API response cache with TTL expiry."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

from drbrain.security import configured_secret_values, redact_sensitive, safe_error

log = logging.getLogger(__name__)


class ApiCache:
    """Simple file-based JSON cache for API responses.

    Each entry is stored as a JSON file containing the data and a timestamp.
    Entries older than `ttl` seconds are considered expired and ignored.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        ttl: int = 86400,
        *,
        secrets: Sequence[str | None] = (),
    ) -> None:
        try:
            raw_directory = os.fspath(cache_dir)
        except (TypeError, ValueError, OSError) as exc:
            raise ValueError("API cache directory must be a local path") from exc
        if not raw_directory:
            raise ValueError("API cache directory must not be empty")
        from drbrain.runtime import _is_uri

        if _is_uri(raw_directory):
            raise ValueError("API cache directory must be a local filesystem path")
        directory = Path(raw_directory).expanduser()
        # API response caches are mutable runtime state.  When a runtime
        # selector is active, reject external paths and lexical symlink aliases
        # before mkdir can redirect writes into another worktree.  Direct
        # library callers without a selector retain explicit shared-cache use.
        if "DRBRAIN_ROOT" in os.environ or "DRBRAIN_RUNTIME_ROOT" in os.environ:
            from drbrain.runtime import RuntimeContext

            runtime = RuntimeContext.create()
            directory = runtime.assert_within_root(directory, label="API cache directory")
        else:
            from drbrain.runtime import _first_symlink_component

            alias = _first_symlink_component(directory)
            if alias is not None:
                raise ValueError(f"API cache directory must not contain a symlink: {alias}")
            directory = directory.resolve()
        if directory.exists() and not directory.is_dir():
            raise ValueError(f"API cache directory is not a directory: {directory}")
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"API cache directory is not a real directory: {directory}")
        self._dir = directory
        self._ttl = ttl
        self._secrets = tuple(
            dict.fromkeys(
                value
                for value in (*configured_secret_values(os.environ), *secrets)
                if isinstance(value, str) and value
            )
        )

    def get(self, key: str) -> dict | list | None:
        """Return cached data if present and not expired, else None."""
        path = self._path(key)
        if path.is_symlink() or not path.is_file():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                entry = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Cache read error for %s: %s", key, safe_error(e))
            return None
        if not isinstance(entry, dict):
            return None
        try:
            cached_at = float(entry.get("cached_at", 0))
        except (TypeError, ValueError):
            return None
        if time.time() - cached_at > self._ttl:
            return None
        return entry.get("data")

    def set(self, key: str, data: dict | list) -> None:
        """Store data in cache with current timestamp."""
        path = self._path(key)
        if path.is_symlink() or (path.exists() and not path.is_file()):
            log.warning("Cache write skipped for unsafe path %s", path)
            return
        temporary_name: str | None = None
        try:
            fd, temporary_name = tempfile.mkstemp(prefix=".cache-", suffix=".tmp", dir=self._dir)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                # Provider error payloads occasionally echo request metadata.
                # Cache only the same credential-free projection used by
                # durable session/ledger writers; the caller's object is not
                # mutated because ``redact_sensitive`` returns a copy.
                projected = redact_sensitive(data)
                if self._secrets:
                    projected = _replace_secrets(projected, self._secrets)
                json.dump({"cached_at": time.time(), "data": projected}, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary_name, path)
            temporary_name = None
        except OSError as e:
            log.warning("Cache write error for %s: %s", key, safe_error(e))
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except OSError:
                    pass

    def delete(self, key: str) -> None:
        """Remove a single cached entry."""
        path = self._path(key)
        if path.is_file() and not path.is_symlink():
            path.unlink(missing_ok=True)

    def clear(self) -> None:
        """Remove all cached entries."""
        for path in self._dir.glob("*.json"):
            if path.is_file() and not path.is_symlink():
                path.unlink(missing_ok=True)

    def _path(self, key: str) -> Path:
        """Map a cache key to a file path using MD5 hash."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self._dir / f"{key_hash}.json"


def _replace_secrets(value: object, secrets: tuple[str, ...]) -> object:
    """Replace configured opaque credentials even in unlabelled payload fields."""
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {key: _replace_secrets(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_secrets(item, secrets) for item in value]
    if isinstance(value, tuple):
        return tuple(_replace_secrets(item, secrets) for item in value)
    return value
