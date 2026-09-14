"""Embedding identity and caching protocol (plan T23).

A vector is only comparable to another vector produced by the *same profile*:
provider, model, dimension, preprocessing, tokenizer and sequence budget all
participate.  This module makes that identity explicit and cacheable so the
tree builder, the vector leg and query embedding can share one profile
instead of judging compatibility from a model name or a short hash.

Rules enforced here:

* The profile id covers provider, model, dimension, preprocessing, tokenizer,
  max sequence length and an optional weights/service revision.  Changing any
  of them yields a different id and therefore a cache miss.
* Same text under the same profile keeps a stable identity; identical text
  under different profiles is two different vectors.
* The in-process cache is keyed by ``(profile_id, content_hash)`` — a short
  hash alone must never decide reuse, and provenance is kept by the caller
  (node identity), not erased by content dedup.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

EMBEDDING_PROFILE_SCHEMA = "embedding-profile-v1"


def _sha256_hex(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def content_hash(text: str) -> str:
    """Full-content hash (64 hex chars) used to key embedding caches."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EmbeddingProfile:
    provider: str = "local"
    model: str = ""
    dimension: int | None = None
    max_seq_length: int | None = None
    preprocessing: str = ""
    tokenizer: str = ""
    tokenizer_revision: str = ""
    normalize: bool = True
    revision: str = ""

    def __post_init__(self) -> None:
        if not str(self.model).strip():
            raise ValueError("embedding profile requires a model name")
        if self.dimension is not None and int(self.dimension) <= 0:
            raise ValueError("dimension must be positive when set")
        if self.max_seq_length is not None and int(self.max_seq_length) <= 0:
            raise ValueError("max_seq_length must be positive when set")

    def canonical(self) -> dict[str, Any]:
        return {
            "schema": EMBEDDING_PROFILE_SCHEMA,
            "provider": self.provider,
            "model": self.model,
            "dimension": self.dimension,
            "max_seq_length": self.max_seq_length,
            "preprocessing": self.preprocessing,
            "tokenizer": self.tokenizer,
            "tokenizer_revision": self.tokenizer_revision,
            "normalize": self.normalize,
            "revision": self.revision,
        }

    def profile_id(self) -> str:
        payload = json.dumps(self.canonical(), sort_keys=True, ensure_ascii=False)
        return "emb-" + _sha256_hex(payload)[:24]

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["profile_id"] = self.profile_id()
        return payload

    def vector_key(self, text: str) -> tuple[str, str]:
        """Cache/identity key for one text under this profile."""
        return (self.profile_id(), content_hash(text))


def profile_from_config(cfg: Any) -> EmbeddingProfile:
    """Build the profile from an EmbedConfig (typed or dict-style)."""

    def get(key: str, default: Any = None) -> Any:
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    provider = str(get("provider", "local") or "local").strip().lower() or "local"
    return EmbeddingProfile(
        provider=provider,
        model=str(get("model", "") or ""),
        dimension=int(get("dim")) if get("dim") is not None else None,
        max_seq_length=int(get("max_seq_length")) if get("max_seq_length") is not None else None,
        preprocessing=str(get("preprocessing", "") or ""),
        tokenizer=str(get("tokenizer", "") or ""),
        tokenizer_revision=str(get("tokenizer_revision", "") or ""),
        normalize=bool(get("normalize", True)),
        revision=str(get("revision", "") or ""),
    )


@dataclass
class EmbeddingCacheStats:
    hits: int = 0
    misses: int = 0
    stored: int = 0


class EmbeddingCache:
    """Thread-safe (profile_id, content_hash) -> vector cache."""

    def __init__(self, *, max_entries: int = 200_000) -> None:
        self._entries: dict[tuple[str, str], tuple[float, ...]] = {}
        self._lock = threading.Lock()
        self._max_entries = int(max_entries)
        self.stats = EmbeddingCacheStats()

    def get(self, profile: EmbeddingProfile, text: str) -> tuple[float, ...] | None:
        key = profile.vector_key(text)
        with self._lock:
            vector = self._entries.get(key)
            if vector is None:
                self.stats.misses += 1
                return None
            self.stats.hits += 1
            return vector

    def put(self, profile: EmbeddingProfile, text: str, vector: Sequence[float]) -> None:
        key = profile.vector_key(text)
        with self._lock:
            if len(self._entries) >= self._max_entries and key not in self._entries:
                # Bounded, deterministic eviction: drop the oldest insertion.
                oldest = next(iter(self._entries))
                self._entries.pop(oldest, None)
            self._entries[key] = tuple(float(value) for value in vector)
            self.stats.stored += 1

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.stats = EmbeddingCacheStats()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


_GLOBAL_CACHE = EmbeddingCache()


def global_cache() -> EmbeddingCache:
    return _GLOBAL_CACHE


def text_cache_key(profile: EmbeddingProfile, text: str) -> str:
    """Single string key for external indexes (ANN metadata, tests)."""
    profile_id, digest = profile.vector_key(text)
    return f"{profile_id}:{digest}"


def cached_embed(
    texts: Sequence[str],
    *,
    profile: EmbeddingProfile,
    compute: Callable[[list[str]], list[list[float]]],
    cache: EmbeddingCache | None = None,
) -> list[list[float]]:
    """Embed ``texts`` with the profile, consulting and filling the cache.

    ``compute`` is only called for cache misses and receives exactly the
    missing texts in order; duplicate texts inside one call are computed once.
    """
    if cache is None:
        cache = global_cache()
    missing: list[str] = []
    missing_index: dict[str, int] = {}
    results: list[list[float] | None] = [None] * len(texts)
    for position, text in enumerate(texts):
        cached = cache.get(profile, text)
        if cached is not None:
            results[position] = list(cached)
            continue
        if text not in missing_index:
            missing_index[text] = len(missing)
            missing.append(text)
    if missing:
        computed = compute(list(missing))
        if len(computed) != len(missing):
            raise ValueError(f"embedder returned {len(computed)} vectors for {len(missing)} texts")
        for text, vector in zip(missing, computed):
            cache.put(profile, text, vector)
            for position, candidate in enumerate(texts):
                if candidate == text and results[position] is None:
                    results[position] = [float(value) for value in vector]
    return [vector if vector is not None else [] for vector in results]


def iter_content_hashes(texts: Iterable[str]) -> list[str]:
    return [content_hash(text) for text in texts]
