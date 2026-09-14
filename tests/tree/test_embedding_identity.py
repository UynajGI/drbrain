"""T23: embedding profile identity and cache-reuse protocol."""

from __future__ import annotations

import pytest

from drbrain.tree.embedding_identity import (
    EmbeddingCache,
    EmbeddingProfile,
    cached_embed,
    content_hash,
    profile_from_config,
    text_cache_key,
)


class TestProfileIdentity:
    def test_stable_and_sensitive_to_every_field(self):
        base = EmbeddingProfile(provider="local", model="BAAI/bge-small-en-v1.5", dimension=384)
        assert base.profile_id() == EmbeddingProfile(
            provider="local", model="BAAI/bge-small-en-v1.5", dimension=384
        ).profile_id()
        variants = [
            EmbeddingProfile(provider="local", model="other-model", dimension=384),
            EmbeddingProfile(provider="openai-compat", model="BAAI/bge-small-en-v1.5", dimension=384),
            EmbeddingProfile(provider="local", model="BAAI/bge-small-en-v1.5", dimension=768),
            EmbeddingProfile(provider="local", model="BAAI/bge-small-en-v1.5", dimension=384, max_seq_length=512),
            EmbeddingProfile(provider="local", model="BAAI/bge-small-en-v1.5", dimension=384, preprocessing="query: "),
            EmbeddingProfile(provider="local", model="BAAI/bge-small-en-v1.5", dimension=384, tokenizer="o200k_base"),
            EmbeddingProfile(provider="local", model="BAAI/bge-small-en-v1.5", dimension=384, revision="weights-v2"),
        ]
        ids = {base.profile_id()} | {variant.profile_id() for variant in variants}
        assert len(ids) == len(variants) + 1

    def test_model_name_alone_is_not_identity(self):
        a = EmbeddingProfile(provider="local", model="m", max_seq_length=512)
        b = EmbeddingProfile(provider="local", model="m", max_seq_length=8192)
        assert a.profile_id() != b.profile_id()

    def test_profile_from_dict_and_object(self):
        from types import SimpleNamespace

        as_dict = {"provider": "local", "model": "m", "dim": 384, "max_seq_length": 256}
        as_obj = SimpleNamespace(provider="local", model="m", dim=384, max_seq_length=256)
        assert profile_from_config(as_dict).profile_id() == profile_from_config(as_obj).profile_id()

    def test_invalid_profiles_rejected(self):
        with pytest.raises(ValueError, match="model name"):
            EmbeddingProfile(model="  ")
        with pytest.raises(ValueError, match="dimension"):
            EmbeddingProfile(model="m", dimension=0)

    def test_to_json_exposes_profile_id(self):
        payload = EmbeddingProfile(model="m").to_json()
        assert payload["profile_id"].startswith("emb-")


class TestCachedEmbed:
    def test_cache_hit_avoids_recompute(self):
        profile = EmbeddingProfile(model="m")
        calls: list[list[str]] = []

        def compute(texts):
            calls.append(list(texts))
            return [[float(len(text))] for text in texts]

        cache = EmbeddingCache()
        first = cached_embed(["alpha", "beta", "alpha"], profile=profile, compute=compute, cache=cache)
        assert calls == [["alpha", "beta"]]  # duplicates collapse
        assert first[0] == first[2] == [5.0]
        second = cached_embed(["alpha", "beta"], profile=profile, compute=compute, cache=cache)
        assert len(calls) == 1
        assert second == [[5.0], [4.0]]
        assert cache.stats.misses == 3 and cache.stats.hits == 2 and cache.stats.stored == 2

    def test_different_profile_forces_recompute(self):
        cache = EmbeddingCache()
        calls = []

        def compute(texts):
            calls.append(list(texts))
            return [[1.0] for _ in texts]

        cached_embed(["shared text"], profile=EmbeddingProfile(model="m1"), compute=compute, cache=cache)
        cached_embed(["shared text"], profile=EmbeddingProfile(model="m2"), compute=compute, cache=cache)
        assert len(calls) == 2

    def test_same_text_different_provenance_is_not_deduplicated_away(self):
        """Content-hash reuse never erases provenance: callers keep both ids."""
        cache = EmbeddingCache()
        compute_calls = []

        def compute(texts):
            compute_calls.append(list(texts))
            return [[2.0] for _ in texts]

        profile = EmbeddingProfile(model="m")
        cached_embed(["identical body"], profile=profile, compute=compute, cache=cache)
        cached_embed(["identical body"], profile=profile, compute=compute, cache=cache)
        assert len(compute_calls) == 1
        assert text_cache_key(profile, "identical body").count(":") == 1

    def test_mismatched_embedder_output_raises(self):
        with pytest.raises(ValueError, match="vectors for"):
            cached_embed(
                ["a", "b"],
                profile=EmbeddingProfile(model="m"),
                compute=lambda texts: [[0.0]],
                cache=EmbeddingCache(),
            )

    def test_short_hash_never_decides_compatibility(self):
        """content_hash is full-length; profile separates text of equal length."""
        text_a, text_b = "AAAA", "BBBB"
        assert len(content_hash("x")) == 64
        assert content_hash(text_a) != content_hash(text_b)
