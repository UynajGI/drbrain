"""Tests for the glossary + LLM-fallback normalizer (GlossaryNormalizer)."""

import json

from drbrain.extractor.glossary_normalizer import GlossaryNormalizer


class _CountingStub:
    """LLM stub: returns a fixed MAPPING JSON, counts calls, records prompts."""

    def __init__(self, mapping=None, *, raw_response=None, raise_error=False):
        self.mapping = mapping or {}
        self.raw_response = raw_response
        self.raise_error = raise_error
        self.calls = 0
        self.prompts = []

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        if self.raise_error:
            raise RuntimeError("llm down")
        if self.raw_response is not None:
            return self.raw_response
        return json.dumps({"MAPPING": self.mapping})


GLOSSARY = {
    "sol gel": "sol-gel",
    "sol-gel method": "sol-gel",
    "chemical vapor deposition": "CVD",
    "hydrothermal method": "hydrothermal",
}


# -- glossary substring matching --


class TestGlossary:
    def test_substring_hit(self):
        n = GlossaryNormalizer(GLOSSARY)
        assert n.normalize("sol gel") == ("sol-gel", "glossary")

    def test_substring_hit_case_insensitive(self):
        n = GlossaryNormalizer(GLOSSARY)
        assert n.normalize("Sol Gel") == ("sol-gel", "glossary")

    def test_substring_inside_longer_name(self):
        n = GlossaryNormalizer(GLOSSARY)
        assert n.normalize("a novel sol gel coating") == ("sol-gel", "glossary")

    def test_longest_key_wins(self):
        n = GlossaryNormalizer({"sol gel": "sol-gel", "sol gel method": "sol-gel-method"})
        assert n.normalize("sol gel method") == ("sol-gel-method", "glossary")

    def test_trailing_dot_and_whitespace_stripped(self):
        n = GlossaryNormalizer(GLOSSARY)
        assert n.normalize("Chemical Vapor Deposition.") == ("CVD", "glossary")

    def test_empty_name(self):
        n = GlossaryNormalizer(GLOSSARY)
        assert n.normalize("") == ("", "identity")
        assert n.normalize("   ") == ("", "identity")


# -- suffix stripping --


class TestSuffixStripping:
    def test_suffix_stripped_when_glossary_misses(self):
        n = GlossaryNormalizer({"sol gel": "sol-gel"})
        assert n.normalize("hydrothermal method") == ("hydrothermal", "glossary")
        assert n.normalize("spray coating technique") == ("spray coating", "glossary")

    def test_suffix_strip_then_glossary_reeheck(self):
        n = GlossaryNormalizer({"wet chemical": "wet-chemical"})
        assert n.normalize("wet chemical process") == ("wet-chemical", "glossary")

    def test_glossary_wins_over_suffix(self):
        # "sol gel method" matches the glossary substring before suffix logic runs.
        n = GlossaryNormalizer(GLOSSARY)
        assert n.normalize("sol gel method") == ("sol-gel", "glossary")

    def test_custom_suffixes(self):
        n = GlossaryNormalizer({}, suffixes=(" reaction",))
        assert n.normalize("aldol reaction") == ("aldol", "glossary")
        # Default suffixes no longer apply when overridden.
        assert n.normalize("aldol method") == ("aldol method", "identity")

    def test_suffix_only_at_end(self):
        n = GlossaryNormalizer({})
        assert n.normalize("method coating") == ("method coating", "identity")


# -- LLM fallback --


class TestLLMFallback:
    def test_miss_routes_to_llm(self):
        stub = _CountingStub({"pulsed laser deposition": "pulsed laser deposition"})
        n = GlossaryNormalizer({}, llm_client=stub)
        assert n.normalize("pulsed laser deposition") == ("pulsed laser deposition", "llm")
        assert stub.calls == 1
        assert "pulsed laser deposition" in stub.prompts[0]

    def test_llm_value_refined_through_glossary(self):
        # LLM proposes a value that itself matches the glossary.
        stub = _CountingStub({"vapour deposition": "chemical vapor deposition"})
        n = GlossaryNormalizer(GLOSSARY, llm_client=stub)
        assert n.normalize("vapour deposition") == ("CVD", "llm")

    def test_llm_response_with_prose_and_fence(self):
        stub = _CountingStub(
            raw_response='Sure! ```json\n{"MAPPING": {"electro spinning": "electrospinning"}}\n```'
        )
        n = GlossaryNormalizer({}, llm_client=stub)
        assert n.normalize("electro spinning") == ("electrospinning", "llm")

    def test_llm_echo_key_case_insensitive_reconcile(self):
        # LLM echoes the name with different casing; the mapping still applies.
        stub = _CountingStub({"Pulsed Laser Deposition": "PLD"})
        n = GlossaryNormalizer({}, llm_client=stub)
        assert n.normalize("pulsed laser deposition") == ("PLD", "llm")

    def test_llm_omits_name_falls_back_to_identity(self):
        stub = _CountingStub({"some other name": "x"})
        n = GlossaryNormalizer({}, llm_client=stub)
        assert n.normalize("weird name") == ("weird name", "identity")

    def test_llm_failure_degrades_to_identity(self):
        stub = _CountingStub(raise_error=True)
        n = GlossaryNormalizer({}, llm_client=stub)
        assert n.normalize("weird item") == ("weird item", "identity")

    def test_llm_garbage_response_degrades_to_identity(self):
        stub = _CountingStub(raw_response="this is not json at all")
        n = GlossaryNormalizer({}, llm_client=stub)
        assert n.normalize("weird item") == ("weird item", "identity")


# -- no LLM: pure glossary --


class TestGlossaryOnly:
    def test_miss_returns_identity_with_original_casing(self):
        n = GlossaryNormalizer(GLOSSARY)
        assert n.normalize("Pulsed Laser Deposition") == ("Pulsed Laser Deposition", "identity")

    def test_batch_without_llm(self):
        n = GlossaryNormalizer(GLOSSARY)
        out = n.normalize_batch(["sol gel", "mystery x", "hydrothermal method"])
        assert out == {
            "sol gel": "sol-gel",
            "mystery x": "mystery x",
            "hydrothermal method": "hydrothermal",
        }


# -- mapping cache / resume --


class TestCache:
    def test_second_call_reuses_cache_without_llm(self, tmp_path):
        cache = tmp_path / "mapping.json"
        stub = _CountingStub({"pulsed laser deposition": "PLD"})
        n = GlossaryNormalizer({}, llm_client=stub, cache_path=cache)
        assert n.normalize("pulsed laser deposition") == ("PLD", "llm")
        assert stub.calls == 1
        # Same instance: cache hit, no further LLM call.
        assert n.normalize("pulsed laser deposition") == ("PLD", "llm")
        assert stub.calls == 1

    def test_new_instance_resumes_from_cache(self, tmp_path):
        cache = tmp_path / "mapping.json"
        stub1 = _CountingStub({"pulsed laser deposition": "PLD"})
        n1 = GlossaryNormalizer({}, llm_client=stub1, cache_path=cache)
        n1.normalize("pulsed laser deposition")
        assert stub1.calls == 1

        # Fresh instance, same cache: skips the LLM entirely.
        stub2 = _CountingStub(raise_error=True)  # would raise if called
        n2 = GlossaryNormalizer({}, llm_client=stub2, cache_path=cache)
        assert n2.normalize("pulsed laser deposition") == ("PLD", "llm")
        assert stub2.calls == 0

    def test_cache_file_holds_mapping(self, tmp_path):
        cache = tmp_path / "mapping.json"
        n = GlossaryNormalizer(
            {}, llm_client=_CountingStub({"foo bar": "foobar"}), cache_path=cache
        )
        n.normalize("foo bar")
        assert json.loads(cache.read_text()) == {"foo bar": "foobar"}

    def test_corrupt_cache_file_resets_to_empty(self, tmp_path):
        cache = tmp_path / "mapping.json"
        cache.write_text("{not valid json", encoding="utf-8")
        n = GlossaryNormalizer(GLOSSARY, llm_client=_CountingStub({}), cache_path=cache)
        assert n.normalize("sol gel") == ("sol-gel", "glossary")

    def test_batch_resumes_across_instances(self, tmp_path):
        cache = tmp_path / "mapping.json"
        names = ["sol gel", "foo bar", "baz qux", "quux corge"]
        stub1 = _CountingStub({"foo bar": "foobar", "baz qux": "bazqux", "quux corge": "quuxcorge"})
        n1 = GlossaryNormalizer(GLOSSARY, llm_client=stub1, cache_path=cache)
        out1 = n1.normalize_batch(names)
        assert out1 == {
            "sol gel": "sol-gel",
            "foo bar": "foobar",
            "baz qux": "bazqux",
            "quux corge": "quuxcorge",
        }

        stub2 = _CountingStub(raise_error=True)
        n2 = GlossaryNormalizer(GLOSSARY, llm_client=stub2, cache_path=cache)
        out2 = n2.normalize_batch(names)
        assert out2 == out1
        assert stub2.calls == 0


# -- batch interface --


class TestBatch:
    def test_batch_routes_only_misses_to_llm(self):
        stub = _CountingStub({"pulsed laser deposition": "PLD"})
        n = GlossaryNormalizer(GLOSSARY, llm_client=stub)
        out = n.normalize_batch(["sol gel", "pulsed laser deposition", "hydrothermal method"])
        assert out == {
            "sol gel": "sol-gel",
            "pulsed laser deposition": "PLD",
            "hydrothermal method": "hydrothermal",
        }
        assert stub.calls == 1
        # Glossary-hit names never reach the LLM prompt.
        assert "sol gel" not in stub.prompts[0]
        assert "pulsed laser deposition" in stub.prompts[0]

    def test_batch_respects_batch_size_chunking(self):
        stub = _CountingStub(
            {"a b": "ab", "c d": "cd", "e f": "ef"},
        )
        n = GlossaryNormalizer({}, llm_client=stub)
        out = n.normalize_batch(["a b", "c d", "e f"], batch_size=1)
        assert out == {"a b": "ab", "c d": "cd", "e f": "ef"}
        assert stub.calls == 3

    def test_batch_llm_failure_keeps_identity(self):
        stub = _CountingStub(raise_error=True)
        n = GlossaryNormalizer({}, llm_client=stub)
        out = n.normalize_batch(["foo bar", "baz qux"])
        assert out == {"foo bar": "foo bar", "baz qux": "baz qux"}

    def test_batch_preserves_input_name_keys(self):
        stub = _CountingStub({"pulsed laser deposition": "PLD"})
        n = GlossaryNormalizer({}, llm_client=stub)
        out = n.normalize_batch(["Pulsed Laser Deposition"])
        assert out == {"Pulsed Laser Deposition": "PLD"}
