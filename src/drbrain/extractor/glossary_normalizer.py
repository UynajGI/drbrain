"""Controlled-vocabulary normalization: deterministic glossary + LLM fallback.

Generic mechanism generalized from a corpus-side batch script (v2,
2026-08-08), which normalized 19,724 distinct synthesis-method names via:

- a deterministic glossary layer: substring-containment matching against a
  curated term table (the script's ``METHOD_MAP``, 30+ entries) plus stripping
  of generic suffixes (``" method"`` / ``" technique"`` / ``" process"``);
- an LLM batch fallback: names that miss the glossary are sent to the LLM in
  batches, which replies with a strict ``{"MAPPING": {raw name: standard name}}``
  JSON object (leniently parsed, see :func:`drbrain.utils.llm_json.parse_llm_json`);
- a mapping cache (the script's ``method_map.json``): every LLM result is
  appended to a JSON file so a rerun resumes where the previous run stopped.

Only the *mechanism* is provided here — the glossary content (the actual
term → standard pairs) is the caller's responsibility. This component is
domain-agnostic and does not embed any materials-science vocabulary.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable

from loguru import logger

from drbrain.utils.checkpoint import Checkpoint
from drbrain.utils.llm_json import parse_llm_json

#: Default suffixes stripped by the deterministic layer, matching the research script.
_DEFAULT_SUFFIXES = (" method", " technique", " process")

#: Default LLM batch size, matching the research script (batch=400).
_DEFAULT_BATCH_SIZE = 400

#: source values returned by :meth:`GlossaryNormalizer.normalize`.
_GLOSSARY = "glossary"
_LLM = "llm"
_IDENTITY = "identity"


def _key(name: str) -> str:
    """Canonical key for matching/caching: strip, lowercase, drop trailing dot."""
    return name.strip().lower().rstrip(".")


def _mapping_from_llm_response(text: str) -> dict[str, str]:
    """Extract a {raw name: standard name} mapping from an LLM response.

    Accepts both the research-script envelope ``{"MAPPING": {...}}`` and a bare
    mapping object; tolerates prose/fences/truncation via ``parse_llm_json``.
    """
    data = parse_llm_json(text)
    if not isinstance(data, dict):
        return {}
    mapping = data.get("MAPPING", data)
    if not isinstance(mapping, dict):
        return {}
    return {str(k): v for k, v in mapping.items() if k and isinstance(v, str)}


class GlossaryNormalizer:
    """Deterministic term table + LLM batch fallback + resumable mapping cache.

    Pipeline per name (mirroring the research script):

    1. ``glossary`` — deterministic substring containment (longest key wins),
       then generic-suffix stripping with a glossary re-check on the stripped
       form.
    2. ``cache`` — a previously LLM-mapped name, resumed across runs from
       *cache_path*.
    3. ``llm`` — batch fallback via *llm_client* (skipped when unset).
    4. ``identity`` — no rule or LLM applied; the name is returned as-is
       (whitespace-stripped, original casing preserved).

    Args:
        glossary: substring/pattern → standard-name pairs, e.g.
            ``{"sol gel": "sol-gel", "chemical vapor deposition": "CVD"}``.
            Matching is case-insensitive; among matching keys the longest one
            wins. Keys are normalized with :func:`_key` at construction time.
        llm_client: optional callable ``(prompt: str) -> str`` returning the
            raw LLM response text; the response is parsed internally with
            ``parse_llm_json``. Wire up any concrete client through a thin
            adapter, e.g. ``lambda prompt: call_text_with_fallback(prompt,
            models, max_tokens=4096) or ""``. When unset, the LLM layers are
            skipped and the component is glossary-only.
        cache_path: optional JSON file persisting LLM mappings (incremental
            updates, atomic saves). On restart, already-mapped names are
            skipped. When unset, mappings live only in memory.
        suffixes: generic suffixes stripped when the glossary misses, e.g.
            ``"hydrothermal method"`` → ``"hydrothermal"``. Defaults to the
            corpus script's ``(" method", " technique", " process")``.

    Pattern source: the corpus synthesis-method normalization batch script.
    """

    def __init__(
        self,
        glossary: dict[str, str],
        llm_client: Callable[[str], str] | None = None,
        cache_path: str | os.PathLike[str] | None = None,
        suffixes: tuple[str, ...] | None = None,
    ) -> None:
        # Normalize + drop empty keys, then longest-first for substring matching.
        self._glossary_items: list[tuple[str, str]] = []
        for raw_key, standard in glossary.items():
            key = _key(raw_key)
            if key and standard:
                self._glossary_items.append((key, standard))
        self._glossary_items.sort(key=lambda kv: len(kv[0]), reverse=True)

        self._suffixes = suffixes if suffixes is not None else _DEFAULT_SUFFIXES
        self._llm_client = llm_client
        self._checkpoint = Checkpoint(cache_path, default={}) if cache_path else None
        state = self._checkpoint.load() if self._checkpoint else {}
        self._cache: dict[str, str] = state if isinstance(state, dict) else {}

    # ------------------------------------------------------------------ public

    def normalize(self, name: str) -> tuple[str, str]:
        """Normalize a single *name* to ``(standard, source)``.

        Lookup order: glossary → suffix stripping → cache → LLM fallback →
        identity. ``source`` is one of ``"glossary"`` (deterministic rule hit,
        including suffix stripping), ``"llm"`` (cached or fresh LLM mapping) or
        ``"identity"`` (returned as-is).
        """
        key = _key(name)
        if not key:
            return name.strip(), _IDENTITY

        standard, source = self._deterministic(key)
        if source == _GLOSSARY:
            return standard, source

        if key in self._cache:
            return self._cache[key], _LLM

        if self._llm_client is not None:
            new_map = self._llm_map([key])
            std = new_map.get(key)
            if std:
                return std, _LLM

        return name.strip(), _IDENTITY

    def normalize_batch(
        self, names: Iterable[str], batch_size: int = _DEFAULT_BATCH_SIZE
    ) -> dict[str, str]:
        """Normalize many names; returns ``{input name: standard name}``.

        Input names are used as keys (first-seen order). Names that miss the
        deterministic layer are checked against the cache, then sent to the
        LLM in *batch_size* chunks (when *llm_client* is set); every LLM result
        is appended to the cache so a later run resumes. LLM failures degrade
        to identity for the affected chunk — the batch never raises.
        """
        result: dict[str, str] = {}
        todo: list[str] = []
        for name in names:
            key = _key(name)
            if not key:
                result[name] = name.strip()
                continue
            standard, source = self._deterministic(key)
            if source == _GLOSSARY:
                result[name] = standard
                continue
            if key in self._cache:
                result[name] = self._cache[key]
                continue
            result[name] = name.strip()  # provisional identity, back-filled below
            todo.append(key)

        if todo and self._llm_client is not None:
            for start in range(0, len(todo), batch_size):
                new_map = self._llm_map(todo[start : start + batch_size])
                if not new_map:
                    continue
                for input_name, _ in list(result.items()):
                    mapped = new_map.get(_key(input_name))
                    if mapped is not None:
                        result[input_name] = mapped
        return result

    # ---------------------------------------------------------------- internal

    def _deterministic(self, key: str) -> tuple[str, str]:
        """Deterministic layer: glossary substring, then suffix stripping.

        Returns ``(standard, "glossary")`` on a hit, else ``("", "identity")``.
        """
        for pattern, standard in self._glossary_items:
            if pattern in key:
                return standard, _GLOSSARY
        for suffix in self._suffixes:
            if not key.endswith(suffix):
                continue
            stripped = key[: -len(suffix)].strip()
            if not stripped:
                continue
            for pattern, standard in self._glossary_items:
                if pattern in stripped:
                    return standard, _GLOSSARY
            return stripped, _GLOSSARY
        return "", _IDENTITY

    def _llm_map(self, names: list[str]) -> dict[str, str]:
        """Map *names* (canonical keys) to standards in one LLM call.

        Persists the new mappings into the cache (incremental, atomic save) and
        degrades to ``{}`` on any LLM or parse failure. Echoed keys from the
        LLM are reconciled against the requested names case-insensitively, and
        proposed standards are refined through the deterministic layer
        (mirroring the research script's ``normalize_method(v) or v``).
        """
        if not names or self._llm_client is None:
            return {}
        prompt = self._build_prompt(names)
        try:
            raw = self._llm_client(prompt)
        except Exception as exc:  # noqa: BLE001 - never let the LLM kill the pipeline
            logger.warning(
                "[glossary] LLM call failed, keeping %d names unmapped: %s", len(names), exc
            )
            return {}
        try:
            mapping = _mapping_from_llm_response(raw)
        except json.JSONDecodeError as exc:
            logger.warning(
                "[glossary] LLM response unparseable, keeping %d names unmapped: %s",
                len(names),
                exc,
            )
            return {}

        by_echo_key: dict[str, str] = {}
        for raw_name, standard in mapping.items():
            refined = self._refine_standard(standard)
            if refined:
                by_echo_key[_key(raw_name)] = refined
        new: dict[str, str] = {}
        for name in names:
            std = by_echo_key.get(name)
            if std is not None:
                new[name] = std
        if new:
            self._cache.update(new)
            self._save_cache()
        return new

    def _refine_standard(self, standard: str) -> str:
        """Run an LLM-proposed standard through the deterministic layer.

        Returns the deterministic hit when one applies (substring match or
        suffix strip — e.g. an LLM value like "Sol-Gel Method" collapses to
        "sol-gel"); otherwise the value is kept whitespace-stripped with its
        original casing, consistent with glossary values (authored casing, e.g.
        "CVD") and with the identity layer.
        """
        key = _key(standard)
        if not key:
            return ""
        det, _ = self._deterministic(key)
        return det if det else standard.strip()

    def _build_prompt(self, names: list[str]) -> str:
        """Build the batch normalization prompt (one name per line)."""
        return (
            "You are a controlled-vocabulary normalization assistant. Map each "
            "input name to its standard/canonical name:\n"
            "- merge synonym variants (e.g. 'PLD' and 'pulsed laser deposition' "
            "-> 'pulsed laser deposition')\n"
            "- keep names that are already standard as-is\n"
            "- drop unnecessary modifiers\n"
            "Reply with ONLY a strict JSON object, no other text: "
            '{"MAPPING": {"original name": "standard name", ...}}\n\n'
            "Names (one per line):\n" + "\n".join(names)
        )

    def _save_cache(self) -> None:
        """Persist the mapping cache atomically (no-op without cache_path)."""
        if self._checkpoint is not None:
            self._checkpoint.save(self._cache)
