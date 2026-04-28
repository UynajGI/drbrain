"""Concept label normalization and alias table for entity alignment."""
from __future__ import annotations

import re


_ARTICLES = {"the", "a", "an", "of", "for", "in", "on", "with", "to"}


def normalize_label(label: str) -> str:
    """Normalize a concept label for canonical matching."""
    t = label.lower().strip()
    t = re.sub(r"[^\w\s]", "", t)
    words = t.split()
    words = [w for w in words if w not in _ARTICLES]
    t = " ".join(words)
    # Simple singularization
    if t.endswith("s") and len(t) > 3:
        t = t[:-1]
    t = re.sub(r"\s+", " ", t).strip()
    return t


class AliasTable:
    """Maps label variants to canonical concept IDs."""

    def __init__(self):
        self._canonical: dict[str, str] = {}
        self._aliases: dict[str, str] = {}
        self._counter = 0

    def add_canonical(self, label: str, canonical_id: str) -> str:
        """Register a canonical label. Returns canonical_id."""
        norm = normalize_label(label)
        self._canonical[norm] = canonical_id
        return canonical_id

    def add_alias(self, variant: str, canonical_id: str) -> None:
        """Register an alias pointing to an existing canonical_id."""
        self._aliases[variant.lower().strip()] = canonical_id

    def lookup(self, label: str) -> str | None:
        """Look up canonical_id by label. Returns None if not found."""
        key = label.lower().strip()
        if key in self._aliases:
            return self._aliases[key]
        norm = normalize_label(label)
        return self._canonical.get(norm)

    def get_or_create(self, label: str) -> str:
        """Look up or create a new canonical_id."""
        existing = self.lookup(label)
        if existing is not None:
            return existing
        self._counter += 1
        new_id = f"concept_{self._counter}"
        self._canonical[normalize_label(label)] = new_id
        return new_id
