"""Shared base classes and utilities for USPTO patent search clients.

Both ``uspto_odp`` (Open Data Platform) and ``uspto_ppubs`` (Publications)
clients share common patterns: patent data classes with ``to_dict()`` /
``google_patents_url()``, publication number normalization, and a public
``search_patents()`` API. This module extracts the shared logic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class PatentBase:
    """Base class for patent result records across USPTO clients.

    Subclasses add API-specific fields. The shared methods
    ``google_patents_url()`` and ``_common_dict()`` reduce duplication.
    """

    publication_number: str = ""
    application_number: str = ""
    title: str = ""

    def google_patents_url(self) -> str:
        """Link to Google Patents viewer."""
        if self.publication_number:
            return f"https://patents.google.com/patent/{self.publication_number}/en"
        if self.application_number:
            return f"https://data.uspto.gov/api/v1/patent/applications/{self.application_number}"
        return ""

    def _common_dict(self) -> dict:
        """Fields shared by all patent result types."""
        return {
            "publication_number": self.publication_number,
            "application_number": self.application_number,
            "title": self.title,
        }


def clean_publication_number(raw: str) -> str:
    """Normalize a patent publication number.

    Removes whitespace, converts to uppercase, strips non-alphanumeric
    characters. Used by both clients.
    """
    if not raw:
        return ""
    cleaned = raw.strip().upper().replace(" ", "")
    # Keep only alphanumeric + hyphen
    cleaned = re.sub(r"[^A-Z0-9]", "", cleaned)
    return cleaned


# Re-export public API from submodules for convenience
def __getattr__(name: str):
    """Lazy attribute access for submodule re-exports."""
    if name == "PatentResult":
        from drbrain.providers.uspto_odp import PatentResult

        return PatentResult
    if name == "PpubsPatent":
        from drbrain.providers.uspto_ppubs import PpubsPatent

        return PpubsPatent
    if name == "search_patents_odp":
        from drbrain.providers.uspto_odp import search_patents

        return search_patents
    if name == "search_patents_ppubs":
        from drbrain.providers.uspto_ppubs import search_patents  # type: ignore[assignment]

        return search_patents
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["PatentBase", "clean_publication_number"]
