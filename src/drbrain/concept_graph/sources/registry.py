"""Registry of corpus source adapters.

Adapters are registered by name with a zero-arg factory. Built-in factories lazy
import their adapter modules so the package imports cleanly even if optional
dependencies for a given source are missing.
"""

from __future__ import annotations

from collections.abc import Callable

from drbrain.concept_graph.sources.base import CorpusSource

_FACTORIES: dict[str, Callable[[], CorpusSource]] = {}


def register_source(name: str, factory: Callable[[], CorpusSource]) -> None:
    """Register a corpus source factory under ``name`` (overwrites existing)."""
    _FACTORIES[name.lower()] = factory


def available_sources() -> list[str]:
    """Return the sorted names of all registered sources."""
    return sorted(_FACTORIES)


def get_source(name: str) -> CorpusSource:
    """Instantiate the corpus source registered under ``name``.

    Args:
        name: Source name (case-insensitive), e.g. ``"sciverse"`` / ``"openalex"``.

    Returns:
        A concrete :class:`CorpusSource` instance.

    Raises:
        KeyError: If no source is registered under ``name``.
    """
    key = name.lower()
    if key not in _FACTORIES:
        raise KeyError(
            f"Unknown corpus source '{name}'. Available: {', '.join(available_sources()) or '(none)'}"
        )
    return _FACTORIES[key]()


def _sciverse_factory() -> CorpusSource:
    from drbrain.concept_graph.sources.sciverse import SciverseSource
    from drbrain.config import load_config

    cfg = load_config()
    api = cfg.api
    return SciverseSource(
        token=api.sciverse_token,
        base_url=api.sciverse_base_url,
        rate_limit=api.sciverse_rate_limit,
    )


def _openalex_factory() -> CorpusSource:
    from drbrain.concept_graph.sources.openalex import OpenAlexSource
    from drbrain.config import load_config

    cfg = load_config()
    # Polite-pool mailto + API key raise OpenAlex's rate cap (~100 rps).
    return OpenAlexSource(
        token=cfg.api.openalex_token or None, mailto=cfg.api.crossref_email
    )


def _crossref_factory() -> CorpusSource:
    from drbrain.concept_graph.sources.crossref import CrossRefSource

    return CrossRefSource()


# Built-in registrations (factories lazy-import the adapter modules).
register_source("sciverse", _sciverse_factory)
register_source("openalex", _openalex_factory)
register_source("crossref", _crossref_factory)
