"""USPTO patent search providers.

Two clients are available:
- ``uspto_odp``: Open Data Platform (requires API key)
- ``uspto_ppubs``: Patent Public Search (no key, session-based)

Both share ``PatentBase`` and ``clean_publication_number()`` from ``base.py``.
"""

from drbrain.providers.base import (
    PatentBase,
    clean_publication_number,
)

__all__ = [
    "PatentBase",
    "clean_publication_number",
]
