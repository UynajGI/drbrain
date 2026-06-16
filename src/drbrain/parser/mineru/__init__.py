"""MinerU PDF parser package."""

from drbrain.parser.mineru.fallback import (  # noqa: F401
    ARXIV_FILENAME,
    HEADING_SECTIONS,
    ID_PATTERN,
    INLINE_SECTION,
    MAX_CHARS,
    YEAR_PATTERN,
    filter_sections,
)
from drbrain.parser.mineru.metadata import (  # noqa: F401
    _fetch_arxiv_metadata,
    _fetch_crossref_metadata,
    _fetch_deepxiv_metadata,
    _fetch_openalex_metadata,
    _fetch_s2_metadata,
    _resolve_metadata,
    _titles_match,
)
from drbrain.parser.mineru.parser import (  # noqa: F401
    MinerUParser,
    ParsedPaper,
    PDFValidation,
    _extract_arxiv_from_filename,
    _find_cli,
    _validate_pdf,
    extract_pdf,
    normalize_arxiv,
    normalize_doi,
)
