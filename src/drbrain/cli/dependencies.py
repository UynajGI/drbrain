"""Optional dependency checking with friendly install hints."""

from __future__ import annotations

import sys

from loguru import logger

_INSTALL_HINTS: dict[str, str] = {
    "sentence_transformers": "pip install sentence-transformers",
    "faiss": "pip install faiss-cpu",
    "bertopic": "pip install bertopic",
    "numpy": "pip install numpy",
    "pymupdf": "pip install pymupdf",
    "pymupdf4llm": "pip install pymupdf4llm",
    "scipy": "pip install scipy",
    "streamlit": "pip install streamlit",
    "pyalex": "pip install pyalex",
    "arxiv": "pip install arxiv",
    "pikepdf": "pip install pikepdf",
    "litellm": "pip install litellm",
    "typer": "pip install typer",
    "rich": "pip install rich",
    "pyyaml": "pip install pyyaml",
    "pydantic": "pip install pydantic",
}


def check_import_error(e: ImportError) -> None:
    """Log a friendly message and exit when an optional dependency is missing.

    Call this inside except ImportError blocks:
        try:
            import optional_lib
        except ImportError as e:
            check_import_error(e)
    """
    # Extract the module name from the ImportError
    name = getattr(e, "name", None) or str(e).split()[-1].strip("'\"")

    # Get the top-level package name
    top_level = name.split(".")[0] if name else ""

    hint = _INSTALL_HINTS.get(top_level) or _INSTALL_HINTS.get(name, "")
    if hint:
        logger.error(f"Missing dependency: {name}\n  Install: {hint}")
    else:
        logger.error(f"Missing dependency: {name}")

    sys.exit(1)
