"""Narrow loaders for the vendored upstream algorithms (plan T02).

Rules this module exists to enforce:

* The upstream package ``__init__`` is never executed: RAPTOR's ``__init__``
  eagerly imports FAISS/T5/sentence-transformers, and PageIndex's pulls in the
  SDK client stack.  We install an alias package whose ``__path__`` points at
  the vendored directory and import only the submodules we audited.
* Missing optional dependencies surface as :class:`MissingTreeDependencyError`
  with the install command, not as a bare ImportError mid-build.
* No loader here touches ``LocalAPI``/``DocStore``/pickle persistence.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any

from drbrain.tree.vendor import REPO_ROOT, VendorSourceError, vendor_path, verify_vendor

RAPTOR_ALIAS = "drbrain_vendored_raptor"
PAGEINDEX_ALIAS = "drbrain_vendored_pageindex"

_TREE_EXTRA_HINT = "install the tree profile: uv sync --extra tree"

#: Modules the adapters may load (an allowlist: a typo or a path outside the
#: audited surface fails loudly instead of importing SDK machinery).
RAPTOR_MODULES = ("tree_structures", "cluster_utils", "utils")
PAGEINDEX_MODULES = (
    "utils",
    "tree_optimize",
    "page_index_md",
    "page_index_classic",
)


class MissingTreeDependencyError(RuntimeError):
    """A vendored adapter needs an optional dependency that is not installed."""


def _install_alias_package(alias: str, package_dir: Path) -> types.ModuleType:
    existing = sys.modules.get(alias)
    if existing is not None:
        return existing
    module = types.ModuleType(alias)
    module.__path__ = [str(package_dir)]  # type: ignore[attr-defined]
    module.__package__ = alias
    module.__file__ = str(package_dir / "__init__.py")
    module.__doc__ = (
        f"Alias package for vendored upstream source at {package_dir} "
        "(upstream __init__ intentionally not executed)"
    )
    sys.modules[alias] = module
    return module


def _load(package: str, module_name: str, *, verified: bool) -> Any:
    if not verified:
        verify_vendor(REPO_ROOT)
    root = vendor_path(package, REPO_ROOT) / ("pageindex" if package == "pageindex" else "raptor")
    alias = PAGEINDEX_ALIAS if package == "pageindex" else RAPTOR_ALIAS
    _install_alias_package(alias, root)
    try:
        return importlib.import_module(f"{alias}.{module_name}")
    except ImportError as exc:  # pragma: no cover - exercised with blocked imports
        raise MissingTreeDependencyError(
            f"loading vendored {package}.{module_name} failed ({exc}); {_TREE_EXTRA_HINT}"
        ) from exc


_verified = False


def _ensure_verified() -> None:
    global _verified
    if not _verified:
        verify_vendor(REPO_ROOT)
        _verified = True


def load_raptor_module(module_name: str) -> Any:
    """Load one vendored RAPTOR module without running its package ``__init__``."""
    if module_name not in RAPTOR_MODULES:
        raise KeyError(
            f"raptor module {module_name!r} is not on the audited allowlist {RAPTOR_MODULES}"
        )
    _ensure_verified()
    return _load("raptor", module_name, verified=True)


def load_pageindex_module(module_name: str) -> Any:
    """Load one vendored PageIndex module without running its package ``__init__``."""
    if module_name not in PAGEINDEX_MODULES:
        raise KeyError(
            f"pageindex module {module_name!r} is not on the audited allowlist {PAGEINDEX_MODULES}"
        )
    _ensure_verified()
    return _load("pageindex", module_name, verified=True)


def reset_caches() -> None:
    """Drop loader state (tests only)."""
    global _verified
    _verified = False
    for alias in (RAPTOR_ALIAS, PAGEINDEX_ALIAS):
        for name in list(sys.modules):
            if name == alias or name.startswith(alias + "."):
                del sys.modules[name]


__all__ = [
    "MissingTreeDependencyError",
    "VendorSourceError",
    "reset_caches",
    "load_pageindex_module",
    "load_raptor_module",
]
