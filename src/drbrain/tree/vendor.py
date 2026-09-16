"""Vendor verification for the fixed upstream sources (plan T01).

The two upstream repositories are pinned as git submodules; this module
verifies the *checked-out* commit, the license file and the exact functions
the adapters rely on.  Checks are offline (local git metadata plus file
reads) and fail closed with an actionable message: a wrong revision must
never silently degrade into another algorithm.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


class VendorSourceError(RuntimeError):
    """The vendored upstream source is missing, wrong, or incomplete."""


@dataclass(frozen=True)
class VendorSpec:
    name: str
    relative_path: str
    commit: str
    license_file: str
    #: file (relative to the vendor dir) -> symbols that must be defined there
    required_symbols: dict[str, tuple[str, ...]] = field(default_factory=dict)


PAGEINDEX_SPEC = VendorSpec(
    name="pageindex",
    relative_path="vendor/pageindex",
    commit="bfbd4b305cd3f0f39a5094779627a2c93634ad79",
    license_file="LICENSE",
    required_symbols={
        "pageindex/page_index_md.py": ("md_to_tree",),
        "pageindex/page_index_classic.py": ("page_index_main",),
        "pageindex/utils.py": ("count_tokens", "generate_summaries_for_structure"),
        "pageindex/tree_optimize.py": ("merge_tree",),
    },
)

RAPTOR_SPEC = VendorSpec(
    name="raptor",
    relative_path="vendor/raptor",
    commit="7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767",
    license_file="LICENSE.txt",
    required_symbols={
        "raptor/cluster_utils.py": (
            "global_cluster_embeddings",
            "local_cluster_embeddings",
            "get_optimal_clusters",
            "GMM_cluster",
            "perform_clustering",
            "RAPTOR_Clustering",
        ),
        "raptor/tree_structures.py": ("Node", "Tree"),
        "raptor/utils.py": ("get_node_list", "get_children", "get_text"),
    },
)

VENDOR_SPECS: tuple[VendorSpec, ...] = (PAGEINDEX_SPEC, RAPTOR_SPEC)


def _symbol_defined(source: str, symbol: str) -> bool:
    pattern = re.compile(rf"^(?:async\s+)?(?:def|class)\s+{re.escape(symbol)}\b", re.MULTILINE)
    return bool(pattern.search(source))


def _head_commit(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def verify_spec(spec: VendorSpec, root: Path = REPO_ROOT) -> list[str]:
    """Return a list of human-readable problems (empty list means verified)."""
    problems: list[str] = []
    vendor_dir = root / spec.relative_path
    if not vendor_dir.is_dir():
        return [
            f"{spec.name}: {spec.relative_path} is missing — "
            "run `git submodule update --init vendor`"
        ]
    head = _head_commit(vendor_dir)
    if head is None:
        problems.append(
            f"{spec.name}: {spec.relative_path} has no git checkout — "
            "re-clone with submodules (`git submodule update --init vendor`)"
        )
    elif head != spec.commit:
        problems.append(
            f"{spec.name}: expected commit {spec.commit}, found {head} — "
            "checkout the pinned revision (do not track main)"
        )
    license_path = vendor_dir / spec.license_file
    if not license_path.is_file() or license_path.stat().st_size < 100:
        problems.append(f"{spec.name}: license file {spec.license_file} missing or empty")
    for relative, symbols in spec.required_symbols.items():
        source_path = vendor_dir / relative
        if not source_path.is_file():
            problems.append(f"{spec.name}: required source {relative} missing")
            continue
        source = source_path.read_text(encoding="utf-8", errors="replace")
        for symbol in symbols:
            if not _symbol_defined(source, symbol):
                problems.append(f"{spec.name}: {relative} does not define {symbol}")
    return problems


def verify_vendor(root: Path = REPO_ROOT) -> dict[str, str]:
    """Verify every vendored source; raise :class:`VendorSourceError` on failure."""
    problems: list[str] = []
    for spec in VENDOR_SPECS:
        problems.extend(verify_spec(spec, root))
    if problems:
        raise VendorSourceError(
            "vendored upstream sources failed verification:\n  - " + "\n  - ".join(problems)
        )
    return {spec.name: spec.commit for spec in VENDOR_SPECS}


def vendor_path(name: str, root: Path = REPO_ROOT) -> Path:
    for spec in VENDOR_SPECS:
        if spec.name == name:
            return root / spec.relative_path
    raise KeyError(f"unknown vendored source {name!r}")
