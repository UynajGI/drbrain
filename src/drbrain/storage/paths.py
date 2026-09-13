"""Centralized path accessors for paper directories.

``papers.local_id`` is a database identifier and is intentionally allowed to
contain characters such as the slash in a DOI.  A filesystem component has a
different contract: it must be one path component and must not be able to
escape the configured papers root.  Evidence locators also use a colon as the
``paper_id:node_id`` separator, so local IDs reject colons at the write
boundary.  The helpers in this module keep those identifiers separate while
retaining read compatibility with the layouts written by older corpus scripts.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import quote, unquote

# Keep generated names portable across POSIX and Windows.  ``%`` is excluded
# deliberately: it is the escape marker used by ``paper_fs_key`` and therefore
# makes the encoding bijective (an encoded key can never be mistaken for an
# unencoded safe local_id).
_SAFE_LOCAL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
# DOI registrant prefixes are normally 4–9 digits, but test fixtures and a few
# legacy providers emit shorter valid-looking prefixes; accepting any digits
# keeps migration lookup lossless without treating arbitrary names as DOI roots.
_DOI_COMPONENT_RE = re.compile(r"^10\.[0-9]+$", re.IGNORECASE)
_MAX_COMPONENT_BYTES = 240  # leave room below common NAME_MAX=255 limits
_WINDOWS_RESERVED_RE = re.compile(r"(?i)^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?$")


def _validate_local_id(local_id: str) -> str:
    """Return a valid local ID or fail before touching the filesystem."""
    if not isinstance(local_id, str):
        raise ValueError("paper local_id must be a string")
    if local_id != local_id.strip():
        # Whitespace is identity-bearing data in SQLite.  Silently trimming it
        # would make two DB rows address the same filesystem directory.
        raise ValueError("paper local_id must not have surrounding whitespace")
    value = local_id
    if not value or "\x00" in value:
        raise ValueError("paper local_id must be a non-empty string without NUL bytes")
    # Evidence locators use ``paper_id:node_id``.  Allowing a colon in the
    # paper identity would make that durable format ambiguous and could bind
    # evidence to the wrong paper after a round trip.
    if ":" in value:
        raise ValueError("paper local_id must not contain ':'")
    return value


def _validate_paper_candidate(
    papers_root: Path,
    candidate: Path,
    *,
    require_directory: bool = False,
) -> Path:
    """Validate a paper path before it is used as a read or write target.

    ``Path.resolve()`` alone is insufficient for a missing path: an existing
    symlink in an intermediate component can still redirect a later
    ``mkdir()`` outside the corpus root.  Walk the lexical components first,
    reject symlink aliases, then check the resolved candidate containment.
    """
    root = Path(papers_root).expanduser()
    # ``root.resolve()`` would erase this identity before the caller has a
    # chance to notice it.  A symlinked corpus root can redirect every newly
    # created paper directory outside the selected runtime namespace, so fail
    # closed even when the link happens to point at an otherwise valid folder.
    if root.is_symlink():
        raise ValueError(f"papers root must not be a symlink: {root}")
    # A non-symlink leaf can still sit below a symlinked parent (``root/link``
    # -> another checkout).  Use the runtime lexical walker rather than
    # ``abspath``: normalizing ``link/../`` first would erase the alias before
    # it can be rejected.
    from drbrain.runtime import _first_symlink_component

    root_alias = _first_symlink_component(root)
    if root_alias is not None:
        raise ValueError(f"papers root must not contain a symlink component: {root_alias}")
    root_resolved = root.resolve()
    candidate = Path(candidate).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate_alias = _first_symlink_component(candidate)
    if candidate_alias is not None:
        try:
            candidate_alias.resolve().relative_to(root_resolved)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"paper path escapes papers root via symlink {root}: {candidate}"
            ) from exc
        raise ValueError(f"paper path contains a symlink component: {candidate_alias}")
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"paper path escapes papers root {root}: {candidate}") from exc

    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            try:
                current.resolve().relative_to(root_resolved)
            except (OSError, ValueError) as exc:
                raise ValueError(
                    f"paper path escapes papers root via symlink {root}: {current}"
                ) from exc
            raise ValueError(f"paper path contains a symlink component: {current}")
        # Existing non-directory parents would make a later write resolve in
        # an implementation-dependent way; fail before touching anything.
        if current != candidate and current.exists() and not current.is_dir():
            raise ValueError(f"paper path parent is not a directory: {current}")

    try:
        candidate.resolve().relative_to(root_resolved)
    except (OSError, ValueError) as exc:
        raise ValueError(f"paper path escapes papers root {root}: {candidate}") from exc
    if candidate.exists() and not candidate.is_dir():
        raise ValueError(f"paper path is not a directory: {candidate}")
    if require_directory and not candidate.is_dir():
        raise ValueError(f"paper path is not a directory: {candidate}")
    return candidate


def _active_runtime_context():
    """Return the selected runtime for direct artifact writers, if any."""
    if "DRBRAIN_ROOT" not in os.environ and "DRBRAIN_RUNTIME_ROOT" not in os.environ:
        return None
    from drbrain.runtime import RuntimeContext

    return RuntimeContext.create()


def _runtime_scoped_papers_root(papers_root: str | Path) -> Path:
    """Resolve a papers root and bind it to the selected runtime namespace."""
    root = Path(papers_root).expanduser()
    runtime = _active_runtime_context()
    if runtime is not None:
        root = runtime.assert_within_root(root, label="papers root")
    return root


def writable_artifact_path(paper_path: str | Path, filename: str) -> Path:
    """Return a fixed artifact path without following a symlink on write.

    Callers should write atomically (temporary file + replace) after obtaining
    this path.  Rejecting a pre-existing symlink prevents a stale artifact
    planted in a paper directory from redirecting bytes elsewhere.
    """
    runtime = _active_runtime_context()
    directory = Path(paper_path)
    if not filename or Path(filename).name != filename or "\x00" in filename:
        raise ValueError("artifact filename must be one path component")
    if runtime is not None:
        # Relative direct calls follow the selected namespace rather than the
        # process CWD; absolute calls are checked as well so a plugin cannot
        # redirect bytes into another worktree.
        directory = runtime.assert_within_root(directory, label="paper artifact parent")
    # When called directly there is no separate ``papers_root`` argument.  A
    # relative paper path therefore has to be interpreted from the caller's
    # working directory (the same convention as ``Path``/``open``), while an
    # absolute path can use the filesystem anchor.  In both cases the
    # validator still walks every ancestor and rejects an intermediate alias,
    # including ``runtime/link/paper`` where ``link`` points elsewhere.
    anchor = (
        runtime.root
        if runtime is not None
        else (Path.cwd() if not directory.is_absolute() else Path(directory.anchor or os.sep))
    )
    try:
        directory = _validate_paper_candidate(anchor, directory, require_directory=True)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"paper artifact parent is not a real directory: {paper_path}: {exc}"
        ) from exc
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"paper artifact parent is not a real directory: {directory}")
    candidate = directory / filename
    if candidate.is_symlink():
        raise ValueError(f"paper artifact is a symlink: {candidate}")
    if candidate.exists() and not candidate.is_file():
        raise ValueError(f"paper artifact is not a regular file: {candidate}")
    return candidate


def writable_artifact_dir(paper_path: str | Path, dirname: str) -> Path:
    """Create and return a safe writable subdirectory of a paper directory.

    The directory name is intentionally limited to one component.  Validation
    happens both before and after ``mkdir`` so a pre-existing or concurrently
    planted symlink cannot be accepted as an artifact destination.
    """
    runtime = _active_runtime_context()
    directory = Path(paper_path)
    if not dirname or Path(dirname).name != dirname or "\x00" in dirname:
        raise ValueError("artifact directory name must be one path component")
    if runtime is not None:
        directory = runtime.assert_within_root(directory, label="paper artifact parent")
    anchor = (
        runtime.root
        if runtime is not None
        else (Path.cwd() if not directory.is_absolute() else Path(directory.anchor or os.sep))
    )
    try:
        directory = _validate_paper_candidate(anchor, directory, require_directory=True)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"paper artifact parent is not a real directory: {paper_path}: {exc}"
        ) from exc
    target = directory / dirname
    try:
        _validate_paper_candidate(directory, target)
        target.mkdir(parents=False, exist_ok=True)
        _validate_paper_candidate(directory, target, require_directory=True)
    except (OSError, ValueError) as exc:
        raise ValueError(f"paper artifact directory is not safe: {target}: {exc}") from exc
    if target.is_symlink() or not target.is_dir():
        raise ValueError(f"paper artifact directory is not a real directory: {target}")
    return target


def paper_fs_key(local_id: str) -> str:
    """Return the canonical, single-component filesystem key for ``local_id``.

    Existing safe IDs (including the generated ``p...`` IDs and historical
    underscore-sanitized IDs) are returned unchanged.  Other IDs are encoded
    using percent-escaped UTF-8, so a DOI such as ``10.1234/a/b`` becomes one
    directory name rather than nested directories.  The encoding is
    reversible via :func:`paper_id_from_fs_key` and is collision-free for all
    IDs representable by a filesystem component.
    """
    value = _validate_local_id(local_id)
    if (
        _SAFE_LOCAL_ID_RE.fullmatch(value)
        and value not in {".", ".."}
        and not value.endswith((".", " "))
        and not _WINDOWS_RESERVED_RE.fullmatch(value)
    ):
        if len(value.encode("utf-8")) > _MAX_COMPONENT_BYTES:
            raise ValueError(
                f"paper local_id is too long for a filesystem key ({len(value.encode('utf-8'))} bytes)"
            )
        return value

    # Exclude ``.`` from the safe alphabet for the special path components.
    if _WINDOWS_RESERVED_RE.fullmatch(value):
        # Windows reserves device names even when they look like ordinary
        # directory components (including ``CON.txt``).  Escape the first
        # character so the key remains reversible but can never resolve to a
        # device path on that platform.
        encoded = f"%{ord(value[0]):02X}" + quote(value[1:], safe="-._")
    else:
        encoded = quote(value, safe="-._")
    # Leading/trailing dots are legal on POSIX but problematic on Windows
    # (and ``.``/``..`` have path semantics).  Escape only boundary dots while
    # retaining the readable DOI prefix in the usual case.
    if encoded.startswith("."):
        encoded = "%2E" + encoded[1:]
    if encoded.endswith("."):
        encoded = encoded[:-1] + "%2E"
    if encoded in {".", ".."}:
        encoded = quote(value, safe="-_")
    if len(encoded.encode("utf-8")) > _MAX_COMPONENT_BYTES:
        raise ValueError(
            f"paper local_id is too long for a filesystem key ({len(encoded.encode('utf-8'))} bytes)"
        )
    return encoded


def paper_id_from_fs_key(fs_key: str) -> str:
    """Decode a canonical filesystem key back to its database local ID."""
    if not isinstance(fs_key, str):
        raise ValueError("filesystem key must be a string")
    key = fs_key
    if not key or "/" in key or "\\" in key or "\x00" in key:
        raise ValueError("filesystem key must be one non-empty path component")
    # Safe IDs never contain ``%``.  Avoiding unquote in that case preserves
    # exact compatibility for all existing p.../underscore directory names.
    decoded = unquote(key) if "%" in key else key
    # Reject malformed/non-canonical escape strings instead of allowing two
    # filesystem names to decode to one identity.  Legacy names are handled by
    # ``resolve_paper_dir`` and do not pass through this decoder.
    if paper_fs_key(decoded) != key:
        raise ValueError(f"non-canonical filesystem key: {fs_key!r}")
    return decoded


def paper_id_from_dir(paper_path: str | Path, papers_root: str | Path | None = None) -> str:
    """Derive a database local ID from an asset directory.

    New canonical directories decode directly from their basename.  For old
    DOI layouts, ``papers_root`` lets us recover the complete relative path
    (``papers/10.1/foo`` → ``10.1/foo``); without it, the common one-level DOI
    shape is still recognized.  Callers that already know the DB ID should
    prefer passing it explicitly to their collection function.
    """
    path = Path(paper_path)
    if papers_root is not None:
        papers_root = _runtime_scoped_papers_root(papers_root)
        # Keep identity derivation subject to the same lexical containment
        # policy as directory resolution.  This matters for indexers that
        # discover paths first and derive IDs afterwards.
        _validate_paper_candidate(Path(papers_root), path)
    name = path.name
    decoded = paper_id_from_fs_key(name)
    if decoded != name:
        return decoded

    if papers_root is not None:
        root = Path(papers_root)
        try:
            relative = path.resolve().relative_to(root.resolve())
        except ValueError:
            relative = None
        if relative is not None and len(relative.parts) > 1:
            first = relative.parts[0]
            if _DOI_COMPONENT_RE.fullmatch(first):
                return "/".join(relative.parts)

    # Best-effort compatibility for direct callers handed a legacy nested DOI
    # path without its root.  Walk upward until the DOI prefix is found so
    # suffixes containing additional slashes are reconstructed too.
    for ancestor in path.parents:
        if _DOI_COMPONENT_RE.fullmatch(ancestor.name):
            relative = path.relative_to(ancestor)
            if relative.parts:
                return "/".join((ancestor.name, *relative.parts))
    return name


def _legacy_paper_dirs(papers_root: Path, local_id: str) -> list[Path]:
    """Return existing pre-canonical candidates for an unsafe local ID."""
    if "/" not in local_id:
        return []
    parts = local_id.split("/")
    # Reject traversal and empty components before constructing any candidate.
    if any(not part or part in {".", ".."} for part in parts):
        return []
    nested = papers_root.joinpath(*parts)
    # Older scripts used two flattened variants: either the complete DOI was
    # flattened (``10.1_foo_bar``), or only the suffix was flattened beneath
    # the DOI registrant directory (``10.1/foo_bar``).  Keep both fallbacks
    # read-only; all new writes use the canonical percent key.
    flattened = papers_root / local_id.replace("/", "_")
    prefix_flat = papers_root / parts[0] / "_".join(parts[1:])
    out: list[Path] = []
    for candidate in (nested, prefix_flat, flattened):
        # Legacy layouts are read during migration, but they are still used as
        # write targets by idempotent ingest.  Resolve every existing candidate
        # before returning it so a symlink (including one in an intermediate
        # DOI component) cannot redirect artifacts outside the selected root.
        if candidate.is_symlink() or candidate.exists():
            if not candidate.is_dir():
                if candidate.is_symlink():
                    raise ValueError(f"paper directory is not a directory: {candidate}")
                continue
            _validate_paper_candidate(papers_root, candidate, require_directory=True)
            if candidate not in out:
                out.append(candidate)
        else:
            # Validate missing candidates too: an intermediate symlink can
            # otherwise be followed by a subsequent mkdir during ingest.
            _validate_paper_candidate(papers_root, candidate)
    return out


def resolve_paper_dir(papers_root: str | Path, local_id: str) -> Path | None:
    """Find an existing paper directory using canonical or legacy layouts.

    The canonical path is always preferred.  If only legacy paths exist, a
    single candidate is returned; if both old nested and flattened candidates
    exist, resolution fails loudly instead of silently selecting the wrong
    paper (the two layouts cannot be disambiguated from the basename alone).
    """
    root = _runtime_scoped_papers_root(papers_root)
    value = _validate_local_id(local_id)
    canonical = root / paper_fs_key(value)
    _validate_paper_candidate(root, canonical)
    if canonical.is_symlink() or canonical.exists():
        _validate_paper_candidate(root, canonical, require_directory=True)
        return canonical
    legacy = _legacy_paper_dirs(root, value)
    if len(legacy) > 1:
        raise ValueError(f"ambiguous paper directory layouts for local_id {value!r}")
    return legacy[0] if legacy else None


def iter_paper_dirs(papers_root: str | Path) -> list[Path]:
    """Return asset directories containing ``tree.json`` recursively.

    Recursive discovery is needed for legacy nested DOI directories; canonical
    flat directories are included by the same traversal.  Symlinked dirs are
    skipped to keep discovery inside the configured root.
    """
    root = _runtime_scoped_papers_root(papers_root)
    if root.is_symlink():
        raise ValueError(f"papers root must not be a symlink: {root}")
    if not root.is_dir():
        return []
    # Validate lexical ancestors as well as the leaf.  ``root.resolve()``
    # alone would accept e.g. ``runtime/link/papers`` when ``link`` points at
    # another worktree.
    _validate_paper_candidate(root, root, require_directory=True)
    try:
        resolved_root = root.resolve()
    except OSError:
        return []
    candidates: list[Path] = []
    for tree_path in root.rglob("tree.json"):
        try:
            # Do not ingest a tree file supplied through a symlink.  A regular
            # parent directory does not guarantee the file itself stays within
            # the configured root (and pathlib may encounter symlinked
            # intermediate components on some platforms).
            if tree_path.is_symlink():
                continue
            tree_path.resolve().relative_to(resolved_root)
            parent = tree_path.parent
            if parent.is_dir() and not parent.is_symlink():
                candidates.append(parent)
        except (OSError, ValueError):
            continue

    # A migration can leave the same paper in its new encoded directory and
    # one or more of the old DOI layouts.  Indexing every tree file would then
    # duplicate concepts and vectors.  First derive the identity from each
    # directory, then associate known DOI IDs with their legacy aliases (the
    # flattened forms cannot be decoded reliably in isolation).
    unique_candidates = sorted(set(candidates), key=lambda p: str(p))
    if not unique_candidates:
        return []

    direct_ids: dict[Path, str] = {}
    for candidate in unique_candidates:
        try:
            direct_ids[candidate] = paper_id_from_dir(candidate, root)
        except (OSError, ValueError):
            # A malformed component is not safe to expose to an indexer.
            continue
    if not direct_ids:
        return []

    candidate_set = set(direct_ids)
    groups: dict[str, set[Path]] = {}
    ambiguous_ids: set[str] = set()
    for candidate, local_id in direct_ids.items():
        groups.setdefault(local_id, set()).add(candidate)

    # Nested/encoded DOI paths provide an unambiguous ID.  Use that ID to
    # recognize the two historical flattened layouts as aliases of the same
    # paper.  A flattened path that could also be a literal underscore ID is
    # treated as ambiguous and removed from that literal group; silently
    # choosing one identity would be worse than skipping it during migration.
    doi_ids = {
        local_id
        for local_id in direct_ids.values()
        if "/" in local_id and _DOI_COMPONENT_RE.fullmatch(local_id.split("/", 1)[0])
    }
    for local_id in doi_ids:
        try:
            legacy_candidates = _legacy_paper_dirs(root, local_id)
        except (OSError, ValueError):
            ambiguous_ids.add(local_id)
            continue
        for legacy_path in legacy_candidates:
            if legacy_path not in candidate_set:
                continue
            direct_id = direct_ids[legacy_path]
            if direct_id != local_id:
                groups.get(direct_id, set()).discard(legacy_path)
                ambiguous_ids.add(direct_id)
            groups.setdefault(local_id, set()).add(legacy_path)

    selected: list[Path] = []
    for local_id, paths in groups.items():
        if local_id in ambiguous_ids:
            continue
        try:
            canonical_key = paper_fs_key(local_id)
            canonical = [
                path
                for path in paths
                if path.parent.resolve() == resolved_root and path.name == canonical_key
            ]
        except (OSError, ValueError):
            # Unrepresentable identities (for example an overlong component)
            # must not abort discovery of the remaining corpus.
            continue
        if len(canonical) == 1:
            selected.append(canonical[0])
        elif len(canonical) > 1:
            # This should not occur on a normal filesystem, but do not index
            # duplicate aliases if a custom path provider reports them.
            continue
        elif len(paths) == 1:
            selected.append(next(iter(paths)))
        # More than one legacy layout for one ID is ambiguous.  Skip the
        # entire group rather than selecting an arbitrary directory.
    return sorted(set(selected), key=lambda p: str(p))


def paper_dir(papers_root: Path, local_id: str) -> Path:
    """Return the canonical per-paper directory path.

    Existing legacy directories are returned when present so reads and
    idempotent re-ingests do not strand already-downloaded PDFs.  New IDs are
    always written below the canonical single-component key.
    """
    root = _runtime_scoped_papers_root(papers_root)
    value = _validate_local_id(local_id)
    canonical = root / paper_fs_key(value)
    _validate_paper_candidate(root, canonical)
    if canonical.is_symlink() or canonical.exists():
        _validate_paper_candidate(root, canonical, require_directory=True)
        return canonical
    legacy = _legacy_paper_dirs(root, value)
    if len(legacy) > 1:
        raise ValueError(f"ambiguous paper directory layouts for local_id {value!r}")
    return legacy[0] if legacy else canonical


def _artifact_path(paper_dir: str | Path, filename: str) -> Path:
    """Return a read/write artifact path without following symlink aliases."""
    directory = Path(paper_dir).expanduser()
    runtime = _active_runtime_context()
    if runtime is not None:
        directory = runtime.assert_within_root(directory, label="paper artifact parent")
    # The directory may be created by the caller immediately afterwards, so
    # validate existing components but do not require the leaf to exist yet.
    from drbrain.runtime import _first_symlink_component

    alias = _first_symlink_component(directory)
    if alias is not None:
        raise ValueError(f"paper artifact parent contains a symlink component: {alias}")
    if directory.exists() and not directory.is_dir():
        raise ValueError(f"paper artifact parent is not a directory: {directory}")
    if not filename or Path(filename).name != filename or "\x00" in filename:
        raise ValueError("artifact filename must be one path component")
    candidate = directory / filename
    if candidate.is_symlink():
        raise ValueError(f"paper artifact is a symlink: {candidate}")
    if candidate.exists() and not candidate.is_file():
        raise ValueError(f"paper artifact is not a regular file: {candidate}")
    return candidate


def raw_md_path(paper_dir: Path) -> Path:
    """Return path to the MinerU markdown file."""
    return _artifact_path(paper_dir, "raw.md")


def tree_json_path(paper_dir: Path) -> Path:
    """Return path to the PageIndex tree JSON file."""
    return _artifact_path(paper_dir, "tree.json")


def source_pdf_path(paper_dir: Path) -> Path:
    """Return path to the source PDF copy."""
    return _artifact_path(paper_dir, "source.pdf")


def images_dir(paper_dir: Path) -> Path:
    """Return path to the extracted images directory."""
    directory = Path(paper_dir).expanduser()
    runtime = _active_runtime_context()
    if runtime is not None:
        directory = runtime.assert_within_root(directory, label="paper artifact parent")
    from drbrain.runtime import _first_symlink_component

    alias = _first_symlink_component(directory)
    if alias is not None:
        raise ValueError(f"paper artifact parent contains a symlink component: {alias}")
    if directory.exists() and not directory.is_dir():
        raise ValueError(f"paper artifact parent is not a directory: {directory}")
    target = directory / "images"
    if target.is_symlink():
        raise ValueError(f"paper images directory is a symlink: {target}")
    if target.exists() and not target.is_dir():
        raise ValueError(f"paper images path is not a directory: {target}")
    return target
