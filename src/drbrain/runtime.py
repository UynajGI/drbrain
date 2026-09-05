"""Runtime-scoped paths and environment for an isolated DrBrain run.

The CLI historically resolved relative paths against whatever directory the
process happened to use, while several bulk scripts carried their own absolute
repository path.  ``RuntimeContext`` is the small boundary shared by both
entrypoints: it gives a run one root, one temporary namespace, and one set of
normalized config paths.
"""

from __future__ import annotations

import copy
import hashlib
import os
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import Any

from drbrain.security import redact_sensitive_text

_SPECIAL_PATHS = {":memory:"}
# Empty values are meaningful only for explicitly optional resources.  An
# empty mutable path otherwise becomes ``Path('')`` (the process CWD), which
# is precisely the kind of silent fallback the runtime boundary is meant to
# prevent.
_OPTIONAL_EMPTY_PATH_FIELDS = {
    ("embed", "cache_dir"),
    ("autoresearch", "plugins_dir"),
}
_RUN_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_URI_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")

# Only values that are local filesystem paths belong here.  Keeping this list
# explicit prevents accidental rewriting of URLs, model names, or remote
# backup destinations.
_PATH_FIELDS: tuple[tuple[str, ...], ...] = (
    ("db", "path"),
    ("dirs", "inbox"),
    ("dirs", "pending"),
    ("dirs", "papers"),
    ("dirs", "reports"),
    ("dirs", "cache"),
    ("dirs", "logs"),
    ("dirs", "backups"),
    ("dirs", "citation_styles"),
    ("embed", "cache_dir"),
    ("llamaindex", "storage_dir"),
    ("llamaindex", "eval", "golden_set"),
    ("autoresearch", "run_dir"),
    ("autoresearch", "plugins_dir"),
)

# Paths that contain the mutable corpus or runtime state.  Model caches and
# remote backup destinations are deliberately excluded: sharing those is a
# supported deployment choice, while sharing a library DB is not.
_ISOLATED_PATH_FIELDS: tuple[tuple[str, ...], ...] = tuple(
    field_path for field_path in _PATH_FIELDS if field_path not in {("embed", "cache_dir")}
)


def _get_child(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _has_child(value: Any, key: str) -> bool:
    """Return whether a config object exposes *key* without coercing errors."""
    if isinstance(value, Mapping):
        return key in value
    return hasattr(value, key)


def _path_parent(config: Any, path_keys: tuple[str, ...]) -> Any | None:
    """Find a path field's parent and reject malformed intermediate sections.

    Missing sections are allowed so callers can pass partial config mappings.
    A section that is present but scalar (or explicitly ``None``) is different:
    treating it as absent would skip the path check and let a malformed config
    fall through to a process-CWD write target.
    """
    parent = config
    for index, key in enumerate(path_keys[:-1]):
        if not _has_child(parent, key):
            return None
        child = _get_child(parent, key)
        next_key = path_keys[index + 1]
        # An empty mapping is a valid partial section (the leaf will simply be
        # absent below).  A scalar/sequence, or an object without the expected
        # next attribute, is a malformed section and must not be treated as
        # missing.
        if child is None or (not isinstance(child, Mapping) and not _has_child(child, next_key)):
            section = ".".join(path_keys[: index + 1])
            raise ValueError(f"configured section '{section}' must be a mapping")
        parent = child
    return parent


def _set_child(value: Any, key: str, child: Any) -> None:
    if isinstance(value, dict):
        value[key] = child
    else:
        setattr(value, key, child)


def _is_special_path(value: Any) -> bool:
    """Return whether a config value is a non-filesystem path sentinel."""
    try:
        value = os.fspath(value)
    except TypeError:
        return False
    return isinstance(value, str) and value in _SPECIAL_PATHS


def _is_empty_path(value: Any) -> bool:
    """Return whether a path value is the empty string (never a path)."""
    try:
        value = os.fspath(value)
    except TypeError:
        return False
    return isinstance(value, str) and value == ""


def _configured_runtime_root() -> str | None:
    """Read the inherited runtime selector without silently accepting empties.

    ``DRBRAIN_ROOT`` has precedence over the legacy alias.  Once a selector is
    present, even an empty value is an explicit configuration error; falling
    through to the current working directory would let a launcher write into
    an unintended worktree.
    """

    if "DRBRAIN_ROOT" in os.environ:
        configured = os.environ["DRBRAIN_ROOT"]
        if _is_empty_path(configured):
            raise ValueError("DRBRAIN_ROOT must not be empty")
        return configured
    if "DRBRAIN_RUNTIME_ROOT" in os.environ:
        configured = os.environ["DRBRAIN_RUNTIME_ROOT"]
        if _is_empty_path(configured):
            raise ValueError("DRBRAIN_RUNTIME_ROOT must not be empty")
        return configured
    return None


def _is_optional_empty_path(path_keys: tuple[str, ...], value: Any) -> bool:
    """Return whether an empty value is allowed for this optional field."""
    return path_keys in _OPTIONAL_EMPTY_PATH_FIELDS and _is_empty_path(value)


def _validate_config_shape(config: Any) -> None:
    """Reject malformed top-level configs before path fields are inspected."""

    if isinstance(config, Mapping):
        return
    if is_dataclass(config) and not isinstance(config, type):
        return
    raise ValueError("runtime config must be a mapping or typed dataclass")


def _is_file_uri(value: Any) -> bool:
    """Return whether *value* is a SQLite ``file:`` URI.

    ``file:`` is intentionally handled separately from ordinary remote URLs:
    SQLite accepts it as a database location and can therefore bypass the
    worktree boundary (including with ``mode=rwc``).  A URI is fine for an
    explicitly shared, non-isolated resource, but never for mutable runtime
    state such as the library DB or papers directory.
    """

    try:
        value = os.fspath(value)
    except TypeError:
        return False
    return isinstance(value, str) and value.lower().startswith("file:")


def _is_uri(value: Any) -> bool:
    """Return whether *value* uses a URI scheme instead of a local path."""

    try:
        value = os.fspath(value)
    except TypeError:
        return False
    return isinstance(value, str) and bool(_URI_SCHEME_RE.match(value))


def _first_symlink_component(value: str | Path) -> Path | None:
    """Return the first lexical symlink in *value*, without resolving it.

    ``Path.resolve()`` is necessary for containment checks, but it erases the
    fact that a caller supplied a symlink alias.  Checking the lexical path
    separately prevents an in-root alias from becoming a mutable write target
    while still allowing ordinary missing path components to be created later.
    """
    try:
        path = Path(value).expanduser()
    except (TypeError, ValueError, OSError):
        return None
    if not path.is_absolute():
        path = Path.cwd() / path

    # Walk the original lexical components.  ``os.path.abspath``/``normpath``
    # would erase ``link/../`` before we inspect ``link`` and could therefore
    # hide a symlink alias from the policy.  ``..`` is handled lexically after
    # the component preceding it has been checked.
    current = Path(path.anchor or os.sep)
    for part in path.parts:
        if part in {"", ".", path.anchor}:
            continue
        if part == "..":
            current = current.parent
            continue
        current /= part
        try:
            if current.is_symlink():
                return current
        except OSError:
            return current
    return None


def _normalize_run_id(value: str | None) -> str:
    """Normalize a run identifier without silently collapsing explicit values."""
    if value is not None:
        raw = value
        if not isinstance(raw, str) or not raw:
            raise ValueError("DRBRAIN_RUN_ID must not be empty")
    elif "DRBRAIN_RUN_ID" in os.environ:
        raw = os.environ["DRBRAIN_RUN_ID"]
        if not raw:
            raise ValueError("DRBRAIN_RUN_ID must not be empty")
    else:
        raw = uuid.uuid4().hex[:12]

    normalized = _RUN_ID_RE.sub("_", str(raw)).strip("._-")
    if not normalized:
        raise ValueError("DRBRAIN_RUN_ID must contain at least one safe character")
    # A run id is incorporated into temporary directory names and log paths;
    # never preserve an explicitly supplied credential-looking value there.
    raw_lower = str(raw).lower()
    if (
        redact_sensitive_text(str(raw)) != str(raw)
        or raw_lower.startswith("sk-")
        or any(token in raw_lower for token in ("api_key", "apikey", "token=", "secret", "password"))
    ):
        digest = hashlib.sha256(str(raw).encode("utf-8")).hexdigest()[:12]
        normalized = f"run-{digest}"
    if len(normalized) > 80:
        # Keep the readable prefix while making truncation collision-resistant.
        suffix = hashlib.sha256(str(raw).encode("utf-8")).hexdigest()[:12]
        normalized = f"{normalized[:67]}-{suffix}"
    return normalized


@dataclass(frozen=True)
class RuntimeContext:
    """Paths and process environment belonging to one isolated run.

    ``root`` is normally the current worktree.  Relative paths in a typed or
    dict-style config are made absolute beneath it by :meth:`apply_config`.
    Explicit absolute paths are preserved for backwards compatibility; bulk
    scripts must use this context instead of carrying a separate hard-coded
    root.
    """

    root: Path
    temp_root: Path
    run_id: str
    overlay_path: Path | None = None

    @classmethod
    def create(
        cls,
        root: str | Path | None = None,
        *,
        run_id: str | None = None,
        temp_root: str | Path | None = None,
        overlay_path: str | Path | None = None,
        create_root: bool = False,
    ) -> RuntimeContext:
        """Create a context rooted at a directory.

        The environment variable is intentionally checked here as well as in
        the shell wrappers, so Python entrypoints launched directly inherit the
        same isolation contract.  ``create_root`` is reserved for the setup
        command: ordinary callers fail closed on a missing root so a typo
        cannot silently create a new namespace.
        """

        if root is not None and _is_empty_path(root):
            raise ValueError("Runtime root must not be empty")
        inherited_root = _configured_runtime_root() if root is None else None
        raw_root = root if root is not None else inherited_root or Path.cwd()
        if _is_uri(raw_root):
            raise ValueError("Runtime root must be a local filesystem path")
        root_alias = _first_symlink_component(raw_root)
        if root_alias is not None:
            raise ValueError(f"Runtime root must not contain a symlink component: {root_alias}")
        root_path = Path(raw_root).expanduser().resolve()
        if not root_path.exists():
            if not create_root:
                raise ValueError(f"Runtime root does not exist: {root_path}")
            try:
                root_path.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ValueError(f"Runtime root could not be created: {root_path}") from exc
            # A concurrent creator may have published a symlink while mkdir
            # was in progress.  Recheck both the lexical alias and resolved
            # directory before any config/artifact path is derived.
            root_alias = _first_symlink_component(raw_root)
            if root_alias is not None:
                raise ValueError(f"Runtime root must not contain a symlink component: {root_alias}")
            root_path = Path(raw_root).expanduser().resolve()
        if not root_path.exists():
            raise ValueError(f"Runtime root does not exist: {root_path}")
        if not root_path.is_dir():
            raise ValueError(f"Runtime root must be a directory: {root_path}")

        resolved_run_id = _normalize_run_id(run_id)
        if temp_root is None and root is None and "DRBRAIN_TEMP_ROOT" in os.environ:
            # Standalone workers inherit this selector from shell launchers.
            # An explicitly empty value is still an error; silently falling
            # back to the default would split one run across namespaces.
            temp_root = os.environ["DRBRAIN_TEMP_ROOT"]
        if temp_root is not None and _is_empty_path(temp_root):
            raise ValueError("Runtime temp root must not be empty")
        if temp_root is None:
            lexical_temp_path = root_path / "data" / ".runtime" / resolved_run_id
        else:
            if _is_uri(temp_root):
                raise ValueError("Runtime temp root must be a local filesystem path")
            lexical_temp_path = Path(temp_root).expanduser()
            if not lexical_temp_path.is_absolute():
                lexical_temp_path = root_path / lexical_temp_path
        # Check the lexical path even when the default is used.  Without this
        # branch a pre-existing ``data/.runtime`` symlink could redirect the
        # supposedly private scratch namespace before ``ensure_temp_root``
        # creates it.
        temp_alias = _first_symlink_component(lexical_temp_path)
        if temp_alias is not None:
            raise ValueError(
                f"Runtime temp root must not contain a symlink component: {temp_alias}"
            )
        temp_path = lexical_temp_path.resolve()
        try:
            temp_path.relative_to(root_path)
        except ValueError as exc:
            raise ValueError(
                f"Runtime temp root escapes runtime root {root_path}: {temp_path}"
            ) from exc

        overlay = None
        if overlay_path is not None and _is_empty_path(overlay_path):
            raise ValueError("Config overlay must not be empty")
        if overlay_path is not None:
            if _is_uri(overlay_path):
                raise ValueError("Config overlay must be a local path")
            lexical_overlay = Path(overlay_path).expanduser()
            if not lexical_overlay.is_absolute():
                lexical_overlay = root_path / lexical_overlay
            overlay = lexical_overlay.resolve()
            try:
                overlay.relative_to(root_path)
            except ValueError as exc:
                raise ValueError(
                    f"Config overlay escapes runtime root {root_path}: {overlay}"
                ) from exc
            overlay_alias = _first_symlink_component(lexical_overlay)
            if overlay_alias is not None:
                raise ValueError(
                    f"Config overlay must not contain a symlink component: {overlay_alias}"
                )

        return cls(
            root=root_path,
            temp_root=temp_path,
            run_id=resolved_run_id,
            overlay_path=overlay,
        )

    @property
    def config_path(self) -> Path | None:
        """Backward-compatible name used by pipeline subprocess launchers."""

        return self.overlay_path

    @property
    def base_config_path(self) -> Path:
        """The base config associated with this runtime root."""

        return self.root / "config.yaml"

    def resolve_path(self, value: str | Path) -> Path | str:
        """Resolve a local path beneath ``root`` while preserving sentinels."""

        if not isinstance(value, (str, Path)):
            raise ValueError(
                f"configured path must be a string or path, got {type(value).__name__}"
            )
        if _is_special_path(value):
            return ":memory:"
        if _is_empty_path(value):
            raise ValueError("configured path must not be empty")
        if _is_uri(value):
            return value
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.root / path
        return path.resolve()

    def apply_config(self, config: Any) -> Any:
        """Return a deep-copied config with local paths scoped to this run.

        Both the current typed ``Config`` and legacy nested dictionaries are
        accepted.  The input is never mutated, which is important for tests and
        for callers that reuse a base config for multiple isolated runs.
        """

        _validate_config_shape(config)
        normalized = copy.deepcopy(config)
        for path_keys in _PATH_FIELDS:
            parent = _path_parent(normalized, path_keys)
            if parent is None:
                continue
            leaf = path_keys[-1]
            if not _has_child(parent, leaf):
                continue
            current = _get_child(parent, leaf)
            if current is None:
                raise ValueError(f"configured {'.'.join(path_keys)} must be a path, not null")
            if _is_optional_empty_path(path_keys, current):
                continue
            if _is_special_path(current):
                if path_keys == ("db", "path"):
                    continue
                raise ValueError(
                    f"configured {'.'.join(path_keys)} does not accept the :memory: sentinel"
                )
            if path_keys in _ISOLATED_PATH_FIELDS and _is_uri(current):
                reason = (
                    "SQLite file: URIs are not allowed"
                    if _is_file_uri(current)
                    else "URI schemes are not allowed"
                )
                raise ValueError(f"configured {'.'.join(path_keys)} must be a local path; {reason}")
            resolved = self.resolve_path(current)
            # Relative traversal (for example ``../shared.db``) is never a
            # valid isolated path.  Absolute paths remain compatible with
            # deployments that deliberately share a resource; callers that
            # require a fully private root use ``validate_config`` below.
            if isinstance(resolved, Path) and not Path(current).expanduser().is_absolute():
                if not self.is_within_root(resolved):
                    raise ValueError(
                        f"configured {'.'.join(path_keys)} escapes runtime root {self.root}: {resolved}"
                    )
            # Config fields are declared as strings.  Preserve that public
            # shape for dict and dataclass callers while ``resolve_path`` can
            # still return a Path for direct filesystem operations.
            if isinstance(current, str) and isinstance(resolved, Path):
                resolved = str(resolved)
            _set_child(parent, leaf, resolved)
        return normalized

    def ensure_temp_root(self) -> Path:
        """Create and return this run's private temporary directory."""

        self.temp_root.mkdir(parents=True, exist_ok=True)
        # Re-check after creation in case an existing parent was replaced by
        # a symlink between context construction and this call.
        alias = _first_symlink_component(self.temp_root)
        if alias is not None:
            raise ValueError(f"Runtime temp root must not contain a symlink component: {alias}")
        if not self.is_within_root(self.temp_root):
            raise ValueError(
                f"Runtime temp root escapes runtime root {self.root}: {self.temp_root}"
            )
        return self.temp_root

    def is_within_root(self, path: str | Path) -> bool:
        """Return whether *path* resolves beneath this runtime root."""

        try:
            candidate = Path(path).expanduser().resolve()
        except (TypeError, ValueError, OSError):
            return False
        try:
            candidate.relative_to(self.root)
        except ValueError:
            return False
        return True

    def assert_within_root(self, path: str | Path, *, label: str = "path") -> Path:
        """Resolve *path* and fail closed when it escapes the runtime root."""

        if _is_uri(path):
            raise ValueError(f"{label} must be a local filesystem path")
        if _is_empty_path(path):
            raise ValueError(f"{label} must not be empty")
        try:
            candidate = Path(path).expanduser()
        except (TypeError, ValueError, OSError) as exc:
            raise ValueError(f"{label} is not a valid path: {path!r}") from exc
        if not candidate.is_absolute():
            candidate = self.root / candidate
        lexical_candidate = candidate
        candidate = candidate.resolve()
        if not self.is_within_root(candidate):
            raise ValueError(f"{label} escapes runtime root {self.root}: {candidate}")
        alias = _first_symlink_component(lexical_candidate)
        if alias is not None:
            lexical_absolute = Path(os.path.abspath(os.fspath(lexical_candidate)))
            if alias == lexical_absolute:
                raise ValueError(f"{label} must not be a symlink: {alias}")
            raise ValueError(f"{label} must not contain a symlink component: {alias}")
        return candidate

    def validate_config_file(
        self,
        path: str | Path,
        *,
        label: str = "config file",
        required: bool = False,
    ) -> Path:
        """Validate a config layer before opening it.

        Config files are read before the typed path policy can run, so a
        symlinked layer could otherwise import another worktree's settings or
        secrets.  Keep this check at the runtime boundary and allow callers to
        probe an optional local layer with ``required=False``.
        """

        candidate = self.assert_within_root(path, label=label)
        lexical = Path(path).expanduser()
        if not lexical.is_absolute():
            lexical = self.root / lexical
        if lexical.is_symlink():
            raise ValueError(f"{label} must not be a symlink: {lexical}")
        if candidate.exists() and not candidate.is_file():
            raise ValueError(f"{label} is not a regular file: {candidate}")
        if required and not candidate.exists():
            raise FileNotFoundError(f"{label} not found: {candidate}")
        return candidate

    def validate_config(self, config: Any) -> None:
        """Fail closed when mutable data paths escape this runtime root.

        Validation is intentionally separate from :meth:`apply_config` so
        callers can retain an explicitly shared model cache or backup target.
        The CLI invokes this check when ``--root``/``DRBRAIN_ROOT`` is used;
        library callers can opt into the same invariant explicitly.
        """

        _validate_config_shape(config)
        for path_keys in _ISOLATED_PATH_FIELDS:
            parent = _path_parent(config, path_keys)
            if parent is None:
                continue
            if not _has_child(parent, path_keys[-1]):
                continue
            value = _get_child(parent, path_keys[-1])
            if value is None:
                raise ValueError(f"configured {'.'.join(path_keys)} must be a path, not null")
            if _is_optional_empty_path(path_keys, value):
                continue
            if _is_special_path(value):
                if path_keys == ("db", "path"):
                    continue
                raise ValueError(
                    f"configured {'.'.join(path_keys)} does not accept the :memory: sentinel"
                )
            if path_keys in _ISOLATED_PATH_FIELDS and _is_uri(value):
                reason = (
                    "SQLite file: URIs are not allowed"
                    if _is_file_uri(value)
                    else "URI schemes are not allowed"
                )
                raise ValueError(f"configured {'.'.join(path_keys)} must be a local path; {reason}")
            try:
                # Validation is the strict opt-in boundary: unlike
                # ``apply_config`` (which retains legacy shared absolute
                # paths), it rejects both resolved escapes and lexical
                # symlink aliases.
                self.assert_within_root(value, label=f"configured {'.'.join(path_keys)}")
            except (TypeError, ValueError, OSError) as exc:
                # Keep the specific reason (escape, symlink, URI, or empty
                # value) visible to callers and CLI diagnostics.
                raise ValueError(
                    f"configured {'.'.join(path_keys)} is not a valid path: {exc}"
                ) from exc
        # Backup identity files are local private inputs, even though the
        # remote target itself is intentionally allowed to be shared.
        backup = _get_child(config, "backup") if _has_child(config, "backup") else None
        targets = _get_child(backup, "targets") if backup is not None and _has_child(backup, "targets") else {}
        if isinstance(targets, Mapping):
            values = targets.values()
        else:
            values = ()
        for target in values:
            identity = _get_child(target, "identity_file") if _has_child(target, "identity_file") else ""
            if identity:
                try:
                    self.assert_within_root(identity, label="backup identity file")
                except (TypeError, ValueError, OSError) as exc:
                    raise ValueError(f"backup identity file is not valid: {exc}") from exc

    def child_env(self) -> dict[str, str]:
        """Return an environment suitable for a child pipeline process."""

        env = os.environ.copy()
        env["DRBRAIN_ROOT"] = str(self.root)
        env["DRBRAIN_RUNTIME_ROOT"] = str(self.root)
        env["DRBRAIN_TEMP_ROOT"] = str(self.temp_root)
        env["DRBRAIN_RUN_ID"] = self.run_id
        # Always publish the effective config anchor.  Leaving an inherited
        # ``DRBRAIN_CONFIG`` in place when this context has no overlay lets a
        # child accidentally load another worktree's config.
        config_path = self.overlay_path or self.base_config_path
        env["DRBRAIN_CONFIG_PATH"] = str(config_path)
        env["DRBRAIN_CONFIG"] = str(config_path)
        return env


def runtime_root() -> Path:
    """Resolve the root for standalone scripts without importing the CLI."""

    configured = _configured_runtime_root()
    if configured:
        if _is_uri(configured):
            raise ValueError("DRBRAIN_ROOT must be a local filesystem path")
        alias = _first_symlink_component(configured)
        if alias is not None:
            raise ValueError(f"DRBRAIN_ROOT must not contain a symlink component: {alias}")
        candidate = Path(configured).expanduser()
        if not candidate.exists():
            raise ValueError(f"runtime root does not exist: {candidate}")
        if not candidate.is_dir():
            raise ValueError(f"runtime root must be a directory: {candidate}")
        return candidate.resolve()
    # Keep the standalone helper aligned with ``RuntimeContext.create()``.
    # Launchers that need a data-only runtime export DRBRAIN_ROOT explicitly;
    # without a selector, the caller's working directory is the only
    # unambiguous namespace and avoids splitting config/DB resolution between
    # CWD and the package checkout.
    return Path.cwd().resolve()
