"""Atomic index publication, pinned readers, and retention; independent of model SDKs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from loguru import logger

from drbrain.config import Config
from drbrain.rag.config import get_llamaindex_config

MANIFEST_NAME = "manifest.json"
ACTIVE_POINTER_NAME = "active.json"
GENERATIONS_DIR_NAME = "generations"
LEGACY_INDEX_GENERATION = "legacy"
GENERATION_RETAIN_COUNT = 3
GENERATION_PRUNE_GRACE_SECONDS = 3600.0
GENERATION_REFERENCES_NAME = "generation-references.json"
GENERATION_REFERENCES_DIR_NAME = "generation-references"


def _storage_dirs(storage_dir: str | Path) -> tuple[Path, Path, Path]:
    """Return (root, vector_dir, bm25_dir) for a configured storage_dir."""
    root = Path(storage_dir)
    return root, root / "vector", root / "bm25"


def _pointer_path(storage_root: Path) -> Path:
    return storage_root / ACTIVE_POINTER_NAME


def _active_storage_root(storage_dir: str | Path) -> tuple[Path, str | None] | None:
    """Return the physical active store and its generation, if one is active.

    A storage directory without ``active.json`` is a pre-generation legacy
    index and remains readable. A malformed or dangling pointer is never
    silently redirected to that legacy index: serving stale data is safer than
    combining generations, but serving an explicitly invalid active pointer is
    not safe at all, so callers receive ``None`` and can surface health failure.
    """
    storage_root = Path(storage_dir)
    pointer = _pointer_path(storage_root)
    if not pointer.exists():
        return storage_root, None
    try:
        raw = json.loads(pointer.read_text(encoding="utf-8"))
        generation = raw.get("generation") if isinstance(raw, dict) else None
        if not isinstance(generation, str) or not generation.strip():
            return None
        generation = generation.strip()
        if generation in {".", ".."} or Path(generation).name != generation:
            return None
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    active_root = storage_root / GENERATIONS_DIR_NAME / generation
    if not active_root.is_dir() or active_root.is_symlink():
        return None
    return active_root, generation


def get_active_index_generation(cfg: Config) -> str | None:
    """Return the active generation id, or ``None`` for legacy/no active index.

    This is additive operational metadata. ``load_index`` keeps its historic
    two-item return value so callers do not need to migrate.
    """
    active = _active_storage_root(get_llamaindex_config(cfg).storage_dir)
    return active[1] if active is not None else None


def capture_index_generation(cfg: Config) -> str | None:
    """Capture the snapshot a long-running caller must keep using.

    ``get_active_index_generation`` keeps its historic ``None`` return for both
    a legacy flat index and an invalid pointer. Durable callers need to
    distinguish them: only a missing pointer is the explicit ``"legacy"``
    snapshot; a malformed or dangling pointer remains unavailable/fail-closed.
    """
    active = _active_storage_root(get_llamaindex_config(cfg).storage_dir)
    if getattr(get_llamaindex_config(cfg), "rag_engine", "llamaindex") == "sql":
        from drbrain.rag.sql_snapshot import resolve_sql_snapshot
        from drbrain.rag.status import RetrievalUnavailableError

        if active is None or active[1] is None:
            return None
        try:
            resolve_sql_snapshot(cfg, active[1])
        except RetrievalUnavailableError:
            return None
        return active[1]
    if active is None:
        return None
    return active[1] or LEGACY_INDEX_GENERATION


def _storage_root_for_generation(storage_dir: str | Path, generation: str | None) -> Path | None:
    """Resolve an optional immutable snapshot without following a newer pointer."""
    root = Path(storage_dir)
    if generation is None:
        active = _active_storage_root(root)
        return active[0] if active is not None else None
    if generation == LEGACY_INDEX_GENERATION:
        return root
    if not generation or generation in {".", ".."} or Path(generation).name != generation:
        return None
    candidate = root / GENERATIONS_DIR_NAME / generation
    return candidate if candidate.is_dir() and not candidate.is_symlink() else None


def _new_generation_id() -> str:
    return f"g-{time.time_ns()}-{uuid.uuid4().hex[:8]}"


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a small JSON control file without torn readers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except OSError:
        # ``Path.write_text`` historically created these public index metadata
        # files as shared-readable on the default deployment filesystem.
        mode = 0o644
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            fd = -1  # ownership moved to the context manager
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, path)
    except Exception:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _load_manifest(storage_dir: str | Path) -> dict[str, Any]:
    active = _active_storage_root(storage_dir)
    if active is None:
        logger.warning("[rag] active generation pointer is invalid at %s", storage_dir)
        return {}
    root, _ = active
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("[rag] ignoring unreadable manifest at %s", manifest_path)
        return {}


def _write_manifest(storage_dir: str | Path, manifest: dict[str, Any]) -> None:
    _write_json_atomically(Path(storage_dir) / MANIFEST_NAME, manifest)


def _generation_references(storage_root: Path) -> dict[str, str] | None:
    """Read durable run references, returning ``None`` for unsafe control data."""
    references: dict[str, str] = {}
    # Read the first PR's single-file shape as a compatibility input, but write
    # only isolated run files below so concurrent run creation cannot lose refs.
    legacy_path = storage_root / GENERATION_REFERENCES_NAME
    if legacy_path.exists():
        try:
            legacy_payload = json.loads(legacy_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.error("[rag] generation references are unreadable at %s", legacy_path)
            return None
        raw = legacy_payload.get("references", {}) if isinstance(legacy_payload, dict) else {}
        if not isinstance(raw, dict):
            logger.error("[rag] generation references are malformed at %s", legacy_path)
            return None
        references.update(
            {
                str(run_id): str(generation)
                for run_id, generation in raw.items()
                if str(run_id).strip() and str(generation).strip()
            }
        )

    references_dir = storage_root / GENERATION_REFERENCES_DIR_NAME
    if not references_dir.exists():
        return references
    try:
        paths = sorted(references_dir.glob("*.json"))
    except OSError:
        logger.error("[rag] generation reference directory is unreadable at %s", references_dir)
        return None
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.error("[rag] generation reference is unreadable at %s", path)
            return None
        if not isinstance(payload, dict):
            logger.error("[rag] generation reference is malformed at %s", path)
            return None
        run_id = str(payload.get("run_id") or "").strip()
        generation = str(payload.get("generation") or "").strip()
        if not run_id or not generation:
            logger.error("[rag] generation reference is malformed at %s", path)
            return None
        references[run_id] = generation
    return references


def _referenced_generations(storage_root: Path) -> set[str]:
    """Return immutable snapshots still required by durable autoresearch runs."""
    references = _generation_references(storage_root)
    if references is not None:
        return set(references.values())
    # A damaged retention ledger must not become permission to delete audit
    # evidence. Keep all completed snapshots until an operator repairs it.
    generations_root = storage_root / GENERATIONS_DIR_NAME
    try:
        return {
            child.name
            for child in generations_root.iterdir()
            if child.is_dir() and child.name.startswith("g-")
        }
    except OSError:
        return set()


def retain_index_generation(cfg: Config, generation: str | None, run_id: str) -> bool:
    """Durably retain a run's index snapshot without introducing another service."""
    resolved_generation = str(generation or "").strip()
    if not resolved_generation or resolved_generation == LEGACY_INDEX_GENERATION:
        return False
    storage_root = Path(get_llamaindex_config(cfg).storage_dir)
    if resolved_generation in {".", ".."} or Path(resolved_generation).name != resolved_generation:
        raise ValueError("invalid index generation")
    generation_root = storage_root / GENERATIONS_DIR_NAME / resolved_generation
    if not generation_root.is_dir():
        raise RuntimeError(f"cannot retain unavailable index generation {resolved_generation!r}")
    if _generation_references(storage_root) is None:
        raise RuntimeError("cannot update unreadable generation references")
    reference_name = hashlib.sha256(str(run_id).encode("utf-8")).hexdigest() + ".json"
    _write_json_atomically(
        storage_root / GENERATION_REFERENCES_DIR_NAME / reference_name,
        {"run_id": str(run_id), "generation": resolved_generation},
    )
    return True


def _prune_inactive_generations(
    storage_root: Path,
    active_generation: str,
    *,
    retain_count: int = GENERATION_RETAIN_COUNT,
    grace_seconds: float = GENERATION_PRUNE_GRACE_SECONDS,
    protected_generations: Iterable[str] = (),
) -> list[str]:
    """Bound completed snapshots while retaining a reader/rollback window.

    At least ``retain_count`` completed generations, including the active one,
    are kept. Older snapshots are only removed after a grace period, allowing
    in-flight readers that resolved a previous pointer to finish loading.
    Staging directories are never considered completed snapshots.
    """
    generations_root = storage_root / GENERATIONS_DIR_NAME
    if retain_count < 1 or not generations_root.is_dir():
        return []

    entries = list(generations_root.iterdir())
    completed = [child for child in entries if child.is_dir() and child.name.startswith("g-")]
    completed.sort(key=lambda child: child.stat().st_mtime, reverse=True)
    protected = {active_generation}
    protected.update(str(generation) for generation in protected_generations if str(generation))
    for child in completed:
        if len(protected) >= retain_count:
            break
        protected.add(child.name)

    cutoff = time.time() - max(0.0, grace_seconds)
    pruned: list[str] = []
    for child in completed:
        if child.name in protected or child.stat().st_mtime > cutoff:
            continue
        try:
            shutil.rmtree(child)
        except OSError as exc:
            logger.warning("[rag] could not prune stale generation %s: %s", child, exc)
        else:
            pruned.append(child.name)
    for child in entries:
        if not child.is_dir() or not child.name.startswith(".staging-g-"):
            continue
        if child.stat().st_mtime > cutoff:
            continue
        try:
            shutil.rmtree(child)
        except OSError as exc:
            logger.warning("[rag] could not prune stale staging directory %s: %s", child, exc)
        else:
            pruned.append(child.name)
    return pruned
