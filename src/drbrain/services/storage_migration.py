"""Deterministic migration planning for legacy artifacts (plan T50).

The plan is a pure function of its inputs: two scans of the same store produce
byte-identical plans (items sorted by paper id, no timestamps inside the item
list), and scanning never writes — the database bytes and the artifact
directory mtimes are untouched.  Every item carries the expected source
revision and content hash so ``migrate --apply`` can refuse to act on a source
that changed after the plan was made, and ambiguous legacy layouts become
explicit conflicts instead of a silent pick.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Actions a plan item can carry.
ACTIONS = ("import", "reuse", "recompute", "conflict", "skip")

_LEGACY_RAW = "raw.md"


@dataclass(frozen=True)
class MigrationItem:
    local_id: str
    action: str
    reason: str
    source: str = ""
    expected_revision: int | None = None
    expected_hash: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise ValueError(f"unsupported migration action {self.action!r}")

    def to_json(self) -> dict[str, Any]:
        return {
            "local_id": self.local_id,
            "action": self.action,
            "reason": self.reason,
            "source": self.source,
            "expected_revision": self.expected_revision,
            "expected_hash": self.expected_hash,
            "detail": dict(self.detail),
        }


@dataclass
class MigrationPlan:
    items: list[MigrationItem] = field(default_factory=list)
    inputs: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 0

    def summary(self) -> dict[str, int]:
        counts = {action: 0 for action in ACTIONS}
        for item in self.items:
            counts[item.action] += 1
        return counts

    def plan_id(self) -> str:
        payload = json.dumps(self.to_json(include_id=False), sort_keys=True, ensure_ascii=False)
        return "mig-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def to_json(self, *, include_id: bool = True) -> dict[str, Any]:
        document: dict[str, Any] = {
            "schema_version": self.schema_version,
            "summary": self.summary(),
            "items": [item.to_json() for item in self.items],
            "inputs": dict(self.inputs),
        }
        if include_id:
            document["plan_id"] = self.plan_id()
        return document


def _file_fingerprint(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError:
        return {"size": 0, "mtime_ns": 0}
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            while chunk := handle.read(1 << 20):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


def _directory_digest(papers_root: Path, directories: list[Path]) -> str:
    payload = []
    for directory in sorted(directories, key=lambda item: str(item)):
        entry: dict[str, object] = {"name": directory.name}
        for filename in (_LEGACY_RAW, "tree.json"):
            candidate = directory / filename
            if candidate.is_file():
                stat = candidate.stat()
                entry[filename] = [int(stat.st_size), int(stat.st_mtime_ns)]
        payload.append(entry)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def open_readonly(db_path: str | Path) -> sqlite3.Connection:
    """Read-only connection (query_only, never migrates)."""
    from drbrain.services.storage_audit import open_readonly as _open

    return _open(db_path)


def build_migration_plan(
    *,
    db_path: str | Path,
    papers_root: str | Path | None = None,
) -> MigrationPlan:
    """Derive the migration plan without touching the store."""
    from drbrain.storage.inbox import first_symlink_component
    from drbrain.storage.paths import paper_id_from_dir, resolve_paper_dir

    plan = MigrationPlan()
    db_file = Path(db_path)
    conn = open_readonly(db_file)
    try:
        # An uninitialized database (no schema yet) is a state, not a crash:
        # the plan then derives from the directory tree alone, and ``--apply``
        # initialises the schema before writing.
        try:
            versions = [
                int(row[0])
                for row in conn.execute(
                    "SELECT version FROM schema_versions ORDER BY version"
                ).fetchall()
            ]
        except sqlite3.OperationalError:
            versions = []
        plan.schema_version = versions[-1] if versions else 0
        try:
            papers = [
                str(row[0])
                for row in conn.execute("SELECT local_id FROM papers ORDER BY local_id").fetchall()
            ]
        except sqlite3.OperationalError:
            papers = []
        canonical: dict[str, dict[str, Any]] = {}
        if plan.schema_version >= 24:
            for row in conn.execute(
                """
                SELECT r.local_id, r.revision, r.canonical_hash, r.state,
                       COUNT(b.block_id) AS blocks,
                       COALESCE(SUM(b.char_end - b.char_start), 0) AS span
                FROM document_revisions r
                LEFT JOIN content_blocks b
                  ON b.local_id = r.local_id AND b.revision = r.revision
                GROUP BY r.local_id, r.revision
                ORDER BY r.local_id, r.revision
                """
            ).fetchall():
                canonical[str(row["local_id"])] = {
                    "revision": int(row["revision"]),
                    "canonical_hash": str(row["canonical_hash"]),
                    "state": str(row["state"]),
                    "blocks": int(row["blocks"]),
                    "span": int(row["span"]),
                }
    finally:
        conn.close()

    root = Path(papers_root) if papers_root is not None else None
    directories: list[Path] = []
    by_paper: dict[str, list[Path]] = {}
    if root is not None and root.is_dir() and first_symlink_component(root) is None:
        for candidate in sorted(
            (item for item in root.rglob("*") if item.is_dir()), key=lambda item: str(item)
        ):
            if first_symlink_component(candidate) is not None:
                continue
            if not (candidate / _LEGACY_RAW).is_file() and not (candidate / "tree.json").is_file():
                continue
            directories.append(candidate)
            try:
                local_id = paper_id_from_dir(candidate, root)
            except Exception:  # noqa: BLE001 - unnamed dirs cannot be planned
                continue
            by_paper.setdefault(local_id, []).append(candidate)

    all_ids = sorted(set(papers) | set(by_paper) | set(canonical))
    for local_id in all_ids:
        # A logical id that resolves to more than one legacy layout is a
        # conflict the plan must surface (never a silent pick).
        if root is not None:
            try:
                resolve_paper_dir(root, local_id)
            except ValueError as exc:
                if "ambiguous" in str(exc):
                    # ``_legacy_paper_dirs`` is the resolver's own definition of
                    # the legacy candidate set; reporting those candidates is
                    # what lets a human (or the apply step) decide.
                    try:
                        from drbrain.storage.paths import _legacy_paper_dirs

                        candidates = [str(path) for path in _legacy_paper_dirs(root, local_id)]
                    except Exception:  # noqa: BLE001 - fall back to grouped dirs
                        candidates = [str(path) for path in by_paper.get(local_id, [])]
                    plan.items.append(
                        MigrationItem(
                            local_id,
                            "conflict",
                            "ambiguous-layout",
                            detail={"directories": candidates, "error": str(exc)},
                        )
                    )
                    continue
        item = _plan_item(
            local_id,
            canonical=canonical.get(local_id),
            directories=by_paper.get(local_id, []),
            resolve=resolve_paper_dir,
            root=root,
        )
        plan.items.append(item)

    plan.inputs = {
        "db": str(db_file),
        "db_fingerprint": _file_fingerprint(db_file),
        "papers_root": str(root) if root is not None else "",
        "paper_dirs": len(directories),
        "dirs_digest": _directory_digest(root, directories) if root is not None else "",
    }
    return plan


def _plan_item(
    local_id: str,
    *,
    canonical: dict[str, Any] | None,
    directories: list[Path],
    resolve,
    root: Path | None,
) -> MigrationItem:
    if len(directories) > 1:
        return MigrationItem(
            local_id,
            "conflict",
            "ambiguous-layout",
            detail={"directories": [str(path) for path in directories]},
        )
    directory = directories[0] if directories else None
    raw_path = directory / _LEGACY_RAW if directory else None
    tree_path = directory / "tree.json" if directory else None
    raw_hash = _hash_file(raw_path) if raw_path and raw_path.is_file() else ""
    tree_ok = False
    tree_error = ""
    if tree_path and tree_path.is_file():
        try:
            payload = json.loads(tree_path.read_text(encoding="utf-8", errors="replace"))
            tree_ok = isinstance(payload, dict) and isinstance(payload.get("structure"), list)
            if not tree_ok:
                tree_error = "structure is not a list"
        except (OSError, ValueError) as exc:
            tree_error = str(exc)

    if canonical is not None and canonical["state"] == "ready" and canonical["blocks"] > 0:
        detail: dict[str, Any] = {"canonical_revision": canonical["revision"]}
        if directory is not None:
            detail["legacy_dir"] = str(directory)
            detail["legacy_tree_ok"] = tree_ok
        reason = "canonical-ready" + ("+legacy-copy" if directory is not None else "")
        return MigrationItem(
            local_id,
            "reuse",
            reason,
            source="canonical",
            expected_revision=canonical["revision"],
            expected_hash=canonical["canonical_hash"],
            detail=detail,
        )

    if canonical is not None and (canonical["state"] != "ready" or canonical["blocks"] == 0):
        # Re-deriving needs a verified legacy source: the guard must follow the
        # *source* file that will be imported, not the broken canonical hash.
        source_hash = raw_hash if raw_hash else canonical["canonical_hash"]
        return MigrationItem(
            local_id,
            "recompute",
            "canonical-inconsistent"
            if canonical["blocks"] == 0
            else f"canonical-{canonical['state']}",
            source="legacy-raw" if raw_hash else "canonical",
            expected_revision=canonical["revision"] + 1 if raw_hash else None,
            expected_hash=source_hash,
            detail={
                "blocks": canonical["blocks"],
                "legacy_dir": str(directory) if directory is not None else "",
                "legacy_source_hash": raw_hash,
            },
        )

    if directory is None:
        return MigrationItem(local_id, "skip", "no-artifacts")
    if raw_hash and tree_ok:
        return MigrationItem(
            local_id,
            "import",
            "legacy-raw+tree",
            source="legacy-raw",
            expected_revision=1,
            expected_hash=raw_hash,
            detail={"legacy_dir": str(directory)},
        )
    if raw_hash:
        return MigrationItem(
            local_id,
            "import",
            "legacy-raw",
            source="legacy-raw",
            expected_revision=1,
            expected_hash=raw_hash,
            detail={"legacy_dir": str(directory), "tree_error": tree_error or "absent"},
        )
    if tree_ok:
        return MigrationItem(
            local_id,
            "conflict",
            "tree-only",
            source="legacy-tree",
            detail={"legacy_dir": str(directory)},
        )
    return MigrationItem(
        local_id,
        "conflict",
        "unusable-legacy-material",
        detail={"legacy_dir": str(directory), "tree_error": tree_error or "absent"},
    )


def plan_rejects_concurrent_change(plan_item: MigrationItem, *, current_hash: str) -> bool:
    """Apply-time guard: the source must still hash to the planned value."""
    return bool(plan_item.expected_hash) and plan_item.expected_hash != current_hash


# ── applying a plan (T51) ───────────────────────────────────────────


@dataclass
class ApplyOutcome:
    job_id: str
    applied: list[str] = field(default_factory=list)
    reused: list[str] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    failed: list[dict[str, str]] = field(default_factory=list)
    remaining: list[str] = field(default_factory=list)
    created_records: list[str] = field(default_factory=list)
    paused: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "applied": list(self.applied),
            "reused": list(self.reused),
            "skipped": list(self.skipped),
            "failed": list(self.failed),
            "remaining": list(self.remaining),
            "created_records": list(self.created_records),
            "paused": self.paused,
            "counts": {
                "applied": len(self.applied),
                "reused": len(self.reused),
                "skipped": len(self.skipped),
                "failed": len(self.failed),
                "remaining": len(self.remaining),
                "created_records": len(self.created_records),
            },
        }


def _title_from_body(text: str) -> str:
    """Best-effort title for a placeholder record (first non-empty line)."""
    for line in text.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return stripped[:200]
    return ""


def _ensure_paper_record(db, local_id: str, text: str, outcome: ApplyOutcome) -> None:
    """Create the minimal library record for a directory-only paper.

    The plan classified this directory as an importable legacy source; the
    record is created as a *placeholder* (never as a verified paper), so
    nothing here invents provenance beyond what the directory provides.
    """
    if db.get_paper(local_id) is not None:
        return
    db.insert_paper(local_id, _title_from_body(text), None, "placeholder")
    db.commit()
    outcome.created_records.append(local_id)


def _legacy_raw_path(root: Path, local_id: str) -> Path | None:
    """Resolve the legacy raw.md for an item (canonical or legacy layout)."""
    from drbrain.storage.paths import resolve_paper_dir

    try:
        directory = resolve_paper_dir(root, local_id)
    except ValueError:
        return None
    if directory is None:
        return None
    candidate = directory / _LEGACY_RAW
    return candidate if candidate.is_file() else None


def _load_checkpoint(store, job_id: str, owner: str):
    claim = store.claim(job_id, owner)
    checkpoint = claim.checkpoint or {}
    done = {str(item) for item in checkpoint.get("done", [])}
    failed = {str(key): str(value) for key, value in (checkpoint.get("failed") or {}).items()}
    return claim, done, failed


def apply_migration_plan(
    db,
    plan: MigrationPlan,
    *,
    papers_root: str | Path,
    job_store=None,
    owner: str | None = None,
    max_items: int | None = None,
    count_tokens=None,
    on_progress=None,
) -> ApplyOutcome:
    """Apply a verified plan one paper at a time, resumable and idempotent.

    * the source file must still hash to the planned value or the item is
      refused (``source-changed``) instead of importing unverified content;
    * every paper is written in its own transaction, so one failure cannot
      roll back another, and failures are recorded, never silently skipped;
    * progress is checkpointed after each item: an interrupted run resumes
      from the checkpoint and re-running a finished plan is a no-op;
    * nothing in the legacy tree is moved or deleted.
    """
    import uuid as _uuid
    from pathlib import Path as _Path

    from drbrain.services.canonical_content import write_canonical_content
    from drbrain.tree.jobs import TreeJobStore

    root = _Path(papers_root)
    store = job_store or TreeJobStore(db)
    owner = owner or f"migrate-{_uuid.uuid4().hex[:8]}"
    scope_key = plan.plan_id()
    # Resume the newest unfinished job for this exact plan, else start one.
    job_id = ""
    for job in db.list_tree_jobs(limit=50):
        if str(job["scope_key"]) == scope_key and str(job["state"]) in {
            "pending",
            "running",
            "paused",
        }:
            job_id = str(job["job_id"])
            break
    if not job_id:
        job_id = store.create(scope_key, kind="migrate")

    claim, done, failed_map = _load_checkpoint(store, job_id, owner)
    if not claim.granted:
        raise RuntimeError(f"migration job {job_id} is already claimed by another worker")

    outcome = ApplyOutcome(job_id=job_id)
    outcome.failed = [
        {"local_id": local_id, "error": reason} for local_id, reason in sorted(failed_map.items())
    ]
    processed = 0
    for item in plan.items:
        if item.local_id in done or item.local_id in failed_map:
            if item.action == "reuse":
                outcome.reused.append(item.local_id)
            continue
        if max_items is not None and processed >= max_items:
            outcome.paused = True
            break
        processed += 1
        if item.action in {"skip", "conflict"}:
            outcome.skipped.append({"local_id": item.local_id, "reason": item.reason})
            done.add(item.local_id)
            if on_progress is not None:
                on_progress(item, "skipped")
            store.save_checkpoint(
                job_id,
                owner=owner,
                checkpoint={"done": sorted(done), "failed": failed_map},
            )
            continue
        if item.action == "reuse":
            outcome.reused.append(item.local_id)
            done.add(item.local_id)
            store.save_checkpoint(
                job_id,
                owner=owner,
                checkpoint={"done": sorted(done), "failed": failed_map},
            )
            continue

        raw_path = _legacy_raw_path(root, item.local_id) if root.is_dir() else None
        if raw_path is None:
            outcome.failed.append({"local_id": item.local_id, "error": "no-legacy-source"})
            failed_map[item.local_id] = "no-legacy-source"
            store.save_checkpoint(
                job_id,
                owner=owner,
                checkpoint={"done": sorted(done), "failed": failed_map},
            )
            continue
        current_hash = _hash_file(raw_path)
        if plan_rejects_concurrent_change(item, current_hash=current_hash):
            outcome.failed.append({"local_id": item.local_id, "error": "source-changed"})
            failed_map[item.local_id] = "source-changed"
            store.save_checkpoint(
                job_id,
                owner=owner,
                checkpoint={"done": sorted(done), "failed": failed_map},
            )
            if on_progress is not None:
                on_progress(item, "source-changed")
            continue

        try:
            text = raw_path.read_text(encoding="utf-8", errors="replace")
            _ensure_paper_record(db, item.local_id, text, outcome)
            result = write_canonical_content(
                db,
                item.local_id,
                text,
                media_type="md",
                source_hash=current_hash,
                parser="legacy-migration",
                stale_previous=item.action == "recompute",
            )
            if not result.get("ok"):
                raise RuntimeError(str(result.get("reason") or "write-failed"))
        except Exception as exc:  # noqa: BLE001 - one paper must not stop the run
            outcome.failed.append({"local_id": item.local_id, "error": str(exc)})
            failed_map[item.local_id] = str(exc)
        else:
            if result.get("reused"):
                outcome.reused.append(item.local_id)
            else:
                outcome.applied.append(item.local_id)
            done.add(item.local_id)
        store.save_checkpoint(
            job_id,
            owner=owner,
            checkpoint={"done": sorted(done), "failed": failed_map},
            metrics={"applied": len(outcome.applied), "failed": len(outcome.failed)},
        )
        if on_progress is not None:
            on_progress(item, "applied" if item.local_id in done else "failed")
    del count_tokens  # reserved for token-aware batching in T60

    outcome.remaining = [
        item.local_id
        for item in plan.items
        if item.local_id not in done and item.local_id not in failed_map
    ]
    if outcome.paused:
        store.pause(job_id, reason="max_items reached")
    elif outcome.remaining:
        store.pause(job_id, reason="leftover items")
    else:
        store.finish(job_id, done=not outcome.failed, reason="applied")
    return outcome
