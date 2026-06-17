"""Backup creation via tar.gz archive and rsync remote sync."""

from __future__ import annotations

import os as _os
import shlex as _shlex
import subprocess as _subprocess
import tarfile
import tempfile as _tempfile
from dataclasses import dataclass as _dataclass
from datetime import datetime
from pathlib import Path

from loguru import logger

from drbrain.config import BackupTargetConfig

BACKUP_DIR = "data/backups"


def _backup_filename() -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"drbrain-{ts}.tar.gz"


def create_backup(
    papers_dir: Path,
    db_path: Path,
    backup_dir: Path | None = None,
    workspace_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> Path:
    """Create a tar.gz backup of papers, DB, workspace, and reports.

    Args:
        papers_dir: Path to data/papers/.
        db_path: Path to drbrain.db.
        backup_dir: Output directory for the archive (default: data/backups).
        workspace_dir: Optional path to workspace/.
        reports_dir: Optional path to data/reports/.

    Returns:
        Path to the created archive.
    """
    backup_dir = Path(backup_dir or BACKUP_DIR)
    backup_dir.mkdir(parents=True, exist_ok=True)

    out_path = backup_dir / _backup_filename()

    with tarfile.open(out_path, "w:gz") as tar:
        if papers_dir.exists():
            tar.add(str(papers_dir), arcname="papers")
        if db_path.exists():
            tar.add(str(db_path), arcname="db/drbrain.db")
        if workspace_dir and workspace_dir.exists():
            for item in workspace_dir.iterdir():
                tar.add(str(item), arcname=f"workspace/{item.name}")
        if reports_dir and reports_dir.exists():
            tar.add(str(reports_dir), arcname="reports")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info("[backup] created %s (%.1f MB)", out_path.name, size_mb)
    return out_path


def list_backups(backup_dir: Path | None = None) -> list[Path]:
    """Return sorted list of backup files (newest first)."""
    d = Path(backup_dir or BACKUP_DIR)
    if not d.exists():
        return []
    backups = sorted(
        d.glob("drbrain-*.tar.gz"),
        key=lambda p: p.name,
        reverse=True,
    )
    return backups


# ── Rsync remote backup ────────────────────────────────────────────


class BackupConfigError(ValueError):
    """Raised when backup configuration is missing or invalid."""


@_dataclass
class BackupRunResult:
    """Structured result from a backup invocation."""

    command: list[str]
    returncode: int
    stdout: str
    stderr: str


def _resolve_target(targets: dict[str, BackupTargetConfig], target_name: str) -> BackupTargetConfig:
    target = targets.get(target_name)
    if target is None:
        raise BackupConfigError(f"Unknown backup target: {target_name}")
    if not target.enabled:
        raise BackupConfigError(f"Backup target is disabled: {target_name}")
    if not target.host:
        raise BackupConfigError(f"Backup target '{target_name}' is missing host")
    if not target.path:
        raise BackupConfigError(f"Backup target '{target_name}' is missing path")
    return target


def _resolve_identity_file(identity_file: str) -> str:
    if not identity_file:
        return ""
    return str(Path(identity_file).expanduser())


def _build_remote_shell(ssh_bin: str, target: BackupTargetConfig) -> str:
    parts = [ssh_bin]
    if target.password:
        parts.extend(
            [
                "-o",
                "PreferredAuthentications=password,keyboard-interactive",
                "-o",
                "PubkeyAuthentication=no",
            ]
        )
    else:
        parts.extend(["-o", "BatchMode=yes"])
    if target.port and target.port != 22:
        parts.extend(["-p", str(target.port)])
    identity_file = _resolve_identity_file(target.identity_file)
    if identity_file:
        parts.extend(["-i", identity_file])
    return _shlex.join(parts)


def _build_password_env(
    target: BackupTargetConfig,
) -> tuple[dict[str, str], str] | tuple[None, None]:
    if not target.password:
        return None, None

    fd, askpass_path = _tempfile.mkstemp(prefix="drbrain-backup-askpass-", text=True)
    try:
        with _os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write("#!/bin/sh\nprintf '%s\\n' \"$DRBRAIN_BACKUP_SSH_PASSWORD\"\n")
        _os.chmod(askpass_path, 0o700)
    except Exception:
        try:
            _os.unlink(askpass_path)
        except OSError:
            pass
        raise

    env = _os.environ.copy()
    env.update(
        {
            "DRBRAIN_BACKUP_SSH_PASSWORD": target.password,
            "SSH_ASKPASS": askpass_path,
            "SSH_ASKPASS_REQUIRE": "force",
            "DISPLAY": "drbrain-backup",
        }
    )
    return env, askpass_path


def _destination_for(target: BackupTargetConfig) -> str:
    remote = f"{target.user}@{target.host}" if target.user else target.host
    return f"{remote}:{target.path.rstrip('/')}/"


def build_rsync_command(
    rsync_bin: str,
    ssh_bin: str,
    targets: dict[str, BackupTargetConfig],
    source_dir: str,
    target_name: str,
    *,
    dry_run: bool = False,
) -> list[str]:
    """Build the rsync command line for a configured backup target."""
    target = _resolve_target(targets, target_name)
    cmd = [rsync_bin, "-a", "--stats", "--human-readable"]
    if target.compress:
        cmd.append("-z")
    if target.mode == "append":
        cmd.append("--append")
    elif target.mode == "append-verify":
        cmd.append("--append-verify")
    if dry_run:
        cmd.append("--dry-run")
    for pattern in target.exclude:
        cmd.extend(["--exclude", pattern])
    cmd.extend(["-e", _build_remote_shell(ssh_bin, target)])
    cmd.append(f"{source_dir.rstrip('/')}/")
    cmd.append(_destination_for(target))
    return cmd


def run_backup(
    rsync_bin: str,
    ssh_bin: str,
    targets: dict[str, BackupTargetConfig],
    source_dir: str,
    target_name: str,
    *,
    dry_run: bool = False,
) -> BackupRunResult:
    """Run an rsync backup for a configured target."""
    target = _resolve_target(targets, target_name)
    cmd = build_rsync_command(
        rsync_bin,
        ssh_bin,
        targets,
        source_dir,
        target_name,
        dry_run=dry_run,
    )
    env, askpass_path = _build_password_env(target)
    run_kwargs: dict = {"check": False, "text": True, "capture_output": True}
    if env is not None:
        run_kwargs["env"] = env
        run_kwargs["stdin"] = _subprocess.DEVNULL
    try:
        completed = _subprocess.run(cmd, **run_kwargs)
    except OSError as exc:
        detail = exc.strerror or str(exc)
        raise BackupConfigError(f"Failed to execute rsync {cmd[0]!r}: {detail}") from exc
    finally:
        if askpass_path:
            try:
                _os.unlink(askpass_path)
            except OSError:
                pass
    return BackupRunResult(
        command=cmd,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


# ── Restore ───────────────────────────────────────────────────────────


def restore_backup(
    backup_path: Path,
    target_dir: Path | None = None,
    *,
    force: bool = False,
) -> list[str]:
    """Restore a tar.gz backup or copy a directory backup to *target_dir*.

    For tar.gz archives, the archive is extracted into *target_dir*.
    For directory backups, contents are copied recursively.

    Args:
        backup_path: Path to a ``.tar.gz`` file or a directory.
        target_dir: Where to restore.  If ``None`` the archive members
            are extracted relative to the current working directory.
        force: When ``False``, refuse to overwrite existing files that are
            newer than the corresponding archive entry.

    Returns:
        A list of top-level entries that were restored.
    """
    if not backup_path.exists():
        raise FileNotFoundError(f"Backup not found: {backup_path}")

    if backup_path.is_file() and backup_path.suffix == ".gz":
        return _restore_tarball(backup_path, target_dir, force=force)
    if backup_path.is_dir():
        return _restore_directory(backup_path, target_dir, force=force)

    raise ValueError(f"Unsupported backup format: {backup_path}")


def _restore_tarball(
    archive: Path,
    target_dir: Path | None,
    *,
    force: bool = False,
) -> list[str]:
    """Extract a tar.gz backup into *target_dir*."""

    target = Path(target_dir) if target_dir is not None else Path.cwd()

    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()

        # Safety check: refuse to overwrite newer files unless --force
        if not force:
            for member in members:
                if not member.isfile():
                    continue
                dest = target / member.name
                if dest.exists() and dest.stat().st_mtime > member.mtime:
                    raise FileExistsError(
                        f"File is newer than backup: {member.name}. Use --force to overwrite."
                    )

        # Extract
        target.mkdir(parents=True, exist_ok=True)
        tar.extractall(path=str(target), filter="data")  # noqa: S202

        # Return top-level entries
        top_level = {m.name.split("/")[0] for m in members}
        return sorted(top_level)


def _restore_directory(
    source: Path,
    target_dir: Path | None,
    *,
    force: bool = False,
) -> list[str]:
    """Copy all contents of a directory backup to *target_dir*."""
    import shutil as _shutil

    target = Path(target_dir) if target_dir is not None else Path.cwd()
    target.mkdir(parents=True, exist_ok=True)

    entries: list[str] = []
    for item in source.iterdir():
        dest = target / item.name
        if dest.exists() and not force:
            raise FileExistsError(f"Target already exists: {item.name}. Use --force to overwrite.")
        if item.is_dir():
            _shutil.copytree(str(item), str(dest), dirs_exist_ok=force)
        else:
            _shutil.copy2(str(item), str(dest))
        entries.append(item.name)

    return sorted(entries)
