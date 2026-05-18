"""Tests for backup creation and rsync remote sync."""

import tarfile

import pytest

from drbrain.config import BackupConfig, BackupTargetConfig
from drbrain.storage.backup import (
    BackupConfigError,
    _backup_filename,
    build_rsync_command,
    create_backup,
    list_backups,
    run_backup,
)


def test_backup_filename():
    """Backup filename follows expected pattern."""
    name = _backup_filename()
    assert name.startswith("drbrain-")
    assert name.endswith(".tar.gz")


def test_create_backup_basic(tmp_path):
    """create_backup creates a tar.gz with the expected contents."""
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    (papers_dir / "dummy.txt").write_text("data")

    db_dir = tmp_path / "db"
    db_dir.mkdir()
    (db_dir / "drbrain.db").write_text("sqlite")

    backups_dir = tmp_path / "backups"

    path = create_backup(
        papers_dir=papers_dir,
        db_path=db_dir / "drbrain.db",
        backup_dir=backups_dir,
    )

    assert path.exists()
    assert path.suffix == ".gz"
    assert path.parent == backups_dir

    with tarfile.open(path, "r:gz") as tar:
        names = tar.getnames()
        assert any("dummy.txt" in n for n in names)
        assert any("drbrain.db" in n for n in names)


def test_create_backup_excludes_cache(tmp_path):
    """Backup does not include cache or log directories."""
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    (papers_dir / "important.md").write_text("keep me")

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "cached.bin").write_text("xxxxx")

    backups_dir = tmp_path / "backups"

    path = create_backup(
        papers_dir=papers_dir,
        db_path=tmp_path / "db" / "drbrain.db",
        backup_dir=backups_dir,
    )

    with tarfile.open(path, "r:gz") as tar:
        names = tar.getnames()
        assert any("important.md" in n for n in names)
        assert not any("cached.bin" in n for n in names)


def test_list_backups_empty(tmp_path):
    """Empty backup directory returns empty list."""
    assert list_backups(tmp_path) == []


def test_list_backups_with_files(tmp_path):
    """list_backups returns sorted list of backup paths (newest first)."""
    (tmp_path / "drbrain-2026-01-01-000000.tar.gz").touch()
    (tmp_path / "drbrain-2026-06-15-120000.tar.gz").touch()
    (tmp_path / "not-a-backup.txt").touch()

    backups = list_backups(tmp_path)
    assert len(backups) == 2
    assert "06-15" in backups[0].name


def test_create_backup_with_workspace_dir(tmp_path):
    """create_backup includes workspace files when workspace_dir is provided."""
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    (papers_dir / "dummy.txt").write_text("data")

    db_dir = tmp_path / "db"
    db_dir.mkdir()
    (db_dir / "drbrain.db").write_text("sqlite")

    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (workspace_dir / "config.yaml").write_text("workspace: test")
    (workspace_dir / "papers.json").write_text('["p1"]')

    backups_dir = tmp_path / "backups"

    path = create_backup(
        papers_dir=papers_dir,
        db_path=db_dir / "drbrain.db",
        backup_dir=backups_dir,
        workspace_dir=workspace_dir,
    )

    assert path.exists()
    with tarfile.open(path, "r:gz") as tar:
        names = tar.getnames()
        assert any("workspace/config.yaml" in n for n in names)
        assert any("workspace/papers.json" in n for n in names)


def test_create_backup_with_reports_dir(tmp_path):
    """create_backup includes reports when reports_dir is provided."""
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    (papers_dir / "dummy.txt").write_text("data")

    db_dir = tmp_path / "db"
    db_dir.mkdir()
    (db_dir / "drbrain.db").write_text("sqlite")

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "report.json").write_text('{"key": "val"}')

    backups_dir = tmp_path / "backups"

    path = create_backup(
        papers_dir=papers_dir,
        db_path=db_dir / "drbrain.db",
        backup_dir=backups_dir,
        reports_dir=reports_dir,
    )

    assert path.exists()
    with tarfile.open(path, "r:gz") as tar:
        names = tar.getnames()
        assert any("reports/report.json" in n for n in names)


def test_create_backup_missing_papers_dir(tmp_path):
    """create_backup does not crash when papers_dir does not exist."""
    papers_dir = tmp_path / "nonexistent_papers"

    backups_dir = tmp_path / "backups"

    path = create_backup(
        papers_dir=papers_dir,
        db_path=tmp_path / "nonexistent" / "drbrain.db",
        backup_dir=backups_dir,
    )

    assert path.exists()
    # archive should be created but (mostly) empty
    with tarfile.open(path, "r:gz") as tar:
        names = tar.getnames()
        assert len(names) == 0


# ── Rsync remote backup ────────────────────────────────────────────

SAMPLE_TARGET = BackupTargetConfig(
    host="backup.example.com",
    user="drbrain",
    path="/backups/drbrain/",
    port=22,
    identity_file="~/.ssh/id_ed25519",
    compress=True,
    enabled=True,
)


class TestResolveTarget:
    def test_valid_target(self):
        from drbrain.storage.backup import _resolve_target

        targets = {"myserver": SAMPLE_TARGET}
        t = _resolve_target(targets, "myserver")
        assert t.host == "backup.example.com"

    def test_unknown_target(self):
        from drbrain.storage.backup import _resolve_target

        with pytest.raises(BackupConfigError, match="Unknown backup target"):
            _resolve_target({}, "missing")

    def test_disabled_target(self):
        from drbrain.storage.backup import _resolve_target

        disabled = BackupTargetConfig(host="x.com", path="/x", enabled=False)
        with pytest.raises(BackupConfigError, match="disabled"):
            _resolve_target({"t": disabled}, "t")

    def test_missing_host(self):
        from drbrain.storage.backup import _resolve_target

        no_host = BackupTargetConfig(path="/x")
        with pytest.raises(BackupConfigError, match="missing host"):
            _resolve_target({"t": no_host}, "t")

    def test_missing_path(self):
        from drbrain.storage.backup import _resolve_target

        no_path = BackupTargetConfig(host="x.com")
        with pytest.raises(BackupConfigError, match="missing path"):
            _resolve_target({"t": no_path}, "t")


class TestBuildRsyncCommand:
    def test_basic_command_structure(self):
        targets = {"myserver": SAMPLE_TARGET}
        cmd = build_rsync_command(
            rsync_bin="rsync",
            ssh_bin="ssh",
            targets=targets,
            source_dir="/data/drbrain",
            target_name="myserver",
        )
        assert cmd[0] == "rsync"
        assert "-a" in cmd
        assert "--stats" in cmd
        assert "-z" in cmd
        assert "-e" in cmd
        assert "/data/drbrain/" in cmd
        assert "drbrain@backup.example.com:/backups/drbrain/" in cmd

    def test_dry_run_flag(self):
        targets = {"myserver": SAMPLE_TARGET}
        cmd = build_rsync_command(
            rsync_bin="rsync",
            ssh_bin="ssh",
            targets=targets,
            source_dir="/data",
            target_name="myserver",
            dry_run=True,
        )
        assert "--dry-run" in cmd

    def test_append_mode(self):
        t = BackupTargetConfig(host="x.com", path="/x", mode="append", compress=False)
        cmd = build_rsync_command(
            rsync_bin="rsync",
            ssh_bin="ssh",
            targets={"t": t},
            source_dir="/d",
            target_name="t",
        )
        assert "--append" in cmd
        assert "-z" not in cmd

    def test_append_verify_mode(self):
        t = BackupTargetConfig(host="x.com", path="/x", mode="append-verify", compress=False)
        cmd = build_rsync_command(
            rsync_bin="rsync",
            ssh_bin="ssh",
            targets={"t": t},
            source_dir="/d",
            target_name="t",
        )
        assert "--append-verify" in cmd

    def test_exclude_patterns(self):
        t = BackupTargetConfig(
            host="x.com",
            path="/x",
            exclude=[".git", "*.tmp"],
            compress=False,
        )
        cmd = build_rsync_command(
            rsync_bin="rsync",
            ssh_bin="ssh",
            targets={"t": t},
            source_dir="/d",
            target_name="t",
        )
        assert "--exclude" in cmd
        assert ".git" in cmd
        assert "*.tmp" in cmd

    def test_ssh_batch_mode_default(self):
        targets = {"myserver": SAMPLE_TARGET}
        cmd = build_rsync_command(
            rsync_bin="rsync",
            ssh_bin="ssh",
            targets=targets,
            source_dir="/d",
            target_name="myserver",
        )
        ssh_str = cmd[cmd.index("-e") + 1]
        assert "BatchMode=yes" in ssh_str

    def test_ssh_password_mode(self):
        t = BackupTargetConfig(
            host="x.com",
            path="/x",
            password="secret",
            compress=False,
        )
        cmd = build_rsync_command(
            rsync_bin="rsync",
            ssh_bin="ssh",
            targets={"t": t},
            source_dir="/d",
            target_name="t",
        )
        ssh_str = cmd[cmd.index("-e") + 1]
        assert "PubkeyAuthentication=no" in ssh_str

    def test_custom_port(self):
        t = BackupTargetConfig(host="x.com", path="/x", port=2222, compress=False)
        cmd = build_rsync_command(
            rsync_bin="rsync",
            ssh_bin="ssh",
            targets={"t": t},
            source_dir="/d",
            target_name="t",
        )
        ssh_str = cmd[cmd.index("-e") + 1]
        assert "2222" in ssh_str

    def test_identity_file(self):
        t = BackupTargetConfig(
            host="x.com",
            path="/x",
            identity_file="~/.ssh/key",
            compress=False,
        )
        cmd = build_rsync_command(
            rsync_bin="rsync",
            ssh_bin="ssh",
            targets={"t": t},
            source_dir="/d",
            target_name="t",
        )
        ssh_str = cmd[cmd.index("-e") + 1]
        assert ".ssh/key" in ssh_str


class TestRunBackup:
    def test_missing_rsync_binary(self):
        t = BackupTargetConfig(host="localhost", path="/tmp")
        with pytest.raises(BackupConfigError, match="Failed to execute rsync"):
            run_backup(
                rsync_bin="/nonexistent/rsync",
                ssh_bin="ssh",
                targets={"t": t},
                source_dir="/tmp",
                target_name="t",
            )

    def test_dry_run_with_missing_binary(self):
        t = BackupTargetConfig(host="localhost", path="/tmp")
        with pytest.raises(BackupConfigError, match="Failed to execute rsync"):
            run_backup(
                rsync_bin="/nonexistent/rsync",
                ssh_bin="ssh",
                targets={"t": t},
                source_dir="/tmp",
                target_name="t",
                dry_run=True,
            )


class TestBackupConfigDefaults:
    def test_default_backup_config(self):
        cfg = BackupConfig()
        assert cfg.ssh_bin == "ssh"
        assert cfg.rsync_bin == "rsync"
        assert cfg.targets == {}

    def test_default_target_config(self):
        t = BackupTargetConfig()
        assert t.port == 22
        assert t.mode == "default"
        assert t.compress is True
        assert t.enabled is True
        assert t.exclude == []
