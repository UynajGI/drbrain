"""Tests for backup creation."""

import tarfile

from drbrain.storage.backup import (
    _backup_filename,
    create_backup,
    list_backups,
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
