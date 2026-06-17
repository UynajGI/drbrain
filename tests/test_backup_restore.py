"""Tests for backup restore command and restore_backup helper."""

from pathlib import Path

import pytest

from drbrain.storage.backup import (
    create_backup,
    restore_backup,
)


def _make_backup(tmp_path) -> Path:
    """Create a small tar.gz backup and return its path."""
    papers = tmp_path / "source_papers"
    papers.mkdir()
    (papers / "p1" / "meta.json").parent.mkdir(parents=True)
    (papers / "p1" / "meta.json").write_text('{"title": "Paper 1"}')
    (papers / "p2" / "meta.json").parent.mkdir(parents=True)
    (papers / "p2" / "meta.json").write_text('{"title": "Paper 2"}')

    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True)
    db_file = db_dir / "drbrain.db"
    db_file.write_text("sqlite-data")

    backup_dir = tmp_path / "backups"
    return create_backup(
        papers_dir=papers,
        db_path=db_file,
        backup_dir=backup_dir,
    )


class TestRestoreTarball:
    def test_restore_creates_files(self, tmp_path):
        """restore_backup from tar.gz recreates original files."""
        archive = _make_backup(tmp_path)
        target = tmp_path / "restored"
        entries = restore_backup(archive, target)

        assert "papers" in entries or "db" in entries
        # Verify at least one file was restored
        restored_files = list(target.rglob("*"))
        assert len(restored_files) > 0

    def test_restore_file_contents_match(self, tmp_path):
        """Restored file contents match the originals."""
        archive = _make_backup(tmp_path)
        target = tmp_path / "restored"
        restore_backup(archive, target)

        # Check DB file
        restored_db = target / "db" / "drbrain.db"
        assert restored_db.exists()
        assert restored_db.read_text() == "sqlite-data"

        # Check a paper meta
        restored_meta = target / "papers" / "p1" / "meta.json"
        assert restored_meta.exists()
        import json

        data = json.loads(restored_meta.read_text())
        assert data["title"] == "Paper 1"

    def test_restore_refuses_newer_without_force(self, tmp_path):
        """restore_backup refuses to overwrite newer files unless --force."""
        archive = _make_backup(tmp_path)
        target = tmp_path / "restored"
        restore_backup(archive, target)

        # Touch the db file so it's "newer"
        (target / "db" / "drbrain.db").touch()

        with pytest.raises(FileExistsError, match="newer than backup"):
            restore_backup(archive, target, force=False)

    def test_restore_force_overwrites(self, tmp_path):
        """restore_backup with force=True overwrites newer files."""
        archive = _make_backup(tmp_path)
        target = tmp_path / "restored"
        restore_backup(archive, target)

        # Corrupt a file
        (target / "db" / "drbrain.db").write_text("corrupted")

        # Force restore should fix it
        restore_backup(archive, target, force=True)
        assert (target / "db" / "drbrain.db").read_text() == "sqlite-data"

    def test_restore_nonexistent_raises(self, tmp_path):
        """restore_backup raises FileNotFoundError for missing path."""
        with pytest.raises(FileNotFoundError, match="not found"):
            restore_backup(tmp_path / "nonexistent.tar.gz")


class TestRestoreDirectory:
    def test_restore_from_directory(self, tmp_path):
        """restore_backup copies a directory backup."""
        source = tmp_path / "dir_backup"
        source.mkdir()
        (source / "papers").mkdir()
        (source / "papers" / "p1.txt").write_text("paper-data")
        (source / "db").mkdir()
        (source / "db" / "drbrain.db").write_text("db-data")

        target = tmp_path / "restored"
        entries = restore_backup(source, target)

        assert "papers" in entries
        assert "db" in entries
        assert (target / "papers" / "p1.txt").read_text() == "paper-data"
        assert (target / "db" / "drbrain.db").read_text() == "db-data"

    def test_restore_directory_refuses_existing(self, tmp_path):
        """restore_backup refuses to overwrite existing files without --force."""
        source = tmp_path / "dir_backup"
        source.mkdir()
        (source / "data.txt").write_text("backup")

        target = tmp_path / "restored"
        target.mkdir()
        (target / "data.txt").write_text("existing")

        with pytest.raises(FileExistsError, match="already exists"):
            restore_backup(source, target, force=False)

    def test_restore_directory_force(self, tmp_path):
        """restore_backup with force overwrites existing directory entries."""
        source = tmp_path / "dir_backup"
        source.mkdir()
        (source / "data.txt").write_text("new")

        target = tmp_path / "restored"
        target.mkdir()
        (target / "data.txt").write_text("old")

        restore_backup(source, target, force=True)
        assert (target / "data.txt").read_text() == "new"


class TestRestoreCommand:
    def test_restore_cmd_runs(self, tmp_path, monkeypatch):
        """The CLI restore_cmd can be invoked via typer runner."""
        from typer.testing import CliRunner

        from drbrain.cli.main import app

        # Create a backup
        archive = _make_backup(tmp_path)
        target = tmp_path / "cli_restored"

        runner = CliRunner()
        result = runner.invoke(app, ["restore", str(archive), "--target", str(target)])
        assert result.exit_code == 0
        assert "Restored" in result.output
        assert (target / "db" / "drbrain.db").exists()

    def test_restore_cmd_json_output(self, tmp_path, monkeypatch):
        """The CLI restore_cmd supports --json output."""
        from typer.testing import CliRunner

        from drbrain.cli.main import app

        archive = _make_backup(tmp_path)
        target = tmp_path / "cli_restored_json"

        runner = CliRunner()
        result = runner.invoke(app, ["restore", str(archive), "--target", str(target), "--json"])
        assert result.exit_code == 0
        import json

        data = json.loads(result.output)
        assert "restored" in data
