---
name: backup
description: >
  Create and manage backups of DrBrain data — local tar.gz archives and rsync remote sync.
  Use when the user wants to back up their library, sync data to a remote server,
  list existing backups, or set up automated backup workflows. Trigger on "back up my data",
  "save my library", "sync to server", "create a backup".
---

# Backup

Two backup modes: local tar.gz archives (always available) and rsync to remote targets.

## Local tar.gz backup

```bash
drbrain backup                        # create backup in data/backups/
drbrain backup --list                 # list existing backups
drbrain backup --output custom.tar.gz # custom output path
```

Backs up: papers, database, workspace, and reports.

## Rsync remote backup

Configure targets in `config.local.yaml`:

```yaml
backup:
  targets:
    myserver:
      host: backup.example.com
      user: drbrain
      path: /backups/drbrain/
      port: 22
      identity_file: "~/.ssh/id_ed25519"
      compress: true
      exclude: []
```

Then sync:

```bash
drbrain backup --list                     # shows both local + remote targets
drbrain backup --target myserver          # rsync to remote
drbrain backup --target myserver --dry-run # preview without transfer
```

## CLI Reference

| Command | What it does |
|---------|--------------|
| `drbrain backup` | Create local tar.gz backup |
| `drbrain backup --list` | List backups and rsync targets |
| `drbrain backup --output <path>` | Custom output path |
| `drbrain backup --target <name>` | Rsync to remote target |
| `drbrain backup --target <name> --dry-run` | Preview rsync |
