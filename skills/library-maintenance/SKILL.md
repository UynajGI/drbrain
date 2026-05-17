---
name: library-maintenance
description: >
  Manage the DrBrain library — setup, health checks, statistics, backups, cleanup, paper deletion,
  confidence queue resolution, and author lineage. Use this skill whenever the user asks to
  "check my setup", "show library stats", "how many papers do I have?", "backup my data",
  "clean up", "delete a paper", "resolve conflicts", "show me a report", "trace author lineage",
  or needs to perform library housekeeping. Also use when the user is setting up DrBrain for the
  first time, troubleshooting configuration issues, wants to free disk space, or needs to inspect
  paper-level details. Trigger proactively for any library management or maintenance task.
---

# Library Maintenance

Manage the DrBrain library: setup, diagnostics, statistics, reporting, cleanup, and data lifecycle.

## Operations

### setup — First-time initialization

Generate config, create data directories, validate environment:

```bash
drbrain setup                  # interactive
drbrain setup --quick          # skip prompts, use defaults
drbrain setup --change-password
```

### check — Environment diagnostics

Verify dependencies, configuration, and environment variables:

```bash
drbrain check
```

Run this first when something breaks. Checks: Python deps, MinerU, API keys, DB integrity,
directory structure.

### stats — Library statistics

Overview of the database:

```bash
drbrain stats
drbrain stats --json
drbrain stats -w my-workspace
```

Shows: paper count, concept count, edge count, embedding status, DB size, last build time.

### report — Single-paper summary

Detailed report for one paper:

```bash
drbrain report p3f8a2
drbrain report p3f8a2 --json
```

Includes: metadata, concept list, edge summary, build stage status, quality flags.

### delete — Remove a paper

Delete a paper and all its associated data (concepts, edges, embeddings):

```bash
drbrain delete p3f8a2            # prompts for confirmation
drbrain delete p3f8a2 --force    # skip confirmation
drbrain delete p3f8a2 --rm-files # also delete source files
drbrain delete p3f8a2 --json     # JSON output
```

### clean — Reset the library

Clear data directories. Keeps inbox PDFs intact:

```bash
drbrain clean                    # prompts for confirmation
drbrain clean --force            # skip confirmation
```

Clears: database, cache, logs, papers/, reports/. Preserves: spool/inbox/, config files.

### backup — Create or list backups

Create tar.gz snapshots of papers, database, and workspaces:

```bash
drbrain backup                                # create with timestamp name
drbrain backup --output my-backup.tar.gz      # custom path
drbrain backup --list                         # list existing backups
drbrain backup --list --json                  # machine-readable list
```

### queue — Confidence queue management

Entries with low confidence from extraction or closure are queued for review:

```bash
drbrain queue                    # list pending items
drbrain queue --json             # machine-readable
drbrain queue resolve            # interactive: accept/reject one item
drbrain queue resolve-all        # batch resolve all pending
```

### lineage — Author/research lineage

Explore author lineage via OpenAlex:

```bash
drbrain lineage --list                         # all actors with counts
drbrain lineage --name "Hinton"                # search by name
drbrain lineage A5023806754                    # specific author ID
drbrain lineage A5023806754 --json             # machine-readable
```

## Common workflows

**First-time setup:**
```bash
drbrain setup && drbrain check
```
After setup, add papers with `drbrain ingest` or `drbrain fetch` (see `paper-ingest` skill).

**Pre-cleanup health check:**
```bash
drbrain audit && drbrain stats
```

**Backup before major operation:**
```bash
drbrain backup && drbrain build --all
```

**Full reset:**
```bash
drbrain clean --force && drbrain setup --quick
```

## CLI Reference

| Command | What it does |
|---------|--------------|
| `drbrain setup` | Interactive first-time setup |
| `drbrain setup --quick` | Non-interactive setup |
| `drbrain check` | Environment diagnostics |
| `drbrain stats` | Library statistics |
| `drbrain stats --json` | Machine-readable stats |
| `drbrain stats -w <ws>` | Workspace-scoped stats |
| `drbrain report <id>` | Single-paper report |
| `drbrain report <id> --json` | Full report JSON |
| `drbrain delete <id>` | Delete paper (with confirmation) |
| `drbrain delete <id> --force --rm-files` | Force delete + source files |
| `drbrain clean` | Clear data (with confirmation) |
| `drbrain clean --force` | Force clear |
| `drbrain backup` | Create timestamped backup |
| `drbrain backup -o path.tar.gz` | Custom backup path |
| `drbrain backup --list` | List existing backups |
| `drbrain queue` | List pending confidence items |
| `drbrain queue resolve` | Resolve one item |
| `drbrain queue resolve-all` | Batch resolve all |
| `drbrain lineage --list` | All authors with paper counts |
| `drbrain lineage --name "X"` | Search author by name |
| `drbrain lineage <openalex_id>` | Specific author details |
