#!/usr/bin/env python
"""统一入库：把 --no-db 缓存的 tree.json + manifest 批量写入主库。

读取 oa ingest manifest 的成功记录（local_id 遵循 canonical-v1 身份契约），
校验 tree.json/raw.md 存在后写主库 papers/paper_ids 表。
经 Database 写接口入库；失败记录使整批中止，已存在 DOI 可安全重跑。

用法:
    uv run python scripts/pipeline/load_ingest_cache.py [--batch-size 500]
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

# Keep code imports anchored to this checkout.  The runtime root is a separate
# data namespace and is resolved afresh inside ``main``.
SOURCE_ROOT = Path(__file__).resolve().parents[2]
SOURCE_SRC = SOURCE_ROOT / "src"
for _import_root in (SOURCE_ROOT, SOURCE_SRC):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

# Compatibility snapshot for older callers; never use this value for data I/O.
from drbrain.dedup.resolver import PaperIDs, canonical_paper_id, normalize_doi  # noqa: E402
from drbrain.runtime import RuntimeContext, runtime_root  # noqa: E402
from drbrain.security import configured_secret_values, safe_error  # noqa: E402

# Compatibility snapshot only; never validate an environment selector during
# import because command-line callers need a clean runtime error instead.
ROOT = SOURCE_ROOT

from drbrain.storage.database import Database  # noqa: E402
from drbrain.storage.paths import paper_dir, paper_fs_key  # noqa: E402
from scripts.pipeline.common import load_cfg, runtime_path  # noqa: E402


def _safe_pipeline_error(value: object, cfg: object | None = None) -> str:
    """Bound and redact loader errors before printing them."""

    try:
        secrets = configured_secret_values(cfg if cfg is not None else os.environ)
    except Exception:  # noqa: BLE001
        secrets = ()
    if isinstance(value, BaseException):
        value = f"{type(value).__name__}: {value}"
    return safe_error(value, secrets=secrets)


def _safe_input_files(pattern: str, root: Path) -> list[Path]:
    """Expand an in-root manifest glob without following symlink aliases."""

    import glob

    lexical_pattern = Path(pattern).expanduser()
    if not lexical_pattern.is_absolute():
        lexical_pattern = root / lexical_pattern
    if not any(marker in pattern for marker in ("*", "?", "[")) and lexical_pattern.is_symlink():
        raise ValueError(f"manifest must not be a symlink: {lexical_pattern}")
    resolved_pattern = runtime_path(pattern, root)
    root = root.resolve()
    files: list[Path] = []
    for raw in sorted(glob.glob(str(resolved_pattern))):
        candidate = Path(raw)
        try:
            relative = candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"manifest escapes runtime root {root}: {candidate}") from exc
        current = root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise ValueError(f"manifest must not contain symlink components: {candidate}")
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"manifest escapes runtime root {root}: {candidate}") from exc
        if not resolved.is_file():
            raise ValueError(f"manifest is not a regular file: {candidate}")
        files.append(resolved)
    return files


def _safe_artifact_path(
    path: Path, *, papers_root: Path, context: RuntimeContext, label: str
) -> Path:
    """Validate one cached paper artifact before reading it.

    ``Path.is_file()`` follows symlinks.  Keep the lexical path check and the
    resolved containment check together so a stale artifact (or an
    intermediate directory alias) cannot make this loader read another
    worktree's ``tree.json``/``raw.md``.
    """
    lexical = Path(path).expanduser()
    if not lexical.is_absolute():
        lexical = papers_root / lexical
    # ``assert_within_root`` checks every lexical component before resolving;
    # it therefore rejects both a symlink leaf and an intermediate alias.
    resolved = context.assert_within_root(lexical, label=label)
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {lexical}")
    return lexical


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifests", type=str, default="data/shards/oa_shard*.ingest.jsonl")
    ap.add_argument("--db", type=str, default="data/drbrain.db")
    ap.add_argument("--papers-dir", type=str, default=None)
    args = ap.parse_args()

    cfg: dict | None = None
    try:
        root = runtime_root()
        # All input artifacts and the destination database share one runtime
        # namespace.  Do not root a second context at ``data/papers``: doing
        # so can inherit a temp root outside it and accidentally weaken the
        # boundary for callers using an isolated worktree.
        runtime = RuntimeContext.create(root)
    except (OSError, TypeError, ValueError) as exc:
        print(f"runtime root error: {_safe_pipeline_error(exc, cfg)}", file=sys.stderr)
        return 1
    try:
        if args.papers_dir:
            papers_dir = runtime.assert_within_root(args.papers_dir, label="papers directory")
        else:
            # Loading cached ingest output does not need an LLM config.  Use
            # it when present, while retaining the runtime-root default for a
            # freshly bootstrapped root that has no config layer yet.
            try:
                cfg = load_cfg(root=root)
            except FileNotFoundError:
                cfg = {}
            cfg_papers_dir = Path(cfg.get("dirs", {}).get("papers", root / "data/papers"))
            if not cfg_papers_dir.is_absolute():
                cfg_papers_dir = root / cfg_papers_dir
            papers_dir = runtime.assert_within_root(cfg_papers_dir, label="papers directory")
    except (OSError, TypeError, ValueError) as exc:
        print(f"runtime/config error: {_safe_pipeline_error(exc, cfg)}", file=sys.stderr)
        return 1

    # 收集全部 manifest 记录（DOI → local_id/title/year）
    records: dict[str, dict] = {}
    records_by_local_id: dict[str, str] = {}

    try:
        files = _safe_input_files(args.manifests, root)
    except (OSError, TypeError, ValueError) as exc:
        print(f"manifest path error: {_safe_pipeline_error(exc, cfg)}", file=sys.stderr)
        return 1
    if not files:
        print(f"无匹配 manifest 文件: {args.manifests}")
        return 1
    parse_errors: list[str] = []
    failed_records = 0
    for mf in files:
        with open(mf, encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError as exc:
                    parse_errors.append(f"{mf.name}:{line_no}: invalid JSON ({type(exc).__name__})")
                    continue
                if not isinstance(r, dict):
                    parse_errors.append(f"{mf.name}:{line_no}: record must be an object")
                    continue
                if not isinstance(r.get("ok"), bool):
                    parse_errors.append(f"{mf.name}:{line_no}: ok must be boolean")
                    continue
                if not r["ok"]:
                    failed_records += 1
                    continue
                local_id = r.get("local_id")
                if not isinstance(local_id, str) or not local_id.strip():
                    parse_errors.append(f"{mf.name}:{line_no}: successful record missing local_id")
                    continue
                try:
                    paper_fs_key(local_id)
                except (TypeError, ValueError) as exc:
                    parse_errors.append(
                        f"{mf.name}:{line_no}: invalid local_id ({type(exc).__name__})"
                    )
                    continue
                raw_doi = r.get("doi")
                # The source filename is not a reversible DOI encoding
                # (``_`` is a legal DOI character), so successful records must
                # carry the explicit normalized DOI.  Never guess an identity
                # from a lossy basename.
                if not isinstance(raw_doi, str) or not raw_doi.strip():
                    parse_errors.append(
                        f"{mf.name}:{line_no}: successful record requires an explicit DOI"
                    )
                    continue
                if r.get("title") is not None and not isinstance(r.get("title"), str):
                    parse_errors.append(f"{mf.name}:{line_no}: title must be a string")
                    continue
                year_value = r.get("year")
                if year_value is not None:
                    # The producer passes source years through verbatim, which
                    # commonly yields digit strings such as "2024".
                    if isinstance(year_value, str) and year_value.strip().isdigit():
                        r["year"] = int(year_value.strip())
                    elif not isinstance(year_value, int):
                        parse_errors.append(f"{mf.name}:{line_no}: year must be an integer")
                        continue
                try:
                    doi = normalize_doi(raw_doi)
                except (TypeError, ValueError) as exc:
                    parse_errors.append(f"{mf.name}:{line_no}: invalid DOI ({type(exc).__name__})")
                    continue
                if not doi:
                    parse_errors.append(f"{mf.name}:{line_no}: successful record has an empty DOI")
                    continue
                # ``file`` was added to the cache manifest after the first
                # loader release.  Keep old records usable, but validate the
                # field strictly whenever it is present; a supplied path is
                # never allowed to become an artifact lookup escape hatch.
                raw_file = r.get("file")
                if raw_file is not None and (
                    not isinstance(raw_file, str)
                    or not raw_file
                    or Path(raw_file).name != raw_file
                    or "\x00" in raw_file
                ):
                    parse_errors.append(f"{mf.name}:{line_no}: file must be a basename")
                    continue
                raw_scheme = r.get("id_scheme")
                if raw_scheme is not None and raw_scheme != "canonical-v1":
                    parse_errors.append(f"{mf.name}:{line_no}: unsupported id_scheme")
                    continue
                try:
                    ids = PaperIDs(
                        doi=doi,
                        arxiv=r.get("arxiv"),
                        s2_id=r.get("s2_id"),
                        openalex_id=r.get("openalex_id"),
                    ).normalized()
                    if raw_scheme == "canonical-v1":
                        expected = canonical_paper_id(
                            ids,
                            title=r.get("title") or "",
                            year=r.get("year"),
                            source_key=raw_file or f"doi:{doi}",
                        )
                        if local_id != expected:
                            parse_errors.append(
                                f"{mf.name}:{line_no}: local_id does not match canonical identity"
                            )
                            continue
                except (TypeError, ValueError) as exc:
                    parse_errors.append(
                        f"{mf.name}:{line_no}: invalid paper identity ({type(exc).__name__})"
                    )
                    continue
                prior = records.get(doi)
                if prior is not None and prior["local_id"] != local_id:
                    parse_errors.append(f"{mf.name}:{line_no}: DOI maps to conflicting local_ids")
                    continue
                prior_doi = records_by_local_id.get(local_id)
                if prior_doi is not None and prior_doi != doi:
                    parse_errors.append(f"{mf.name}:{line_no}: local_id maps to conflicting DOIs")
                    continue
                records[doi] = {
                    "local_id": local_id,
                    "doi": doi,
                    "arxiv": ids.arxiv,
                    "s2_id": ids.s2_id,
                    "openalex_id": ids.openalex_id,
                    "title": r.get("title") or "",
                    "year": r.get("year"),
                    "id_scheme": raw_scheme,
                }
                records_by_local_id[local_id] = doi
    if parse_errors:
        print(f"manifest 校验失败: {len(parse_errors)} 条")
        for error in parse_errors[:10]:
            print(f"  {error}")
        return 1
    if not records:
        print("没有可入库的成功记录")
        return 1
    print(f"manifest 待入库记录: {len(records)}")

    # Verify every artifact before opening the writable database.  A missing
    # tree must not leave a partially populated destination that looks usable.
    missing_files: list[str] = []
    for doi, rec in records.items():
        try:
            paper_path = paper_dir(papers_dir, rec["local_id"])
            _safe_artifact_path(
                paper_path / "tree.json",
                papers_root=papers_dir,
                context=runtime,
                label=f"tree.json for {rec['local_id']}",
            )
            _safe_artifact_path(
                paper_path / "raw.md",
                papers_root=papers_dir,
                context=runtime,
                label=f"raw.md for {rec['local_id']}",
            )
        except (OSError, ValueError) as exc:
            missing_files.append(f"{rec['local_id']}: {type(exc).__name__}: {exc}")
            continue
    if missing_files:
        print(f"纸稿缓存校验失败: {len(missing_files)} 条", file=sys.stderr)
        for error in missing_files[:10]:
            print(f"  {error}", file=sys.stderr)
        return 1
    if failed_records:
        print(f"manifest 含失败记录: {failed_records} 条", file=sys.stderr)
        return 1

    try:
        db_path = (
            Path(":memory:")
            if args.db == ":memory:"
            else runtime.assert_within_root(args.db, label="database path")
        )
        db = Database(db_path)
    except (sqlite3.Error, OSError, TypeError, ValueError) as exc:
        print(f"database path error: {_safe_pipeline_error(exc, cfg)}", file=sys.stderr)
        return 1
    t0 = time.monotonic()
    inserted = skipped_no_tree = skipped_exists = 0
    batch: list[tuple] = []

    def flush() -> None:
        nonlocal inserted
        if not batch:
            return
        try:
            for local_id, title, year, doi, arxiv, s2_id, openalex_id in batch:
                # Use the Database write surface so identifier validation and
                # conflict semantics stay consistent with normal ingest.
                db.insert_paper(local_id, title, year, "uploaded")
                db.insert_paper_ids(
                    local_id,
                    doi=doi,
                    arxiv=arxiv,
                    s2_id=s2_id,
                    openalex_id=openalex_id,
                    strict=True,
                )
            db.commit()
        except Exception:
            db.conn.rollback()
            raise
        inserted += len(batch)
        batch.clear()

    try:
        # Check existing unique mappings before the first write.  INSERT OR
        # REPLACE could otherwise delete a valid paper when a stale manifest
        # reuses its DOI or local_id.
        conflicts: list[str] = []
        for doi, rec in records.items():
            lid = rec["local_id"]
            doi_row = db.conn.execute(
                "SELECT local_id FROM paper_ids WHERE doi = ?", (doi,)
            ).fetchone()
            if doi_row and doi_row[0] != lid:
                conflicts.append(f"DOI {doi!r} already belongs to {doi_row[0]!r}")
            for kind, value in (
                ("arxiv", rec.get("arxiv")),
                ("s2_id", rec.get("s2_id")),
                ("openalex_id", rec.get("openalex_id")),
            ):
                if value:
                    owner = db.get_paper_by_external_id(kind, value)
                    if owner and owner != lid:
                        conflicts.append(f"{kind} {value!r} already belongs to {owner!r}")
            lid_row = db.conn.execute(
                "SELECT doi FROM paper_ids WHERE local_id = ?", (lid,)
            ).fetchone()
            if lid_row and (lid_row[0] or "").strip().lower() != doi:
                conflicts.append(f"local_id {lid!r} already belongs to {lid_row[0]!r}")
        if conflicts:
            print(f"paper identity conflict: {len(conflicts)}", file=sys.stderr)
            for conflict in conflicts[:10]:
                print(f"  {conflict}", file=sys.stderr)
            return 1

        for i, (doi, rec) in enumerate(records.items(), 1):
            lid = rec["local_id"]
            # 已存在则跳过（幂等）
            row = db.conn.execute("SELECT 1 FROM paper_ids WHERE doi = ?", (doi,)).fetchone()
            if row:
                skipped_exists += 1
                continue
            # title/year 从 raw.md 首行或 meta 拿不到（--no-db 模式没存），用 DOI 占位
            batch.append(
                (
                    lid,
                    rec.get("title") or doi,
                    rec.get("year"),
                    doi,
                    rec.get("arxiv"),
                    rec.get("s2_id"),
                    rec.get("openalex_id"),
                )
            )
            if len(batch) >= 500:
                flush()
            if i % 5000 == 0:
                print(
                    f"[{i}/{len(records)}] 入库={inserted} 已存在={skipped_exists} "
                    f"缺文件={skipped_no_tree} elapsed={time.monotonic() - t0:.0f}s",
                    flush=True,
                )
        flush()
        print(
            f"\n完成: 入库={inserted} 已存在跳过={skipped_exists} 缺文件={skipped_no_tree} "
            f"({time.monotonic() - t0:.0f}s)"
        )
        return 0
    except Exception as exc:
        try:
            db.conn.rollback()
        except Exception:
            pass
        print(f"database write error: {_safe_pipeline_error(exc, cfg)}", file=sys.stderr)
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
