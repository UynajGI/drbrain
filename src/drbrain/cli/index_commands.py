"""``drbrain index`` — prepare, inspect and verify the searchable index.

Main line: ``ingest → index build → search / ask``.  ``index build`` fills the
enabled retrieval legs from the canonical store — the incremental lexical BM25
index, canonical FTS, shared leaf/region vectors and the unified tree hierarchy
— and publishes one queryable tree generation.  ``index status`` separates the
three states a corpus can be in (ingested / indexed / retrievable) and reports
per-leg readiness, versions, backlog and failure reasons.  ``index verify``
re-checks exactly what ``search`` and ``ask`` actually read; the wider storage
audit stays in ``drbrain storage audit``.

Bare ``drbrain index`` keeps the historical incremental BM25 contract: the
group callback runs it when no subcommand is given, so existing callers, exit
codes and ``--json`` shapes stay valid.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import typer

from drbrain.cli._common import open_db, runtime_data_path
from drbrain.security import redact_sensitive, safe_error

DEFAULT_TREE_STORAGE = "data/tree"

index_app = typer.Typer(
    help="Prepare, inspect and verify the searchable index (lexical + FTS + vectors + tree)",
    invoke_without_command=True,
    no_args_is_help=False,
)


def _runtime_option(value: Any, default: Any) -> Any:
    """Normalize a Typer ``OptionInfo`` for direct (non-CLI) callers."""
    if isinstance(value, typer.models.OptionInfo):
        return default if value.default is None else value.default
    return value


def _embed_section(cfg: Any) -> Any:
    """The ``embed`` config section as a typed object or dict."""
    if isinstance(cfg, dict):
        return cfg.get("embed")
    return getattr(cfg, "embed", None)


def _embedding_profile(cfg: Any) -> tuple[Any | None, str]:
    """The current embedding profile plus a reason when it cannot be built."""
    from drbrain.tree.embedding_identity import profile_from_config

    try:
        return profile_from_config(_embed_section(cfg)), ""
    except ValueError as exc:
        return None, safe_error(exc)


def _tree_storage_root(ctx: typer.Context, cfg: Any, override: str = "") -> Path:
    """Resolve the unified tree storage root the same way ``index build`` does."""
    from drbrain.rag.config import get_llamaindex_config

    li = get_llamaindex_config(cfg)
    configured = (
        str(override or "").strip()
        or str(getattr(li, "tree_storage", "") or "").strip()
        or DEFAULT_TREE_STORAGE
    )
    return Path(runtime_data_path(ctx, configured, label="tree storage"))


def _vector_backlog(db: Any, profile: Any) -> dict[str, Any]:
    """Ready leaves/regions whose stored vector is missing or stale.

    The predicate is the same :func:`~drbrain.tree.vector_store.needs_write_meta`
    the vector stage uses, so ``pending == 0`` means the next ``index build``
    would embed nothing.
    """
    from drbrain.tree.prepare import ready_node_rows
    from drbrain.tree.vector_store import needs_write_meta

    rows = ready_node_rows(db)
    stored: dict[str, dict[str, Any]] = {}
    for node_id, state, revision, content_hash, profile_id in db.conn.execute(
        "SELECT node_id, state, node_revision, content_hash, profile_id FROM node_vectors"
    ):
        stored[str(node_id)] = {
            "state": state,
            "node_revision": revision,
            "content_hash": content_hash,
            "profile_id": profile_id,
        }
    profile_id = profile.profile_id()
    pending = [
        row["node_id"]
        for row in rows
        if needs_write_meta(
            stored.get(row["node_id"]),
            node_revision=row["revision"],
            content_hash=row["content_hash"],
            profile_id=profile_id,
        )
    ]
    return {
        "profile_id": profile_id,
        "nodes": len(rows),
        "pending": len(pending),
        "sample": pending[:5],
    }


# ── the historical incremental BM25 stage (bare ``drbrain index``) ───────────


@contextmanager
def _open_database(cfg: Any, db_path: str = ""):
    """Open the configured database, or an explicit shard override.

    The shard pipelines keep their per-shard databases; ``index build --db``
    targets one exactly like the historical ``embed --tree --db`` did.
    """
    if str(db_path or "").strip():
        from drbrain.storage.database import Database

        db = Database(db_path)
        try:
            yield db
        finally:
            db.close()
        return
    with open_db(cfg) as db:
        yield db


def rebuild_lexical_index(
    cfg: Any,
    *,
    rebuild: bool = False,
    notify: Callable[[str], None] | None = None,
    db_path: str = "",
) -> dict[str, Any]:
    """Rebuild the lexical BM25 index (the pre-``index build`` behavior).

    Incremental by default: skips the rebuild when no paper changed since the
    last successful run.  Returns the historical JSON shape — ``{"documents",
    "indexed"}`` (plus ``"up_to_date": True`` on the skip path) — so both the
    legacy command and ``index build`` report the same lexical stage result.
    ``db_path`` overrides the configured database (shard pipelines).
    """
    with _open_database(cfg, db_path) as db:
        from drbrain.query.bm25 import build_bm25_index

        if not rebuild:
            last_run = db.get_last_run("index")
            max_ts = db.get_max_paper_timestamp()
            if last_run is not None and (max_ts is None or max_ts <= last_run):
                count = db.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
                return {"documents": count, "indexed": False, "up_to_date": True}

        if notify is not None:
            notify("Building BM25 index...")
        index = build_bm25_index(db)
        doc_count = len(index._documents)
        db.set_last_run("index")
        db.commit()
    return {"documents": doc_count, "indexed": True}


def emit_lexical_result(payload: dict[str, Any], *, json_output: bool) -> None:
    """Print the lexical stage result in the historical (pre-refactor) format."""
    if json_output:
        typer.echo(json.dumps(payload))
    elif payload.get("up_to_date"):
        typer.echo(
            f"Index up to date ({payload['documents']} documents, no changes since last run)"
        )
    else:
        typer.echo(f"Indexed {payload['documents']} documents")


@index_app.callback()
def index_callback(
    ctx: typer.Context,
    rebuild: bool = typer.Option(False, "--rebuild", help="Force full rebuild"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON"),
) -> None:
    """Prepare, inspect and verify the searchable index — the main-line index entry.

    ``drbrain index build`` prepares every enabled retrieval leg (lexical BM25
    + canonical FTS + shared vectors + unified tree, and the LlamaIndex
    generation when ``rag_engine: llamaindex``) and publishes what search/ask
    read; ``drbrain index status`` reports ingested / indexed / retrievable
    per leg, and ``drbrain index verify`` re-checks exactly that.

    A bare invocation keeps the historical incremental BM25 rebuild: it skips
    the rebuild when no paper changed since the last successful run; use
    --rebuild to force a full rebuild.
    """
    if ctx.invoked_subcommand is not None:
        # Options declared on the group belong to the bare command only.
        return
    cfg = ctx.obj["config"]
    payload = rebuild_lexical_index(
        cfg,
        rebuild=bool(_runtime_option(rebuild, False)),
        notify=None if json_output else typer.echo,
    )
    emit_lexical_result(payload, json_output=bool(_runtime_option(json_output, False)))


# ── index build ─────────────────────────────────────────────────────────────


def _prepare_llamaindex_generation(
    cfg: Any,
    db: Any,
    *,
    force: bool,
    notify: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Prepare the LlamaIndex generation when that engine is selected.

    Mirrors ``rag index`` semantics so the remedy command really prepares what
    ``search``/``ask`` read: only ``rag_engine: llamaindex`` prepares this
    generation (the deprecated SQL snapshot path is never republished), and a
    missing llama-index stack or a build error is a reported *stage failure*
    rather than a silent skip.  The unified store stages are engine
    independent and always run.
    """
    from drbrain.rag.config import get_llamaindex_config

    li = get_llamaindex_config(cfg)
    engine = str(getattr(li, "rag_engine", "") or "").strip().lower()
    if engine != "llamaindex":
        return {"status": "skipped", "reason": f"rag_engine={engine or 'sql'}"}
    try:
        from drbrain.rag.indexer import _LLAMA_INDEX_AVAILABLE, build_index
    except ImportError as exc:  # pragma: no cover - defensive
        return {"status": "failed", "error": safe_error(exc)}
    if not _LLAMA_INDEX_AVAILABLE:
        return {
            "status": "failed",
            "error": (
                "llama-index is not installed; run: "
                "uv add llama-index-core llama-index-retrievers-bm25"
            ),
        }
    if notify is not None:
        notify("Preparing the LlamaIndex generation...")
    try:
        stats = build_index(cfg, db, force=bool(force), max_node_tokens=li.max_node_tokens)
    except Exception as exc:  # noqa: BLE001 - a failed stage is a reported state
        return {"status": "failed", "error": safe_error(exc)}
    return {"status": "ok", **dict(stats)}


@index_app.command("build")
def index_build_cmd(
    ctx: typer.Context,
    force: bool = typer.Option(False, "--force", "-f", help="Force a full rebuild of every stage"),
    tree_storage: str = typer.Option(
        "",
        "--tree-storage",
        help="Storage root for unified tree generations (default: data/tree)",
    ),
    db_path: str = typer.Option(
        "", "--db", help="Override db path (shard databases; default cfg db.path)"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
) -> None:
    """Prepare every enabled retrieval leg and publish a queryable generation.

    The lexical BM25 index, canonical FTS, the shared vector collection and the
    unified tree hierarchy are filled incrementally from the canonical store;
    when ``llamaindex.rag_engine: llamaindex`` the LlamaIndex generation is
    prepared in the same run (the deprecated SQL snapshot path is not
    republished).  A tree generation is published only when something changed.
    Exit code 1 when any stage failed — an unfinished or failed stage is never
    reported as ready.
    """
    cfg = ctx.obj["config"]
    _force = bool(_runtime_option(force, False))
    _json = bool(_runtime_option(json_output, False))
    tree_storage = str(_runtime_option(tree_storage, "") or "")
    db_path = str(_runtime_option(db_path, "") or "")

    lexical = rebuild_lexical_index(
        cfg,
        rebuild=_force,
        notify=None if _json else typer.echo,
        db_path=db_path,
    )

    from drbrain.rag.config import get_llamaindex_config
    from drbrain.tree.prepare import prepare_unified_index

    li = get_llamaindex_config(cfg)
    embed_cfg = _embed_section(cfg)
    profile, profile_error = _embedding_profile(cfg)
    if profile is None:
        typer.echo(
            f"[index] embedding profile unavailable: {profile_error}",
            err=True,
        )
        raise typer.Exit(1)

    root = _tree_storage_root(ctx, cfg, tree_storage)
    started = time.monotonic()
    with _open_database(cfg, db_path) as db:
        outcome = prepare_unified_index(
            db,
            storage_dir=root,
            profile=profile,
            embed_cfg=embed_cfg,
            config=cfg,
            summary_max_tokens=li.summary_max_tokens,
            summary_input_budget=li.summary_input_budget,
            force=_force,
        )
        llamaindex_stage = _prepare_llamaindex_generation(
            cfg,
            db,
            force=_force,
            notify=None if _json else typer.echo,
        )
    payload = outcome.to_json()
    payload["lexical"] = lexical
    payload["llamaindex"] = llamaindex_stage
    if llamaindex_stage.get("status") in {"failed", "partial"}:
        payload["failed_stages"] = [*payload["failed_stages"], "llamaindex"]
        payload["ok"] = False
    payload["duration_ms"] = round((time.monotonic() - started) * 1000, 3)
    if _json:
        typer.echo(json.dumps(redact_sensitive(payload), indent=2, ensure_ascii=False, default=str))
    else:
        typer.echo(
            f"Index build ({'full' if _force else 'incremental'}): "
            f"{'ok' if payload['ok'] else 'FAILED'}"
        )
        typer.echo(f"  lexical:    {lexical['documents']} documents, indexed={lexical['indexed']}")
        for stage in ("fts", "vectors", "hierarchy", "publication"):
            typer.echo(f"  {stage + ':':<12} {payload[stage].get('status', '')}")
        if llamaindex_stage.get("status") != "skipped":
            typer.echo(f"  {'llamaindex:':<12} {llamaindex_stage.get('status', '')}")
        typer.echo(
            f"  changed={payload['changed']} published={payload['published'] or 'none'} "
            f"duration_ms={payload['duration_ms']}"
        )
        if payload["failed_stages"]:
            typer.echo(f"Failed stages: {', '.join(payload['failed_stages'])}", err=True)
    if not payload["ok"]:
        raise typer.Exit(code=1)


# ── index status ────────────────────────────────────────────────────────────


def _tree_leg_status(ctx: typer.Context, cfg: Any, db: Any, profile: Any) -> dict[str, Any]:
    """Readiness of the unified tree leg: published generation and hierarchy.

    Unpromoted leaves (``leaves_missing_parent``) are legal multi-roots in the
    frozen protocol, so they are reported as *pending* assignment work, not as
    a readiness failure: the next ``index build`` retries them.
    """
    from drbrain.tree.publish import get_active_tree_generation, resolve_tree_generation

    root = _tree_storage_root(ctx, cfg)
    generation = get_active_tree_generation(root) or ""
    leg: dict[str, Any] = {
        "ready": False,
        "generation": generation,
        "tree_storage": str(root),
        "manifest_profile_id": "",
        "profile_id": profile.profile_id() if profile is not None else "",
        "leaves": db.count_tree_nodes(kind="leaf", state="ready"),
        "regions": db.count_tree_nodes(kind="region", state="ready"),
        "leaves_missing_parent": len(db.leaves_missing_parent()),
        "reasons": [],
    }
    if not generation:
        leg["reasons"].append("no_published_generation")
        return leg
    try:
        manifest = resolve_tree_generation(root, generation)["manifest"]
    except Exception as exc:  # noqa: BLE001 - an unreadable generation is a reported state
        leg["reasons"].append(f"generation_unreadable: {safe_error(exc)}")
        return leg
    leg["manifest_profile_id"] = str(manifest.get("profile_id") or "")
    if leg["leaves"] + leg["regions"] <= 0:
        leg["reasons"].append("no_ready_nodes")
    if profile is not None and leg["manifest_profile_id"] != profile.profile_id():
        leg["reasons"].append("embedding_profile_changed")
    leg["ready"] = not leg["reasons"]
    return leg


def build_index_status(ctx: typer.Context, cfg: Any) -> dict[str, Any]:
    """The three-state readiness report (ingested / indexed / retrievable)."""
    from drbrain.rag.config import get_llamaindex_config
    from drbrain.rag.legs import normalize_legs
    from drbrain.tree.observability import tree_snapshot

    li = get_llamaindex_config(cfg)
    profile, profile_error = _embedding_profile(cfg)
    requested = [str(item) for item in (li.retrievers or [])]
    try:
        normalized = normalize_legs(requested)
        route = {
            "requested": requested,
            "legs": list(normalized.legs),
            "extras": list(normalized.extras),
            "notes": list(normalized.notes),
        }
    except Exception as exc:  # noqa: BLE001 - an invalid route is a reported state
        route = {"requested": requested, "legs": [], "extras": [], "notes": [safe_error(exc)]}

    with open_db(cfg) as db:
        snapshot = tree_snapshot(db, storage_dir=_tree_storage_root(ctx, cfg))
        fts = db.content_fts_status()
        vectors_ready = db.count_node_vectors(state="ready")
        vectors_staging = db.count_node_vectors(state="staging")
        last_run = db.get_last_run("index")
        max_ts = db.get_max_paper_timestamp()
        papers = int(db.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0])
        if profile is not None:
            backlog = _vector_backlog(db, profile)
        else:
            backlog = {"profile_id": "", "nodes": 0, "pending": None, "sample": []}
        tree_leg = _tree_leg_status(ctx, cfg, db, profile)

    up_to_date = last_run is not None and (max_ts is None or max_ts <= last_run)
    lexical = {
        "ready": papers == 0 or up_to_date,
        "documents": papers,
        "indexed": up_to_date,
        "last_run": last_run,
        "max_paper_timestamp": max_ts,
        "reasons": [] if (papers == 0 or up_to_date) else ["lexical_index_stale"],
    }
    fts_leg = {
        "ready": bool(fts.get("consistent")),
        "indexed": int(fts.get("indexed") or 0),
        "blocks": int(fts.get("blocks") or 0),
        "sampled": int(fts.get("sampled") or 0),
        "reasons": [] if fts.get("consistent") else ["content_fts_inconsistent"],
    }
    vector_leg: dict[str, Any] = {
        "ready": False,
        "ready_count": vectors_ready,
        "staging_count": vectors_staging,
        "nodes": backlog["nodes"],
        "pending": backlog["pending"],
        "sample": backlog["sample"],
        "profile_id": backlog["profile_id"],
        "reasons": [],
    }
    if profile is None:
        vector_leg["reasons"].append(f"embedding_profile_unavailable: {profile_error}")
    else:
        if not vectors_ready:
            vector_leg["reasons"].append("no_ready_vectors")
        if backlog["pending"]:
            vector_leg["reasons"].append(f"vectors_pending:{backlog['pending']}")
        vector_leg["ready"] = not vector_leg["reasons"]

    ingested = {
        "ready": bool(snapshot.documents.get("ready")),
        "documents": dict(snapshot.documents),
        "blocks": snapshot.blocks,
        "papers": papers,
        "reasons": [],
    }
    if snapshot.documents.get("failed"):
        ingested["reasons"].append(f"documents_failed:{snapshot.documents['failed']}")
    if snapshot.documents.get("stale"):
        ingested["reasons"].append(f"documents_stale:{snapshot.documents['stale']}")

    legs = {
        "lexical": lexical,
        "fts": fts_leg,
        "vector": vector_leg,
        "tree": tree_leg,
    }
    backend = _backend_health(cfg)
    indexed_ready = fts_leg["ready"] and vector_leg["ready"] and tree_leg["ready"]
    # Retrievability follows the *route* the engine will actually run: the
    # LlamaIndex engine reads its persisted generation for bm25/vector and the
    # unified generation for the tree leg; the SQL engine reads a pinned SQL
    # snapshot when one is published and otherwise serves a tree request from
    # the unified generation alone.
    engine = str(getattr(li, "rag_engine", "") or "llamaindex").strip().lower()
    route_legs = set(route["legs"])
    retrievable_reasons: list[str] = []
    if not li.enabled:
        retrievable_reasons.append("llamaindex_disabled")
    if "tree" in route_legs and not tree_leg["ready"]:
        retrievable_reasons.append("tree_unavailable")
    if profile is None:
        retrievable_reasons.append("embedding_profile_unavailable")
    if engine == "llamaindex":
        if not backend.get("ready"):
            retrievable_reasons.append("llamaindex_generation_not_ready")
    elif backend.get("generation"):
        if not backend.get("ready"):
            retrievable_reasons.append("sql_snapshot_unavailable")
    elif "tree" not in route_legs:
        # No pinned SQL snapshot and no tree leg to fall back to.
        retrievable_reasons.append("no_published_index")
    retrievable = {
        "ready": not retrievable_reasons,
        "reasons": retrievable_reasons,
        "backend": backend.get("status", ""),
    }

    pending = {
        "documents_stale": int(snapshot.documents.get("stale") or 0),
        "documents_failed": int(snapshot.documents.get("failed") or 0),
        "vectors_pending": backlog["pending"],
        "vectors_staging": vectors_staging,
        "leaves_missing_parent": tree_leg["leaves_missing_parent"],
        "summaries_failed": int(snapshot.summaries.get("failed") or 0),
    }
    reasons = [
        f"state.{name}: {reason}"
        for name, leg in {**legs, "retrievable": retrievable}.items()
        for reason in leg["reasons"]
    ]
    ok = bool(ingested["ready"] and indexed_ready and retrievable["ready"])
    if not li.enabled:
        status = "disabled"
    elif ok:
        status = "ready"
    elif ingested["ready"] or any(leg["ready"] for leg in legs.values()):
        status = "partial"
    else:
        status = "not_ready"
    return {
        "ok": ok,
        "status": status,
        "engine": str(getattr(li, "rag_engine", "llamaindex") or "llamaindex"),
        "enabled": bool(li.enabled),
        "tree_storage": str(_tree_storage_root(ctx, cfg)),
        "route": route,
        "generation": tree_leg["generation"],
        "states": {
            "ingested": ingested,
            "indexed": {"ready": indexed_ready, "legs": ["fts", "vector", "tree"]},
            "retrievable": retrievable,
        },
        "legs": legs,
        "backend": backend,
        "pending": pending,
        "reasons": reasons,
        "tree_state": snapshot.stage_state().value,
    }


def _backend_health(cfg: Any) -> dict[str, Any]:
    """The configured RAG backend's own readiness report (read-only)."""
    from drbrain.rag.indexer import get_index_health

    try:
        report = dict(get_index_health(cfg))
    except Exception as exc:  # noqa: BLE001 - a broken backend is a reported state
        return {
            "ready": False,
            "status": "unavailable",
            "generation": None,
            "reasons": [safe_error(exc)],
        }
    return {
        "ready": bool(report.get("ready")),
        "status": str(report.get("status") or ""),
        "generation": report.get("generation"),
        "reasons": [str(reason) for reason in (report.get("reasons") or [])],
        "storage_dir": str(report.get("storage_dir") or ""),
        "vector_backend": report.get("vector_backend"),
    }


def _emit_status(report: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        typer.echo(json.dumps(redact_sensitive(report), indent=2, ensure_ascii=False, default=str))
        return
    typer.echo(f"Index status: {report['status']} (engine: {report['engine']})")
    states = report["states"]
    typer.echo(
        f"  Ingested:    {'yes' if states['ingested']['ready'] else 'no'}"
        f" — {states['ingested']['papers']} papers,"
        f" {states['ingested']['blocks']} blocks, documents {states['ingested']['documents']}"
    )
    typer.echo(f"  Indexed:     {'yes' if states['indexed']['ready'] else 'no'}")
    typer.echo(
        f"  Retrievable: {'yes' if states['retrievable']['ready'] else 'no'}"
        f" (generation: {report['generation'] or 'none'})"
    )
    for name, leg in report["legs"].items():
        mark = "ready" if leg["ready"] else "not ready"
        detail = ", ".join(
            f"{key}={value}"
            for key, value in leg.items()
            if key
            in {
                "documents",
                "indexed",
                "ready_count",
                "nodes",
                "pending",
                "generation",
                "leaves",
                "regions",
                "leaves_missing_parent",
            }
            and value is not None
        )
        typer.echo(f"  [{name}] {mark} ({detail})")
        for reason in leg["reasons"]:
            typer.echo(f"      reason: {reason}")
    if report["reasons"]:
        typer.echo("Reasons:")
        for reason in report["reasons"]:
            typer.echo(f"  - {reason}")


@index_app.command("status")
def index_status_cmd(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
) -> None:
    """Report ingested / indexed / retrievable state per retrieval leg.

    Read-only: it never embeds, builds or writes.  ``ready`` for a leg means the
    next ``search``/``ask`` can read it as-is; ``pending`` counts the work
    ``index build`` would still do.  Exit code 0 — the report itself is the
    result.
    """
    cfg = ctx.obj["config"]
    report = build_index_status(ctx, cfg)
    _emit_status(report, json_output=bool(_runtime_option(json_output, False)))


# ── index verify ────────────────────────────────────────────────────────────


def build_index_verify(ctx: typer.Context, cfg: Any) -> dict[str, Any]:
    """Consistency checks over exactly what ``search``/``ask`` read.

    Reuses the storage audit and the publication verifier, then adds the
    retrieval-readiness checks they do not cover: the engine's own generation
    (LlamaIndex store or active SQL snapshot), ready nodes missing their
    current vector, leaves that lost their parent, whether the published
    generation still describes the live database, and whether its manifest was
    produced by the embedding profile that is configured now.
    """
    from drbrain.rag.config import get_llamaindex_config
    from drbrain.services.storage_audit import audit_storage
    from drbrain.tree.publish import (
        compute_watermarks,
        get_active_tree_generation,
        resolve_tree_generation,
        verify_tree_generation,
    )

    profile, profile_error = _embedding_profile(cfg)
    root = _tree_storage_root(ctx, cfg)
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []

    def add(name: str, ok: bool, reason: str = "", *, severity: str = "error", **detail: Any):
        record = {
            "name": name,
            "ok": bool(ok),
            "severity": severity,
            "reason": reason,
            "detail": detail,
        }
        checks.append(record)
        if not ok and reason:
            (errors if severity == "error" else warnings).append(f"{name}: {reason}")

    generation = get_active_tree_generation(root) or ""
    manifest: dict[str, Any] = {}
    if not generation:
        add("tree_generation", False, "no published generation under the tree storage")
    else:
        try:
            manifest = resolve_tree_generation(root, generation)["manifest"]
        except Exception as exc:  # noqa: BLE001 - report instead of raising
            add("tree_generation", False, f"unreadable generation: {safe_error(exc)}")

    with open_db(cfg) as db:
        db_path = db.path
        if generation and manifest:
            try:
                check = verify_tree_generation(root, generation)
            except Exception as exc:  # noqa: BLE001 - a broken generation is a finding
                add("tree_generation", False, f"verification failed: {safe_error(exc)}")
            else:
                add(
                    "tree_generation",
                    bool(check.get("ok")),
                    f"manifest mismatch: {', '.join(check.get('mismatches') or [])}",
                    generation=generation,
                    fingerprint=check.get("fingerprint", ""),
                )
            manifest_profile = str(manifest.get("profile_id") or "")
            if profile is None:
                add(
                    "embedding_profile",
                    False,
                    f"embedding profile unavailable: {profile_error}",
                )
            else:
                add(
                    "embedding_profile",
                    manifest_profile == profile.profile_id(),
                    f"published profile {manifest_profile or '(none)'} differs from the "
                    f"configured {profile.profile_id()}",
                    manifest_profile_id=manifest_profile,
                    profile_id=profile.profile_id(),
                )
                live = compute_watermarks(db, profile_id=manifest_profile or None)
                expected = manifest.get("watermarks") or {}
                stale_sections = [
                    section
                    for section in ("content", "nodes", "vectors")
                    if live.get(section, {}).get("digest")
                    != (expected.get(section) or {}).get("digest")
                ]
                add(
                    "generation_freshness",
                    not stale_sections,
                    f"live database moved past the published generation: "
                    f"{', '.join(stale_sections)}",
                    stale_sections=stale_sections,
                    live=live,
                )

        fts = db.content_fts_status()
        add(
            "content_fts",
            bool(fts.get("consistent")),
            f"FTS found {fts.get('indexed')} of {fts.get('sampled')} sampled blocks",
            indexed=fts.get("indexed"),
            blocks=fts.get("blocks"),
            sampled=fts.get("sampled"),
        )
        if profile is not None:
            backlog = _vector_backlog(db, profile)
            add(
                "node_vectors",
                not backlog["pending"],
                f"{backlog['pending']} ready nodes lack a current vector",
                nodes=backlog["nodes"],
                pending=backlog["pending"],
                sample=backlog["sample"],
                profile_id=backlog["profile_id"],
            )
        else:
            add("node_vectors", False, f"embedding profile unavailable: {profile_error}")
        missing_parent = db.leaves_missing_parent()
        add(
            "leaf_reachability",
            not missing_parent,
            f"{len(missing_parent)} ready leaves have no ready parent "
            "(unpromoted leaves are legal multi-roots; the next index build retries them)",
            severity="warning",
            sample=missing_parent[:5],
        )

    # The engine's own generation: the LlamaIndex store always serves the
    # bm25/vector legs in that mode, a pinned SQL snapshot only when published.
    li = get_llamaindex_config(cfg)
    engine = str(getattr(li, "rag_engine", "") or "llamaindex").strip().lower()
    backend = _backend_health(cfg)
    if engine == "llamaindex" or backend.get("generation"):
        name = "llamaindex_generation" if engine == "llamaindex" else "sql_snapshot"
        add(
            name,
            bool(backend.get("ready")),
            f"{name} is not ready: {', '.join(backend.get('reasons') or [])}",
            status=backend.get("status"),
            generation=backend.get("generation"),
        )
    else:
        add(
            "engine_generation",
            True,
            "",
            status=backend.get("status"),
            note="no pinned engine generation; retrieval is served by the unified tree leg",
        )

    audit = audit_storage(db_path=db_path, storage_dir=root)
    audit_errors = [
        finding.to_json()
        for finding in audit.findings
        if finding.severity == "error" and finding.category not in {"generation_inconsistent"}
    ]
    add(
        "storage_audit",
        not audit_errors,
        f"{len(audit_errors)} storage audit error(s)",
        audit=audit.to_json(),
    )
    for finding in audit.findings:
        if finding.severity == "warning" and finding.category in {
            "no_active_generation",
            "generation_unreadable",
        }:
            warnings.append(f"storage_audit: {finding.message}")
    return {
        "ok": not errors,
        "generation": generation,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
    }


@index_app.command("verify")
def index_verify_cmd(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Output JSON to stdout"),
) -> None:
    """Verify what ``search``/``ask`` read: FTS, vectors, membership, generation.

    Read-only.  Exit 1 when any check reports an error, 0 otherwise.  The broad
    storage-integrity audit remains ``drbrain storage audit``; this command only
    covers retrieval readiness and never duplicates that audit.
    """
    cfg = ctx.obj["config"]
    report = build_index_verify(ctx, cfg)
    if bool(_runtime_option(json_output, False)):
        typer.echo(json.dumps(redact_sensitive(report), indent=2, ensure_ascii=False, default=str))
    else:
        typer.echo(
            f"Index verify: {'ok' if report['ok'] else 'failed'}"
            f" (generation: {report['generation'] or 'none'})"
        )
        for check in report["checks"]:
            if check["ok"]:
                mark = "ok"
            else:
                mark = "warn" if check.get("severity") == "warning" else "FAIL"
            suffix = f" — {check['reason']}" if check["reason"] else ""
            typer.echo(f"  [{mark}] {check['name']}{suffix}")
        for warning in report["warnings"]:
            typer.echo(f"  [warn] {warning}", err=True)
    if not report["ok"]:
        raise typer.Exit(code=1)


__all__ = ["build_index_status", "build_index_verify", "index_app", "rebuild_lexical_index"]
