"""Read-only index reporting shared by the CLI and the WebUI (04-arch A2).

``index status`` / ``index verify`` used to be CLI-only builders that reached
into ``typer.Context`` for path policy.  The WebUI needs the very same numbers
(the three states, per-leg readiness, versions, backlog, verification), so the
builders live here and both callers adapt:

* the CLI keeps its runtime policy and passes ``tree_storage`` explicitly;
* ``app/service.py`` passes both ``db`` (its own runtime/symlink policy) and
  ``tree_storage``, so neither caller re-implements the other's path rules.

Nothing here writes: these are the read side of the index.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from drbrain.security import safe_error
from drbrain.storage.database import Database

#: Historical default tree storage root (kept identical to the CLI's).
DEFAULT_TREE_STORAGE = "data/tree"


def _resolve_tree_storage(cfg: Any, tree_storage: str | Path | None) -> Path:
    """Tree storage root when the caller did not resolve one itself.

    CLI callers resolve with their runtime policy; the app passes the service
    layer's.  This fallback keeps the historical default chain
    (``llamaindex.tree_storage`` → ``data/tree``) and honors an active runtime
    root so direct embedded callers cannot silently write outside it.
    """
    if tree_storage:
        return Path(tree_storage).expanduser()
    from drbrain.rag.config import get_llamaindex_config

    configured = str(getattr(get_llamaindex_config(cfg), "tree_storage", "") or "").strip()
    raw = Path(configured or DEFAULT_TREE_STORAGE).expanduser()
    if raw.is_absolute():
        return raw
    if "DRBRAIN_ROOT" in os.environ or "DRBRAIN_RUNTIME_ROOT" in os.environ:
        from drbrain.runtime import RuntimeContext

        return Path(RuntimeContext.create().assert_within_root(raw, label="tree storage"))
    return raw.resolve()


@contextmanager
def _open_db(cfg: Any, db: Any = None) -> Iterator[Any]:
    """Use the caller's connection, or open the configured one for CLI callers."""
    if db is not None:
        yield db
        return
    handle = Database(str(cfg["db"]["path"]))
    try:
        yield handle
    finally:
        handle.close()


def embed_section(cfg: Any) -> Any:
    """The ``embed`` config section as a typed object or dict."""
    if isinstance(cfg, dict):
        return cfg.get("embed")
    return getattr(cfg, "embed", None)


def embedding_profile(cfg: Any) -> tuple[Any | None, str]:
    """The current embedding profile plus a reason when it cannot be built."""
    from drbrain.tree.embedding_identity import profile_from_config

    try:
        return profile_from_config(embed_section(cfg)), ""
    except ValueError as exc:
        return None, safe_error(exc)


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


def _last_build_record(db: Any) -> dict[str, Any]:
    """The latest recorded prepare outcome (``{}`` when never recorded).

    Written by ``prepare_unified_index`` (``LAST_BUILD_KEY``, overwritten on
    every run), so ``index status``/``index verify`` can tell a *failed* build
    apart from retired-vector debris without guessing.  An absent record (a
    corpus last built before the key existed) is "no signal", not a failure.
    """
    from drbrain.tree.prepare import LAST_BUILD_KEY

    raw = db.get_vector_metadata(LAST_BUILD_KEY)
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _tree_leg_status(
    cfg: Any, db: Any, profile: Any, *, tree_storage: str | Path | None = None
) -> dict[str, Any]:
    """Readiness of the unified tree leg: published generation and hierarchy.

    Unpromoted leaves (``leaves_missing_parent``) are legal multi-roots in the
    frozen protocol, so they are reported as *pending* assignment work, not as
    a readiness failure: the next ``index build`` retries them.
    """
    from drbrain.tree.publish import get_active_tree_generation, resolve_tree_generation

    root = _resolve_tree_storage(cfg, tree_storage)
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
    last_build = _last_build_record(db)
    if last_build:
        failed_stages = [str(stage) for stage in (last_build.get("failed_stages") or [])]
        leg["last_build"] = {
            "ok": bool(last_build.get("ok")),
            "failed_stages": failed_stages,
            "published": str(last_build.get("published") or ""),
        }
        if not last_build.get("ok"):
            # A failed stage must never be reported as ready, even while an
            # older generation is still readable; the next successful build
            # overwrites the record and clears this reason.
            failed = ", ".join(failed_stages) or "unknown"
            leg["reasons"].append(f"last_build_failed: {failed}")
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


def build_index_status(
    cfg: Any, *, db: Any = None, tree_storage: str | Path | None = None
) -> dict[str, Any]:
    """The three-state readiness report (ingested / indexed / retrievable)."""
    from drbrain.rag.config import get_llamaindex_config
    from drbrain.rag.legs import normalize_legs
    from drbrain.tree.observability import tree_snapshot

    li = get_llamaindex_config(cfg)
    profile, profile_error = embedding_profile(cfg)
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

    with _open_db(cfg, db) as db:
        snapshot = tree_snapshot(db, storage_dir=_resolve_tree_storage(cfg, tree_storage))
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
        tree_leg = _tree_leg_status(cfg, db, profile, tree_storage=tree_storage)

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

    ingested: dict[str, Any] = {
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

    legs: dict[str, dict[str, Any]] = {
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
    elif not tree_leg.get("generation"):
        # BM25/vector also read the unified generation when no SQL corpus exists.
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
        "tree_storage": str(_resolve_tree_storage(cfg, tree_storage)),
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


def build_index_verify(
    cfg: Any, *, db: Any = None, tree_storage: str | Path | None = None
) -> dict[str, Any]:
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

    profile, profile_error = embedding_profile(cfg)
    root = _resolve_tree_storage(cfg, tree_storage)
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

    with _open_db(cfg, db) as db:
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

        last_build = _last_build_record(db)
        if last_build:
            failed = ", ".join(str(stage) for stage in (last_build.get("failed_stages") or []))
            add(
                "last_build",
                bool(last_build.get("ok")),
                f"the most recent index build failed: {failed or 'unknown stage'}",
                failed_stages=list(last_build.get("failed_stages") or []),
                published=str(last_build.get("published") or ""),
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


__all__ = [
    "DEFAULT_TREE_STORAGE",
    "build_index_status",
    "build_index_verify",
    "embed_section",
    "embedding_profile",
]
