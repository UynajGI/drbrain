"""One-command incremental preparation of the unified index (plan T45).

Stages fill only what is invalid, in one call:

1. ``fts``       — verify ``content_fts`` against the canonical blocks by
   matching sampled tokens; rebuild only on drift.  Verified content is never
   re-parsed — the canonical store is the input.
2. ``vectors``   — embed ready canonical nodes whose shared-store entry is
   missing or stale for the current embedding profile; unchanged nodes are
   reused, so a second prepare performs zero embedding work.
3. ``hierarchy`` — run the bounded builder only while the ready-leaf
   signature changed; a complete and fresh tree reports ``complete`` without
   resolving or calling a model.
4. ``publish``   — publish a new tree generation only when a stage changed
   the store; otherwise the active pointer stays untouched.

No KG build/closure dependency and no copy of the legacy retrieval database:
the main store plus the shared ANN directory are the only facts.  Every stage
is resumable — per-node vectors are idempotent, the builder persists nodes and
summaries per round, and a failed stage is reported without corrupting the
store for the next run.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from drbrain.tree.builder import BuilderConfig, TreeBuilder
from drbrain.tree.embedding_identity import EmbeddingProfile
from drbrain.tree.publish import publish_tree_generation
from drbrain.tree.summary import identify_contract
from drbrain.tree.vector_store import UnifiedVectorStore, VectorEntry, needs_write_meta

#: Working ANN collection under the storage root (published by ``publish_tree_generation``).
WORKING_VECTORS_DIR = "vectors"

#: Watermark key for the hierarchy stage signature.
HIERARCHY_WATERMARK = "tree.prepare.hierarchy"

#: Builder stop reasons that mean an execution failure (a clustered or a
#: summarised group could not even be attempted), not a valid stop.
FAILED_BUILD_REASONS = frozenset({"clustering_failed", "summary_failed"})

DEFAULT_BATCH_SIZE = 256


@dataclass
class PrepareOutcome:
    fts: dict[str, Any] = field(default_factory=dict)
    vectors: dict[str, Any] = field(default_factory=dict)
    hierarchy: dict[str, Any] = field(default_factory=dict)
    publication: dict[str, Any] = field(default_factory=dict)
    published: str | None = None
    changed: bool = False
    duration_ms: float = 0.0

    @property
    def failed_stages(self) -> list[str]:
        """Stages that are not in a usable state.

        ``partial`` counts: the frozen protocol keeps unfinished work and
        errors out of ``ready`` (T06), so a run that only got part way must not
        be presented as ok.
        """
        failed = [
            name
            for name, payload in (
                ("fts", self.fts),
                ("vectors", self.vectors),
                ("hierarchy", self.hierarchy),
                ("publish", self.publication),
            )
            if str(payload.get("status")) in {"failed", "partial"}
        ]
        return failed

    @property
    def ok(self) -> bool:
        return not self.failed_stages

    def to_json(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "changed": self.changed,
            "published": self.published,
            "failed_stages": self.failed_stages,
            "fts": self.fts,
            "vectors": self.vectors,
            "hierarchy": self.hierarchy,
            "publication": self.publication,
            "duration_ms": round(self.duration_ms, 3),
        }


class IndexModelSummary:
    """Adapt the T19 :class:`IndexModel` to the summary-service protocol."""

    def __init__(self, model: Any) -> None:
        self._model = model
        self.calls = 0

    def complete(self, prompt: str, *, max_tokens: int) -> Any:
        self.calls += 1
        return self._model.call_text(prompt, max_tokens=int(max_tokens))


def prepare_unified_index(
    db,
    *,
    storage_dir: str | Path,
    profile: EmbeddingProfile,
    embed: Callable[[Sequence[str]], list[list[float]]] | None = None,
    embed_cfg: Any = None,
    summary_service: Any = None,
    summary_model: Any = None,
    config: Any = None,
    builder_config: BuilderConfig | None = None,
    summary_max_tokens: int | None = None,
    summary_input_budget: int | None = None,
    seed_nodes: Sequence[str] | None = None,
    force: bool = False,
    publish: bool = True,
    sample: int = 10,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> PrepareOutcome:
    """Prepare FTS, shared vectors and the unified hierarchy, then publish."""
    if builder_config is None and (summary_max_tokens or summary_input_budget):
        import dataclasses as _dataclasses

        base = BuilderConfig()
        builder_config = _dataclasses.replace(
            base,
            contract=_dataclasses.replace(
                base.contract,
                max_output_tokens=max(
                    1, int(summary_max_tokens or base.contract.max_output_tokens)
                ),
                input_budget=max(1, int(summary_input_budget or base.contract.input_budget)),
            ),
        )
    outcome = PrepareOutcome()
    started = time.perf_counter()
    if embed is None and embed_cfg is not None:

        def embed(texts: Sequence[str]) -> list[list[float]]:
            from drbrain.services.embedding import _embed_batch

            return _embed_batch(list(texts), embed_cfg)

    root = Path(storage_dir)
    outcome.fts = _prepare_fts(db, force=force, sample=sample)
    try:
        with UnifiedVectorStore(
            root / WORKING_VECTORS_DIR, create=True, dimension=profile.dimension
        ) as store:
            outcome.vectors = _prepare_vectors(
                db, store, profile, embed=embed, force=force, batch_size=batch_size
            )
            if outcome.vectors.get("status") == "failed":
                outcome.hierarchy = {"status": "skipped", "reason": "vectors-failed"}
            else:
                outcome.hierarchy = _prepare_hierarchy(
                    db,
                    store,
                    profile,
                    embed=embed,
                    summary_service=summary_service,
                    summary_model=summary_model,
                    config=config,
                    builder_config=builder_config,
                    seed_nodes=seed_nodes,
                    force=force,
                )
    except Exception as exc:  # noqa: BLE001 - an unusable collection is a reported state
        logger.warning("[tree] vector store unavailable: {}", exc)
        outcome.vectors = {"status": "failed", "error": f"vector store unavailable: {exc}"}
        outcome.hierarchy = {"status": "skipped", "reason": "vectors-failed"}
    outcome.changed = bool(outcome.vectors.get("embedded")) or bool(
        outcome.hierarchy.get("created")
    )
    if not publish:
        outcome.publication = {"status": "skipped", "reason": "publish-disabled"}
    elif not outcome.ok:
        outcome.publication = {"status": "skipped", "reason": "stage-failed"}
    elif not outcome.changed:
        outcome.publication = {"status": "skipped", "reason": "unchanged"}
    else:
        try:
            result = publish_tree_generation(
                db,
                root,
                profile_id=profile.profile_id(),
                vector_dir=root / WORKING_VECTORS_DIR,
            )
        except Exception as exc:  # noqa: BLE001 - publication stays a reported state
            logger.warning("[tree] publication failed: {}", exc)
            outcome.publication = {"status": "failed", "error": str(exc)}
        else:
            outcome.published = str(result.get("generation") or "")
            outcome.publication = {
                "status": "published",
                "generation": outcome.published,
                "vector_count": result.get("vector_count"),
            }
    outcome.duration_ms = (time.monotonic() - started) * 1000
    return outcome


# ── stages ──────────────────────────────────────────────────────────────────


def _prepare_fts(db, *, force: bool, sample: int) -> dict[str, Any]:
    try:
        status = db.content_fts_status(sample=max(1, int(sample)))
        if force or not status.get("consistent", False):
            indexed = db.rebuild_content_fts()
            return {
                "status": "rebuilt",
                "indexed": indexed,
                "blocks": status.get("blocks", 0),
            }
        return {
            "status": "ok",
            "blocks": status.get("blocks", 0),
            "sampled": status.get("sampled", 0),
        }
    except Exception as exc:  # noqa: BLE001 - a stage failure is a reported state
        logger.warning("[tree] fts stage failed: {}", exc)
        return {"status": "failed", "error": str(exc)}


def ready_node_rows(db) -> list[dict[str, Any]]:
    """Ready leaves/regions with the identity a vector write is keyed on.

    Shared by the vector stage and the read-only index status/verify commands
    so "this node needs a vector" is decided by one enumeration.
    """
    rows = db.conn.execute(
        "SELECT node_id, kind, revision, content_hash, local_id, layer FROM tree_nodes "
        "WHERE state = 'ready' AND kind IN ('leaf', 'region') ORDER BY layer, node_id"
    ).fetchall()
    return [
        {
            "node_id": str(node_id),
            "kind": str(kind),
            "revision": max(1, int(revision or 1)),
            "content_hash": str(content_hash or ""),
            "local_id": str(local_id or ""),
            "layer": int(layer or 0),
        }
        for node_id, kind, revision, content_hash, local_id, layer in rows
    ]


def _node_text(db, node_id: str) -> str:
    from drbrain.storage.node_projection import read_node_text

    return read_node_text(db.conn, node_id)


def _prepare_vectors(
    db,
    store,
    profile: EmbeddingProfile,
    *,
    embed,
    force: bool,
    batch_size: int,
) -> dict[str, Any]:
    if embed is None:
        return {
            "status": "failed",
            "error": "no embedder configured (pass embed or embed_cfg)",
        }
    if profile.dimension is None:
        return {"status": "failed", "error": "embedding profile needs a dimension"}
    profile_id = profile.profile_id()
    rows = ready_node_rows(db)
    pending: list[dict[str, Any]] = []
    for row in rows:
        if not force and not needs_write_meta(
            db.get_node_vector(row["node_id"]),
            node_revision=row["revision"],
            content_hash=row["content_hash"],
            profile_id=profile_id,
        ):
            continue
        pending.append(row)
    if not pending:
        return {"status": "ok", "nodes": len(rows), "embedded": 0, "pending": 0}

    texts: list[str] = []
    records: list[dict[str, Any]] = []
    empty: list[str] = []
    for row in pending:
        text = _node_text(db, row["node_id"])
        if not text:
            empty.append(row["node_id"])
            continue
        texts.append(text)
        records.append(row)

    embedded = 0
    size = max(1, int(batch_size))
    try:
        for start in range(0, len(records), size):
            chunk = records[start : start + size]
            vectors = embed(texts[start : start + size])
            if len(vectors) != len(chunk):
                raise ValueError(f"embedder returned {len(vectors)} vectors for {len(chunk)} texts")
            entries = []
            for record, vector in zip(chunk, vectors):
                values = tuple(float(value) for value in vector)
                if len(values) != int(profile.dimension):
                    raise ValueError(
                        f"embedder returned dimension {len(values)}, "
                        f"profile requires {profile.dimension}"
                    )
                entries.append(
                    VectorEntry(
                        node_id=record["node_id"],
                        node_revision=record["revision"],
                        kind=record["kind"],
                        local_id=record["local_id"],
                        layer=record["layer"],
                        content_hash=record["content_hash"],
                        profile_id=profile_id,
                        vector=values,
                    )
                )
            for entry in entries:
                db.upsert_node_vector(
                    entry.node_id,
                    node_revision=entry.node_revision,
                    kind=entry.kind,
                    profile_id=entry.profile_id,
                    content_hash=entry.content_hash,
                    dimension=len(entry.vector),
                    local_id=entry.local_id,
                    layer=entry.layer,
                    state="staging",
                )
            store.upsert(entries)
            for entry in entries:
                db.upsert_node_vector(
                    entry.node_id,
                    node_revision=entry.node_revision,
                    kind=entry.kind,
                    profile_id=entry.profile_id,
                    content_hash=entry.content_hash,
                    dimension=len(entry.vector),
                    local_id=entry.local_id,
                    layer=entry.layer,
                    state="ready",
                )
            embedded += len(entries)
    except Exception as exc:  # noqa: BLE001 - staging rows are retried next run
        logger.warning("[tree] vector stage failed: {}", exc)
        return {
            "status": "failed",
            "embedded": embedded,
            "error": str(exc),
            "pending": len(records) - embedded,
        }
    return {
        "status": "ok",
        "nodes": len(rows),
        "embedded": embedded,
        "pending": 0,
        "empty": empty,
    }


def _param_digest(value: Any) -> str:
    """Stable digest of one parameter object (contract, dataclass, mapping)."""
    if value is None:
        return ""
    digest = getattr(value, "digest", None)
    if callable(digest):
        return str(digest() or "")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        payload: Any = dataclasses.asdict(value)
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        payload = str(value)
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _hierarchy_signature(db, config: BuilderConfig, profile: EmbeddingProfile | None = None) -> str:
    """Everything the published hierarchy depends on (T36 invalidation).

    Leaves are the content side; the contract — which now carries the resolved
    model and tokenizer identity — the embedding profile and the clustering and
    cost parameters are the deployment side.  A change on either side has to
    move the signature, otherwise summaries produced by an old deployment keep
    being certified as current.
    """
    leaves = db.conn.execute(
        "SELECT node_id, content_hash FROM tree_nodes "
        "WHERE state = 'ready' AND kind = 'leaf' ORDER BY node_id"
    ).fetchall()
    profile_identity = ""
    if profile is not None:
        getter = getattr(profile, "profile_id", None)
        profile_identity = str(getter() if callable(getter) else (getter or ""))
    payload = "|".join(
        [
            *[f"{node_id}@{content_hash}" for node_id, content_hash in leaves],
            _param_digest(getattr(config, "contract", None)),
            _param_digest(getattr(config, "clustering", None)),
            _param_digest(getattr(config, "cost", None)),
            str(int(getattr(config, "max_layers", 0))),
            str(getattr(config, "lam", "")),
            str(int(getattr(config, "min_frontier", 0))),
            str(int(getattr(config, "max_new_nodes_per_round", 0))),
            str(bool(getattr(config, "use_structure_hints", False))),
            str(int(getattr(config, "structure_hints_per_document", 0))),
            profile_identity,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _contract_json(contract: Any) -> str:
    """Canonical contract JSON, matching how ``tree_nodes`` stores it."""
    canonical = getattr(contract, "canonical", None)
    data = canonical() if callable(canonical) else dict(contract or {})
    return json.dumps(dict(data), ensure_ascii=False, sort_keys=True)


def _contract_identities(model: Any, summary_service: Any) -> tuple[str, str]:
    """Resolved ``(model, tokenizer)`` identities for the summary contract.

    The model identity comes from the bound endpoint role (endpoint name,
    base URL and model), so two deployments that share a model name never share
    cached summaries; the tokenizer identity comes from the counter that
    actually measures the budgets.
    """
    inner = getattr(model, "_model", None)
    role = getattr(inner, "role", None)
    model_identity = str(getattr(role, "identity", "") or "")
    if not model_identity:
        model_identity = str(getattr(model, "identity", "") or "")
    counter = getattr(summary_service, "_count_tokens", None)
    if counter is None:
        from drbrain.services.tokens import count_tokens

        # Production passes no summary service, so the builder ends up on the
        # shared default counter: bind that identity rather than leaving the
        # field empty, or a counter change slips past the cache key.
        counter = count_tokens
    module = str(getattr(counter, "__module__", "") or "")
    name = str(getattr(counter, "__name__", type(counter).__name__))
    tokenizer_identity = f"{module}.{name}".lstrip(".")
    return model_identity, tokenizer_identity


def _prepare_hierarchy(
    db,
    store,
    profile: EmbeddingProfile,
    *,
    embed,
    summary_service: Any,
    summary_model: Any,
    config: Any,
    builder_config: BuilderConfig | None,
    seed_nodes: Sequence[str] | None,
    force: bool,
) -> dict[str, Any]:
    builder_config = builder_config or BuilderConfig()
    model = summary_model
    if model is None:
        try:
            model = _resolve_index_summary_model(config)
        except Exception as exc:  # noqa: BLE001 - reported, retried next run
            return {"status": "failed", "error": str(exc), "reason": "index-model-unavailable"}
    # Bind the resolved deployment identity before anything derives from the
    # contract: the cache key and the region node identity both include it, so
    # the signature has to see the identities the summaries will actually carry.
    model_identity, tokenizer_identity = _contract_identities(model, summary_service)
    builder_config = dataclasses.replace(
        builder_config,
        contract=identify_contract(
            builder_config.contract,
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
        ),
    )
    signature = _hierarchy_signature(db, builder_config, profile)
    previous = db.get_vector_metadata(HIERARCHY_WATERMARK)
    # Retire every ready region built under a different contract before the
    # unchanged-signature return: a region node id embeds the contract digest,
    # so superseded nodes are stale by construction and must not keep certifying
    # the tree.  This must **not** depend on the watermark — an old library, a
    # killed build, a previous partial run or a rolled-back deployment can leave
    # the watermark equal to the current signature (or unset) while another
    # contract's regions are still ready.  Steady state costs one read.
    retired = db.retire_regions_with_other_contract(_contract_json(builder_config.contract))
    if retired:
        logger.info("[tree] retired {} stale region(s) for the current contract", len(retired))
        db.retire_node_vectors(retired)
        if store is not None:
            store.delete(retired)
    if not force and previous == signature and not db.leaves_missing_parent():
        return {
            "status": "complete",
            "reason": "signature-unchanged",
            "frontier": 0,
            "created": 0,
        }
    frontier = (
        [str(node) for node in seed_nodes] if seed_nodes is not None else db.leaves_missing_parent()
    )
    if not force and not frontier:
        db.set_vector_metadata(HIERARCHY_WATERMARK, signature)
        return {"status": "complete", "reason": "no-unassigned-leaves", "created": 0}
    if len(frontier) <= int(getattr(builder_config, "min_frontier", 1)):
        return {
            "status": "skipped",
            "reason": "frontier_below_minimum",
            "frontier": len(frontier),
            "created": 0,
        }

    builder = TreeBuilder(
        db,
        vectors=store,
        summary_service=summary_service,
        embed=embed,
        config=builder_config,
        model=model,
        profile_id=profile.profile_id(),
    )
    try:
        result = builder.build(frontier)
    except Exception as exc:  # noqa: BLE001 - reported, retried next run
        logger.warning("[tree] hierarchy stage failed: {}", exc)
        return {"status": "failed", "error": str(exc), "frontier": len(frontier)}
    payload = result.to_json()
    failures = dict(payload.get("failed") or {})
    stop_reason = str(payload.get("stop_reason") or "")
    if failures or stop_reason in FAILED_BUILD_REASONS:
        # An execution failure is its own state, not a valid stop (T06/T35):
        # nothing is certified, the leaves stay in the frontier because no
        # parent was published, and the watermark does not move — so the very
        # same command retries once the endpoint is healthy again.
        payload["status"] = "partial" if payload.get("created") else "failed"
        payload["reason"] = stop_reason or next(iter(failures), "stage-failed")
        return payload
    payload["status"] = "built" if payload.get("created") else "complete"
    db.set_vector_metadata(HIERARCHY_WATERMARK, signature)
    return payload


def _resolve_index_summary_model(config: Any) -> Any:
    from drbrain.services.index_model import IndexModel
    from drbrain.services.model_roles import ROLE_INDEX, resolve_model_role

    role = resolve_model_role(config, ROLE_INDEX)
    return IndexModelSummary(IndexModel(role=role))


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "FAILED_BUILD_REASONS",
    "HIERARCHY_WATERMARK",
    "IndexModelSummary",
    "PrepareOutcome",
    "WORKING_VECTORS_DIR",
    "prepare_unified_index",
    "ready_node_rows",
]
