"""Legacy vector migration: confirmed rows into the shared store (plan T52).

The legacy ``tree_vectors`` table keeps float32 blobs in SQLite, keyed by
``{paper_id}:{local_node_id}``, and records **no** producer identity — no
model, no dimension, no normalization flag.  A legacy vector is therefore
reusable only when the caller declares what produced it
(:class:`LegacyVectorIdentity`) and every declared field matches the target
:class:`EmbeddingProfile`; the rest of the basis is verified structurally:
the blob must decode to the profile dimension, the global node id must agree
with the row's ``paper_id``, and the legacy content hash must resolve to the
canonical node(s) whose text was embedded.

Confirmed vectors are written to the shared Zvec index plus the metadata
table — the same path the unified builder uses (``VectorEntry`` +
``node_vectors``).  Nothing is ever written back into ``tree_vectors`` or its
sqlite-vec shadow, and no vector is written into SQLite as a float copy: the
unified path does not continue the legacy dual-write.  Rows that fail any
check are reported as ``pending`` for the embedding pipeline, never dropped
silently and never "resolved" by deleting legacy data.
"""

from __future__ import annotations

import struct
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from drbrain.tree.embedding_identity import EmbeddingProfile
from drbrain.tree.vector_store import VectorEntry, needs_write

#: The legacy pipeline stored ``sha256(text)[:16]`` as its content hash.
LEGACY_CONTENT_HASH_LENGTH = 16

LEGACY_VECTOR_TABLE = "tree_vectors"


@dataclass(frozen=True)
class LegacyVectorIdentity:
    """Declared producer identity of the legacy ``tree_vectors`` rows.

    The legacy schema records none of this, so every field the target profile
    uses must be affirmatively declared (``from_profile`` covers the common
    case where the legacy rows came from the same setup); an undeclared or
    disagreeing field blocks reuse instead of being assumed compatible.
    """

    provider: str = ""
    model: str = ""
    normalize: bool | None = None
    max_seq_length: int | None = None
    preprocessing: str = ""
    tokenizer: str = ""
    tokenizer_revision: str = ""
    revision: str = ""

    @classmethod
    def from_profile(cls, profile: EmbeddingProfile) -> LegacyVectorIdentity:
        return cls(
            provider=profile.provider,
            model=profile.model,
            normalize=profile.normalize,
            max_seq_length=profile.max_seq_length,
            preprocessing=profile.preprocessing,
            tokenizer=profile.tokenizer,
            tokenizer_revision=profile.tokenizer_revision,
            revision=profile.revision,
        )

    def missing_required(self) -> tuple[str, ...]:
        missing: list[str] = []
        if not str(self.provider).strip():
            missing.append("provider")
        if not str(self.model).strip():
            missing.append("model")
        if self.normalize is None:
            missing.append("normalize")
        return tuple(missing)

    def difference(self, profile: EmbeddingProfile) -> str | None:
        """The first undeclared or disagreeing identity field, else ``None``."""
        if self.missing_required():
            return "identity-unknown"
        checks = (
            ("provider", "provider-mismatch"),
            ("model", "model-mismatch"),
            ("normalize", "normalization-mismatch"),
            ("max_seq_length", "sequence-budget-mismatch"),
            ("preprocessing", "preprocessing-mismatch"),
            ("tokenizer", "tokenizer-mismatch"),
            ("tokenizer_revision", "tokenizer-revision-mismatch"),
            ("revision", "revision-mismatch"),
        )
        for name, reason in checks:
            declared = getattr(self, name)
            target = getattr(profile, name)
            if name == "provider":
                declared = str(declared).strip().lower()
                target = str(target).strip().lower()
            if declared is None or declared == "":
                if target is not None and target != "":
                    return f"undeclared-{name}"
                continue
            if declared != target:
                return reason
        return None


@dataclass(frozen=True)
class LegacyVectorRow:
    node_id: str
    paper_id: str
    content_hash: str
    tree_layer: str
    blob: bytes


@dataclass(frozen=True)
class CanonicalTarget:
    node_id: str
    local_id: str
    kind: str
    node_revision: int
    layer: int
    content_hash: str


@dataclass(frozen=True)
class VectorVerdict:
    row: LegacyVectorRow
    decision: str  # "reuse" | "pending"
    reason: str = ""
    targets: tuple[CanonicalTarget, ...] = ()
    vector: tuple[float, ...] = ()


def decode_legacy_vector(blob: bytes | None) -> tuple[float, ...] | None:
    """Decode one legacy float32 blob (little-endian, as written by struct)."""
    if not blob or len(blob) % 4 != 0:
        return None
    return struct.unpack(f"<{len(blob) // 4}f", bytes(blob))


def legacy_node_paper(node_id: str) -> str | None:
    """The paper part of a legacy global node id ``{paper}:{local}``."""
    prefix, separator, local = str(node_id).partition(":")
    if not separator or not prefix or not local:
        return None
    return prefix


def iter_legacy_vector_rows(db) -> Iterator[LegacyVectorRow]:
    """Read the legacy vector table in a stable order (read-only)."""
    tables = {
        str(row[0])
        for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    }
    if LEGACY_VECTOR_TABLE not in tables:
        logger.info("[tree] no {} table; nothing to migrate", LEGACY_VECTOR_TABLE)
        return
    cursor = db.conn.execute(
        f"SELECT node_id, paper_id, content_hash, tree_layer, embedding "
        f"FROM {LEGACY_VECTOR_TABLE} ORDER BY node_id"
    )
    for node_id, paper_id, content_hash, tree_layer, blob in cursor:
        yield LegacyVectorRow(
            node_id=str(node_id),
            paper_id=str(paper_id),
            content_hash=str(content_hash or ""),
            tree_layer=str(tree_layer or ""),
            blob=bytes(blob or b""),
        )


def canonical_target_resolver(
    db,
) -> Callable[[str, str], tuple[CanonicalTarget, ...]]:
    """Resolve ``(paper_id, legacy_content_hash)`` to ready canonical nodes."""
    cursor = db.conn.execute(
        "SELECT node_id, local_id, kind, revision, layer, content_hash "
        "FROM tree_nodes WHERE state = 'ready'"
    )
    index: dict[tuple[str, str], list[CanonicalTarget]] = {}
    for node_id, local_id, kind, revision, layer, content_hash in cursor.fetchall():
        digest = str(content_hash or "")
        if not digest:
            continue
        target = CanonicalTarget(
            node_id=str(node_id),
            local_id=str(local_id),
            kind=str(kind),
            node_revision=max(1, int(revision)),
            layer=int(layer),
            content_hash=digest,
        )
        key = (str(local_id), digest[:LEGACY_CONTENT_HASH_LENGTH])
        index.setdefault(key, []).append(target)

    def resolve(paper_id: str, content_hash: str) -> tuple[CanonicalTarget, ...]:
        key = (str(paper_id), str(content_hash or "").strip().lower())
        return tuple(index.get(key, ()))

    return resolve


def classify_legacy_vector(
    row: LegacyVectorRow,
    *,
    profile: EmbeddingProfile,
    declared: LegacyVectorIdentity | None,
    resolve_target: Callable[[str, str], Sequence[CanonicalTarget]],
) -> VectorVerdict:
    """Decide whether one legacy row can be reused without re-embedding."""
    if declared is None:
        return VectorVerdict(row, "pending", reason="identity-unknown")
    difference = declared.difference(profile)
    if difference is not None:
        return VectorVerdict(row, "pending", reason=difference)
    if profile.dimension is None:
        return VectorVerdict(row, "pending", reason="profile-dimension-unknown")
    vector = decode_legacy_vector(row.blob)
    if vector is None:
        return VectorVerdict(row, "pending", reason="unusable-blob")
    if len(vector) != int(profile.dimension):
        return VectorVerdict(row, "pending", reason="dimension-mismatch")
    if legacy_node_paper(row.node_id) != row.paper_id:
        return VectorVerdict(row, "pending", reason="id-mismatch")
    targets = tuple(resolve_target(row.paper_id, row.content_hash))
    if not targets:
        return VectorVerdict(row, "pending", reason="no-canonical-match")
    return VectorVerdict(row, "reuse", targets=targets, vector=vector)


@dataclass
class VectorMigrationReport:
    rows: int = 0
    reused: list[dict[str, Any]] = field(default_factory=list)
    already: list[dict[str, Any]] = field(default_factory=list)
    pending: list[dict[str, Any]] = field(default_factory=list)
    failed: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "counts": {
                "reused": len(self.reused),
                "already": len(self.already),
                "pending": len(self.pending),
                "failed": len(self.failed),
            },
            "reused": self.reused,
            "already": self.already,
            "pending": self.pending,
            "failed": self.failed,
        }


def migrate_legacy_vectors(
    db,
    store,
    *,
    profile: EmbeddingProfile,
    declared_identity: LegacyVectorIdentity | None,
    resolve_target: Callable[[str, str], Sequence[CanonicalTarget]] | None = None,
    rows: Iterable[LegacyVectorRow] | None = None,
    batch_size: int = 256,
) -> VectorMigrationReport:
    """Write confirmed legacy vectors into the shared index, resumably.

    Metadata is marked ``staging`` before the index write and ``ready`` after
    it, so an interrupted run re-processes exactly the unfinished entries.
    ``already`` counts targets whose metadata is already ready for this
    profile/revision/hash; the report's ``pending`` list is the work list for
    the embedding pipeline.
    """
    resolver = resolve_target or canonical_target_resolver(db)
    report = VectorMigrationReport()
    pending_batch: list[tuple[LegacyVectorRow, CanonicalTarget, tuple[float, ...]]] = []
    profile_id = profile.profile_id()

    def flush() -> None:
        current = list(pending_batch)
        pending_batch.clear()
        if not current:
            return
        entries = [
            VectorEntry(
                node_id=target.node_id,
                node_revision=target.node_revision,
                kind=target.kind,
                local_id=target.local_id,
                layer=target.layer,
                content_hash=target.content_hash,
                profile_id=profile_id,
                vector=vector,
            )
            for _, target, vector in current
        ]
        try:
            store.upsert(entries)
        except Exception as exc:  # noqa: BLE001 - one batch failure stays contained
            for row, target, _ in current:
                report.failed.append(
                    {"legacy": row.node_id, "target": target.node_id, "error": str(exc)}
                )
            return
        for row, target, vector in current:
            try:
                db.upsert_node_vector(
                    target.node_id,
                    node_revision=target.node_revision,
                    kind=target.kind,
                    profile_id=profile_id,
                    content_hash=target.content_hash,
                    dimension=len(vector),
                    local_id=target.local_id,
                    layer=target.layer,
                    state="ready",
                )
            except Exception as exc:  # noqa: BLE001 - vector stays reusable next run
                report.failed.append(
                    {"legacy": row.node_id, "target": target.node_id, "error": str(exc)}
                )
            else:
                report.reused.append(
                    {"legacy": row.node_id, "target": target.node_id, "local_id": target.local_id}
                )

    source = rows if rows is not None else iter_legacy_vector_rows(db)
    for row in source:
        report.rows += 1
        verdict = classify_legacy_vector(
            row, profile=profile, declared=declared_identity, resolve_target=resolver
        )
        if verdict.decision != "reuse":
            report.pending.append({"legacy": row.node_id, "reason": verdict.reason})
            continue
        for target in verdict.targets:
            entry = VectorEntry(
                node_id=target.node_id,
                node_revision=target.node_revision,
                kind=target.kind,
                local_id=target.local_id,
                layer=target.layer,
                content_hash=target.content_hash,
                profile_id=profile_id,
                vector=verdict.vector,
            )
            if not needs_write(db.get_node_vector(target.node_id), entry):
                report.already.append({"legacy": row.node_id, "target": target.node_id})
                continue
            try:
                db.upsert_node_vector(
                    target.node_id,
                    node_revision=target.node_revision,
                    kind=target.kind,
                    profile_id=profile_id,
                    content_hash=target.content_hash,
                    dimension=len(verdict.vector),
                    local_id=target.local_id,
                    layer=target.layer,
                    state="staging",
                )
            except Exception as exc:  # noqa: BLE001 - keep the row isolated
                report.failed.append(
                    {"legacy": row.node_id, "target": target.node_id, "error": str(exc)}
                )
                continue
            pending_batch.append((row, target, verdict.vector))
            if len(pending_batch) >= max(1, int(batch_size)):
                flush()
    flush()
    return report
