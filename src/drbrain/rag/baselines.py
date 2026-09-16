"""Evaluation-only baseline adapters (plan T56).

Each baseline is a standalone comparison algorithm registered *only* for the
evaluation entry: BM25+vector fusion, the real PageIndex iterative tree search
(model-guided over ``tree.json``; never SQL LIKE), the unified-tree retrieval
ablation (flat all-layer ANN over the unified builder's own generation) and a
simple concatenation arm.  Baselines share the unified model/context caps,
record the algorithm they actually ran plus cost counters, and never become
production routes.

The unified-tree arm is **not** an independent RAPTOR baseline: it reads the
unified build products (structural candidates, conditioned groups and parent
cost gates) and says so in its own name, notes and report fields.  The real
RAPTOR comparison is ``raptor_collapsed``, which may only score a generation
produced and fingerprinted by an independent RAPTOR build; a missing or
mismatched provenance record fails closed with
:class:`BaselineProvenanceError` instead of silently scoring the unified tree
(review finding 8).

Dependencies are injectable so tests exercise the algorithms without a
database, a vector store or a model.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

#: The unified-tree arm replaced the mislabelled RAPTOR arm (review finding 8).
UNIFIED_TREE_ABLATION = "unified_tree_flat"
#: The independent RAPTOR comparison; provenance-gated.
RAPTOR_BASELINE = "raptor_collapsed"

#: Registered evaluation baselines → the real algorithm they run.
BASELINES: dict[str, str] = {
    "bm25_vector": "BM25 + vector (RRF fusion of both legs)",
    "pageindex": "PageIndex iterative tree search (model-guided over tree.json; no SQL LIKE)",
    UNIFIED_TREE_ABLATION: (
        "unified tree retrieval ablation (flat all-layer ANN over one unified-tree generation: "
        "structural candidates, conditioned groups, parent cost gates; no expansion, no navigation)"
    ),
    RAPTOR_BASELINE: (
        "RAPTOR collapsed tree over an independent RAPTOR build "
        "(provenance fingerprint verified; refuses to score a unified-tree generation)"
    ),
    "concat": "simple concatenation (BM25 top-k then vector top-k, deduplicated)",
}

#: What the unified-tree arm actually reads; reported with every outcome so no
#: reader can mistake it for an independent RAPTOR comparison (finding 8).
UNIFIED_TREE_ABLATION_NOTE = (
    "reads the unified-tree generation (structural candidates, conditioned groups, parent cost "
    "gates); this is a retrieval ablation of the unified builder, not an independent RAPTOR baseline"
)

#: Independent-RAPTOR provenance (plan T56/T62).  ``raptor_collapsed`` may only
#: score a generation whose own build published this record next to its
#: manifest.
RAPTOR_PROVENANCE_NAME = "raptor-provenance.json"
RAPTOR_ALGORITHM = "raptor"
#: Root of the independent RAPTOR build; never the unified-tree storage.
DEFAULT_RAPTOR_STORAGE = "data/raptor"
#: A record must carry all of these, non-empty, or the arm fails closed.
RAPTOR_PROVENANCE_FIELDS: tuple[str, ...] = (
    "algorithm",
    "algorithm_version",
    "build_params",
    "corpus",
    "generated_at",
    "generation",
    "manifest_fingerprint",
    "content_hash",
    "members_hash",
)

_SQL_SEARCH = Callable[..., list[dict[str, Any]]]
_TREE_SEARCH = Callable[[str, int], list[dict[str, Any]]]
_PAGEINDEX_SEARCH = Callable[[str, Sequence[str], int], list[dict[str, Any]]]
_RAPTOR_SEARCH = Callable[[str, int], list[dict[str, Any]]]


class BaselineProvenanceError(RuntimeError):
    """A baseline input is not the independent build product it claims to be."""


def baseline_algorithm(name: str) -> str:
    """The documented algorithm of a registered baseline."""
    key = str(name).strip().lower()
    if key not in BASELINES:
        raise ValueError(f"unknown baseline {name!r}; expected one of {sorted(BASELINES)}")
    return BASELINES[key]


def raptor_provenance_fingerprint(record: Mapping[str, Any] | None) -> str:
    """Canonical fingerprint over a declared RAPTOR provenance record.

    Recorded in the evaluation report so a run names the exact build it scored.
    """
    payload = {
        field_name: (record or {}).get(field_name) for field_name in RAPTOR_PROVENANCE_FIELDS
    }
    encoded = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def verify_raptor_provenance(
    provenance: Mapping[str, Any] | None,
    *,
    generation: str,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail-closed check that a generation is an independent RAPTOR build.

    Raises :class:`BaselineProvenanceError` when the record is absent,
    incomplete, declares another algorithm or generation, or does not bind to
    the published manifest fingerprint.  Callers get an auditable summary only
    for a verified record; nothing here ever falls back to another build.
    """
    if not isinstance(provenance, Mapping) or not provenance:
        raise BaselineProvenanceError(
            f"no RAPTOR provenance record for generation {generation!r}; expected "
            f"{RAPTOR_PROVENANCE_NAME} from an independent RAPTOR build"
        )
    missing = [
        field_name
        for field_name in RAPTOR_PROVENANCE_FIELDS
        if provenance.get(field_name) in (None, "", {}, [])
    ]
    if missing:
        raise BaselineProvenanceError(
            "RAPTOR provenance is missing required fields: " + ", ".join(missing)
        )
    algorithm = str(provenance.get("algorithm") or "").strip().lower()
    if algorithm != RAPTOR_ALGORITHM:
        raise BaselineProvenanceError(
            f"generation {generation!r} was built by {algorithm!r}, not by {RAPTOR_ALGORITHM!r}; "
            "it cannot be scored as the RAPTOR baseline"
        )
    declared = str(provenance.get("generation") or "")
    if declared != generation:
        raise BaselineProvenanceError(
            f"provenance names generation {declared!r} but {generation!r} was requested"
        )
    manifest_fingerprint = str(provenance.get("manifest_fingerprint"))
    manifest_checked = manifest is not None
    if manifest is not None:
        published = str(manifest.get("fingerprint") or "")
        if not published:
            raise BaselineProvenanceError(
                f"generation {generation!r} has no manifest fingerprint to bind its provenance to"
            )
        if manifest_fingerprint != published:
            raise BaselineProvenanceError(
                "provenance does not match the published manifest fingerprint of generation "
                f"{generation!r}"
            )
        manifest_fingerprint = published
    return {
        "generation": generation,
        "algorithm": algorithm,
        "algorithm_version": str(provenance.get("algorithm_version")),
        "generated_at": str(provenance.get("generated_at")),
        "corpus": provenance.get("corpus"),
        "build_params": provenance.get("build_params"),
        "content_hash": str(provenance.get("content_hash")),
        "members_hash": str(provenance.get("members_hash")),
        "manifest_fingerprint": manifest_fingerprint,
        "manifest_checked": manifest_checked,
        "provenance_fingerprint": raptor_provenance_fingerprint(provenance),
    }


def read_raptor_provenance(generation_path: str | Path) -> dict[str, Any]:
    """Read a generation's own provenance record; a missing file is an error."""
    path = Path(generation_path) / RAPTOR_PROVENANCE_NAME
    if not path.is_file():
        raise BaselineProvenanceError(
            f"generation at {generation_path} has no {RAPTOR_PROVENANCE_NAME}; only an "
            "independent RAPTOR build may be scored as the RAPTOR baseline"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BaselineProvenanceError(f"unreadable {RAPTOR_PROVENANCE_NAME} at {path}") from exc
    if not isinstance(payload, dict):
        raise BaselineProvenanceError(f"{RAPTOR_PROVENANCE_NAME} at {path} must be a JSON object")
    return payload


@dataclass(frozen=True)
class RaptorBuild:
    """A published independent RAPTOR build: root, generation, manifest, provenance."""

    storage_root: str
    generation: str
    manifest: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def verify(self) -> dict[str, Any]:
        """Fail-closed provenance audit of this build (raises on any mismatch)."""
        return verify_raptor_provenance(
            self.provenance, generation=self.generation, manifest=self.manifest
        )


def load_raptor_build(cfg: Any = None, *, storage_root: str | Path | None = None) -> RaptorBuild:
    """Load the published independent RAPTOR generation and its provenance.

    Unlike the unified-tree arm this never looks at the unified-tree storage:
    the RAPTOR root defaults to ``data/raptor`` under the selected runtime
    (``llamaindex.raptor_storage`` overrides it when configured).
    """
    from drbrain.tree.publish import get_active_tree_generation, resolve_tree_generation

    if storage_root is not None:
        root = Path(storage_root).expanduser()
    else:
        if cfg is None:
            raise BaselineProvenanceError(
                "no configuration to locate the independent RAPTOR build; the RAPTOR baseline "
                "needs its own published generation (plan T62)"
            )
        from drbrain.rag.config import get_llamaindex_config

        li = get_llamaindex_config(cfg)
        root = Path(
            _runtime_scoped_path(
                getattr(li, "raptor_storage", None) or DEFAULT_RAPTOR_STORAGE,
                label="raptor storage",
            )
        ).expanduser()
    generation = get_active_tree_generation(root)
    if not generation:
        raise BaselineProvenanceError(
            f"no independent RAPTOR generation under {root}; the RAPTOR baseline needs its own "
            "build (plan T62)"
        )
    resolved = resolve_tree_generation(root, generation)
    return RaptorBuild(
        storage_root=str(root),
        generation=generation,
        manifest=dict(resolved.get("manifest") or {}),
        provenance=read_raptor_provenance(resolved["path"]),
    )


@dataclass
class BaselineOutcome:
    name: str
    algorithm: str
    status: str = "ok"  # "ok" | "empty" | "unavailable"
    keys: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "baseline": self.name,
            "algorithm": self.algorithm,
            "status": self.status,
            "keys": list(self.keys),
            "details": self.details,
            "notes": list(self.notes),
        }


def _key_of(row: dict[str, Any]) -> str:
    paper = str(row.get("paper_id") or "")
    node = str(row.get("node_id") or "")
    if not node and row.get("evidence_id"):
        return str(row["evidence_id"])
    return f"{paper}:{node}" if paper and node else str(row.get("key") or "")


def _dedup(keys: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for key in keys:
        text = str(key)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _tree_keys(candidates: Sequence[Mapping[str, Any]], top_k: int) -> tuple[str, ...]:
    """Evidence keys of flat all-layer tree candidates, deduplicated and capped."""
    return tuple(
        _dedup(
            [
                f"{item.get('local_id', '')}:{item.get('node_id', '')}"
                if item.get("local_id")
                else str(item.get("node_id") or "")
                for item in candidates
            ]
        )[:top_k]
    )


@dataclass
class BaselineRunner:
    """Runs one registered baseline with injectable algorithm dependencies."""

    cfg: Any = None
    db: Any = None
    sql_search: _SQL_SEARCH | None = None
    tree_search: _TREE_SEARCH | None = None
    pageindex_search: _PAGEINDEX_SEARCH | None = None
    raptor_search: _RAPTOR_SEARCH | None = None
    #: The independent RAPTOR build to score; when absent it is loaded from the
    #: RAPTOR root and a missing/invalid build fails closed.
    raptor_build: RaptorBuild | None = None

    # ── algorithms ──────────────────────────────────────────────────────

    def _run_sql(self, legs: Sequence[str], query: str, top_k: int) -> tuple[list[dict], int]:
        search = self.sql_search or _default_sql_search(self.cfg, self.db)
        started = time.perf_counter()
        rows = list(search(list(legs), query, top_k) or [])
        return rows, int((time.perf_counter() - started) * 1000)

    def _run_sql_soft(
        self, legs: Sequence[str], query: str, top_k: int
    ) -> tuple[list[dict], str | None, int]:
        """One leg of a concatenation arm; a failure degrades, not aborts."""
        try:
            rows, elapsed = self._run_sql(legs, query, top_k)
            return rows, None, elapsed
        except Exception as exc:  # noqa: BLE001 - the arm reports the failure
            logger.info("[rag] concat leg {} unavailable: {}", "+".join(legs), exc)
            return [], f"{type(exc).__name__}: {exc}", 0

    def run(
        self,
        name: str,
        query: str,
        *,
        top_k: int = 10,
        papers: Sequence[str] | None = None,
    ) -> BaselineOutcome:
        key = str(name).strip().lower()
        algorithm = baseline_algorithm(key)
        try:
            if key == "bm25_vector":
                return self._bm25_vector(query, top_k, algorithm)
            if key == "concat":
                return self._concat(query, top_k, algorithm)
            if key == UNIFIED_TREE_ABLATION:
                return self._unified_tree_flat(query, top_k, algorithm)
            if key == RAPTOR_BASELINE:
                return self._raptor_collapsed(query, top_k, algorithm)
            return self._pageindex(query, top_k, papers, algorithm)
        except BaselineProvenanceError as exc:
            # fail closed: never report keys from a build the baseline did not verify
            logger.warning("[rag] baseline {} refused: {}", key, exc)
            return BaselineOutcome(
                name=key,
                algorithm=algorithm,
                status="unavailable",
                details={"error": f"{type(exc).__name__}: {exc}", "fail_closed": True},
                notes=("fail-closed: the input is not the independent build this baseline claims",),
            )
        except Exception as exc:  # noqa: BLE001 - a baseline failure is a reported state
            logger.warning("[rag] baseline {} unavailable: {}", key, exc)
            return BaselineOutcome(
                name=key,
                algorithm=algorithm,
                status="unavailable",
                details={"error": f"{type(exc).__name__}: {exc}"},
            )

    def _bm25_vector(self, query: str, top_k: int, algorithm: str) -> BaselineOutcome:
        rows, elapsed = self._run_sql(["bm25", "vector"], query, top_k)
        keys = _dedup([_key_of(row) for row in rows])[:top_k]
        return BaselineOutcome(
            name="bm25_vector",
            algorithm=algorithm,
            status="ok" if keys else "empty",
            keys=tuple(keys),
            details={"top_k": top_k, "candidates": len(rows), "elapsed_ms": elapsed},
        )

    def _concat(self, query: str, top_k: int, algorithm: str) -> BaselineOutcome:
        bm25_rows, bm25_error, ms_bm25 = self._run_sql_soft(["bm25"], query, top_k)
        vector_rows, vector_error, ms_vector = self._run_sql_soft(["vector"], query, top_k)
        keys = _dedup([_key_of(row) for row in bm25_rows] + [_key_of(row) for row in vector_rows])
        notes = tuple(
            note
            for note in (
                f"bm25 unavailable: {bm25_error}" if bm25_error else "",
                f"vector unavailable: {vector_error}" if vector_error else "",
            )
            if note
        )
        if keys:
            status = "ok"
        elif bm25_error and vector_error:
            status = "unavailable"
        else:
            status = "empty"
        return BaselineOutcome(
            name="concat",
            algorithm=algorithm,
            status=status,
            keys=tuple(keys[: top_k * 2]),
            details={
                "top_k": top_k,
                "bm25": len(bm25_rows),
                "vector": len(vector_rows),
                "elapsed_ms": ms_bm25 + ms_vector,
            },
            notes=notes,
        )

    def _unified_tree_flat(self, query: str, top_k: int, algorithm: str) -> BaselineOutcome:
        """Flat all-layer ANN over the unified builder's own generation."""
        search = self.tree_search or _default_unified_tree_search(self.cfg, self.db)
        started = time.perf_counter()
        candidates = list(search(query, top_k) or [])
        elapsed = int((time.perf_counter() - started) * 1000)
        keys = _tree_keys(candidates, top_k)
        return BaselineOutcome(
            name=UNIFIED_TREE_ABLATION,
            algorithm=algorithm,
            status="ok" if keys else "empty",
            keys=keys,
            details={
                "top_k": top_k,
                "candidates": len(candidates),
                "view": "all",
                "elapsed_ms": elapsed,
                "source": "unified-tree generation",
            },
            notes=(UNIFIED_TREE_ABLATION_NOTE,),
        )

    def _raptor_collapsed(self, query: str, top_k: int, algorithm: str) -> BaselineOutcome:
        """Flat all-layer ANN over an *independent, verified* RAPTOR build."""
        build = self.raptor_build or load_raptor_build(self.cfg)
        audit = build.verify()  # fail-closed: raises BaselineProvenanceError
        search = self.raptor_search or _default_raptor_search(build, self.cfg)
        started = time.perf_counter()
        candidates = list(search(query, top_k) or [])
        elapsed = int((time.perf_counter() - started) * 1000)
        keys = _tree_keys(candidates, top_k)
        return BaselineOutcome(
            name=RAPTOR_BASELINE,
            algorithm=algorithm,
            status="ok" if keys else "empty",
            keys=keys,
            details={
                "top_k": top_k,
                "candidates": len(candidates),
                "view": "all",
                "elapsed_ms": elapsed,
                "generation": build.generation,
                "build": {
                    "algorithm": audit["algorithm"],
                    "algorithm_version": audit["algorithm_version"],
                    "generated_at": audit["generated_at"],
                    "corpus": audit["corpus"],
                    "build_params": audit["build_params"],
                    "manifest_fingerprint": audit["manifest_fingerprint"],
                    "provenance_fingerprint": audit["provenance_fingerprint"],
                    "manifest_checked": audit["manifest_checked"],
                },
            },
        )

    def _pageindex(
        self,
        query: str,
        top_k: int,
        papers: Sequence[str] | None,
        algorithm: str,
    ) -> BaselineOutcome:
        search = self.pageindex_search or _default_pageindex_search(self.cfg, self.db)
        started = time.perf_counter()
        results = list(search(query, list(papers or []), top_k) or [])
        elapsed = int((time.perf_counter() - started) * 1000)
        keys = _dedup(
            [
                f"{item.get('paper_id', '')}:{item.get('node_id', '')}"
                for item in results
                if item.get("node_id")
            ]
        )[:top_k]
        return BaselineOutcome(
            name="pageindex",
            algorithm=algorithm,
            status="ok" if keys else "empty",
            keys=tuple(keys),
            details={
                "top_k": top_k,
                "papers": len(papers or []),
                "candidates": len(results),
                "elapsed_ms": elapsed,
                "uses_sql_like": False,
            },
        )


# ── default wiring ───────────────────────────────────────────────────────────


def _runtime_scoped_path(value: Any, *, label: str) -> Path:
    """Resolve a config path under the selected runtime root (mirrors the CLI).

    Config values stay relative; the CLI resolves them against ``DRBRAIN_ROOT``
    when a runtime is selected.  The eval entry must do the same or it reads
    the repository's directory instead of the runtime's.
    """
    import os

    path = Path(str(value or "")).expanduser()
    if not str(path):
        return path
    if "DRBRAIN_ROOT" not in os.environ and "DRBRAIN_RUNTIME_ROOT" not in os.environ:
        return path
    from drbrain.runtime import RuntimeContext

    return RuntimeContext.create().assert_within_root(path, label=label)


def _default_sql_search(cfg: Any, db: Any) -> _SQL_SEARCH:
    from drbrain.rag.config import get_llamaindex_config
    from drbrain.rag.sql_retrie import retrieve_documents_sql

    def search(legs: Sequence[str], query: str, top_k: int) -> list[dict[str, Any]]:
        li = get_llamaindex_config(cfg)
        route_cfg = cfg
        if list(getattr(li, "retrievers", []) or []) != list(legs):
            route_cfg = dataclasses.replace(
                cfg, llamaindex=dataclasses.replace(li, retrievers=list(legs))
            )
        rows = retrieve_documents_sql(route_cfg, db, query, top_k=top_k)
        return [dict(row) for row in rows]

    return search


def _default_unified_tree_search(cfg: Any, db: Any) -> _TREE_SEARCH:
    """Flat all-layer ANN over the active unified-tree generation."""

    def search(query: str, top_k: int) -> list[dict[str, Any]]:
        from drbrain.rag.config import get_llamaindex_config
        from drbrain.services.embedding import _embed_batch
        from drbrain.tree.embedding_identity import profile_from_config
        from drbrain.tree.publish import get_active_tree_generation, resolve_tree_generation
        from drbrain.tree.search import TreeSearch
        from drbrain.tree.vector_store import UnifiedVectorStore

        li = get_llamaindex_config(cfg)
        root = _runtime_scoped_path(li.tree_storage or "data/tree", label="tree storage")
        generation = get_active_tree_generation(root)
        if not generation:
            raise RuntimeError(f"no active tree generation under {root}")
        resolved = resolve_tree_generation(root, generation)
        profile = profile_from_config(cfg.embed)
        with UnifiedVectorStore(Path(resolved["vectors"]), dimension=profile.dimension) as store:
            searcher = TreeSearch(store, profile_id=profile.profile_id(), top_k=top_k)
            candidates = searcher.search_from_text(
                lambda texts: _embed_batch(list(texts), cfg.embed),
                query,
                top_k=top_k,
                view="all",
            )
        return [candidate.to_json() for candidate in candidates]

    return search


def _default_raptor_search(build: RaptorBuild, cfg: Any) -> _RAPTOR_SEARCH:
    """Flat all-layer ANN over the *verified* independent RAPTOR generation.

    Same ANN step as RAPTOR's collapsed tree, but the generation is the one the
    provenance audit accepted — never the unified-tree storage.
    """

    def search(query: str, top_k: int) -> list[dict[str, Any]]:
        from drbrain.services.embedding import _embed_batch
        from drbrain.tree.embedding_identity import profile_from_config
        from drbrain.tree.publish import resolve_tree_generation
        from drbrain.tree.search import TreeSearch
        from drbrain.tree.vector_store import UnifiedVectorStore

        resolved = resolve_tree_generation(build.storage_root, build.generation)
        profile = profile_from_config(cfg.embed)
        with UnifiedVectorStore(Path(resolved["vectors"]), dimension=profile.dimension) as store:
            searcher = TreeSearch(store, profile_id=profile.profile_id(), top_k=top_k)
            candidates = searcher.search_from_text(
                lambda texts: _embed_batch(list(texts), cfg.embed),
                query,
                top_k=top_k,
                view="all",
            )
        return [candidate.to_json() for candidate in candidates]

    return search


def _default_pageindex_search(cfg: Any, db: Any) -> _PAGEINDEX_SEARCH:
    def search(query: str, papers: Sequence[str], top_k: int) -> list[dict[str, Any]]:
        from drbrain.query.tree_retrieval import query_by_structure
        from drbrain.storage.paths import paper_dir, tree_json_path

        models = list(getattr(cfg.llm, "models", None) or [])
        if not models:
            raise RuntimeError("PageIndex baseline needs llm.models")
        papers_root = _runtime_scoped_path(
            getattr(getattr(cfg, "dirs", None), "papers", "data/papers"), label="papers root"
        )
        selected = list(papers) or [
            str(row["local_id"]) for row in (db.get_all_papers() if db is not None else [])
        ]
        out: list[dict[str, Any]] = []
        for local_id in selected:
            directory = paper_dir(papers_root, local_id)
            if not tree_json_path(directory).is_file():
                continue
            results = asyncio.run(query_by_structure(query, directory, models))
            for item in results or []:
                if not isinstance(item, dict):
                    continue
                out.append(
                    {
                        "paper_id": local_id,
                        "node_id": str(item.get("node_id") or ""),
                        "score": float(item.get("score") or 0.0),
                    }
                )
        out.sort(key=lambda row: row["score"], reverse=True)
        return out[:top_k]

    return search


# ── scoring (evaluation entry only) ─────────────────────────────────────────


@dataclass
class BaselineEval:
    name: str
    algorithm: str
    queries: int = 0
    k: int = 10
    hit_rate_paper: float = 0.0
    hit_rate_node: float = 0.0
    mrr_paper: float = 0.0
    mrr_node: float = 0.0
    cost: dict[str, Any] = field(default_factory=dict)
    per_query: list[dict[str, Any]] = field(default_factory=list)
    notes: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "baseline": self.name,
            "algorithm": self.algorithm,
            "queries": self.queries,
            "k": self.k,
            "hit_rate_paper": round(self.hit_rate_paper, 4),
            "hit_rate_node": round(self.hit_rate_node, 4),
            "mrr_paper": round(self.mrr_paper, 4),
            "mrr_node": round(self.mrr_node, 4),
            "cost": self.cost,
            "per_query": self.per_query,
            "notes": list(self.notes),
        }


def _rank_metrics(keys: Sequence[str], entry: dict[str, Any], k: int) -> dict[str, Any]:
    papers = {str(p) for p in entry.get("relevant_papers") or []}
    nodes = {str(n) for n in entry.get("relevant_nodes") or []}
    paper_rank: int | None = None
    node_rank: int | None = None
    for rank, key in enumerate(keys[:k], start=1):
        paper_id, _, node_id = str(key).partition(":")
        if paper_rank is None and paper_id in papers:
            paper_rank = rank
        if node_rank is None and node_id in nodes:
            node_rank = rank
        if paper_rank is not None and node_rank is not None:
            break
    return {"paper_rank": paper_rank, "node_rank": node_rank}


def evaluate_baseline(
    name: str,
    cfg: Any,
    db: Any,
    entries: Sequence[dict[str, Any]],
    *,
    k: int = 10,
    runner: BaselineRunner | None = None,
    papers_for: Callable[[dict[str, Any]], Sequence[str] | None] | None = None,
) -> BaselineEval:
    """Run one baseline over a golden split and score it (evaluation only)."""
    runner = runner or BaselineRunner(cfg=cfg, db=db)
    result = BaselineEval(name=name, algorithm=baseline_algorithm(name), k=k)
    if not entries:
        result.notes = ("no golden entries",)
        return result
    total_paper = total_node = 0.0
    elapsed_ms = 0
    for entry in entries:
        papers = papers_for(entry) if papers_for else entry.get("relevant_papers")
        outcome = runner.run(name, str(entry.get("query") or ""), top_k=k, papers=papers)
        metrics = _rank_metrics(outcome.keys, entry, k)
        elapsed_ms += int(outcome.details.get("elapsed_ms") or 0)
        result.queries += 1
        if metrics["paper_rank"]:
            total_paper += 1.0 / metrics["paper_rank"]
        if metrics["node_rank"]:
            total_node += 1.0 / metrics["node_rank"]
        result.per_query.append({"id": entry.get("id"), "status": outcome.status, **metrics})
    result.hit_rate_paper = sum(1 for row in result.per_query if row["paper_rank"]) / result.queries
    result.hit_rate_node = sum(1 for row in result.per_query if row["node_rank"]) / result.queries
    result.mrr_paper = total_paper / result.queries
    result.mrr_node = total_node / result.queries
    unavailable = sum(1 for row in result.per_query if row["status"] == "unavailable")
    result.cost = {
        "elapsed_ms": elapsed_ms,
        "queries": result.queries,
        "k": k,
        "unavailable_queries": unavailable,
        # a run that could not execute is not a comparison, even if it scores 0
        "usable": unavailable < result.queries,
        "production_route": False,
    }
    if unavailable == result.queries:
        result.notes = (
            *result.notes,
            "every query was unavailable; these metrics are not a usable comparison",
        )
    return result


__all__ = [
    "BASELINES",
    "DEFAULT_RAPTOR_STORAGE",
    "RAPTOR_ALGORITHM",
    "RAPTOR_BASELINE",
    "RAPTOR_PROVENANCE_FIELDS",
    "RAPTOR_PROVENANCE_NAME",
    "UNIFIED_TREE_ABLATION",
    "UNIFIED_TREE_ABLATION_NOTE",
    "BaselineEval",
    "BaselineOutcome",
    "BaselineProvenanceError",
    "BaselineRunner",
    "RaptorBuild",
    "baseline_algorithm",
    "evaluate_baseline",
    "load_raptor_build",
    "raptor_provenance_fingerprint",
    "read_raptor_provenance",
    "verify_raptor_provenance",
]
