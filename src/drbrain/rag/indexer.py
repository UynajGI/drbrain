"""Index build orchestration and compatible index-loading APIs.

Node preparation, embedding execution and immutable generation management live
in index_nodes, index_embeddings and index_generations. SQL publication copies
the corpus through the same generation lifecycle without running embeddings."""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from loguru import logger

from drbrain.config import Config
from drbrain.rag.config import get_llamaindex_config
from drbrain.rag.index_embeddings import (
    _default_embed_model as _default_embed_model,
)
from drbrain.rag.index_embeddings import (
    _embed_chunk_worker as _embed_chunk_worker,
)
from drbrain.rag.index_embeddings import (
    _embed_devices as _embed_devices,
)
from drbrain.rag.index_embeddings import (
    _embed_model_path as _embed_model_path,
)
from drbrain.rag.index_embeddings import (
    _init_embed_worker as _init_embed_worker,
)
from drbrain.rag.index_embeddings import (
    _load_old_embeddings as _load_old_embeddings,
)
from drbrain.rag.index_embeddings import (
    _load_old_index_nodes as _load_old_index_nodes,
)
from drbrain.rag.index_embeddings import preserve_gc_state
from drbrain.rag.index_generations import (
    ACTIVE_POINTER_NAME as ACTIVE_POINTER_NAME,
)
from drbrain.rag.index_generations import (
    GENERATION_PRUNE_GRACE_SECONDS as GENERATION_PRUNE_GRACE_SECONDS,
)
from drbrain.rag.index_generations import (
    GENERATION_REFERENCES_DIR_NAME as GENERATION_REFERENCES_DIR_NAME,
)
from drbrain.rag.index_generations import (
    GENERATION_REFERENCES_NAME as GENERATION_REFERENCES_NAME,
)
from drbrain.rag.index_generations import (
    GENERATION_RETAIN_COUNT as GENERATION_RETAIN_COUNT,
)
from drbrain.rag.index_generations import (
    GENERATIONS_DIR_NAME as GENERATIONS_DIR_NAME,
)
from drbrain.rag.index_generations import (
    LEGACY_INDEX_GENERATION as LEGACY_INDEX_GENERATION,
)
from drbrain.rag.index_generations import (
    MANIFEST_NAME as MANIFEST_NAME,
)
from drbrain.rag.index_generations import (
    _active_storage_root as _active_storage_root,
)
from drbrain.rag.index_generations import (
    _generation_references as _generation_references,
)
from drbrain.rag.index_generations import (
    _load_manifest as _load_manifest,
)
from drbrain.rag.index_generations import (
    _new_generation_id as _new_generation_id,
)
from drbrain.rag.index_generations import (
    _pointer_path as _pointer_path,
)
from drbrain.rag.index_generations import (
    _prune_inactive_generations as _prune_inactive_generations,
)
from drbrain.rag.index_generations import (
    _referenced_generations as _referenced_generations,
)
from drbrain.rag.index_generations import (
    _storage_dirs as _storage_dirs,
)
from drbrain.rag.index_generations import (
    _storage_root_for_generation as _storage_root_for_generation,
)
from drbrain.rag.index_generations import (
    _write_json_atomically as _write_json_atomically,
)
from drbrain.rag.index_generations import (
    _write_manifest as _write_manifest,
)
from drbrain.rag.index_generations import (
    capture_index_generation as capture_index_generation,
)
from drbrain.rag.index_generations import (
    get_active_index_generation as get_active_index_generation,
)
from drbrain.rag.index_generations import (
    retain_index_generation as retain_index_generation,
)
from drbrain.rag.index_nodes import (
    CHARS_PER_TOKEN as CHARS_PER_TOKEN,
)
from drbrain.rag.index_nodes import (
    DEFAULT_MAX_NODE_TOKENS as DEFAULT_MAX_NODE_TOKENS,
)
from drbrain.rag.index_nodes import (
    TREE_LAYER_PAGEINDEX as TREE_LAYER_PAGEINDEX,
)
from drbrain.rag.index_nodes import (
    _chunk_document as _chunk_document,
)
from drbrain.rag.index_nodes import (
    _content_hash as _content_hash,
)
from drbrain.rag.index_nodes import (
    _node_key as _node_key,
)
from drbrain.rag.index_nodes import (
    collect_tree_nodes as collect_tree_nodes,
)
from drbrain.storage.paths import (
    resolve_paper_dir,
)

try:  # pragma: no cover - exercised in environments without llama-index
    from llama_index.core import VectorStoreIndex, load_index_from_storage
    from llama_index.core.schema import Document, TextNode
    from llama_index.core.storage import StorageContext
    from llama_index.core.vector_stores import SimpleVectorStore
    from llama_index.retrievers.bm25 import BM25Retriever

    _LLAMA_INDEX_AVAILABLE = True
except ImportError:  # pragma: no cover - envs without llama-index
    Document = TextNode = VectorStoreIndex = load_index_from_storage = None  # type: ignore[assignment,misc]
    StorageContext = SimpleVectorStore = BM25Retriever = None  # type: ignore[assignment,misc]
    _LLAMA_INDEX_AVAILABLE = False

__all__ = [
    "_LLAMA_INDEX_AVAILABLE",
    "LEGACY_INDEX_GENERATION",
    "MANIFEST_NAME",
    "build_index",
    "capture_index_generation",
    "collect_tree_nodes",
    "get_active_index_generation",
    "get_index_health",
    "load_index",
    "retain_index_generation",
]


def _resolve_paper_dir(papers_root: Path, pid: str) -> Path | None:
    """Locate a paper's asset directory across on-disk layouts.

    Corpus ingests use different layouts: newer papers live in flat
    ``<papers>/<local_id>/`` dirs (e.g. ``p00005bb6``), while DOI-keyed
    papers are nested one level (``<papers>/<prefix>/<suffix>``) or use an
    underscore-sanitized flat name.  Returns the first existing layout.
    """
    return resolve_paper_dir(papers_root, pid)


# ── Build ────────────────────────────────────────────────────────────────────


@preserve_gc_state
def build_index(
    cfg: Config,
    db: Any,
    paper_ids: Iterable[str] | None = None,
    force: bool = False,
    embed_model: Any | None = None,
    max_node_tokens: int | None = None,
) -> dict[str, Any]:
    """Publish the selected backend's immutable retrieval generation.

    SQL publishes a copy of the prepared corpus database; it does not build
    embeddings and rejects paper selection or embedding-build overrides.

    For every target paper (``paper_ids``, or all papers known to ``db`` when
    ``None``), collects its PageIndex Documents, embeds only the nodes whose
    ``content_hash`` changed since the last build (``force=True`` re-embeds
    everything), persists ``VectorStoreIndex`` + ``BM25Retriever`` under
    ``llamaindex.storage_dir``, and records hashes in ``manifest.json``.

    For LlamaIndex, ``max_node_tokens`` (default ``llamaindex.max_node_tokens``)
    estimates a character cap per embedded fragment. Oversized nodes become
    exact parent-text slices with unique ids and explicit parent locators.
    Change detection stays per parent node — an
    unchanged node reuses all its chunk embeddings; only changed nodes (or
    nodes with missing chunk embeddings) are re-embedded. Non-positive values
    retain the safe default cap.

    ``db`` only needs ``get_all_papers() -> list[{"local_id": str}]``. Nodes
    whose embedding cannot be reused (no persisted store, or the configured
    embed model changed) are re-embedded automatically.

    Returns historic stats plus additive ``generation`` and
    ``previous_generation`` after a successful publish.
    """
    if get_llamaindex_config(cfg).rag_engine == "sql":
        from drbrain.rag.sql_snapshot import publish_sql_snapshot

        if paper_ids is not None or embed_model is not None or max_node_tokens is not None:
            raise ValueError(
                "SQL publication snapshots the corpus; per-paper embedding options are unsupported"
            )
        return publish_sql_snapshot(cfg)
    if not _LLAMA_INDEX_AVAILABLE:
        raise RuntimeError(
            "llama-index is not installed; run `uv add llama-index-core llama-index-retrievers-bm25`"
        )
    # A full-library build holds millions of tracked containers; CPython's
    # cyclic GC scans them repeatedly during the alloc-heavy phases and
    # burns minutes of CPU. This function never relies on cycle collection,
    # so turn it off for the duration.
    import gc as _gc

    _gc_was_enabled = _gc.isenabled()
    _gc.disable()

    li = get_llamaindex_config(cfg)
    storage_root = Path(li.storage_dir)
    active_store = _active_storage_root(li.storage_dir)
    if active_store is None:
        # A broken pointer must not be used as a cache source. A successful
        # rebuild below can still recover by publishing a fresh generation.
        logger.warning("[rag] active generation pointer is invalid; rebuilding without reuse")
        source_root, previous_generation = storage_root, None
    else:
        source_root, previous_generation = active_store
    _, vector_dir, bm25_dir = _storage_dirs(source_root)
    papers_root = Path(cfg.dirs.papers)
    top_k = cfg.embed.top_k or 10
    if max_node_tokens is None:
        max_node_tokens = int(getattr(li, "max_node_tokens", DEFAULT_MAX_NODE_TOKENS))
    else:
        max_node_tokens = int(max_node_tokens)
    if max_node_tokens <= 0:
        logger.warning(
            "[rag] max_node_tokens=%s is unsafe; using default %d",
            max_node_tokens,
            DEFAULT_MAX_NODE_TOKENS,
        )
        max_node_tokens = DEFAULT_MAX_NODE_TOKENS

    target_ids = (
        list(paper_ids) if paper_ids is not None else [p["local_id"] for p in db.get_all_papers()]
    )

    # 1. Collect parent documents + split into chunk documents.
    #    ``docs_by_key`` holds one Document per tree node (parent), keyed by
    #    ``paper_id:node_id``; ``chunk_docs`` holds the (possibly split) chunk
    #    Documents keyed by ``paper_id:node_id#i``. Change detection runs on
    #    parent keys; embedding + indexing run on chunk keys.
    docs_by_key: dict[str, Document] = {}
    chunk_docs: dict[str, Document] = {}
    parent_chunks: dict[str, list[str]] = {}
    paper_keys: dict[str, list[str]] = {}

    import concurrent.futures as _cf

    def _collect_one(pid: str) -> tuple[str, list[Document] | None]:
        paper_dir = _resolve_paper_dir(papers_root, pid)
        if paper_dir is None:
            return pid, None
        return pid, collect_tree_nodes(paper_dir, paper_id=pid)

    missing_dirs = 0
    sorted_ids = sorted(str(p) for p in target_ids)
    with _cf.ThreadPoolExecutor(max_workers=32) as _ex:
        for pid, docs in _ex.map(_collect_one, sorted_ids, chunksize=64):
            if docs is None:
                missing_dirs += 1
                continue
            if not docs:
                logger.info("[rag] no PageIndex nodes for %s (missing tree.json?)", pid)
                continue
            paper_keys[pid] = [doc.id_ for doc in docs]
            docs_by_key.update((doc.id_, doc) for doc in docs)
            for doc in docs:
                chunks = _chunk_document(doc, max_node_tokens) if max_node_tokens > 0 else [doc]
                parent_chunks[doc.id_] = [c.id_ for c in chunks]
                chunk_docs.update((c.id_, c) for c in chunks)
    if missing_dirs:
        logger.warning("[rag] {} papers had no dir on disk and were skipped", missing_dirs)

    hashes = {key: _content_hash(doc.text) for key, doc in docs_by_key.items()}

    # 2. Diff against the previous manifest (per parent node).
    manifest = _load_manifest(li.storage_dir)
    old_papers: dict[str, dict[str, str]] = manifest.get("papers", {})
    old_model = manifest.get("embed_model")
    new_model = cfg.embed.model
    format_changed = bool(manifest) and (
        manifest.get("fragment_format") != 2 or manifest.get("max_node_tokens") != max_node_tokens
    )
    if format_changed and paper_ids is not None:
        raise ValueError("fragment format changed; run a full rag index before indexing a subset")
    model_changed = (bool(old_model) and old_model != new_model) or format_changed

    changed: set[str] = set()
    if force or model_changed:
        changed = set(hashes)
    else:
        for key, h in hashes.items():
            if old_papers.get(key.split(":", 1)[0], {}).get(key) != h:
                changed.add(key)

    removed: set[str] = set()
    for pid, old_keys in old_papers.items():
        if pid in paper_keys:
            removed.update(k for k in old_keys if k not in hashes)

    # 3. Embed only what needs fresh embeddings: every changed node's chunks,
    #    plus any unchanged node whose chunk embeddings are missing from the
    #    previous store (crash-recovery).
    old_embeddings = {} if force else _load_old_embeddings(vector_dir)
    embed = embed_model or _default_embed_model(cfg)

    still_changed = set(changed)
    for key in hashes:
        if key in changed:
            continue
        if any(ck not in old_embeddings for ck in parent_chunks[key]):
            still_changed.add(key)

    embed_keys: list[str] = [ck for key in sorted(still_changed) for ck in parent_chunks[key]]
    embedded_texts = [chunk_docs[k].text for k in embed_keys]
    new_embeddings: dict[str, list[float]] = {}
    # Reuse pipeline-computed vectors (tree_vectors, pageindex layer) when the
    # content hash matches: node keys are identical ("paper:node") and both
    # sides use the same sha256[:16] hash, so a matching hash proves identical
    # text. This skips re-embedding leaf nodes the corpus pipeline already
    # embedded; split chunks (#i) and unseen nodes still embed below.
    if embed_keys and not force:
        try:
            import sqlite3 as _sqlite3

            import numpy as _np

            _conn = _sqlite3.connect(f"file:{cfg.db.path}?mode=ro", uri=True)
            _parent_keys = sorted({k for k in embed_keys if "#" not in k})
            _batch_size = 10000
            # Pass 1: hash-only probe (hash column is tiny; fetching the 4KB
            # embedding blob for every row costs GBs of pointless transfer).
            _matched: list[str] = []
            for _s in range(0, len(_parent_keys), _batch_size):
                _batch = _parent_keys[_s : _s + _batch_size]
                _q = ",".join("?" * len(_batch))
                for _nid, _h in _conn.execute(
                    "SELECT node_id, content_hash FROM tree_vectors "
                    f"WHERE tree_layer = 'pageindex' AND node_id IN ({_q})",
                    _batch,
                ):
                    if hashes.get(_nid) == _h:
                        _matched.append(_nid)
            # Pass 2: fetch blobs for matches only.
            from drbrain.storage.vector_index import embedding_byte_len as _eb_len

            _expected_bytes = _eb_len(_conn)
            _reuse: dict[str, list[float]] = {}
            for _s in range(0, len(_matched), _batch_size):
                _batch = _matched[_s : _s + _batch_size]
                _q = ",".join("?" * len(_batch))
                for _nid, _blob in _conn.execute(
                    f"SELECT node_id, embedding FROM tree_vectors WHERE node_id IN ({_q})",
                    _batch,
                ):
                    if len(_blob) == _expected_bytes:
                        _reuse[_nid] = _np.frombuffer(_blob, dtype=_np.float32).tolist()
            _conn.close()
            if _reuse:
                new_embeddings.update(_reuse)
                embed_keys = [k for k in embed_keys if k not in _reuse]
                embedded_texts = [chunk_docs[k].text for k in embed_keys]
                logger.info(
                    "[rag] reused {} vectors from tree_vectors ({} left to embed)",
                    len(_reuse),
                    len(embed_keys),
                )
        except Exception as exc:  # noqa: BLE001 - reuse is an optimization only
            logger.warning("[rag] tree_vectors reuse skipped: %s", exc)
    if embedded_texts:
        # Chunked embedding: one giant batch holds the whole corpus in the
        # provider call (hours for millions of nodes, zero crash tolerance).
        # Chunks give progress visibility and bound provider-side memory.
        _embed_chunk = 5000
        vectors: list[list[float]] = []
        total_chunks = (len(embedded_texts) + _embed_chunk - 1) // _embed_chunk
        _devices = _embed_devices(cfg) if getattr(cfg.embed, "provider", "") == "local" else []
        if len(_devices) > 1 and total_chunks > 1:
            # Fan chunks out over multiple GPUs (spawn pool, one model per GPU).
            import multiprocessing as _mp

            _model_path = _embed_model_path(cfg)
            _chunks = [
                embedded_texts[s : s + _embed_chunk]
                for s in range(0, len(embedded_texts), _embed_chunk)
            ]
            _jobs = [
                (_model_path, _devices[i % len(_devices)], c, i + 1, total_chunks)
                for i, c in enumerate(_chunks)
            ]
            _ctx = _mp.get_context("spawn")
            _worker_counter = _ctx.Value("i", 0)
            logger.info("[rag] parallel embedding on GPUs {}", _devices)
            with _ctx.Pool(
                len(_devices), initializer=_init_embed_worker, initargs=(_devices, _worker_counter)
            ) as _pool:
                for _arr in _pool.imap(_embed_chunk_worker, _jobs, chunksize=2):
                    vectors.extend(_arr.tolist() if hasattr(_arr, "tolist") else _arr)
        else:
            for ci, start in enumerate(range(0, len(embedded_texts), _embed_chunk), 1):
                logger.info(
                    "[rag] embedding chunk {}/{} ({} nodes)",
                    ci,
                    total_chunks,
                    min(_embed_chunk, len(embedded_texts) - start),
                )
                vectors.extend(
                    embed.get_text_embedding_batch(embedded_texts[start : start + _embed_chunk])
                )
        if len(vectors) != len(embed_keys):
            raise RuntimeError(
                "embedding batch returned "
                f"{len(vectors)} vectors for {len(embed_keys)} requested nodes; generation not published"
            )
        for key, vec in zip(embed_keys, vectors):
            if not vec:
                raise RuntimeError(
                    f"embedding batch returned an empty vector for {key}; generation not published"
                )
            new_embeddings[key] = vec

    # 3b. For a --paper subset rebuild, carry over non-target papers' already
    #     indexed nodes so the persisted index keeps covering the whole library.
    carried: dict[str, TextNode] = {}
    if paper_ids is not None:
        target_set = {str(p) for p in target_ids}
        for key, node in _load_old_index_nodes(vector_dir, embed).items():
            if node.metadata.get("paper_id") not in target_set:
                carried[key] = node

    # 4. Assemble pre-embedded TextNodes (one per chunk; sub-cap nodes stay
    #    single). Chunk order follows parent-node order then chunk index.
    nodes: list[TextNode] = []
    for key in sorted(hashes):
        for ck in parent_chunks[key]:
            doc = chunk_docs[ck]
            embedding = new_embeddings.get(ck) or old_embeddings.get(ck)
            if embedding is None:  # pragma: no cover - defensive
                logger.warning("[rag] node %s has no embedding; skipping", ck)
                continue
            nodes.append(
                TextNode(
                    text=doc.text,
                    id_=ck,
                    metadata=dict(doc.metadata),
                    embedding=embedding,
                )
            )
    nodes.extend(carried[key] for key in sorted(carried))

    stats: dict[str, Any] = {
        "papers": len(paper_keys),
        "nodes": len(nodes),
        "chunked": sum(1 for ks in parent_chunks.values() if len(ks) > 1),
        "carried": len(carried),
        "embedded": len(embed_keys),
        "unchanged": len(hashes) - len(still_changed),
        "removed": len(removed),
        "bm25_nodes": 0,
        "storage_dir": str(storage_root),
        "previous_generation": previous_generation,
    }
    if not nodes:
        logger.warning("[rag] nothing to index; leaving existing indexes untouched")
        if _gc_was_enabled:
            _gc.enable()
        return stats

    # 5. Build a complete new generation. Nothing below the active pointer is
    # touched until both artifacts deserialize successfully.
    generations_root = storage_root / GENERATIONS_DIR_NAME
    generation = _new_generation_id()
    stage_root = generations_root / f".staging-{generation}"
    final_root = generations_root / generation
    _, stage_vector_dir, stage_bm25_dir = _storage_dirs(stage_root)
    generations_root.mkdir(parents=True, exist_ok=True)
    if stage_root.exists() or final_root.exists():  # never overwrite a generation
        raise RuntimeError(f"index generation collision: {generation}")
    try:
        index = VectorStoreIndex(nodes=nodes, embed_model=embed)
        index.storage_context.persist(str(stage_vector_dir))
        logger.info(
            "[rag] staged vector index (%d nodes) at %s",
            len(nodes),
            stage_vector_dir,
        )

        bm25 = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k)
        bm25.persist(str(stage_bm25_dir))
        stats["bm25_nodes"] = len(bm25.corpus)
        logger.info("[rag] staged BM25 index (%d docs) at %s", len(bm25.corpus), stage_bm25_dir)

        # 6. Update manifest: target papers replaced, others preserved.
        new_papers: dict[str, dict[str, str]] = {
            pid: {k: hashes[k] for k in keys if k in hashes} for pid, keys in paper_keys.items()
        }
        for pid, old_keys in old_papers.items():
            if pid not in new_papers:
                new_papers[pid] = dict(old_keys)
        manifest = {
            "generation": generation,
            "embed_model": new_model,
            "fragment_format": 2,
            "max_node_tokens": max_node_tokens,
            "vector_store": li.vector_store,
            "papers": new_papers,
        }
        _write_manifest(stage_root, manifest)

        # Validate both staged artifacts before activation. A failed validation
        # leaves the prior active generation untouched for readers and operators.
        staged_context = StorageContext.from_defaults(persist_dir=str(stage_vector_dir))
        load_index_from_storage(staged_context, embed_model=embed)
        BM25Retriever.from_persist_dir(str(stage_bm25_dir))
    except Exception as exc:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise RuntimeError(
            f"staged generation validation failed; generation not published: {exc}"
        ) from exc

    try:
        stage_root.replace(final_root)
    except Exception as exc:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise RuntimeError(
            f"could not finalize staged generation; generation not published: {exc}"
        ) from exc
    _write_json_atomically(
        _pointer_path(storage_root),
        {"generation": generation, "updated_at_ns": time.time_ns()},
    )
    # The root manifest is a non-authoritative compatibility mirror. A mirror
    # failure must not report a published generation as a failed build.
    try:
        _write_manifest(storage_root, manifest)
    except OSError as exc:
        logger.warning(
            "[rag] published generation %s but could not mirror manifest: %s", generation, exc
        )
    try:
        stats["pruned_generations"] = _prune_inactive_generations(
            storage_root,
            generation,
            protected_generations=_referenced_generations(storage_root),
        )
    except OSError as exc:
        logger.warning(
            "[rag] published generation %s but could not prune stale generations: %s",
            generation,
            exc,
        )
        stats["pruned_generations"] = []
    stats["generation"] = generation
    logger.info("[rag] index build done: %s", stats)
    if _gc_was_enabled:
        _gc.enable()
    return stats


# ── Load ─────────────────────────────────────────────────────────────────────


def load_index(
    cfg: Config,
    embed_model: Any | None = None,
    *,
    generation: str | None = None,
) -> tuple[Any | None, Any | None]:
    """Load the persisted ``(VectorStoreIndex, BM25Retriever)`` from disk.

    Returns ``(None, None)`` when llama-index is unavailable or no index has
    been built yet. The embed model defaults to the T1 DrbrainEmbedding
    adapter (lazy — no model load happens here unless a query embeds).

    ``generation`` is additive.  When supplied it resolves that immutable
    generation (or the explicit ``"legacy"`` snapshot) instead of following
    ``active.json``; callers that omit it keep the historic active-pointer
    behavior.
    """
    if not _LLAMA_INDEX_AVAILABLE:
        return None, None

    li = get_llamaindex_config(cfg)
    active_root = _storage_root_for_generation(li.storage_dir, generation)
    if active_root is None:
        if generation is None:
            logger.error("[rag] active generation pointer is invalid at %s", li.storage_dir)
        else:
            logger.error(
                "[rag] requested generation %r is unavailable at %s", generation, li.storage_dir
            )
        return None, None
    _, vector_dir, bm25_dir = _storage_dirs(active_root)
    embed = embed_model or _default_embed_model(cfg)
    top_k = cfg.embed.top_k or 10

    index = None
    if (vector_dir / "docstore.json").exists():
        try:
            sc = StorageContext.from_defaults(persist_dir=str(vector_dir))
            index = load_index_from_storage(sc, embed_model=embed)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("[rag] failed to load vector index from %s: %s", vector_dir, exc)

    bm25 = None
    if (bm25_dir / "corpus.jsonl").exists():
        try:
            bm25 = BM25Retriever.from_persist_dir(str(bm25_dir))
            bm25.similarity_top_k = top_k
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("[rag] failed to load BM25 index from %s: %s", bm25_dir, exc)

    return index, bm25


def get_index_health(cfg: Config) -> dict[str, Any]:
    """Inspect persisted RAG readiness without querying, embedding, or writing.

    The existing :func:`load_index` tuple is intentionally unchanged for every
    caller. This additive report makes deployment automation able to distinguish
    a disabled feature, an unavailable dependency, broken artifacts, and a
    deserialization failure before accepting retrieval traffic. The final
    validation deserializes persisted indexes, so this is a read-only but not
    constant-time operation.
    """
    li = get_llamaindex_config(cfg)
    if li.rag_engine == "sql":
        from drbrain.rag.sql_snapshot import sql_index_health

        return sql_index_health(cfg)
    root = Path(li.storage_dir)
    pointer_path = _pointer_path(root)
    active_store = _active_storage_root(li.storage_dir)
    active_root = active_store[0] if active_store is not None else root
    active_generation = active_store[1] if active_store is not None else None
    _, vector_dir, bm25_dir = _storage_dirs(active_root)
    manifest_path = active_root / MANIFEST_NAME
    vector_docstore = vector_dir / "docstore.json"
    vector_store = vector_dir / "default__vector_store.json"
    bm25_corpus = bm25_dir / "corpus.jsonl"
    checks: dict[str, Any] = {
        "config_enabled": bool(li.enabled),
        "llama_index_available": _LLAMA_INDEX_AVAILABLE,
        "generation": {
            "pointer_path": str(pointer_path),
            "pointer_exists": pointer_path.exists(),
            "active": active_generation,
            "active_path": str(active_root),
            "valid": active_store is not None,
        },
        "manifest": {
            "path": str(manifest_path),
            "exists": manifest_path.exists(),
            "valid": False,
            "embed_model": None,
            "embed_model_matches_config": False,
            "vector_store": None,
            "vector_store_matches_config": False,
            "paper_count": 0,
            "parent_node_count": 0,
        },
        "vector": {
            "path": str(vector_dir),
            "docstore_exists": vector_docstore.exists(),
            "vector_store_exists": vector_store.exists(),
            "loadable": None,
        },
        "bm25": {
            "path": str(bm25_dir),
            "corpus_exists": bm25_corpus.exists(),
            "loadable": None,
        },
    }
    reasons: list[str] = []

    if not li.enabled:
        reasons.append("config_disabled")
    if not _LLAMA_INDEX_AVAILABLE:
        reasons.append("llama_index_unavailable")
    if pointer_path.exists() and active_store is None:
        reasons.append("active_generation_invalid")
    if reasons:
        return {
            "ready": False,
            "status": "not_ready",
            "storage_dir": str(root),
            "checks": checks,
            "reasons": reasons,
        }

    manifest: dict[str, Any] | None = None
    if manifest_path.exists():
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("manifest must be an object")
            papers = raw.get("papers")
            valid_papers = isinstance(papers, dict) and all(
                isinstance(paper_id, str)
                and isinstance(nodes, dict)
                and all(
                    isinstance(node_id, str) and isinstance(content_hash, str)
                    for node_id, content_hash in nodes.items()
                )
                for paper_id, nodes in papers.items()
            )
            if (
                not isinstance(raw.get("embed_model"), str)
                or not isinstance(raw.get("vector_store"), str)
                or not valid_papers
            ):
                raise ValueError("manifest has an unsupported shape")
            manifest = raw
        except (OSError, ValueError, json.JSONDecodeError):
            reasons.append("manifest_invalid")
    else:
        reasons.append("manifest_missing")

    if manifest is not None:
        papers = manifest["papers"]
        parent_node_count = sum(len(nodes) for nodes in papers.values() if isinstance(nodes, dict))
        embed_model_matches = manifest["embed_model"] == cfg.embed.model
        vector_store_matches = manifest["vector_store"] == li.vector_store
        checks["manifest"].update(
            {
                "valid": True,
                "embed_model": manifest["embed_model"],
                "embed_model_matches_config": embed_model_matches,
                "vector_store": manifest["vector_store"],
                "vector_store_matches_config": vector_store_matches,
                "paper_count": len(papers),
                "parent_node_count": parent_node_count,
            }
        )
        if not embed_model_matches:
            reasons.append("embed_model_mismatch")
        if not vector_store_matches:
            reasons.append("vector_store_mismatch")

    if not vector_docstore.exists():
        reasons.append("vector_docstore_missing")
    if not vector_store.exists():
        reasons.append("vector_store_missing")
    if not bm25_corpus.exists():
        reasons.append("bm25_corpus_missing")

    static_ready = not reasons
    if static_ready:
        index, bm25 = load_index(cfg)
        checks["vector"]["loadable"] = index is not None
        checks["bm25"]["loadable"] = bm25 is not None
        if index is None:
            reasons.append("vector_unloadable")
        if bm25 is None:
            reasons.append("bm25_unloadable")

    ready = not reasons
    return {
        "ready": ready,
        "status": "ready" if ready else "not_ready",
        "storage_dir": str(root),
        "checks": checks,
        "reasons": reasons,
    }
