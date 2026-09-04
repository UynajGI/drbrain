"""Index layer: tree.json/raw.md → LlamaIndex Documents/Nodes; VectorStoreIndex + BM25.

Ticket: T3 (索引层). Depends on T1.

Converts drbrain's PageIndex assets (``tree.json`` + ``raw.md`` per paper) into
LlamaIndex objects:

* :func:`collect_tree_nodes` — one :class:`~llama_index.core.schema.Document`
  per tree node (``paper_id:node_id`` unique key, PageIndex metadata).
* :func:`build_index` — incremental ``VectorStoreIndex`` build (embeds only
  nodes whose ``content_hash`` changed) + persistent BM25 inverted index.
  New builds are staged as a complete generation, validated, then activated by
  an atomic pointer swap so readers never observe mixed vector/BM25 artifacts.
* :func:`load_index` — restore ``(VectorStoreIndex, BM25Retriever)`` from disk
  without rebuilding.

Legacy persistence layout under ``storage_dir``::

    storage_dir/
      manifest.json      # embed_model + {paper_id: {node_key: content_hash}}
      vector/            # StorageContext.persist (docstore, index_store, SimpleVectorStore)
      bm25/              # BM25Retriever.persist (bm25s index + corpus)

New builds preserve the legacy files for migration compatibility but read from
the active generation::

    storage_dir/
      active.json         # atomically swapped {"generation": "..."}
      generations/<id>/
        manifest.json
        vector/
        bm25/

Everything degrades gracefully when llama-index is not installed: the CLI
fails with a clear message instead of a traceback, and importing the module
never raises.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from loguru import logger

from drbrain.config import Config
from drbrain.rag.config import get_llamaindex_config

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

#: Layer tag stored on PageIndex nodes (mirrors ``tree_vectors.tree_layer``).
TREE_LAYER_PAGEINDEX = "pageindex"
#: Filename of the incremental-update manifest under ``storage_dir``.
MANIFEST_NAME = "manifest.json"
#: Atomically swapped pointer to the complete index generation readers use.
ACTIVE_POINTER_NAME = "active.json"
#: Directory holding immutable, fully-built index generations.
GENERATIONS_DIR_NAME = "generations"
#: Explicit snapshot label for the pre-generation flat storage layout.
LEGACY_INDEX_GENERATION = "legacy"
#: Completed generations retained for rollback and in-flight readers.
GENERATION_RETAIN_COUNT = 3
#: Minimum completed-generation age before automatic retention pruning.
GENERATION_PRUNE_GRACE_SECONDS = 3600.0
#: Durable autoresearch run -> immutable generation references.
GENERATION_REFERENCES_NAME = "generation-references.json"
#: Per-run references avoid read-modify-write races between concurrent directors.
GENERATION_REFERENCES_DIR_NAME = "generation-references"
#: Default cap for a single embedded node, in LLM tokens. PageIndex nodes can
#: exceed this (the real corpus has 39-94KB Abstract/References nodes ≈ 9-23k
#: tokens); embedding them in one forward pass OOMs a 16GB fp32 GPU (T3/T7
#: finding). Nodes above the cap are split into paragraph chunks at
#: ``4 chars/token`` — each chunk inherits the parent node_id + ``#i`` suffix
#: (T9 decision: split, not truncate — preserves full content).
#:
#: 4000, not 8000: the GPU memory profile of the Qwen3-Embedding-0.6B fp32
#: forward pass is quadratic in sequence length (measured per-sample: 4096
#: tokens ≈ 3.6GB, 8192 tokens ≈ 12.2GB). On a 16GB V100 (≈2.5GB weights +
#: ~1GB overhead) a single 8000-token sequence lands at ~15GB and OOMs;
#: 4000-token chunks leave comfortable headroom (T9 GPU verification).
DEFAULT_MAX_NODE_TOKENS = 4000
#: Chars-per-token heuristic used to convert ``max_node_tokens`` → char cap.
CHARS_PER_TOKEN = 4


def _content_hash(text: str) -> str:
    """Stable content hash for incremental update detection (sha256[:16])."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _node_key(paper_id: str, node_id: str) -> str:
    """Globally-unique node key (tree ``node_id``s are only unique per paper)."""
    return f"{paper_id}:{node_id}"


def _paragraph_chunks(text: str, max_chars: int) -> list[str]:
    """Split ``text`` into paragraph-boundary chunks of at most ``max_chars``.

    Greedy accumulation over ``\n\n``-separated paragraphs, preserving the
    original text verbatim (only boundaries are chosen). A single paragraph
    longer than the cap is hard-sliced at the cap, so the worst case stays
    bounded even for ``\n``-only bodies.
    """
    if len(text) <= max_chars:
        return [text]
    paras = text.split("\n\n")
    chunks: list[str] = []
    cur = ""
    for p in paras:
        while len(p) > max_chars:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(p[:max_chars])
            p = p[max_chars:]
        if not p:
            continue
        if cur and len(cur) + len(p) + 2 > max_chars:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks or [text]


def _chunk_document(doc: Document, max_node_tokens: int) -> list[Document]:
    """Split an over-long Document into paragraph chunks (T9 OOM fix).

    A node whose text exceeds ``max_node_tokens`` tokens (≈ 4 chars/token) is
    split at paragraph boundaries. Every chunk:
      * keeps the parent's metadata (``paper_id``/``node_id``/``line_*``) so
        downstream consumers (sources, eval node-level grading) still see the
        original tree node;
      * adds ``chunk_index`` / ``chunk_count``;
      * is re-prefixed with the section title (aids BM25/vector matching);
      * gets id ``<paper_id:node_id>#<i>`` (distinct from the parent key so
        fusion dedup and the manifest never collide).

    Sub-cap nodes are returned as-is (one Document, no ``#`` suffix).
    """
    max_chars = max(1, int(max_node_tokens)) * CHARS_PER_TOKEN
    if len(doc.text) <= max_chars:
        return [doc]
    title = str(doc.metadata.get("title") or "")
    body = doc.text
    if title and body.startswith(title + "\n"):
        body = body[len(title) + 1 :]
    parts = _paragraph_chunks(body, max_chars)
    out: list[Document] = []
    for i, part in enumerate(parts):
        md = dict(doc.metadata)
        md["chunk_index"] = i
        md["chunk_count"] = len(parts)
        chunk_text = f"{title}\n{part}".strip() if title else part
        out.append(
            Document(
                text=chunk_text,
                id_=f"{doc.id_}#{i}",
                metadata=md,
            )
        )
    return out


# ── Document collection ──────────────────────────────────────────────────────


def collect_tree_nodes(
    paper_dir: str | Path,
    tree_json: str | Path | dict | None = None,
    max_node_tokens: int | None = None,
) -> list[Document]:
    """Collect one :class:`Document` per PageIndex tree node.

    ``tree_json`` may be a path, a raw parsed dict, or ``None`` (defaults to
    ``<paper_dir>/tree.json``). Each node becomes a Document with
    ``text = "<title>\\n<body>"`` where the body is loaded from ``raw.md`` by
    line range, and metadata::

        {paper_id, node_id, title, line_start, line_end, tree_layer: "pageindex"}

    When ``max_node_tokens`` is given (e.g. 8000), nodes above the cap are
    split into paragraph chunks via :func:`_chunk_document` — each chunk keeps
    the parent metadata plus ``chunk_index``/``chunk_count`` and an id with a
    ``#i`` suffix (T9: bounds single-sequence embedding size on GPU). Without
    the parameter the node↔Document mapping is 1:1 (backward compatible).

    Body resolution order (mirrors ``services.embedding._collect_tree_nodes``
    semantics, but also handles the actual tree.json format which carries
    ``line_num`` + inline ``text``):

    1. explicit ``line_start``/``line_end`` → ``raw.md[line_start:line_end]``
    2. ``line_num`` (1-based header line) → flat range up to the next node's
       header line in ``raw.md`` (same computation the PageIndex builder used)
    3. inline node ``text`` → used verbatim (when ``raw.md`` is missing)

    ``raw.md`` is only read when at least one node needs line-based extraction
    (body loaded on demand). Documents whose text is empty are dropped.
    """
    if not _LLAMA_INDEX_AVAILABLE:  # pragma: no cover - envs without llama-index
        raise RuntimeError("llama-index is not installed; cannot collect Documents")

    paper_dir = Path(paper_dir)
    paper_id = paper_dir.name

    if tree_json is None or isinstance(tree_json, (str, Path)):
        tree_path = Path(tree_json) if tree_json else paper_dir / "tree.json"
        if not tree_path.exists():
            return []
        try:
            tree = json.loads(tree_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("[rag] cannot parse tree.json at %s: %s", tree_path, exc)
            return []
    elif isinstance(tree_json, dict):
        tree = tree_json
    else:  # pragma: no cover - defensive
        raise TypeError("tree_json must be a path or parsed dict")

    # Flatten the hierarchy in document order (pre-order) so sibling/aunt
    # headers bound each node's raw.md line range, exactly like the builder.
    flat: list[dict[str, Any]] = []

    def _flatten(nodes: list[dict]) -> None:
        for node in nodes:
            flat.append(node)
            children = node.get("nodes")
            if isinstance(children, list) and children:
                _flatten(children)

    structure = tree.get("structure")
    if isinstance(structure, list):
        _flatten(structure)

    if not flat:
        return []

    # Load raw.md on demand: only needed when some node wants line extraction.
    raw_lines: list[str] | None = None
    if any(
        node.get("line_num") is not None
        or (node.get("line_start") is not None and node.get("line_end") is not None)
        for node in flat
    ):
        raw_path = paper_dir / "raw.md"
        if raw_path.exists():
            raw_lines = raw_path.read_text(encoding="utf-8").split("\n")

    docs: list[Document] = []
    for i, node in enumerate(flat):
        nid = str(node.get("node_id") or "").strip()
        title = str(node.get("title") or "").strip()
        if not nid:
            continue

        body = ""
        line_start: int | None = None
        line_end: int | None = None

        ls, le = node.get("line_start"), node.get("line_end")
        if ls is not None and le is not None and raw_lines is not None:
            line_start, line_end = int(ls), int(le)
            body = "\n".join(raw_lines[line_start:line_end])
        elif node.get("line_num") is not None and raw_lines is not None:
            start0 = int(node["line_num"]) - 1  # 1-based header line → 0-based
            end0 = len(raw_lines)
            nxt = flat[i + 1] if i + 1 < len(flat) else None
            if nxt is not None and nxt.get("line_num") is not None:
                end0 = int(nxt["line_num"]) - 1
            line_start, line_end = start0, max(start0, end0)
            body = "\n".join(raw_lines[start0:end0])
        elif node.get("text"):
            body = str(node["text"])

        text = f"{title}\n{body}".strip()
        if not text:
            continue

        docs.append(
            Document(
                text=text,
                id_=_node_key(paper_id, nid),
                metadata={
                    "paper_id": paper_id,
                    "node_id": nid,
                    "title": title,
                    "line_start": line_start,
                    "line_end": line_end,
                    "tree_layer": TREE_LAYER_PAGEINDEX,
                },
            )
        )
    if max_node_tokens and max_node_tokens > 0:
        out: list[Document] = []
        for doc in docs:
            out.extend(_chunk_document(doc, int(max_node_tokens)))
        return out
    return docs


# ── Persistence helpers ──────────────────────────────────────────────────────


def _storage_dirs(storage_dir: str | Path) -> tuple[Path, Path, Path]:
    """Return (root, vector_dir, bm25_dir) for a configured storage_dir."""
    root = Path(storage_dir)
    return root, root / "vector", root / "bm25"


def _pointer_path(storage_root: Path) -> Path:
    return storage_root / ACTIVE_POINTER_NAME


def _active_storage_root(storage_dir: str | Path) -> tuple[Path, str | None] | None:
    """Return the physical active store and its generation, if one is active.

    A storage directory without ``active.json`` is a pre-generation legacy
    index and remains readable. A malformed or dangling pointer is never
    silently redirected to that legacy index: serving stale data is safer than
    combining generations, but serving an explicitly invalid active pointer is
    not safe at all, so callers receive ``None`` and can surface health failure.
    """
    storage_root = Path(storage_dir)
    pointer = _pointer_path(storage_root)
    if not pointer.exists():
        return storage_root, None
    try:
        raw = json.loads(pointer.read_text(encoding="utf-8"))
        generation = raw.get("generation") if isinstance(raw, dict) else None
        if not isinstance(generation, str) or not generation.strip():
            return None
        generation = generation.strip()
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    active_root = storage_root / GENERATIONS_DIR_NAME / generation
    if not active_root.is_dir():
        return None
    return active_root, generation


def get_active_index_generation(cfg: Config) -> str | None:
    """Return the active generation id, or ``None`` for legacy/no active index.

    This is additive operational metadata. ``load_index`` keeps its historic
    two-item return value so callers do not need to migrate.
    """
    active = _active_storage_root(get_llamaindex_config(cfg).storage_dir)
    return active[1] if active is not None else None


def capture_index_generation(cfg: Config) -> str | None:
    """Capture the snapshot a long-running caller must keep using.

    ``get_active_index_generation`` keeps its historic ``None`` return for both
    a legacy flat index and an invalid pointer. Durable callers need to
    distinguish them: only a missing pointer is the explicit ``"legacy"``
    snapshot; a malformed or dangling pointer remains unavailable/fail-closed.
    """
    active = _active_storage_root(get_llamaindex_config(cfg).storage_dir)
    if getattr(get_llamaindex_config(cfg), "rag_engine", "llamaindex") == "sql":
        # SQL-native engine: the evidence anchor is the content fingerprint of
        # the working copy (node_texts + tree_vectors in drbrain_rag.db).
        from drbrain.rag.sql_retrie import _default_rag_db, _generation_id

        rag_db = _default_rag_db(cfg)
        if not rag_db.exists():
            return None
        import sqlite3 as _sqlite3

        _conn = _sqlite3.connect(f"file:{rag_db}?mode=ro", uri=True)
        try:
            return _generation_id(_conn)
        finally:
            _conn.close()
    if active is None:
        return None
    return active[1] or LEGACY_INDEX_GENERATION


def _storage_root_for_generation(storage_dir: str | Path, generation: str | None) -> Path | None:
    """Resolve an optional immutable snapshot without following a newer pointer."""
    root = Path(storage_dir)
    if generation is None:
        active = _active_storage_root(root)
        return active[0] if active is not None else None
    if generation == LEGACY_INDEX_GENERATION:
        return root
    if not generation or Path(generation).name != generation:
        return None
    candidate = root / GENERATIONS_DIR_NAME / generation
    return candidate if candidate.is_dir() else None


def _new_generation_id() -> str:
    return f"g-{time.time_ns()}-{uuid.uuid4().hex[:8]}"


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a small JSON control file without torn readers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except OSError:
        # ``Path.write_text`` historically created these public index metadata
        # files as shared-readable on the default deployment filesystem.
        mode = 0o644
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            fd = -1  # ownership moved to the context manager
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, path)
    except Exception:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _load_manifest(storage_dir: str | Path) -> dict[str, Any]:
    active = _active_storage_root(storage_dir)
    if active is None:
        logger.warning("[rag] active generation pointer is invalid at %s", storage_dir)
        return {}
    root, _ = active
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("[rag] ignoring unreadable manifest at %s", manifest_path)
        return {}


def _write_manifest(storage_dir: str | Path, manifest: dict[str, Any]) -> None:
    _write_json_atomically(Path(storage_dir) / MANIFEST_NAME, manifest)


def _generation_references(storage_root: Path) -> dict[str, str] | None:
    """Read durable run references, returning ``None`` for unsafe control data."""
    references: dict[str, str] = {}
    # Read the first PR's single-file shape as a compatibility input, but write
    # only isolated run files below so concurrent run creation cannot lose refs.
    legacy_path = storage_root / GENERATION_REFERENCES_NAME
    if legacy_path.exists():
        try:
            legacy_payload = json.loads(legacy_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.error("[rag] generation references are unreadable at %s", legacy_path)
            return None
        raw = legacy_payload.get("references", {}) if isinstance(legacy_payload, dict) else {}
        if not isinstance(raw, dict):
            logger.error("[rag] generation references are malformed at %s", legacy_path)
            return None
        references.update(
            {
                str(run_id): str(generation)
                for run_id, generation in raw.items()
                if str(run_id).strip() and str(generation).strip()
            }
        )

    references_dir = storage_root / GENERATION_REFERENCES_DIR_NAME
    if not references_dir.exists():
        return references
    try:
        paths = sorted(references_dir.glob("*.json"))
    except OSError:
        logger.error("[rag] generation reference directory is unreadable at %s", references_dir)
        return None
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.error("[rag] generation reference is unreadable at %s", path)
            return None
        if not isinstance(payload, dict):
            logger.error("[rag] generation reference is malformed at %s", path)
            return None
        run_id = str(payload.get("run_id") or "").strip()
        generation = str(payload.get("generation") or "").strip()
        if not run_id or not generation:
            logger.error("[rag] generation reference is malformed at %s", path)
            return None
        references[run_id] = generation
    return references


def _referenced_generations(storage_root: Path) -> set[str]:
    """Return immutable snapshots still required by durable autoresearch runs."""
    references = _generation_references(storage_root)
    if references is not None:
        return set(references.values())
    # A damaged retention ledger must not become permission to delete audit
    # evidence. Keep all completed snapshots until an operator repairs it.
    generations_root = storage_root / GENERATIONS_DIR_NAME
    try:
        return {
            child.name
            for child in generations_root.iterdir()
            if child.is_dir() and child.name.startswith("g-")
        }
    except OSError:
        return set()


def retain_index_generation(cfg: Config, generation: str | None, run_id: str) -> bool:
    """Durably retain a run's index snapshot without introducing another service."""
    resolved_generation = str(generation or "").strip()
    if not resolved_generation or resolved_generation == LEGACY_INDEX_GENERATION:
        return False
    if getattr(get_llamaindex_config(cfg), "rag_engine", "llamaindex") == "sql":
        # SQL-native engine: the generation is a fingerprint of the working
        # copy, not a prunable filesystem generation — nothing to retain.
        return True
    storage_root = Path(get_llamaindex_config(cfg).storage_dir)
    generation_root = storage_root / GENERATIONS_DIR_NAME / resolved_generation
    if not generation_root.is_dir():
        raise RuntimeError(f"cannot retain unavailable index generation {resolved_generation!r}")
    if _generation_references(storage_root) is None:
        raise RuntimeError("cannot update unreadable generation references")
    reference_name = hashlib.sha256(str(run_id).encode("utf-8")).hexdigest() + ".json"
    _write_json_atomically(
        storage_root / GENERATION_REFERENCES_DIR_NAME / reference_name,
        {"run_id": str(run_id), "generation": resolved_generation},
    )
    return True


def _prune_inactive_generations(
    storage_root: Path,
    active_generation: str,
    *,
    retain_count: int = GENERATION_RETAIN_COUNT,
    grace_seconds: float = GENERATION_PRUNE_GRACE_SECONDS,
    protected_generations: Iterable[str] = (),
) -> list[str]:
    """Bound completed snapshots while retaining a reader/rollback window.

    At least ``retain_count`` completed generations, including the active one,
    are kept. Older snapshots are only removed after a grace period, allowing
    in-flight readers that resolved a previous pointer to finish loading.
    Staging directories are never considered completed snapshots.
    """
    generations_root = storage_root / GENERATIONS_DIR_NAME
    if retain_count < 1 or not generations_root.is_dir():
        return []

    entries = list(generations_root.iterdir())
    completed = [child for child in entries if child.is_dir() and child.name.startswith("g-")]
    completed.sort(key=lambda child: child.stat().st_mtime, reverse=True)
    protected = {active_generation}
    protected.update(str(generation) for generation in protected_generations if str(generation))
    for child in completed:
        if len(protected) >= retain_count:
            break
        protected.add(child.name)

    cutoff = time.time() - max(0.0, grace_seconds)
    pruned: list[str] = []
    for child in completed:
        if child.name in protected or child.stat().st_mtime > cutoff:
            continue
        try:
            shutil.rmtree(child)
        except OSError as exc:
            logger.warning("[rag] could not prune stale generation %s: %s", child, exc)
        else:
            pruned.append(child.name)
    for child in entries:
        if not child.is_dir() or not child.name.startswith(".staging-g-"):
            continue
        if child.stat().st_mtime > cutoff:
            continue
        try:
            shutil.rmtree(child)
        except OSError as exc:
            logger.warning("[rag] could not prune stale staging directory %s: %s", child, exc)
        else:
            pruned.append(child.name)
    return pruned


def _load_old_embeddings(vector_dir: Path) -> dict[str, list[float]]:
    """Load node_id → embedding from a previously persisted SimpleVectorStore."""
    if not _LLAMA_INDEX_AVAILABLE or not (vector_dir / "default__vector_store.json").exists():
        return {}
    try:
        store = SimpleVectorStore.from_persist_dir(str(vector_dir))
        return dict(store._data.embedding_dict)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("[rag] could not read previous vector store: %s", exc)
        return {}


def _load_old_index_nodes(vector_dir: Path, embed_model: Any) -> dict[str, TextNode]:
    """Load all previously indexed nodes (with embeddings) from a persisted index.

    Used to carry non-target papers through a ``--paper`` subset rebuild so the
    persisted index keeps covering the whole library.
    """
    if not _LLAMA_INDEX_AVAILABLE or not (vector_dir / "docstore.json").exists():
        return {}
    try:
        sc = StorageContext.from_defaults(persist_dir=str(vector_dir))
        idx = load_index_from_storage(sc, embed_model=embed_model)
        out: dict[str, TextNode] = {}
        for node in idx.docstore.docs.values():
            if not isinstance(node, TextNode):
                continue
            # ``.get(node_id)`` is the duck-typed vector-store lookup the
            # concrete store (e.g. SimpleVectorStore) implements; the base
            # ``BasePydanticVectorStore`` type does not declare it.
            embedding = node.embedding or idx._vector_store.get(  # type: ignore[attr-defined]
                node.node_id
            )
            if embedding is None:
                continue
            node.embedding = embedding
            out[node.node_id] = node
        return out
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("[rag] could not load previous index nodes: %s", exc)
        return {}


def _default_embed_model(cfg: Config) -> Any:
    """T1 DrbrainEmbedding adapter (lazy; loads the model on first embed call)."""
    from drbrain.rag.llm import DrbrainEmbedding

    return DrbrainEmbedding(cfg)


def _resolve_paper_dir(papers_root: Path, pid: str) -> Path | None:
    """Locate a paper's asset directory across on-disk layouts.

    Corpus ingests use different layouts: newer papers live in flat
    ``<papers>/<local_id>/`` dirs (e.g. ``p00005bb6``), while DOI-keyed
    papers are nested one level (``<papers>/<prefix>/<suffix>``) or use an
    underscore-sanitized flat name.  Returns the first existing layout.
    """
    candidates = [papers_root / pid]
    if "/" in pid:
        prefix, _, suffix = pid.partition("/")
        candidates.append(papers_root / prefix / suffix.replace("/", "_"))
        candidates.append(papers_root / pid.replace("/", "_"))
    for cand in candidates:
        if cand.is_dir():
            return cand
    return None


# ── Build ────────────────────────────────────────────────────────────────────


def _embed_devices(cfg: Config) -> list[int]:
    """GPU ids for parallel chunk embedding: configured cuda:N plus extra_gpus."""
    devices: list[int] = []
    dev = str(getattr(cfg.embed, "device", "") or "")
    if dev.startswith("cuda:"):
        try:
            devices.append(int(dev.split(":", 1)[1]))
        except ValueError:
            pass
    for g in getattr(cfg.embed, "extra_gpus", None) or []:
        if int(g) not in devices:
            devices.append(int(g))
    return devices


def _embed_model_path(cfg: Config) -> str:
    from drbrain.services.embedding import _resolve_model_path

    path = _resolve_model_path(
        cfg.embed.model, os.path.expanduser(cfg.embed.cache_dir), cfg.embed.source
    )
    if not path:
        raise RuntimeError("cannot resolve local model path for parallel embedding")
    return str(path)


_WORKER_MODEL: Any = None


def _embed_chunk_worker(args: tuple[Any, int, list[str], int, int]) -> Any:
    """Spawn-pool worker: pin one GPU, embed a chunk, return vectors."""
    model_path, gpu, texts, ci, total = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    global _WORKER_MODEL
    if _WORKER_MODEL is None:
        from sentence_transformers import SentenceTransformer

        _WORKER_MODEL = SentenceTransformer(model_path, device="cuda")
        # Align with scripts/serve_embedding.py: truncate to 512 tokens so
        # these vectors match the corpus-pipeline embedding semantics.
        try:
            _WORKER_MODEL.max_seq_length = 512
        except Exception:  # noqa: BLE001 - non-fatal optimization
            pass
    logger.info("[rag] embedding chunk {}/{} ({} nodes) on GPU {}", ci, total, len(texts), gpu)
    return _WORKER_MODEL.encode(texts, normalize_embeddings=True, batch_size=64)


def build_index(
    cfg: Config,
    db: Any,
    paper_ids: Iterable[str] | None = None,
    force: bool = False,
    embed_model: Any | None = None,
    max_node_tokens: int | None = None,
) -> dict[str, Any]:
    """Build (or incrementally update) the LlamaIndex vector + BM25 indexes.

    For every target paper (``paper_ids``, or all papers known to ``db`` when
    ``None``), collects its PageIndex Documents, embeds only the nodes whose
    ``content_hash`` changed since the last build (``force=True`` re-embeds
    everything), persists ``VectorStoreIndex`` + ``BM25Retriever`` under
    ``llamaindex.storage_dir``, and records hashes in ``manifest.json``.

    ``max_node_tokens`` (default ``llamaindex.max_node_tokens``) caps
    the size of a single embedded sequence: nodes above the cap are split into
    paragraph chunks (each chunk = one indexed TextNode carrying the parent's
    metadata + ``#i`` suffix). Change detection stays per parent node — an
    unchanged node reuses all its chunk embeddings; only changed nodes (or
    nodes with missing chunk embeddings) are re-embedded. Non-positive values
    retain the safe default cap.

    ``db`` only needs ``get_all_papers() -> list[{"local_id": str}]``. Nodes
    whose embedding cannot be reused (no persisted store, or the configured
    embed model changed) are re-embedded automatically.

    Returns historic stats plus additive ``generation`` and
    ``previous_generation`` after a successful publish.
    """
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
        return pid, collect_tree_nodes(paper_dir)

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
    model_changed = bool(old_model) and old_model != new_model

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
            logger.info("[rag] parallel embedding on GPUs {}", _devices)
            with _ctx.Pool(len(_devices)) as _pool:
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
