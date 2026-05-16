"""Tree node embedding service (ScholarAIO pattern).

Lightweight vectors for semantically-complete tree nodes.
Provider=none disables all vectors; falls back to BM25 + LLM navigation.

Usage:
    from drbrain.services.embedding import build_tree_vectors, search_tree
    build_tree_vectors(db_path, paper_dir, cfg)
    results = search_tree("turbulent drag reduction", db_path, top_k=5)
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import struct
import time
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from drbrain.config import EmbedConfig

# ── Module-level caches ──────────────────────────────────────────────────────

_model_cache: dict = {}  # key: (model_name, cache_dir, device) -> SentenceTransformer
_GPU_PROFILE_FILE = Path("~/.cache/drbrain/gpu_profile.json").expanduser()


def _profile_cache_key(model_name: str, gpu_name: str) -> str:
    """Cache key combining model name and GPU model for reuse across sessions."""
    return f"{model_name}::{gpu_name}"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _embed_provider(cfg: EmbedConfig | None = None) -> str:
    """Return normalized embedding backend provider."""
    if cfg is None:
        return "local"
    provider = (cfg.provider or "local").strip().lower()
    return provider or "local"


def _content_hash(text: str) -> str:
    """Stable content hash for incremental update detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    import numpy as np

    va = np.asarray(a, dtype="float32")
    vb = np.asarray(b, dtype="float32")
    dot = float(np.dot(va, vb))
    return dot  # assumes normalized input


# ── Model resolution ─────────────────────────────────────────────────────────


def _looks_like_sentence_transformer_dir(path: Path) -> bool:
    """Heuristic for a locally cached sentence-transformers model directory."""
    return (
        path.is_dir()
        and (path / "modules.json").exists()
        and (path / "config_sentence_transformers.json").exists()
        and ((path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists())
    )


def _find_local_model_path(model_name: str, cache_dir: str) -> str | None:
    """Return a directly discoverable cached ModelScope model path, if present."""
    parts = model_name.split("/", 1)
    if len(parts) != 2:
        return None

    org, repo = parts
    root = Path(cache_dir).expanduser()
    org_dir = root / org
    if not org_dir.exists():
        return None

    repo_variants = [repo, repo.replace(".", "_"), repo.replace(".", "___")]
    for variant in repo_variants:
        candidate = org_dir / variant
        if _looks_like_sentence_transformer_dir(candidate):
            return str(candidate)

    for candidate in org_dir.iterdir():
        if (
            candidate.is_dir()
            and candidate.name.startswith(repo.split(".", 1)[0])
            and _looks_like_sentence_transformer_dir(candidate)
        ):
            return str(candidate)

    return None


def _resolve_model_path(model_name: str, cache_dir: str, source: str) -> str | None:
    """Find local model path or download via ModelScope.

    Args:
        model_name: Model ID (e.g. ``"Qwen/Qwen3-Embedding-0.6B"``).
        cache_dir: Local cache directory.
        source: ``"modelscope"`` or ``"huggingface"``.

    Returns:
        Local folder path if found or downloaded, ``None`` to fall back
        to HuggingFace (SentenceTransformer handles download internally).
    """
    if source != "modelscope":
        return None

    local_path = _find_local_model_path(model_name, cache_dir)
    if local_path:
        return local_path

    try:
        from modelscope import snapshot_download  # noqa: PLC0415
    except ImportError:
        return None
    import logging

    logging.getLogger("modelscope").setLevel(logging.ERROR)

    # Check if already cached locally
    try:
        local_path = snapshot_download(model_name, cache_dir=cache_dir, local_files_only=True)
        return local_path
    except Exception:
        logger.debug("model not cached locally: %s", model_name)

    # Download from ModelScope
    try:
        logger.info("[embed] downloading model %s from ModelScope", model_name)
        return snapshot_download(model_name, cache_dir=cache_dir)
    except Exception:
        logger.warning(
            "[embed] ModelScope download failed for %s, falling back to HuggingFace", model_name
        )
    return None


# ── Model loading ────────────────────────────────────────────────────────────


def _load_model(cfg: EmbedConfig | None = None):
    """Load SentenceTransformer with module-level cache.

    Resolves model path via ModelScope first, falls back to HuggingFace.
    Cached by (model_name, cache_dir, device) to avoid reloading.
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if cfg is not None:
        model_name = cfg.model
        cache_dir = os.path.expanduser(cfg.cache_dir)
        device_cfg = cfg.device
        source = cfg.source
    else:
        model_name = "Qwen/Qwen3-Embedding-0.6B"
        cache_dir = os.path.expanduser("~/.cache/modelscope/hub/models")
        device_cfg = "auto"
        source = "modelscope"

    if source == "modelscope":
        os.environ["MODELSCOPE_CACHE"] = cache_dir

    SentenceTransformer = importlib.import_module("sentence_transformers").SentenceTransformer  # noqa: N806

    if device_cfg == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = device_cfg

    cache_key = (model_name, cache_dir, device)
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    # Try to find or download the model
    local_path = _resolve_model_path(model_name, cache_dir, source)
    if local_path:
        model = SentenceTransformer(local_path, device=device)
    else:
        model = SentenceTransformer(model_name, device=device)

    _model_cache[cache_key] = model
    return model


# ── GPU profiling & adaptive batching ────────────────────────────────────────


def _run_profile(model, cfg: EmbedConfig | None = None) -> dict:
    """Profile GPU memory per sample at various sequence lengths.

    Generates dummy texts at several token counts, encodes one at a time,
    and records peak GPU memory. Results are cached to disk so this only
    runs once per model + GPU combination.

    Returns:
        ``{"gpu_total_bytes": int, "per_sample": {token_len: bytes, ...},
           "model_name": str, "gpu_name": str, "profiled_at": str,
           "baseline_bytes": int}`` or empty dict if CPU-only.
    """
    try:
        import torch
    except ImportError:
        return {}

    if not torch.cuda.is_available():
        return {}

    device = next(
        model.parameters() if hasattr(model, "parameters") else model[0].parameters()
    ).device
    if device.type != "cuda":
        return {}

    gpu_props = torch.cuda.get_device_properties(device)
    gpu_name = gpu_props.name
    gpu_total = gpu_props.total_memory

    tokenizer = model.tokenizer

    per_sample: dict[int, int] = {}
    filler = "turbulence flow particle dynamics simulation "

    # Measure baseline: model weights already on GPU
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    tiny = filler[:20]
    model.encode([tiny], normalize_embeddings=True, batch_size=1)
    baseline = torch.cuda.memory_allocated(device)

    model_name = cfg.model if cfg is not None else "Qwen/Qwen3-Embedding-0.6B"

    logger.info(
        "[gpu-profile] Profiling GPU memory for %s on %s (baseline=%.0f MB, total=%.0f MB) ...",
        model_name,
        gpu_name,
        baseline / 1024**2,
        gpu_total / 1024**2,
    )

    # Probe from 64 tokens, doubling each time, until OOM
    tgt_tokens = 64
    max_tokens = getattr(model, "max_seq_length", 32768) or 32768
    while tgt_tokens <= max_tokens:
        raw = filler * (tgt_tokens // 4 + 10)
        ids = tokenizer.encode(raw)[:tgt_tokens]
        text = tokenizer.decode(ids, skip_special_tokens=True)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        try:
            model.encode([text], normalize_embeddings=True, batch_size=1)
            peak = torch.cuda.max_memory_allocated(device)
            incremental = peak - baseline
            per_sample[tgt_tokens] = incremental
            logger.info(
                "[gpu-profile]   tokens=%5d  incremental=%6.0f MB  (peak=%.0f MB)",
                tgt_tokens,
                incremental / 1024**2,
                peak / 1024**2,
            )
        except torch.cuda.OutOfMemoryError:
            logger.info(
                "[gpu-profile]   tokens=%5d  OOM -- max single-sample capacity found", tgt_tokens
            )
            torch.cuda.empty_cache()
            break

        tgt_tokens *= 2

    return {
        "gpu_total_bytes": gpu_total,
        "baseline_bytes": baseline,
        "gpu_name": gpu_name,
        "model_name": model_name,
        "per_sample": {str(k): v for k, v in per_sample.items()},
        "profiled_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _load_or_create_profile(model, cfg: EmbedConfig | None = None) -> dict:
    """Load cached GPU profile or run profiling."""
    try:
        import torch
    except ImportError:
        return {}

    if not torch.cuda.is_available():
        return {}

    device = next(
        model.parameters() if hasattr(model, "parameters") else model[0].parameters()
    ).device
    if device.type != "cuda":
        return {}

    gpu_name = torch.cuda.get_device_properties(device).name
    model_name = cfg.model if cfg is not None else "Qwen/Qwen3-Embedding-0.6B"
    cache_key = _profile_cache_key(model_name, gpu_name)

    # Try loading from disk
    if _GPU_PROFILE_FILE.exists():
        try:
            all_profiles = json.loads(_GPU_PROFILE_FILE.read_text("utf-8"))
            if cache_key in all_profiles:
                logger.debug("[gpu-profile] loaded cached profile for %s", cache_key)
                return all_profiles[cache_key]
        except Exception:
            pass

    # Run profiling
    profile = _run_profile(model, cfg)
    if not profile:
        return {}

    # Save to disk
    _GPU_PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
    all_profiles = {}
    if _GPU_PROFILE_FILE.exists():
        try:
            all_profiles = json.loads(_GPU_PROFILE_FILE.read_text("utf-8"))
        except Exception:
            pass
    all_profiles[cache_key] = profile
    _GPU_PROFILE_FILE.write_text(
        json.dumps(all_profiles, indent=2, ensure_ascii=False) + "\n", "utf-8"
    )
    logger.info("[gpu-profile] saved profile to %s", _GPU_PROFILE_FILE)
    return profile


def _estimate_mem_per_sample(est_tokens: int, profile: dict) -> int:
    """Interpolate/extrapolate memory per sample from profile data.

    For sequence lengths beyond the profiled range, extrapolates using
    quadratic scaling (attention is O(n^2)).
    """
    per_sample = profile.get("per_sample", {})
    if not per_sample:
        return 0

    # Convert keys to int, sort
    points = sorted((int(k), v) for k, v in per_sample.items())

    if est_tokens <= points[0][0]:
        return points[0][1]

    # Linear interpolation within profiled range
    for i in range(len(points) - 1):
        t0, m0 = points[i]
        t1, m1 = points[i + 1]
        if t0 <= est_tokens <= t1:
            frac = (est_tokens - t0) / (t1 - t0)
            return int(m0 + frac * (m1 - m0))

    # Extrapolate beyond max profiled point with quadratic scaling
    t_max, m_max = points[-1]
    ratio = est_tokens / t_max
    return int(m_max * ratio * ratio)


def _compute_batch_size(est_tokens: int, profile: dict, safety_factor: float = 0.85) -> int:
    """Compute optimal batch_size for texts of a given token length.

    Uses incremental memory per sample (peak minus baseline) from the
    profile, so model weight memory is excluded from the calculation.
    """
    if not profile or not profile.get("per_sample"):
        return 8  # conservative default

    gpu_total = profile["gpu_total_bytes"]
    baseline = profile.get("baseline_bytes", 0)
    mem_per_sample = _estimate_mem_per_sample(est_tokens, profile)

    if mem_per_sample <= 0:
        return 8

    # Available = total GPU memory * safety - baseline (model weights etc.)
    available = gpu_total * safety_factor - baseline
    if available <= 0:
        return 1

    bs = int(available / mem_per_sample)
    return max(1, min(bs, 128))


# ── Post-filter ──────────────────────────────────────────────────────────────


def _post_filter(
    results: list[dict],
    min_score: float = 0.0,
    require_text: bool = True,
) -> list[dict]:
    """Filter vector search results by quality thresholds.

    Args:
        results: List of result dicts with at least ``score`` and ``node_id`` keys.
        min_score: Minimum cosine similarity score (default 0.0).
        require_text: If True, remove results with empty/None ``node_id``.

    Returns:
        Filtered result list.
    """
    filtered = [r for r in results if r.get("score", 0.0) >= min_score]
    if require_text:
        filtered = [r for r in filtered if r.get("node_id")]
    return filtered


# ── OpenAI-compatible embedding ───────────────────────────────────────────────


def _embed_batch_openai_compat(texts: list[str], cfg: EmbedConfig) -> list[list[float]]:
    """Embed texts via OpenAI-compatible /v1/embeddings API.

    Splits large batches into chunks of cfg.batch_size. Uses exponential
    backoff retry on transient errors.
    """
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    if not cfg.api_base:
        raise ValueError("embed.api_base is required for openai-compat provider")
    if not cfg.api_key:
        raise ValueError("embed.api_key is required for openai-compat provider")

    endpoint = cfg.api_base.rstrip("/") + "/embeddings"
    model = cfg.model or "text-embedding-3-small"
    chunk_size = max(1, cfg.batch_size or 64)

    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))

    all_vectors: list[list[float]] = []

    for i in range(0, len(texts), chunk_size):
        chunk = texts[i : i + chunk_size]
        body = {"model": model, "input": chunk}

        try:
            resp = session.post(
                endpoint,
                json=body,
                headers={"Authorization": f"Bearer {cfg.api_key}"},
                timeout=60,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.warning("[embed] openai-compat request failed: %s", e)
            if i == 0:
                raise
            break

        data = resp.json().get("data", [])
        # Sort by index to preserve input order
        data.sort(key=lambda d: d.get("index", 0))
        for item in data:
            emb = item.get("embedding", [])
            if emb:
                all_vectors.append(emb)

    return all_vectors


# ── Embedding dispatch ──────────────────────────────────────────────────────


def _embed_batch(texts: list[str], cfg: EmbedConfig | None = None) -> list[list[float]]:
    """Embed a batch of texts via the configured provider.

    Returns list of float vectors. Returns empty list when provider is ``"none"``.
    """
    provider = _embed_provider(cfg)
    if provider == "none":
        return []
    if provider == "openai-compat":
        if cfg is None:
            return []
        return _embed_batch_openai_compat(texts, cfg)
    return _embed_batch_local(texts, cfg)


def _embed_batch_local(texts: list[str], cfg: EmbedConfig | None = None) -> list[list[float]]:
    """Embed texts using local sentence-transformers model.

    When running on CUDA, adaptively tunes batch_size based on a one-time
    GPU memory profile. Falls back to config batch_size on CPU or when
    profiling is unavailable.
    """
    model = _load_model(cfg)

    # Determine batch_size
    bs = cfg.batch_size if cfg is not None else 64
    device_str = str(getattr(model, "device", "cpu"))

    if "cuda" in device_str:
        try:
            profile = _load_or_create_profile(model, cfg)
            if profile:
                # Estimate max tokens in batch for conservative sizing
                tokenizer = model.tokenizer
                max_tokens = max(len(tokenizer.encode(t)) for t in texts) if texts else 128
                adaptive_bs = _compute_batch_size(max_tokens, profile)
                bs = adaptive_bs
                logger.debug("[embed] adaptive batch_size=%d for max_tokens=%d", bs, max_tokens)
        except Exception:
            logger.debug("[embed] GPU adaptive batching unavailable, using fixed batch_size=%d", bs)

    embeddings = model.encode(texts, normalize_embeddings=True, batch_size=bs)
    return [e.tolist() for e in embeddings]


# ── Build ────────────────────────────────────────────────────────────────────


def _collect_tree_nodes(paper_dir: Path) -> list[dict]:
    """Collect all tree nodes from a paper's tree.json, with their text content."""
    tree_path = paper_dir / "tree.json"
    if not tree_path.exists():
        return []

    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    structure = tree.get("structure", [])

    raw_md_path = paper_dir / "raw.md"
    if raw_md_path.exists():
        raw_text = raw_md_path.read_text(encoding="utf-8")
    else:
        raw_text = ""

    def _walk(nodes: list[dict], path_prefix: str = "") -> list[dict]:
        result = []
        for node in nodes:
            nid = node.get("node_id", "")
            title = node.get("title", "")
            text = f"{title}\n"

            # Try to extract content from raw.md using line ranges
            line_start = node.get("line_start")
            line_end = node.get("line_end")
            if line_start is not None and line_end is not None and raw_text:
                text += "\n".join(raw_text.split("\n")[line_start:line_end])
            elif node.get("content"):
                text += str(node.get("content", ""))

            result.append(
                {
                    "node_id": nid,
                    "title": title,
                    "text": text.strip(),
                }
            )

            child_nodes = node.get("nodes", [])
            if child_nodes:
                result.extend(_walk(child_nodes, f"{path_prefix}{nid}/"))
        return result

    return _walk(structure)


def build_tree_vectors(
    db_path: Path,
    paper_dir: Path,
    cfg: EmbedConfig | None = None,
) -> int:
    """Embed all tree nodes for a paper and store in tree_vectors.

    Args:
        db_path: SQLite database path.
        paper_dir: Paper directory containing tree.json and raw.md.
        cfg: Optional EmbedConfig.

    Returns:
        Number of vectors written.
    """
    import sqlite3

    provider = _embed_provider(cfg)
    if provider == "none":
        logger.info("embed.provider=none; tree vector generation is disabled")
        return 0

    nodes = _collect_tree_nodes(paper_dir)
    if not nodes:
        return 0

    # Check existing hashes for incremental update
    conn = sqlite3.connect(str(db_path))
    try:
        existing_hashes: dict[str, str] = {}
        for row in conn.execute("SELECT node_id, content_hash FROM tree_vectors").fetchall():
            existing_hashes[row[0]] = row[1]

        paper_id = paper_dir.name

        # Collect nodes that need embedding
        to_embed_texts: list[str] = []
        to_embed_nodes: list[dict] = []
        for node in nodes:
            nhash = _content_hash(node["text"])
            if existing_hashes.get(node["node_id"]) == nhash:
                continue  # unchanged, skip
            to_embed_texts.append(node["text"])
            to_embed_nodes.append(node)
            to_embed_nodes[-1]["_hash"] = nhash

        if not to_embed_texts:
            return 0

        # Embed in batch
        vectors = _embed_batch(to_embed_texts, cfg)
        if not vectors:
            return 0

        # Store
        for node, vec in zip(to_embed_nodes, vectors):
            blob = struct.pack(f"{len(vec)}f", *vec)
            conn.execute(
                "INSERT OR REPLACE INTO tree_vectors "
                "(node_id, paper_id, embedding, content_hash, tree_layer) "
                "VALUES (?, ?, ?, ?, ?)",
                (node["node_id"], paper_id, blob, node["_hash"], "pageindex"),
            )

        conn.commit()
        return len(to_embed_nodes)

    finally:
        conn.close()


async def build_paper_tree_vectors(
    paper_dir: Path,
    db_path: Path,
    embed_cfg: EmbedConfig | None = None,
    llm_models: list[dict] | None = None,
) -> int:
    """Build PageIndex tree vectors + RAPTOR recursive summaries for a single paper.

    Combines both layers:
    1. build_tree_vectors — embed PageIndex leaf nodes
    2. build_raptor_tree — recursive GMM clustering + LLM summarization

    Args:
        paper_dir: Paper directory with tree.json and raw.md.
        db_path: SQLite database path.
        embed_cfg: Embedding configuration.
        llm_models: LLM model list for RAPTOR summarization.

    Returns:
        Total number of vectors + summaries created.
    """
    from drbrain.extractor.raptor import build_raptor_tree

    pageindex_count = build_tree_vectors(db_path, paper_dir, embed_cfg)

    raptor_count = 0
    if llm_models:
        try:
            raptor_count = await build_raptor_tree(paper_dir, db_path, embed_cfg, llm_models)
        except Exception:
            logger.warning(
                "RAPTOR tree build failed for %s, PageIndex vectors still created",
                paper_dir.name,
            )

    return pageindex_count + raptor_count


# ── Search ───────────────────────────────────────────────────────────────────


def search_tree(
    query: str,
    db_path: Path,
    top_k: int = 10,
    cfg: EmbedConfig | None = None,
) -> list[dict]:
    """Vector search over tree_vectors using cosine similarity.

    Args:
        query: Natural language query text.
        db_path: SQLite database path.
        top_k: Number of results to return.
        cfg: Optional EmbedConfig.

    Returns:
        List of {node_id, paper_id, score, tree_layer}.
    """
    import sqlite3

    import numpy as np

    provider = _embed_provider(cfg)
    if provider == "none":
        return []

    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    try:
        # Check if tree_vectors has data
        count = conn.execute("SELECT COUNT(*) FROM tree_vectors").fetchone()[0]
        if count == 0:
            return []

        # Embed query
        query_vec = _embed_batch([query], cfg)[0]
        qv = np.asarray(query_vec, dtype="float32")
        query_dim = len(query_vec)

        # Cosine similarity over all stored vectors
        results = []
        for row in conn.execute(
            "SELECT node_id, paper_id, embedding, tree_layer FROM tree_vectors"
        ):
            blob = row[2]
            stored_dim = len(blob) // 4  # float32 = 4 bytes
            if stored_dim != query_dim:
                logger.warning(
                    "Dimension mismatch in tree_vectors node_id={}: stored={} query={}",
                    row[0],
                    stored_dim,
                    query_dim,
                )
                continue
            stored_vec = np.asarray(struct.unpack(f"{stored_dim}f", blob), dtype="float32")
            sim = float(np.dot(qv, stored_vec))
            results.append(
                {
                    "node_id": row[0],
                    "paper_id": row[1],
                    "score": sim,
                    "tree_layer": row[3],
                }
            )

        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:top_k]

        # Post-filter: remove low-quality results
        results = _post_filter(results)

        return results

    finally:
        conn.close()
