"""Logical section Documents and exact, bounded physical index fragments."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
from loguru import logger
from drbrain.storage.paths import paper_id_from_dir, raw_md_path, tree_json_path
try:
    from llama_index.core.schema import Document
    _LLAMA_INDEX_AVAILABLE = True
except ImportError:
    Document = None
    _LLAMA_INDEX_AVAILABLE = False
TREE_LAYER_PAGEINDEX = "pageindex"
DEFAULT_MAX_NODE_TOKENS = 4000
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
    """Bound physical index fragments while retaining exact parent-text offsets.

    Offsets are Python character offsets into the logical parent Document
    (title plus body), not raw.md byte offsets or inherited section line ranges.
    """
    max_chars = max(1, int(max_node_tokens)) * CHARS_PER_TOKEN
    if len(doc.text) <= max_chars:
        return [doc]
    spans = []
    start = 0
    while start < len(doc.text):
        end = min(start + max_chars, len(doc.text))
        if end < len(doc.text):
            boundary = doc.text.rfind("\n\n", start, end)
            if boundary > start:
                end = boundary + 2
        spans.append((start, end))
        start = end
    out = []
    for i, (start, end) in enumerate(spans):
        md = dict(doc.metadata)
        parent = str(md.get("node_id") or doc.id_)
        md.update(
            {
                "parent_node_id": parent,
                "parent_document_id": doc.id_,
                "parent_checksum": hashlib.sha256(doc.text.encode("utf-8")).hexdigest(),
                "node_id": f"{parent}#{i}",
                "char_start": start,
                "char_end": end,
                "offset_basis": "parent_document",
                "chunk_index": i,
                "chunk_count": len(spans),
            }
        )
        for key in ("line_start", "line_end"):
            if key in md:
                md["parent_" + key] = md.pop(key)
        out.append(Document(text=doc.text[start:end], id_=f"{doc.id_}#{i}", metadata=md))
    return out

def collect_tree_nodes(
    paper_dir: str | Path,
    tree_json: str | Path | dict | None = None,
    max_node_tokens: int | None = None,
    *,
    paper_id: str | None = None,
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
    # ``paper_id`` is the DB identity, not necessarily the directory basename:
    # canonical DOI keys are percent-encoded and legacy DOI assets may be
    # nested.  Index builds pass the DB id explicitly; direct callers get a
    # best-effort decode from the path.
    resolved_paper_id = paper_id or paper_id_from_dir(paper_dir)

    if tree_json is None or isinstance(tree_json, (str, Path)):
        tree_path = Path(tree_json) if tree_json else tree_json_path(paper_dir)
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
        raw_path = raw_md_path(paper_dir)
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
                id_=_node_key(resolved_paper_id, nid),
                metadata={
                    "paper_id": resolved_paper_id,
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
