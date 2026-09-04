"""Custom ``BaseRetriever`` wrappers over drbrain-exclusive retrieval assets.

Ticket: T4 (检索器统一). Depends on T2 (LLM bridge) / T3 (index layer).

Wraps three drbrain assets LlamaIndex has no equivalent for, as
:class:`~llama_index.core.retrievers.BaseRetriever` implementations so they can
be fused alongside BM25/vector legs (see :mod:`drbrain.rag.fusion`):

* :class:`DrbrainTreeRetriever` — PageIndex tree navigation. Reuses
  ``query_by_structure_hybrid`` semantics: LLM selection over the tree
  skeleton is the PRIMARY reasoning path, vector search pre-filters
  candidates, and section bodies are loaded on demand from ``raw.md``.
* :class:`DrbrainRAPTORRetriever` — RAPTOR two-stage tree traversal. Reuses
  ``tree_traversal_search`` semantics: layer-by-layer cosine traversal of
  ``tree_vectors`` descending via ``tree_summaries.source_node_ids``, with a
  collapsed-tree fallback. RAPTOR summary rows become ``IndexNode`` (with
  ``source_node_ids`` preserved), pageindex leaves become ``TextNode``.
* :class:`DrbrainGraphRetriever` — knowledge-graph retrieval. ``search_concepts``
  (BM25 over the concepts table) is the entry point; each matched concept is
  expanded via ``get_neighbors`` (graph traversal) so results carry both the
  directly-matched concepts and their graph context.

All three implement ``_retrieve(query_bundle) -> List[NodeWithScore]``
(LlamaIndex 0.14.23 protocol: ``_retrieve`` takes a ``QueryBundle``, not a
plain string) and accept an optional ``paper_id`` filter.

Big-node guard (T3 遗留): real corpora contain ~34 KB tree nodes; node text
loaded by these retrievers is truncated to ``MAX_NODE_CHARS`` before entering
the LLM context.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from drbrain.config import Config

try:
    from llama_index.core.async_utils import asyncio_run
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import IndexNode, NodeWithScore, QueryBundle, TextNode

    _LLAMA_INDEX_AVAILABLE = True
except ImportError:  # pragma: no cover - envs without llama-index
    asyncio_run = None  # type: ignore[assignment,misc]
    BaseRetriever = None  # type: ignore[assignment,misc]
    IndexNode = None  # type: ignore[assignment,misc]
    NodeWithScore = None  # type: ignore[assignment,misc]
    QueryBundle = None  # type: ignore[assignment,misc]
    TextNode = None  # type: ignore[assignment,misc]
    _LLAMA_INDEX_AVAILABLE = False

log = logging.getLogger(__name__)

__all__ = [
    "_LLAMA_INDEX_AVAILABLE",
    "DrbrainGraphRetriever",
    "DrbrainRAPTORRetriever",
    "DrbrainTreeRetriever",
]

#: Layer tag of PageIndex nodes (mirrors ``tree_vectors.tree_layer``).
TREE_LAYER_PAGEINDEX = "pageindex"
#: Max chars of node text handed to the LLM (~8k tokens). Guards against the
#: T3 big-node issue (real nodes up to 34 KB) when loading section bodies.
MAX_NODE_CHARS = 32000


def _truncate_for_llm(text: str, max_chars: int = MAX_NODE_CHARS) -> str:
    """Truncate long node text so it stays inside the LLM context budget."""
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"


def _find_title(structure: list[dict], node_id: str) -> str:
    """Recursive title lookup by node_id in a tree structure."""
    for node in structure:
        if node.get("node_id") == node_id:
            return str(node.get("title") or "")
        children = node.get("nodes")
        if isinstance(children, list) and children:
            title = _find_title(children, node_id)
            if title:
                return title
    return ""


def _find_node(structure: list[dict], node_id: str) -> dict | None:
    """Recursive node-dict lookup by node_id in a tree structure."""
    for node in structure:
        if node.get("node_id") == node_id:
            return node
        children = node.get("nodes")
        if isinstance(children, list) and children:
            found = _find_node(children, node_id)
            if found is not None:
                return found
    return None


def _line_offsets(node: Any) -> tuple[int | None, int | None]:
    """Sanitized ``(line_start, line_end)`` of a tree.json node (None-safe).

    Latex-built trees carry exact 0-based, end-exclusive offsets into
    ``raw.md`` (see ``parser.latex_md.markdown_to_tree``); trees from the
    scibase/oa pipeline only carry ``line_num`` → ``(None, None)``.
    """
    if not isinstance(node, dict):
        return None, None
    ls, le = node.get("line_start"), node.get("line_end")
    if ls is None or le is None:
        return None, None
    try:
        return int(ls), int(le)
    except (TypeError, ValueError):
        return None, None


def _tree_node_offsets(papers_dir, paper_id: str, node_id: str) -> tuple[int | None, int | None]:
    """``(line_start, line_end)`` of one PageIndex node, read from tree.json.

    On-demand companion to :func:`_pageindex_section` for paths that already
    hold the section body from elsewhere (the tree navigator) and only need
    the locator. Returns ``(None, None)`` when the tree/node/offsets are
    unavailable.
    """
    if not papers_dir:
        return None, None
    tree_path = Path(papers_dir) / paper_id / "tree.json"
    if not tree_path.exists():
        return None, None
    try:
        tree = json.loads(tree_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # pragma: no cover - defensive
        return None, None
    structure = tree.get("structure", [])
    return _line_offsets(_find_node(structure if isinstance(structure, list) else [], node_id))


def _pageindex_section(papers_dir, paper_id: str, node_id: str) -> tuple[str, str, dict | None]:
    """Return ``(title, body, node)`` for a PageIndex tree node, loaded on demand.

    Reads ``tree.json`` + ``raw.md`` under ``papers_dir/<paper_id>`` (same
    source the vector index was built from). ``node`` is the resolved
    tree.json node dict — its ``line_start``/``line_end`` locate ``body`` in
    ``raw.md`` — or ``None`` when unresolved. Returns ``("", "", None)`` when
    either file is missing or the node cannot be resolved.
    """
    if not papers_dir:
        return "", "", None
    paper_dir = Path(papers_dir) / paper_id
    try:
        from drbrain.parser.pageindex_parser import get_node_content
        from drbrain.storage.paths import raw_md_path, tree_json_path
    except ImportError:  # pragma: no cover - defensive
        return "", "", None
    tree_path = tree_json_path(paper_dir)
    md_path = raw_md_path(paper_dir)
    if not tree_path.exists() or not md_path.exists():
        return "", "", None
    try:
        tree = json.loads(tree_path.read_text(encoding="utf-8"))
        structure = tree.get("structure", [])
        body = get_node_content(md_path, structure, node_id) or ""
    except (OSError, ValueError):  # pragma: no cover - defensive
        return "", "", None
    return _find_title(structure, node_id), body.strip(), _find_node(structure, node_id)


def _read_tree_summary(db_path, paper_id: str, node_id: str) -> tuple[str, list]:
    """Return ``(summary_text, source_node_ids)`` from ``tree_summaries``.

    ``source_node_ids`` lists the child nodes the RAPTOR summary was built
    from (preserved as metadata for downstream drill-down). Falls back to
    ``("", [])`` when the row is missing or the DB is unavailable.
    """
    if db_path is None or not Path(db_path).exists():
        return "", []
    from drbrain.storage.connection import connect_wal

    conn = connect_wal(db_path)
    try:
        row = conn.execute(
            "SELECT summary_text, source_node_ids FROM tree_summaries "
            "WHERE node_id = ? AND paper_id = ?",
            (node_id, paper_id),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return "", []
    try:
        source_ids = json.loads(row[1]) if row[1] else []
    except (ValueError, TypeError):  # pragma: no cover - defensive
        source_ids = []
    return row[0] or "", list(source_ids)


#: Matches a markdown ATX header (``#`` … ``######``), mirroring the parser's
#: header detection used to derive the ``line_num`` ranges stored in
#: ``tree.json``.
_HEADER_RE = re.compile(r"^(#{1,6})\s")


def _parent_and_path(structure: list[dict], node_id: str) -> tuple[dict | None, list[str]]:
    """Return ``(parent_node, ancestor_ids)`` for ``node_id``.

    ``parent_node`` is the immediate parent dict (or ``None`` when ``node_id``
    is a top-level node or absent); ``ancestor_ids`` is the chain of ancestor
    node_ids from the root down to (and including) the parent. ``tree.json``
    carries no explicit parent pointer — only nested ``nodes`` lists — so the
    parent is recovered by walking the recursion path.
    """
    for node in structure:
        children = node.get("nodes")
        if not isinstance(children, list) or not children:
            continue
        for child in children:
            if child.get("node_id") == node_id:
                return node, [str(node.get("node_id") or "")]
        for child in children:
            parent, path = _parent_and_path([child], node_id)
            if parent is not None:
                return parent, [str(node.get("node_id") or "")] + path
    return None, []


def _full_section_body(md_path: str | Path, node: dict) -> str:
    """Return the FULL body of ``node``'s section (header + all children).

    Slices ``raw.md`` from ``node["line_num"]`` to the next header at the same
    or shallower level — i.e. the whole subtree. Unlike :func:`get_node_content`,
    which stops at the first child header (so a parent's body would collapse to
    its heading), this keeps every descendant chunk. A section's complete
    condition list therefore survives even when the matched leaf was only one
    chunk of it.
    """
    line_num = node.get("line_num")
    if not line_num:
        return ""
    md_path = Path(md_path)
    if not md_path.exists():
        return ""
    lines = md_path.read_text(encoding="utf-8").splitlines()
    start = int(line_num) - 1
    if start < 0 or start >= len(lines):
        return ""
    m = _HEADER_RE.match(lines[start].lstrip())
    level = len(m.group(1)) if m else 1
    end = len(lines)
    for i in range(start + 1, len(lines)):
        hm = _HEADER_RE.match(lines[i].lstrip())
        if hm and len(hm.group(1)) <= level:
            end = i
            break
    return "\n".join(lines[start:end]).strip()


def _parent_section(
    papers_dir, paper_id: str, node_id: str
) -> tuple[str, str, str, dict | None]:
    """Return ``(parent_title, full_parent_body, parent_node_id, parent_node)``.

    Resolves the PARENT of ``node_id`` (the "context unit" surrounding a
    matched leaf) from the paper's ``tree.json`` + ``raw.md`` and returns the
    parent's title together with its FULL section body (header + all children).
    ``parent_node`` is the resolved tree.json parent dict — its
    ``line_start``/``line_end`` locate the body in ``raw.md`` — or ``None``.
    Returns ``("", "", "", None)`` when ``node_id`` is a top-level node (no
    parent) or the files cannot be resolved.

    ``node_id`` must be the bare per-paper id (no ``paper_id:`` prefix) — the
    form ``tree.json`` is keyed by.
    """
    if not papers_dir:
        return "", "", "", None
    paper_dir = Path(papers_dir) / paper_id
    try:
        from drbrain.storage.paths import raw_md_path, tree_json_path
    except ImportError:  # pragma: no cover - defensive
        return "", "", "", None
    tree_path = tree_json_path(paper_dir)
    md_path = raw_md_path(paper_dir)
    if not tree_path.exists() or not md_path.exists():
        return "", "", "", None
    try:
        tree = json.loads(tree_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # pragma: no cover - defensive
        return "", "", "", None
    structure = tree.get("structure", [])
    parent, _ancestors = _parent_and_path(structure, node_id)
    if parent is None:
        return "", "", "", None
    parent_id = str(parent.get("node_id") or "")
    parent_title = str(parent.get("title") or "")
    return parent_title, _full_section_body(md_path, parent), parent_id, parent


if _LLAMA_INDEX_AVAILABLE:

    class DrbrainTreeRetriever(BaseRetriever):
        """PageIndex tree navigation wrapped as a LlamaIndex retriever.

        Faithful port of ``query_by_structure_hybrid`` (single-paper or
        cross-paper): for each target paper the LLM selects promising sections
        from the tree skeleton (PRIMARY), vector search pre-filters candidates
        when a vector store is available (AUXILIARY), and section bodies are
        fetched on demand from ``raw.md``. LLM-picked sections keep priority;
        returned nodes are ranked by that priority so RRF fusion sees a
        meaningful order.

        Nodes are ``TextNode`` keyed ``<paper_id>:<node_id>`` with metadata
        ``{paper_id, node_id, title, source: "tree", pick, tree_layer}`` —
        ``pick`` records which sub-path selected the node (``llm``/``vector``/
        ``llm+vector``), kept separate from the fusion-level ``source``.

        With ``expand_to_parent`` (default on) each matched leaf's text is
        replaced by its PARENT section's full body (header + all children), so
        a section split across sibling chunks reaches the LLM whole. Metadata
        keeps the leaf ``node_id``/``title`` and adds ``parent_node_id``/
        ``parent_title`` for provenance, plus ``line_start``/``line_end`` —
        the raw.md offsets of the section whose text is actually shown (the
        parent when expanded, else the leaf) so checksummed rows stay
        re-locatable (R-I3). ``None`` when the tree carries no offsets
        (line_num-only scibase/oa trees).
        """

        def __init__(
            self,
            cfg: Config,
            paper_id: str | None = None,
            top_k: int = 5,
            db_path=None,
            models: list[dict] | None = None,
            papers_dir=None,
            embed_cfg=None,
            cache=None,
            expand_to_parent: bool = True,
        ) -> None:
            super().__init__()  # sets callback_manager/object_map (IndexNode resolution)
            self.paper_id = paper_id
            self.top_k = int(top_k)
            self._cfg = cfg
            self._models = list(models) if models is not None else list(cfg.llm.models)
            self._papers_dir = Path(papers_dir) if papers_dir else Path(cfg.dirs.papers)
            #: Vector pre-filter needs the SQLite path; ``None`` → pure-LLM mode.
            self._db_path = Path(db_path) if db_path else None
            self._embed_cfg = embed_cfg if embed_cfg is not None else cfg.embed
            self._cache = cache  # ApiCache | None, built lazily on first call
            #: Expand a matched leaf to its parent section's full body (T4).
            self._expand_to_parent = bool(expand_to_parent)

        # ── internals ──────────────────────────────────────────────────

        def _get_cache(self):
            """ApiCache built lazily from config (same rule as DrbrainLLM)."""
            if self._cache is None:
                api = getattr(self._cfg, "api", None)
                ttl = getattr(api, "cache_ttl", 0) or 0
                if ttl > 0:
                    dirs = getattr(self._cfg, "dirs", None)
                    cache_dir = getattr(dirs, "cache", None) or "data/cache"
                    from drbrain.extractor.cache import ApiCache

                    self._cache = ApiCache(cache_dir, ttl=ttl)
            return self._cache

        def _paper_dirs(self) -> list[Path]:
            """Target paper dirs: the filtered paper, or every dir with tree.json."""
            if self.paper_id:
                target = self._papers_dir / self.paper_id
                return [target] if target.is_dir() else []
            if not self._papers_dir.is_dir():
                return []
            return sorted(
                d for d in self._papers_dir.iterdir() if d.is_dir() and (d / "tree.json").exists()
            )

        # ── LlamaIndex protocol ────────────────────────────────────────

        def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
            # asyncio_run handles the nested-loop case (runs in a thread).
            return asyncio_run(self._aretrieve(query_bundle))

        async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
            from drbrain.query.tree_retrieval import query_by_structure_hybrid

            question = query_bundle.query_str
            out: list[NodeWithScore] = []
            for paper_dir in self._paper_dirs():
                sections = await self._navigate_paper(
                    query_by_structure_hybrid, question, paper_dir
                )
                if not sections:
                    continue
                # The legacy navigator returns LLM picks (an unordered set) before
                # vector additions; stabilise the order deterministically so fused
                # rankings don't flip run-to-run: LLM picks first, then vector,
                # node_id as tie-breaker.
                sections = sorted(
                    sections,
                    key=lambda sec: (
                        0 if sec.get("source") in ("llm", "llm+vector", None) else 1,
                        str(sec.get("node_id") or ""),
                    ),
                )
                paper_id = paper_dir.name
                n = len(sections)
                for i, sec in enumerate(sections):
                    nid = str(sec.get("node_id") or "")
                    body = str(sec.get("content") or "").strip()
                    if not nid or not body:
                        continue
                    line_start: int | None = None
                    line_end: int | None = None
                    parent_node_id = ""
                    parent_title = ""
                    if self._expand_to_parent:
                        # Retrieval unit (leaf) ≠ context unit (parent section):
                        # swap the leaf chunk for its parent's full body so a
                        # section split across siblings reaches the LLM whole.
                        parent_title, parent_body, parent_node_id, parent_node = (
                            _parent_section(self._papers_dir, paper_id, nid)
                        )
                        if parent_node_id and parent_body:
                            body = parent_body
                            # R-I3: the offsets must locate the text that is
                            # actually displayed (the parent section), so the
                            # checksummed excerpt can be re-located in raw.md.
                            line_start, line_end = _line_offsets(parent_node)
                        else:
                            parent_node_id = ""
                            parent_title = ""
                    if line_start is None:
                        # Leaf body shown as-is (no expansion, or no parent):
                        # travel with the leaf's own raw.md offsets.
                        line_start, line_end = _tree_node_offsets(
                            self._papers_dir, paper_id, nid
                        )
                    node = TextNode(
                        text=_truncate_for_llm(body),
                        id_=f"{paper_id}:{nid}",
                        metadata={
                            "paper_id": paper_id,
                            "node_id": nid,
                            "title": str(sec.get("title") or ""),
                            "parent_node_id": parent_node_id,
                            "parent_title": parent_title,
                            "source": "tree",
                            "pick": str(sec.get("source") or "llm"),
                            "tree_layer": TREE_LAYER_PAGEINDEX,
                            "line_start": line_start,
                            "line_end": line_end,
                        },
                    )
                    # LLM picks first (position-derived score, priority order).
                    out.append(NodeWithScore(node=node, score=max(0.0, (n - i) / n)))
            return out

        async def _navigate_paper(
            self, navigator, question: str, paper_dir: Path
        ) -> list[dict] | None:
            """Run the PageIndex hybrid navigation for one paper (fault-tolerant)."""
            try:
                return await navigator(
                    question,
                    paper_dir,
                    self._db_path or Path("unused"),
                    self._models,
                    cfg=None if self._db_path is None else self._embed_cfg,
                    top_k=self.top_k,
                    _cache=self._get_cache(),
                )
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("[rag] tree navigation failed for %s: %s", paper_dir.name, exc)
                return None

    class DrbrainRAPTORRetriever(BaseRetriever):
        """RAPTOR two-stage tree traversal wrapped as a LlamaIndex retriever.

        Faithful port of ``tree_traversal_search``: starting from the root
        RAPTOR layer, per-layer cosine scoring keeps top-k nodes, children are
        collected from ``tree_summaries.source_node_ids``, and traversal
        descends to the pageindex leaf layer; when fewer than ``min_results``
        survive, a collapsed-tree fallback runs across all layers.

        Rows are mapped to nodes by layer:
        * ``raptor_L*`` summary rows → ``IndexNode`` (summary text, metadata
          keeps ``source_node_ids`` for drill-down)
        * ``pageindex`` leaves → ``TextNode`` (body loaded on demand from
          ``tree.json`` + ``raw.md``)

        Node ids are ``<paper_id>:<node_id>``; metadata carries ``tree_layer``.

        With ``expand_to_parent`` (default on) a pageindex leaf's text is
        replaced by its PARENT section's full body (header + all children);
        metadata keeps the leaf ``node_id``/``title`` and adds
        ``parent_node_id``/``parent_title``, plus ``line_start``/``line_end`` —
        the raw.md offsets of the section whose text is actually shown (the
        parent when expanded, else the leaf) so checksummed rows stay
        re-locatable (R-I3).
        """

        def __init__(
            self,
            cfg: Config,
            paper_id: str | None = None,
            top_k: int = 5,
            min_results: int = 3,
            db_path=None,
            papers_dir=None,
            embed_cfg=None,
            expand_to_parent: bool = True,
        ) -> None:
            super().__init__()  # sets callback_manager/object_map (IndexNode resolution)
            self.paper_id = paper_id
            self.top_k = int(top_k)
            self.min_results = int(min_results)
            self._db_path = Path(db_path) if db_path else None
            self._papers_dir = Path(papers_dir) if papers_dir else Path(cfg.dirs.papers)
            self._embed_cfg = embed_cfg if embed_cfg is not None else cfg.embed
            #: Expand a matched leaf to its parent section's full body (T4).
            self._expand_to_parent = bool(expand_to_parent)

        def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
            from drbrain.query.tree_retrieval import tree_traversal_search

            if self._db_path is None or not self._db_path.exists():
                return []
            try:
                rows = tree_traversal_search(
                    query_bundle.query_str,
                    self._db_path,
                    top_k=self.top_k,
                    min_results=self.min_results,
                    cfg=self._embed_cfg,
                )
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("[rag] RAPTOR traversal failed: %s", exc)
                return []
            rows = rows or []
            if self.paper_id:
                rows = [r for r in rows if r.get("paper_id") == self.paper_id]
            out: list[NodeWithScore] = []
            for row in rows:
                node = self._row_to_node(row)
                if node is not None:
                    out.append(NodeWithScore(node=node, score=float(row.get("score") or 0.0)))
            return out

        def _row_to_node(self, row: dict) -> Any | None:
            """Map one tree_vectors row to an IndexNode/TextNode (by layer)."""
            paper_id = str(row.get("paper_id") or "")
            nid = str(row.get("node_id") or "")
            layer = str(row.get("tree_layer") or "")
            if not paper_id or not nid:
                return None
            # tree_vectors.node_id is globally unique ("{paper_id}:{local}").
            # The LlamaIndex node id keeps that global form; the per-paper
            # local id (what tree.json / raw.md are keyed by) is stored in
            # metadata and used for on-demand body/title lookups.
            from drbrain.services.embedding import _local_node_id

            local_nid = _local_node_id(paper_id, nid)
            node_id = f"{paper_id}:{local_nid}"
            if layer.startswith("raptor_L"):
                summary, source_ids = _read_tree_summary(self._db_path, paper_id, nid)
                # ``metadata`` is the public alias of IndexNode's (deprecated)
                # ``extra_info`` field; mypy's synthesized ``__init__`` exposes
                # only the field name, so the alias is flagged as unexpected.
                return IndexNode(  # type: ignore[call-arg]
                    text=_truncate_for_llm(summary)
                    if summary
                    else f"[RAPTOR summary: {local_nid}]",
                    id_=node_id,
                    index_id=node_id,
                    metadata={
                        "paper_id": paper_id,
                        "node_id": local_nid,
                        "title": "",
                        "source": "raptor",
                        "tree_layer": layer,
                        "source_node_ids": source_ids,
                    },
                )
            # pageindex leaf: body loaded on demand from raw.md.
            title, body, leaf_node = _pageindex_section(self._papers_dir, paper_id, local_nid)
            line_start: int | None = None
            line_end: int | None = None
            parent_node_id = ""
            parent_title = ""
            if self._expand_to_parent:
                # Retrieval unit (leaf) ≠ context unit (parent section): swap
                # the leaf chunk for its parent's full body so a section split
                # across siblings reaches the LLM whole.
                parent_title, parent_body, parent_node_id, parent_node = _parent_section(
                    self._papers_dir, paper_id, local_nid
                )
                if parent_node_id and parent_body:
                    body = parent_body
                else:
                    parent_node_id = ""
                    parent_title = ""
            if body:
                # R-I3: offsets of the section whose text is actually shown —
                # the parent when the leaf was expanded into it, else the leaf
                # — so checksummed rows can be re-located in raw.md.
                shown_node = parent_node if parent_node_id else leaf_node
                line_start, line_end = _line_offsets(shown_node)
            return TextNode(
                text=_truncate_for_llm(body) if body else f"[section: {local_nid}]",
                id_=node_id,
                metadata={
                    "paper_id": paper_id,
                    "node_id": local_nid,
                    "title": title,
                    "parent_node_id": parent_node_id,
                    "parent_title": parent_title,
                    "source": "raptor",
                    "tree_layer": layer or TREE_LAYER_PAGEINDEX,
                    "line_start": line_start,
                    "line_end": line_end,
                },
            )

    class DrbrainGraphRetriever(BaseRetriever):
        """Knowledge-graph retrieval wrapped as a LlamaIndex retriever.

        ``search_concepts`` (BM25 over concept labels) is the entry point;
        each matched concept is enriched from the ``concepts`` table
        (``type``/``confidence``/``section``, ``paper_id`` filter applied
        here) and expanded via ``get_neighbors`` (1-hop graph traversal).
        Output ``TextNode``s: seed concepts (score = BM25 score) followed by
        their neighbors (score = seed score decayed by hop distance), keyed
        ``concept:<label>`` with metadata ``{type, confidence, source: "graph",
        role: concept|neighbor}``.
        """

        def __init__(
            self,
            db=None,
            graph=None,
            paper_id: str | None = None,
            top_k: int = 5,
            max_neighbors: int = 3,
        ) -> None:
            super().__init__()  # sets callback_manager/object_map (IndexNode resolution)
            self.paper_id = paper_id
            self.top_k = int(top_k)
            self.max_neighbors = int(max_neighbors)
            self._db = db
            self._graph = graph

        def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
            if self._db is None:
                return []
            from drbrain.extractor.agent_tools import search_concepts

            try:
                concepts = search_concepts(self._db, query_bundle.query_str, limit=self.top_k)
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("[rag] graph concept search failed: %s", exc)
                concepts = []
            concepts = concepts or []

            # Dedup seed concepts by label, keeping the best-scoring row.
            best: dict[str, dict] = {}
            for c in concepts:
                label = str(c.get("label") or "").strip()
                if not label:
                    continue
                score = float(c.get("score") or 0.0)
                if label not in best or score > float(best[label].get("score") or 0.0):
                    best[label] = {**c, "score": score}

            seeds = self._enrich_concepts(list(best.values()))
            out: list[NodeWithScore] = []
            for seed in seeds:
                label = seed["label"]
                score = float(seed.get("score") or 0.0)
                out.append(
                    NodeWithScore(
                        node=TextNode(
                            text=_concept_text(seed),
                            id_=f"concept:{label}",
                            metadata={
                                "paper_id": seed.get("local_id") or "",
                                "node_id": f"concept:{label}",
                                "type": seed.get("type") or "",
                                "confidence": seed.get("confidence"),
                                "source": "graph",
                                "role": "concept",
                            },
                        ),
                        score=score,
                    )
                )
                out.extend(self._neighbor_nodes(label, score))
            return out

        def _enrich_concepts(self, concepts: list[dict]) -> list[dict]:
            """Join search hits back to the concepts table; apply paper filter."""
            if not self._db or not concepts:
                return []
            out: list[dict] = []
            for c in concepts:
                label = str(c.get("label") or "").strip()
                ctype = str(c.get("type") or "").strip()
                if not label:
                    continue
                sql = "SELECT local_id, type, label, confidence, section FROM concepts WHERE label = ?"
                params: list = [label]
                if ctype:
                    sql += " AND type = ?"
                    params.append(ctype)
                rows = self._db.conn.execute(sql + " ORDER BY confidence DESC", params).fetchall()
                if self.paper_id:
                    rows = [r for r in rows if r[0] == self.paper_id]
                if rows:
                    r = rows[0]
                    out.append(
                        {
                            "local_id": r[0],
                            "type": r[1],
                            "label": r[2],
                            "confidence": r[3],
                            "section": r[4],
                            "score": float(c.get("score") or 0.0),
                        }
                    )
            return out

        def _neighbor_nodes(self, label: str, seed_score: float) -> list[NodeWithScore]:
            """Expand one seed concept with 1-hop neighbors (score-decayed)."""
            if self._graph is None:
                return []
            from drbrain.extractor.agent_tools import get_neighbors

            try:
                neighbors = get_neighbors(self._graph, label, hops=1, direction="both")
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("[rag] graph neighbor expansion failed for %s: %s", label, exc)
                return []
            out: list[NodeWithScore] = []
            for nb in (neighbors or [])[: self.max_neighbors]:
                target = str(nb.get("target") or "").strip()
                if not target or target == label:
                    continue
                relation = ""
                path = nb.get("path") or []
                if path:
                    relation = str(path[0].get("relation") or "")
                distance = int(nb.get("distance") or 1) or 1
                text = (
                    f"{label} → {target}" if not relation else f"{label} --[{relation}]--> {target}"
                )
                out.append(
                    NodeWithScore(
                        node=TextNode(
                            text=text,
                            id_=f"concept:{target}",
                            metadata={
                                "paper_id": "",
                                "node_id": f"concept:{target}",
                                "type": "Neighbor",
                                "confidence": None,
                                "source": "graph",
                                "role": "neighbor",
                                "relation": relation,
                            },
                        ),
                        score=seed_score * 0.5 / distance,
                    )
                )
            return out


def _concept_text(row: dict) -> str:
    """Compose a concept's text: label + type + section (description)."""
    parts = [str(row.get("label") or "")]
    if row.get("type"):
        parts.append(f"({row['type']})")
    if row.get("section"):
        parts.append(f"— {row['section']}")
    return " ".join(parts).strip() or row.get("label") or ""
