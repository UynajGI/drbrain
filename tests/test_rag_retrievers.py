"""T4 retriever-unification tests: custom retrievers + RRF fusion.

Covers :mod:`drbrain.rag.retrievers` and :mod:`drbrain.rag.fusion`:

* :class:`DrbrainTreeRetriever` — LLM navigation (mocked) + vector pre-filter
  (mocked) over real test-run papers
* :class:`DrbrainRAPTORRetriever` — synthetic tree_vectors/tree_summaries
  traversal over real papers' raw.md
* :class:`DrbrainGraphRetriever` — real concepts table + fake graph
* :class:`FusionRetriever` / :func:`build_fusion_retriever` /
  :func:`get_retrievers` — mock legs + real T3-built index legs

Unit tests need no GPU/network (LLM and embedding calls are mocked); the one
live-LLM test is marked ``integration``.
"""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from drbrain.config import Config, DirsConfig, EmbedConfig, LlamaIndexConfig
from drbrain.rag.agent import _retrieval_rows
from drbrain.rag.fusion import FusionRetriever, build_fusion_retriever, get_retrievers
from drbrain.rag.indexer import build_index, load_index
from drbrain.rag.retrievers import (
    MAX_NODE_CHARS,
    DrbrainGraphRetriever,
    DrbrainRAPTORRetriever,
    DrbrainTreeRetriever,
    _full_section_body,
    _pageindex_section,
    _parent_and_path,
    _parent_section,
    _truncate_for_llm,
)
from drbrain.storage.database import Database

_HAS_LLAMA_INDEX = importlib.util.find_spec("llama_index") is not None

if _HAS_LLAMA_INDEX:
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import IndexNode, NodeWithScore, TextNode
    from pydantic import PrivateAttr

TEST_RUN = Path(__file__).resolve().parents[1] / "test-run"
REAL_PAPERS = TEST_RUN / "papers"
PAPER_A = "10.1002_adma.202308655"  # 3 tree nodes: 0000/0001/0002
PAPER_B = "10.3390_ma15134622"  # 4 tree nodes: 0000..0003

pytestmark = pytest.mark.skipif(not _HAS_LLAMA_INDEX, reason="llama_index not installed")

_MODELS = [{"provider": "openai", "model": "gpt-4o", "api_key": "k", "base_url": None}]


# ── Shared helpers ───────────────────────────────────────────────────────────


class _CountingEmbed(BaseEmbedding):
    """Deterministic embed adapter (same as test_rag_indexer's)."""

    _embedded_texts: list[str] = PrivateAttr(default_factory=list)

    def __init__(self) -> None:
        super().__init__(model_name="fake-embed", embed_batch_size=8)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            self._embedded_texts.append(t)
            out.append([float(len(t)), 1.0, 0.0])
        return out

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embeddings([text])[0]

    def _get_query_embedding(self, query: str) -> list[float]:
        return [float(len(query)), 1.0, 0.0]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)


class _PaperDB:
    """Minimal db stand-in exposing get_all_papers()."""

    def __init__(self, ids: list[str]) -> None:
        self._ids = ids

    def get_all_papers(self) -> list[dict]:
        return [{"local_id": pid} for pid in self._ids]


class _StaticRetriever(BaseRetriever):
    """Returns a fixed NodeWithScore list (or raises when configured)."""

    def __init__(self, nodes, error: Exception | None = None) -> None:
        self._nodes = list(nodes or [])
        self._error = error

    def _retrieve(self, query_bundle):  # noqa: ANN001
        if self._error is not None:
            raise self._error
        return list(self._nodes)


class _PathElem:
    """Mirror of GraphEngine's path-step object (attribute access)."""

    def __init__(self, src, relation, dst) -> None:
        self.src = src
        self.relation = relation
        self.dst = dst


class _TraverseResult:
    """Mirror of GraphEngine.TraverseResult (attribute access)."""

    def __init__(self, target, source, distance, path) -> None:
        self.target = target
        self.source = source
        self.distance = distance
        self.path = path


class _FakeGraph:
    """Minimal GraphEngine stand-in with a per-label neighbor map."""

    def __init__(self, neighbors: dict[str, list[dict]] | None = None) -> None:
        self._neighbors = neighbors or {}

    def traverse(self, start_nodes, hops: int = 1, direction: str = "both") -> list:
        out = []
        for node in start_nodes:
            for nb in self._neighbors.get(node, []):
                path = [
                    _PathElem(s.get("src", ""), s.get("relation", ""), s.get("dst", ""))
                    for s in nb.get("path", [])
                ]
                out.append(
                    _TraverseResult(
                        target=nb["target"],
                        source=nb["source"],
                        distance=nb.get("distance", 1),
                        path=path,
                    )
                )
        return out


def _mk_node(
    nid: str, text: str = "body", meta: dict | None = None, score: float = 1.0
) -> NodeWithScore:
    return NodeWithScore(node=TextNode(text=text, id_=nid, metadata=meta or {}), score=score)


def _make_cfg(tmp_path, papers_dir=None) -> Config:
    return Config(
        llamaindex=LlamaIndexConfig(
            enabled=True, vector_store="memory", storage_dir=str(tmp_path / "li")
        ),
        dirs=DirsConfig(papers=str(papers_dir or REAL_PAPERS)),
        embed=EmbedConfig(provider="local", model="fake-embed", top_k=5),
    )


def _write_synthetic_paper(papers_dir: Path, pid: str) -> None:
    """tree.json with line_num only + matching raw.md (2 sections)."""
    paper_dir = papers_dir / pid
    paper_dir.mkdir(parents=True, exist_ok=True)
    (paper_dir / "raw.md").write_text(
        "cover line\n"
        "# Intro\n"
        "polymer drag reduction basics\n"
        "turbulent channel flow experiments\n"
        "# Methods\n"
        "solvothermal nanoparticle synthesis\n"
        "characterization via XRD and SEM\n",
        encoding="utf-8",
    )
    (paper_dir / "tree.json").write_text(
        json.dumps(
            {
                "doc_name": pid,
                "line_count": 7,
                "structure": [
                    {"title": "Intro", "node_id": "0000", "line_num": 2, "nodes": []},
                    {"title": "Methods", "node_id": "0001", "line_num": 5, "nodes": []},
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_structured_paper(papers_dir: Path, pid: str, sections: list[tuple[str, str]]) -> None:
    """Write tree.json + raw.md for a paper from explicit ``(title, body)`` sections."""
    paper_dir = papers_dir / pid
    paper_dir.mkdir(parents=True, exist_ok=True)
    lines = ["cover line"]
    structure = []
    for i, (title, body) in enumerate(sections):
        header_line = len(lines) + 1  # 1-based header line
        lines.append(f"# {title}")
        lines.extend(body.split("\n"))
        structure.append(
            {"title": title, "node_id": f"{i:04d}", "line_num": header_line, "nodes": []}
        )
    (paper_dir / "raw.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (paper_dir / "tree.json").write_text(
        json.dumps({"doc_name": pid, "line_count": len(lines), "structure": structure}),
        encoding="utf-8",
    )


def _write_nested_paper(papers_dir: Path, pid: str) -> None:
    """tree.json with a 2-level hierarchy + matching raw.md (parent + 2 children)."""
    paper_dir = papers_dir / pid
    paper_dir.mkdir(parents=True, exist_ok=True)
    (paper_dir / "raw.md").write_text(
        "# Parent Section\n"
        "intro to the section\n"
        "## Child A\n"
        "condition list part 1\n"
        "## Child B\n"
        "condition list part 2\n"
        "# Sibling Section\n"
        "unrelated content\n",
        encoding="utf-8",
    )
    (paper_dir / "tree.json").write_text(
        json.dumps(
            {
                "doc_name": pid,
                "line_count": 8,
                "structure": [
                    {
                        "title": "Parent Section",
                        "node_id": "0000",
                        "line_num": 1,
                        "nodes": [
                            {"title": "Child A", "node_id": "0001", "line_num": 3, "nodes": []},
                            {"title": "Child B", "node_id": "0002", "line_num": 5, "nodes": []},
                        ],
                    },
                    {"title": "Sibling Section", "node_id": "0003", "line_num": 7, "nodes": []},
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_nested_paper_with_offsets(papers_dir: Path, pid: str) -> None:
    """Same nested paper, but every node carries exact line offsets.

    Mirrors the latex-built tree format (``parser.latex_md.markdown_to_tree``):
    0-based, end-exclusive ``line_start``/``line_end`` into raw.md, where a
    parent's range spans its whole section (header + all children).
    """
    paper_dir = papers_dir / pid
    paper_dir.mkdir(parents=True, exist_ok=True)
    raw = (
        "# Parent Section\n"
        "intro to the section\n"
        "## Child A\n"
        "condition list part 1\n"
        "## Child B\n"
        "condition list part 2\n"
        "# Sibling Section\n"
        "unrelated content\n"
    )
    (paper_dir / "raw.md").write_text(raw, encoding="utf-8")
    (paper_dir / "tree.json").write_text(
        json.dumps(
            {
                "doc_name": pid,
                "line_count": 8,
                "structure": [
                    {
                        "title": "Parent Section",
                        "node_id": "0000",
                        "line_num": 1,
                        "line_start": 0,
                        "line_end": 6,
                        "nodes": [
                            {
                                "title": "Child A",
                                "node_id": "0001",
                                "line_num": 3,
                                "line_start": 2,
                                "line_end": 4,
                                "nodes": [],
                            },
                            {
                                "title": "Child B",
                                "node_id": "0002",
                                "line_num": 5,
                                "line_start": 4,
                                "line_end": 6,
                                "nodes": [],
                            },
                        ],
                    },
                    {
                        "title": "Sibling Section",
                        "node_id": "0003",
                        "line_num": 7,
                        "line_start": 6,
                        "line_end": 8,
                        "nodes": [],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


# PAPER_A: 3 small nodes (0000/0001/0002) — the synthetic stand-in for the
# former test-run paper used by the tree/fusion/retriever tests.
_PAPER_A_SECTIONS = [
    ("Intro", "polymer drag reduction basics\nturbulent channel flow experiments"),
    ("Methods", "solvothermal nanoparticle synthesis\ncharacterization via XRD and SEM"),
    ("Results", "measured drag reduction of 40 percent"),
]


async def _fake_acall(prompt, models, system_prompt=None, max_tokens=1024, _cache=None):
    """Mock of tree_retrieval.acall_with_fallback: pick nodes 0000+0002."""
    del prompt, models, system_prompt, max_tokens, _cache
    return {"node_ids": ["0000", "0002"], "reasoning": "mocked"}


def _fake_search_tree(*args, **kwargs):
    """Mock of embedding.search_tree: one vector hit on node 0001."""
    del args, kwargs
    return [{"node_id": "0001", "paper_id": PAPER_A, "score": 0.9, "tree_layer": "pageindex"}]


# ── DrbrainTreeRetriever ─────────────────────────────────────────────────────


def test_tree_retriever_llm_primary_and_vector_merge(tmp_path, monkeypatch):
    papers_dir = tmp_path / "papers"
    _write_structured_paper(papers_dir, PAPER_A, _PAPER_A_SECTIONS)
    cfg = _make_cfg(tmp_path, papers_dir)
    monkeypatch.setattr("drbrain.query.tree_retrieval.acall_with_fallback", _fake_acall)
    monkeypatch.setattr("drbrain.services.embedding.search_tree", _fake_search_tree)

    retriever = DrbrainTreeRetriever(
        cfg, paper_id=PAPER_A, top_k=5, db_path=Path("unused-db"), models=_MODELS
    )
    nodes = retriever.retrieve("which sections cover the methods?")
    assert nodes, "tree retriever must return sections"
    # LLM picks (0000, 0002) first, vector adds 0001 → 3 nodes, 0000 leads.
    assert nodes[0].node.node_id == f"{PAPER_A}:0000"
    assert nodes[0].score >= nodes[-1].score
    for nws in nodes:
        node = nws.node
        assert node.node_id.startswith(f"{PAPER_A}:")
        assert node.metadata["paper_id"] == PAPER_A
        assert node.metadata["source"] == "tree"
        assert node.metadata["tree_layer"] == "pageindex"
        assert node.metadata["node_id"]
        assert node.metadata["title"]
        assert node.metadata["pick"] in ("llm", "vector", "llm+vector")
        assert node.text, "node must carry the section body"
    picks = [n.node.metadata["pick"] for n in nodes]
    assert "vector" in picks, "vector pre-filter leg must contribute a candidate"


def test_tree_retriever_paper_filter(tmp_path, monkeypatch):
    papers_dir = tmp_path / "papers"
    _write_synthetic_paper(papers_dir, "pa")
    _write_synthetic_paper(papers_dir, "pb")
    cfg = _make_cfg(tmp_path, papers_dir)
    monkeypatch.setattr(
        "drbrain.query.tree_retrieval.acall_with_fallback",
        _fake_acall,  # picks 0000/0002
    )
    monkeypatch.setattr("drbrain.services.embedding.search_tree", _fake_search_tree)
    # paper filter: only pa
    filtered = DrbrainTreeRetriever(cfg, paper_id="pa", top_k=5, db_path=None, models=_MODELS)
    nodes = filtered.retrieve("q")
    assert nodes
    assert all(n.node.metadata["paper_id"] == "pa" for n in nodes)
    assert all(n.node.node_id.startswith("pa:") for n in nodes)
    # no filter: both papers navigated (node ids stay unique via paper prefix)
    both = DrbrainTreeRetriever(cfg, top_k=5, db_path=None, models=_MODELS)
    nodes_both = both.retrieve("q")
    assert {n.node.metadata["paper_id"] for n in nodes_both} == {"pa", "pb"}
    assert len({n.node.node_id for n in nodes_both}) == len(nodes_both)


def test_tree_retriever_truncates_big_node(tmp_path, monkeypatch):
    papers_dir = tmp_path / "papers"
    _write_synthetic_paper(papers_dir, "pa")
    cfg = _make_cfg(tmp_path, papers_dir)

    async def _big_acall(prompt, models, system_prompt=None, max_tokens=1024, _cache=None):
        del prompt, models, system_prompt, max_tokens, _cache
        return {"node_ids": ["0000"], "reasoning": "mocked"}

    monkeypatch.setattr("drbrain.query.tree_retrieval.acall_with_fallback", _big_acall)
    monkeypatch.setattr("drbrain.services.embedding.search_tree", lambda *a, **k: [])
    # raw.md body beyond MAX_NODE_CHARS → node text must be truncated
    raw = papers_dir / "pa" / "raw.md"
    raw.write_text("# Intro\n" + "x" * (MAX_NODE_CHARS + 500) + "\n", encoding="utf-8")

    retriever = DrbrainTreeRetriever(cfg, paper_id="pa", top_k=5, db_path=None, models=_MODELS)
    nodes = retriever.retrieve("q")
    assert nodes
    assert len(nodes[0].node.text) <= MAX_NODE_CHARS + 20
    assert nodes[0].node.text.endswith("...[truncated]")
    # plain truncation helper
    assert _truncate_for_llm("short") == "short"


def test_tree_retriever_paper_without_tree_json(tmp_path, monkeypatch):
    papers_dir = tmp_path / "papers"
    (papers_dir / "pa").mkdir(parents=True)  # no tree.json
    cfg = _make_cfg(tmp_path, papers_dir)
    monkeypatch.setattr("drbrain.query.tree_retrieval.acall_with_fallback", _fake_acall)
    retriever = DrbrainTreeRetriever(cfg, paper_id="pa", top_k=5, db_path=None, models=_MODELS)
    assert retriever.retrieve("q") == []


# ── parent-section expansion (retrieval unit ≠ context unit) ─────────────────


def test_parent_section_helper(tmp_path):
    papers_dir = tmp_path / "papers"
    _write_nested_paper(papers_dir, "pa")
    tree = json.loads((papers_dir / "pa" / "tree.json").read_text(encoding="utf-8"))
    structure = tree["structure"]

    # Leaf "0001" resolves its parent via the recursion path (no explicit ptr).
    parent, path = _parent_and_path(structure, "0001")
    assert parent is not None
    assert parent["node_id"] == "0000"
    assert path == ["0000"]

    # Leaf body stops at its own chunk; parent body spans both children.
    title, body, leaf_node = _pageindex_section(papers_dir, "pa", "0001")
    assert title == "Child A"
    assert "part 1" in body and "part 2" not in body
    assert leaf_node is not None and leaf_node["node_id"] == "0001"
    # line_num-only tree: no line offsets on the node dict.
    assert leaf_node.get("line_start") is None and leaf_node.get("line_end") is None

    ptitle, pbody, pnid, pnode = _parent_section(papers_dir, "pa", "0001")
    assert (ptitle, pnid) == ("Parent Section", "0000")
    assert pnode is not None and pnode["node_id"] == "0000"
    assert "part 1" in pbody and "part 2" in pbody, "parent body must cover the full section"
    assert "Sibling Section" not in pbody, "parent body must stop before the next sibling"

    # Top-level nodes have no parent → empty expansion.
    assert _parent_section(papers_dir, "pa", "0000") == ("", "", "", None)
    assert _parent_section(papers_dir, "pa", "0003") == ("", "", "", None)

    # _full_section_body mirrors the parent slice directly from raw.md.
    assert _full_section_body(papers_dir / "pa" / "raw.md", parent) == pbody


async def _pick_leaf_acall(prompt, models, system_prompt=None, max_tokens=1024, _cache=None):
    """Mock navigator: pick a single leaf node (0001) that has a parent."""
    del prompt, models, system_prompt, max_tokens, _cache
    return {"node_ids": ["0001"], "reasoning": "mocked"}


def test_tree_retriever_expands_to_parent(tmp_path, monkeypatch):
    papers_dir = tmp_path / "papers"
    _write_nested_paper(papers_dir, "pa")
    cfg = _make_cfg(tmp_path, papers_dir)
    monkeypatch.setattr("drbrain.query.tree_retrieval.acall_with_fallback", _pick_leaf_acall)
    monkeypatch.setattr("drbrain.services.embedding.search_tree", lambda *a, **k: [])

    retriever = DrbrainTreeRetriever(cfg, paper_id="pa", top_k=5, db_path=None, models=_MODELS)
    nodes = retriever.retrieve("what are the conditions?")
    assert len(nodes) == 1
    node = nodes[0].node
    # text is the PARENT section (both children), not just the matched leaf.
    assert "part 1" in node.text and "part 2" in node.text
    assert node.metadata["node_id"] == "0001"  # leaf id kept for provenance
    assert node.metadata["title"] == "Child A"  # leaf title kept
    assert node.metadata["parent_node_id"] == "0000"
    assert node.metadata["parent_title"] == "Parent Section"


def test_tree_retriever_expand_to_parent_disabled(tmp_path, monkeypatch):
    papers_dir = tmp_path / "papers"
    _write_nested_paper(papers_dir, "pa")
    cfg = _make_cfg(tmp_path, papers_dir)
    monkeypatch.setattr("drbrain.query.tree_retrieval.acall_with_fallback", _pick_leaf_acall)
    monkeypatch.setattr("drbrain.services.embedding.search_tree", lambda *a, **k: [])

    retriever = DrbrainTreeRetriever(
        cfg, paper_id="pa", top_k=5, db_path=None, models=_MODELS, expand_to_parent=False
    )
    nodes = retriever.retrieve("what are the conditions?")
    assert len(nodes) == 1
    node = nodes[0].node
    assert "part 1" in node.text and "part 2" not in node.text  # leaf chunk only
    assert node.metadata["node_id"] == "0001"
    assert node.metadata["parent_node_id"] == ""
    assert node.metadata["parent_title"] == ""


# ── R-I3: line offsets travel with the displayed text ────────────────────────


def test_tree_retriever_expanded_leaf_carries_parent_line_offsets(tmp_path, monkeypatch):
    """Leaf expanded to the parent body → offsets locate the PARENT section."""
    papers_dir = tmp_path / "papers"
    _write_nested_paper_with_offsets(papers_dir, "pa")
    cfg = _make_cfg(tmp_path, papers_dir)
    monkeypatch.setattr("drbrain.query.tree_retrieval.acall_with_fallback", _pick_leaf_acall)
    monkeypatch.setattr("drbrain.services.embedding.search_tree", lambda *a, **k: [])

    retriever = DrbrainTreeRetriever(cfg, paper_id="pa", top_k=5, db_path=None, models=_MODELS)
    nodes = retriever.retrieve("what are the conditions?")
    assert len(nodes) == 1
    node = nodes[0].node
    assert "part 1" in node.text and "part 2" in node.text  # parent body shown
    assert node.metadata["node_id"] == "0001"  # leaf id kept for provenance
    assert node.metadata["parent_node_id"] == "0000"
    # offsets of the displayed parent section (0-based, end-exclusive)
    assert node.metadata["line_start"] == 0
    assert node.metadata["line_end"] == 6
    # the declared parent slice reproduces the displayed body
    assert "\n".join(
        (papers_dir / "pa" / "raw.md").read_text(encoding="utf-8").split("\n")[0:6]
    ).strip() == node.text


def test_tree_retriever_leaf_keeps_its_own_line_offsets_without_expansion(
    tmp_path, monkeypatch
):
    """No expansion → the leaf body is shown → the leaf's own offsets travel."""
    papers_dir = tmp_path / "papers"
    _write_nested_paper_with_offsets(papers_dir, "pa")
    cfg = _make_cfg(tmp_path, papers_dir)
    monkeypatch.setattr("drbrain.query.tree_retrieval.acall_with_fallback", _pick_leaf_acall)
    monkeypatch.setattr("drbrain.services.embedding.search_tree", lambda *a, **k: [])

    retriever = DrbrainTreeRetriever(
        cfg, paper_id="pa", top_k=5, db_path=None, models=_MODELS, expand_to_parent=False
    )
    nodes = retriever.retrieve("what are the conditions?")
    assert len(nodes) == 1
    node = nodes[0].node
    assert "part 2" not in node.text  # leaf chunk only
    assert node.metadata["line_start"] == 2
    assert node.metadata["line_end"] == 4


def test_tree_retriever_line_offsets_absent_for_line_num_only_trees(tmp_path, monkeypatch):
    """scibase/oa trees carry line_num only → offsets stay None (key-compatible)."""
    papers_dir = tmp_path / "papers"
    _write_nested_paper(papers_dir, "pa")
    cfg = _make_cfg(tmp_path, papers_dir)
    monkeypatch.setattr("drbrain.query.tree_retrieval.acall_with_fallback", _pick_leaf_acall)
    monkeypatch.setattr("drbrain.services.embedding.search_tree", lambda *a, **k: [])

    retriever = DrbrainTreeRetriever(cfg, paper_id="pa", top_k=5, db_path=None, models=_MODELS)
    nodes = retriever.retrieve("what are the conditions?")
    assert len(nodes) == 1
    node = nodes[0].node
    assert "part 1" in node.text and "part 2" in node.text  # expansion still works
    assert node.metadata["line_start"] is None
    assert node.metadata["line_end"] is None


# ── DrbrainRAPTORRetriever ───────────────────────────────────────────────────


def _seed_raptor_db(db_path: Path) -> None:
    """tree_vectors (2 RAPTOR layers + pageindex) + tree_summaries links."""
    db = Database(db_path)

    def blob(*v: float) -> bytes:
        arr = np.asarray(v, dtype="float32")
        return (arr / np.linalg.norm(arr)).astype("float32").tobytes()

    rows = [
        ("0000", PAPER_A, blob(1, 0, 0, 0), "pageindex"),
        ("0001", PAPER_A, blob(0, 1, 0, 0), "pageindex"),
        ("1000", PAPER_B, blob(0, 0, 1, 0), "pageindex"),
        ("1001", PAPER_B, blob(0, 0, 0, 1), "pageindex"),
        ("r1", PAPER_A, blob(0.5, 0.5, 0, 0), "raptor_L1"),
        ("r0", PAPER_A, blob(1, 1, 0, 0), "raptor_L2"),
    ]
    for nid, pid, emb, layer in rows:
        db.conn.execute(
            "INSERT INTO tree_vectors (node_id, paper_id, embedding, content_hash, tree_layer) "
            "VALUES (?,?,?,?,?)",
            (nid, pid, emb, "hash", layer),
        )
    db.conn.execute(
        "INSERT INTO tree_summaries (node_id, paper_id, summary_text, source_node_ids, tree_layer) "
        "VALUES (?,?,?,?,?)",
        ("r1", PAPER_A, "summary of sections 0000 and 0001", '["0000","0001"]', 1),
    )
    db.conn.execute(
        "INSERT INTO tree_summaries (node_id, paper_id, summary_text, source_node_ids, tree_layer) "
        "VALUES (?,?,?,?,?)",
        ("r0", PAPER_A, "root summary over section clusters", '["r1"]', 2),
    )
    db.conn.commit()
    db.close()


def test_raptor_retriever_traversal_with_fallback(tmp_path, monkeypatch):
    """Traversal descends raptor_L2 → raptor_L1 → pageindex; fallback adds rows."""
    db_path = tmp_path / "raptor.db"
    _seed_raptor_db(db_path)
    cfg = _make_cfg(tmp_path, REAL_PAPERS)
    monkeypatch.setattr(
        "drbrain.services.embedding._embed_batch", lambda texts, cfg=None: [[1.0, 0.0, 0.0, 0.0]]
    )

    retriever = DrbrainRAPTORRetriever(cfg, top_k=5, min_results=3, db_path=db_path)
    nodes = retriever.retrieve("polymer drag reduction")
    assert nodes, "RAPTOR traversal must return nodes"

    by_id = {nws.node.node_id: nws for nws in nodes}
    # pageindex leaf with real raw.md body loaded on demand
    leaf = by_id.get(f"{PAPER_A}:0000")
    assert leaf is not None
    assert isinstance(leaf.node, TextNode) and not isinstance(leaf.node, IndexNode)
    assert leaf.node.metadata["tree_layer"] == "pageindex"
    assert leaf.node.metadata["source"] == "raptor"
    assert leaf.node.text, "leaf body must load from raw.md"
    # RAPTOR summary rows became IndexNodes with parent links preserved
    raptor_nodes = [n for n in nodes if n.node.metadata["tree_layer"].startswith("raptor_L")]
    assert raptor_nodes, "fallback must surface RAPTOR summary rows"
    summary = [n for n in raptor_nodes if n.node.metadata["tree_layer"] == "raptor_L1"][0]
    assert isinstance(summary.node, IndexNode)
    assert summary.node.metadata["source_node_ids"] == ["0000", "0001"]
    assert "summary of sections" in summary.node.text
    # scores preserved from traversal, descending
    scores = [nws.score for nws in nodes]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] >= 0.5  # query [1,0,0,0] hits node 0000 (sim 1.0)


def test_raptor_retriever_paper_filter(tmp_path, monkeypatch):
    db_path = tmp_path / "raptor.db"
    _seed_raptor_db(db_path)
    cfg = _make_cfg(tmp_path, REAL_PAPERS)
    monkeypatch.setattr(
        "drbrain.services.embedding._embed_batch", lambda texts, cfg=None: [[1.0, 0.0, 0.0, 0.0]]
    )

    unfiltered = DrbrainRAPTORRetriever(cfg, top_k=5, min_results=3, db_path=db_path)
    all_papers = {n.node.metadata["paper_id"] for n in unfiltered.retrieve("q")}
    assert len(all_papers) >= 2, "unfiltered traversal must cross papers"

    filtered = DrbrainRAPTORRetriever(
        cfg, paper_id=PAPER_A, top_k=5, min_results=3, db_path=db_path
    )
    nodes = filtered.retrieve("q")
    assert nodes
    assert all(n.node.metadata["paper_id"] == PAPER_A for n in nodes)


def test_raptor_retriever_missing_db(tmp_path):
    cfg = _make_cfg(tmp_path, REAL_PAPERS)
    retriever = DrbrainRAPTORRetriever(cfg, db_path=tmp_path / "nope.db")
    assert retriever.retrieve("q") == []


def _seed_pageindex_db(db_path: Path, paper_id: str, node_id: str, vec: list[float]) -> None:
    """tree_vectors with a single pageindex leaf (no RAPTOR layers)."""
    db = Database(db_path)
    arr = np.asarray(vec, dtype="float32")
    blob = (arr / np.linalg.norm(arr)).astype("float32").tobytes()
    db.conn.execute(
        "INSERT INTO tree_vectors (node_id, paper_id, embedding, content_hash, tree_layer) "
        "VALUES (?,?,?,?,?)",
        (node_id, paper_id, blob, "hash", "pageindex"),
    )
    db.conn.commit()
    db.close()


def test_raptor_retriever_expands_to_parent(tmp_path, monkeypatch):
    papers_dir = tmp_path / "papers"
    _write_nested_paper(papers_dir, "pa")
    db_path = tmp_path / "raptor.db"
    _seed_pageindex_db(db_path, "pa", "0001", [1.0, 0.0, 0.0])
    cfg = _make_cfg(tmp_path, papers_dir)
    monkeypatch.setattr(
        "drbrain.services.embedding._embed_batch",
        lambda texts, cfg=None: [[1.0, 0.0, 0.0]],
    )

    retriever = DrbrainRAPTORRetriever(cfg, top_k=5, min_results=3, db_path=db_path)
    nodes = retriever.retrieve("what are the conditions?")
    assert len(nodes) == 1
    node = nodes[0].node
    assert node.node_id == "pa:0001"
    # leaf (0001) expanded to its parent section (0000) covering both children.
    assert "part 1" in node.text and "part 2" in node.text
    assert node.metadata["node_id"] == "0001"
    assert node.metadata["title"] == "Child A"
    assert node.metadata["parent_node_id"] == "0000"
    assert node.metadata["parent_title"] == "Parent Section"


def test_raptor_leaf_row_carries_displayed_text_offsets_and_checksum(tmp_path, monkeypatch):
    """R-I3 end-to-end: expanded leaf → parent offsets + checksum over shown text.

    The row keeps the leaf ``node_id`` but its checksums must re-verify the
    text actually displayed (the parent section body), and its line offsets
    must locate that same text in raw.md — not the leaf chunk's.
    """
    papers_dir = tmp_path / "papers"
    _write_nested_paper_with_offsets(papers_dir, "pa")
    db_path = tmp_path / "raptor.db"
    _seed_pageindex_db(db_path, "pa", "0001", [1.0, 0.0, 0.0])
    cfg = _make_cfg(tmp_path, papers_dir)
    monkeypatch.setattr(
        "drbrain.services.embedding._embed_batch",
        lambda texts, cfg=None: [[1.0, 0.0, 0.0]],
    )

    retriever = DrbrainRAPTORRetriever(cfg, top_k=5, min_results=3, db_path=db_path)
    nodes = retriever.retrieve("what are the conditions?")
    assert len(nodes) == 1
    node = nodes[0].node
    assert node.metadata["node_id"] == "0001"  # leaf id kept for provenance
    assert node.metadata["parent_node_id"] == "0000"
    assert "part 1" in node.text and "part 2" in node.text  # parent body displayed
    # offsets of the DISPLAYED parent section, not the matched leaf
    assert node.metadata["line_start"] == 0
    assert node.metadata["line_end"] == 6

    rows = _retrieval_rows(nodes, generation="g-1", query="what are the conditions?")
    row = rows[0]
    assert row["node_id"] == "0001"
    assert row["line_start"] == 0
    assert row["line_end"] == 6
    # checksums bind the actual displayed text (full chunk + 500-char excerpt)
    assert row["content_checksum"] == hashlib.sha256(node.text.encode("utf-8")).hexdigest()
    assert row["excerpt_checksum"] == hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
    leaf_only = "condition list part 1"
    assert row["content_checksum"] != hashlib.sha256(leaf_only.encode("utf-8")).hexdigest()
    # the offsets re-locate exactly the checksummed display text in raw.md
    located = "\n".join(
        (papers_dir / "pa" / "raw.md").read_text(encoding="utf-8").split("\n")[0:6]
    ).strip()
    assert located == node.text
    assert row["content_checksum"] == hashlib.sha256(located.encode("utf-8")).hexdigest()


# ── DrbrainGraphRetriever ────────────────────────────────────────────────────


def _seed_concepts(db: Database) -> None:
    for pid, title in ((PAPER_A, "Polymer drag reduction"), (PAPER_B, "Another paper")):
        db.conn.execute(
            "INSERT INTO papers (local_id, title, year) VALUES (?,?,?)", (pid, title, 2024)
        )
    concepts = [
        (PAPER_A, "Method", "polymer drag reduction", 0.9, "Methods"),
        (PAPER_A, "Problem", "drag reduction", 0.8, "Abstract"),
        (PAPER_B, "Method", "polymer drag reduction", 0.7, "Methods"),
        # filler rows: keep corpus large enough that BM25 IDF stays positive
        # (rank_bm25 clamps negative-IDF terms; tiny corpora then round to 0)
        (PAPER_A, "Method", "thermal conductivity", 0.8, "Methods"),
        (PAPER_A, "Problem", "bandgap engineering", 0.7, "Abstract"),
        (PAPER_B, "Conclusion", "perovskite solar cells", 0.9, "Conclusions"),
        (PAPER_B, "Method", "lithium battery cycling", 0.8, "Methods"),
        (PAPER_B, "Problem", "catalyst degradation", 0.6, "Abstract"),
    ]
    for pid, ctype, label, conf, section in concepts:
        db.conn.execute(
            "INSERT INTO concepts (local_id, type, label, confidence, section) VALUES (?,?,?,?,?)",
            (pid, ctype, label, conf, section),
        )
    db.conn.commit()


def test_graph_retriever_concepts_and_neighbors(tmp_db):
    _seed_concepts(tmp_db)
    graph = _FakeGraph(
        {
            "polymer drag reduction": [
                {
                    "target": "nanoparticle synthesis",
                    "source": "polymer drag reduction",
                    "distance": 1,
                    "path": [
                        {
                            "src": "polymer drag reduction",
                            "relation": "proposes",
                            "dst": "nanoparticle synthesis",
                        }
                    ],
                }
            ]
        }
    )
    retriever = DrbrainGraphRetriever(db=tmp_db, graph=graph, top_k=5, max_neighbors=3)
    nodes = retriever.retrieve("polymer drag reduction")
    assert nodes, "graph retriever must return concepts"

    seed = next(n for n in nodes if n.node.metadata["role"] == "concept")
    assert seed.node.metadata["source"] == "graph"
    assert seed.node.metadata["type"] == "Method"
    assert seed.node.metadata["confidence"] == 0.9
    assert seed.node.metadata["paper_id"] == PAPER_A
    assert "polymer drag reduction" in seed.node.text
    assert "(Method)" in seed.node.text
    assert seed.score > 0

    neighbor = next(n for n in nodes if n.node.metadata["role"] == "neighbor")
    assert neighbor.node.metadata["relation"] == "proposes"
    assert "nanoparticle synthesis" in neighbor.node.text
    assert neighbor.score == seed.score * 0.5  # 1-hop decay
    assert neighbor.score < seed.score


def test_graph_retriever_paper_filter(tmp_db):
    _seed_concepts(tmp_db)
    retriever = DrbrainGraphRetriever(db=tmp_db, paper_id=PAPER_B, top_k=5)
    nodes = retriever.retrieve("polymer drag reduction")
    # dedup keeps the best-scoring row, then the filter narrows it to paper B
    assert nodes
    assert all(n.node.metadata["paper_id"] == PAPER_B for n in nodes)
    assert nodes[0].node.metadata["confidence"] == 0.7


def test_graph_retriever_no_db_no_graph():
    assert DrbrainGraphRetriever(db=None, graph=None).retrieve("q") == []
    assert DrbrainGraphRetriever(db=None, graph=_FakeGraph()).retrieve("q") == []


# ── FusionRetriever ──────────────────────────────────────────────────────────


def test_fusion_rrf_dedup_ordering_and_source_annotation():
    bm25 = _StaticRetriever([_mk_node("p1:0000", score=0.9), _mk_node("p1:0001", score=0.8)])
    vector = _StaticRetriever([_mk_node("p1:0000", score=0.7), _mk_node("p2:0000", score=0.6)])
    fused = FusionRetriever([bm25, vector], sources=["bm25", "vector"], top_k=5)
    out = fused.retrieve("q")

    # p1:0000 appears in both lists → highest RRF; dedup by node_id, not hash.
    assert out[0].node.node_id == "p1:0000"
    assert len(out) == 3
    assert out[0].score > out[1].score >= out[2].score

    top = out[0].node.metadata
    assert top["source"] == "bm25,vector"
    assert top["sources"] == ["bm25", "vector"]
    contrib = top["contributions"]
    assert contrib["bm25"]["rank"] == 1
    assert contrib["vector"]["rank"] == 1
    assert contrib["vector"]["weight"] == 1.0


def test_fusion_weighted_mode():
    a = _StaticRetriever([_mk_node("x", score=1.0)])
    b = _StaticRetriever([_mk_node("y", score=1.0)])
    fused = FusionRetriever(
        [a, b], sources=["a", "b"], mode="weighted", weights={"a": 5.0, "b": 1.0}, top_k=2
    )
    out = fused.retrieve("q")
    assert [n.node.node_id for n in out] == ["x", "y"]  # a weighted 5×


def test_fusion_fault_tolerant():
    bad = _StaticRetriever([], error=RuntimeError("boom"))
    good = _StaticRetriever([_mk_node("z", score=0.5)])
    fused = FusionRetriever([bad, good], sources=["bad", "good"], top_k=5)
    out = fused.retrieve("q")
    assert [n.node.node_id for n in out] == ["z"]
    assert out[0].node.metadata["source"] == "good"


def test_fusion_top_k_and_mode_validation():
    legs = [_StaticRetriever([_mk_node(f"n{i}", score=float(i))]) for i in range(5)]
    fused = FusionRetriever(legs, sources=[f"s{i}" for i in range(5)], top_k=2)
    assert len(fused.retrieve("q")) == 2

    with pytest.raises(ValueError, match="fusion mode"):
        FusionRetriever(legs, mode="nope")
    with pytest.raises(ValueError, match="equal length"):
        FusionRetriever([legs[0]], sources=["a", "b"])


def test_build_fusion_retriever_real_legs(tmp_path):
    """Real T3-built index legs + one custom (mock tree) leg."""
    papers_dir = tmp_path / "papers"
    _write_structured_paper(papers_dir, PAPER_A, _PAPER_A_SECTIONS)
    cfg = _make_cfg(tmp_path, papers_dir)
    db = _PaperDB([PAPER_A])
    embed = _CountingEmbed()
    stats = build_index(cfg, db, embed_model=embed)
    assert stats["nodes"] == 3
    index, bm25 = load_index(cfg, embed_model=embed)
    assert index is not None and bm25 is not None

    tree = _StaticRetriever([_mk_node(f"{PAPER_B}:9999", score=0.5)])  # node only the tree leg sees
    fused = build_fusion_retriever(
        cfg,
        vector_index=index,
        bm25_retriever=bm25,
        custom_retrievers={"tree": tree},
        top_k=20,  # wide window so the tree-only node (rank 1/61 RRF) survives
    )
    assert isinstance(fused, FusionRetriever)
    out = fused.retrieve("drag reduction polymer")
    assert out, "real legs must return fused results"
    # the tree-only node survives fusion with its own annotation
    tree_hit = next(n for n in out if n.node.node_id == f"{PAPER_B}:9999")
    assert tree_hit.node.metadata["sources"] == ["tree"]
    assert tree_hit.node.metadata["source"] == "tree"
    # bm25/vector legs share the index docstore → ids match the built index
    for nws in out:
        if nws.node.node_id == f"{PAPER_B}:9999":
            continue
        assert nws.node.node_id.startswith(f"{PAPER_A}:")
        assert nws.node.metadata["source"], "every fused node must carry a source annotation"
        assert nws.node.metadata["sources"]


def test_build_fusion_retriever_none_without_legs(tmp_path):
    cfg = _make_cfg(tmp_path, REAL_PAPERS)
    assert build_fusion_retriever(cfg) is None
    assert build_fusion_retriever(cfg, custom_retrievers={}) is None


def test_get_retrievers_config_list(tmp_path):
    papers_dir = tmp_path / "papers"
    _write_structured_paper(papers_dir, PAPER_A, _PAPER_A_SECTIONS)
    cfg = _make_cfg(tmp_path, papers_dir)
    build_index(cfg, _PaperDB([PAPER_A]), embed_model=_CountingEmbed())

    retrievers = get_retrievers(cfg)
    assert set(retrievers) == {"bm25", "vector"}

    cfg.llamaindex.retrievers = ["tree"]
    retrievers = get_retrievers(cfg)
    assert set(retrievers) == {"tree"}
    assert isinstance(retrievers["tree"], DrbrainTreeRetriever)

    cfg.llamaindex.retrievers = ["graph"]
    retrievers = get_retrievers(cfg, db=_PaperDB([PAPER_A]), graph=_FakeGraph())
    assert set(retrievers) == {"graph"}
    assert isinstance(retrievers["graph"], DrbrainGraphRetriever)


# ── integration: one live LLM call ───────────────────────────────────────────


@pytest.mark.integration
def test_tree_retriever_real_llm_smoke(tmp_path):
    """Live LLM tree navigation over one real paper (opencode key, no vectors).

    Skips unless ``test-run/config.yaml`` (the opencode test key) is present.
    """
    test_cfg_path = TEST_RUN / "config.yaml"
    if not test_cfg_path.exists():
        pytest.skip("test-run/config.yaml (opencode test key) not present")
    cfg = Config.from_yaml(
        str(test_cfg_path), local_path=test_cfg_path.parent / "config.local.yaml"
    )
    assert cfg.llm.models, "test-run config must define llm.models"
    cfg.dirs.papers = str(REAL_PAPERS)

    retriever = DrbrainTreeRetriever(cfg, paper_id=PAPER_A, top_k=3, db_path=None)
    nodes = retriever.retrieve("Which statements describe the data availability?")
    assert nodes, "real LLM navigation must find sections"
    for nws in nodes:
        assert nws.node.metadata["paper_id"] == PAPER_A
        assert nws.node.metadata["source"] == "tree"
        assert nws.node.text
