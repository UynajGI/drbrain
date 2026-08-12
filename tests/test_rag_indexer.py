"""T3 indexer tests: tree.json/raw.md → LlamaIndex Documents/Nodes + indexes.

Covers :mod:`drbrain.rag.indexer`:

* :func:`collect_tree_nodes` — real test-run paper + synthetic line-range tree
* :func:`build_index` — full build, incremental (content_hash), ``--paper``
  subset carry-over, ``--force`` rebuild
* :func:`load_index` — restore from disk, retrieve, missing-storage fallback

Embedding uses a deterministic fake (:class:`_CountingEmbed`) so the suite
needs no GPU/network. A real-model smoke test is marked ``integration``.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from drbrain.config import Config, DirsConfig, EmbedConfig, LlamaIndexConfig
from drbrain.rag.indexer import MANIFEST_NAME, build_index, collect_tree_nodes, load_index

_HAS_LLAMA_INDEX = importlib.util.find_spec("llama_index") is not None

if _HAS_LLAMA_INDEX:
    from llama_index.core.embeddings import BaseEmbedding
    from pydantic import PrivateAttr

TEST_RUN = Path(__file__).resolve().parents[1] / "test-run"
REAL_PAPERS = TEST_RUN / "papers"
PAPER_A = "10.1002_adma.202308655"  # 3 tree nodes
PAPER_B = "10.3390_ma15134622"  # 4 tree nodes

pytestmark = pytest.mark.skipif(not _HAS_LLAMA_INDEX, reason="llama_index not installed")


# ── Helpers ──────────────────────────────────────────────────────────────────


class _CountingEmbed(BaseEmbedding):
    """Deterministic embed adapter that records every text it embeds."""

    _embedded_texts: list[str] = PrivateAttr(default_factory=list)

    def __init__(self) -> None:
        super().__init__(model_name="fake-embed", embed_batch_size=8)

    @property
    def embedded_texts(self) -> list[str]:
        return self._embedded_texts

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


def _make_cfg(tmp_path: Path, papers_dir: Path) -> Config:
    return Config(
        llamaindex=LlamaIndexConfig(
            enabled=True, vector_store="memory", storage_dir=str(tmp_path / "li")
        ),
        dirs=DirsConfig(papers=str(papers_dir)),
        embed=EmbedConfig(provider="none", model="fake-embed", top_k=5),
    )


def _write_synthetic_paper(papers_dir: Path, pid: str) -> None:
    """tree.json with line_num only (no inline text) + matching raw.md."""
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


# ── collect_tree_nodes ───────────────────────────────────────────────────────


def test_collect_tree_nodes_real_paper():
    docs = collect_tree_nodes(REAL_PAPERS / PAPER_A)
    assert len(docs) > 0
    for doc in docs:
        assert doc.id_.startswith(f"{PAPER_A}:")
        assert doc.metadata["paper_id"] == PAPER_A
        assert doc.metadata["node_id"]
        assert doc.metadata["title"]
        assert doc.metadata["tree_layer"] == "pageindex"
        assert doc.metadata["line_start"] is not None
        assert doc.metadata["line_end"] is not None
        assert doc.text.startswith(doc.metadata["title"])
    # ids unique across the paper
    ids = [d.id_ for d in docs]
    assert len(set(ids)) == len(ids)


def test_collect_tree_nodes_from_dict_with_line_ranges(tmp_path):
    raw = "line0\n# Section A\ncontent A1\ncontent A2\n# Section B\ncontent B\n"
    paper_dir = tmp_path / "paper1"
    paper_dir.mkdir()
    (paper_dir / "raw.md").write_text(raw, encoding="utf-8")
    tree = {
        "structure": [
            {"title": "Section A", "node_id": "0000", "line_num": 2, "nodes": []},
            {"title": "Section B", "node_id": "0001", "line_num": 5, "nodes": []},
        ]
    }
    docs = collect_tree_nodes(paper_dir, tree)
    assert [d.metadata["node_id"] for d in docs] == ["0000", "0001"]
    # 0-based range [line_num-1, next_line_num-1): line 2 → line 4
    assert docs[0].text == "Section A\n# Section A\ncontent A1\ncontent A2"
    assert docs[0].metadata["line_start"] == 1
    assert docs[0].metadata["line_end"] == 4
    # last node runs to EOF (split("\n") yields 7 lines incl. trailing "")
    assert docs[1].text == "Section B\n# Section B\ncontent B"
    assert docs[1].metadata["line_end"] == 7
    # explicit line_start/line_end wins over line_num
    tree2 = {
        "structure": [
            {
                "title": "S",
                "node_id": "0000",
                "line_num": 2,
                "line_start": 4,
                "line_end": 6,
                "nodes": [],
            }
        ]
    }
    docs2 = collect_tree_nodes(paper_dir, tree2)
    assert docs2[0].metadata["line_start"] == 4
    assert docs2[0].metadata["line_end"] == 6
    assert docs2[0].text == "S\n# Section B\ncontent B"


def test_collect_tree_nodes_falls_back_to_inline_text(tmp_path):
    paper_dir = tmp_path / "paper1"
    paper_dir.mkdir()
    tree = {"structure": [{"title": "T", "node_id": "0000", "text": "inline body text"}]}
    docs = collect_tree_nodes(paper_dir, tree)  # no raw.md present
    assert len(docs) == 1
    assert docs[0].text == "T\ninline body text"
    assert docs[0].metadata["line_start"] is None
    assert docs[0].metadata["line_end"] is None


def test_collect_tree_nodes_missing_tree_json(tmp_path):
    assert collect_tree_nodes(tmp_path / "nope") == []


# ── build_index / load_index ─────────────────────────────────────────────────


def test_build_index_full_and_load_retrieve(tmp_path):
    cfg = _make_cfg(tmp_path, REAL_PAPERS)
    db = _PaperDB([PAPER_A, PAPER_B])
    embed = _CountingEmbed()

    stats = build_index(cfg, db, embed_model=embed)
    # PAPER_A: 3 small nodes → 3 chunks. PAPER_B: 3 small + 1 oversized
    # Abstract (34,326 chars > 4000 tokens ≈ 16000 chars) → 3 chunks.
    assert stats["papers"] == 2
    assert stats["nodes"] == 9  # 3 + (3 + 3 chunks)
    assert stats["chunked"] == 1  # only B's Abstract was split
    assert stats["embedded"] == 9
    assert stats["unchanged"] == 0
    assert stats["bm25_nodes"] == 9
    assert stats["carried"] == 0
    assert (tmp_path / "li" / MANIFEST_NAME).exists()

    index, bm25 = load_index(cfg, embed_model=embed)
    assert index is not None
    assert bm25 is not None
    assert len(index.docstore.docs) == 9

    # every indexed node is bounded below the chunk cap (+title slack)
    max_chars = 4000 * 4
    for node in index.docstore.docs.values():
        assert len(node.text) <= max_chars + 2000, f"oversized node {node.node_id}"

    hits = index.as_retriever(similarity_top_k=3).retrieve("drag reduction polymer")
    assert hits, "vector retrieval must return hits"
    assert all(h.node_id.startswith((f"{PAPER_A}:", f"{PAPER_B}:")) for h in hits)

    bm25_hits = bm25.retrieve("nanoparticle synthesis")
    assert bm25_hits, "BM25 retrieval must return hits"
    assert all(n.node_id in index.docstore.docs for n in bm25_hits)


def test_build_index_no_chunking_when_cap_disabled(tmp_path):
    """A huge max_node_tokens keeps the 1:1 node↔Document mapping."""
    cfg = _make_cfg(tmp_path, REAL_PAPERS)
    db = _PaperDB([PAPER_B])
    embed = _CountingEmbed()
    stats = build_index(cfg, db, embed_model=embed, max_node_tokens=100_000)
    assert stats["nodes"] == 4  # 4 tree nodes, none split
    assert stats["chunked"] == 0


def test_build_index_chunk_metadata_and_sizes(tmp_path):
    cfg = _make_cfg(tmp_path, REAL_PAPERS)
    db = _PaperDB([PAPER_B])
    embed = _CountingEmbed()
    stats = build_index(cfg, db, embed_model=embed)
    assert stats["chunked"] == 1

    index, _ = load_index(cfg, embed_model=embed)
    # B's Abstract (node_id 0000) spans chunks #0..#2 with parent metadata kept
    chunks = sorted(
        (
            n
            for n in index.docstore.docs.values()
            if n.metadata["paper_id"] == PAPER_B and n.metadata["node_id"] == "0000"
        ),
        key=lambda n: n.metadata["chunk_index"],
    )
    assert len(chunks) == 3
    assert [n.metadata["chunk_index"] for n in chunks] == [0, 1, 2]
    assert all(n.metadata["chunk_count"] == 3 for n in chunks)
    assert all(n.metadata["title"] == chunks[0].metadata["title"] for n in chunks)
    assert all(n.metadata["line_start"] == chunks[0].metadata["line_start"] for n in chunks)
    assert chunks[0].node_id == f"{PAPER_B}:0000#0"
    assert chunks[2].node_id == f"{PAPER_B}:0000#2"
    # chunk text re-prefixes the section title
    assert chunks[2].text.startswith(chunks[2].metadata["title"])
    # concatenation of chunk bodies reconstructs the parent content (minus
    # hard-sliced overlong paragraphs, if any)
    joined = "\n\n".join(n.text[len(n.metadata["title"]) + 1 :].strip() for n in chunks)
    assert joined


def test_collect_tree_nodes_max_node_tokens_splits(tmp_path):
    """collect_tree_nodes honours max_node_tokens directly (T9 param)."""
    raw = "line0\n# Big\n" + ("para one content\n\n" * 600)
    paper_dir = tmp_path / "big"
    paper_dir.mkdir()
    (paper_dir / "raw.md").write_text(raw, encoding="utf-8")
    tree = {
        "structure": [
            {"title": "Big", "node_id": "0000", "line_num": 2, "nodes": []},
        ]
    }
    docs = collect_tree_nodes(paper_dir, tree, max_node_tokens=1000)
    assert len(docs) > 1, "oversized node must be split"
    for d in docs:
        assert d.id_.startswith("big:0000#")
        assert d.metadata["chunk_count"] == len(docs)
        assert len(d.text) <= 1000 * 4 + 2000
    # without the cap: single document
    single = collect_tree_nodes(paper_dir, tree)
    assert len(single) == 1
    assert single[0].id_ == "big:0000"


def test_build_index_incremental_with_chunked_node(tmp_path):
    """Incremental build: an unchanged chunked node reuses all chunk embeds."""
    papers_dir = tmp_path / "papers"
    _write_synthetic_paper(papers_dir, "paper1")
    # inflate the last node's body (Methods runs to EOF) so it crosses the
    # 8000-token (32K char) cap
    raw_path = papers_dir / "paper1" / "raw.md"
    raw_path.write_text(
        raw_path.read_text(encoding="utf-8") + ("chunk filler text\n\n" * 3000),
        encoding="utf-8",
    )
    cfg = _make_cfg(tmp_path, papers_dir)
    db = _PaperDB(["paper1"])
    embed = _CountingEmbed()

    stats1 = build_index(cfg, db, embed_model=embed)
    assert stats1["chunked"] == 1
    embedded_before = len(embed.embedded_texts)
    stats2 = build_index(cfg, db, embed_model=embed)
    assert len(embed.embedded_texts) == embedded_before  # nothing re-embedded
    assert stats2["embedded"] == 0
    assert stats2["unchanged"] == 2  # Intro + Methods (chunked) parents unchanged


def test_build_index_incremental(tmp_path):
    papers_dir = tmp_path / "papers"
    _write_synthetic_paper(papers_dir, "paper1")
    cfg = _make_cfg(tmp_path, papers_dir)
    db = _PaperDB(["paper1"])
    embed = _CountingEmbed()

    stats1 = build_index(cfg, db, embed_model=embed)
    assert stats1["embedded"] == 2  # Intro + Methods

    # Modify raw.md content inside node 0001's range → only that node changes.
    raw = papers_dir / "paper1" / "raw.md"
    raw.write_text(
        raw.read_text(encoding="utf-8").replace(
            "solvothermal nanoparticle synthesis",
            "solvothermal synthesis of zinc oxide nanoparticles",
        ),
        encoding="utf-8",
    )
    embedded_before = len(embed.embedded_texts)
    stats2 = build_index(cfg, db, embed_model=embed)
    assert len(embed.embedded_texts) - embedded_before == 1
    assert stats2["embedded"] == 1
    assert stats2["unchanged"] == 1
    assert stats2["nodes"] == 2

    # Unchanged node kept its cached embedding; retrieval still works.
    index, _ = load_index(cfg, embed_model=embed)
    assert len(index.docstore.docs) == 2
    assert index.as_retriever(similarity_top_k=2).retrieve("turbulent channel flow")


def test_build_index_incremental_no_changes_reembeds_nothing(tmp_path):
    papers_dir = tmp_path / "papers"
    _write_synthetic_paper(papers_dir, "paper1")
    cfg = _make_cfg(tmp_path, papers_dir)
    db = _PaperDB(["paper1"])
    embed = _CountingEmbed()

    build_index(cfg, db, embed_model=embed)
    embedded_before = len(embed.embedded_texts)
    stats = build_index(cfg, db, embed_model=embed)
    assert len(embed.embedded_texts) == embedded_before  # nothing re-embedded
    assert stats["embedded"] == 0
    assert stats["unchanged"] == 2


def test_build_index_force_reembeds_all(tmp_path):
    papers_dir = tmp_path / "papers"
    _write_synthetic_paper(papers_dir, "paper1")
    cfg = _make_cfg(tmp_path, papers_dir)
    db = _PaperDB(["paper1"])
    embed = _CountingEmbed()

    build_index(cfg, db, embed_model=embed)
    embedded_before = len(embed.embedded_texts)
    stats = build_index(cfg, db, embed_model=embed, force=True)
    assert len(embed.embedded_texts) - embedded_before == 2
    assert stats["embedded"] == 2
    assert stats["unchanged"] == 0


def test_build_index_paper_subset_carries_other_papers(tmp_path):
    cfg = _make_cfg(tmp_path, REAL_PAPERS)
    db = _PaperDB([PAPER_A, PAPER_B])
    embed = _CountingEmbed()

    build_index(cfg, db, embed_model=embed)  # full index: A + B (9 chunk nodes)
    embedded_before = len(embed.embedded_texts)
    stats = build_index(cfg, db, paper_ids=[PAPER_B], embed_model=embed)

    # Only B's 4 parents are re-collected; A's 3 nodes are carried over
    # untouched. B's nodes are unchanged (hash match, incl. the chunked
    # Abstract) → nothing re-embedded.
    assert stats["carried"] == 3
    assert stats["papers"] == 1
    assert stats["nodes"] == 9
    assert stats["embedded"] == 0
    assert stats["unchanged"] == 4
    assert len(embed.embedded_texts) - embedded_before == 0

    index, bm25 = load_index(cfg, embed_model=embed)
    assert len(index.docstore.docs) == 9  # A still present
    assert len(bm25.corpus) == 9
    # manifest still records paper A
    manifest = json.loads((tmp_path / "li" / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert any(pid.startswith(PAPER_A) for pid in manifest["papers"])


def test_load_index_without_storage_returns_none(tmp_path):
    cfg = _make_cfg(tmp_path, REAL_PAPERS)
    index, bm25 = load_index(cfg, embed_model=_CountingEmbed())
    assert index is None
    assert bm25 is None


def test_drbrain_embedding_keeps_cfg_reference():
    """T3 regression: pydantic init wiped the mixin's _cfg, breaking _embed."""
    from drbrain.rag.llm import DrbrainEmbedding

    cfg = _make_cfg(Path("."), REAL_PAPERS)
    emb = DrbrainEmbedding(cfg)
    assert emb._cfg is cfg
    assert emb.model_name == cfg.embed.model
    # _embed_one without touching the real model: provider=none → empty vec
    assert emb._embed_one("anything") == []


def test_build_index_missing_paper_dir_skips(tmp_path):
    cfg = _make_cfg(tmp_path, REAL_PAPERS)
    db = _PaperDB([PAPER_A])
    stats = build_index(
        cfg, db, paper_ids=["does-not-exist", PAPER_A], embed_model=_CountingEmbed()
    )
    assert stats["papers"] == 1  # missing dir skipped
    assert stats["nodes"] == 3


@pytest.mark.integration
def test_build_index_real_embedding_smoke(tmp_path):
    """Real embed provider (Qwen3-Embedding-0.6B) over one small paper."""
    from drbrain.rag.indexer import _default_embed_model

    cfg = _make_cfg(tmp_path, REAL_PAPERS)
    cfg.embed = EmbedConfig(provider="local", model="Qwen/Qwen3-Embedding-0.6B")
    db = _PaperDB([PAPER_A])
    stats = build_index(cfg, db, embed_model=_default_embed_model(cfg))
    assert stats["nodes"] == 3
    index, bm25 = load_index(cfg, embed_model=_default_embed_model(cfg))
    assert index.as_retriever(similarity_top_k=2).retrieve("drag reduction")
