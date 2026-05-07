"""Tests for Layer 4: Tree retrieval v2 (LLM-primary hybrid, cross-paper, scoring)."""

import json
import struct
import tempfile
from pathlib import Path
from unittest import mock

# ── Helper ───────────────────────────────────────────────────────────────────


def _setup_paper_with_vectors(base: Path, paper_id: str, node_data: list[dict]):
    """Create a paper dir with tree.json and populate tree_vectors."""
    import numpy as np

    from drbrain.storage.database import Database

    paper_dir = base / paper_id
    paper_dir.mkdir()

    tree = {"structure": node_data}
    (paper_dir / "tree.json").write_text(json.dumps(tree), encoding="utf-8")
    (paper_dir / "raw.md").write_text("# Test\n\nContent.", encoding="utf-8")

    db_path = base / "test.db"
    db = Database(db_path)
    db.conn.execute(
        "INSERT OR IGNORE INTO papers (local_id, title) VALUES (?, ?)",
        (paper_id, f"Paper {paper_id}"),
    )
    db.conn.commit()

    # Store fake vectors
    for i, node in enumerate(node_data):
        vec = np.zeros(8, dtype="float32")
        vec[i % 8] = 1.0
        vec = vec / np.linalg.norm(vec)
        blob = struct.pack("8f", *vec)
        db.conn.execute(
            "INSERT OR REPLACE INTO tree_vectors "
            "(node_id, paper_id, embedding, content_hash, tree_layer) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                node["node_id"],
                paper_id,
                blob,
                f"hash-{node['node_id']}",
                "pageindex",
            ),
        )
    db.conn.commit()

    return db_path


# ── Cross-paper retrieval ───────────────────────────────────────────────────


def test_query_cross_paper_returns_results_from_multiple_papers():
    """query_cross_paper searches across all papers in tree_vectors."""
    import numpy as np

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)

        nodes1 = [
            {"node_id": "p1_s1", "title": "Attention Mechanisms"},
            {"node_id": "p1_s2", "title": "Transformer Architecture"},
        ]
        db_path = _setup_paper_with_vectors(base, "paper-1", nodes1)

        nodes2 = [
            {"node_id": "p2_s1", "title": "Graph Neural Networks"},
            {"node_id": "p2_s2", "title": "Attention in GNNs"},
        ]
        _setup_paper_with_vectors(base, "paper-2", nodes2)

        from drbrain.config import EmbedConfig
        from drbrain.query.tree_retrieval import query_cross_paper

        with mock.patch(
            "drbrain.services.embedding._embed_batch",
            return_value=[np.zeros(8, dtype="float32").tolist()],
        ):
            results = query_cross_paper(
                "attention mechanism",
                db_path,
                top_k=3,
                cfg=EmbedConfig(provider="local"),
            )

        assert len(results) >= 1
        assert len(results) <= 3
        for r in results:
            assert "node_id" in r
            assert "paper_id" in r
            assert "score" in r


def test_query_cross_paper_empty_db_returns_empty():
    """Empty tree_vectors returns empty list, not error."""
    from drbrain.storage.database import Database

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        Database(db_path)

        from drbrain.config import EmbedConfig
        from drbrain.query.tree_retrieval import query_cross_paper

        results = query_cross_paper("test", db_path, cfg=EmbedConfig(provider="local"))
        assert results == []


def test_query_cross_paper_provider_none_returns_empty():
    """provider=none returns empty (vector-free path)."""
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        nodes = [{"node_id": "n1", "title": "Test"}]
        db_path = _setup_paper_with_vectors(base, "paper-1", nodes)

        from drbrain.config import EmbedConfig
        from drbrain.query.tree_retrieval import query_cross_paper

        results = query_cross_paper("test", db_path, cfg=EmbedConfig(provider="none"))
        assert results == []


# ── Hybrid scoring ──────────────────────────────────────────────────────────


def test_hybrid_score_weighted_sum():
    """_hybrid_score combines BM25 and vector scores with alpha weight."""
    from drbrain.query.tree_retrieval import _hybrid_score

    bm25_results = [
        {"id": "a", "bm25_score": 10.0},
        {"id": "b", "bm25_score": 5.0},
        {"id": "c", "bm25_score": 1.0},
    ]
    vector_results = [
        {"id": "b", "score": 0.9},
        {"id": "c", "score": 0.8},
        {"id": "a", "score": 0.3},
    ]

    merged = _hybrid_score(bm25_results, vector_results, bm25_key="bm25_score", alpha=0.5)
    assert len(merged) == 3
    assert merged[0]["id"] == "b"  # strongest combined


def test_hybrid_score_missing_in_vector():
    """Items only in BM25 still appear in hybrid results."""
    from drbrain.query.tree_retrieval import _hybrid_score

    bm25_results = [
        {"id": "a", "bm25_score": 8.0},
        {"id": "b", "bm25_score": 6.0},
    ]
    vector_results = [
        {"id": "a", "score": 0.9},
    ]

    merged = _hybrid_score(bm25_results, vector_results, bm25_key="bm25_score", alpha=0.5)
    assert len(merged) == 2
    assert merged[1]["id"] == "b"


def test_rrf_score():
    """Reciprocal Rank Fusion combines multiple ranked lists."""
    from drbrain.query.tree_retrieval import _rrf_score

    list_a = ["x", "y", "z"]
    list_b = ["y", "z", "x"]
    list_c = ["z", "x", "y"]

    merged = _rrf_score([list_a, list_b, list_c], k=60)
    assert len(merged) == 3
    scores = [s for _, s in merged]
    assert max(scores) - min(scores) < 0.01


# ── LLM-primary hybrid retrieval ────────────────────────────────────────────


def test_query_by_structure_hybrid_llm_primary():
    """Hybrid mode: LLM navigation is PRIMARY, vectors augment."""
    import asyncio

    import numpy as np

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        nodes = [
            {
                "node_id": "n1",
                "title": "Introduction to Attention",
                "nodes": [
                    {"node_id": "n1.1", "title": "Self-Attention"},
                    {"node_id": "n1.2", "title": "Multi-Head"},
                ],
            },
            {"node_id": "n2", "title": "Results"},
        ]
        db_path = _setup_paper_with_vectors(base, "test-paper", nodes)
        paper_dir = base / "test-paper"

        from drbrain.config import EmbedConfig
        from drbrain.query.tree_retrieval import query_by_structure_hybrid

        qv = np.zeros(8, dtype="float32")
        qv[1] = 1.0
        qv = qv / np.linalg.norm(qv)

        llm_response = '{"nodes": [{"node_id": "n1.1", "reason": "relevant"}]}'

        with (
            mock.patch(
                "drbrain.services.embedding._embed_batch",
                return_value=[qv.tolist()],
            ),
            mock.patch(
                "drbrain.query.tree_retrieval.acall_with_fallback",
                return_value=llm_response,
            ),
        ):
            sections = asyncio.run(
                query_by_structure_hybrid(
                    "attention",
                    paper_dir,
                    db_path,
                    models=[{"provider": "openai", "model": "gpt-4o"}],
                    cfg=EmbedConfig(provider="local"),
                    top_k=2,
                )
            )

        assert sections is not None
        assert len(sections) >= 1
        for s in sections:
            assert "node_id" in s
            assert "title" in s
            assert "source" in s  # llm, vector, or llm+vector
        # LLM-selected node should be present
        assert any(s["node_id"] == "n1.1" for s in sections)


def test_query_by_structure_hybrid_no_vectors_still_llm():
    """Without vectors, LLM navigation still works (not a 'fallback' — the default)."""
    import asyncio

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        paper_dir = base / "test-paper"
        paper_dir.mkdir()
        tree = {
            "structure": [
                {"node_id": "n1", "title": "Introduction"},
                {"node_id": "n2", "title": "Methods"},
            ]
        }
        (paper_dir / "tree.json").write_text(json.dumps(tree), encoding="utf-8")
        (paper_dir / "raw.md").write_text(
            "# Intro\n\nSome text.\n\n# Methods\n\nMore text.", encoding="utf-8"
        )

        from drbrain.storage.database import Database

        db_path = base / "test.db"
        db = Database(db_path)
        db.conn.execute(
            "INSERT INTO papers (local_id, title) VALUES (?, ?)",
            ("test-paper", "Test"),
        )
        db.conn.commit()

        from drbrain.query.tree_retrieval import query_by_structure_hybrid

        llm_response = '{"nodes": [{"node_id": "n1", "reason": "matches query"}]}'

        with mock.patch(
            "drbrain.query.tree_retrieval.acall_with_fallback",
            return_value=llm_response,
        ):
            sections = asyncio.run(
                query_by_structure_hybrid(
                    "introduction",
                    paper_dir,
                    db_path,
                    models=[{"provider": "openai", "model": "gpt-4o"}],
                    cfg=None,  # No vectors at all
                )
            )

        assert sections is not None
        assert len(sections) >= 1
        assert sections[0]["source"] == "llm"  # pure LLM, no vector augmentation
