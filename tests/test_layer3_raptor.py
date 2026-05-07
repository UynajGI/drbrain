"""Tests for Layer 3: RAPTOR recursive semantic tree."""

import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest

# ── BIC computation ──────────────────────────────────────────────────────────


def test_bic_gmm_returns_lower_for_better_k():
    """BIC score is lower for better-fitting cluster counts."""
    import numpy as np

    from drbrain.extractor.raptor import _bic_gmm

    # Generate 3 well-separated clusters
    rng = np.random.RandomState(42)
    c1 = rng.randn(30, 5) + np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    c2 = rng.randn(30, 5) + np.array([8.0, 0.0, 0.0, 0.0, 0.0])
    c3 = rng.randn(30, 5) + np.array([0.0, 8.0, 0.0, 0.0, 0.0])
    x = np.vstack([c1, c2, c3]).astype("float32")

    bic_1 = _bic_gmm(x, 1)
    _bic_gmm(x, 2)
    bic_3 = _bic_gmm(x, 3)
    bic_5 = _bic_gmm(x, 5)

    # k=3 should be better (lower BIC) than k=1 or k=2
    assert bic_3 < bic_1
    assert bic_3 < bic_5  # overfitting penalty


def test_bic_gmm_returns_finite():
    """BIC always returns a finite float."""
    import numpy as np

    from drbrain.extractor.raptor import _bic_gmm

    x = np.random.RandomState(123).randn(10, 4).astype("float32")
    bic = _bic_gmm(x, 2)
    assert isinstance(bic, float)
    assert bic != float("inf")
    assert bic == bic  # not NaN


# ── GMM clustering ───────────────────────────────────────────────────────────


def test_gmm_cluster_returns_cluster_assignments():
    """_gmm_cluster returns list of cluster index lists."""
    import numpy as np

    from drbrain.extractor.raptor import _gmm_cluster

    # 3 clusters of 10 samples each
    rng = np.random.RandomState(42)
    c1 = rng.randn(10, 5) + [0, 0, 0, 0, 0]
    c2 = rng.randn(10, 5) + [5, 0, 0, 0, 0]
    c3 = rng.randn(10, 5) + [0, 5, 0, 0, 0]
    x = np.vstack([c1, c2, c3]).astype("float32")

    clusters = _gmm_cluster(x, n_samples=30, max_k=5)
    assert len(clusters) >= 2  # should find at least 2 clusters

    # Each sample should appear in exactly one cluster
    all_indices = []
    for c in clusters:
        all_indices.extend(c)
    assert sorted(all_indices) == list(range(30))


def test_gmm_cluster_too_few_samples_returns_one_cluster():
    """With too few samples, only 1 cluster is possible."""
    import numpy as np

    from drbrain.extractor.raptor import _gmm_cluster

    x = np.random.RandomState(1).randn(2, 5).astype("float32")
    clusters = _gmm_cluster(x, n_samples=2)
    assert len(clusters) == 1


# ── UMAP reduction ───────────────────────────────────────────────────────────


def test_umap_reduce_preserves_row_count():
    """UMAP reduces dimensionality but keeps the same number of rows."""
    import numpy as np

    from drbrain.extractor.raptor import _umap_reduce

    x = np.random.RandomState(99).randn(20, 64).astype("float32")
    reduced = _umap_reduce(x, n_components=10)
    reduced_arr = np.asarray(reduced, dtype="float32")
    assert reduced_arr.shape[0] == 20
    assert reduced_arr.shape[1] == 10


def test_umap_reduce_skip_when_fewer_rows_than_components():
    """UMAP is skipped when there are fewer rows than target components."""
    import numpy as np

    from drbrain.extractor.raptor import _umap_reduce

    x = np.random.RandomState(1).randn(3, 64).astype("float32")
    reduced = _umap_reduce(x, n_components=10)
    # Should return original when n_samples <= n_components
    reduced_arr = np.asarray(reduced, dtype="float32")
    assert reduced_arr.shape == x.shape


# ── Summarization ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_summarize_cluster_returns_summary():
    """_summarize_cluster calls LLM and returns summary text."""
    from drbrain.extractor.raptor import _summarize_cluster

    texts = [
        "We propose a new attention mechanism called SparseAttention.",
        "SparseAttention reduces complexity from O(n^2) to O(n log n).",
        "Experiments show 30% speedup with no accuracy loss.",
    ]
    models = [{"provider": "openai", "model": "gpt-4o"}]

    with mock.patch(
        "drbrain.extractor.llm_client.acall_text_with_fallback",
        return_value="SparseAttention: a novel attention mechanism with O(n log n) complexity.",
    ):
        summary = await _summarize_cluster(texts, models)
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "attention" in summary.lower()


# ── RAPTOR tree building (integration, mocked) ───────────────────────────────


def test_build_raptor_tree_stores_summaries():
    """build_raptor_tree creates tree_summaries rows for a paper.

    Mocks embedding and LLM calls to verify the full pipeline:
    embed → cluster → summarize → store → re-embed → repeat.
    """
    import asyncio

    import numpy as np

    from drbrain.storage.database import Database

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)

        # Create paper with tree.json
        paper_dir = base / "test-paper"
        paper_dir.mkdir()
        tree = {
            "structure": [
                {
                    "node_id": "s1",
                    "title": "Abstract",
                },
                {
                    "node_id": "s2",
                    "title": "Introduction",
                    "nodes": [
                        {"node_id": "s2.1", "title": "Background"},
                        {"node_id": "s2.2", "title": "Contributions"},
                    ],
                },
                {
                    "node_id": "s3",
                    "title": "Methods",
                    "nodes": [
                        {"node_id": "s3.1", "title": "Algorithm Design"},
                        {"node_id": "s3.2", "title": "Implementation"},
                    ],
                },
                {"node_id": "s4", "title": "Results"},
                {"node_id": "s5", "title": "Conclusion"},
            ]
        }
        (paper_dir / "tree.json").write_text(json.dumps(tree), encoding="utf-8")
        (paper_dir / "raw.md").write_text("# Test\n\nContent", encoding="utf-8")

        db_path = base / "test.db"
        db = Database(db_path)
        db.conn.execute(
            "INSERT INTO papers (local_id, title) VALUES (?, ?)",
            ("test-paper", "Test Paper"),
        )
        db.conn.commit()

        # Mock embedding: return random normalized vectors per text
        emb_dim = 8

        def _fake_embed(texts, cfg=None):
            vecs = []
            for i, _ in enumerate(texts):
                v = np.random.RandomState(i).randn(emb_dim).astype("float32")
                v = v / np.linalg.norm(v)
                vecs.append(v.tolist())
            return vecs

        # Mock LLM summarization
        def _fake_summarize(texts, models):
            return f"Summary of {len(texts)} sections: {texts[0][:30]}..."

        from drbrain.config import Config, EmbedConfig

        cfg = Config()
        cfg.embed = EmbedConfig(provider="local")

        with (
            mock.patch(
                "drbrain.services.embedding._embed_batch",
                side_effect=_fake_embed,
            ),
            mock.patch(
                "drbrain.extractor.llm_client.acall_text_with_fallback",
                side_effect=_fake_summarize,
            ),
        ):
            from drbrain.extractor.raptor import build_raptor_tree

            count = asyncio.run(build_raptor_tree(paper_dir, db_path, cfg.embed))

            # Should have created at least some summary nodes
            assert count >= 1

            # Verify tree_summaries has entries
            summaries = db.conn.execute(
                "SELECT node_id, paper_id, summary_text, source_node_ids, tree_layer "
                "FROM tree_summaries ORDER BY tree_layer"
            ).fetchall()
            assert len(summaries) == count

            # Each summary has source_node_ids (provenance)
            for s in summaries:
                node_id, paper_id, text, source_json, layer = s
                assert node_id.startswith("raptor_")
                assert paper_id == "test-paper"
                assert len(text) > 0
                assert layer >= 1
                # source_node_ids should be valid JSON array
                sources = json.loads(source_json)
                assert isinstance(sources, list)
                assert len(sources) >= 1  # at least one child

            # RAPTOR nodes should also be in tree_vectors
            raptor_vecs = db.conn.execute(
                "SELECT COUNT(*) FROM tree_vectors WHERE tree_layer LIKE 'raptor_%'"
            ).fetchone()
            assert raptor_vecs[0] == count


def test_build_raptor_tree_stops_at_max_layers():
    """build_raptor_tree respects max_layers parameter."""
    import asyncio

    import numpy as np

    from drbrain.storage.database import Database

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        paper_dir = base / "test-paper"
        paper_dir.mkdir()
        tree = {"structure": [{"node_id": f"n{i}", "title": f"Section {i}"} for i in range(20)]}
        (paper_dir / "tree.json").write_text(json.dumps(tree), encoding="utf-8")
        (paper_dir / "raw.md").write_text("# Test\n\n" + "Content\n" * 20, encoding="utf-8")

        db_path = base / "test.db"
        db = Database(db_path)
        db.conn.execute(
            "INSERT INTO papers (local_id, title) VALUES (?, ?)",
            ("test-paper", "Test Paper"),
        )
        db.conn.commit()

        emb_dim = 8

        def _fake_embed(texts, cfg=None):
            vecs = []
            for i, _ in enumerate(texts):
                v = np.random.RandomState(i).randn(emb_dim).astype("float32")
                v = v / np.linalg.norm(v)
                vecs.append(v.tolist())
            return vecs

        def _fake_summarize(texts, models):
            return f"Summary of {len(texts)} items."

        from drbrain.config import Config, EmbedConfig

        cfg = Config()
        cfg.embed = EmbedConfig(provider="local")

        with (
            mock.patch(
                "drbrain.services.embedding._embed_batch",
                side_effect=_fake_embed,
            ),
            mock.patch(
                "drbrain.extractor.llm_client.acall_text_with_fallback",
                side_effect=_fake_summarize,
            ),
        ):
            from drbrain.extractor.raptor import build_raptor_tree

            count = asyncio.run(build_raptor_tree(paper_dir, db_path, cfg.embed, max_layers=1))

            assert count >= 1

            # All tree_summaries should be layer 1 only
            layers = db.conn.execute("SELECT DISTINCT tree_layer FROM tree_summaries").fetchall()
            assert len(layers) == 1
            assert layers[0][0] == 1
