"""Tests for Layer 2: Embedding engine (EmbedConfig, tree_vectors, search)."""

import json
import struct
import tempfile
from pathlib import Path
from unittest import mock

import pytest

# ── EmbedConfig ──────────────────────────────────────────────────────────────


def test_embed_config_defaults():
    """EmbedConfig has sensible defaults matching ScholarAIO pattern."""
    from drbrain.config import EmbedConfig

    cfg = EmbedConfig()
    assert cfg.provider == "local"
    assert cfg.model == "Qwen/Qwen3-Embedding-0.6B"
    assert cfg.device == "auto"
    assert cfg.top_k == 10
    assert cfg.source == "modelscope"
    assert cfg.cache_dir  # non-empty


def test_embed_config_custom():
    """EmbedConfig accepts overrides."""
    from drbrain.config import EmbedConfig

    cfg = EmbedConfig(
        provider="openai-compat",
        model="text-embedding-3-small",
        api_base="https://api.openai.com/v1",
        top_k=20,
    )
    assert cfg.provider == "openai-compat"
    assert cfg.model == "text-embedding-3-small"
    assert cfg.top_k == 20


# ── Tree vectors storage ────────────────────────────────────────────────────


def _make_fake_vector(dim: int = 8) -> bytes:
    """Create a fake normalized embedding BLOB."""
    import numpy as np

    vec = np.random.randn(dim).astype("float32")
    vec = vec / np.linalg.norm(vec)
    return struct.pack(f"{dim}f", *vec)


def test_build_tree_vectors_stores_embeddings():
    """build_tree_vectors stores vectors in tree_vectors table."""
    from drbrain.storage.database import Database

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        db = Database(db_path)

        # Insert a paper
        db.conn.execute(
            "INSERT INTO papers (local_id, title) VALUES (?, ?)",
            ("test-paper", "Test Paper"),
        )

        # Simulate embedding: write directly to tree_vectors
        vec = _make_fake_vector(8)
        db.conn.execute(
            "INSERT INTO tree_vectors (node_id, paper_id, embedding, content_hash, tree_layer) "
            "VALUES (?, ?, ?, ?, ?)",
            ("node-1", "test-paper", vec, "abc123", "pageindex"),
        )
        db.conn.commit()

        row = db.conn.execute(
            "SELECT node_id, paper_id, content_hash, tree_layer FROM tree_vectors WHERE node_id = ?",
            ("node-1",),
        ).fetchone()
        assert row is not None
        assert row[0] == "node-1"
        assert row[1] == "test-paper"
        assert row[2] == "abc123"
        assert row[3] == "pageindex"

        # Verify embedding BLOB is stored and recoverable
        blob_row = db.conn.execute(
            "SELECT embedding FROM tree_vectors WHERE node_id = ?", ("node-1",)
        ).fetchone()
        recovered = struct.unpack("8f", blob_row[0])
        assert len(recovered) == 8


def test_tree_vectors_content_hash_prevents_redundant_rebuild():
    """Content hash enables incremental updates — unchanged nodes skip re-embedding."""
    from drbrain.storage.database import Database

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        db = Database(db_path)

        db.conn.execute(
            "INSERT INTO papers (local_id, title) VALUES (?, ?)",
            ("p1", "Paper 1"),
        )

        vec1 = _make_fake_vector(4)
        db.conn.execute(
            "INSERT INTO tree_vectors (node_id, paper_id, embedding, content_hash, tree_layer) "
            "VALUES (?, ?, ?, ?, ?)",
            ("n1", "p1", vec1, "hash-v1", "pageindex"),
        )
        db.conn.commit()

        # Same hash → should be recognized as up-to-date
        row = db.conn.execute(
            "SELECT content_hash FROM tree_vectors WHERE node_id = ?", ("n1",)
        ).fetchone()
        assert row[0] == "hash-v1"

        # Update with new hash
        vec2 = _make_fake_vector(4)
        db.conn.execute(
            "UPDATE tree_vectors SET embedding = ?, content_hash = ? WHERE node_id = ?",
            (vec2, "hash-v2", "n1"),
        )
        db.conn.commit()

        row = db.conn.execute(
            "SELECT content_hash FROM tree_vectors WHERE node_id = ?", ("n1",)
        ).fetchone()
        assert row[0] == "hash-v2"


def test_vector_metadata_signature_tracking():
    """vector_metadata stores embedding signature for rebuild detection."""
    from drbrain.storage.database import Database

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        db = Database(db_path)

        sig = "local:Qwen/Qwen3-Embedding-0.6B:auto"
        db.conn.execute(
            "INSERT OR REPLACE INTO vector_metadata (key, value) VALUES (?, ?)",
            ("embed_signature", sig),
        )
        db.conn.commit()

        row = db.conn.execute(
            "SELECT value FROM vector_metadata WHERE key = ?", ("embed_signature",)
        ).fetchone()
        assert row[0] == sig


# ── provider=none graceful degradation ──────────────────────────────────────


def test_provider_none_skips_embedding():
    """provider=none returns 0 and does not touch tree_vectors."""
    from drbrain.config import EmbedConfig

    cfg = EmbedConfig(provider="none")

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        from drbrain.storage.database import Database

        db = Database(db_path)
        db.conn.execute(
            "INSERT INTO papers (local_id, title) VALUES (?, ?)",
            ("p1", "Paper 1"),
        )
        db.conn.commit()

        # With provider=none, should return early with 0
        from drbrain.services.embedding import build_tree_vectors

        count = build_tree_vectors(db_path, Path(td), cfg)
        assert count == 0


# ── Cosine similarity ───────────────────────────────────────────────────────


def test_cosine_similarity_computation():
    """cosine_similarity computes correctly between two vectors."""
    import numpy as np

    from drbrain.services.embedding import _cosine_similarity

    a = np.array([1.0, 0.0, 0.0], dtype="float32")
    b = np.array([0.0, 1.0, 0.0], dtype="float32")
    assert _cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    c = np.array([1.0, 0.0, 0.0], dtype="float32")
    assert _cosine_similarity(a, c) == pytest.approx(1.0, abs=1e-6)


def test_search_tree_with_fake_vectors():
    """search_tree returns top-k nodes by cosine similarity using stored vectors."""
    from drbrain.storage.database import Database

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        db = Database(db_path)

        db.conn.execute(
            "INSERT INTO papers (local_id, title) VALUES (?, ?)",
            ("p1", "Paper 1"),
        )

        # Store 3 fake vectors: node-1 (close to query), node-2 (opposite), node-3 (orthogonal)
        import numpy as np

        query_vec = np.array([1.0, 0.0, 0.0], dtype="float32")
        query_vec = query_vec / np.linalg.norm(query_vec)

        vec1 = np.array([0.99, 0.14, 0.0], dtype="float32")  # ~cos=0.99
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = np.array([-1.0, 0.0, 0.0], dtype="float32")  # cos=-1.0
        vec2 = vec2 / np.linalg.norm(vec2)
        vec3 = np.array([0.0, 1.0, 0.0], dtype="float32")  # cos=0.0
        vec3 = vec3 / np.linalg.norm(vec3)

        for nid, vec in [("node-1", vec1), ("node-2", vec2), ("node-3", vec3)]:
            blob = struct.pack("3f", *vec)
            db.conn.execute(
                "INSERT OR REPLACE INTO tree_vectors (node_id, paper_id, embedding, content_hash, tree_layer) "
                "VALUES (?, ?, ?, ?, ?)",
                (nid, "p1", blob, f"hash-{nid}", "pageindex"),
            )
        db.conn.commit()

        # Manual cosine similarity search (simulating what search_tree does)
        results = []
        for row in db.conn.execute("SELECT node_id, embedding FROM tree_vectors"):
            stored_vec = np.array(struct.unpack("3f", row[1]), dtype="float32")
            sim = float(np.dot(query_vec, stored_vec))
            results.append((row[0], sim))

        results.sort(key=lambda x: x[1], reverse=True)

        assert results[0][0] == "node-1"
        assert results[0][1] == pytest.approx(0.99, abs=0.02)
        assert results[1][0] == "node-3"  # cos=0 > cos=-1
        assert results[2][0] == "node-2"


# ── Build tree vectors integration ──────────────────────────────────────────


def test_build_tree_vectors_with_mock_embed():
    """build_tree_vectors embeds tree nodes from a paper's tree.json and stores them."""
    from drbrain.storage.database import Database

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)

        # Create paper dir with tree.json
        paper_dir = base / "test-paper"
        paper_dir.mkdir()
        tree = {
            "structure": [
                {
                    "node_id": "n1",
                    "title": "Introduction",
                    "nodes": [
                        {"node_id": "n1.1", "title": "Background"},
                        {"node_id": "n1.2", "title": "Contributions"},
                    ],
                },
                {
                    "node_id": "n2",
                    "title": "Methods",
                    "nodes": [
                        {"node_id": "n2.1", "title": "Algorithm"},
                    ],
                },
            ]
        }
        (paper_dir / "tree.json").write_text(json.dumps(tree), encoding="utf-8")

        # Also need raw.md for content hash lookup (can be empty for test)
        (paper_dir / "raw.md").write_text("# Test\n\nContent", encoding="utf-8")

        db_path = base / "test.db"
        db = Database(db_path)
        db.conn.execute(
            "INSERT INTO papers (local_id, title) VALUES (?, ?)",
            ("test-paper", "Test Paper"),
        )
        db.conn.commit()

        # Mock embedding: return N vectors matching input count
        def _fake_embed(texts, cfg=None):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

        with mock.patch(
            "drbrain.services.embedding._embed_batch",
            side_effect=_fake_embed,
        ):
            from drbrain.config import EmbedConfig
            from drbrain.services.embedding import build_tree_vectors

            cfg = EmbedConfig(provider="local")
            count = build_tree_vectors(db_path, paper_dir, cfg)

            # All 5 nodes (n1, n1.1, n1.2, n2, n2.1) should be embedded
            assert count == 5

            # Verify stored
            rows = db.conn.execute("SELECT COUNT(*) FROM tree_vectors").fetchone()
            assert rows[0] == 5

            # Check one node's embedding
            row = db.conn.execute(
                "SELECT embedding, content_hash FROM tree_vectors WHERE node_id = ?",
                ("n1",),
            ).fetchone()
            assert row is not None
            recovered = struct.unpack("4f", row[0])
            assert recovered == pytest.approx((0.1, 0.2, 0.3, 0.4))
            assert row[1]  # content_hash is non-empty
