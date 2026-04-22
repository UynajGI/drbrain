"""Integration test for the ingest pipeline database layer."""
import tempfile
from pathlib import Path
from brbrain.storage.database import Database
from brbrain.graph.engine import GraphEngine

def test_ingest_pipeline_with_mock_data():
    """Full pipeline: create paper, insert concepts, edges, verify DB state."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        db = Database(str(db_path))
        graph = GraphEngine()

        # Insert a paper
        db.insert_paper("p1", "Test Paper", 2024, "uploaded")
        db.insert_paper_ids("p1", doi="10.1234/test")
        db.commit()

        # Insert concepts
        db.insert_concept("p1", "Problem", "long-range dependency", 0.9)
        db.insert_concept("p1", "Method", "attention mechanism", 0.95)
        db.commit()

        # Insert edge
        db.insert_edge("attention mechanism", "long-range dependency", "addresses", "p1")
        db.commit()

        # Verify
        concepts = db.conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
        assert concepts == 2

        edges = db.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        assert edges == 1

        # Verify graph engine can load from DB
        graph.load_from_db(db)
        assert graph.graph.number_of_nodes() >= 2

        db.close()
