"""Tests for OKF (Open Knowledge Format) export."""

from __future__ import annotations

import re

import pytest

from drbrain.graph.engine import GraphEngine
from drbrain.storage.database import Database
from drbrain.storage.okf_export import _slugify, export_okf


@pytest.fixture
def populated_db():
    """In-memory DB with two papers, three concepts, one argument, one edge."""
    db = Database(":memory:")
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status, abstract) "
        "VALUES ('p1', 'Attention Is All You Need', 2017, 'extracted', 'We propose transformers.')"
    )
    db.conn.execute(
        "INSERT INTO papers (local_id, title, year, status) "
        "VALUES ('p2', 'ResNet', 2016, 'extracted')"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, first_seen, last_seen) "
        "VALUES ('p1', 'Method', 'transformer', 0.95, 2017, 2017)"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, first_seen, last_seen) "
        "VALUES ('p1', 'Problem', 'long-range dependency', 0.9, 2017, 2018)"
    )
    db.conn.execute(
        "INSERT INTO concepts (local_id, type, label, confidence, first_seen, last_seen) "
        "VALUES ('p2', 'Method', 'resnet', 0.92, 2016, 2016)"
    )
    db.conn.execute(
        "INSERT INTO arguments (source_paper, claim, claim_type, target_label, target_type, "
        "evidence_type, mechanism, confidence) "
        "VALUES ('p2', 'ResNet enables deeper nets', 'supports', 'resnet', 'Method', "
        "'empirical', 'skip connections ease gradient flow', 0.9)"
    )
    db.commit()
    yield db
    db.close()


@pytest.fixture
def populated_graph():
    """GraphEngine with cross-concept edges."""
    g = GraphEngine()
    g.add_edge(
        "transformer", "long-range dependency", relation="solves", source_paper="p1", weight=1.0
    )
    g.add_edge("resnet", "transformer", relation="extends", source_paper="p2", weight=0.8)
    return g


# ── slugify ───────────────────────────────────────────────────────────────


class TestSlugify:
    def test_handles_special_chars(self):
        """Spaces and slashes collapse to hyphens; lowercased."""
        seen: set[str] = set()
        assert _slugify("Long-Range Dependency", seen) == "long-range-dependency"
        seen2: set[str] = set()
        assert _slugify("a/b\\c", seen2) == "a-b-c"
        seen3: set[str] = set()
        assert _slugify("UPPER", seen3) == "upper"

    def test_collision_dedup(self):
        """Two different labels with the same slug get -2 suffix on the second."""
        seen: set[str] = set()
        s1 = _slugify("Transformer", seen)
        s2 = _slugify("Transformer!", seen)  # same slug after sanitize
        assert s1 == "transformer"
        assert s2 == "transformer-2"

    def test_empty_label_fallback(self):
        seen: set[str] = set()
        assert _slugify("", seen) == "concept"


# ── export_okf ────────────────────────────────────────────────────────────


class TestExportOkf:
    def test_concept_md_has_frontmatter_and_type(self, populated_graph, populated_db, tmp_path):
        """Each concept .md has parseable YAML frontmatter with a non-empty type."""
        export_okf(populated_graph, populated_db, tmp_path / "bundle")
        md = (tmp_path / "bundle" / "concepts" / "method" / "transformer.md").read_text()
        # Frontmatter delimited by ---
        assert md.startswith("---\n")
        fm_end = md.index("\n---\n", 4)
        fm = md[4:fm_end]
        assert re.search(r"^type:\s*\S+", fm, re.MULTILINE), "type field missing or empty"
        assert "title:" in fm

    def test_concept_md_has_cross_links(self, populated_graph, populated_db, tmp_path):
        """Edges render as markdown links to the target concept file."""
        export_okf(populated_graph, populated_db, tmp_path / "bundle")
        md = (tmp_path / "bundle" / "concepts" / "method" / "transformer.md").read_text()
        # transformer solves long-range dependency → link to that concept
        assert "long-range-dependency.md" in md
        assert "**solves**" in md
        # resnet extends transformer → incoming link from resnet
        assert "resnet.md" in md

    def test_arguments_render_in_body(self, populated_graph, populated_db, tmp_path):
        """Arguments appear under a # Arguments section in the concept body."""
        export_okf(populated_graph, populated_db, tmp_path / "bundle")
        md = (tmp_path / "bundle" / "concepts" / "method" / "resnet.md").read_text()
        assert "## Arguments" in md or "Arguments" in md
        assert "ResNet enables deeper nets" in md
        assert "**supports**" in md

    def test_paper_md_has_metadata_and_concepts(self, populated_graph, populated_db, tmp_path):
        """Paper .md has frontmatter + abstract + concept listing."""
        export_okf(populated_graph, populated_db, tmp_path / "bundle")
        md = (tmp_path / "bundle" / "papers" / "p1.md").read_text()
        assert md.startswith("---\n")
        assert "Attention Is All You Need" in md
        assert "transformers." in md  # abstract
        # Concepts section lists this paper's concepts
        assert "transformer" in md

    def test_index_md_lists_concepts_by_type(self, populated_graph, populated_db, tmp_path):
        """Root index.md groups concepts by type and lists them."""
        export_okf(populated_graph, populated_db, tmp_path / "bundle")
        idx = (tmp_path / "bundle" / "index.md").read_text()
        assert "## Concepts by Type" in idx
        assert "Method" in idx
        assert "Problem" in idx
        # Links to concept files present
        assert "transformer.md" in idx or "concepts/method/transformer" in idx

    def test_stats_returned(self, populated_graph, populated_db, tmp_path):
        """export_okf returns a stats dict with expected keys."""
        stats = export_okf(populated_graph, populated_db, tmp_path / "bundle")
        assert stats["concepts"] == 3
        assert stats["papers"] == 2
        assert stats["edges"] == 2
        assert stats["arguments"] == 1

    def test_okf_conformance_every_md_has_type(self, populated_graph, populated_db, tmp_path):
        """OKF §9: every non-index .md has frontmatter with non-empty type."""
        import yaml

        export_okf(populated_graph, populated_db, tmp_path / "bundle")
        bundle = tmp_path / "bundle"
        md_files = [
            p
            for p in bundle.rglob("*.md")
            if p.name != "index.md"  # index has no frontmatter per OKF §6
        ]
        assert len(md_files) >= 4  # 3 concepts + 2 papers - at least
        for md_path in md_files:
            text = md_path.read_text()
            assert text.startswith("---\n"), f"{md_path.name} missing frontmatter"
            fm_end = text.index("\n---\n", 4)
            fm = yaml.safe_load(text[4:fm_end])
            assert isinstance(fm, dict), f"{md_path.name} frontmatter not a dict"
            assert fm.get("type"), f"{md_path.name} missing/empty type field"

    def test_paper_filter_restricts_concepts(self, populated_graph, populated_db, tmp_path):
        """paper_ids filter exports only that paper's concepts."""
        export_okf(populated_graph, populated_db, tmp_path / "bundle", paper_ids=["p2"])
        bundle = tmp_path / "bundle"
        # p2 owns 'resnet' only; transformer/long-range belong to p1
        assert (bundle / "concepts" / "method" / "resnet.md").exists()
        assert not (bundle / "concepts" / "method" / "transformer.md").exists()
        # Papers filtered too
        assert (bundle / "papers" / "p2.md").exists()
        assert not (bundle / "papers" / "p1.md").exists()

    def test_broken_link_tolerated(self, populated_db, tmp_path):
        """An edge to a concept not in the DB renders as plain text (OKF §5.3)."""
        g = GraphEngine()
        # Edge references a concept that has no concept row
        g.add_edge("resnet", "ghost-concept", relation="extends", source_paper="p2", weight=1.0)
        export_okf(g, populated_db, tmp_path / "bundle")
        md = (tmp_path / "bundle" / "concepts" / "method" / "resnet.md").read_text()
        # ghost-concept has no file, so it should appear as plain text, not a link
        assert "ghost-concept" in md
        assert "ghost-concept.md" not in md
