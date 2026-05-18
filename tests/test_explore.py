"""Tests for explore silos — literature discovery collections.

TDD: tests written before implementation.
"""

from __future__ import annotations

import pytest


class TestExploreStore:
    """Test the explore store layer."""

    def test_module_imports(self):
        from drbrain.storage import explore

        assert explore is not None

    def test_create_silo(self, tmp_path):
        from drbrain.storage.explore import (
            create_explore_silo,
            list_explore_silos,
        )

        create_explore_silo(tmp_path, "turbulence", description="Turbulence papers")

        silos = list_explore_silos(tmp_path)
        assert len(silos) == 1
        assert silos[0]["name"] == "turbulence"
        assert silos[0]["description"] == "Turbulence papers"

    def test_create_duplicate_is_idempotent(self, tmp_path):
        from drbrain.storage.explore import create_explore_silo

        s1 = create_explore_silo(tmp_path, "nlp")
        s2 = create_explore_silo(tmp_path, "nlp")
        assert s1["name"] == s2["name"]
        assert s1["created_at"] == s2["created_at"]

    def test_add_paper_to_silo(self, tmp_path):
        from drbrain.storage.explore import (
            add_paper_to_silo,
            create_explore_silo,
            get_silo_papers,
        )

        create_explore_silo(tmp_path, "nlp")
        add_paper_to_silo(
            tmp_path,
            "nlp",
            {
                "title": "Attention Is All You Need",
                "authors": ["Vaswani, Ashish"],
                "year": 2017,
                "doi": "10.1234/attention",
            },
        )

        papers = get_silo_papers(tmp_path, "nlp")
        assert len(papers) == 1
        assert papers[0]["title"] == "Attention Is All You Need"

    def test_add_paper_nonexistent_silo(self, tmp_path):
        from drbrain.storage.explore import add_paper_to_silo

        with pytest.raises(ValueError, match="Silo not found"):
            add_paper_to_silo(tmp_path, "ghost", {"title": "X"})

    def test_search_silo(self, tmp_path):
        from drbrain.storage.explore import (
            add_paper_to_silo,
            create_explore_silo,
            search_silo,
        )

        create_explore_silo(tmp_path, "ml")
        add_paper_to_silo(
            tmp_path,
            "ml",
            {
                "title": "Deep Learning with PyTorch",
                "authors": ["Smith, John"],
                "year": 2020,
            },
        )
        add_paper_to_silo(
            tmp_path,
            "ml",
            {
                "title": "Graph Neural Networks in Practice",
                "authors": ["Jones, Bob"],
                "year": 2021,
            },
        )

        results = search_silo(tmp_path, "ml", "deep learning")
        assert len(results) == 1
        assert "PyTorch" in results[0]["title"]

        results2 = search_silo(tmp_path, "ml", "neural")
        assert len(results2) == 1
        assert "Neural" in results2[0]["title"]

    def test_search_silo_case_insensitive(self, tmp_path):
        from drbrain.storage.explore import (
            add_paper_to_silo,
            create_explore_silo,
            search_silo,
        )

        create_explore_silo(tmp_path, "ml")
        add_paper_to_silo(
            tmp_path,
            "ml",
            {
                "title": "TRANSFORMER Architectures",
                "authors": [],
                "year": 2023,
            },
        )

        results = search_silo(tmp_path, "ml", "transformer")
        assert len(results) == 1

    def test_delete_silo(self, tmp_path):
        from drbrain.storage.explore import (
            create_explore_silo,
            delete_explore_silo,
            list_explore_silos,
        )

        create_explore_silo(tmp_path, "temp")
        assert len(list_explore_silos(tmp_path)) == 1

        delete_explore_silo(tmp_path, "temp")
        assert len(list_explore_silos(tmp_path)) == 0

    def test_invalid_silo_name(self, tmp_path):
        from drbrain.storage.explore import create_explore_silo

        with pytest.raises(ValueError):
            create_explore_silo(tmp_path, "bad/name")
        with pytest.raises(ValueError):
            create_explore_silo(tmp_path, "..")
        with pytest.raises(ValueError):
            create_explore_silo(tmp_path, "")
