"""Tests for proceedings management.

TDD: tests written before implementation.
"""

from __future__ import annotations

import pytest


class TestProceedingsStore:
    """Test the proceedings data store layer."""

    def test_module_imports(self):
        from drbrain.storage import proceedings

        assert proceedings is not None

    def test_create_proceeding(self, tmp_path):
        from drbrain.storage.proceedings import (
            create_proceeding,
            load_proceedings,
        )

        store_path = tmp_path / "proceedings.json"
        create_proceeding(store_path, "NeurIPS", 2024, venue="Vancouver")

        data = load_proceedings(store_path)
        assert len(data) == 1
        assert data[0]["name"] == "NeurIPS"
        assert data[0]["year"] == 2024
        assert "id" in data[0]

    def test_create_duplicate_returns_existing(self, tmp_path):
        from drbrain.storage.proceedings import create_proceeding

        store_path = tmp_path / "proceedings.json"
        p1 = create_proceeding(store_path, "ICML", 2024)
        p2 = create_proceeding(store_path, "ICML", 2024)
        assert p1["id"] == p2["id"]

    def test_add_paper_to_proceeding(self, tmp_path):
        from drbrain.storage.proceedings import (
            add_paper,
            create_proceeding,
            load_proceedings,
        )

        store_path = tmp_path / "proceedings.json"
        p = create_proceeding(store_path, "NeurIPS", 2024)
        add_paper(store_path, p["id"], "paper-001")
        add_paper(store_path, p["id"], "paper-002")

        data = load_proceedings(store_path)
        assert set(data[0]["papers"]) == {"paper-001", "paper-002"}

    def test_add_paper_idempotent(self, tmp_path):
        from drbrain.storage.proceedings import (
            add_paper,
            create_proceeding,
            load_proceedings,
        )

        store_path = tmp_path / "proceedings.json"
        p = create_proceeding(store_path, "NeurIPS", 2024)
        add_paper(store_path, p["id"], "paper-001")
        add_paper(store_path, p["id"], "paper-001")  # duplicate

        data = load_proceedings(store_path)
        assert len(data[0]["papers"]) == 1

    def test_add_paper_unknown_proceeding_raises(self, tmp_path):
        from drbrain.storage.proceedings import add_paper

        store_path = tmp_path / "proceedings.json"
        with pytest.raises(ValueError, match="not found"):
            add_paper(store_path, "nonexistent", "paper-001")

    def test_list_proceedings(self, tmp_path):
        from drbrain.storage.proceedings import (
            create_proceeding,
            list_proceedings,
        )

        store_path = tmp_path / "proceedings.json"
        create_proceeding(store_path, "NeurIPS", 2024)
        create_proceeding(store_path, "ICML", 2023)

        result = list_proceedings(store_path)
        assert len(result) == 2

    def test_get_proceeding(self, tmp_path):
        from drbrain.storage.proceedings import (
            create_proceeding,
            get_proceeding,
        )

        store_path = tmp_path / "proceedings.json"
        p = create_proceeding(store_path, "NeurIPS", 2024)

        found = get_proceeding(store_path, p["id"])
        assert found["name"] == "NeurIPS"

    def test_get_proceeding_not_found(self, tmp_path):
        from drbrain.storage.proceedings import get_proceeding

        store_path = tmp_path / "proceedings.json"
        assert get_proceeding(store_path, "nonexistent") is None
