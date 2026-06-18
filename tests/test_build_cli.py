"""Tests for drbrain.cli.build_commands — build / embed / translate commands.

Strategy: invoke commands through a minimal Typer app whose callback injects a
config dict into ctx.obj. Mock Database, open_db, and any LLM/embedding
side-effects so no network or real DB writes occur.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import typer
from typer.testing import CliRunner

from drbrain.cli.build_commands import (
    _build_extraction_summary,
    build_cmd,
    embed_cmd,
    translate_cmd,
)


def _make_app(config: dict) -> typer.Typer:
    """Build a Typer app whose callback injects *config* into ctx.obj."""
    app = typer.Typer()

    @app.callback()
    def _setup(ctx: typer.Context) -> None:
        ctx.obj = {"config": config}

    app.command("build")(build_cmd)
    app.command("embed")(embed_cmd)
    app.command("translate")(translate_cmd)
    return app


runner = CliRunner()


def _cfg(**overrides):
    base = {
        "db": {"path": "/tmp/fake.db"},
        "llm": {"models": [{"model": "fake-model"}]},
        "dirs": {"papers": "/tmp/fake_papers"},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# build_cmd
# ---------------------------------------------------------------------------


@patch("drbrain.cli.build_commands.Database")
def test_build_no_papers_message(mock_db_cls):
    """build_cmd with empty unprocessed-paper list prints guidance and exits 0."""
    mock_db = MagicMock()
    mock_db.get_all_papers.return_value = []
    mock_db_cls.return_value = mock_db

    app = _make_app(_cfg())
    result = runner.invoke(app, ["build"])
    assert result.exit_code == 0
    assert "No papers to build" in result.output
    mock_db.close.assert_called()


@patch("drbrain.cli.build_commands.Database")
def test_build_all_empty(mock_db_cls):
    """build --all with empty DB also reports no papers."""
    mock_db = MagicMock()
    mock_db.get_all_papers.return_value = []
    mock_db_cls.return_value = mock_db

    app = _make_app(_cfg())
    result = runner.invoke(app, ["build", "--all"])
    assert result.exit_code == 0
    assert "No papers to build" in result.output


@patch("drbrain.cli.build_commands.Database")
def test_build_paper_not_found(mock_db_cls):
    """build with unknown paper_id: echoes 'not found', no papers processed."""
    mock_db = MagicMock()
    mock_db.get_paper.return_value = None  # unknown id
    mock_db.get_all_papers.return_value = []
    mock_db_cls.return_value = mock_db

    app = _make_app(_cfg())
    result = runner.invoke(app, ["build", "nonexistent-001"])
    assert result.exit_code == 0
    assert "Paper not found: nonexistent-001" in result.output
    assert "No papers to build" in result.output


@patch("drbrain.cli.build_commands.Database")
def test_build_no_llm_models_exits_1(mock_db_cls):
    """build with a paper present but no LLM models configured → exit 1."""
    mock_db = MagicMock()
    mock_db.get_all_papers.return_value = [{"local_id": "p1", "title": "T", "status": "uploaded"}]
    mock_db_cls.return_value = mock_db

    app = _make_app(_cfg(llm={"models": []}))
    result = runner.invoke(app, ["build"])
    assert result.exit_code == 1
    assert "No LLM models configured" in result.output
    mock_db.close.assert_called()


@patch("drbrain.cli.build_commands.Database")
def test_build_paper_missing_raw_md_skipped(mock_db_cls, tmp_path):
    """build for a paper whose raw.md is missing → echoes skip message."""
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    # paper dir exists but no raw.md / tree.json
    (papers_dir / "p2").mkdir()

    mock_db = MagicMock()
    mock_db.get_all_papers.return_value = [
        {"local_id": "p2", "title": "My Paper", "status": "uploaded"}
    ]
    mock_db_cls.return_value = mock_db

    app = _make_app(_cfg(dirs={"papers": str(papers_dir)}))
    result = runner.invoke(app, ["build"])
    assert result.exit_code == 0
    assert "No raw.md" in result.output


# ---------------------------------------------------------------------------
# embed_cmd
# ---------------------------------------------------------------------------


@patch("drbrain.cli.build_commands.GraphEngine")
@patch("drbrain.cli.build_commands.Database")
def test_embed_no_graph_data_exits_1(mock_db_cls, mock_graph_cls):
    """embed with empty graph → exit 1 with guidance."""
    mock_db = MagicMock()
    mock_db.load_embeddings.return_value = []
    mock_db_cls.return_value = mock_db

    mock_graph = MagicMock()
    mock_graph.graph.number_of_nodes.return_value = 0
    mock_graph_cls.return_value = mock_graph

    app = _make_app(_cfg())
    result = runner.invoke(app, ["embed"])
    assert result.exit_code == 1
    assert "No graph data" in result.output
    mock_db.close.assert_called()


@patch("drbrain.config.EmbedConfig")
@patch("drbrain.cli.build_commands.Database")
def test_embed_tree_provider_none_skips(mock_db_cls, mock_embed_cfg_cls, tmp_path):
    """embed --tree with provider=none → prints disabled message and returns."""
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()

    embed_cfg = MagicMock()
    embed_cfg.provider = "none"
    mock_embed_cfg_cls.return_value = embed_cfg

    mock_db = MagicMock()
    mock_db_cls.return_value = mock_db

    app = _make_app(_cfg(dirs={"papers": str(papers_dir)}))
    result = runner.invoke(app, ["embed", "--tree"])
    assert result.exit_code == 0
    assert "disabled" in result.output
    mock_db.close.assert_called()


@patch("drbrain.cli.build_commands.GraphEngine")
@patch("drbrain.cli.build_commands.Database")
def test_embed_trains_and_saves(mock_db_cls, mock_graph_cls):
    """embed happy path: trains TransE and persists embeddings."""
    mock_db = MagicMock()
    mock_db.load_embeddings.return_value = []
    mock_db_cls.return_value = mock_db

    mock_graph = MagicMock()
    mock_graph.graph.number_of_nodes.return_value = 5
    mock_graph_cls.return_value = mock_graph

    fake_transe = MagicMock()
    fake_transe.entities = {"A": [0.1], "B": [0.2]}
    fake_transe.relations = {"rel": [0.3]}

    app = _make_app(_cfg())
    with patch("drbrain.graph.embedding.TransE", return_value=fake_transe):
        result = runner.invoke(app, ["embed", "--dim", "64", "--epochs", "5"])

    assert result.exit_code == 0, result.output
    assert "Trained 2 entities" in result.output
    assert "incremental" not in result.output or "from scratch" in result.output
    # save_embedding invoked for entities + relations
    assert mock_db.save_embedding.call_count >= 3
    mock_db.commit.assert_called()
    mock_db.close.assert_called()


@patch("drbrain.cli.build_commands.GraphEngine")
@patch("drbrain.cli.build_commands.Database")
def test_embed_retrain_uses_no_init_entities(mock_db_cls, mock_graph_cls):
    """embed --retrain should skip loading initial entities."""
    mock_db = MagicMock()
    mock_db.load_embeddings.return_value = [{"label": "X"}]
    mock_db_cls.return_value = mock_db

    mock_graph = MagicMock()
    mock_graph.graph.number_of_nodes.return_value = 3
    mock_graph_cls.return_value = mock_graph

    fake_transe = MagicMock()
    fake_transe.entities = {}
    fake_transe.relations = {}

    app = _make_app(_cfg())
    with patch("drbrain.graph.embedding.TransE", return_value=fake_transe) as m_transe:
        result = runner.invoke(app, ["embed", "--retrain"])

    assert result.exit_code == 0
    # init_entities should be None when retrain=True
    _, kwargs = m_transe.return_value.train.call_args
    assert kwargs.get("init_entities") is None
    mock_db.clear_embeddings.assert_called()


# ---------------------------------------------------------------------------
# translate_cmd
# ---------------------------------------------------------------------------


def test_translate_paper_not_found_exits_1():
    """translate_cmd with unknown paper → exit 1."""
    cfg = _cfg()
    app = _make_app(cfg)

    fake_db = MagicMock()
    fake_db.get_paper.return_value = None
    with patch("drbrain.cli.build_commands.open_db") as m_open:
        # open_db is used as context manager
        m_open.return_value.__enter__.return_value = fake_db
        result = runner.invoke(app, ["translate", "ghost-001"])

    assert result.exit_code == 1
    assert "Paper not found: ghost-001" in result.output


def test_translate_no_raw_md_exits_1(tmp_path):
    """translate for a paper without raw.md → exit 1."""
    papers_dir = tmp_path / "papers"
    (papers_dir / "p3").mkdir(parents=True)  # dir but no raw.md
    cfg = _cfg(dirs={"papers": str(papers_dir)})
    app = _make_app(cfg)

    fake_db = MagicMock()
    fake_db.get_paper.return_value = {"local_id": "p3", "title": "T"}
    with patch("drbrain.cli.build_commands.open_db") as m_open:
        m_open.return_value.__enter__.return_value = fake_db
        result = runner.invoke(app, ["translate", "p3"])

    assert result.exit_code == 1
    assert "No raw.md found" in result.output


def test_translate_no_llm_models_exits_1(tmp_path):
    """translate with raw.md present but no LLM models → exit 1."""
    papers_dir = tmp_path / "papers"
    (papers_dir / "p4").mkdir(parents=True)
    (papers_dir / "p4" / "raw.md").write_text("# Hello", encoding="utf-8")

    cfg = _cfg(llm={"models": []}, dirs={"papers": str(papers_dir)})
    app = _make_app(cfg)

    fake_db = MagicMock()
    fake_db.get_paper.return_value = {"local_id": "p4", "title": "T"}
    with patch("drbrain.cli.build_commands.open_db") as m_open:
        m_open.return_value.__enter__.return_value = fake_db
        result = runner.invoke(app, ["translate", "p4"])

    assert result.exit_code == 1
    assert "No LLM models configured" in result.output


def test_translate_success(tmp_path):
    """translate happy path — echoes translated output path."""
    papers_dir = tmp_path / "papers"
    (papers_dir / "p5").mkdir(parents=True)
    (papers_dir / "p5" / "raw.md").write_text("body", encoding="utf-8")

    cfg = _cfg(dirs={"papers": str(papers_dir)})
    app = _make_app(cfg)

    fake_db = MagicMock()
    fake_db.get_paper.return_value = {"local_id": "p5", "title": "Paper Five"}
    fake_result = MagicMock()
    fake_result.ok = True
    fake_result.path = papers_dir / "p5" / "translated.md"
    fake_result.completed_chunks = 3
    fake_result.total_chunks = 3

    with (
        patch("drbrain.cli.build_commands.open_db") as m_open,
        patch(
            "drbrain.services.translate.translate_paper", return_value=fake_result
        ) as m_translate,
    ):
        m_open.return_value.__enter__.return_value = fake_db
        result = runner.invoke(app, ["translate", "p5", "--lang", "zh"])

    assert result.exit_code == 0, result.output
    assert "Translating: Paper Five" in result.output
    assert "Translated:" in result.output
    m_translate.assert_called_once()
    # target_lang passed through
    _, kwargs = m_translate.call_args
    assert kwargs["target_lang"] == "zh"


def test_translate_json_output(tmp_path):
    """translate --json emits JSON with paper/output/chunks."""
    papers_dir = tmp_path / "papers"
    (papers_dir / "p6").mkdir(parents=True)
    (papers_dir / "p6" / "raw.md").write_text("body", encoding="utf-8")

    cfg = _cfg(dirs={"papers": str(papers_dir)})
    app = _make_app(cfg)

    fake_db = MagicMock()
    fake_db.get_paper.return_value = {"local_id": "p6", "title": "Six"}
    fake_result = MagicMock()
    fake_result.ok = True
    fake_result.path = "/some/path/zh.md"
    fake_result.completed_chunks = 2
    fake_result.total_chunks = 2

    import json

    with (
        patch("drbrain.cli.build_commands.open_db") as m_open,
        patch("drbrain.services.translate.translate_paper", return_value=fake_result),
    ):
        m_open.return_value.__enter__.return_value = fake_db
        result = runner.invoke(app, ["translate", "p6", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output.strip().splitlines()[-1])
    assert data["paper"] == "p6"
    assert data["completed_chunks"] == 2


# ---------------------------------------------------------------------------
# _build_extraction_summary (pure function — fast coverage)
# ---------------------------------------------------------------------------


def test_build_extraction_summary_empty():
    out = _build_extraction_summary("p1", [], [], [], [])
    assert "Extraction results for paper p1" in out
    # no concept/relation sections when empty
    assert "Concepts" not in out


def test_build_extraction_summary_with_concepts():
    concepts = [
        {"type": "Problem", "label": "X"},
        {"type": "Problem", "label": "Y"},
        {"type": "Method", "label": "Z"},
    ]
    out = _build_extraction_summary("p2", concepts, [], [], [])
    assert "Concepts (3 total)" in out
    assert "Problem: X, Y" in out
    assert "Method: Z" in out


def test_build_extraction_summary_truncates_long_lists():
    concepts = [{"type": "T", "label": f"L{i}"} for i in range(15)]
    out = _build_extraction_summary("p3", concepts, [], [], [])
    assert "+5 more" in out


def test_build_extraction_summary_relations_and_merges():
    relations = [{"head": "A", "rel": "uses", "tail": "B"}]
    merges = [{"canonical": "C", "variants": ["c1", "c2"]}]
    corrections = [{"description": "fix typo"}]
    out = _build_extraction_summary("p4", [], relations, merges, corrections)
    assert "Relations (1 total)" in out
    assert "A --[uses]--> B" in out
    assert "Coreference merges (1 total)" in out
    assert "C <- ['c1', 'c2']" in out
    assert "Refinement corrections (1 total)" in out
    assert "fix typo" in out
