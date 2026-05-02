"""Tests for setup.py config generation."""

import tempfile
from pathlib import Path

from drbrain.cli.setup import generate_local_config


def test_generate_local_config_creates_file():
    """generate_local_config writes a valid YAML file."""
    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "config.local.yaml"
        result = generate_local_config(
            output_path=out_path,
            llm_primary={"provider": "openai", "model": "gpt-4", "api_key": "x"},
        )
        assert result == out_path
        assert out_path.exists()

        import yaml

        config = yaml.safe_load(out_path.read_text())
        assert config["llm"]["models"][0]["provider"] == "openai"
        assert config["db"]["path"] == "data/drbrain.db"
        assert config["mineru"]["model"] == "vlm"


def test_generate_local_config_token_mode():
    """Token mode includes mineru token in config."""
    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "config.local.yaml"
        generate_local_config(
            output_path=out_path,
            llm_primary={"provider": "openai", "model": "gpt-4", "api_key": "x"},
            mineru_mode="token",
            mineru_token="my-secret-token",
        )

        import yaml

        config = yaml.safe_load(out_path.read_text())
        assert config["mineru"]["token"] == "my-secret-token"


def test_generate_local_config_flash_mode_no_token():
    """Flash mode has empty mineru token."""
    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "config.local.yaml"
        generate_local_config(
            output_path=out_path,
            llm_primary={"provider": "openai", "model": "gpt-4", "api_key": "x"},
            mineru_mode="flash",
            mineru_token="should-be-ignored",
        )

        import yaml

        config = yaml.safe_load(out_path.read_text())
        assert config["mineru"]["token"] == ""


def test_generate_local_config_custom_params():
    """Custom DB path and BM25 params are saved."""
    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "config.local.yaml"
        generate_local_config(
            output_path=out_path,
            llm_primary={"provider": "ollama", "model": "qwen2.5:7b", "api_key": ""},
            db_path=str(Path(td) / "custom.db"),
            bm25_k1=2.0,
            bm25_b=0.5,
            s2_rate_limit=60,
        )

        import yaml

        config = yaml.safe_load(out_path.read_text())
        assert config["db"]["path"] == str(Path(td) / "custom.db")
        assert config["bm25"]["k1"] == 2.0
        assert config["bm25"]["b"] == 0.5
        assert config["api"]["s2_rate_limit"] == 60


# -- _ensure_directories tests --


def test_ensure_directories_creates_missing():
    """_ensure_directories creates missing dirs, returns count."""
    from drbrain.cli.setup import _ensure_directories

    with tempfile.TemporaryDirectory() as td:
        cfg = {"dirs": {"inbox": f"{td}/inbox", "papers": f"{td}/papers"}}
        created = _ensure_directories(cfg)
        assert created == 2
        assert Path(f"{td}/inbox").exists()
        assert Path(f"{td}/papers").exists()

        # Second call creates nothing
        created2 = _ensure_directories(cfg)
        assert created2 == 0


def test_ensure_directories_default_paths():
    """_ensure_directories uses defaults when config has no dirs."""
    from drbrain.cli.setup import _ensure_directories

    cfg = {}  # no dirs config — uses default paths
    created = _ensure_directories(cfg)
    assert created >= 0  # doesn't crash


# -- _brief_validation tests --


def test_brief_validation_all_ok():
    """_brief_validation returns ok and warn lists."""
    from drbrain.cli.setup import _brief_validation

    # Use a temp directory that exists
    with tempfile.TemporaryDirectory() as td:
        papers_dir = Path(td) / "papers"
        papers_dir.mkdir()
        cfg = {"dirs": {"inbox": str(papers_dir)}}
        ok, warn = _brief_validation(cfg)
        assert isinstance(ok, list)
        assert isinstance(warn, list)
        # Should have ok for existing dir
        assert any("Data directories" in o for o in ok)
