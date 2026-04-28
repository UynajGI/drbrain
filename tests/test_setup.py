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
