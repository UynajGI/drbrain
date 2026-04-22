"""Tests for setup wizard — tests the config generation logic, not interactive I/O."""
from pathlib import Path
import tempfile
from brbrain.config import load_config
from brbrain.cli.setup import generate_local_config

def test_generate_local_config_writes_valid_yaml():
    """generate_local_config produces a file that load_config can read."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "config.local.yaml"
        generate_local_config(
            output_path=out,
            llm_primary={"provider": "openai", "model": "gpt-4o", "api_key": "sk-123", "base_url": None},
            mineru_mode="flash",
            mineru_model="vlm",
            db_path="data/test.db",
        )
        assert out.exists()
        cfg = load_config(
            base_path=Path(__file__).parent.parent / "config.yaml",
            local_path=out,
        )
        assert cfg["llm"]["models"][0]["provider"] == "openai"
        assert cfg["mineru"]["token"] == ""

def test_generate_local_config_token_mode():
    """Token mode writes the token value."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "config.local.yaml"
        generate_local_config(
            output_path=out,
            llm_primary={"provider": "openai", "model": "gpt-4o", "api_key": "sk-123", "base_url": None},
            mineru_mode="token",
            mineru_token="abc-token-123",
            mineru_model="pipeline",
            db_path="data/test.db",
        )
        cfg = load_config(
            base_path=Path(__file__).parent.parent / "config.yaml",
            local_path=out,
        )
        assert cfg["mineru"]["token"] == "abc-token-123"
