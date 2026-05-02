"""Tests for setup.py config generation."""

import tempfile
from pathlib import Path
from unittest import mock

from drbrain.cli.setup import generate_local_config, setup_cmd


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


# -- setup_cmd tests --


@mock.patch("drbrain.cli.setup.generate_local_config")
@mock.patch("typer.confirm")
@mock.patch("typer.prompt")
@mock.patch("typer.echo")
def test_setup_cmd_runs_with_confirm_true(mock_echo, mock_prompt, mock_confirm, mock_gen_config):
    """setup_cmd writes config when user confirms 'Write config.local.yaml?'."""
    mock_prompt.side_effect = [
        "openai",  # [1/16] provider
        "gpt-4o",  # [1/16] model
        "sk-key",  # [1/16] api_key
        "",  # [1/16] base_url
        "flash",  # [4/16] mineru_mode
        "vlm",  # [6/16] mineru_model
        "data/drbrain.db",  # [10/16] db_path
        100,  # [14/16] s2_rate_limit
        "",  # [15/16] crossref_email
        "",  # [15/16] openalex_token
        1.5,  # [16/16] bm25_k1
        0.75,  # [16/16] bm25_b
    ]
    mock_confirm.side_effect = [
        False,  # [2/16] has_fallback → no
        False,  # [7/16] is_ocr → no
        True,  # [7/16] enable_formula → yes
        True,  # [7/16] enable_table → yes
        True,  # [16/16] Write config.local.yaml? → yes
    ]
    mock_gen_config.return_value = Path("config.local.yaml")

    setup_cmd()

    mock_gen_config.assert_called_once()


@mock.patch("drbrain.cli.setup.generate_local_config")
@mock.patch("typer.confirm")
@mock.patch("typer.prompt")
@mock.patch("typer.echo")
def test_setup_cmd_cancelled_on_false(mock_echo, mock_prompt, mock_confirm, mock_gen_config):
    """setup_cmd does NOT write config when user declines confirmation."""
    mock_prompt.side_effect = [
        "openai",  # provider
        "gpt-4o",  # model
        "sk-key",  # api_key
        "",  # base_url
        "flash",  # mineru_mode
        "vlm",  # mineru_model
        "data/drbrain.db",  # db_path
        100,  # s2_rate_limit
        "",  # crossref_email
        "",  # openalex_token
        1.5,  # bm25_k1
        0.75,  # bm25_b
    ]
    mock_confirm.side_effect = [
        False,  # has_fallback → no
        False,  # is_ocr → no
        True,  # enable_formula → yes
        True,  # enable_table → yes
        False,  # Write config.local.yaml? → NO
    ]
    mock_gen_config.return_value = Path("config.local.yaml")

    setup_cmd()

    # generate_local_config should NOT be called
    mock_gen_config.assert_not_called()
