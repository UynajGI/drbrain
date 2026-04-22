"""LLM client with YAML-configured fallback chain."""
from __future__ import annotations

import json
import logging

import litellm

log = logging.getLogger(__name__)


class LLMClient:
    """Calls LLM with provider/model from config, supports fallback chain."""

    def __init__(self, models: list[dict]):
        """models: list of {provider, model, api_key, base_url} from config."""
        self.models = models

    def call(self, prompt: str, system_prompt: str = "", max_tokens: int = 4096) -> dict | None:
        """Call first successful model in chain. Returns parsed JSON dict or None."""
        return call_with_fallback(prompt, self.models, system_prompt, max_tokens)


def _build_litellm_kwargs(model_cfg: dict, prompt: str, system_prompt: str, max_tokens: int) -> dict:
    """Build litellm completion kwargs from model config."""
    name = f"{model_cfg['provider']}/{model_cfg['model']}"
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    kwargs = {
        "model": name,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }
    if model_cfg.get("api_key"):
        kwargs["api_key"] = model_cfg["api_key"]
    if model_cfg.get("base_url"):
        kwargs["api_base"] = model_cfg["base_url"]
    return kwargs


def call_with_fallback(
    prompt: str,
    models: list[dict],
    system_prompt: str = "",
    max_tokens: int = 4096,
) -> dict | None:
    """Try models in order, return first successful parsed JSON response."""
    for i, model_cfg in enumerate(models):
        try:
            kwargs = _build_litellm_kwargs(model_cfg, prompt, system_prompt, max_tokens)
            response = litellm.completion(**kwargs)
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            name = f"{model_cfg['provider']}/{model_cfg['model']}"
            log.warning(f"Model {name} failed (attempt {i+1}/{len(models)}): {e}")
            continue
    log.error(f"All {len(models)} models failed")
    return None


async def acall_with_fallback(
    prompt: str,
    models: list[dict],
    system_prompt: str = "",
    max_tokens: int = 4096,
) -> dict | None:
    """Async version of call_with_fallback."""
    for i, model_cfg in enumerate(models):
        try:
            kwargs = _build_litellm_kwargs(model_cfg, prompt, system_prompt, max_tokens)
            response = await litellm.acompletion(**kwargs)
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            name = f"{model_cfg['provider']}/{model_cfg['model']}"
            log.warning(f"Model {name} failed (attempt {i+1}/{len(models)}): {e}")
            continue
    log.error(f"All {len(models)} models failed")
    return None
