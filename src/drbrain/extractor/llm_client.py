"""LLM client with YAML-configured fallback chain + token tracking."""

from __future__ import annotations

import json
import time

import litellm
from loguru import logger


class LLMClient:
    """Calls LLM with provider/model from config, supports fallback chain."""

    def __init__(self, models: list[dict]):
        self.models = models

    def call(self, prompt: str, system_prompt: str = "", max_tokens: int = 16384) -> dict | None:
        return call_with_fallback(prompt, self.models, system_prompt, max_tokens)


def _build_litellm_kwargs(
    model_cfg: dict, prompt: str, system_prompt: str, max_tokens: int
) -> dict:
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
        "timeout": 60,
    }
    if model_cfg.get("api_key"):
        kwargs["api_key"] = model_cfg["api_key"]
    if model_cfg.get("base_url"):
        kwargs["api_base"] = model_cfg["base_url"]
    return kwargs


def _record_llm(model_name: str, provider: str, response, start_time: float) -> None:
    """Record LLM usage to metrics store."""
    try:
        from drbrain.metrics import get_metrics

        duration_ms = int((time.monotonic() - start_time) * 1000)
        usage = getattr(response, "usage", None)
        tokens_in = getattr(usage, "prompt_tokens", 0) if usage else 0
        tokens_out = getattr(usage, "completion_tokens", 0) if usage else 0
        get_metrics().record_llm(
            model=model_name,
            provider=provider,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            duration_ms=duration_ms,
        )
    except Exception:
        pass


def call_with_fallback(
    prompt: str,
    models: list[dict],
    system_prompt: str = "",
    max_tokens: int = 16384,
) -> dict | None:
    """Try models in order, return first successful parsed JSON response."""
    for i, model_cfg in enumerate(models):
        name = f"{model_cfg['provider']}/{model_cfg['model']}"
        try:
            start = time.monotonic()
            kwargs = _build_litellm_kwargs(model_cfg, prompt, system_prompt, max_tokens)
            response = litellm.completion(**kwargs)
            _record_llm(model_cfg["model"], model_cfg.get("provider", ""), response, start)
            content = response.choices[0].message.content
            logger.debug(f"LLM call success: {name} in {int((time.monotonic() - start) * 1000)}ms")
            return json.loads(content)
        except Exception as e:
            logger.warning(f"Model {name} failed (attempt {i + 1}/{len(models)}): {e}")
            continue
    logger.error(f"All {len(models)} models failed")
    return None


async def acall_with_fallback(
    prompt: str,
    models: list[dict],
    system_prompt: str = "",
    max_tokens: int = 16384,
) -> dict | list | None:
    """Async version of call_with_fallback."""
    for i, model_cfg in enumerate(models):
        name = f"{model_cfg['provider']}/{model_cfg['model']}"
        try:
            start = time.monotonic()
            kwargs = _build_litellm_kwargs(model_cfg, prompt, system_prompt, max_tokens)
            response = await litellm.acompletion(**kwargs)
            _record_llm(model_cfg["model"], model_cfg.get("provider", ""), response, start)
            content = response.choices[0].message.content
            logger.debug(f"LLM call success: {name} in {int((time.monotonic() - start) * 1000)}ms")
            return json.loads(content)
        except Exception as e:
            logger.warning(f"Model {name} failed (attempt {i + 1}/{len(models)}): {e}")
            continue
    logger.error(f"All {len(models)} models failed")
    return None


async def acall_text_with_fallback(
    prompt: str,
    models: list[dict],
    system_prompt: str = "",
    max_tokens: int = 1024,
) -> str | None:
    """Async text call with fallback. Returns raw text (not JSON)."""
    for i, model_cfg in enumerate(models):
        name = f"{model_cfg['provider']}/{model_cfg['model']}"
        try:
            start = time.monotonic()
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            kwargs = {
                "model": name,
                "messages": messages,
                "temperature": 0,
                "max_tokens": max_tokens,
                "timeout": 60,
            }
            if model_cfg.get("api_key"):
                kwargs["api_key"] = model_cfg["api_key"]
            if model_cfg.get("base_url"):
                kwargs["api_base"] = model_cfg["base_url"]
            response = await litellm.acompletion(**kwargs)
            _record_llm(model_cfg["model"], model_cfg.get("provider", ""), response, start)
            logger.debug(
                f"LLM text call success: {name} in {int((time.monotonic() - start) * 1000)}ms"
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Model {name} failed (attempt {i + 1}/{len(models)}): {e}")
            continue
    logger.error(f"All {len(models)} models failed")
    return None
