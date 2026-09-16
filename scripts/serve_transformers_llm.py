#!/usr/bin/env python
"""Qwen3.5-9B 本地推理服务（transformers 4bit, 单卡 V100）— OpenAI 兼容 /v1/chat/completions。

V100 (sm_70) 跑不动 vLLM 0.19（SymmMem/FA2/Triton 均不支持），改用 transformers。
9B 4bit ≈ 5.5GB 权重，单卡 16GB 足够。串行推理锁防并发 OOM。

用法: /home/jiangyuan/mineru-venv/bin/python scripts/serve_transformers_llm.py
"""

from __future__ import annotations

import threading
import os

import torch
import uvicorn
from fastapi import FastAPI, Request

MODEL = os.environ.get(
    "DRBRAIN_SERVE_MODEL_PATH",
    "/home/jiangyuan/.cache/modelscope/models/Qwen--Qwen3.5-9B",
)
MODEL_ID = os.environ.get("DRBRAIN_SERVE_MODEL_ID", "local-model")
# Keep the KV cache bounded on 16 GB cards.  Spark extraction does not need a
# 2k-token answer; large JSON budgets were causing the worker to be killed.
MAX_INPUT_TOKENS = int(os.environ.get("DRBRAIN_MAX_INPUT_TOKENS", "4096"))
MAX_NEW_TOKENS = int(os.environ.get("DRBRAIN_MAX_NEW_TOKENS", "512"))

app = FastAPI()
_infer_lock = threading.Lock()
_tok = None
_model = None


@app.on_event("startup")
def _load() -> None:
    global _tok, _model
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    _tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    quant = None
    if os.environ.get("DRBRAIN_NO_QUANT", "0") != "1":
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    device_map = {"": "cuda:0"} if os.environ.get("DRBRAIN_SINGLE_GPU", "0") == "1" else "auto"
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        config=config,
        dtype=torch.float16,
        device_map=device_map,
        quantization_config=quant,
        trust_remote_code=True,
    )
    _model.eval()


@app.post("/v1/chat/completions")
async def chat(req: Request) -> dict:
    body = await req.json()
    messages = body.get("messages", [])
    max_tokens = int(body.get("max_tokens", 2048))
    temperature = float(body.get("temperature", 0.1))
    # PageIndex expects JSON even when its LiteLLM adapter omits
    # ``response_format`` from the request.
    json_mode = bool(body.get("response_format")) or os.environ.get("DRBRAIN_JSON_MODE", "0") == "1"
    if json_mode:
        messages = list(messages)
        messages.append({"role": "user", "content": "Return ONLY one valid JSON object. Do not include markdown fences, explanations, or trailing text."})
        max_tokens = min(max_tokens, MAX_NEW_TOKENS)
    max_tokens = min(max_tokens, MAX_NEW_TOKENS)
    # 禁用 thinking（抽取场景：快、不吃 max_tokens）
    prompt = _tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = _tok(prompt, return_tensors="pt")
    # 输入截断：build 的 section 全文可能超长，超限截断避免 KV cache OOM
    if inputs["input_ids"].shape[1] > MAX_INPUT_TOKENS:
        inputs = {k: v[:, -MAX_INPUT_TOKENS:] for k, v in inputs.items()}
    # With ``device_map=auto`` Spark may place its input embedding on a
    # different GPU (for example cuda:3).  Inputs must start on that device;
    # Transformers will route activations across the remaining mapped layers.
    input_device = _model.get_input_embeddings().weight.device
    inputs = {k: v.to(input_device) for k, v in inputs.items()}
    with _infer_lock:
        with torch.inference_mode():
            out = _model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
                top_k=0,
            )
    text = _tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    if json_mode:
        text = text.replace("```json", "").replace("```", "").strip()
        if "<think>" in text:
            text = text.split("</think>", 1)[-1].strip()
    in_tokens = inputs["input_ids"].shape[1]
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": in_tokens, "completion_tokens": max_tokens},
    }


@app.get("/v1/models")
async def models() -> dict:
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


if __name__ == "__main__":
    import sys

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
