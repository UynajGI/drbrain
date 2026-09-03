#!/usr/bin/env python
"""Qwen3-Embedding-0.6B 本地嵌入服务 — OpenAI 兼容 /v1/embeddings。

常驻进程，模型只加载一次，避免每次 embed 都重新加载 0.6B（省时间）。
用法: /home/jiangyuan/mineru-venv/bin/python scripts/serve_embedding.py
"""

from __future__ import annotations

import threading

import uvicorn
from fastapi import FastAPI, Request

MODEL = "/home/jiangyuan/.cache/modelscope/models/Qwen--Qwen3-Embedding-0.6B/snapshots/master"

app = FastAPI()
_infer_lock = threading.Lock()
_model = None


@app.on_event("startup")
def _load() -> None:
    global _model
    from sentence_transformers import SentenceTransformer

    # 本地 modelscope 缓存路径，local_files_only 避免联网下载（本机 HF 不通）
    _model = SentenceTransformer(MODEL, device="cuda", local_files_only=True)
    # 全摘要文本可达数千 token，不截断会把 attention 激活打爆 16GB 显存；
    # 512 是检索嵌入标准截断（覆盖绝大多数摘要），内存硬上限 ~3GB/实例
    _model.max_seq_length = 512


@app.post("/v1/embeddings")
async def embeddings(req: Request) -> dict:
    body = await req.json()
    texts = body.get("input", [])
    if isinstance(texts, str):
        texts = [texts]
    with _infer_lock:
        # batch_size 上限：全摘要长文本（数百 token）一次性 encode 会把
        # attention 激活打爆 16GB 显存 —— 分小批推理，内存稳定 ~4GB/实例
        vecs = _model.encode(texts, normalize_embeddings=True, batch_size=8).tolist()
    return {
        "object": "list",
        "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vecs)],
        "model": MODEL,
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


if __name__ == "__main__":
    import sys

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
    uvicorn.run(app, host="0.0.0.0", port=port)
