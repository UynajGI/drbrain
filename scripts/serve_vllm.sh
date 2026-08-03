#!/usr/bin/env bash
# Start the local vLLM concept-extraction server on GPU 0.
# vLLM 0.18.0 + torch 2.10 cu128 (driver 570 max CUDA 12.8).
set -e
cd /home/jiangyuan/drbrain
export CUDA_VISIBLE_DEVICES=0
export FLASHINFER_DISABLE_VERSION_CHECK=1
exec .venv-vllm/bin/vllm serve /home/jiangyuan/models/Qwen3.5-9B \
  --served-model-name qwen3.5-9b \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.75
