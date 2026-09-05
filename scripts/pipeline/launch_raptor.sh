#!/bin/bash
# RAPTOR 摘要补跑(先缓存后入库):16 进程 × 8 worker = 128 LLM 并发(实测最优)。
# config.local.yaml(hy3 前置)——hy3 限流时 fallback ox-alpha-free(同一批 key,
# 双 config 分池无效,已实测 18/18 key 重叠)。
# 只算不写(embed_batch --raptor-out),跑完统一 load_raptor.py 入库,零锁竞争。
# OMP_NUM_THREADS=2 限制 GMM/OpenBLAS 线程,防 16 进程 CPU 争抢。
set -u
source "$(dirname "$0")/runtime.sh"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
runtime_init_selected_root "$ROOT"
for i in $(seq 0 15); do
  OMP_NUM_THREADS=2 EMBED_WORKERS=4 EMBED_PAPER_TIMEOUT=900     nohup uv run python scripts/pipeline/embed_batch.py \
      --ids-file "/tmp/raptor_shard_${i}.txt" \
      --db data/drbrain.db \
      --config config.local.yaml \
      --raptor-out "data/spool/raptor_out_${i}.jsonl" \
      >> "/tmp/raptor_${i}.log" 2>&1 &
  echo "raptor shard${i} started (pid $!)"
done
wait
echo "ALL_RAPTOR_DONE"
