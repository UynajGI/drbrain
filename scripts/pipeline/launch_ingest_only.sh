#!/usr/bin/env bash
# launch_ingest_only.sh — 只跑 ingest（--no-db 只缓存文件，不写 db）
# 8 片 oa 并行，scibase 已完成跳过。INGEST_CONCURRENCY=8 吃满 ox-alpha-free。
# 用法: bash scripts/pipeline/launch_ingest_only.sh
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

for i in 0 1 2 3 4 5 6 7; do
  if [ $((i % 2)) -eq 0 ]; then
    EMBED=config.embed1.yaml
  else
    EMBED=config.embed2.yaml
  fi
  nohup bash -c "
    cd '$ROOT'
    INGEST_CONCURRENCY=8 uv run python -u scripts/pipeline/ingest_openalex.py \
      --shard-id $i --shard-total 8 --no-db \
      --db data/shards/oa_shard$i.db --manifest data/shards/oa_shard$i.ingest.jsonl \
      --config $EMBED >> /tmp/oa_shard$i.log 2>&1
  " >> "/tmp/launch_oa$i.log" 2>&1 &
  echo "oa$i ingest launched (pid $!)"
done
echo "=== 8 片 oa ingest 全部启动（--no-db 只缓存文件, 并发16）==="