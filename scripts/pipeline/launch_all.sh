#!/usr/bin/env bash
# launch_all.sh — 一次性启动 16 片全量增强管线（8 scibase + 8 openalex）
# 用法: bash scripts/pipeline/launch_all.sh
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# 8 scibase 分片（ingest 已完成 → 直接 build，hy3 主抽）
for i in 0 1 2 3 4 5 6 7; do
  if [ $((i % 2)) -eq 0 ]; then
    EMBED=config.embed1.yaml
  else
    EMBED=config.embed2.yaml
  fi
  nohup bash scripts/pipeline/shard_pipeline.sh \
    "data/spool/scibase_shards8/shard$i" "data/shards/shard$i.db" "$EMBED" \
    >> "/tmp/launch_scibase$i.log" 2>&1 &
  echo "scibase shard$i launched (pid $!)"
done

# 8 openalex 分片（ingest 未完成 → 继续 ingest，ox-alpha-free）
for i in 0 1 2 3 4 5 6 7; do
  if [ $((i % 2)) -eq 0 ]; then
    EMBED=config.embed1.yaml
  else
    EMBED=config.embed2.yaml
  fi
  nohup bash scripts/pipeline/oa_shard_pipeline.sh \
    "$i" "data/shards/oa_shard$i.db" "$EMBED" \
    >> "/tmp/launch_oa$i.log" 2>&1 &
  echo "oa shard$i launched (pid $!)"
done

echo "=== 16 片全部启动 ==="