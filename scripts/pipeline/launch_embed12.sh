#!/usr/bin/env bash
# launch_embed12.sh — 启动 12 路 embed（分片→配置映射固定）
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

declare -A CFG
CFG[0]=e8006; CFG[1]=e8007; CFG[2]=e8008; CFG[3]=e8009
CFG[4]=e8010; CFG[5]=e8011; CFG[6]=e8012; CFG[7]=e8013
CFG[8]=e8006; CFG[9]=e8008; CFG[10]=e8010; CFG[11]=e8012

for i in 0 1 2 3 4 5 6 7 8 9 10 11; do
  nohup .venv/bin/python -u scripts/pipeline/embed_batch.py \
    --ids-file /tmp/emb_shard$i.txt \
    --config config.${CFG[$i]}.yaml \
    --db data/drbrain.db --skip-raptor >> /tmp/embed_s$i.log 2>&1 &
  echo "s$i -> ${CFG[$i]} (pid $!)"
done
echo "=== 12 路 embed 启动 ==="