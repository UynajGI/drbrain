#!/usr/bin/env bash
# shard_pipeline.sh — scibase 单分片全链：ingest → build(jsonl-out) → load → embed → DONE
# 用法: bash scripts/pipeline/shard_pipeline.sh <shard_dir> <shard_db> <embed_config>
#   shard_dir    如 data/spool/scibase_shards8/shard0
#   shard_db     如 data/shards/shard0.db
#   embed_config 如 config.embed1.yaml（指向 8001 嵌入服务）
# 每步幂等断点续传；崩溃重启本脚本即可续跑。失败不 touch DONE。
set -u
source "$(dirname "$0")/runtime.sh"
SHARD_DIR="$1"
SHARD_DB="$2"
EMBED_CFG="${3:-config.embed1.yaml}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
runtime_init_selected_root "$ROOT"

SHARD_NAME="$(basename "$SHARD_DIR")"
mkdir -p data/shards
INGEST_MANIFEST="data/shards/${SHARD_NAME}.ingest.jsonl"
BUILD_MANIFEST="data/shards/${SHARD_NAME}.build.jsonl"
LOG="/tmp/shard_${SHARD_NAME}.log"

log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

log "=== shard $SHARD_NAME start: dir=$SHARD_DIR db=$SHARD_DB embed_cfg=$EMBED_CFG ==="

log "stage 1/3 ingest (片内并行多篇)..."
INGEST_CONCURRENCY=4 uv run python -u scripts/pipeline/ingest_scibase.py \
  --source "$SHARD_DIR" --resume \
  --db "$SHARD_DB" --manifest "$INGEST_MANIFEST" >> "$LOG" 2>&1
log "stage 1/3 ingest exit=$?"

log "stage 2/3 build (jsonl-out, hy3 主抽 5.7s/次, 并发 20×10叶子=3200路)..."
BUILD_CONCURRENCY=20 uv run python -u scripts/pipeline/build.py \
  --from-manifest "$INGEST_MANIFEST" --resume --config config.build.yaml \
  --db "$SHARD_DB" --manifest "$BUILD_MANIFEST" \
  --jsonl-out "data/shards/${SHARD_NAME}.build.jsonl" >> "$LOG" 2>&1
BUILD_EXIT=$?
log "stage 2/3 build exit=$BUILD_EXIT"
if [ "$BUILD_EXIT" -ne 0 ]; then
  log "build failed — 不 touch DONE，重启脚本续跑"
  exit 1
fi

log "stage 2.5/3 入库 (build jsonl → db)..."
uv run python -u scripts/pipeline/load_build.py \
  --jsonl "data/shards/${SHARD_NAME}.build.jsonl" --db "$SHARD_DB" >> "$LOG" 2>&1
LOAD_EXIT=$?
log "stage 2.5/3 入库 exit=$LOAD_EXIT"
if [ "$LOAD_EXIT" -ne 0 ]; then
  log "入库 failed — 不 touch DONE，重启脚本续跑"
  exit 1
fi

PIDS=$(uv run python -c "
import json
ids=[]
for line in open('$INGEST_MANIFEST'):
    try:
        r=json.loads(line)
        if r.get('ok') and r.get('local_id'): ids.append(r['local_id'])
    except Exception: pass
print(','.join(ids))
" 2>/dev/null)
log "stage 3/3 embed ($(( $(echo "$PIDS" | tr ',' '\n' | wc -l) )) papers)..."
uv run drbrain --config "$EMBED_CFG" embed --tree --db "$SHARD_DB" --papers "$PIDS" >> "$LOG" 2>&1
EMBED_EXIT=$?
log "stage 3/3 embed exit=$EMBED_EXIT"
if [ "$EMBED_EXIT" -ne 0 ]; then
  log "embed failed — 不 touch DONE，重启脚本续跑"
  exit 1
fi

touch "data/shards/${SHARD_NAME}.DONE"
log "=== shard $SHARD_NAME SHARD_DONE ==="
