#!/usr/bin/env bash
# launch_ingest_only.sh — 只跑 ingest（--no-db 只缓存文件，不写 db）
# 8 片 oa 并行，scibase 已完成跳过。INGEST_CONCURRENCY=8 吃满 ox-alpha-free。
# 用法: bash scripts/pipeline/launch_ingest_only.sh [base_config]
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SOURCE_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
# shellcheck source=runtime.sh
source "$SCRIPT_DIR/runtime.sh"
WAIT_FOR_WORKERS="$(runtime_env_selector DRBRAIN_WAIT_FOR_WORKERS 1 "DRBRAIN_WAIT_FOR_WORKERS")"
if [[ "${1:-}" == "--no-wait" ]]; then
  WAIT_FOR_WORKERS=0
  shift
elif [[ "${1:-}" == "--wait" ]]; then
  WAIT_FOR_WORKERS=1
  shift
fi
case "$WAIT_FOR_WORKERS" in
  0|1) ;;
  *) runtime_die "DRBRAIN_WAIT_FOR_WORKERS must be 0 or 1" ;;
esac
DEFAULT_ROOT="$SOURCE_ROOT"
runtime_init_selected_root "$DEFAULT_ROOT"
ROOT="$RUNTIME_ROOT"
cd "$ROOT"
if [[ $# -gt 0 ]]; then
  [[ -n "$1" ]] || runtime_die "base config is empty"
  BASE_CFG_VALUE="$1"
elif [[ "${DRBRAIN_CONFIG+x}" == x ]]; then
  [[ -n "$DRBRAIN_CONFIG" ]] || runtime_die "DRBRAIN_CONFIG is empty"
  BASE_CFG_VALUE="$DRBRAIN_CONFIG"
else
  BASE_CFG_VALUE="config.yaml"
fi
BASE_CFG="$(runtime_path "$BASE_CFG_VALUE" "base config")"
export DRBRAIN_CONFIG="$BASE_CFG"
export DRBRAIN_CONFIG_PATH="$BASE_CFG"
command -v uv >/dev/null 2>&1 || {
  echo "uv executable not found in PATH" >&2
  exit 127
}
RUN_TAG="$(runtime_run_tag)"
TEMP_ROOT="$(runtime_env_path DRBRAIN_TEMP_ROOT "$ROOT/data/.runtime" "temporary root")"
export DRBRAIN_TEMP_ROOT="$TEMP_ROOT"
LOG_DIR="$(runtime_env_path DRBRAIN_LOG_DIR "$TEMP_ROOT/drbrain-$RUN_TAG" "log directory")"
LOG_DIR="$(runtime_prepare_dir "$LOG_DIR" "log directory")"
export DRBRAIN_LOG_DIR="$LOG_DIR"

PIDS=()
trap runtime_cleanup_workers_on_failure EXIT
trap 'exit 143' HUP INT TERM

for i in 0 1 2 3 4 5 6 7; do
  if [ $((i % 2)) -eq 0 ]; then
    EMBED=config.embed1.yaml
  else
    EMBED=config.embed2.yaml
  fi
  EMBED="$(runtime_path "$EMBED" "embedding config")"
  SHARD_DB="$(runtime_path "data/shards/oa_shard$i.db" "OpenAlex shard database")"
  MANIFEST="$(runtime_path "data/shards/oa_shard$i.ingest.jsonl" "OpenAlex ingest manifest")"
  LOG_FILE="$(runtime_path "$LOG_DIR/launch_oa$i.log" "OpenAlex launcher log")"
  nohup env \
    DRBRAIN_ROOT="$ROOT" DRBRAIN_CONFIG="$BASE_CFG" \
    INGEST_CONCURRENCY=8 \
    uv run --project "$SOURCE_ROOT" --directory "$SOURCE_ROOT" python -u "$SCRIPT_DIR/ingest_openalex.py" \
      --shard-id "$i" --shard-total 8 --no-db \
      --db "$SHARD_DB" \
      --manifest "$MANIFEST" \
      --config "$EMBED" \
    >> "$LOG_FILE" 2>&1 &
  pid=$!
  PIDS+=("$pid")
  echo "oa$i ingest launched (pid $pid)"
done
echo "=== 8 片 oa ingest 全部启动（--no-db 只缓存文件, 并发16；logs: $LOG_DIR） ==="

if [[ "$WAIT_FOR_WORKERS" == "1" ]]; then
  wait_status=0
  for pid in "${PIDS[@]}"; do
    if wait "$pid"; then
      continue
    else
      child_status=$?
    fi
    if [[ "$wait_status" -eq 0 ]]; then
      wait_status="$child_status"
      runtime_stop_workers
    fi
  done
  if [[ "$wait_status" -ne 0 ]]; then
    echo "=== ingest worker failure detected (exit $wait_status; see logs: $LOG_DIR) ===" >&2
    exit "$wait_status"
  fi
  echo "=== 8 个 ingest worker 全部完成 ==="
fi
