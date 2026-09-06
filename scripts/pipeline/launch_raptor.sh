#!/usr/bin/env bash
# RAPTOR 摘要补跑(先缓存后入库):16 进程 × 8 worker = 128 LLM 并发(实测最优)。
# config.local.yaml(hy3 前置)——hy3 限流时 fallback ox-alpha-free(同一批 key,
# 双 config 分池无效,已实测 18/18 key 重叠)。
# 只算不写(embed_batch --raptor-out),跑完统一 load_raptor.py 入库,零锁竞争。
# OMP_NUM_THREADS=2 限制 GMM/OpenBLAS 线程,防 16 进程 CPU 争抢。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SOURCE_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
# shellcheck source=runtime.sh
source "$SCRIPT_DIR/runtime.sh"
DEFAULT_ROOT="$SOURCE_ROOT"
runtime_init_selected_root "$DEFAULT_ROOT"
ROOT="$RUNTIME_ROOT"
if [[ $# -gt 0 ]]; then
  [[ -n "$1" ]] || runtime_die "RAPTOR config is empty"
  RAPTOR_CFG_VALUE="$1"
else
  RAPTOR_CFG_VALUE="$(runtime_env_selector DRBRAIN_RAPTOR_CONFIG config.local.yaml "DRBRAIN_RAPTOR_CONFIG")"
fi
RAPTOR_CFG="$(runtime_path "$RAPTOR_CFG_VALUE" "RAPTOR config")"
DB_PATH="$(runtime_env_path RAPTOR_DB "data/drbrain.db" "RAPTOR database")"
cd "$ROOT"
IDS_DIR="$(runtime_env_path RAPTOR_IDS_DIR "$ROOT/data/.runtime" "RAPTOR ids directory")"
OUTPUT_DIR="$(runtime_env_path RAPTOR_OUTPUT_DIR "data/spool" "RAPTOR output directory")"
export DRBRAIN_CONFIG="$RAPTOR_CFG"
export DRBRAIN_CONFIG_PATH="$RAPTOR_CFG"
command -v uv >/dev/null 2>&1 || {
  echo "uv executable not found in PATH" >&2
  exit 127
}
RUN_TAG="$(runtime_run_tag)"
TEMP_ROOT="$(runtime_env_path DRBRAIN_TEMP_ROOT "$ROOT/data/.runtime" "temporary root")"
export DRBRAIN_TEMP_ROOT="$TEMP_ROOT"
LOG_DIR="$(runtime_env_path DRBRAIN_LOG_DIR "$TEMP_ROOT/drbrain-$RUN_TAG" "log directory")"
LOG_DIR="$(runtime_prepare_dir "$LOG_DIR" "log directory")"
OUTPUT_DIR="$(runtime_prepare_dir "$OUTPUT_DIR" "RAPTOR output directory")"
export DRBRAIN_LOG_DIR="$LOG_DIR"

[[ -d "$IDS_DIR" ]] || runtime_die "RAPTOR ids directory does not exist: $IDS_DIR"

for i in $(seq 0 15); do
  ids_file="$(runtime_existing_path "$IDS_DIR/raptor_shard_${i}.txt" "RAPTOR ids file")"
  [[ -f "$ids_file" ]] || runtime_die "RAPTOR ids file is not regular: $ids_file"
done

PIDS=()
trap runtime_cleanup_workers_on_failure EXIT
trap 'exit 143' HUP INT TERM
for i in $(seq 0 15); do
  ids_file="$(runtime_existing_path "$IDS_DIR/raptor_shard_${i}.txt" "RAPTOR ids file")"
  RAPTOR_OUT="$(runtime_path "$OUTPUT_DIR/raptor_out_${i}.jsonl" "RAPTOR output file")"
  LOG_FILE="$(runtime_path "$LOG_DIR/raptor_${i}.log" "RAPTOR log file")"
  OMP_NUM_THREADS=2 EMBED_WORKERS=4 EMBED_PAPER_TIMEOUT=900     nohup uv run --project "$SOURCE_ROOT" --directory "$SOURCE_ROOT" python "$SCRIPT_DIR/embed_batch.py" \
      --ids-file "$ids_file" \
      --db "$DB_PATH" \
      --config "$RAPTOR_CFG" \
      --raptor-out "$RAPTOR_OUT" \
      >> "$LOG_FILE" 2>&1 &
  PIDS+=("$!")
  echo "raptor shard${i} started (pid $!)"
done

status=0
for pid in "${PIDS[@]}"; do
  if wait "$pid"; then
    :
  else
    code=$?
    if [[ "$status" -eq 0 ]]; then
      status="$code"
      runtime_stop_workers
    fi
  fi
done
if [[ "$status" -ne 0 ]]; then
  echo "RAPTOR failed (exit $status); inspect logs in $LOG_DIR" >&2
  exit "$status"
fi
echo "ALL_RAPTOR_DONE (logs: $LOG_DIR)"
