#!/usr/bin/env bash
# launch_embed12.sh — 启动 12 路 embed（分片→配置映射固定）
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
DB_PATH="$(runtime_env_path EMBED_DB "data/drbrain.db" "embedding database")"
cd "$ROOT"
IDS_DIR="$(runtime_env_path EMBED_IDS_DIR "$ROOT/data/.runtime" "embedding ids directory")"
EMBED_BASE_CFG="$(runtime_path "$(runtime_env_selector DRBRAIN_CONFIG config.yaml "DRBRAIN_CONFIG")" "base config")"
export DRBRAIN_CONFIG="$EMBED_BASE_CFG"
export DRBRAIN_CONFIG_PATH="$EMBED_BASE_CFG"
PYTHON_BIN="${DRBRAIN_PYTHON:-}"
if [[ -n "$PYTHON_BIN" ]]; then
  # An explicitly supplied interpreter is an intentional shared deployment
  # dependency; do not reinterpret it as a runtime data path.
  [[ -x "$PYTHON_BIN" ]] || runtime_die "Python executable is not usable: $PYTHON_BIN"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  # Preserve the historical worktree-local environment when root is the
  # checkout itself.
  PYTHON_BIN="$ROOT/.venv/bin/python"
elif [[ -x "$SOURCE_ROOT/.venv/bin/python" ]]; then
  # A data-only runtime root uses the source checkout's environment.
  PYTHON_BIN="$SOURCE_ROOT/.venv/bin/python"
else
  echo "Python executable not found in runtime or source checkout" >&2
  exit 127
fi
RUN_TAG="$(runtime_run_tag)"
TEMP_ROOT="$(runtime_env_path DRBRAIN_TEMP_ROOT "$ROOT/data/.runtime" "temporary root")"
export DRBRAIN_TEMP_ROOT="$TEMP_ROOT"
LOG_DIR="$(runtime_env_path DRBRAIN_LOG_DIR "$TEMP_ROOT/drbrain-$RUN_TAG" "log directory")"
LOG_DIR="$(runtime_prepare_dir "$LOG_DIR" "log directory")"
export DRBRAIN_LOG_DIR="$LOG_DIR"

PIDS=()
trap runtime_cleanup_workers_on_failure EXIT
trap 'exit 143' HUP INT TERM

[[ -d "$IDS_DIR" ]] || runtime_die "embedding ids directory does not exist: $IDS_DIR"

declare -A CFG
CFG[0]=e8006; CFG[1]=e8007; CFG[2]=e8008; CFG[3]=e8009
CFG[4]=e8010; CFG[5]=e8011; CFG[6]=e8012; CFG[7]=e8013
CFG[8]=e8006; CFG[9]=e8008; CFG[10]=e8010; CFG[11]=e8012

for i in 0 1 2 3 4 5 6 7 8 9 10 11; do
  IDS_FILE="$(runtime_existing_path "$IDS_DIR/emb_shard$i.txt" "embedding ids file")"
  [[ -f "$IDS_FILE" ]] || runtime_die "embedding ids file is not regular: $IDS_FILE"
  CFG_FILE="$(runtime_path "config.${CFG[$i]}.yaml" "embedding config")"
  LOG_FILE="$(runtime_path "$LOG_DIR/embed_s$i.log" "embedding log file")"
  nohup "$PYTHON_BIN" -u "$SCRIPT_DIR/embed_batch.py" \
    --ids-file "$IDS_FILE" \
    --config "$CFG_FILE" \
    --db "$DB_PATH" --skip-raptor >> "$LOG_FILE" 2>&1 &
  pid=$!
  PIDS+=("$pid")
  echo "s$i -> ${CFG[$i]} (pid $pid)"
done
echo "=== 12 路 embed 启动（logs: $LOG_DIR） ==="

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
    echo "=== embed worker failure detected (exit $wait_status; see logs: $LOG_DIR) ===" >&2
    exit "$wait_status"
  fi
  echo "=== 12 个 embed worker 全部完成 ==="
fi
