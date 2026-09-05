#!/usr/bin/env bash
# Shared path boundary helpers for standalone pipeline launchers.
#
# This file is sourced by the shell wrappers.  It deliberately does not set
# shell options so that the caller retains control of ``set -euo pipefail``.
# Pipeline logs and intermediate manifests can contain provider errors or
# identifiers; keep newly-created artifacts private by default.
umask 077

runtime_die() {
  # Do not echo path/selector values: launchers persist stderr in logs and a
  # malformed value may contain a credential or another worktree's location.
  local message="${*%%:*}"
  printf 'runtime isolation error: %s\n' "$message" >&2
  exit 2
}

runtime_realpath() {
  local value="${1-}"
  if command -v realpath >/dev/null 2>&1; then
    realpath -m -- "$value"
    return
  fi
  if command -v readlink >/dev/null 2>&1; then
    readlink -m -- "$value"
    return
  fi
  runtime_die "realpath/readlink is required to resolve pipeline paths"
}

# ``realpath -m`` is useful for containment checks, but it follows symlinks.
# Inspect the lexical path first so a root alias (including an intermediate
# component) cannot silently retarget a launcher into another checkout.
runtime_reject_symlink_components() {
  local value="${1-}"
  [[ -n "$value" ]] || runtime_die "path is empty"
  local candidate="$value"
  if [[ "$candidate" != /* ]]; then
    candidate="$PWD/$candidate"
  fi

  local current="/"
  local part
  local -a parts=()
  IFS='/' read -r -a parts <<< "$candidate"
  for part in "${parts[@]}"; do
    [[ -z "$part" || "$part" == "." ]] && continue
    if [[ "$part" == ".." ]]; then
      current="${current%/*}"
      [[ -n "$current" ]] || current="/"
      continue
    fi
    if [[ "$current" == "/" ]]; then
      current="/$part"
    else
      current="$current/$part"
    fi
    if [[ -L "$current" ]]; then
      runtime_die "path must not contain a symlink component: $current"
    fi
  done
}

runtime_init_root() {
  local candidate="${1-}"
  [[ -n "$candidate" ]] || runtime_die "runtime root is empty"
  # Keep the namespace identity explicit.  Resolving a symlink here would make
  # a seemingly isolated invocation write into another checkout (for example
  # the shared phy worktree), so callers must pass the real directory path.
  runtime_reject_symlink_components "$candidate"
  [[ -d "$candidate" ]] || runtime_die "runtime root does not exist: $candidate"
  RUNTIME_ROOT="$(runtime_realpath "$candidate")" || runtime_die "cannot resolve runtime root: $candidate"
  [[ -d "$RUNTIME_ROOT" ]] || runtime_die "runtime root is not a directory: $RUNTIME_ROOT"
  export RUNTIME_ROOT
  export DRBRAIN_ROOT="$RUNTIME_ROOT"
  # Keep both legacy and current runtime selectors aligned for child
  # processes; an inherited value must never point at another worktree.
  export DRBRAIN_RUNTIME_ROOT="$RUNTIME_ROOT"
}

# Select the inherited namespace without treating an explicitly empty
# selector as unset.  Keeping this in the shared helper ensures every
# launcher applies the same precedence as ``drbrain.runtime.runtime_root``.
runtime_init_selected_root() {
  local fallback="${1-}"
  if [[ "${DRBRAIN_ROOT+x}" == x ]]; then
    runtime_init_root "$DRBRAIN_ROOT"
  elif [[ "${DRBRAIN_RUNTIME_ROOT+x}" == x ]]; then
    runtime_init_root "$DRBRAIN_RUNTIME_ROOT"
  else
    runtime_init_root "$fallback"
  fi
}

# Stop and reap worker PIDs owned by a launcher.  The launchers intentionally
# keep the array name ``PIDS`` so this helper can be shared without requiring
# Bash nameref support.  Invalid/stale entries are ignored; a cleanup path
# must never mask the original failure status.
runtime_stop_workers() {
  local pid
  for pid in "${PIDS[@]-}"; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    [[ "$pid" == "$$" ]] && continue
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${PIDS[@]-}"; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    [[ "$pid" == "$$" ]] && continue
    wait "$pid" 2>/dev/null || true
  done
}

# EXIT-trap adapter: preserve a successful detached launcher, but close down
# workers whenever setup or signal handling exits non-zero.  This also covers
# ``--no-wait``: that mode is detached only after every worker was started.
runtime_cleanup_workers_on_failure() {
  local status=$?
  if [[ "$status" -ne 0 ]]; then
    runtime_stop_workers
  fi
  return "$status"
}

# Resolve a path against RUNTIME_ROOT.  ``mode=input`` is reserved for a
# read-only source supplied by the caller; every mutable path uses the default
# ``mode=write`` and must remain inside the selected worktree.
runtime_path() {
  local value="${1-}"
  local label="${2:-path}"
  local mode="${3:-write}"
  [[ -n "$value" ]] || runtime_die "$label is empty"
  # URI schemes are never valid local pipeline paths.  Without this guard,
  # ``file:...`` and ``https://...`` would be interpreted as relative names
  # beneath the runtime root and could hide a caller configuration mistake.
  if [[ "$value" =~ ^[A-Za-z][A-Za-z0-9+.-]*: ]]; then
    runtime_die "$label must be a local filesystem path: $value"
  fi

  local candidate="$value"
  if [[ "$candidate" != /* ]]; then
    candidate="$RUNTIME_ROOT/$candidate"
  fi
  if [[ "$mode" != "external" ]]; then
    runtime_reject_symlink_components "$candidate"
  fi
  candidate="$(runtime_realpath "$candidate")" || runtime_die "cannot resolve $label: $value"

  if [[ "$mode" != "input" && "$mode" != "external" ]]; then
    case "$candidate" in
      "$RUNTIME_ROOT"|"$RUNTIME_ROOT"/*) ;;
      *) runtime_die "$label escapes runtime root $RUNTIME_ROOT: $value" ;;
    esac
  fi
  printf '%s\n' "$candidate"
}

runtime_existing_path() {
  local value="$1"
  local label="${2:-path}"
  local mode="${3:-write}"
  local resolved
  resolved="$(runtime_path "$value" "$label" "$mode")"
  [[ -e "$resolved" ]] || runtime_die "$label does not exist: $resolved"
  printf '%s\n' "$resolved"
}

runtime_prepare_dir() {
  local value="$1"
  local label="${2:-directory}"
  local mode="${3:-write}"
  [[ "$mode" == "write" ]] || runtime_die "$label is mutable and cannot use external mode"
  local resolved
  resolved="$(runtime_path "$value" "$label" "$mode")"
  if [[ -e "$resolved" && ! -d "$resolved" ]]; then
    runtime_die "$label is not a directory: $resolved"
  fi
  mkdir -p -- "$resolved" || runtime_die "cannot create $label: $resolved"
  [[ -d "$resolved" ]] || runtime_die "$label is not a directory: $resolved"
  printf '%s\n' "$resolved"
}

runtime_env_path() {
  local variable="$1"
  local fallback="$2"
  local label="${3:-$variable}"
  local mode="${4:-write}"
  local value
  if [[ "${!variable+x}" == x ]]; then
    value="${!variable}"
    [[ -n "$value" ]] || runtime_die "$label is empty"
  else
    value="$fallback"
  fi
  runtime_path "$value" "$label" "$mode"
}

# Select an environment-backed scalar without treating an explicitly empty
# value as absent.  Path selectors use ``runtime_env_path``; this helper is for
# flags and config names that are validated by their caller.
runtime_env_selector() {
  local variable="$1"
  local fallback="$2"
  local label="${3:-$variable}"
  local value
  if [[ "${!variable+x}" == x ]]; then
    value="${!variable}"
    [[ -n "$value" ]] || runtime_die "$label is empty"
  else
    value="$fallback"
  fi
  printf '%s\n' "$value"
}

runtime_run_tag() {
  local root_tag
  root_tag="$(printf '%s' "$RUNTIME_ROOT" | sha256sum | cut -c1-8)"
  local raw
  if [[ "${DRBRAIN_RUN_ID+x}" == x ]]; then
    [[ -n "$DRBRAIN_RUN_ID" ]] || runtime_die "DRBRAIN_RUN_ID is empty"
    raw="$DRBRAIN_RUN_ID"
  else
    raw="$(basename "$RUNTIME_ROOT")-$root_tag"
  fi
  local raw_lower="${raw,,}"
  if [[ "$raw_lower" == sk-* || "$raw_lower" == *api_key* || "$raw_lower" == *apikey* || "$raw_lower" == *token* || "$raw_lower" == *secret* || "$raw_lower" == *password* ]]; then
    raw="run-$(printf '%s' "$raw" | sha256sum | cut -c1-12)"
  fi
  local value="$raw"
  value="${value//[^A-Za-z0-9_.-]/_}"
  [[ -n "$value" ]] || value="run-$root_tag"
  if (( ${#value} > 80 )); then
    local suffix
    suffix="$(printf '%s' "$raw" | sha256sum | cut -c1-12)"
    value="${value:0:67}-$suffix"
  fi
  printf '%s\n' "$value"
}
