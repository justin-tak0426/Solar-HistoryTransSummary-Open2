#!/usr/bin/env bash

set -euo pipefail
umask 022

readonly SOLAR_ROOT="/NHNHOME/WORKSPACE/26mfds002_B/jhtak/solar-open-2"
readonly SOLAR_PROJECT="$SOLAR_ROOT/project"
readonly SOLAR_RUNTIME="$SOLAR_ROOT/runtime"
readonly SOLAR_DATA="$SOLAR_ROOT/data"
readonly B200_LAUNCH_DIR="$SOLAR_PROJECT/launch/b200"
readonly B200_CONFIG_DIR="$SOLAR_PROJECT/config/b200"
readonly B200_REVISION="8d1dda5ad2af58207c65348d825772d2f4d72481"
readonly B200_SOURCE_DIR="$SOLAR_RUNTIME/hf_cache/models--upstage--Solar-Open2-250B/snapshots/$B200_REVISION"
readonly B200_TRAIN_EXPERT_DIR="$SOLAR_RUNTIME/models/Solar-Open2-250B-NF4-layer-flat-experts"
readonly B200_INFER_EXPERT_DIR="$SOLAR_RUNTIME/models/Solar-Open2-250B-NF4-per-expert-fallback"
readonly B200_PYTHON="$SOLAR_PROJECT/solar-open-2/bin/python"
readonly B200_METRIC_HF_HOME="$SOLAR_RUNTIME/metric_hf"
readonly B200_METRIC_HUB_CACHE="$B200_METRIC_HF_HOME/hub"

readonly B200_ASSET_RECEIPT="$SOLAR_RUNTIME/receipts/b200_asset_integrity_large_v1.json"
readonly B200_ENV_RECEIPT="$SOLAR_RUNTIME/receipts/b200_environment_v1.json"
readonly B200_KERNEL_RECEIPT="$SOLAR_RUNTIME/receipts/b200_kernel_probe_v1.json"
readonly B200_TOKENIZER_RECEIPT="$SOLAR_RUNTIME/receipts/b200_tokenizer_contract_v1.json"
readonly B200_PREFLIGHT_RECEIPT="$SOLAR_RUNTIME/receipts/b200_model_preflight_v1.json"
readonly B200_SMOKE_RECEIPT="$SOLAR_RUNTIME/receipts/b200_smoke_2gpu_len9600_v1.json"
readonly B200_APPROVAL_RECEIPT="$SOLAR_RUNTIME/receipts/b200_production_approval_v1.json"
readonly B200_TRANSFER_RECEIPT="$SOLAR_RUNTIME/receipts/b200_transfer_integrity_full.json"
readonly B200_SOURCE_RECEIPT="$SOLAR_RUNTIME/receipts/source_bf16_download.json"

readonly B200_SMOKE_RUN_DIR="$SOLAR_RUNTIME/runs/b200_smoke_len9600_v1"
readonly B200_SMOKE_INFER_DIR="$SOLAR_RUNTIME/runs/b200_smoke_inference_v1"
readonly B200_BASE_PREDICTIONS="$B200_SMOKE_INFER_DIR/base/validation_smoke.jsonl"
readonly B200_TUNED_PREDICTIONS="$B200_SMOKE_INFER_DIR/tuned/validation_smoke.jsonl"
readonly B200_PRODUCTION_RUN_DIR="$SOLAR_RUNTIME/runs/b200_1epoch_len9600_v1"
readonly B200_COMBINED_EVAL="$SOLAR_DATA/combined_test_and_gold_nodict.jsonl"
readonly B200_FULL_EVAL="$SOLAR_DATA/full_test_nodict.jsonl"
readonly B200_GOLD_EVAL="$SOLAR_DATA/human_gold_nodict.jsonl"
readonly B200_EVALUATION_DIR="$SOLAR_RUNTIME/runs/b200_evaluation_v1"
readonly B200_EVALUATION_BASE_LABEL="base"
readonly B200_EVALUATION_TUNED_LABEL="b200_best_adapter"
readonly B200_EVALUATION_BASE_COMBINED="$B200_EVALUATION_DIR/predictions/base.combined.jsonl"
readonly B200_EVALUATION_TUNED_COMBINED="$B200_EVALUATION_DIR/predictions/tuned.combined.jsonl"
readonly B200_EVALUATION_AUDIT_JSON="$B200_EVALUATION_DIR/audit/output_audit.json"
readonly B200_EVALUATION_AUDIT_MD="$B200_EVALUATION_DIR/audit/output_audit.md"
readonly B200_EVALUATION_REPORT_JSON="$B200_EVALUATION_DIR/report/evaluation_report.json"
readonly B200_EVALUATION_REPORT_MD="$B200_EVALUATION_DIR/report/evaluation_report.md"
readonly B200_EVALUATION_WANDB_DIR="$SOLAR_RUNTIME/wandb/b200_evaluation_v2"
readonly B200_EVALUATION_WANDB_RUN_ID="solar_open2_b200_evaluation_20260802_v2"
readonly B200_EVALUATION_WANDB_NAME="Solar Open2 250B base vs best adapter evaluation v1"

export HF_HOME="$SOLAR_RUNTIME/hf_cache"
export HUGGINGFACE_HUB_CACHE="$SOLAR_RUNTIME/hf_cache/hub"
export XDG_CACHE_HOME="$SOLAR_RUNTIME/cache/xdg"
export PIP_CACHE_DIR="$SOLAR_RUNTIME/cache/pip"
export UV_CACHE_DIR="$SOLAR_RUNTIME/cache/uv"
export TRANSFORMERS_CACHE="$SOLAR_RUNTIME/cache/transformers"
export TMPDIR="$SOLAR_RUNTIME/tmp"
export TORCHINDUCTOR_CACHE_DIR="$SOLAR_RUNTIME/cache/torchinductor"
export CUDA_CACHE_PATH="$SOLAR_RUNTIME/cache/cuda"
export NUMBA_CACHE_DIR="$SOLAR_RUNTIME/cache/numba"
export WANDB_DIR="$SOLAR_RUNTIME/wandb"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

b200_die() {
  printf 'B200 launcher FAIL: %s\n' "$*" >&2
  exit 1
}

b200_prepare_mutable_dirs() {
  mkdir -p \
    "$SOLAR_RUNTIME/cache/xdg" \
    "$SOLAR_RUNTIME/cache/pip" \
    "$SOLAR_RUNTIME/cache/uv" \
    "$SOLAR_RUNTIME/cache/transformers" \
    "$SOLAR_RUNTIME/cache/torchinductor" \
    "$SOLAR_RUNTIME/cache/cuda" \
    "$SOLAR_RUNTIME/cache/numba" \
    "$SOLAR_RUNTIME/logs" \
    "$SOLAR_RUNTIME/receipts" \
    "$SOLAR_RUNTIME/runs" \
    "$SOLAR_RUNTIME/tmp" \
    "$SOLAR_RUNTIME/wandb"
}

b200_assert_common_contract() {
  [[ "$(/usr/bin/id -un)" == "e202" ]] || b200_die "expected user e202"
  [[ "$(/usr/bin/realpath -e "$SOLAR_ROOT")" == "$SOLAR_ROOT" ]] || b200_die "owned root changed"
  [[ "$(/usr/bin/stat -c '%U' "$SOLAR_ROOT")" == "e202" ]] || b200_die "owned root owner changed"
  [[ "$(/usr/bin/realpath -e "$B200_LAUNCH_DIR")" == "$B200_LAUNCH_DIR" ]] || b200_die "launcher directory changed"
  [[ -x "$B200_PYTHON" ]] || b200_die "project venv Python is missing"
  [[ -d "$B200_CONFIG_DIR" ]] || b200_die "B200 config directory is missing"
  [[ -d "$B200_SOURCE_DIR" ]] || b200_die "pinned BF16 source is missing"
  [[ -d "$B200_TRAIN_EXPERT_DIR" ]] || b200_die "training NF4 artifact is missing"
  [[ -d "$B200_INFER_EXPERT_DIR" ]] || b200_die "inference NF4 artifact is missing"
  local prefix
  prefix="$(CUDA_VISIBLE_DEVICES= "$B200_PYTHON" -c 'import pathlib,sys; print(pathlib.Path(sys.prefix).resolve())')"
  [[ "$prefix" == "$SOLAR_PROJECT/solar-open-2" ]] || b200_die "venv sys.prefix changed: $prefix"
}

b200_use_cpu_only() {
  export CUDA_VISIBLE_DEVICES=
  export TRITON_CACHE_DIR="$SOLAR_RUNTIME/cache/triton_cpu"
  mkdir -p "$TRITON_CACHE_DIR"
}

b200_use_gpu_6_7() {
  export CUDA_VISIBLE_DEVICES=6,7
  export TRITON_CACHE_DIR="$SOLAR_RUNTIME/cache/triton_b200"
  mkdir -p "$TRITON_CACHE_DIR"
  [[ "$CUDA_VISIBLE_DEVICES" == "6,7" ]] || b200_die "CUDA visibility changed"
}

b200_require_existing_file() {
  local path="$1"
  local resolved
  [[ "$path" == "$SOLAR_ROOT/"* ]] || b200_die "path is outside owned root: $path"
  [[ ! -L "$path" && -f "$path" ]] || b200_die "expected regular non-symlink file: $path"
  resolved="$(/usr/bin/realpath -e "$path")"
  [[ "$resolved" == "$path" ]] || b200_die "file path changed after resolution: $path"
}

b200_require_fresh_path() {
  local path="$1"
  local parent
  local resolved_parent
  [[ "$path" == "$SOLAR_ROOT/"* ]] || b200_die "output is outside owned root: $path"
  [[ ! -e "$path" && ! -L "$path" ]] || b200_die "refusing existing output or symlink: $path"
  parent="$(dirname "$path")"
  mkdir -p "$parent"
  [[ ! -L "$parent" && -d "$parent" ]] || b200_die "output parent is invalid: $parent"
  resolved_parent="$(/usr/bin/realpath -e "$parent")"
  [[ "$resolved_parent" == "$SOLAR_ROOT" || "$resolved_parent" == "$SOLAR_ROOT/"* ]] || b200_die "output parent escaped root"
  [[ "$resolved_parent/$(basename "$path")" == "$path" ]] || b200_die "output path is not canonical: $path"
}

b200_begin_log() {
  local log_path="$1"
  b200_require_fresh_path "$log_path"
  if ! (set -o noclobber; : >"$log_path") 2>/dev/null; then
    b200_die "could not atomically create fresh log: $log_path"
  fi
  exec >>"$log_path" 2>&1
  printf '[b200-launch] log=%s stage_pid=%s\n' "$log_path" "$$"
}

b200_acquire_stage_lock() {
  local lock_path="$SOLAR_RUNTIME/tmp/b200_launcher.lock"
  [[ ! -L "$lock_path" ]] || b200_die "launcher lock is a symlink"
  exec {B200_STAGE_LOCK_FD}>"$lock_path"
  /usr/bin/flock -n "$B200_STAGE_LOCK_FD" || b200_die "another B200 launcher stage is active"
  readonly B200_STAGE_LOCK_FD
}

b200_release_stage_lock() {
  [[ -n "${B200_STAGE_LOCK_FD:-}" ]] || b200_die "stage lock is not held"
  /usr/bin/flock -u "$B200_STAGE_LOCK_FD" || b200_die "could not release stage lock"
}

b200_prepare_resumable_dir() {
  local path="$1"
  local resolved
  if [[ -e "$path" || -L "$path" ]]; then
    [[ ! -L "$path" && -d "$path" ]] || b200_die "resumable path is not a regular directory: $path"
    resolved="$(/usr/bin/realpath -e "$path")"
    [[ "$resolved" == "$path" && "$resolved" == "$SOLAR_ROOT/"* ]] || b200_die "resumable path escaped owned root: $path"
  else
    b200_require_fresh_path "$path"
    mkdir -p "$path"
  fi
}

b200_gate_receipts() {
  local stage="$1"
  CUDA_VISIBLE_DEVICES= "$B200_PYTHON" "$B200_LAUNCH_DIR/launcher_gate.py" --stage "$stage"
}

b200_run_metric_scorer() {
  HF_HOME="$B200_METRIC_HF_HOME" \
    HF_HUB_CACHE="$B200_METRIC_HUB_CACHE" \
    HUGGINGFACE_HUB_CACHE="$B200_METRIC_HUB_CACHE" \
    TRANSFORMERS_CACHE="$B200_METRIC_HUB_CACHE" \
    "$@"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  b200_die "common_env.sh is a sourced library, not a launcher"
fi
