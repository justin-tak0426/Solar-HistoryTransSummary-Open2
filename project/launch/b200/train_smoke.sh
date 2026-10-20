#!/usr/bin/env bash
set -euo pipefail
umask 022
source "/NHNHOME/WORKSPACE/26mfds002_B/jhtak/solar-open-2/project/launch/b200/common_env.sh"
source "$B200_LAUNCH_DIR/gpu_idle_gate.sh"
b200_prepare_mutable_dirs
b200_assert_common_contract
b200_acquire_stage_lock
b200_gate_receipts train-smoke
b200_require_fresh_path "$B200_SMOKE_RUN_DIR"
b200_begin_log "$SOLAR_RUNTIME/logs/b200_smoke_len9600_v1.log"
export WANDB_MODE=disabled
export WANDB_DIR="$SOLAR_RUNTIME/wandb/b200_smoke_len9600_v2"
export WANDB_ENTITY="junhyuk-tak-markr"
export WANDB_PROJECT="modern-korean-lm"
export WANDB_RUN_ID="solar_open2_b200_smoke_20260802_v2"
export WANDB_NAME="Solar Open2 250B 2xB200 smoke len9600 optimized v2"
b200_require_fresh_path "$WANDB_DIR"
b200_use_gpu_6_7
b200_gpu_idle_gate
mkdir -p "$WANDB_DIR"
cd "$SOLAR_PROJECT"
"$B200_PYTHON" "$SOLAR_PROJECT/scripts/train.py" \
  --config-dir "$B200_CONFIG_DIR" \
  --source-dir "$B200_SOURCE_DIR" \
  --expert-nf4-dir "$B200_TRAIN_EXPERT_DIR" \
  --data-dir "$SOLAR_DATA" \
  --output-dir "$B200_SMOKE_RUN_DIR" \
  --asset-integrity-receipt "$B200_ASSET_RECEIPT" \
  --preflight-receipt "$B200_PREFLIGHT_RECEIPT" \
  --smoke \
  --wandb-mode disabled
