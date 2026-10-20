#!/usr/bin/env bash
set -euo pipefail
umask 022
source "/NHNHOME/WORKSPACE/26mfds002_B/jhtak/solar-open-2/project/launch/b200/common_env.sh"
source "$B200_LAUNCH_DIR/gpu_idle_gate.sh"
b200_prepare_mutable_dirs
b200_assert_common_contract
b200_acquire_stage_lock
b200_gate_receipts probe
b200_require_fresh_path "$B200_KERNEL_RECEIPT"
b200_begin_log "$SOLAR_RUNTIME/logs/b200_kernel_probe_v1.log"
b200_use_gpu_6_7
b200_gpu_idle_gate
cd "$SOLAR_PROJECT"
"$B200_PYTHON" "$SOLAR_PROJECT/migration/b200/probe_b200.py" \
  --config-dir "$B200_CONFIG_DIR" \
  --environment-receipt "$B200_ENV_RECEIPT" \
  --receipt "$B200_KERNEL_RECEIPT"
