#!/usr/bin/env bash
set -euo pipefail
umask 022
source "/NHNHOME/WORKSPACE/26mfds002_B/jhtak/solar-open-2/project/launch/b200/common_env.sh"
b200_prepare_mutable_dirs
b200_assert_common_contract
b200_acquire_stage_lock
b200_use_cpu_only
b200_gate_receipts certify-smoke
b200_require_existing_file "$B200_SMOKE_RUN_DIR/run_manifest.json"
b200_require_existing_file "$B200_BASE_PREDICTIONS"
b200_require_existing_file "$B200_TUNED_PREDICTIONS"
b200_require_fresh_path "$B200_SMOKE_RECEIPT"
b200_begin_log "$SOLAR_RUNTIME/logs/b200_smoke_certification_v1.log"
cd "$SOLAR_PROJECT"
"$B200_PYTHON" "$SOLAR_PROJECT/scripts/certify_smoke.py" \
  --config-dir "$B200_CONFIG_DIR" \
  --source-dir "$B200_SOURCE_DIR" \
  --training-expert-nf4-dir "$B200_TRAIN_EXPERT_DIR" \
  --inference-expert-nf4-dir "$B200_INFER_EXPERT_DIR" \
  --environment-receipt "$B200_ENV_RECEIPT" \
  --asset-integrity-receipt "$B200_ASSET_RECEIPT" \
  --kernel-probe-receipt "$B200_KERNEL_RECEIPT" \
  --tokenizer-contract-receipt "$B200_TOKENIZER_RECEIPT" \
  --preflight-receipt "$B200_PREFLIGHT_RECEIPT" \
  --training-manifest "$B200_SMOKE_RUN_DIR/run_manifest.json" \
  --base-predictions "$B200_BASE_PREDICTIONS" \
  --tuned-predictions "$B200_TUNED_PREDICTIONS" \
  --output "$B200_SMOKE_RECEIPT"
