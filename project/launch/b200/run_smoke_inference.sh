#!/usr/bin/env bash
set -euo pipefail
umask 022
source "/NHNHOME/WORKSPACE/26mfds002_B/jhtak/solar-open-2/project/launch/b200/common_env.sh"
source "$B200_LAUNCH_DIR/gpu_idle_gate.sh"
b200_prepare_mutable_dirs
b200_assert_common_contract
b200_acquire_stage_lock
b200_gate_receipts infer-smoke
b200_require_existing_file "$B200_SMOKE_RUN_DIR/_SUCCESS"
b200_require_existing_file "$B200_SMOKE_RUN_DIR/run_manifest.json"
b200_require_existing_file "$B200_SMOKE_RUN_DIR/final_adapter/_SUCCESS"
b200_require_fresh_path "$B200_SMOKE_INFER_DIR"
b200_begin_log "$SOLAR_RUNTIME/logs/b200_smoke_inference_v1.log"
mkdir -p "$(dirname "$B200_BASE_PREDICTIONS")" "$(dirname "$B200_TUNED_PREDICTIONS")"
b200_use_gpu_6_7
b200_gpu_idle_gate
cd "$SOLAR_PROJECT"
"$B200_PYTHON" "$SOLAR_PROJECT/scripts/infer.py" \
  --config-dir "$B200_CONFIG_DIR" \
  --mode b200-validation-smoke \
  --source-dir "$B200_SOURCE_DIR" \
  --expert-nf4-dir "$B200_INFER_EXPERT_DIR" \
  --asset-integrity-receipt "$B200_ASSET_RECEIPT" \
  --model-label base \
  --eval-jsonl "$SOLAR_DATA/validation.jsonl" \
  --output-jsonl "$B200_BASE_PREDICTIONS" \
  --tensor-parallel-size 2 \
  --max-model-len 16384 \
  --translation-max-new-tokens 8192 \
  --summarization-max-new-tokens 6144 \
  --generation-batch-size 2
b200_gpu_idle_gate
"$B200_PYTHON" "$SOLAR_PROJECT/scripts/infer.py" \
  --config-dir "$B200_CONFIG_DIR" \
  --mode b200-validation-smoke \
  --source-dir "$B200_SOURCE_DIR" \
  --expert-nf4-dir "$B200_INFER_EXPERT_DIR" \
  --asset-integrity-receipt "$B200_ASSET_RECEIPT" \
  --adapter-dir "$B200_SMOKE_RUN_DIR/final_adapter" \
  --model-label tuned \
  --eval-jsonl "$SOLAR_DATA/validation.jsonl" \
  --output-jsonl "$B200_TUNED_PREDICTIONS" \
  --tensor-parallel-size 2 \
  --max-model-len 16384 \
  --translation-max-new-tokens 8192 \
  --summarization-max-new-tokens 6144 \
  --generation-batch-size 2
