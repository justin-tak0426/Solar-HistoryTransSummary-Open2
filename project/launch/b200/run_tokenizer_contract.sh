#!/usr/bin/env bash
set -euo pipefail
umask 022
source "/NHNHOME/WORKSPACE/26mfds002_B/jhtak/solar-open-2/project/launch/b200/common_env.sh"
b200_prepare_mutable_dirs
b200_assert_common_contract
b200_acquire_stage_lock
b200_use_cpu_only
b200_gate_receipts tokenizer
b200_require_fresh_path "$B200_TOKENIZER_RECEIPT"
b200_begin_log "$SOLAR_RUNTIME/logs/b200_tokenizer_contract_v1.log"
cd "$SOLAR_PROJECT"
"$B200_PYTHON" "$SOLAR_PROJECT/scripts/check_tokenizer_output_contract.py" \
  --config-dir "$B200_CONFIG_DIR" \
  --source-dir "$B200_SOURCE_DIR" \
  --data-dir "$SOLAR_DATA" \
  --receipt "$B200_TOKENIZER_RECEIPT"
