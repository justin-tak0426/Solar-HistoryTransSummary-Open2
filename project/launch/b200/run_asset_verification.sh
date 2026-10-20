#!/usr/bin/env bash
set -euo pipefail
umask 022
source "/NHNHOME/WORKSPACE/26mfds002_B/jhtak/solar-open-2/project/launch/b200/common_env.sh"
b200_prepare_mutable_dirs
b200_assert_common_contract
b200_acquire_stage_lock
b200_use_cpu_only
b200_require_fresh_path "$B200_ASSET_RECEIPT"
b200_begin_log "$SOLAR_RUNTIME/logs/b200_asset_integrity_large_v1.log"
cd "$SOLAR_PROJECT"
"$B200_PYTHON" "$SOLAR_PROJECT/migration/b200/verify_assets.py" \
  --root "$SOLAR_ROOT" \
  --mode full \
  --large-hashes \
  --receipt "$B200_ASSET_RECEIPT"
