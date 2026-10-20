#!/usr/bin/env bash
set -euo pipefail
umask 022
source "/NHNHOME/WORKSPACE/26mfds002_B/jhtak/solar-open-2/project/launch/b200/common_env.sh"
b200_prepare_mutable_dirs
b200_assert_common_contract
b200_acquire_stage_lock
b200_use_cpu_only
b200_gate_receipts cpu-tests
b200_begin_log "$SOLAR_RUNTIME/logs/b200_cpu_tests_v1.log"
cd "$SOLAR_PROJECT"
"$B200_PYTHON" -m unittest discover -s scripts -p 'test_*.py'
"$B200_PYTHON" -m unittest discover -s migration/b200 -p 'test_*.py'
