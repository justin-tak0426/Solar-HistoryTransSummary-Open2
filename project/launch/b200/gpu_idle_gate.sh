#!/usr/bin/env bash

if ! declare -F b200_die >/dev/null 2>&1; then
  # shellcheck source=common_env.sh
  source "/NHNHOME/WORKSPACE/26mfds002_B/jhtak/solar-open-2/project/launch/b200/common_env.sh"
fi

b200_gpu_idle_gate() {
  [[ "${CUDA_VISIBLE_DEVICES:-}" == "6,7" ]] || b200_die "GPU gate requires CUDA_VISIBLE_DEVICES=6,7"
  [[ -x /usr/bin/nvidia-smi ]] || b200_die "nvidia-smi is unavailable"
  local identity_output
  local process_output
  identity_output="$(/usr/bin/nvidia-smi -i 6,7 --query-gpu=index,name,memory.total,memory.used,compute_cap,driver_version --format=csv,noheader,nounits)" || b200_die "GPU identity query failed"
  printf '%s\n' "$identity_output"
  if ! printf '%s\n' "$identity_output" | awk -F, '
    function trim(value) { gsub(/^[[:space:]]+|[[:space:]]+$/, "", value); return value }
    {
      gpu_index = trim($1); name = toupper(trim($2)); total = trim($3) + 0;
      used = trim($4) + 0; capability = trim($5);
      if (gpu_index != 6 && gpu_index != 7) exit 1;
      if (seen[gpu_index]++) exit 1;
      if (name !~ /B200/) exit 1;
      if (total < 163840 || used > 1024) exit 1;
      if (capability != "10.0") exit 1;
      rows++;
    }
    END { if (rows != 2 || !seen[6] || !seen[7]) exit 1 }
  '; then
    b200_die "GPU 6,7 identity, memory, or compute capability gate failed"
  fi
  process_output="$(/usr/bin/nvidia-smi -i 6,7 --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits)" || b200_die "GPU compute-process query failed"
  if [[ -n "${process_output//[[:space:]]/}" ]]; then
    printf '%s\n' "$process_output" >&2
    b200_die "GPU 6 or 7 has an unknown compute process; no process was terminated"
  fi
  printf '[b200-gpu-gate] PASS physical=6,7 logical=0,1 idle_threshold_mib=1024\n'
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  b200_prepare_mutable_dirs
  b200_assert_common_contract
  b200_acquire_stage_lock
  b200_use_gpu_6_7
  b200_gpu_idle_gate
fi
