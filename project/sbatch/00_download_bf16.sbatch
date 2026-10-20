#!/usr/bin/env bash
#SBATCH --job-name=mk03-solar2-download
#SBATCH --partition=batch
#SBATCH --qos=high
#SBATCH --nodelist=gpu03
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@1200
#SBATCH --open-mode=append
#SBATCH --output=/data/home/mk03/workspace/model_training/Solar-HistoryTransSummary-Open2/outputs/logs/download_%j.out
#SBATCH --error=/data/home/mk03/workspace/model_training/Solar-HistoryTransSummary-Open2/outputs/logs/download_%j.out

set -euo pipefail

PROJECT=/data/home/mk03/workspace/model_training/Solar-HistoryTransSummary-Open2
RUNTIME_ROOT=/data1/jhtak/Solar-HistoryTransSummary-Open2
CACHE_DIR=$RUNTIME_ROOT/hf_cache
RECEIPT=$RUNTIME_ROOT/receipts/bf16_download.json
PYTHON=${STAGE_PYTHON:-$PROJECT/solar-open-2/bin/python}

[[ "$(id -un)" == "mk03" ]] || exit 2
for path in "$CACHE_DIR" "$RECEIPT"; do
  resolved=$(realpath -m "$path")
  case "$resolved" in
    "$RUNTIME_ROOT"/*) ;;
    *) echo "Refusing path outside $RUNTIME_ROOT: $resolved" >&2; exit 2 ;;
  esac
done
[[ -x "$PYTHON" ]] || exit 2
mkdir -p "$CACHE_DIR" "$RUNTIME_ROOT/receipts"
cd "$PROJECT"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$CACHE_DIR"
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_XET_HIGH_PERFORMANCE=1

child_pid=""
requeue_requested=0
request_requeue() {
  requeue_requested=1
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill -TERM "$child_pid" 2>/dev/null || true
  fi
}
trap request_requeue USR1

"$PYTHON" -u scripts/download_bf16_snapshot.py \
  --cache-dir "$CACHE_DIR" \
  --receipt "$RECEIPT" \
  --max-workers 8 &
child_pid=$!
set +e
while true; do
  wait "$child_pid"
  status=$?
  if kill -0 "$child_pid" 2>/dev/null; then
    continue
  fi
  # A trapped USR1 can interrupt wait before the child exits. Reap once more
  # after it is no longer alive so status is the child's actual exit status.
  wait "$child_pid"
  status=$?
  break
done
set -e
child_pid=""

if (( requeue_requested )); then
  [[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || exit 75
  record=$(scontrol --oneliner show job "$SLURM_JOB_ID")
  if [[ "$record" != *"JobId=${SLURM_JOB_ID} "* || "$record" != *"UserId=mk03("* ]]; then
    echo "Current-job ownership verification failed; refusing requeue" >&2
    exit 75
  fi
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$status"
