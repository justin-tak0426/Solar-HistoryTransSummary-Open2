#!/usr/bin/env bash
#SBATCH --job-name=mk03-solar2-nf4-infer
#SBATCH --partition=batch
#SBATCH --qos=high
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@3600
#SBATCH --open-mode=append
#SBATCH --output=/data/home/mk03/workspace/model_training/Solar-HistoryTransSummary-Open2/outputs/logs/convert_per_expert_%j.out
#SBATCH --error=/data/home/mk03/workspace/model_training/Solar-HistoryTransSummary-Open2/outputs/logs/convert_per_expert_%j.out

set -euo pipefail

PROJECT=/data/home/mk03/workspace/model_training/Solar-HistoryTransSummary-Open2
RUNTIME_ROOT=/data1/jhtak/Solar-HistoryTransSummary-Open2
REVISION=8d1dda5ad2af58207c65348d825772d2f4d72481
SOURCE_DIR=$RUNTIME_ROOT/hf_cache/models--upstage--Solar-Open2-250B/snapshots/$REVISION
OUTPUT_DIR=$RUNTIME_ROOT/models/Solar-Open2-250B-NF4-per-expert-fallback
DOWNLOAD_RECEIPT=$RUNTIME_ROOT/receipts/bf16_download.json
ENV_RECEIPT=$RUNTIME_ROOT/receipts/environment_promoted.json
TOKENIZER_RECEIPT=$RUNTIME_ROOT/receipts/tokenizer_output_contract.json
PROMOTION_SUCCESS=$RUNTIME_ROOT/receipts/environment_promotion.SUCCESS
PYTHON=$PROJECT/solar-open-2/bin/python

[[ "$(id -un)" == "mk03" ]] || exit 2
[[ -x "$PYTHON" && -f "$DOWNLOAD_RECEIPT" && -d "$SOURCE_DIR" ]] || exit 2
[[ -f "$ENV_RECEIPT" && -f "$TOKENIZER_RECEIPT" && -f "$PROMOTION_SUCCESS" ]] || exit 2
for path in \
  "$SOURCE_DIR" "$OUTPUT_DIR" "$DOWNLOAD_RECEIPT" \
  "$ENV_RECEIPT" "$TOKENIZER_RECEIPT" "$PROMOTION_SUCCESS"; do
  resolved=$(realpath -m "$path")
  case "$resolved" in
    "$RUNTIME_ROOT"/*) ;;
    *) echo "Refusing path outside $RUNTIME_ROOT: $resolved" >&2; exit 2 ;;
  esac
done
"$PYTHON" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="PASS"; assert r["snapshot_dir"]==sys.argv[2]' "$DOWNLOAD_RECEIPT" "$SOURCE_DIR"
read -r promotion_status tokenizer_sha environment_sha < "$PROMOTION_SUCCESS"
[[ "$promotion_status" == "PASS" ]] || exit 2
[[ "$(sha256sum "$TOKENIZER_RECEIPT" | awk '{print $1}')" == "$tokenizer_sha" ]] || exit 2
[[ "$(sha256sum "$ENV_RECEIPT" | awk '{print $1}')" == "$environment_sha" ]] || exit 2

mkdir -p "$RUNTIME_ROOT/models" "$RUNTIME_ROOT/cache/triton_convert_per_expert"
cd "$PROJECT"
export PYTHONDONTWRITEBYTECODE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=$RUNTIME_ROOT/hf_cache
export TRITON_CACHE_DIR=$RUNTIME_ROOT/cache/triton_convert_per_expert
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PYTHON" -u scripts/check_environment.py \
  --require-cuda-count 1 \
  --receipt "$RUNTIME_ROOT/receipts/environment_convert_per_expert.json"

child_pid=""
requeue_requested=0
request_requeue() {
  requeue_requested=1
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill -INT "$child_pid" 2>/dev/null || true
  fi
}
trap request_requeue USR1

"$PYTHON" -u -m scripts.quantization.convert \
  --source-dir "$SOURCE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda:0 \
  --experts-per-shard 16 \
  --execute &
child_pid=$!
set +e
while true; do
  wait "$child_pid"
  status=$?
  if kill -0 "$child_pid" 2>/dev/null; then
    continue
  fi
  wait "$child_pid"
  status=$?
  break
done
set -e
child_pid=""

if (( requeue_requested )); then
  [[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || exit 75
  record=$(scontrol --oneliner show job "$SLURM_JOB_ID")
  [[ "$record" == *"JobId=${SLURM_JOB_ID} "* && "$record" == *"UserId=mk03("* ]] || exit 75
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$status"
