#!/usr/bin/env bash
#SBATCH --job-name=mk03-solar2-eval-es-v2
#SBATCH --partition=batch
#SBATCH --qos=high
#SBATCH --gres=gpu:rtx6000:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@7200
#SBATCH --open-mode=append
#SBATCH --output=/data/home/mk03/workspace/model_training/Solar-HistoryTransSummary-Open2/outputs/logs/eval_earlystop_v2_%j.out
#SBATCH --error=/data/home/mk03/workspace/model_training/Solar-HistoryTransSummary-Open2/outputs/logs/eval_earlystop_v2_%j.out

set -euo pipefail

PROJECT=/data/home/mk03/workspace/model_training/Solar-HistoryTransSummary-Open2
RUNTIME_ROOT=/data1/jhtak/Solar-HistoryTransSummary-Open2
REVISION=8d1dda5ad2af58207c65348d825772d2f4d72481
SOURCE_DIR=$RUNTIME_ROOT/hf_cache/models--upstage--Solar-Open2-250B/snapshots/$REVISION
INFER_EXPERT_DIR=$RUNTIME_ROOT/models/Solar-Open2-250B-NF4-per-expert-fallback
DATA_DIR=/data/home/mk03/workspace/model_training/modern_korean_lm/artifacts/nodict_official_20260719/data
TRAIN_OUT=${TRAIN_OUT:-$RUNTIME_ROOT/runs/train_2epoch_earlystop_step18631_len9600_v2}
EVAL_OUT=${EVAL_OUT:-$RUNTIME_ROOT/runs/eval_2epoch_earlystop_validation_best_v2}
PYTHON=$PROJECT/solar-open-2/bin/python
REFERENCE=$PROJECT/scripts/reference/shared_nodict_evaluation.py
REFERENCE_SOURCE=/data/home/mk01/NIPAHistoryTransSummary/scripts/shared_nodict_evaluation.py
REFERENCE_SHA=a6e68d29f3db569ff78283f15b10680c11c2ff9f6472453dc8fd6ef9a7320b23
BASE_LABEL=solar_open2_base_nf4
TUNED_LABEL=Solar-HistoryTransSummary-Open2-validation-best
BASE_COMBINED=$EVAL_OUT/base/combined_predictions.jsonl
TUNED_COMBINED=$EVAL_OUT/tuned/combined_predictions.jsonl
ADAPTER_RECEIPT=$EVAL_OUT/best_adapter_preflight.json
BASE_SPLIT_RECEIPT=$EVAL_OUT/base/split_receipt.json
TUNED_SPLIT_RECEIPT=$EVAL_OUT/tuned/split_receipt.json
FULL_METRICS_RECEIPT=$EVAL_OUT/full_test/metrics/metrics.receipt.json
GOLD_METRICS_RECEIPT=$EVAL_OUT/human_gold/metrics/metrics.receipt.json

[[ "$(id -un)" == "mk03" ]] || exit 2
[[ -x "$PYTHON" && -d "$SOURCE_DIR" && -d "$INFER_EXPERT_DIR" ]] || exit 2
[[ -f "$TRAIN_OUT/_SUCCESS" && -f "$TRAIN_OUT/best_adapter/_SUCCESS" ]] || {
  echo "Validation-best adapter from the authorized terminal run is incomplete" >&2
  exit 2
}
for path in "$SOURCE_DIR" "$INFER_EXPERT_DIR" "$TRAIN_OUT" "$EVAL_OUT"; do
  resolved=$(realpath -m "$path")
  case "$resolved" in
    "$RUNTIME_ROOT"/*) ;;
    *) echo "Refusing path outside $RUNTIME_ROOT: $resolved" >&2; exit 2 ;;
  esac
done
[[ "$(sha256sum "$REFERENCE" | awk '{print $1}')" == "$REFERENCE_SHA" ]] || exit 2
[[ "$(sha256sum "$REFERENCE_SOURCE" | awk '{print $1}')" == "$REFERENCE_SHA" ]] || exit 2
cmp -s "$REFERENCE" "$REFERENCE_SOURCE" || exit 2

mkdir -p "$EVAL_OUT" "$RUNTIME_ROOT/cache" "$RUNTIME_ROOT/wandb"
cd "$PROJECT"

export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=$RUNTIME_ROOT/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR=$RUNTIME_ROOT/cache/triton_eval_earlystop_v2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_ENTITY=${WANDB_ENTITY:-junhyuk-tak-markr}
export WANDB_PROJECT=${WANDB_PROJECT:-modern-korean-lm}

requeue_requested=0
child_pid=""

request_requeue() {
  requeue_requested=1
  echo "[slurm] USR1 received; requesting an evaluation restart" >&2
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill -USR1 "$child_pid" 2>/dev/null || true
  fi
}
trap request_requeue USR1

requeue_self() {
  [[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || exit 75
  record=$(scontrol --oneliner show job "$SLURM_JOB_ID")
  if [[ "$record" != *"JobId=${SLURM_JOB_ID} "* || "$record" != *"UserId=mk03("* ]]; then
    echo "Current-job ownership verification failed; refusing requeue" >&2
    exit 75
  fi
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}

run_step() {
  if (( requeue_requested )); then
    requeue_self
  fi
  "$@" &
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
  if (( requeue_requested )) || [[ "$status" -eq 75 ]]; then
    requeue_self
  fi
  return "$status"
}

run_step "$PYTHON" -u scripts/check_environment.py \
  --require-cuda-count 3 \
  --receipt "$RUNTIME_ROOT/receipts/environment_eval.json"

run_step "$PYTHON" -u scripts/verify_final_adapter.py \
  --training-manifest "$TRAIN_OUT/run_manifest.json" \
  --adapter-dir "$TRAIN_OUT/best_adapter" \
  --receipt "$ADAPTER_RECEIPT"

run_step "$PYTHON" -u scripts/infer.py \
  --source-dir "$SOURCE_DIR" \
  --expert-nf4-dir "$INFER_EXPERT_DIR" \
  --model-label "$BASE_LABEL" \
  --eval-jsonl "$DATA_DIR/combined_test_and_gold_nodict.jsonl" \
  --output-jsonl "$BASE_COMBINED" \
  --max-model-len 16384 \
  --translation-max-new-tokens 8192 \
  --summarization-max-new-tokens 6144 \
  --generation-batch-size 4 \
  --resume

run_step "$PYTHON" -u scripts/infer.py \
  --source-dir "$SOURCE_DIR" \
  --expert-nf4-dir "$INFER_EXPERT_DIR" \
  --adapter-dir "$TRAIN_OUT/best_adapter" \
  --model-label "$TUNED_LABEL" \
  --eval-jsonl "$DATA_DIR/combined_test_and_gold_nodict.jsonl" \
  --output-jsonl "$TUNED_COMBINED" \
  --max-model-len 16384 \
  --translation-max-new-tokens 8192 \
  --summarization-max-new-tokens 6144 \
  --generation-batch-size 4 \
  --resume

run_step "$PYTHON" -u scripts/audit_outputs.py \
  --model "$BASE_LABEL" "$BASE_COMBINED" \
  --model "$TUNED_LABEL" "$TUNED_COMBINED" \
  --eval-jsonl "$DATA_DIR/combined_test_and_gold_nodict.jsonl" \
  --training-manifest "$TRAIN_OUT/run_manifest.json" \
  --tuned-label "$TUNED_LABEL" \
  --output-json "$EVAL_OUT/output_audit.json" \
  --output-md "$EVAL_OUT/output_audit.md"

for role in base tuned; do
  if [[ "$role" == "base" ]]; then
    model_label=$BASE_LABEL
    split_receipt=$BASE_SPLIT_RECEIPT
  else
    model_label=$TUNED_LABEL
    split_receipt=$TUNED_SPLIT_RECEIPT
  fi
  run_step "$PYTHON" -u scripts/finalize.py split \
    --combined-predictions "$EVAL_OUT/$role/combined_predictions.jsonl" \
    --model-label "$model_label" \
    --audit-json "$EVAL_OUT/output_audit.json" \
    --full-eval "$DATA_DIR/full_test_nodict.jsonl" \
    --human-gold-eval "$DATA_DIR/human_gold_nodict.jsonl" \
    --full-predictions "$EVAL_OUT/$role/full_test/predictions.jsonl" \
    --gold-predictions "$EVAL_OUT/$role/human_gold/predictions.jsonl" \
    --receipt "$split_receipt"
done

METRIC_HF=/tmp/mk03/solar_metric_hf_${SLURM_JOB_ID}
rm -rf "$METRIC_HF"
mkdir -p "$METRIC_HF/hub"
run_step cp -r --no-preserve=mode,ownership,timestamps \
  /data/home/mk01/NIPAHistoryTransSummary/.hf_home/hub/models--xlm-roberta-large \
  "$METRIC_HF/hub/"
export HF_HOME=$METRIC_HF

run_step "$PYTHON" -u scripts/finalize.py score-and-seal \
  --dataset full_test \
  --split-receipt "$BASE_SPLIT_RECEIPT" \
  --split-receipt "$TUNED_SPLIT_RECEIPT" \
  --audit-json "$EVAL_OUT/output_audit.json" \
  --out-dir "$EVAL_OUT/full_test/metrics" \
  --receipt "$FULL_METRICS_RECEIPT"

run_step "$PYTHON" -u scripts/finalize.py score-and-seal \
  --dataset human_gold \
  --split-receipt "$BASE_SPLIT_RECEIPT" \
  --split-receipt "$TUNED_SPLIT_RECEIPT" \
  --audit-json "$EVAL_OUT/output_audit.json" \
  --out-dir "$EVAL_OUT/human_gold/metrics" \
  --receipt "$GOLD_METRICS_RECEIPT"

run_step "$PYTHON" -u scripts/finalize.py report \
  --full-summary "$EVAL_OUT/full_test/metrics/metrics.summary.json" \
  --gold-summary "$EVAL_OUT/human_gold/metrics/metrics.summary.json" \
  --full-metrics-receipt "$FULL_METRICS_RECEIPT" \
  --gold-metrics-receipt "$GOLD_METRICS_RECEIPT" \
  --training-manifest "$TRAIN_OUT/run_manifest.json" \
  --audit-json "$EVAL_OUT/output_audit.json" \
  --base-label "$BASE_LABEL" \
  --tuned-label "$TUNED_LABEL" \
  --output-json "$EVAL_OUT/report.json" \
  --output-md "$EVAL_OUT/report.md"

run_step "$PYTHON" -u scripts/upload_wandb.py \
  --report-json "$EVAL_OUT/report.json" \
  --report-md "$EVAL_OUT/report.md" \
  --training-manifest "$TRAIN_OUT/run_manifest.json" \
  --audit-json "$EVAL_OUT/output_audit.json" \
  --audit-md "$EVAL_OUT/output_audit.md" \
  --full-summary "$EVAL_OUT/full_test/metrics/metrics.summary.json" \
  --gold-summary "$EVAL_OUT/human_gold/metrics/metrics.summary.json" \
  --full-metrics-receipt "$FULL_METRICS_RECEIPT" \
  --gold-metrics-receipt "$GOLD_METRICS_RECEIPT" \
  --receipt "$EVAL_OUT/wandb_upload.json" \
  --success-marker "$EVAL_OUT/_WANDB_SUCCESS" \
  --wandb-dir "$RUNTIME_ROOT/wandb" \
  --entity "$WANDB_ENTITY" \
  --project "$WANDB_PROJECT" \
  --run-id solar_open2_hardterm_eval_earlystop_20260727_v2 \
  --run-name "Solar Open2 early-stop evaluation base vs validation-best test and human-gold" \
  --training-run-id solar_open2_hardterm_2epoch_earlystop_20260727_v2

rm -rf "$METRIC_HF"
echo "SOLAR_OPEN2_VALIDATION_BEST_EVAL_DONE job=${SLURM_JOB_ID} report=${EVAL_OUT}/report.md"
