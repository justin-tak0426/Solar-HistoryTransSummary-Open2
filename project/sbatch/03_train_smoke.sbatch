#!/usr/bin/env bash
#SBATCH --job-name=mk03-solar2-smk-v7
#SBATCH --partition=batch
#SBATCH --qos=high
#SBATCH --gres=gpu:rtx6000:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@7200
#SBATCH --open-mode=append
#SBATCH --output=/data/home/mk03/workspace/model_training/Solar-HistoryTransSummary-Open2/outputs/logs/train_smoke_v7_9600_earlystop_%j.out
#SBATCH --error=/data/home/mk03/workspace/model_training/Solar-HistoryTransSummary-Open2/outputs/logs/train_smoke_v7_9600_earlystop_%j.out

set -euo pipefail

PROJECT=/data/home/mk03/workspace/model_training/Solar-HistoryTransSummary-Open2
RUNTIME_ROOT=/data1/jhtak/Solar-HistoryTransSummary-Open2
REVISION=8d1dda5ad2af58207c65348d825772d2f4d72481
SOURCE_DIR=$RUNTIME_ROOT/hf_cache/models--upstage--Solar-Open2-250B/snapshots/$REVISION
TRAIN_EXPERT_DIR=$RUNTIME_ROOT/models/Solar-Open2-250B-NF4-layer-flat-experts
INFER_EXPERT_DIR=$RUNTIME_ROOT/models/Solar-Open2-250B-NF4-per-expert-fallback
DOWNLOAD_RECEIPT=$RUNTIME_ROOT/receipts/bf16_download.json
DATA_DIR=/data/home/mk03/workspace/model_training/modern_korean_lm/artifacts/nodict_official_20260719/data
PYTHON=$PROJECT/solar-open-2/bin/python
TOKENIZER_CONTRACT_RECEIPT=$RUNTIME_ROOT/receipts/tokenizer_output_contract.json
PREFLIGHT=$RUNTIME_ROOT/receipts/model_preflight.json
SMOKE_OUT=$RUNTIME_ROOT/runs/gpu_smoke_v7_9600_earlystop
SMOKE_RECEIPT=$RUNTIME_ROOT/receipts/gpu_smoke_v7_9600_earlystop.json
BASE_PRED=$SMOKE_OUT/base_predictions.jsonl
TUNED_PRED=$SMOKE_OUT/tuned_predictions.jsonl

[[ "$(id -un)" == "mk03" ]] || exit 2
[[ -x "$PYTHON" && -d "$SOURCE_DIR" ]] || exit 2
[[ -d "$TRAIN_EXPERT_DIR" && -d "$INFER_EXPERT_DIR" ]] || exit 2
[[ -f "$DOWNLOAD_RECEIPT" ]] || exit 2
for path in \
  "$SOURCE_DIR" "$TRAIN_EXPERT_DIR" "$INFER_EXPERT_DIR" \
  "$DOWNLOAD_RECEIPT" "$TOKENIZER_CONTRACT_RECEIPT" "$PREFLIGHT" \
  "$SMOKE_OUT" "$SMOKE_RECEIPT"; do
  resolved=$(realpath -m "$path")
  case "$resolved" in
    "$RUNTIME_ROOT"/*) ;;
    *) echo "Refusing path outside $RUNTIME_ROOT: $resolved" >&2; exit 2 ;;
  esac
done

mkdir -p "$RUNTIME_ROOT/receipts" "$SMOKE_OUT" "$RUNTIME_ROOT/cache"
rm -f "$TOKENIZER_CONTRACT_RECEIPT" "$SMOKE_RECEIPT"
cd "$PROJECT"

export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=$RUNTIME_ROOT/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR=$RUNTIME_ROOT/cache/triton_smoke_v7_earlystop
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

requeue_requested=0
child_pid=""

request_requeue() {
  requeue_requested=1
  echo "[slurm] USR1 received; requesting a boundary-safe smoke restart" >&2
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
  --receipt "$RUNTIME_ROOT/receipts/environment_smoke.json"

run_step "$PYTHON" -c '
import json
import sys
from pathlib import Path

recipe = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "recipe_name": "solar_open2_hardterm_fancy_v3_earlystop",
    "epochs": 2,
    "optimizer_steps": 18631,
    "max_length": 9600,
}
observed = {key: recipe.get(key) for key in expected}
if observed != expected:
    raise RuntimeError(f"smoke recipe mismatch: {observed!r} != {expected!r}")
deck = recipe.get("training_deck") or {}
if (
    deck.get("policy") != "deterministic_global_augmented_epoch_deck"
    or (deck.get("total_occurrences") or {}).get("total") != 298096
):
    raise RuntimeError("smoke deterministic two-epoch deck contract is missing")
validation = recipe.get("validation") or {}
if validation.get("full_validation_steps") != [9316, 13973, 18631]:
    raise RuntimeError("smoke validation candidate schedule changed")
expected_early_stopping = {
    "enabled": True,
    "monitor": "selection_loss",
    "mode": "min",
    "minimum_epochs": 1.0,
    "minimum_step": 9316,
    "subset_baseline_policy": "last_fixed_subset_before_minimum_step",
    "subset_patience_evaluations": 3,
    "subset_min_delta": {
        "absolute": 0.001,
        "relative": 0.0025,
        "combination": "max",
    },
    "full_validation_confirmation_required": True,
    "confirmation_step": 13973,
    "full_min_delta": {
        "absolute": 0.0005,
        "relative": 0.001,
        "combination": "max",
    },
    "continue_if_either_task_materially_improves": True,
    "earliest_stop_step": 13973,
    "latest_stop_step": 18631,
    "restore_best_full_validation": True,
    "sealed_evaluation_not_monitor": True,
}
if validation.get("early_stopping") != expected_early_stopping:
    raise RuntimeError("smoke early-stopping policy changed")
' "$PROJECT/config/recipe.json"

run_step "$PYTHON" -u scripts/check_tokenizer_output_contract.py \
  --source-dir "$SOURCE_DIR" \
  --data-dir "$DATA_DIR" \
  --receipt "$TOKENIZER_CONTRACT_RECEIPT"

run_step "$PYTHON" -u scripts/preflight_model.py \
  --source-dir "$SOURCE_DIR" \
  --expert-nf4-dir "$TRAIN_EXPERT_DIR" \
  --download-receipt "$DOWNLOAD_RECEIPT" \
  --receipt "$PREFLIGHT"

run_step "$PYTHON" -u scripts/infer.py \
  --source-dir "$SOURCE_DIR" \
  --expert-nf4-dir "$INFER_EXPERT_DIR" \
  --model-label solar_open2_base_smoke_v7 \
  --eval-jsonl "$DATA_DIR/combined_test_and_gold_nodict.jsonl" \
  --output-jsonl "$BASE_PRED" \
  --max-model-len 16384 \
  --translation-max-new-tokens 512 \
  --summarization-max-new-tokens 512 \
  --generation-batch-size 2 \
  --limit-per-task 1 \
  --resume

run_step "$PYTHON" -c '
import json
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "scripts"))
from common import output_only_violations
from infer import read_prediction_rows

path = Path(sys.argv[1])
rows = read_prediction_rows(path)
if len(rows) != 2:
    raise RuntimeError(f"base smoke predictions must have exactly two rows: {len(rows)}")
tasks = {str(row.get("task") or "") for row in rows}
if tasks != {"translation", "summarization"}:
    raise RuntimeError(f"base smoke predictions must cover both tasks: {sorted(tasks)}")
for line_number, row in enumerate(rows, 1):
    prediction = row.get("prediction")
    if not isinstance(prediction, str) or not prediction.strip():
        raise RuntimeError(f"base smoke row {line_number} has an empty prediction")
    invalid_unicode = [
        f"U+{ord(character):04X}"
        for character in prediction
        if character == "\ufffd"
        or unicodedata.category(character) == "Cs"
        or (
            unicodedata.category(character) == "Cc"
            and character not in {"\n", "\t"}
        )
    ]
    if invalid_unicode:
        raise RuntimeError(
            f"base smoke row {line_number} has invalid Unicode: {invalid_unicode[:8]}"
        )
    task = str(row["task"])
    maximum = 512
    prediction_tokens = row.get("prediction_tokens")
    if (
        isinstance(prediction_tokens, bool)
        or not isinstance(prediction_tokens, int)
        or not 0 < prediction_tokens < maximum
    ):
        raise RuntimeError(
            f"base smoke row {line_number} did not terminate before its cap: "
            f"{prediction_tokens!r}/{maximum}"
        )
    if row.get("generation_stop_reason") != "eos":
        raise RuntimeError(
            f"base smoke row {line_number} did not terminate with EOS: "
            f"{row.get('generation_stop_reason')!r}"
        )
    if row.get("tokenizer_roundtrip_exact") is not True:
        raise RuntimeError(
            f"base smoke row {line_number} failed tokenizer roundtrip"
        )
    violations = output_only_violations(prediction)
    if violations:
        raise RuntimeError(
            f"base smoke row {line_number} is not answer-body-only: {violations}"
        )
print(json.dumps({
    "status": "PASS",
    "gate": "pretraining_base_inference_output_only",
    "rows": len(rows),
    "tasks": sorted(tasks),
}, sort_keys=True))
' "$BASE_PRED"

run_step "$PYTHON" -u scripts/train.py \
  --source-dir "$SOURCE_DIR" \
  --expert-nf4-dir "$TRAIN_EXPERT_DIR" \
  --data-dir "$DATA_DIR" \
  --output-dir "$SMOKE_OUT/train" \
  --preflight-receipt "$PREFLIGHT" \
  --smoke \
  --wandb-mode disabled

run_step "$PYTHON" -u scripts/infer.py \
  --source-dir "$SOURCE_DIR" \
  --expert-nf4-dir "$INFER_EXPERT_DIR" \
  --adapter-dir "$SMOKE_OUT/train/best_adapter" \
  --model-label solar_open2_lora_smoke_v7 \
  --eval-jsonl "$DATA_DIR/combined_test_and_gold_nodict.jsonl" \
  --output-jsonl "$TUNED_PRED" \
  --max-model-len 16384 \
  --translation-max-new-tokens 512 \
  --summarization-max-new-tokens 512 \
  --generation-batch-size 2 \
  --limit-per-task 1 \
  --resume

run_step "$PYTHON" -u scripts/certify_smoke.py \
  --source-dir "$SOURCE_DIR" \
  --training-expert-nf4-dir "$TRAIN_EXPERT_DIR" \
  --inference-expert-nf4-dir "$INFER_EXPERT_DIR" \
  --preflight-receipt "$PREFLIGHT" \
  --tokenizer-contract-receipt "$TOKENIZER_CONTRACT_RECEIPT" \
  --training-manifest "$SMOKE_OUT/train/run_manifest.json" \
  --base-predictions "$BASE_PRED" \
  --tuned-predictions "$TUNED_PRED" \
  --output "$SMOKE_RECEIPT"

echo "SOLAR_OPEN2_TRAIN_SMOKE_V7_9600_EARLYSTOP_PASS job=${SLURM_JOB_ID} receipt=${SMOKE_RECEIPT}"
