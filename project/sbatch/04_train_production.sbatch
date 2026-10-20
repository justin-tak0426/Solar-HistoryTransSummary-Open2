#!/usr/bin/env bash
#SBATCH --job-name=mk03-solar2-trn-es-v2
#SBATCH --partition=batch
#SBATCH --qos=high
#SBATCH --gres=gpu:rtx6000:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@7200
#SBATCH --open-mode=append
#SBATCH --output=/data/home/mk03/workspace/model_training/Solar-HistoryTransSummary-Open2/outputs/logs/train_production_earlystop_v2_9600_%j.out
#SBATCH --error=/data/home/mk03/workspace/model_training/Solar-HistoryTransSummary-Open2/outputs/logs/train_production_earlystop_v2_9600_%j.out

set -euo pipefail

PROJECT=/data/home/mk03/workspace/model_training/Solar-HistoryTransSummary-Open2
RUNTIME_ROOT=/data1/jhtak/Solar-HistoryTransSummary-Open2
REVISION=8d1dda5ad2af58207c65348d825772d2f4d72481
SOURCE_DIR=$RUNTIME_ROOT/hf_cache/models--upstage--Solar-Open2-250B/snapshots/$REVISION
TRAIN_EXPERT_DIR=$RUNTIME_ROOT/models/Solar-Open2-250B-NF4-layer-flat-experts
DATA_DIR=/data/home/mk03/workspace/model_training/modern_korean_lm/artifacts/nodict_official_20260719/data
PYTHON=$PROJECT/solar-open-2/bin/python
PREFLIGHT=$RUNTIME_ROOT/receipts/model_preflight.json
SMOKE_RECEIPT=$RUNTIME_ROOT/receipts/gpu_smoke_v7_9600_earlystop.json
TRAIN_OUT=${TRAIN_OUT:-$RUNTIME_ROOT/runs/train_2epoch_earlystop_step18631_len9600_v2}

[[ "$(id -un)" == "mk03" ]] || exit 2
[[ -x "$PYTHON" && -d "$SOURCE_DIR" && -d "$TRAIN_EXPERT_DIR" ]] || exit 2
[[ -f "$PREFLIGHT" && -f "$SMOKE_RECEIPT" ]] || {
  echo "Production gate is incomplete" >&2
  exit 2
}
for path in "$SOURCE_DIR" "$TRAIN_EXPERT_DIR" "$PREFLIGHT" "$SMOKE_RECEIPT" "$TRAIN_OUT"; do
  resolved=$(realpath -m "$path")
  case "$resolved" in
    "$RUNTIME_ROOT"/*) ;;
    *) echo "Refusing path outside $RUNTIME_ROOT: $resolved" >&2; exit 2 ;;
  esac
done
mkdir -p "$TRAIN_OUT" "$RUNTIME_ROOT/cache/triton_train_earlystop_v2" "$RUNTIME_ROOT/wandb"
cd "$PROJECT"

export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=$RUNTIME_ROOT/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR=$RUNTIME_ROOT/cache/triton_train_earlystop_v2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=online
export WANDB_DIR=$RUNTIME_ROOT/wandb
export WANDB_ENTITY=${WANDB_ENTITY:-junhyuk-tak-markr}
export WANDB_PROJECT=${WANDB_PROJECT:-modern-korean-lm}
export WANDB_RUN_ID=${WANDB_RUN_ID:-solar_open2_hardterm_2epoch_earlystop_20260727_v2}
export WANDB_NAME=${WANDB_NAME:-Solar-Open2-250B hard-term translation-summary 2epoch earlystop step18631 maxlen9600 v2}

"$PYTHON" -u scripts/check_environment.py \
  --require-cuda-count 3 \
  --receipt "$RUNTIME_ROOT/receipts/environment_train.json"

"$PYTHON" -c '
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
    raise RuntimeError(f"production recipe mismatch: {observed!r} != {expected!r}")
deck = recipe.get("training_deck") or {}
if (
    deck.get("policy") != "deterministic_global_augmented_epoch_deck"
    or (deck.get("total_occurrences") or {}).get("total") != 298096
):
    raise RuntimeError("production deterministic two-epoch deck contract is missing")
validation = recipe.get("validation") or {}
if validation.get("full_validation_steps") != [9316, 13973, 18631]:
    raise RuntimeError("production validation candidate schedule changed")
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
    raise RuntimeError("production early-stopping policy changed")
' "$PROJECT/config/recipe.json"

child_pid=""
usr1_pending=0
forward_usr1() {
  usr1_pending=1
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill -USR1 "$child_pid" 2>/dev/null || true
  fi
}
trap forward_usr1 USR1

"$PYTHON" -u scripts/train.py \
  --source-dir "$SOURCE_DIR" \
  --expert-nf4-dir "$TRAIN_EXPERT_DIR" \
  --data-dir "$DATA_DIR" \
  --output-dir "$TRAIN_OUT" \
  --preflight-receipt "$PREFLIGHT" \
  --smoke-receipt "$SMOKE_RECEIPT" \
  --wandb-mode online &
child_pid=$!
if (( usr1_pending )); then
  kill -USR1 "$child_pid" 2>/dev/null || true
fi
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

if (( usr1_pending )) || [[ "$status" -eq 75 ]]; then
  [[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || exit 75
  record=$(scontrol --oneliner show job "$SLURM_JOB_ID")
  [[ "$record" == *"JobId=${SLURM_JOB_ID} "* && "$record" == *"UserId=mk03("* ]] || exit 75
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$status"
