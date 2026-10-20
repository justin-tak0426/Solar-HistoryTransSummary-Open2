#!/usr/bin/env bash
#SBATCH --job-name=mk03-solar2-env
#SBATCH --partition=batch
#SBATCH --qos=high
#SBATCH --nodelist=gpu03
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/data/home/mk03/workspace/model_training/Solar-HistoryTransSummary-Open2/outputs/logs/environment_%j.out
#SBATCH --error=/data/home/mk03/workspace/model_training/Solar-HistoryTransSummary-Open2/outputs/logs/environment_%j.out

set -euo pipefail

PROJECT=/data/home/mk03/workspace/model_training/Solar-HistoryTransSummary-Open2
RUNTIME_ROOT=/data1/jhtak/Solar-HistoryTransSummary-Open2
REVISION=8d1dda5ad2af58207c65348d825772d2f4d72481
SOURCE_DIR=$RUNTIME_ROOT/hf_cache/models--upstage--Solar-Open2-250B/snapshots/$REVISION
DATA_DIR=/data/home/mk03/workspace/model_training/modern_korean_lm/artifacts/nodict_official_20260719/data
DOWNLOAD_RECEIPT=$RUNTIME_ROOT/receipts/bf16_download.json
NEXT=$PROJECT/.solar-open-2.next
FINAL=$PROJECT/solar-open-2
BACKUP=$PROJECT/.solar-open-2.bootstrap-rollback
ENV_RECEIPT=$RUNTIME_ROOT/receipts/environment_promoted.json
TOKENIZER_RECEIPT=$RUNTIME_ROOT/receipts/tokenizer_output_contract.json
PROMOTION_SUCCESS=$RUNTIME_ROOT/receipts/environment_promotion.SUCCESS

[[ "$(id -un)" == "mk03" ]] || exit 2
for path in \
  "$PROJECT" "$RUNTIME_ROOT" "$SOURCE_DIR" "$DATA_DIR" \
  "$DOWNLOAD_RECEIPT" "$NEXT" "$FINAL" "$BACKUP" \
  "$ENV_RECEIPT" "$TOKENIZER_RECEIPT" "$PROMOTION_SUCCESS"; do
  resolved=$(realpath -m "$path")
  case "$resolved" in
    "$PROJECT"|"$PROJECT"/*|"$RUNTIME_ROOT"|"$RUNTIME_ROOT"/*|"$DATA_DIR") ;;
    *) echo "Refusing non-project path: $resolved" >&2; exit 2 ;;
  esac
done
[[ -f "$DOWNLOAD_RECEIPT" && -d "$SOURCE_DIR" && -d "$DATA_DIR" ]] || exit 2
mkdir -p "$RUNTIME_ROOT/receipts"
rm -f "$PROMOTION_SUCCESS"

validate_staging() {
  [[ -d "$NEXT" && ! -L "$NEXT" && -x "$NEXT/bin/python" ]] || exit 2
  "$NEXT/bin/python" -c \
    'import pathlib,sys; p=pathlib.Path(sys.argv[1]).resolve(); assert pathlib.Path(sys.prefix).resolve()==p; assert pathlib.Path(sys.base_prefix).resolve()==p' \
    "$NEXT"
  "$NEXT/bin/python" -m pip check
  diff -u \
    <("$NEXT/bin/python" -m pip freeze | sort) \
    <(sort "$PROJECT/requirements/solar-open-2.lock.txt")
}

if [[ -d "$NEXT" && ! -L "$NEXT" ]]; then
  validate_staging
  if [[ -d "$FINAL" && ! -L "$FINAL" && ! -e "$BACKUP" ]]; then
    mv "$FINAL" "$BACKUP"
  elif [[ ! -e "$FINAL" && -d "$BACKUP" && ! -L "$BACKUP" ]]; then
    echo "[environment] resuming after bootstrap backup" >&2
  else
    echo "Ambiguous environment promotion state" >&2
    exit 2
  fi
  if ! mv "$NEXT" "$FINAL"; then
    if [[ ! -e "$FINAL" && -d "$BACKUP" ]]; then
      mv "$BACKUP" "$FINAL"
    fi
    exit 2
  fi
elif [[ -d "$FINAL" && ! -L "$FINAL" && -d "$BACKUP" && ! -L "$BACKUP" ]]; then
  echo "[environment] already promoted; revalidating" >&2
else
  echo "Environment promotion inputs are incomplete" >&2
  exit 2
fi

PYTHON=$FINAL/bin/python
[[ -x "$PYTHON" ]] || exit 2
cd "$PROJECT"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=$RUNTIME_ROOT/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

"$PYTHON" -u scripts/check_environment.py \
  --require-cuda-count 0 \
  --receipt "$ENV_RECEIPT"

"$PYTHON" -u scripts/check_tokenizer_output_contract.py \
  --source-dir "$SOURCE_DIR" \
  --data-dir "$DATA_DIR" \
  --receipt "$TOKENIZER_RECEIPT"

temporary_success=$PROMOTION_SUCCESS.${SLURM_JOB_ID}.tmp
printf 'PASS %s %s\n' \
  "$(sha256sum "$TOKENIZER_RECEIPT" | awk '{print $1}')" \
  "$(sha256sum "$ENV_RECEIPT" | awk '{print $1}')" \
  > "$temporary_success"
mv "$temporary_success" "$PROMOTION_SUCCESS"
echo "SOLAR_OPEN2_ENVIRONMENT_PASS job=${SLURM_JOB_ID} receipt=${TOKENIZER_RECEIPT}"
