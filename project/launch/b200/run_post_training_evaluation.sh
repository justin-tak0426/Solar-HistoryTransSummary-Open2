#!/usr/bin/env bash
set -euo pipefail
umask 022

source "/NHNHOME/WORKSPACE/26mfds002_B/jhtak/solar-open-2/project/launch/b200/common_env.sh"
source "$B200_LAUNCH_DIR/gpu_idle_gate.sh"

b200_prepare_mutable_dirs
b200_assert_common_contract
b200_acquire_stage_lock
b200_gate_receipts production

readonly training_manifest="$B200_PRODUCTION_RUN_DIR/run_manifest.json"
readonly selected_adapter="$B200_PRODUCTION_RUN_DIR/best_adapter"
readonly adapter_receipt="$B200_EVALUATION_DIR/receipts/best_adapter_verified.json"
readonly base_split_receipt="$B200_EVALUATION_DIR/receipts/base_split.json"
readonly tuned_split_receipt="$B200_EVALUATION_DIR/receipts/tuned_split.json"
readonly full_metrics_receipt="$B200_EVALUATION_DIR/receipts/full_test_metrics.json"
readonly gold_metrics_receipt="$B200_EVALUATION_DIR/receipts/human_gold_metrics.json"
readonly evaluation_wandb_receipt="$SOLAR_RUNTIME/receipts/b200_evaluation_wandb_v1.json"
readonly base_full_predictions="$B200_EVALUATION_DIR/splits/base.full_test.jsonl"
readonly base_gold_predictions="$B200_EVALUATION_DIR/splits/base.human_gold.jsonl"
readonly tuned_full_predictions="$B200_EVALUATION_DIR/splits/tuned.full_test.jsonl"
readonly tuned_gold_predictions="$B200_EVALUATION_DIR/splits/tuned.human_gold.jsonl"
readonly full_metrics_dir="$B200_EVALUATION_DIR/metrics/full_test"
readonly gold_metrics_dir="$B200_EVALUATION_DIR/metrics/human_gold"

b200_require_existing_file "$B200_PRODUCTION_RUN_DIR/_SUCCESS"
b200_require_existing_file "$training_manifest"
b200_require_existing_file "$selected_adapter/_SUCCESS"
b200_require_existing_file "$B200_COMBINED_EVAL"
b200_require_existing_file "$B200_FULL_EVAL"
b200_require_existing_file "$B200_GOLD_EVAL"
b200_prepare_resumable_dir "$B200_EVALUATION_DIR"
b200_prepare_resumable_dir "$B200_EVALUATION_WANDB_DIR"
mkdir -p \
  "$B200_EVALUATION_DIR/predictions" \
  "$B200_EVALUATION_DIR/audit" \
  "$B200_EVALUATION_DIR/receipts" \
  "$B200_EVALUATION_DIR/splits" \
  "$B200_EVALUATION_DIR/metrics" \
  "$B200_EVALUATION_DIR/report"

evaluation_log="$SOLAR_RUNTIME/logs/b200_evaluation_v1.log"
if [[ -e "$evaluation_log" || -L "$evaluation_log" ]]; then
  evaluation_log="$SOLAR_RUNTIME/logs/b200_evaluation_v1.resume-$(/usr/bin/date -u +%Y%m%dT%H%M%SZ)-$$.log"
fi
b200_begin_log "$evaluation_log"

b200_use_cpu_only
"$B200_PYTHON" "$SOLAR_PROJECT/scripts/verify_final_adapter.py" \
  --config-dir "$B200_CONFIG_DIR" \
  --training-manifest "$training_manifest" \
  --adapter-dir "$selected_adapter" \
  --receipt "$adapter_receipt"

b200_use_gpu_6_7
b200_gpu_idle_gate
"$B200_PYTHON" "$SOLAR_PROJECT/scripts/infer.py" \
  --config-dir "$B200_CONFIG_DIR" \
  --mode production-evaluation \
  --source-dir "$B200_SOURCE_DIR" \
  --expert-nf4-dir "$B200_INFER_EXPERT_DIR" \
  --asset-integrity-receipt "$B200_ASSET_RECEIPT" \
  --terminal-training-dir "$B200_PRODUCTION_RUN_DIR" \
  --model-label "$B200_EVALUATION_BASE_LABEL" \
  --eval-jsonl "$B200_COMBINED_EVAL" \
  --output-jsonl "$B200_EVALUATION_BASE_COMBINED" \
  --tensor-parallel-size 2 \
  --max-model-len 16384 \
  --translation-max-new-tokens 8192 \
  --summarization-max-new-tokens 6144 \
  --generation-batch-size 2 \
  --resume

b200_gpu_idle_gate
"$B200_PYTHON" "$SOLAR_PROJECT/scripts/infer.py" \
  --config-dir "$B200_CONFIG_DIR" \
  --mode production-evaluation \
  --source-dir "$B200_SOURCE_DIR" \
  --expert-nf4-dir "$B200_INFER_EXPERT_DIR" \
  --asset-integrity-receipt "$B200_ASSET_RECEIPT" \
  --terminal-training-dir "$B200_PRODUCTION_RUN_DIR" \
  --adapter-dir "$selected_adapter" \
  --model-label "$B200_EVALUATION_TUNED_LABEL" \
  --eval-jsonl "$B200_COMBINED_EVAL" \
  --output-jsonl "$B200_EVALUATION_TUNED_COMBINED" \
  --tensor-parallel-size 2 \
  --max-model-len 16384 \
  --translation-max-new-tokens 8192 \
  --summarization-max-new-tokens 6144 \
  --generation-batch-size 2 \
  --resume

b200_use_cpu_only
"$B200_PYTHON" "$SOLAR_PROJECT/scripts/audit_outputs.py" \
  --config-dir "$B200_CONFIG_DIR" \
  --model "$B200_EVALUATION_BASE_LABEL" "$B200_EVALUATION_BASE_COMBINED" \
  --model "$B200_EVALUATION_TUNED_LABEL" "$B200_EVALUATION_TUNED_COMBINED" \
  --eval-jsonl "$B200_COMBINED_EVAL" \
  --training-manifest "$training_manifest" \
  --tuned-label "$B200_EVALUATION_TUNED_LABEL" \
  --output-json "$B200_EVALUATION_AUDIT_JSON" \
  --output-md "$B200_EVALUATION_AUDIT_MD"

"$B200_PYTHON" "$SOLAR_PROJECT/scripts/finalize.py" split \
  --config-dir "$B200_CONFIG_DIR" \
  --combined-predictions "$B200_EVALUATION_BASE_COMBINED" \
  --model-label "$B200_EVALUATION_BASE_LABEL" \
  --audit-json "$B200_EVALUATION_AUDIT_JSON" \
  --full-eval "$B200_FULL_EVAL" \
  --human-gold-eval "$B200_GOLD_EVAL" \
  --full-predictions "$base_full_predictions" \
  --gold-predictions "$base_gold_predictions" \
  --receipt "$base_split_receipt"

"$B200_PYTHON" "$SOLAR_PROJECT/scripts/finalize.py" split \
  --config-dir "$B200_CONFIG_DIR" \
  --combined-predictions "$B200_EVALUATION_TUNED_COMBINED" \
  --model-label "$B200_EVALUATION_TUNED_LABEL" \
  --audit-json "$B200_EVALUATION_AUDIT_JSON" \
  --full-eval "$B200_FULL_EVAL" \
  --human-gold-eval "$B200_GOLD_EVAL" \
  --full-predictions "$tuned_full_predictions" \
  --gold-predictions "$tuned_gold_predictions" \
  --receipt "$tuned_split_receipt"

b200_use_gpu_6_7
b200_gate_receipts production
b200_gpu_idle_gate
b200_run_metric_scorer \
  "$B200_PYTHON" "$SOLAR_PROJECT/scripts/evaluation.py" score \
  --prediction_jsonl "$base_full_predictions" "$tuned_full_predictions" \
  --out_dir "$full_metrics_dir" \
  --bert_model xlm-roberta-large \
  --bert_batch_size 8 \
  --device cuda \
  --bleu_tokenizer ko-mecab

b200_gpu_idle_gate
b200_run_metric_scorer \
  "$B200_PYTHON" "$SOLAR_PROJECT/scripts/evaluation.py" score \
  --prediction_jsonl "$base_gold_predictions" "$tuned_gold_predictions" \
  --out_dir "$gold_metrics_dir" \
  --bert_model xlm-roberta-large \
  --bert_batch_size 8 \
  --device cuda \
  --bleu_tokenizer ko-mecab

b200_use_cpu_only
"$B200_PYTHON" "$SOLAR_PROJECT/scripts/finalize.py" seal-metrics \
  --config-dir "$B200_CONFIG_DIR" \
  --dataset full_test \
  --split-receipt "$base_split_receipt" \
  --split-receipt "$tuned_split_receipt" \
  --audit-json "$B200_EVALUATION_AUDIT_JSON" \
  --summary "$full_metrics_dir/metrics.summary.json" \
  --rows "$full_metrics_dir/metrics.rows.jsonl" \
  --receipt "$full_metrics_receipt"

"$B200_PYTHON" "$SOLAR_PROJECT/scripts/finalize.py" seal-metrics \
  --config-dir "$B200_CONFIG_DIR" \
  --dataset human_gold \
  --split-receipt "$base_split_receipt" \
  --split-receipt "$tuned_split_receipt" \
  --audit-json "$B200_EVALUATION_AUDIT_JSON" \
  --summary "$gold_metrics_dir/metrics.summary.json" \
  --rows "$gold_metrics_dir/metrics.rows.jsonl" \
  --receipt "$gold_metrics_receipt"

"$B200_PYTHON" "$SOLAR_PROJECT/scripts/finalize.py" report \
  --config-dir "$B200_CONFIG_DIR" \
  --full-summary "$full_metrics_dir/metrics.summary.json" \
  --gold-summary "$gold_metrics_dir/metrics.summary.json" \
  --full-metrics-receipt "$full_metrics_receipt" \
  --gold-metrics-receipt "$gold_metrics_receipt" \
  --training-manifest "$training_manifest" \
  --audit-json "$B200_EVALUATION_AUDIT_JSON" \
  --base-label "$B200_EVALUATION_BASE_LABEL" \
  --tuned-label "$B200_EVALUATION_TUNED_LABEL" \
  --output-json "$B200_EVALUATION_REPORT_JSON" \
  --output-md "$B200_EVALUATION_REPORT_MD"

"$B200_PYTHON" "$SOLAR_PROJECT/scripts/upload_wandb.py" \
  --config-dir "$B200_CONFIG_DIR" \
  --report-json "$B200_EVALUATION_REPORT_JSON" \
  --report-md "$B200_EVALUATION_REPORT_MD" \
  --training-manifest "$training_manifest" \
  --audit-json "$B200_EVALUATION_AUDIT_JSON" \
  --audit-md "$B200_EVALUATION_AUDIT_MD" \
  --full-summary "$full_metrics_dir/metrics.summary.json" \
  --gold-summary "$gold_metrics_dir/metrics.summary.json" \
  --full-metrics-receipt "$full_metrics_receipt" \
  --gold-metrics-receipt "$gold_metrics_receipt" \
  --receipt "$evaluation_wandb_receipt" \
  --success-marker "$B200_EVALUATION_DIR/_SUCCESS" \
  --wandb-dir "$B200_EVALUATION_WANDB_DIR" \
  --entity junhyuk-tak-markr \
  --project modern-korean-lm \
  --run-id "$B200_EVALUATION_WANDB_RUN_ID" \
  --run-name "$B200_EVALUATION_WANDB_NAME" \
  --training-run-id solar_open2_b200_1epoch_20260802_v2

printf '[b200-evaluation] PASS scorer=%s training_run=%s evaluation_run=%s\n' \
  "$SOLAR_PROJECT/scripts/evaluation.py" \
  "solar_open2_b200_1epoch_20260802_v2" \
  "$B200_EVALUATION_WANDB_RUN_ID"
