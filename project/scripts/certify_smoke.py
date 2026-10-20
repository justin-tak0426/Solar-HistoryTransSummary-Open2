#!/usr/bin/env python3
"""Fail-closed certification for the Solar training and inference smoke."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import b200_runtime
from check_tokenizer_output_contract import (
    B200_VALIDATION_UNICODE_EXPECTED,
    EXPECTED_SOURCE_DIR as EXPECTED_TOKENIZER_SOURCE_DIR,
    EXPECTED_TEMPLATE_KWARGS as EXPECTED_TOKENIZER_TEMPLATE_KWARGS,
    PROMPT_BUDGET_TOKENS,
    SOURCE_TOKENIZER_FILES,
    validate_b200_input_unicode_inventory,
)
from common import (
    CONFIG_DIR,
    PROJECT,
    REFERENCE_SHA256,
    TRAINING_RESULT_LABEL_IDS,
    atomic_json,
    canonical_sha256,
    output_only_violations,
    pipeline_code_identity,
    read_json,
    require_owned_path,
    sha256,
)
from infer import (
    B200_MAX_MODEL_LEN,
    B200_TASK_MAX_NEW_TOKENS,
    B200_VALIDATION_SMOKE_MODE,
    PINNED_COMBINED_EVAL_ROWS,
    PINNED_COMBINED_EVAL_SHA256,
    PINNED_COMBINED_EVAL_TASK_COUNTS,
    PROVENANCE_SCHEMA,
    PROVENANCE_SCHEMA_VERSION,
    TEMPLATE_KWARGS,
    expected_row_provenance,
    generation_integrity_violations,
    lightweight_expert_identity,
    lightweight_source_identity,
    read_prediction_rows,
    validate_adapter,
    verify_recorded_runtime_identity,
)
from quantization.contract import (
    BASE_MODEL_ID,
    BASE_REVISION,
    EXPECTED_SOURCE_INDEX_SHA256,
    EXPECTED_SOURCE_SHARD_COUNT,
    EXPECTED_SOURCE_TENSOR_COUNT,
    EXPERT_INDEX_FILENAME,
    LAYER_FLAT_SCHEMA_NAME,
    MANIFEST_FILENAME,
    SCHEMA_VERSION,
)
from quantization.output_checkpoint import read_json_no_duplicates
from training_plan import early_stopping_decision, positive_cosine_learning_rate


EXPECTED_TASKS = {"translation", "summarization"}
EXPECTED_LORA_MODULES = 528
EXPECTED_LORA_TENSORS = 1_056
EXPECTED_TRAINABLE_PARAMETERS = 133_177_344
SMOKE_MAX_MODEL_LEN = 16_384
SMOKE_MAX_NEW_TOKENS = 512
SMOKE_GENERATION_BATCH_SIZE = 2
PRODUCTION_MAX_LENGTH = 9_600
ABSOLUTE_TRAIN_MAX_TOKENS = 9_560
PRODUCTION_EPOCHS = 2
PRODUCTION_OPTIMIZER_STEPS = 18_631
PRODUCTION_OCCURRENCES = 298_096
PRODUCTION_ROWS_PER_UPDATE = 16
PRODUCTION_FULL_VALIDATION_STEPS = (9_316, 13_973, 18_631)
PRODUCTION_EARLIEST_STOP_STEP = 13_973
EXPECTED_EARLY_STOPPING_POLICY = {
    "enabled": True,
    "monitor": "selection_loss",
    "mode": "min",
    "minimum_epochs": 1.0,
    "minimum_step": 9_316,
    "subset_baseline_policy": "last_fixed_subset_before_minimum_step",
    "subset_patience_evaluations": 3,
    "subset_min_delta": {
        "absolute": 0.001,
        "relative": 0.0025,
        "combination": "max",
    },
    "full_validation_confirmation_required": True,
    "confirmation_step": 13_973,
    "full_min_delta": {
        "absolute": 0.0005,
        "relative": 0.001,
        "combination": "max",
    },
    "continue_if_either_task_materially_improves": True,
    "earliest_stop_step": 13_973,
    "latest_stop_step": 18_631,
    "restore_best_full_validation": True,
    "sealed_evaluation_not_monitor": True,
}
TIMING_UNTIMED_WARMUP_ITERATIONS = 1
TIMING_TIMED_REPEATS = 3
TIMING_ESTIMATOR = "median_p50"
SMOKE_UPDATE_WARMUP_REPETITIONS = 2
SMOKE_UPDATE_TIMED_REPETITIONS = 10
PRODUCTION_BUCKETS = (
    (1, 1_024, 8),
    (1_025, 2_048, 4),
    (2_049, 4_096, 2),
    (4_097, 9_600, 1),
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RUNTIME_SAFETY_FACTOR = 1.25
ALLOCATION_LIMIT_SECONDS = 7 * 24 * 60 * 60
ALLOCATION_REQUEUE_RESERVE_SECONDS = 2 * 60 * 60
USABLE_ALLOCATION_SECONDS = (
    ALLOCATION_LIMIT_SECONDS - ALLOCATION_REQUEUE_RESERVE_SECONDS
)
TOKENIZER_CONTRACT_CHECKS = (
    "owned_pinned_source_and_data",
    "pinned_data_hashes",
    "exact_reference_prompt",
    "reasoning_effort_none",
    "prompt_loss_masking",
    "body_prefix_and_termination",
    "termination_is_configured_generation_eos",
    "forbidden_framing_absent",
    "train_inference_prompt_identity",
    "inference_body_label_not_required",
    "combined_eval_prompt_budget_no_truncation",
)
TOKENIZER_OUTPUT_ROW_FIELDS = {
    "id",
    "task",
    "coverage_role",
    "split",
    "body_sha256",
    "prompt_ids_sha256",
    "prompt_token_count",
    "answer_token_count",
    "assistant_termination_token_count",
    "assistant_termination_token_id",
    "template_suffix_token_count",
    "supervised_labels_are_body_plus_single_eos",
    "answer_label_prefix_removed",
    "forbidden_body_framing",
    "train_inference_prompt_identical",
    "inference_requires_body_label",
}
TOKENIZER_ORDINARY_ROLES = {
    "train_translation_representative": ("train", "translation"),
    "train_summarization_representative": ("train", "summarization"),
    "validation_translation_representative": (
        "validation",
        "translation",
    ),
    "validation_summarization_representative": (
        "validation",
        "summarization",
    ),
}
TOKENIZER_CONTRACT_ROWS = len(TRAINING_RESULT_LABEL_IDS) + len(TOKENIZER_ORDINARY_ROLES)
TOKENIZER_CONTRACT_TASK_COUNTS = {
    "translation": len(TRAINING_RESULT_LABEL_IDS) + 2,
    "summarization": 2,
}
B200_RECEIPT_ROOT = b200_runtime.OWNED_ROOT / "runtime/receipts"
B200_ENVIRONMENT_RECEIPT = B200_RECEIPT_ROOT / "b200_environment_v1.json"
B200_ASSET_INTEGRITY_RECEIPT = B200_RECEIPT_ROOT / "b200_asset_integrity_large_v1.json"
B200_KERNEL_PROBE_RECEIPT = B200_RECEIPT_ROOT / "b200_kernel_probe_v1.json"
B200_TOKENIZER_CONTRACT_RECEIPT = B200_RECEIPT_ROOT / "b200_tokenizer_contract_v1.json"
B200_MODEL_PREFLIGHT_RECEIPT = B200_RECEIPT_ROOT / "b200_model_preflight_v1.json"
B200_SMOKE_RECEIPT = B200_RECEIPT_ROOT / "b200_smoke_2gpu_len9600_v1.json"
B200_TAIL_SHA256 = "2d9059e3f0130eb99223b4cab2a22da09090decfd4a4aa920a7528418e7eb14f"
B200_RECOMMENDED_HOURS = 144.0
B200_ABSOLUTE_LIMIT_HOURS = 168.0


def _fail(label: str, actual: Any, expected: Any) -> None:
    raise RuntimeError(f"{label} mismatch: actual={actual!r}, expected={expected!r}")


def _expect(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        _fail(label, actual, expected)


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _positive_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise RuntimeError(f"{label} must be finite and positive: {value!r}")
    return result


def _integer(
    value: Any,
    label: str,
    *,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be an integer: {value!r}")
    if minimum is not None and value < minimum:
        raise RuntimeError(f"{label} must be >= {minimum}: {value!r}")
    return value


def _runtime_budget_projection(
    training_bucket_benchmarks: dict[str, dict[str, Any]],
    validation_bucket_benchmarks: dict[str, dict[str, Any]],
    production_update_benchmark: dict[str, Any],
    validation_row_seconds: float,
    optimizer_boundary_seconds: float,
    one_time_gradient_audit_seconds: float,
    production_training_plan: dict[str, Any],
    subset_validation_plan: dict[str, Any],
    full_validation_plan: dict[str, Any],
) -> dict[str, Any]:
    validation_row_seconds = _positive_finite(
        validation_row_seconds, "smoke validation_row_seconds"
    )
    optimizer_boundary_seconds = _positive_finite(
        optimizer_boundary_seconds, "smoke optimizer_boundary_seconds"
    )
    one_time_gradient_audit_seconds = _positive_finite(
        one_time_gradient_audit_seconds,
        "smoke one_time_gradient_audit_seconds",
    )
    recipe = read_json(CONFIG_DIR / "recipe.json")
    data_contract = read_json(CONFIG_DIR / "data_contract.json")
    optimizer_steps = int(recipe["optimizer_steps"])
    rows_per_update = int(recipe["effective_batch_size"])
    eval_every = int(recipe["validation"]["eval_every_steps"])
    subset_rows = 2 * int(recipe["validation"]["subset_per_task"])
    full_steps = {int(step) for step in recipe["validation"]["full_validation_steps"]}
    if not full_steps or max(full_steps) != optimizer_steps:
        raise RuntimeError("final optimizer step must be a full validation candidate")
    subset_steps = [
        step
        for step in range(eval_every, optimizer_steps + 1, eval_every)
        if step not in full_steps
    ]
    subset_events = len(subset_steps)
    early_stopping = _object(
        recipe["validation"].get("early_stopping"),
        "recipe early-stopping policy",
    )
    if early_stopping != EXPECTED_EARLY_STOPPING_POLICY:
        raise RuntimeError("recipe early-stopping policy changed")
    earliest_stop_step = int(early_stopping["earliest_stop_step"])
    early_subset_steps = [step for step in subset_steps if step <= earliest_stop_step]
    early_full_steps = sorted(step for step in full_steps if step <= earliest_stop_step)
    if (
        earliest_stop_step != PRODUCTION_EARLIEST_STOP_STEP
        or early_subset_steps != list(range(1_000, 14_000, 1_000))
        or early_full_steps != [9_316, 13_973]
    ):
        raise RuntimeError("early-stop validation schedule changed")
    validation_row_count = int(data_contract["files"]["validation"]["rows"])
    total_validation_rows = (
        subset_events * subset_rows + len(full_steps) * validation_row_count
    )

    expected_occurrences = int(recipe["training_deck"]["total_occurrences"]["total"])
    if production_training_plan.get("optimizer_updates") != optimizer_steps:
        raise RuntimeError("training plan optimizer-step count differs from recipe")
    if production_training_plan.get("occurrences") != expected_occurrences:
        raise RuntimeError("training plan occurrence count differs from recipe")
    if expected_occurrences != optimizer_steps * rows_per_update:
        raise RuntimeError("recipe optimizer steps do not consume the exact deck")
    if subset_validation_plan.get("occurrences") != subset_rows:
        raise RuntimeError("subset validation plan row count differs from recipe")
    if full_validation_plan.get("occurrences") != validation_row_count:
        raise RuntimeError("full validation plan row count differs from data")
    subset_validation_microbatches = subset_events * int(
        subset_validation_plan["microbatches"]
    )
    full_validation_microbatches = len(full_steps) * int(
        full_validation_plan["microbatches"]
    )
    total_validation_microbatches = (
        subset_validation_microbatches + full_validation_microbatches
    )

    benchmark_seconds: dict[str, float] = {}
    validation_benchmark_seconds: dict[str, float] = {}
    expected_bucket_keys = {
        str(bucket["max_length_inclusive"])
        for bucket in recipe["microbatch_policy"]["buckets"]
    }
    if set(training_bucket_benchmarks) != expected_bucket_keys:
        raise RuntimeError(
            "training benchmark buckets differ from recipe: "
            f"{sorted(training_bucket_benchmarks)} != "
            f"{sorted(expected_bucket_keys)}"
        )
    if set(validation_bucket_benchmarks) != expected_bucket_keys:
        raise RuntimeError(
            "validation benchmark buckets differ from recipe: "
            f"{sorted(validation_bucket_benchmarks)} != "
            f"{sorted(expected_bucket_keys)}"
        )
    for bucket in recipe["microbatch_policy"]["buckets"]:
        key = str(bucket["max_length_inclusive"])
        benchmark = training_bucket_benchmarks[key]
        seconds = float(benchmark["seconds"])
        if not math.isfinite(seconds) or seconds <= 0:
            raise RuntimeError(
                f"training bucket {key} has invalid benchmark time: {seconds}"
            )
        if int(benchmark["batch_size"]) != int(bucket["target_micro_batch_size"]):
            raise RuntimeError(
                f"training bucket {key} benchmark batch size differs from recipe"
            )
        benchmark_seconds[key] = seconds
        validation_benchmark = validation_bucket_benchmarks[key]
        validation_seconds = float(validation_benchmark["seconds"])
        if not math.isfinite(validation_seconds) or validation_seconds <= 0:
            raise RuntimeError(
                f"validation bucket {key} has invalid benchmark time: "
                f"{validation_seconds}"
            )
        if int(validation_benchmark["batch_size"]) != int(
            bucket["target_micro_batch_size"]
        ):
            raise RuntimeError(
                f"validation bucket {key} benchmark batch size differs " "from recipe"
            )
        validation_benchmark_seconds[key] = validation_seconds

    update_p50_seconds = float(production_update_benchmark["seconds_p50"])
    update_p90_seconds = float(production_update_benchmark["seconds_p90"])
    for label, value in (
        ("production update p50", update_p50_seconds),
        ("production update p90", update_p90_seconds),
    ):
        if not math.isfinite(value) or value <= 0:
            raise RuntimeError(f"{label} must be finite and positive")
    if update_p90_seconds < update_p50_seconds:
        raise RuntimeError("production update p90 is below p50")
    if (
        int(production_update_benchmark["timed_repetitions"])
        != SMOKE_UPDATE_TIMED_REPETITIONS
        or production_update_benchmark.get("estimator") != TIMING_ESTIMATOR
        or production_update_benchmark.get("benchmark_state_unchanged") is not True
    ):
        raise RuntimeError("production update benchmark contract changed")

    bucket_microbatches = production_training_plan["bucket_microbatches"]
    raw_train_seconds = sum(
        int(bucket_microbatches[key]) * benchmark_seconds[key]
        for key in expected_bucket_keys
    )
    single_subset_validation_seconds = sum(
        int(subset_validation_plan["bucket_microbatches"][key])
        * validation_benchmark_seconds[key]
        for key in expected_bucket_keys
    )
    single_full_validation_seconds = sum(
        int(full_validation_plan["bucket_microbatches"][key])
        * validation_benchmark_seconds[key]
        for key in expected_bucket_keys
    )
    raw_subset_validation_seconds = subset_events * single_subset_validation_seconds
    raw_full_validation_seconds = len(full_steps) * single_full_validation_seconds
    raw_validation_seconds = raw_subset_validation_seconds + raw_full_validation_seconds
    raw_boundary_seconds = optimizer_boundary_seconds * optimizer_steps
    raw_one_time_audit_seconds = one_time_gradient_audit_seconds
    raw_total_seconds = (
        raw_train_seconds
        + raw_validation_seconds
        + raw_boundary_seconds
        + raw_one_time_audit_seconds
    )
    projected_total_seconds = raw_total_seconds * RUNTIME_SAFETY_FACTOR
    projected_full_validation_seconds = (
        sum(
            int(full_validation_plan["bucket_microbatches"][key])
            * validation_benchmark_seconds[key]
            for key in expected_bucket_keys
        )
        * RUNTIME_SAFETY_FACTOR
    )
    projected_single_microbatch_seconds = (
        max(benchmark_seconds.values()) * RUNTIME_SAFETY_FACTOR
    )
    projected_single_boundary_seconds = (
        optimizer_boundary_seconds * RUNTIME_SAFETY_FACTOR
    )
    projected_single_validation_microbatch_seconds = (
        max(validation_benchmark_seconds.values()) * RUNTIME_SAFETY_FACTOR
    )
    raw_empirical_p50_total_seconds = (
        update_p50_seconds * optimizer_steps
        + raw_validation_seconds
        + raw_boundary_seconds
        + raw_one_time_audit_seconds
    )
    raw_empirical_p90_total_seconds = (
        update_p90_seconds * optimizer_steps
        + raw_validation_seconds
        + raw_boundary_seconds
        + raw_one_time_audit_seconds
    )
    projected_empirical_p50_total_seconds = (
        raw_empirical_p50_total_seconds * RUNTIME_SAFETY_FACTOR
    )
    projected_empirical_p90_total_seconds = (
        raw_empirical_p90_total_seconds * RUNTIME_SAFETY_FACTOR
    )
    saved_optimizer_updates = optimizer_steps - earliest_stop_step
    early_validation_microbatches = len(early_subset_steps) * int(
        subset_validation_plan["microbatches"]
    ) + len(early_full_steps) * int(full_validation_plan["microbatches"])
    early_raw_validation_seconds = (
        len(early_subset_steps) * single_subset_validation_seconds
        + len(early_full_steps) * single_full_validation_seconds
    )
    early_raw_boundary_seconds = optimizer_boundary_seconds * earliest_stop_step
    early_raw_empirical_p50_total_seconds = (
        update_p50_seconds * earliest_stop_step
        + early_raw_validation_seconds
        + early_raw_boundary_seconds
        + raw_one_time_audit_seconds
    )
    early_raw_empirical_p90_total_seconds = (
        update_p90_seconds * earliest_stop_step
        + early_raw_validation_seconds
        + early_raw_boundary_seconds
        + raw_one_time_audit_seconds
    )
    early_projected_empirical_p50_total_seconds = (
        early_raw_empirical_p50_total_seconds * RUNTIME_SAFETY_FACTOR
    )
    early_projected_empirical_p90_total_seconds = (
        early_raw_empirical_p90_total_seconds * RUNTIME_SAFETY_FACTOR
    )
    early_allocation_count = max(
        1,
        math.ceil(
            early_projected_empirical_p50_total_seconds / USABLE_ALLOCATION_SECONDS
        ),
    )
    early_last_allocation_seconds = (
        early_projected_empirical_p50_total_seconds
        - (early_allocation_count - 1) * USABLE_ALLOCATION_SECONDS
    )
    feasibility_checks = {
        "single_longest_microbatch_fits_usable_allocation": (
            projected_single_microbatch_seconds <= USABLE_ALLOCATION_SECONDS
        ),
        "single_longest_validation_microbatch_fits_usable_allocation": (
            projected_single_validation_microbatch_seconds <= USABLE_ALLOCATION_SECONDS
        ),
        "single_optimizer_boundary_fits_usable_allocation": (
            projected_single_boundary_seconds <= USABLE_ALLOCATION_SECONDS
        ),
        "validation_progress_is_resumable": True,
    }
    failed = [name for name, passed in feasibility_checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "runtime budget is infeasible within one usable allocation: " f"{failed}"
        )
    allocation_count = max(
        1,
        math.ceil(projected_empirical_p50_total_seconds / USABLE_ALLOCATION_SECONDS),
    )
    last_allocation_seconds = projected_empirical_p50_total_seconds - (
        (allocation_count - 1) * USABLE_ALLOCATION_SECONDS
    )
    observed_workload = {
        "optimizer_steps": optimizer_steps,
        "rows_per_optimizer_step": rows_per_update,
        "training_occurrences": expected_occurrences,
        "train_microbatches": int(production_training_plan["microbatches"]),
        "training_plan": production_training_plan,
        "validation_subset_steps": subset_steps,
        "validation_subset_evaluations": subset_events,
        "validation_rows_per_subset": subset_rows,
        "validation_subset_rows": subset_events * subset_rows,
        "validation_subset_plan": subset_validation_plan,
        "full_validation_steps": sorted(full_steps),
        "full_validation_evaluations": len(full_steps),
        "full_validation_rows": validation_row_count,
        "full_validation_plan": full_validation_plan,
        "total_validation_rows": total_validation_rows,
        "total_validation_microbatches": total_validation_microbatches,
        "optimizer_boundaries": optimizer_steps,
        "full_gradient_audits": 1,
    }
    return {
        "schema_version": 5,
        "measurement": {
            "training_bucket_benchmarks": training_bucket_benchmarks,
            "validation_bucket_benchmarks": validation_bucket_benchmarks,
            "production_update_benchmark": production_update_benchmark,
            "validation_row_seconds": validation_row_seconds,
            "optimizer_boundary_seconds": optimizer_boundary_seconds,
            "one_time_gradient_audit_seconds": (one_time_gradient_audit_seconds),
            "clock": "time.perf_counter",
            "cuda_synchronized": True,
            "microbatch_scope": (
                "one excluded shape warm-up then median of three synchronized "
                "target-batch forward/backward repeats per length bucket; "
                "tokenization excluded"
            ),
            "validation_row_scope": (
                "longest eligible row per task; one excluded warm-up then "
                "median of three tokenization plus eval/no-grad loss repeats"
            ),
            "validation_microbatch_scope": (
                "one excluded shape warm-up then median of three synchronized "
                "target-batch eval/no-grad repeats per length bucket; "
                "tokenization excluded"
            ),
            "production_update_scope": (
                "ten deterministic deck updates sampled across the exact "
                "configured horizon "
                "after two excluded deck-update warm-ups; grouping, "
                "tokenization, forward, and backward included; optimizer "
                "boundary excluded"
            ),
            "optimizer_boundary_scope": (
                "first real gradient clipping and AdamW step; includes initial "
                "optimizer-state allocation"
            ),
            "excluded_from_optimizer_boundary": [
                "one-time full gradient audit",
                "first-step adapter digest",
                "first-step frozen parameter probe",
            ],
        },
        "production_workload": observed_workload,
        "projection": {
            "safety_factor": RUNTIME_SAFETY_FACTOR,
            "raw_train_microbatch_seconds": raw_train_seconds,
            "raw_validation_seconds": raw_validation_seconds,
            "raw_optimizer_boundary_seconds": raw_boundary_seconds,
            "raw_one_time_gradient_audit_seconds": (raw_one_time_audit_seconds),
            "raw_total_seconds": raw_total_seconds,
            "projected_train_microbatch_seconds": (
                raw_train_seconds * RUNTIME_SAFETY_FACTOR
            ),
            "projected_validation_seconds": (
                raw_validation_seconds * RUNTIME_SAFETY_FACTOR
            ),
            "projected_optimizer_boundary_seconds": (
                raw_boundary_seconds * RUNTIME_SAFETY_FACTOR
            ),
            "projected_one_time_gradient_audit_seconds": (
                raw_one_time_audit_seconds * RUNTIME_SAFETY_FACTOR
            ),
            "projected_full_validation_seconds": (projected_full_validation_seconds),
            "full_validation_requires_partial_resume": (
                projected_full_validation_seconds > USABLE_ALLOCATION_SECONDS
            ),
            "projected_total_seconds": projected_total_seconds,
            "projected_total_hours": projected_total_seconds / 3_600,
        },
        "empirical_update_projection": {
            "estimator": "ten_update_sample_p50_and_p90",
            "raw_p50_total_seconds": raw_empirical_p50_total_seconds,
            "raw_p90_total_seconds": raw_empirical_p90_total_seconds,
            "projected_p50_total_seconds": (projected_empirical_p50_total_seconds),
            "projected_p90_total_seconds": (projected_empirical_p90_total_seconds),
            "projected_p50_total_hours": (
                projected_empirical_p50_total_seconds / 3_600
            ),
            "projected_p90_total_hours": (
                projected_empirical_p90_total_seconds / 3_600
            ),
        },
        "early_stopping_projection": {
            "enabled": True,
            "earliest_stop_step": earliest_stop_step,
            "maximum_step": optimizer_steps,
            "saved_optimizer_updates": saved_optimizer_updates,
            "maximum_training_fraction_saved": (
                saved_optimizer_updates / optimizer_steps
            ),
            "subset_validation_steps": early_subset_steps,
            "subset_validation_evaluations": len(early_subset_steps),
            "full_validation_steps": early_full_steps,
            "full_validation_evaluations": len(early_full_steps),
            "projected_validation_microbatches": (early_validation_microbatches),
            "raw_validation_seconds": early_raw_validation_seconds,
            "raw_optimizer_boundary_seconds": early_raw_boundary_seconds,
            "raw_one_time_gradient_audit_seconds": (raw_one_time_audit_seconds),
            "raw_empirical_p50_total_seconds": (early_raw_empirical_p50_total_seconds),
            "raw_empirical_p90_total_seconds": (early_raw_empirical_p90_total_seconds),
            "projected_empirical_p50_total_seconds": (
                early_projected_empirical_p50_total_seconds
            ),
            "projected_empirical_p90_total_seconds": (
                early_projected_empirical_p90_total_seconds
            ),
            "projected_empirical_p50_total_hours": (
                early_projected_empirical_p50_total_seconds / 3_600
            ),
            "projected_empirical_p90_total_hours": (
                early_projected_empirical_p90_total_seconds / 3_600
            ),
            "allocation": {
                "projected_p50_allocation_count": early_allocation_count,
                "projected_p50_last_allocation_seconds": (
                    early_last_allocation_seconds
                ),
                "projected_p50_last_allocation_headroom_seconds": (
                    USABLE_ALLOCATION_SECONDS - early_last_allocation_seconds
                ),
            },
        },
        "allocation": {
            "allocation_limit_seconds": ALLOCATION_LIMIT_SECONDS,
            "requeue_reserve_seconds": ALLOCATION_REQUEUE_RESERVE_SECONDS,
            "usable_allocation_seconds": USABLE_ALLOCATION_SECONDS,
            "projection_basis": (
                "empirical_update_projection.projected_p50_total_seconds"
            ),
            "projected_allocation_count": allocation_count,
            "projected_allocation_headroom_seconds": (
                allocation_count * USABLE_ALLOCATION_SECONDS
                - projected_empirical_p50_total_seconds
            ),
            "projected_last_allocation_seconds": last_allocation_seconds,
            "projected_last_allocation_headroom_seconds": (
                USABLE_ALLOCATION_SECONDS - last_allocation_seconds
            ),
            "explicit_requeue": True,
            "max_batch_requeue_is_not_total_allocation_limit": True,
        },
        "feasibility": {
            "status": "PASS",
            **feasibility_checks,
            "projected_single_longest_microbatch_seconds": (
                projected_single_microbatch_seconds
            ),
            "projected_single_longest_validation_microbatch_seconds": (
                projected_single_validation_microbatch_seconds
            ),
            "projected_single_optimizer_boundary_seconds": (
                projected_single_boundary_seconds
            ),
        },
        "limitations": {
            "explicitly_unmeasured_overheads": [
                "token-length preflight on each allocation",
                "model load on each allocation",
                "recovery and candidate checkpoint I/O",
                "W&B network I/O",
                "final and best adapter publication",
                "cold shape compilation outside the explicitly recorded "
                "benchmark warm-ups",
            ],
            "interpretation": (
                "bucket-ceiling and ten-update empirical compute projections "
                "for the configured optimizer horizon, plus an empirical-only "
                "earliest-stop projection, with a 25% aggregate safety "
                "factor; none is a wall-clock completion guarantee"
            ),
        },
    }


def _validate_recipe_v3(
    recipe: dict[str, Any],
    data_contract: dict[str, Any],
) -> None:
    for key, expected in (
        ("recipe_name", "solar_open2_hardterm_fancy_v3_earlystop"),
        ("epochs", PRODUCTION_EPOCHS),
        ("optimizer_steps", PRODUCTION_OPTIMIZER_STEPS),
        ("effective_batch_size", PRODUCTION_ROWS_PER_UPDATE),
        ("max_length", PRODUCTION_MAX_LENGTH),
        ("minimum_learning_rate_ratio", 0.1),
        ("expert_oversampling_factor", 8),
        ("expert_quality_label", "gold_expert_corrected"),
    ):
        _expect(f"recipe v3 {key}", recipe.get(key), expected)

    policy = _object(
        recipe.get("microbatch_policy"),
        "recipe dynamic microbatch policy",
    )
    for key, expected in (
        ("mode", "dynamic_length_bucket_target_subject_to_gpu_smoke"),
        ("fallback_micro_batch_size", 1),
        ("fallback_gradient_accumulation_steps", 16),
        ("optimizer_step_occurrences", PRODUCTION_ROWS_PER_UPDATE),
        (
            "deck_order_contract",
            "consecutive_16_occurrence_optimizer_groups; bucket only "
            "within each group; no cross-step pull-ahead",
        ),
    ):
        _expect(f"recipe microbatch policy {key}", policy.get(key), expected)
    expected_buckets = [
        {
            "min_length_inclusive": lower,
            "max_length_inclusive": upper,
            "target_micro_batch_size": batch_size,
        }
        for lower, upper, batch_size in PRODUCTION_BUCKETS
    ]
    _expect(
        "recipe dynamic microbatch buckets",
        policy.get("buckets"),
        expected_buckets,
    )

    train_file = _object(
        _object(data_contract.get("files"), "data-contract files").get("train"),
        "training data contract",
    )
    for key, expected in (
        ("rows", 142_713),
        (
            "by_task",
            {"translation": 97_044, "summarization": 45_669},
        ),
        (
            "quality",
            {
                "gold_expert_corrected": 905,
                "silver_model_generated": 141_808,
            },
        ),
    ):
        _expect(f"training data {key}", train_file.get(key), expected)

    deck = _object(recipe.get("training_deck"), "recipe training deck")
    for key, expected in (
        ("policy", "deterministic_global_augmented_epoch_deck"),
        ("shuffle", "deterministic_per_epoch_seed"),
        ("base_unique_rows", 142_713),
        ("expert_unique_rows", 905),
        ("expert_occurrences_per_row_per_epoch", 8),
        (
            "occurrences_per_epoch",
            {
                "translation": 102_224,
                "summarization": 46_824,
                "total": 149_048,
            },
        ),
        (
            "total_occurrences",
            {
                "translation": 204_448,
                "summarization": 93_648,
                "total": PRODUCTION_OCCURRENCES,
            },
        ),
        ("derived_optimizer_steps", PRODUCTION_OPTIMIZER_STEPS),
    ):
        _expect(f"recipe training deck {key}", deck.get(key), expected)
    _expect(
        "recipe exact occurrence/update division",
        int(deck["total_occurrences"]["total"]),
        int(recipe["optimizer_steps"]) * int(recipe["effective_batch_size"]),
    )

    validation = _object(recipe.get("validation"), "recipe validation")
    _expect(
        "recipe full-validation candidate steps",
        validation.get("full_validation_steps"),
        list(PRODUCTION_FULL_VALIDATION_STEPS),
    )
    expected_candidates = [
        {
            "target_epoch": 1.0,
            "step": 9_316,
            "consumed_occurrences": 149_056,
            "target_occurrences": 149_048,
            "occurrence_offset": 8,
        },
        {
            "target_epoch": 1.5,
            "step": 13_973,
            "consumed_occurrences": 223_568,
            "target_occurrences": 223_572,
            "occurrence_offset": -4,
        },
        {
            "target_epoch": 2.0,
            "step": 18_631,
            "consumed_occurrences": 298_096,
            "target_occurrences": 298_096,
            "occurrence_offset": 0,
        },
    ]
    _expect(
        "recipe full-validation candidate contract",
        validation.get("full_validation_candidates"),
        expected_candidates,
    )
    _expect(
        "recipe early-stopping policy",
        validation.get("early_stopping"),
        EXPECTED_EARLY_STOPPING_POLICY,
    )


def _validate_microbatch_plan(
    value: Any,
    *,
    label: str,
    expected_updates: int,
    expected_occurrences: int,
) -> dict[str, Any]:
    plan = _object(value, label)
    _expect(
        f"{label} fields",
        set(plan),
        {
            "optimizer_updates",
            "occurrences",
            "microbatches",
            "bucket_microbatches",
            "bucket_rows",
        },
    )
    _expect(
        f"{label} optimizer updates",
        _integer(plan.get("optimizer_updates"), f"{label} optimizer updates"),
        expected_updates,
    )
    _expect(
        f"{label} occurrences",
        _integer(plan.get("occurrences"), f"{label} occurrences"),
        expected_occurrences,
    )
    microbatches = _integer(
        plan.get("microbatches"),
        f"{label} microbatches",
        minimum=1,
    )
    expected_keys = {str(upper) for _lower, upper, _size in PRODUCTION_BUCKETS}
    bucket_microbatches = _object(
        plan.get("bucket_microbatches"),
        f"{label} bucket microbatches",
    )
    bucket_rows = _object(plan.get("bucket_rows"), f"{label} bucket rows")
    _expect(
        f"{label} bucket-microbatch coverage",
        set(bucket_microbatches),
        expected_keys,
    )
    _expect(f"{label} bucket-row coverage", set(bucket_rows), expected_keys)
    counted_microbatches = 0
    counted_rows = 0
    for _lower, upper, capacity in PRODUCTION_BUCKETS:
        key = str(upper)
        groups = _integer(
            bucket_microbatches[key],
            f"{label} bucket {key} microbatches",
            minimum=0,
        )
        rows = _integer(
            bucket_rows[key],
            f"{label} bucket {key} rows",
            minimum=0,
        )
        if rows > groups * capacity:
            raise RuntimeError(f"{label} bucket {key} rows exceed microbatch capacity")
        if (rows == 0) != (groups == 0):
            raise RuntimeError(f"{label} bucket {key} rows/microbatches disagree")
        counted_microbatches += groups
        counted_rows += rows
    _expect(f"{label} microbatch total", counted_microbatches, microbatches)
    _expect(f"{label} row total", counted_rows, expected_occurrences)
    return plan


def _validate_token_length_contract(
    value: Any,
    *,
    label: str,
    expected_rows: int,
    expected_by_task: dict[str, int],
    expected_maximum: int | None,
) -> dict[str, Any]:
    contract = _object(value, label)
    for key, expected in (
        ("rows", expected_rows),
        ("by_task", expected_by_task),
        ("max_length", PRODUCTION_MAX_LENGTH),
        ("over_max_count", 0),
        ("over_max_rows", []),
    ):
        _expect(f"{label} {key}", contract.get(key), expected)
    ordered = [
        _integer(contract.get(key), f"{label} {key}", minimum=1)
        for key in ("minimum", "p50", "p90", "p95", "p99", "p99_9", "maximum")
    ]
    if ordered != sorted(ordered):
        raise RuntimeError(f"{label} token quantiles are not monotonic")
    if ordered[-1] > PRODUCTION_MAX_LENGTH:
        raise RuntimeError(f"{label} exceeds the 9,600-token contract")
    if expected_maximum is not None:
        _expect(f"{label} absolute maximum", ordered[-1], expected_maximum)
    return contract


def _validate_post_warmup_median_timing(
    value: dict[str, Any],
    *,
    label: str,
    require_warmup_seconds: bool,
) -> float:
    for field, expected in (
        ("warmup_repetitions", TIMING_UNTIMED_WARMUP_ITERATIONS),
        ("timed_repetitions", TIMING_TIMED_REPEATS),
        ("estimator", TIMING_ESTIMATOR),
    ):
        _expect(f"{label} {field}", value.get(field), expected)
    if require_warmup_seconds:
        _positive_finite(value.get("warmup_seconds"), f"{label} warmup seconds")
    _boolean(
        value.get("benchmark_state_unchanged"),
        f"{label} benchmark state",
        True,
    )
    samples = value.get("seconds_samples")
    if not isinstance(samples, list) or len(samples) != TIMING_TIMED_REPEATS:
        raise RuntimeError(
            f"{label} must contain exactly {TIMING_TIMED_REPEATS} " "timed samples"
        )
    measured = [
        _positive_finite(sample, f"{label} sample {index}")
        for index, sample in enumerate(samples)
    ]
    p50 = _positive_finite(value.get("seconds_p50"), f"{label} p50")
    _expect(f"{label} recomputed p50", p50, statistics.median(measured))
    _expect(
        f"{label} projection timing alias",
        _positive_finite(value.get("seconds"), f"{label} seconds"),
        p50,
    )
    return p50


def _validate_production_update_benchmark(
    value: Any,
) -> dict[str, Any]:
    benchmark = _object(value, "production update benchmark")
    _expect(
        "production update benchmark fields",
        set(benchmark),
        {
            "warmup_repetitions",
            "warmup_step_numbers",
            "timed_repetitions",
            "timed_step_numbers",
            "seconds_samples",
            "seconds_p50",
            "seconds_p90",
            "seconds_mean",
            "estimator",
            "occurrences_per_update",
            "microbatch_samples",
            "timed_plan",
            "row_order_sha256",
            "benchmark_state_unchanged",
        },
    )
    for field, expected in (
        ("warmup_repetitions", SMOKE_UPDATE_WARMUP_REPETITIONS),
        ("timed_repetitions", SMOKE_UPDATE_TIMED_REPETITIONS),
        ("estimator", TIMING_ESTIMATOR),
        ("occurrences_per_update", PRODUCTION_ROWS_PER_UPDATE),
    ):
        _expect(f"production update {field}", benchmark.get(field), expected)
    _boolean(
        benchmark.get("benchmark_state_unchanged"),
        "production update benchmark state",
        True,
    )
    _expect(
        "production update warmup steps",
        benchmark.get("warmup_step_numbers"),
        list(range(1, SMOKE_UPDATE_WARMUP_REPETITIONS + 1)),
    )
    expected_timed_steps = [
        min(
            PRODUCTION_OPTIMIZER_STEPS,
            max(
                SMOKE_UPDATE_WARMUP_REPETITIONS + 1,
                round(
                    (sample_index + 0.5)
                    * PRODUCTION_OPTIMIZER_STEPS
                    / SMOKE_UPDATE_TIMED_REPETITIONS
                ),
            ),
        )
        for sample_index in range(SMOKE_UPDATE_TIMED_REPETITIONS)
    ]
    _expect(
        "production update timed steps",
        benchmark.get("timed_step_numbers"),
        expected_timed_steps,
    )
    samples = benchmark.get("seconds_samples")
    if not isinstance(samples, list) or len(samples) != SMOKE_UPDATE_TIMED_REPETITIONS:
        raise RuntimeError(
            "production update benchmark must contain exactly "
            f"{SMOKE_UPDATE_TIMED_REPETITIONS} timed samples"
        )
    measured = [
        _positive_finite(sample, f"production update sample {index}")
        for index, sample in enumerate(samples)
    ]
    ordered = sorted(measured)
    p50 = _positive_finite(
        benchmark.get("seconds_p50"),
        "production update p50",
    )
    p90 = _positive_finite(
        benchmark.get("seconds_p90"),
        "production update p90",
    )
    _expect("production update recomputed p50", p50, statistics.median(measured))
    _expect(
        "production update recomputed p90",
        p90,
        ordered[math.ceil(0.9 * len(ordered)) - 1],
    )
    _expect(
        "production update recomputed mean",
        _positive_finite(
            benchmark.get("seconds_mean"),
            "production update mean",
        ),
        sum(measured) / len(measured),
    )
    microbatches = benchmark.get("microbatch_samples")
    if (
        not isinstance(microbatches, list)
        or len(microbatches) != SMOKE_UPDATE_TIMED_REPETITIONS
        or any(
            isinstance(count, bool) or not isinstance(count, int) or count <= 0
            for count in microbatches
        )
    ):
        raise RuntimeError("production update benchmark has invalid microbatch samples")
    timed_plan = _validate_microbatch_plan(
        benchmark.get("timed_plan"),
        label="production timed-update microbatch plan",
        expected_updates=SMOKE_UPDATE_TIMED_REPETITIONS,
        expected_occurrences=(
            SMOKE_UPDATE_TIMED_REPETITIONS * PRODUCTION_ROWS_PER_UPDATE
        ),
    )
    _expect(
        "production update sampled microbatch total",
        sum(microbatches),
        timed_plan["microbatches"],
    )
    _sha256_text(
        benchmark.get("row_order_sha256"),
        "production update row order",
    )
    return benchmark


def _validate_bucket_benchmarks(
    value: Any,
    *,
    absolute_maximum: int,
    phase: str,
) -> tuple[dict[str, dict[str, Any]], float]:
    benchmarks = _object(value, f"{phase} bucket benchmarks")
    expected_keys = {str(upper) for _lower, upper, _size in PRODUCTION_BUCKETS}
    _expect(f"{phase} benchmark bucket coverage", set(benchmarks), expected_keys)
    measured_seconds: list[float] = []
    for lower, upper, batch_size in PRODUCTION_BUCKETS:
        key = str(upper)
        benchmark = _object(
            benchmarks.get(key),
            f"{phase} bucket {lower}-{upper} benchmark",
        )
        for field, expected in (
            ("minimum_length", lower),
            ("maximum_length", upper),
            ("batch_size", batch_size),
        ):
            _expect(
                f"{phase} bucket {key} {field}",
                benchmark.get(field),
                expected,
            )
        row_ids = benchmark.get("row_ids")
        lengths = benchmark.get("observed_lengths")
        if not isinstance(row_ids, list) or len(row_ids) != batch_size:
            raise RuntimeError(
                f"{phase} bucket {key} did not benchmark its target batch"
            )
        if any(not isinstance(row_id, str) or not row_id for row_id in row_ids):
            raise RuntimeError(f"{phase} bucket {key} has an invalid row ID")
        if not isinstance(lengths, list) or len(lengths) != batch_size:
            raise RuntimeError(f"{phase} bucket {key} lacks target-batch token lengths")
        for position, length in enumerate(lengths):
            tokens = _integer(
                length,
                f"{phase} bucket {key} row {position} tokens",
                minimum=1,
            )
            if not lower <= tokens <= upper:
                raise RuntimeError(
                    f"{phase} bucket {key} row is outside its length range: "
                    f"{tokens}"
                )
        measured_seconds.append(
            _validate_post_warmup_median_timing(
                benchmark,
                label=f"{phase} bucket {key} timing",
                require_warmup_seconds=True,
            )
        )
    _expect(
        f"absolute longest {phase} row benchmark",
        benchmarks[str(PRODUCTION_MAX_LENGTH)]["observed_lengths"],
        [absolute_maximum],
    )
    return benchmarks, max(measured_seconds)


def _sha256_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise RuntimeError(f"{label} is not a SHA-256 digest: {value!r}")
    return value


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is missing or symlinked: {path}")
    return path


def _boolean(value: Any, label: str, expected: bool) -> None:
    if not isinstance(value, bool) or value is not expected:
        raise RuntimeError(f"{label} must be {expected!r}: {value!r}")


def _validate_gradient_checkpointing_contract(
    value: Any,
    label: str,
) -> dict[str, Any]:
    contract = _object(value, label)
    expected_fields = {
        "status",
        "implementation",
        "use_reentrant",
        "preserve_rng_state",
        "determinism_check",
        "early_stop",
        "decoder_layer_count",
        "boundary_layers",
        "layer_input_devices",
    }
    _expect(f"{label} fields", set(contract), expected_fields)
    _expect(f"{label} status", contract.get("status"), "PASS")
    _expect(
        f"{label} implementation",
        contract.get("implementation"),
        "device_aware_non_reentrant",
    )
    _boolean(contract.get("use_reentrant"), f"{label} use_reentrant", False)
    _boolean(
        contract.get("preserve_rng_state"),
        f"{label} preserve_rng_state",
        True,
    )
    _expect(
        f"{label} determinism_check",
        contract.get("determinism_check"),
        "default",
    )
    _boolean(contract.get("early_stop"), f"{label} early_stop", True)
    _expect(
        f"{label} decoder_layer_count",
        _integer(
            contract.get("decoder_layer_count"),
            f"{label} decoder_layer_count",
            minimum=1,
        ),
        48,
    )
    boundaries = contract.get("boundary_layers")
    if not isinstance(boundaries, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in boundaries
    ):
        raise RuntimeError(f"{label} boundary layers are invalid")
    _expect(f"{label} boundary_layers", boundaries, [16, 32])
    devices = _object(
        contract.get("layer_input_devices"),
        f"{label} layer input devices",
    )
    expected_devices = {str(layer): f"cuda:{layer // 16}" for layer in range(48)}
    _expect(f"{label} layer_input_devices", devices, expected_devices)
    return contract


def _current_file_map(directory: Path) -> dict[str, str]:
    if directory.is_symlink() or not directory.is_dir():
        raise RuntimeError(f"adapter directory is missing or symlinked: {directory}")
    result: dict[str, str] = {}
    for path in sorted(directory.iterdir()):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"adapter contains a non-regular entry: {path}")
        result[path.name] = sha256(path)
    return result


def _current_identities() -> dict[str, Any]:
    pipeline = pipeline_code_identity()
    return {
        "model_contract_sha256": sha256(CONFIG_DIR / "model.json"),
        "recipe_sha256": sha256(CONFIG_DIR / "recipe.json"),
        "data_contract_sha256": sha256(CONFIG_DIR / "data_contract.json"),
        "pipeline_code": pipeline,
        "environment_lock_sha256": sha256(
            PROJECT / "requirements/solar-open-2.lock.txt"
        ),
    }


def _current_tokenizer_source_identity(
    source_dir: Path,
    *,
    expected_source_dir: Path,
) -> dict[str, Any]:
    expected_source_dir = expected_source_dir.resolve()
    _expect("tokenizer contract pinned source path", source_dir, expected_source_dir)
    config_path = source_dir / "config.json"
    if not config_path.is_file():
        raise RuntimeError(
            f"tokenizer contract source config is missing: {config_path}"
        )
    config = read_json_no_duplicates(config_path)
    _expect(
        "tokenizer contract source model_type",
        config.get("model_type"),
        "solar_open2",
    )
    if "SolarOpen2ForCausalLM" not in (config.get("architectures") or []):
        raise RuntimeError("tokenizer contract source architecture changed")
    hashes = {
        name: sha256(source_dir / name)
        for name in SOURCE_TOKENIZER_FILES
        if (source_dir / name).is_file()
    }
    required = {
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
    }
    missing = sorted(required - set(hashes))
    if missing:
        raise RuntimeError(f"tokenizer contract source files are missing: {missing}")
    if not {"tokenizer.json", "tokenizer.model"} & set(hashes):
        raise RuntimeError("tokenizer contract source vocabulary is incomplete")
    generation = read_json_no_duplicates(source_dir / "generation_config.json")
    raw_eos = generation.get("eos_token_id")
    if isinstance(raw_eos, int) and not isinstance(raw_eos, bool):
        generation_eos_token_ids = [raw_eos]
    elif (
        isinstance(raw_eos, list)
        and raw_eos
        and all(
            isinstance(value, int) and not isinstance(value, bool) for value in raw_eos
        )
    ):
        generation_eos_token_ids = [int(value) for value in raw_eos]
    else:
        raise RuntimeError("tokenizer contract generation EOS IDs are invalid")
    return {
        "model_id": BASE_MODEL_ID,
        "revision": BASE_REVISION,
        "path": str(source_dir),
        "config_sha256": sha256(config_path),
        "generation_eos_token_ids": generation_eos_token_ids,
        "tokenizer_files_sha256": hashes,
    }


def _current_tokenizer_data_identity() -> dict[str, Any]:
    contract_path = _regular_file(
        CONFIG_DIR / "data_contract.json",
        "tokenizer contract data contract",
    )
    contract = read_json_no_duplicates(contract_path)
    _expect(
        "tokenizer contract reference evaluator SHA",
        contract.get("reference_evaluator_sha256"),
        REFERENCE_SHA256,
    )
    data_dir = Path(str(contract.get("prepared_data_dir") or "")).resolve()
    manifest_path = Path(str(contract.get("prepared_manifest") or "")).resolve()
    _expect(
        "tokenizer contract prepared manifest path",
        manifest_path,
        data_dir / "manifest.json",
    )
    _regular_file(manifest_path, "tokenizer contract prepared manifest")
    manifest_sha256 = sha256(manifest_path)
    _expect(
        "tokenizer contract prepared manifest SHA",
        manifest_sha256,
        contract.get("prepared_manifest_sha256"),
    )
    files: dict[str, Any] = {}
    configured_files = _object(
        contract.get("files"),
        "tokenizer contract configured files",
    )
    for role, entry_value in configured_files.items():
        entry = _object(entry_value, f"tokenizer contract configured {role}")
        path = _regular_file(
            data_dir / str(entry.get("name") or ""),
            f"tokenizer contract data {role}",
        )
        actual_sha256 = sha256(path)
        _expect(
            f"tokenizer contract data {role} SHA",
            actual_sha256,
            entry.get("sha256"),
        )
        files[role] = {
            "path": str(path),
            "sha256": actual_sha256,
            "rows": entry.get("rows"),
            "by_task": entry.get("by_task"),
        }
    return {
        "path": str(data_dir),
        "data_contract_sha256": sha256(contract_path),
        "prepared_manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha256,
        },
        "files": files,
    }


def _validate_tokenizer_output_rows(
    output_contract: dict[str, Any],
    *,
    source_generation_eos_ids: list[int],
) -> None:
    expected_fields = {
        "status",
        "template_kwargs",
        "training_add_generation_prompt",
        "training_full_sequence_add_generation_prompt",
        "inference_add_generation_prompt",
        "inference_requires_response_field",
        "inference_requires_body_label",
        "rows",
        "coverage",
        "supervised_termination_token_ids",
        "configured_generation_eos_token_ids",
    }
    _expect(
        "tokenizer output contract fields",
        set(output_contract),
        expected_fields,
    )
    _expect("tokenizer output contract status", output_contract.get("status"), "PASS")
    template_kwargs = _object(
        output_contract.get("template_kwargs"),
        "tokenizer output template kwargs",
    )
    _expect(
        "tokenizer output template kwargs",
        template_kwargs,
        EXPECTED_TOKENIZER_TEMPLATE_KWARGS,
    )
    _boolean(
        template_kwargs.get("provider_system_prompt"),
        "tokenizer output provider_system_prompt",
        False,
    )
    _expect(
        "inference/tokenizer template kwargs",
        TEMPLATE_KWARGS,
        EXPECTED_TOKENIZER_TEMPLATE_KWARGS,
    )
    for key, expected in (
        ("training_add_generation_prompt", True),
        ("training_full_sequence_add_generation_prompt", False),
        ("inference_add_generation_prompt", True),
        ("inference_requires_response_field", False),
        ("inference_requires_body_label", False),
    ):
        _boolean(
            output_contract.get(key),
            f"tokenizer output contract {key}",
            expected,
        )

    configured_value = output_contract.get("configured_generation_eos_token_ids")
    if not isinstance(configured_value, list) or not configured_value:
        raise RuntimeError("configured generation EOS IDs must be a nonempty list")
    configured_ids = [
        _integer(value, "configured generation EOS token ID", minimum=0)
        for value in configured_value
    ]
    _expect(
        "configured generation EOS token IDs",
        configured_ids,
        sorted(set(source_generation_eos_ids)),
    )
    supervised_value = output_contract.get("supervised_termination_token_ids")
    if not isinstance(supervised_value, list):
        raise RuntimeError("supervised termination token IDs must be a list")
    supervised_ids = [
        _integer(value, "supervised termination token ID", minimum=0)
        for value in supervised_value
    ]
    if len(supervised_ids) != 1 or not set(supervised_ids) <= set(configured_ids):
        raise RuntimeError(
            "supervised termination must be one configured generation EOS"
        )

    rows = output_contract.get("rows")
    if not isinstance(rows, list) or len(rows) != TOKENIZER_CONTRACT_ROWS:
        raise RuntimeError(
            "tokenizer output contract must contain exactly "
            f"{TOKENIZER_CONTRACT_ROWS} rows"
        )
    row_ids: set[str] = set()
    pinned_ids: set[str] = set()
    ordinary_roles: set[str] = set()
    suffix_token_counts: set[int] = set()
    counts: Counter[str] = Counter()
    for index, row_value in enumerate(rows):
        row = _object(row_value, f"tokenizer output row {index}")
        _expect(
            f"tokenizer output row {index} fields",
            set(row),
            TOKENIZER_OUTPUT_ROW_FIELDS,
        )
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id or row_id in row_ids:
            raise RuntimeError(
                f"tokenizer output row has an invalid/duplicate ID: {row_id!r}"
            )
        row_ids.add(row_id)
        task = row.get("task")
        if task not in EXPECTED_TASKS:
            raise RuntimeError(f"tokenizer output row task is invalid: {task!r}")
        split = row.get("split")
        role = row.get("coverage_role")
        is_pinned = row_id in TRAINING_RESULT_LABEL_IDS
        if is_pinned:
            _expect(
                f"tokenizer pinned row {row_id} role",
                role,
                "pinned_answer_prefix",
            )
            _expect(f"tokenizer pinned row {row_id} split", split, "train")
            _expect(f"tokenizer pinned row {row_id} task", task, "translation")
            pinned_ids.add(row_id)
        else:
            if role not in TOKENIZER_ORDINARY_ROLES:
                raise RuntimeError(f"tokenizer ordinary row role is invalid: {role!r}")
            expected_split, expected_task = TOKENIZER_ORDINARY_ROLES[role]
            _expect(f"tokenizer ordinary row {row_id} split", split, expected_split)
            _expect(f"tokenizer ordinary row {row_id} task", task, expected_task)
            if role in ordinary_roles:
                raise RuntimeError(f"duplicate tokenizer coverage role: {role}")
            ordinary_roles.add(role)
        _sha256_text(row.get("body_sha256"), f"tokenizer row {row_id} body SHA")
        _sha256_text(
            row.get("prompt_ids_sha256"),
            f"tokenizer row {row_id} prompt IDs SHA",
        )
        for key in ("prompt_token_count", "answer_token_count"):
            _integer(
                row.get(key),
                f"tokenizer row {row_id} {key}",
                minimum=1,
            )
        _expect(
            f"tokenizer row {row_id} termination count",
            _integer(
                row.get("assistant_termination_token_count"),
                f"tokenizer row {row_id} termination count",
                minimum=1,
            ),
            1,
        )
        _expect(
            f"tokenizer row {row_id} termination token ID",
            _integer(
                row.get("assistant_termination_token_id"),
                f"tokenizer row {row_id} termination token ID",
                minimum=0,
            ),
            supervised_ids[0],
        )
        suffix_token_counts.add(
            _integer(
                row.get("template_suffix_token_count"),
                f"tokenizer row {row_id} template suffix tokens",
                minimum=1,
            )
        )
        for key, expected in (
            ("supervised_labels_are_body_plus_single_eos", True),
            ("answer_label_prefix_removed", is_pinned),
            ("train_inference_prompt_identical", True),
            ("inference_requires_body_label", False),
        ):
            _boolean(
                row.get(key),
                f"tokenizer row {row_id} {key}",
                expected,
            )
        _expect(
            f"tokenizer row {row_id} forbidden framing",
            row.get("forbidden_body_framing"),
            [],
        )
        counts[str(task)] += 1

    _expect(
        "tokenizer pinned output rows",
        pinned_ids,
        set(TRAINING_RESULT_LABEL_IDS),
    )
    _expect(
        "tokenizer ordinary output roles",
        ordinary_roles,
        set(TOKENIZER_ORDINARY_ROLES),
    )
    _expect(
        "tokenizer output task counts",
        dict(counts),
        TOKENIZER_CONTRACT_TASK_COUNTS,
    )
    if len(suffix_token_counts) != 1:
        raise RuntimeError(
            "tokenizer output rows disagree on template suffix length: "
            f"{sorted(suffix_token_counts)}"
        )
    _expect(
        "tokenizer output coverage",
        output_contract.get("coverage"),
        {
            "rows": TOKENIZER_CONTRACT_ROWS,
            "by_task": TOKENIZER_CONTRACT_TASK_COUNTS,
            "pinned_answer_prefix_ids": sorted(TRAINING_RESULT_LABEL_IDS),
            "ordinary_task_representatives": {
                "train": ["summarization", "translation"],
                "validation": ["summarization", "translation"],
            },
        },
    )


def _validate_combined_prompt_budget(
    value: Any,
    *,
    combined_data: dict[str, Any],
) -> None:
    budget = _object(value, "tokenizer combined prompt budget")
    expected_fields = {
        "status",
        "path",
        "rows",
        "by_task",
        "prompt_budget_tokens",
        "truncation_count",
        "exact_output_only_preamble_count",
        "prompt_tokens",
    }
    _expect(
        "tokenizer combined prompt budget fields",
        set(budget),
        expected_fields,
    )
    for key, expected in (
        ("status", "PASS"),
        ("path", combined_data["path"]),
        ("by_task", PINNED_COMBINED_EVAL_TASK_COUNTS),
        ("prompt_budget_tokens", PROMPT_BUDGET_TOKENS),
    ):
        _expect(f"tokenizer combined prompt {key}", budget.get(key), expected)
    for key, expected in (
        ("rows", PINNED_COMBINED_EVAL_ROWS),
        ("truncation_count", 0),
        ("exact_output_only_preamble_count", PINNED_COMBINED_EVAL_ROWS),
    ):
        _expect(
            f"tokenizer combined prompt {key}",
            _integer(
                budget.get(key),
                f"tokenizer combined prompt {key}",
                minimum=0,
            ),
            expected,
        )
    prompt_tokens = _object(
        budget.get("prompt_tokens"),
        "tokenizer combined prompt token statistics",
    )
    _expect(
        "tokenizer combined prompt token tasks",
        set(prompt_tokens),
        EXPECTED_TASKS,
    )
    for task in sorted(EXPECTED_TASKS):
        stats = _object(
            prompt_tokens.get(task),
            f"tokenizer {task} prompt token statistics",
        )
        _expect(
            f"tokenizer {task} prompt token statistic fields",
            set(stats),
            {"minimum", "maximum", "maximum_row"},
        )
        minimum = _integer(
            stats.get("minimum"),
            f"tokenizer {task} minimum prompt tokens",
            minimum=1,
        )
        maximum = _integer(
            stats.get("maximum"),
            f"tokenizer {task} maximum prompt tokens",
            minimum=minimum,
        )
        if maximum > int(PROMPT_BUDGET_TOKENS[task]):
            raise RuntimeError(f"tokenizer {task} prompt maximum exceeds budget")
        maximum_row = _object(
            stats.get("maximum_row"),
            f"tokenizer {task} maximum prompt row",
        )
        _expect(
            f"tokenizer {task} maximum prompt row fields",
            set(maximum_row),
            {"id", "tokens"},
        )
        row_id = maximum_row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise RuntimeError(f"tokenizer {task} maximum prompt row ID is empty")
        _expect(
            f"tokenizer {task} maximum prompt row tokens",
            _integer(
                maximum_row.get("tokens"),
                f"tokenizer {task} maximum prompt row tokens",
                minimum=1,
            ),
            maximum,
        )


def _validate_tokenizer_contract_receipt(
    path: Path,
    *,
    source_dir: Path,
    identities: dict[str, Any],
    expected_source_dir: Path,
    expected_data: dict[str, Any] | None,
) -> dict[str, str]:
    expanded_path = path.expanduser()
    if not expanded_path.is_absolute():
        expanded_path = Path.cwd() / expanded_path
    _regular_file(expanded_path, "tokenizer contract receipt")
    path = expanded_path.resolve()
    receipt_sha256 = sha256(path)
    receipt = read_json_no_duplicates(path)
    expected_fields = {
        "status",
        "user",
        "project",
        "source",
        "data",
        "reference",
        "tokenizer",
        "output_only_contract",
        "combined_eval_prompt_budget",
        "pipeline_code",
        "checks",
    }
    _expect("tokenizer contract receipt fields", set(receipt), expected_fields)
    for key, expected in (
        ("status", "PASS"),
        ("user", "mk03"),
        ("project", str(PROJECT)),
        ("pipeline_code", identities["pipeline_code"]),
    ):
        _expect(f"tokenizer contract receipt {key}", receipt.get(key), expected)

    expected_source = _current_tokenizer_source_identity(
        source_dir,
        expected_source_dir=expected_source_dir,
    )
    _expect(
        "tokenizer contract source identity",
        receipt.get("source"),
        expected_source,
    )
    data = (
        _current_tokenizer_data_identity() if expected_data is None else expected_data
    )
    _expect("tokenizer contract data identity", receipt.get("data"), data)
    combined = _object(
        _object(data.get("files"), "tokenizer contract data files").get(
            "combined_eval"
        ),
        "tokenizer contract combined data",
    )
    for key, expected in (
        ("sha256", PINNED_COMBINED_EVAL_SHA256),
        ("rows", PINNED_COMBINED_EVAL_ROWS),
        ("by_task", PINNED_COMBINED_EVAL_TASK_COUNTS),
    ):
        _expect(f"tokenizer contract combined data {key}", combined.get(key), expected)

    reference_path = _regular_file(
        PROJECT / "scripts/reference/shared_nodict_evaluation.py",
        "tokenizer contract reference evaluator",
    )
    _expect(
        "tokenizer contract reference identity",
        receipt.get("reference"),
        {
            "path": str(reference_path),
            "sha256": REFERENCE_SHA256,
        },
    )
    _expect(
        "tokenizer contract current reference SHA",
        sha256(reference_path),
        REFERENCE_SHA256,
    )
    tokenizer = _object(receipt.get("tokenizer"), "tokenizer contract tokenizer")
    _expect(
        "tokenizer contract tokenizer fields",
        set(tokenizer),
        {"class", "name_or_path", "is_fast"},
    )
    tokenizer_class = tokenizer.get("class")
    if not isinstance(tokenizer_class, str) or not tokenizer_class:
        raise RuntimeError("tokenizer contract tokenizer class is empty")
    _expect(
        "tokenizer contract tokenizer name_or_path",
        tokenizer.get("name_or_path"),
        str(source_dir),
    )
    _boolean(
        tokenizer.get("is_fast"),
        "tokenizer contract fast tokenizer",
        True,
    )
    _validate_tokenizer_output_rows(
        _object(
            receipt.get("output_only_contract"),
            "tokenizer output-only contract",
        ),
        source_generation_eos_ids=expected_source["generation_eos_token_ids"],
    )
    _validate_combined_prompt_budget(
        receipt.get("combined_eval_prompt_budget"),
        combined_data=combined,
    )
    checks = _object(receipt.get("checks"), "tokenizer contract checks")
    _expect(
        "tokenizer contract check fields",
        set(checks),
        set(TOKENIZER_CONTRACT_CHECKS),
    )
    failed = [name for name in TOKENIZER_CONTRACT_CHECKS if checks.get(name) != "PASS"]
    if failed:
        raise RuntimeError(
            f"tokenizer contract receipt lacks required PASS checks: {failed}"
        )
    return {"path": str(path), "sha256": receipt_sha256}


def _expected_evaluation_identity() -> dict[str, Any]:
    data_contract = read_json(CONFIG_DIR / "data_contract.json")
    entry = data_contract["files"]["combined_eval"]
    path = (Path(data_contract["prepared_data_dir"]) / str(entry["name"])).resolve()
    _regular_file(path, "combined evaluation")
    actual_sha256 = sha256(path)
    _expect("combined evaluation configured SHA", actual_sha256, entry["sha256"])
    _expect(
        "combined evaluation pinned SHA",
        actual_sha256,
        PINNED_COMBINED_EVAL_SHA256,
    )
    _expect("combined evaluation rows", entry["rows"], PINNED_COMBINED_EVAL_ROWS)
    _expect(
        "combined evaluation task counts",
        entry["by_task"],
        PINNED_COMBINED_EVAL_TASK_COUNTS,
    )
    return {
        "path": str(path),
        "sha256": actual_sha256,
        "rows": PINNED_COMBINED_EVAL_ROWS,
        "by_task": PINNED_COMBINED_EVAL_TASK_COUNTS,
    }


def _validate_download_receipt(
    summary: dict[str, Any],
    source_dir: Path,
    source: dict[str, Any],
) -> dict[str, Any]:
    path = require_owned_path(Path(str(summary.get("path") or "")))
    _regular_file(path, "download receipt")
    _expect("preflight download receipt SHA", sha256(path), summary.get("sha256"))
    receipt = read_json_no_duplicates(path)
    wanted = {
        "status": "PASS",
        "repo_id": BASE_MODEL_ID,
        "requested_revision": BASE_REVISION,
        "repo_sha": BASE_REVISION,
        "snapshot_dir": str(source_dir),
        "indexed_shards": EXPECTED_SOURCE_SHARD_COUNT,
        "indexed_tensors": EXPECTED_SOURCE_TENSOR_COUNT,
        "local_dir_copy": False,
        "resumable_hf_cache": True,
    }
    for key, expected in wanted.items():
        _expect(f"download receipt {key}", receipt.get(key), expected)
    hashes = _object(receipt.get("contract_hashes"), "download contract_hashes")
    license_path = source_dir / "LICENSE"
    if not license_path.is_file():
        raise RuntimeError(f"source LICENSE is missing: {license_path}")
    for name, expected in (
        ("config.json", source["config_sha256"]),
        ("model.safetensors.index.json", source["index_sha256"]),
        # Hugging Face snapshots intentionally symlink files into the blob cache.
        ("LICENSE", sha256(license_path)),
    ):
        _expect(f"download source hash {name}", hashes.get(name), expected)
    _expect("preflight download repo SHA", summary.get("repo_sha"), BASE_REVISION)
    _expect(
        "preflight download weight bytes",
        summary.get("weight_bytes"),
        receipt.get("verified_tensor_bytes"),
    )
    _expect(
        "preflight download cache bytes",
        summary.get("cache_used_bytes"),
        receipt.get("cache_used_bytes"),
    )
    return {"path": str(path), "sha256": sha256(path)}


def _validate_training_expert(
    expert_dir: Path,
    source_dir: Path,
    source: dict[str, Any],
    preflight_artifact: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = _regular_file(
        expert_dir / MANIFEST_FILENAME, "training expert manifest"
    )
    index_path = _regular_file(
        expert_dir / EXPERT_INDEX_FILENAME, "training expert index"
    )
    manifest = read_json_no_duplicates(manifest_path)
    wanted = {
        "schema": LAYER_FLAT_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "SEALED",
        "complete_model_checkpoint": False,
        "base_model_id": BASE_MODEL_ID,
        "base_revision": BASE_REVISION,
    }
    for key, expected in wanted.items():
        _expect(f"training expert manifest {key}", manifest.get(key), expected)
    manifest_sha256 = sha256(manifest_path)
    index_sha256 = sha256(index_path)
    _expect(
        "training expert sealed index SHA",
        manifest.get("expert_index_sha256"),
        index_sha256,
    )
    sealed_source = _object(manifest.get("source"), "training expert source provenance")
    for key, expected in (
        ("source_dir", str(source_dir)),
        ("index_sha256", source["index_sha256"]),
        ("config_sha256", source["config_sha256"]),
    ):
        _expect(f"training expert source {key}", sealed_source.get(key), expected)
    for key, expected in (
        ("status", "PASS"),
        ("production_eligible", True),
        ("checkpoint_dir", str(expert_dir)),
        ("manifest_sha256", manifest_sha256),
        ("expert_index_sha256", index_sha256),
    ):
        _expect(
            f"preflight training artifact {key}",
            preflight_artifact.get(key),
            expected,
        )
    return {
        "path": str(expert_dir),
        "manifest_sha256": manifest_sha256,
        "expert_index_sha256": index_sha256,
        "source_index_sha256": sealed_source["index_sha256"],
        "source_config_sha256": sealed_source["config_sha256"],
    }


def _validate_preflight(
    path: Path,
    *,
    source_dir: Path,
    training_expert_dir: Path,
    source: dict[str, Any],
    identities: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    _regular_file(path, "preflight receipt")
    receipt = read_json_no_duplicates(path)
    wanted = {
        "status": "PASS",
        "user": "mk03",
        "project": str(PROJECT),
        "source_dir": str(source_dir),
        "expert_nf4_dir": str(training_expert_dir),
        **identities,
    }
    for key, expected in wanted.items():
        _expect(f"preflight {key}", receipt.get(key), expected)
    hardware = _object(receipt.get("hardware"), "preflight hardware")
    _expect("preflight GPU count", hardware.get("device_count"), 3)
    _validate_download_receipt(
        _object(receipt.get("download"), "preflight download"),
        source_dir,
        source,
    )
    training_expert = _validate_training_expert(
        training_expert_dir,
        source_dir,
        source,
        _object(receipt.get("expert_artifact"), "preflight expert artifact"),
    )
    return receipt, training_expert


def _validate_stats(value: Any, label: str) -> None:
    stats = _object(value, label)
    _boolean(stats.get("finite"), f"{label} finite", True)
    _expect(
        f"{label} element_count",
        stats.get("element_count"),
        EXPECTED_TRAINABLE_PARAMETERS,
    )
    _expect(
        f"{label} parameter_count",
        stats.get("parameter_count"),
        EXPECTED_LORA_TENSORS,
    )
    _expect(
        f"{label} dtype parameter counts",
        stats.get("dtype_parameter_counts"),
        {"torch.float32": EXPECTED_LORA_TENSORS},
    )
    _expect(
        f"{label} dtype element counts",
        stats.get("dtype_element_counts"),
        {"torch.float32": EXPECTED_TRAINABLE_PARAMETERS},
    )
    for name in ("l1", "l2"):
        number = stats.get(name)
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise RuntimeError(f"{label} {name} must be numeric")
        if not math.isfinite(float(number)) or float(number) < 0:
            raise RuntimeError(f"{label} {name} is invalid: {number!r}")


def _validate_training(
    path: Path,
    *,
    source_dir: Path,
    training_expert_dir: Path,
    source: dict[str, Any],
    preflight: dict[str, Any],
    identities: dict[str, Any],
) -> tuple[dict[str, Any], Path, dict[str, str]]:
    _regular_file(path, "training manifest")
    training = read_json_no_duplicates(path)
    for key, expected in (
        ("status", "SMOKE_COMPLETED"),
        ("source_dir", str(source_dir)),
        ("expert_nf4_dir", str(training_expert_dir)),
        ("completed_updates", 1),
        ("target_updates", 1),
    ):
        _expect(f"training {key}", training.get(key), expected)

    data_contract = read_json(CONFIG_DIR / "data_contract.json")
    recipe = read_json(CONFIG_DIR / "recipe.json")
    _validate_recipe_v3(recipe, data_contract)
    _expect("training embedded recipe", training.get("config"), recipe)
    _expect(
        "training output-only contract",
        training.get("output_only_contract"),
        recipe["output_only_contract"],
    )
    state = _object(training.get("state_contract"), "training state_contract")
    expected_state = {
        "format_version": 4,
        "source_dir": str(source_dir),
        "expert_nf4_dir": str(training_expert_dir),
        "download_receipt_sha256": preflight["download"]["sha256"],
        "expert_manifest_sha256": preflight["expert_artifact"]["manifest_sha256"],
        "model_contract_sha256": identities["model_contract_sha256"],
        "recipe_sha256": identities["recipe_sha256"],
        "data_contract_sha256": identities["data_contract_sha256"],
        "train_sha256": data_contract["files"]["train"]["sha256"],
        "validation_sha256": data_contract["files"]["validation"]["sha256"],
        "pipeline_code_sha256": identities["pipeline_code"]["sha256"],
        "environment_lock_sha256": identities["environment_lock_sha256"],
        "smoke": True,
    }
    _expect("training state contract", state, expected_state)

    token_lengths = _object(
        training.get("token_length_contract"),
        "training token-length contract",
    )
    train_length_contract = _validate_token_length_contract(
        token_lengths.get("train"),
        label="training-set token-length contract",
        expected_rows=142_713,
        expected_by_task={
            "translation": 97_044,
            "summarization": 45_669,
        },
        expected_maximum=ABSOLUTE_TRAIN_MAX_TOKENS,
    )
    validation_length_contract = _validate_token_length_contract(
        token_lengths.get("validation"),
        label="validation-set token-length contract",
        expected_rows=17_850,
        expected_by_task={
            "translation": 12_133,
            "summarization": 5_717,
        },
        expected_maximum=None,
    )

    completion = _object(
        training.get("training_completion"),
        "smoke training completion",
    )
    for key, expected in (
        ("status", "SMOKE_COMPLETED"),
        ("epochs", PRODUCTION_EPOCHS),
        ("optimizer_steps", 1),
        ("consumed_occurrences", PRODUCTION_ROWS_PER_UPDATE),
        ("expected_optimizer_steps", PRODUCTION_OPTIMIZER_STEPS),
        ("expected_occurrences", PRODUCTION_OCCURRENCES),
    ):
        _expect(f"smoke training completion {key}", completion.get(key), expected)
    _expect(
        "smoke consumed occurrences",
        training.get("consumed_occurrences"),
        PRODUCTION_ROWS_PER_UPDATE,
    )
    coverage = _object(
        training.get("training_coverage"),
        "smoke training coverage",
    )
    for key, expected in (
        ("total_occurrences", PRODUCTION_ROWS_PER_UPDATE),
        ("skipped_occurrences", 0),
        (
            "unconsumed_planned_occurrences",
            PRODUCTION_OCCURRENCES - PRODUCTION_ROWS_PER_UPDATE,
        ),
        ("unique_raw_rows", None),
    ):
        _expect(f"smoke training coverage {key}", coverage.get(key), expected)
    for key in ("task_occurrences", "quality_occurrences"):
        counts = _object(
            coverage.get(key),
            f"smoke training coverage {key}",
        )
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count <= 0
            for count in counts.values()
        ):
            raise RuntimeError(f"smoke training coverage {key} has invalid counts")
        _expect(
            f"smoke training coverage {key} total",
            sum(counts.values()),
            PRODUCTION_ROWS_PER_UPDATE,
        )
    if not set(coverage["task_occurrences"]).issubset(EXPECTED_TASKS):
        raise RuntimeError("smoke task coverage contains an unsupported task")
    if not set(coverage["quality_occurrences"]).issubset(
        {"gold_expert_corrected", "silver_model_generated"}
    ):
        raise RuntimeError("smoke quality coverage contains an unsupported label")

    expected_sampler = {
        "base_rows": 142_713,
        "expert_rows": 905,
        "expert_boost": 8,
        "epochs": PRODUCTION_EPOCHS,
        "examples_per_epoch": 149_048,
        "total_examples": PRODUCTION_OCCURRENCES,
        "base_task_counts": {
            "translation": 97_044,
            "summarization": 45_669,
        },
        "expert_task_counts": {
            "translation": 740,
            "summarization": 165,
        },
        "task_examples_per_epoch": {
            "translation": 102_224,
            "summarization": 46_824,
        },
        "task_total_examples": {
            "translation": 204_448,
            "summarization": 93_648,
        },
    }
    _expect("production sampler plan", training.get("sampler"), expected_sampler)

    validation_resume = _object(
        training.get("validation_resume_contract"),
        "validation resume contract",
    )
    _expect(
        "validation resume contract fields",
        set(validation_resume),
        {
            "schema_version",
            "progress_path",
            "commit_every_groups",
            "bound_to_trainable_parameter_sha256",
            "signal_safe",
            "zero_drop",
        },
    )
    _expect(
        "validation resume contract schema_version",
        _integer(
            validation_resume.get("schema_version"),
            "validation resume contract schema_version",
        ),
        2,
    )
    bound_to_trainable = validation_resume.get("bound_to_trainable_parameter_sha256")
    if bound_to_trainable is not True:
        _fail(
            "validation resume contract " "bound_to_trainable_parameter_sha256",
            bound_to_trainable,
            True,
        )
    for key, expected in (
        ("commit_every_groups", 25),
        ("signal_safe", True),
        ("zero_drop", True),
    ):
        _expect(
            f"validation resume contract {key}",
            validation_resume.get(key),
            expected,
        )
    output_dir = Path(str(training.get("output_dir") or ""))
    if not output_dir.is_absolute():
        raise RuntimeError("training output_dir must be absolute")
    storage = _object(training.get("storage_audit"), "training storage audit")
    _expect(
        "training storage audit fields",
        set(storage),
        {"path", "free_bytes", "minimum_required_bytes", "status"},
    )
    _expect("training storage audit path", storage.get("path"), str(output_dir))
    _expect("training storage audit status", storage.get("status"), "PASS")
    free_bytes = _integer(
        storage.get("free_bytes"),
        "training storage free bytes",
        minimum=1,
    )
    minimum_required_bytes = _integer(
        storage.get("minimum_required_bytes"),
        "training minimum required storage bytes",
        minimum=1,
    )
    if minimum_required_bytes < 16 * 1024**3:
        raise RuntimeError("training storage minimum is below the 16-GiB contract")
    if free_bytes < minimum_required_bytes:
        raise RuntimeError("training storage audit has insufficient free bytes")
    _expect(
        "validation progress path",
        validation_resume.get("progress_path"),
        str(output_dir / "validation_progress.json"),
    )
    progress_path = Path(str(validation_resume["progress_path"]))
    if progress_path.exists() or progress_path.is_symlink():
        raise RuntimeError("completed smoke retained partial validation progress")
    history = training.get("history")
    if not isinstance(history, list) or not history:
        raise RuntimeError("completed smoke lacks validation history")
    for record in history:
        record = _object(record, "smoke validation history record")
        _expect(
            "smoke validation zero-drop evidence",
            record.get("dropped_over_max"),
            {},
        )
    expected_early_stopping_decision = early_stopping_decision(
        history,
        completed_step=1,
        policy=recipe["validation"]["early_stopping"],
        optimizer_steps=PRODUCTION_OPTIMIZER_STEPS,
        full_validation_steps=PRODUCTION_FULL_VALIDATION_STEPS,
        occurrences_per_step=PRODUCTION_ROWS_PER_UPDATE,
        occurrences_per_epoch=int(
            recipe["training_deck"]["occurrences_per_epoch"]["total"]
        ),
    )
    _expect(
        "training early-stopping decision",
        training.get("early_stopping_decision"),
        expected_early_stopping_decision,
    )
    for key, expected in (
        ("completed_epochs", expected_early_stopping_decision["completed_epochs"]),
        ("early_stopping_triggered", False),
    ):
        _expect(
            f"smoke training completion {key}",
            completion.get(key),
            expected,
        )

    loader = _object(training.get("loader_audit"), "training loader audit")
    _expect("training loader status", loader.get("status"), "PASS")
    _expect("training loader base revision", loader.get("base_revision"), BASE_REVISION)
    loader_hardware = _object(loader.get("hardware"), "training loader hardware")
    _expect("training loader GPU count", loader_hardware.get("device_count"), 3)
    source_inventory = _object(
        loader.get("source_inventory"), "training loader source inventory"
    )
    for key, expected in (
        ("source_dir", str(source_dir)),
        ("index_sha256", source["index_sha256"]),
        ("config_sha256", source["config_sha256"]),
        ("tensor_count", EXPECTED_SOURCE_TENSOR_COUNT),
        ("shard_count", EXPECTED_SOURCE_SHARD_COUNT),
    ):
        _expect(f"training loader source {key}", source_inventory.get(key), expected)
    expert_preflight = _object(
        loader.get("expert_preflight"), "training loader expert preflight"
    )
    for key, expected in (
        ("status", "PASS"),
        ("production_eligible", True),
        ("checkpoint_dir", str(training_expert_dir)),
        (
            "manifest_sha256",
            preflight["expert_artifact"]["manifest_sha256"],
        ),
        (
            "expert_index_sha256",
            preflight["expert_artifact"]["expert_index_sha256"],
        ),
    ):
        _expect(f"training loader expert {key}", expert_preflight.get(key), expected)
    loaded_model = _object(loader.get("model_audit"), "training loaded-model audit")
    for key, expected in (
        ("status", "PASS"),
        ("lora_target_bf16_linear_count", EXPECTED_LORA_MODULES),
        ("meta_parameter_count", 0),
        ("meta_buffer_count", 0),
        ("base_trainable_parameter_count", 0),
        ("model_parallel_hook_count", 51),
        ("gradient_checkpointing", True),
        ("device_aware_checkpoint_layer_count", 48),
    ):
        _expect(f"training loaded-model {key}", loaded_model.get(key), expected)
    _expect(
        "training loader gradient checkpointing",
        loader.get("gradient_checkpointing"),
        True,
    )
    loader_checkpointing = _validate_gradient_checkpointing_contract(
        loader.get("gradient_checkpointing_contract"),
        "training loader gradient-checkpointing contract",
    )
    loaded_device_map = _object(
        loaded_model.get("device_map"),
        "training loaded-model device map",
    )
    for layer in range(48):
        _expect(
            f"training layer {layer} checkpoint/device-map binding",
            loader_checkpointing["layer_input_devices"][str(layer)],
            loaded_device_map.get(f"model.layers.{layer}"),
        )
    training_checkpointing = _validate_gradient_checkpointing_contract(
        training.get("gradient_checkpointing_contract"),
        "training final gradient-checkpointing contract",
    )
    _expect(
        "loader/final gradient-checkpointing contract",
        training_checkpointing,
        loader_checkpointing,
    )

    smoke = _object(training.get("smoke"), "training smoke")
    _expect(
        "smoke early-stopping decision mirror",
        smoke.get("early_stopping_decision"),
        expected_early_stopping_decision,
    )
    _expect(
        "smoke maximum-length contract",
        smoke.get("max_length_contract"),
        PRODUCTION_MAX_LENGTH,
    )
    _expect(
        "smoke absolute longest train tokens",
        smoke.get("observed_tokens"),
        ABSOLUTE_TRAIN_MAX_TOKENS,
    )
    _expect(
        "smoke/token-contract absolute longest train tokens",
        smoke.get("observed_tokens"),
        train_length_contract["maximum"],
    )
    _expect(
        "smoke required absolute-longest tokens",
        smoke.get("absolute_longest_required_tokens"),
        ABSOLUTE_TRAIN_MAX_TOKENS,
    )
    _expect(
        "smoke occurrence count",
        smoke.get("consumed_occurrences"),
        PRODUCTION_ROWS_PER_UPDATE,
    )
    if int(smoke.get("hard_term_weighted_tokens", 0)) <= 0:
        raise RuntimeError("smoke did not exercise hard-term weighted loss")
    _expect(
        "smoke occurrence backward audit",
        smoke.get("backward_passes"),
        PRODUCTION_ROWS_PER_UPDATE,
    )
    _expect("smoke GPU count", smoke.get("gpu_count"), 3)
    _expect(
        "training hard-term token total",
        training.get("total_hard_term_weighted_tokens"),
        smoke["hard_term_weighted_tokens"],
    )
    _expect(
        "training backward total",
        training.get("backward_passes"),
        smoke["backward_passes"],
    )
    _expect(
        "training observed-token total",
        training.get("max_observed_tokens"),
        smoke["observed_tokens"],
    )
    smoke_update_plan = _validate_microbatch_plan(
        smoke.get("update_microbatch_plan"),
        label="smoke update microbatch plan",
        expected_updates=1,
        expected_occurrences=PRODUCTION_ROWS_PER_UPDATE,
    )
    if any(int(count) <= 0 for count in smoke_update_plan["bucket_rows"].values()):
        raise RuntimeError("smoke update did not exercise every length bucket")
    production_training_plan = _validate_microbatch_plan(
        smoke.get("production_training_plan"),
        label="production training microbatch plan",
        expected_updates=PRODUCTION_OPTIMIZER_STEPS,
        expected_occurrences=PRODUCTION_OCCURRENCES,
    )
    if any(
        int(count) <= 0 for count in production_training_plan["bucket_rows"].values()
    ):
        raise RuntimeError(
            "production training plan does not exercise every length bucket"
        )
    _expect(
        "top-level production training plan",
        training.get("production_training_plan"),
        production_training_plan,
    )
    subset_validation_plan = _validate_microbatch_plan(
        smoke.get("subset_validation_plan"),
        label="production subset-validation microbatch plan",
        expected_updates=1,
        expected_occurrences=1_024,
    )
    _expect(
        "top-level subset validation plan",
        training.get("subset_validation_plan"),
        subset_validation_plan,
    )
    full_validation_plan = _validate_microbatch_plan(
        smoke.get("full_validation_plan"),
        label="production full-validation microbatch plan",
        expected_updates=1,
        expected_occurrences=17_850,
    )
    _expect(
        "top-level full validation plan",
        training.get("full_validation_plan"),
        full_validation_plan,
    )

    microbatch_phase_seconds = _positive_finite(
        smoke.get("microbatch_phase_seconds"),
        "smoke microbatch_phase_seconds",
    )
    validation_row_seconds = _positive_finite(
        smoke.get("validation_row_seconds"),
        "smoke validation_row_seconds",
    )
    optimizer_boundary_seconds = _positive_finite(
        smoke.get("optimizer_boundary_seconds"),
        "smoke optimizer_boundary_seconds",
    )
    one_time_gradient_audit_seconds = _positive_finite(
        smoke.get("one_time_gradient_audit_seconds"),
        "smoke one_time_gradient_audit_seconds",
    )
    _expect(
        "training microbatch timing",
        training.get("microbatch_phase_seconds"),
        microbatch_phase_seconds,
    )
    _expect(
        "training validation-row timing",
        training.get("validation_row_seconds"),
        validation_row_seconds,
    )
    _expect(
        "training optimizer boundary timing",
        training.get("optimizer_boundary_seconds"),
        optimizer_boundary_seconds,
    )
    _expect(
        "training one-time gradient-audit timing",
        training.get("one_time_gradient_audit_seconds"),
        one_time_gradient_audit_seconds,
    )
    validation_benchmark = _object(
        smoke.get("validation_benchmark"),
        "smoke validation benchmark",
    )
    _expect(
        "training validation benchmark",
        training.get("validation_benchmark"),
        validation_benchmark,
    )
    training_bucket_benchmarks, maximum_bucket_seconds = _validate_bucket_benchmarks(
        validation_benchmark.get("training_buckets"),
        absolute_maximum=train_length_contract["maximum"],
        phase="training",
    )
    validation_bucket_benchmarks, _maximum_validation_bucket_seconds = (
        _validate_bucket_benchmarks(
            validation_benchmark.get("validation_buckets"),
            absolute_maximum=validation_length_contract["maximum"],
            phase="validation",
        )
    )
    _expect(
        "smoke maximum bucket timing",
        microbatch_phase_seconds,
        maximum_bucket_seconds,
    )
    _expect(
        "validation benchmark warm-up",
        validation_benchmark.get("warmup"),
        "one excluded shape-matched run before three timed repeats",
    )
    _boolean(
        validation_benchmark.get("benchmark_state_unchanged"),
        "runtime benchmark aggregate state",
        True,
    )
    _expect(
        "validation benchmark selection",
        validation_benchmark.get("selection"),
        "absolute longest validation row per task",
    )
    by_task = _object(
        validation_benchmark.get("by_task"),
        "validation benchmark tasks",
    )
    _expect("validation benchmark task coverage", set(by_task), EXPECTED_TASKS)
    measured_seconds: list[float] = []
    for task in sorted(EXPECTED_TASKS):
        task_benchmark = _object(
            by_task.get(task),
            f"validation benchmark {task}",
        )
        row_id = task_benchmark.get("row_id")
        if not isinstance(row_id, str) or not row_id:
            raise RuntimeError(f"validation benchmark {task} row ID is invalid")
        tokens = _integer(
            task_benchmark.get("tokens"),
            f"validation benchmark {task} tokens",
            minimum=1,
        )
        if tokens > PRODUCTION_MAX_LENGTH:
            raise RuntimeError(
                f"validation benchmark {task} exceeds max length: {tokens}"
            )
        measured_seconds.append(
            _validate_post_warmup_median_timing(
                task_benchmark,
                label=f"validation max-context {task} timing",
                require_warmup_seconds=False,
            )
        )
    _expect(
        "validation benchmark maximum",
        validation_benchmark.get("maximum_seconds"),
        max(measured_seconds),
    )
    _expect(
        "validation benchmark/runtime timing",
        validation_row_seconds,
        max(measured_seconds),
    )
    production_update_benchmark = _validate_production_update_benchmark(
        validation_benchmark.get("production_update_benchmark")
    )
    _expect(
        "smoke production update benchmark",
        smoke.get("production_update_benchmark"),
        production_update_benchmark,
    )
    expected_runtime_budget = _runtime_budget_projection(
        training_bucket_benchmarks,
        validation_bucket_benchmarks,
        production_update_benchmark,
        validation_row_seconds,
        optimizer_boundary_seconds,
        one_time_gradient_audit_seconds,
        production_training_plan,
        subset_validation_plan,
        full_validation_plan,
    )
    _expect(
        "smoke runtime budget projection",
        smoke.get("runtime_budget_projection"),
        expected_runtime_budget,
    )
    _expect(
        "training runtime budget projection",
        training.get("runtime_budget_projection"),
        expected_runtime_budget,
    )
    _expect(
        "runtime budget schema version",
        expected_runtime_budget.get("schema_version"),
        5,
    )
    runtime_feasibility = _object(
        expected_runtime_budget.get("feasibility"),
        "runtime feasibility",
    )
    _boolean(
        runtime_feasibility.get("validation_progress_is_resumable"),
        "runtime validation-progress resumability",
        True,
    )
    workload = _object(
        expected_runtime_budget.get("production_workload"),
        "runtime production workload",
    )
    for key, expected in (
        ("optimizer_steps", PRODUCTION_OPTIMIZER_STEPS),
        ("rows_per_optimizer_step", PRODUCTION_ROWS_PER_UPDATE),
        ("training_occurrences", PRODUCTION_OCCURRENCES),
        ("validation_subset_evaluations", 18),
        ("validation_subset_rows", 18 * 1_024),
        (
            "full_validation_steps",
            list(PRODUCTION_FULL_VALIDATION_STEPS),
        ),
        ("full_validation_evaluations", 3),
        ("full_validation_rows", 17_850),
        ("total_validation_rows", 18 * 1_024 + 3 * 17_850),
        (
            "total_validation_microbatches",
            18 * int(subset_validation_plan["microbatches"])
            + 3 * int(full_validation_plan["microbatches"]),
        ),
        ("optimizer_boundaries", PRODUCTION_OPTIMIZER_STEPS),
        ("full_gradient_audits", 1),
    ):
        _expect(f"runtime workload {key}", workload.get(key), expected)
    measurement = _object(
        expected_runtime_budget.get("measurement"),
        "runtime measurement",
    )
    _expect(
        "runtime first optimizer boundary scope",
        measurement.get("optimizer_boundary_scope"),
        "first real gradient clipping and AdamW step; includes initial "
        "optimizer-state allocation",
    )
    excluded = measurement.get("excluded_from_optimizer_boundary")
    if not isinstance(excluded, list) or "one-time full gradient audit" not in excluded:
        raise RuntimeError(
            "runtime boundary measurement does not exclude the full gradient audit"
        )
    projection = _object(
        expected_runtime_budget.get("projection"),
        "runtime projection",
    )
    _expect(
        "runtime one-time gradient-audit projection",
        projection.get("raw_one_time_gradient_audit_seconds"),
        one_time_gradient_audit_seconds,
    )
    _expect(
        "runtime steady-state boundary projection",
        projection.get("raw_optimizer_boundary_seconds"),
        optimizer_boundary_seconds * PRODUCTION_OPTIMIZER_STEPS,
    )
    expected_raw_total = (
        projection["raw_train_microbatch_seconds"]
        + projection["raw_validation_seconds"]
        + projection["raw_optimizer_boundary_seconds"]
        + one_time_gradient_audit_seconds
    )
    _expect(
        "runtime total with one non-repeated gradient audit",
        projection.get("raw_total_seconds"),
        expected_raw_total,
    )
    _expect(
        "runtime production training plan",
        workload.get("training_plan"),
        production_training_plan,
    )
    _expect(
        "runtime subset validation plan",
        workload.get("validation_subset_plan"),
        subset_validation_plan,
    )
    _expect(
        "runtime full validation plan",
        workload.get("full_validation_plan"),
        full_validation_plan,
    )
    early_projection = _object(
        expected_runtime_budget.get("early_stopping_projection"),
        "runtime earliest-stop projection",
    )
    _expect(
        "runtime earliest-stop projection fields",
        set(early_projection),
        {
            "enabled",
            "earliest_stop_step",
            "maximum_step",
            "saved_optimizer_updates",
            "maximum_training_fraction_saved",
            "subset_validation_steps",
            "subset_validation_evaluations",
            "full_validation_steps",
            "full_validation_evaluations",
            "projected_validation_microbatches",
            "raw_validation_seconds",
            "raw_optimizer_boundary_seconds",
            "raw_one_time_gradient_audit_seconds",
            "raw_empirical_p50_total_seconds",
            "raw_empirical_p90_total_seconds",
            "projected_empirical_p50_total_seconds",
            "projected_empirical_p90_total_seconds",
            "projected_empirical_p50_total_hours",
            "projected_empirical_p90_total_hours",
            "allocation",
        },
    )
    expected_early_subset_steps = list(range(1_000, 14_000, 1_000))
    for key, expected in (
        ("enabled", True),
        ("earliest_stop_step", PRODUCTION_EARLIEST_STOP_STEP),
        ("maximum_step", PRODUCTION_OPTIMIZER_STEPS),
        (
            "saved_optimizer_updates",
            PRODUCTION_OPTIMIZER_STEPS - PRODUCTION_EARLIEST_STOP_STEP,
        ),
        (
            "maximum_training_fraction_saved",
            (PRODUCTION_OPTIMIZER_STEPS - PRODUCTION_EARLIEST_STOP_STEP)
            / PRODUCTION_OPTIMIZER_STEPS,
        ),
        ("subset_validation_steps", expected_early_subset_steps),
        ("subset_validation_evaluations", 13),
        ("full_validation_steps", [9_316, 13_973]),
        ("full_validation_evaluations", 2),
        (
            "projected_validation_microbatches",
            13 * int(subset_validation_plan["microbatches"])
            + 2 * int(full_validation_plan["microbatches"]),
        ),
        (
            "raw_optimizer_boundary_seconds",
            optimizer_boundary_seconds * PRODUCTION_EARLIEST_STOP_STEP,
        ),
        (
            "raw_one_time_gradient_audit_seconds",
            one_time_gradient_audit_seconds,
        ),
    ):
        _expect(f"runtime earliest-stop {key}", early_projection.get(key), expected)
    expected_early_p50_raw = (
        production_update_benchmark["seconds_p50"] * PRODUCTION_EARLIEST_STOP_STEP
        + early_projection["raw_validation_seconds"]
        + early_projection["raw_optimizer_boundary_seconds"]
        + one_time_gradient_audit_seconds
    )
    expected_early_p90_raw = (
        production_update_benchmark["seconds_p90"] * PRODUCTION_EARLIEST_STOP_STEP
        + early_projection["raw_validation_seconds"]
        + early_projection["raw_optimizer_boundary_seconds"]
        + one_time_gradient_audit_seconds
    )
    for estimator, raw_seconds in (
        ("p50", expected_early_p50_raw),
        ("p90", expected_early_p90_raw),
    ):
        projected_seconds = raw_seconds * RUNTIME_SAFETY_FACTOR
        _expect(
            f"runtime earliest-stop raw empirical {estimator}",
            early_projection.get(f"raw_empirical_{estimator}_total_seconds"),
            raw_seconds,
        )
        _expect(
            f"runtime earliest-stop projected empirical {estimator}",
            early_projection.get(f"projected_empirical_{estimator}_total_seconds"),
            projected_seconds,
        )
        _expect(
            f"runtime earliest-stop projected empirical {estimator} hours",
            early_projection.get(f"projected_empirical_{estimator}_total_hours"),
            projected_seconds / 3_600,
        )
    output_only = _object(
        smoke.get("output_only_supervision"),
        "training output-only supervision",
    )
    for key, expected in (
        ("prompt_labels_all_masked", True),
        ("prompt_weights_all_zero", True),
        ("supervised_prefix_is_answer_body", True),
        ("template_suffix_starts_with_eos", True),
        ("supervised_sequence_ends_at_eos", True),
        ("translation_or_summarization_body_only", True),
        ("reasoning_effort", "none"),
    ):
        _expect(f"training output-only {key}", output_only.get(key), expected)
    for key in ("prompt_token_count", "answer_token_count"):
        if int(output_only.get(key, 0)) <= 0:
            raise RuntimeError(f"training output-only {key} must be positive")
    termination_count = output_only.get("assistant_termination_token_count")
    if (
        isinstance(termination_count, bool)
        or not isinstance(termination_count, int)
        or termination_count != 1
    ):
        raise RuntimeError(
            "training output-only assistant termination count must be exactly one"
        )
    termination_token_id = output_only.get("assistant_termination_token_id")
    if (
        isinstance(termination_token_id, bool)
        or not isinstance(termination_token_id, int)
        or termination_token_id < 0
    ):
        raise RuntimeError(
            "training output-only assistant termination token id is invalid"
        )
    suffix_token_count = output_only.get("template_suffix_token_count")
    if (
        isinstance(suffix_token_count, bool)
        or not isinstance(suffix_token_count, int)
        or suffix_token_count < 1
    ):
        raise RuntimeError(
            "training output-only template suffix token count is invalid"
        )
    expected_first_output = dict(output_only)
    expected_first_output.pop("translation_or_summarization_body_only")
    _expect(
        "training first output-only supervision",
        training.get("first_output_only_supervision"),
        expected_first_output,
    )

    gradient = _object(smoke.get("gradient_audit"), "LoRA gradient audit")
    _expect("LoRA gradient finite", gradient.get("finite"), True)
    if int(gradient.get("parameters_with_gradient", 0)) <= 0:
        raise RuntimeError("LoRA gradient audit has no gradients")
    if int(gradient.get("nonzero_gradient_elements", 0)) <= 0:
        raise RuntimeError("LoRA gradient audit is entirely zero")
    _positive_finite(gradient.get("l2"), "LoRA gradient l2")
    _expect(
        "training first gradient audit",
        training.get("first_update_gradient_audit"),
        gradient,
    )

    update = _object(smoke.get("adapter_update"), "LoRA adapter update")
    before_sha = _sha256_text(update.get("before_sha256"), "LoRA before digest")
    after_sha = _sha256_text(update.get("after_sha256"), "LoRA after digest")
    _boolean(update.get("changed"), "LoRA update changed flag", True)
    if before_sha == after_sha:
        raise RuntimeError("LoRA before/after digests are identical")
    _expect(
        "LoRA effective update optimizer step",
        _integer(
            update.get("optimizer_step"),
            "LoRA effective update optimizer step",
            minimum=1,
        ),
        1,
    )
    warmup_updates = int(float(recipe["warmup_ratio"]) * int(recipe["optimizer_steps"]))
    _expect("production scheduler warmup updates", warmup_updates, 558)
    expected_first_learning_rate = positive_cosine_learning_rate(
        update=1,
        total_updates=int(recipe["optimizer_steps"]),
        warmup_updates=warmup_updates,
        peak_learning_rate=float(recipe["learning_rate"]),
        minimum_learning_rate_ratio=float(recipe["minimum_learning_rate_ratio"]),
    )
    _expect(
        "production first-update warmup formula",
        expected_first_learning_rate,
        float(recipe["learning_rate"]) / warmup_updates,
    )
    observed_first_learning_rate = _positive_finite(
        update.get("learning_rate_used"),
        "LoRA effective update learning rate",
    )
    if not math.isclose(
        observed_first_learning_rate,
        expected_first_learning_rate,
        rel_tol=1e-12,
        abs_tol=0.0,
    ):
        _fail(
            "LoRA effective update learning rate",
            observed_first_learning_rate,
            expected_first_learning_rate,
        )
    _validate_stats(update.get("before"), "LoRA pre-update stats")
    _validate_stats(update.get("after"), "LoRA post-update stats")
    _expect(
        "training first adapter update",
        training.get("first_update_adapter_delta"),
        update,
    )
    replay = _object(
        smoke.get("post_update_recompute_audit"),
        "post-update checkpoint recompute audit",
    )
    _expect(
        "post-update checkpoint recompute fields",
        set(replay),
        {
            "status",
            "phase",
            "row_id",
            "observed_tokens",
            "hard_term_weighted_tokens",
            "loss",
            "seconds",
            "gradient_audit",
            "adapter_parameters_unchanged",
            "frozen_parameters_unchanged",
        },
    )
    _expect(
        "training post-update checkpoint recompute audit",
        training.get("post_update_recompute_audit"),
        replay,
    )
    _expect("post-update recompute status", replay.get("status"), "PASS")
    _expect(
        "post-update recompute phase",
        replay.get("phase"),
        "after_first_positive_lr_optimizer_update",
    )
    row_id = replay.get("row_id")
    if not isinstance(row_id, str) or not row_id:
        raise RuntimeError("post-update recompute row ID is invalid")
    replay_tokens = _integer(
        replay.get("observed_tokens"),
        "post-update recompute observed tokens",
        minimum=1,
    )
    _expect(
        "post-update absolute-longest F+B token count",
        replay_tokens,
        ABSOLUTE_TRAIN_MAX_TOKENS,
    )
    _expect(
        "post-update/token-contract absolute-longest token count",
        replay_tokens,
        train_length_contract["maximum"],
    )
    _expect(
        "post-update/first-update max-length token count",
        replay_tokens,
        smoke["observed_tokens"],
    )
    _expect(
        "post-update/training-benchmark absolute-longest row",
        row_id,
        training_bucket_benchmarks[str(PRODUCTION_MAX_LENGTH)]["row_ids"][0],
    )
    _integer(
        replay.get("hard_term_weighted_tokens"),
        "post-update recompute hard-term weighted tokens",
        minimum=0,
    )
    _positive_finite(replay.get("loss"), "post-update recompute loss")
    _positive_finite(replay.get("seconds"), "post-update recompute seconds")
    replay_gradient = _object(
        replay.get("gradient_audit"),
        "post-update recompute gradient audit",
    )
    _expect(
        "post-update gradient fields",
        set(replay_gradient),
        {
            "parameters_with_gradient",
            "nonzero_gradient_elements",
            "finite",
            "l2",
        },
    )
    _boolean(
        replay_gradient.get("finite"),
        "post-update gradient finite",
        True,
    )
    replay_parameters = _integer(
        replay_gradient.get("parameters_with_gradient"),
        "post-update parameters with gradient",
        minimum=1,
    )
    if replay_parameters > EXPECTED_LORA_TENSORS:
        raise RuntimeError(
            "post-update recompute reports too many parameters with gradients"
        )
    replay_nonzero = _integer(
        replay_gradient.get("nonzero_gradient_elements"),
        "post-update nonzero gradient elements",
        minimum=1,
    )
    if replay_nonzero > EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError(
            "post-update recompute reports too many nonzero gradient elements"
        )
    _positive_finite(
        replay_gradient.get("l2"),
        "post-update recompute LoRA gradient l2",
    )
    _boolean(
        replay.get("adapter_parameters_unchanged"),
        "post-update recompute adapter unchanged",
        True,
    )
    _boolean(
        replay.get("frozen_parameters_unchanged"),
        "post-update recompute frozen parameters unchanged",
        True,
    )

    before_probe = _object(
        training.get("frozen_probe_before"), "frozen base probe before"
    )
    after_probe = _object(
        training.get("frozen_probe_after_update"), "frozen base probe after"
    )
    _expect("frozen base probe equality", after_probe, before_probe)
    _expect(
        "frozen base unchanged flag",
        smoke.get("frozen_base_probe_unchanged"),
        True,
    )
    _expect(
        "frozen base sampled parameters",
        before_probe.get("sampled_parameter_count"),
        32,
    )
    _sha256_text(before_probe.get("values_sha256"), "frozen base probe digest")

    module_audit = _object(
        training.get("lora_module_audit"), "training LoRA module audit"
    )
    _expect("training LoRA module status", module_audit.get("status"), "PASS")
    _expect(
        "training LoRA module count",
        module_audit.get("module_count"),
        EXPECTED_LORA_MODULES,
    )
    _expect(
        "training trainable parameter count",
        training.get("trainable_parameters"),
        EXPECTED_TRAINABLE_PARAMETERS,
    )
    device = _object(training.get("device_audit"), "training device audit")
    _expect("training CPU/disk offload", device.get("cpu_disk_offload"), False)
    for value in device.get("hf_device_map_devices") or []:
        if str(value).lower() in {"cpu", "disk", "meta"}:
            raise RuntimeError(f"forbidden training device-map entry: {value}")
    parameter_devices = _object(
        device.get("parameter_devices"), "training parameter devices"
    )
    if not parameter_devices or any(
        not str(name).startswith("cuda") for name in parameter_devices
    ):
        raise RuntimeError("training manifest contains non-CUDA parameters")

    smoke_validation = _object(history[-1], "final smoke validation")
    for key, expected in (
        ("step", 1),
        ("scope", "fixed_subset"),
        ("translation_count", 1),
        ("summarization_count", 1),
        ("dropped_over_max", {}),
    ):
        _expect(f"final smoke validation {key}", smoke_validation.get(key), expected)

    final_checkpoint = _object(
        training.get("final_checkpoint"),
        "smoke final checkpoint",
    )
    final_dir = require_owned_path(Path(str(final_checkpoint.get("adapter_dir") or "")))
    _expect(
        "smoke final checkpoint directory",
        final_dir,
        output_dir / recipe["checkpoint"]["final_adapter_dirname"],
    )
    _expect("smoke final checkpoint step", final_checkpoint.get("step"), 1)
    final_success = _regular_file(
        final_dir / "_SUCCESS",
        "smoke final adapter _SUCCESS",
    )
    _expect(
        "smoke final adapter success marker",
        final_success.read_text(encoding="utf-8").strip(),
        "PASS",
    )
    final_files = _current_file_map(final_dir)
    _expect(
        "smoke final adapter file hashes",
        final_checkpoint.get("files_sha256"),
        final_files,
    )

    expected_selection = {
        "candidate_steps": [1],
        "candidates": [
            {
                "step": 1,
                "selection_loss": smoke_validation["selection_loss"],
                "translation_loss": smoke_validation["translation_loss"],
                "summarization_loss": smoke_validation["summarization_loss"],
                "adapter_dir": str(final_dir),
                "files_sha256": final_files,
            }
        ],
        "selected_step": 1,
        "selected_selection_loss": smoke_validation["selection_loss"],
        "policy": "smoke_final_step",
        "tie_break": "not_applicable",
    }
    _expect(
        "smoke checkpoint selection record",
        training.get("selection_record"),
        expected_selection,
    )

    selected = _object(training.get("selected_checkpoint"), "selected checkpoint")
    adapter_dir = require_owned_path(Path(str(selected.get("adapter_dir") or "")))
    expected_selected = {
        "policy": "smoke_final_step",
        "tie_break": "not_applicable",
        "step": 1,
        "selection_loss": smoke_validation["selection_loss"],
        "source_checkpoint_dir": str(final_dir),
        "adapter_dir": str(output_dir / recipe["checkpoint"]["best_adapter_dirname"]),
    }
    for key, expected in expected_selected.items():
        _expect(f"selected checkpoint {key}", selected.get(key), expected)
    _expect(
        "selected best-adapter directory",
        adapter_dir,
        output_dir / recipe["checkpoint"]["best_adapter_dirname"],
    )
    success = _regular_file(adapter_dir / "_SUCCESS", "selected adapter _SUCCESS")
    _expect(
        "selected adapter success marker",
        success.read_text(encoding="utf-8").strip(),
        "PASS",
    )
    current_files = _current_file_map(adapter_dir)
    _expect("selected adapter file hashes", selected.get("files_sha256"), current_files)
    _expect(
        "smoke adapter file hashes",
        smoke.get("adapter_files_sha256"),
        current_files,
    )
    return training, adapter_dir, current_files


def _validate_implementation(
    implementation: dict[str, Any], identities: dict[str, Any]
) -> None:
    wanted = {
        "infer_py_sha256": sha256(Path(__file__).with_name("infer.py")),
        "inference_loader_sha256": sha256(
            Path(__file__).with_name("solar_inference_loader.py")
        ),
        "reference_evaluator_sha256": REFERENCE_SHA256,
        "model_contract_sha256": identities["model_contract_sha256"],
        "recipe_sha256": identities["recipe_sha256"],
        "data_contract_sha256": identities["data_contract_sha256"],
        "environment_lock_sha256": identities["environment_lock_sha256"],
        "pipeline_code_sha256": identities["pipeline_code"]["sha256"],
    }
    for key, expected in wanted.items():
        _expect(f"inference implementation {key}", implementation.get(key), expected)
    for key in ("transformers_version", "peft_version", "bitsandbytes_version"):
        if not str(implementation.get(key) or "").strip():
            raise RuntimeError(f"inference implementation lacks {key}")


def _validate_prediction_file(
    path: Path,
    *,
    role: str,
    source: dict[str, Any],
    inference_expert: dict[str, Any],
    adapter: dict[str, Any] | None,
    identities: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    _regular_file(path, f"{role} predictions")
    manifest_path = _regular_file(
        path.with_name(path.name + ".manifest.json"),
        f"{role} prediction manifest",
    )
    manifest = read_json_no_duplicates(manifest_path)
    for key, expected in (
        ("schema", PROVENANCE_SCHEMA),
        ("schema_version", PROVENANCE_SCHEMA_VERSION),
        ("status", "COMPLETE"),
        ("expected_rows", 2),
        ("completed_rows", 2),
        ("output_jsonl_sha256", sha256(path)),
    ):
        _expect(f"{role} manifest {key}", manifest.get(key), expected)

    contract = _object(manifest.get("contract"), f"{role} inference contract")
    if "adapter" not in contract:
        raise RuntimeError(f"{role} inference contract lacks adapter=null/object")
    contract_sha256 = canonical_sha256(contract)
    _expect(
        f"{role} inference contract SHA",
        manifest.get("contract_sha256"),
        contract_sha256,
    )
    for key, expected in (
        ("schema", PROVENANCE_SCHEMA),
        ("schema_version", PROVENANCE_SCHEMA_VERSION),
        ("source", source),
        ("per_expert_nf4", inference_expert),
        ("adapter", adapter),
    ):
        _expect(f"{role} contract {key}", contract.get(key), expected)

    contract_eval = _object(contract.get("evaluation"), f"{role} evaluation")
    for key, expected in (
        ("role", "full_test_plus_human_gold_sealed_union"),
        ("path", evaluation["path"]),
        ("sha256", evaluation["sha256"]),
        ("pinned_total_rows", PINNED_COMBINED_EVAL_ROWS),
        ("pinned_by_task", PINNED_COMBINED_EVAL_TASK_COUNTS),
        ("selected_rows", 2),
        ("limit_per_task", 1),
    ):
        _expect(f"{role} evaluation {key}", contract_eval.get(key), expected)
    _validate_implementation(
        _object(contract.get("implementation"), f"{role} implementation"),
        identities,
    )
    _expect(
        f"{role} prompt",
        contract.get("prompt"),
        {"variant": "no_dictionary_context", **TEMPLATE_KWARGS},
    )
    decoding = _object(contract.get("decoding"), f"{role} decoding")
    _expect(
        f"{role} decoding SHA",
        contract.get("decoding_sha256"),
        canonical_sha256(decoding),
    )
    for key, expected in (
        ("padding_side", "left"),
        ("raw_output_policy", "outer strip only; no filtering or repair"),
    ):
        _expect(f"{role} decoding {key}", decoding.get(key), expected)
    for key, expected in (("temperature", 0.0), ("top_p", 1.0)):
        value = decoding.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"{role} decoding {key} must be numeric")
        _expect(f"{role} decoding {key}", float(value), expected)
    _boolean(
        decoding.get("do_sample"),
        f"{role} decoding do_sample",
        False,
    )
    _expect(
        f"{role} decoding num_beams",
        _integer(decoding.get("num_beams"), f"{role} decoding num_beams"),
        1,
    )
    for key, expected in (
        ("max_model_len", SMOKE_MAX_MODEL_LEN),
        ("translation_max_new_tokens", SMOKE_MAX_NEW_TOKENS),
        ("summarization_max_new_tokens", SMOKE_MAX_NEW_TOKENS),
    ):
        _expect(
            f"{role} decoding {key}",
            _integer(decoding.get(key), f"{role} decoding {key}"),
            expected,
        )
    runtime = _object(contract.get("runtime"), f"{role} runtime")
    _expect(
        f"{role} runtime strategy",
        runtime.get("strategy"),
        "single_process_three_gpu_model_parallel",
    )
    _expect(
        f"{role} routed expert format",
        runtime.get("routed_expert_format"),
        "per_expert_nf4",
    )
    _expect(
        f"{role} generation batch size",
        _integer(
            runtime.get("generation_batch_size"),
            f"{role} generation batch size",
        ),
        SMOKE_GENERATION_BATCH_SIZE,
    )

    rows = read_prediction_rows(path)
    if len(rows) != 2:
        raise RuntimeError(f"{role} smoke predictions must have exactly two rows")
    keys = [(str(row.get("id") or ""), str(row.get("task") or "")) for row in rows]
    if any(not row_id for row_id, _task in keys) or len(set(keys)) != 2:
        raise RuntimeError(f"{role} prediction keys are blank or duplicated")
    if {task for _row_id, task in keys} != EXPECTED_TASKS:
        raise RuntimeError(f"{role} predictions do not cover exactly two tasks")
    _expect(
        f"{role} selected key SHA",
        contract_eval.get("selected_keys_sha256"),
        canonical_sha256([list(key) for key in keys]),
    )
    labels = {str(row.get("model_label") or "") for row in rows}
    if len(labels) != 1 or "" in labels:
        raise RuntimeError(f"{role} prediction labels are inconsistent")
    _expect(
        f"{role} row/contract label",
        next(iter(labels)),
        contract.get("model_label"),
    )
    invariants = expected_row_provenance(contract, contract_sha256)
    for line_number, row in enumerate(rows, 1):
        for field, expected in invariants.items():
            _expect(
                f"{role} row {line_number} provenance {field}",
                row.get(field),
                expected,
            )
        for field, expected in (
            ("prompt_variant", "no_dictionary_context"),
            ("reasoning_effort", "none"),
        ):
            _expect(f"{role} row {line_number} {field}", row.get(field), expected)
        for field in ("origin", "reference", "prediction"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise RuntimeError(f"{role} row {line_number} has empty {field}")
        prediction = row["prediction"]
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
                f"{role} row {line_number} has invalid Unicode: "
                f"{invalid_unicode[:8]}"
            )
        prompt_tokens = row.get("prompt_tokens")
        prediction_tokens = row.get("prediction_tokens")
        for field, value in (
            ("prompt_tokens", prompt_tokens),
            ("prediction_tokens", prediction_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise RuntimeError(
                    f"{role} row {line_number} has invalid {field}: {value!r}"
                )
        task = str(row["task"])
        maximum = int(decoding[f"{task}_max_new_tokens"])
        if prediction_tokens >= maximum:
            raise RuntimeError(
                f"{role} row {line_number} exhausted its generation cap: "
                f"{prediction_tokens}/{maximum}"
            )
        _expect(
            f"{role} row {line_number} generation stop reason",
            row.get("generation_stop_reason"),
            "eos",
        )
        _boolean(
            row.get("tokenizer_roundtrip_exact"),
            f"{role} row {line_number} tokenizer roundtrip",
            True,
        )
        violations = output_only_violations(prediction)
        if violations:
            raise RuntimeError(
                f"{role} row {line_number} has non-body prediction: " f"{violations}"
            )

    audit = _object(manifest.get("runtime_audit"), f"{role} runtime audit")
    model_audit = _object(audit.get("model_audit"), f"{role} model audit")
    _expect(f"{role} model audit status", model_audit.get("status"), "PASS")
    _boolean(
        audit.get("hash_verification"),
        f"{role} runtime hash verification",
        True,
    )
    _boolean(
        audit.get("production_eligible"),
        f"{role} runtime production eligibility",
        True,
    )
    verify_recorded_runtime_identity(audit, contract)
    if "adapter_audit" not in audit:
        raise RuntimeError(f"{role} runtime audit lacks adapter_audit=null/object")
    adapter_audit = audit.get("adapter_audit")
    if adapter is None:
        _expect(f"{role} base adapter audit", adapter_audit, None)
    else:
        adapter_audit = _object(adapter_audit, f"{role} adapter audit")
        wanted_audit = {
            "status": "PASS",
            "module_count": EXPECTED_LORA_MODULES,
            "adapter_tensor_count": EXPECTED_LORA_TENSORS,
            "adapter_tensor_names_sha256": adapter["serialized_keys_sha256"],
            "adapter_parameter_count": EXPECTED_TRAINABLE_PARAMETERS,
            "adapter_dtypes": {"torch.float32": EXPECTED_LORA_TENSORS},
            "trainable_parameter_count": 0,
            "offloaded_adapter_tensor_count": 0,
        }
        for key, expected in wanted_audit.items():
            _expect(f"{role} adapter audit {key}", adapter_audit.get(key), expected)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "contract_sha256": contract_sha256,
        "model_label": next(iter(labels)),
        "keys": keys,
        "evaluation": contract_eval,
        "decoding": decoding,
    }


def _b200_owned_regular_file(
    path: Path,
    label: str,
    *,
    expected_path: Path | None = None,
) -> Path:
    raw = path.expanduser()
    if raw.is_symlink():
        raise RuntimeError(f"{label} must not be a symlink: {raw}")
    resolved = b200_runtime.require_owned_path(raw, must_exist=True)
    if not resolved.is_file():
        raise RuntimeError(f"{label} is not a regular file: {resolved}")
    if expected_path is not None and resolved != expected_path.resolve():
        raise RuntimeError(
            f"{label} path mismatch: actual={resolved}, "
            f"expected={expected_path.resolve()}"
        )
    return resolved


def _b200_receipt(
    path: Path,
    label: str,
    *,
    expected_path: Path | None = None,
) -> tuple[Path, dict[str, Any], dict[str, str]]:
    resolved = _b200_owned_regular_file(
        path,
        label,
        expected_path=expected_path,
    )
    value = b200_runtime.read_json_no_duplicates(resolved)
    return (
        resolved,
        value,
        {
            "path": str(resolved),
            "sha256": b200_runtime.sha256_file(resolved),
        },
    )


def _b200_current_identities(config_dir: Path) -> dict[str, Any]:
    bundle = b200_runtime.load_b200_config_bundle(config_dir)
    config = b200_runtime.b200_config_identity(config_dir)
    pipeline = pipeline_code_identity(config_dir=bundle["config_dir"])
    return {
        "bundle": bundle,
        "config": config,
        "pipeline_code": pipeline,
        "environment_lock_sha256": b200_runtime.sha256_file(
            b200_runtime.PROJECT_DIR / "requirements/solar-open-2.lock.txt"
        ),
    }


def _validate_b200_gpu_evidence(value: Any, label: str) -> dict[str, Any]:
    hardware = _object(value, label)
    for key, expected in (
        ("status", "PASS"),
        ("physical_cuda_visible_devices", "6,7"),
        ("logical_gpu_count", 2),
        ("boundary_layers", [24]),
    ):
        _expect(f"{label} {key}", hardware.get(key), expected)
    if "sm_100" not in (hardware.get("torch_arch_list") or []):
        raise RuntimeError(f"{label} lacks sm_100")
    devices = hardware.get("logical_to_physical")
    if not isinstance(devices, list) or len(devices) != 2:
        raise RuntimeError(f"{label} must contain exactly two GPU records")
    for logical_index, device_value in enumerate(devices):
        device = _object(device_value, f"{label} cuda:{logical_index}")
        for key, expected in (
            ("logical_index", logical_index),
            ("logical_device", f"cuda:{logical_index}"),
            ("physical_index", 6 + logical_index),
            ("compute_capability", [10, 0]),
        ):
            _expect(
                f"{label} cuda:{logical_index} {key}",
                device.get(key),
                expected,
            )
        if "B200" not in str(device.get("name") or "").upper():
            raise RuntimeError(f"{label} cuda:{logical_index} is not B200")
        memory = _integer(
            device.get("total_memory_bytes"),
            f"{label} cuda:{logical_index} total memory",
            minimum=1,
        )
        if memory < b200_runtime.MINIMUM_MEMORY_BYTES_PER_GPU:
            raise RuntimeError(f"{label} cuda:{logical_index} memory is too small")
    return hardware


def _validate_b200_hf_device_map(value: Any, label: str) -> dict[str, str]:
    device_map = _object(value, label)
    expected = dict(b200_runtime.EXPECTED_HF_DEVICE_MAP)
    _expect(label, device_map, expected)
    if list(device_map.values()).count("cuda:0") != 25:
        raise RuntimeError(f"{label} does not contain embed plus 24 layers on cuda:0")
    if list(device_map.values()).count("cuda:1") != 26:
        raise RuntimeError(
            f"{label} does not contain 24 layers plus norm/head on cuda:1"
        )
    if any(
        str(device).lower() in {"cpu", "disk", "meta", "cuda:2"}
        for device in device_map.values()
    ):
        raise RuntimeError(f"{label} contains forbidden offload/device entries")
    return device_map


def _validate_b200_environment_receipt(
    path: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    _path, receipt, identity = _b200_receipt(
        path,
        "B200 environment receipt",
        expected_path=B200_ENVIRONMENT_RECEIPT,
    )
    for key, expected in (
        ("schema", "solar_open2_b200_environment"),
        ("schema_version", 1),
        ("status", "PASS"),
        ("root", str(b200_runtime.OWNED_ROOT)),
        ("user", b200_runtime.EXPECTED_USER),
        ("secrets_recorded", False),
        ("b200_production_authorized", False),
    ):
        _expect(f"B200 environment {key}", receipt.get(key), expected)
    python = _object(receipt.get("python"), "B200 environment Python")
    _expect(
        "B200 environment sys.prefix",
        python.get("sys_prefix"),
        str(b200_runtime.VENV_DIR),
    )
    lock = _object(
        receipt.get("dependency_lock"),
        "B200 environment dependency lock",
    )
    lock_path = _b200_owned_regular_file(
        Path(str(lock.get("path") or "")),
        "B200 dependency lock",
        expected_path=(b200_runtime.PROJECT_DIR / "requirements/solar-open-2.lock.txt"),
    )
    _expect(
        "B200 environment dependency lock SHA",
        lock.get("sha256"),
        b200_runtime.sha256_file(lock_path),
    )
    torch = _object(receipt.get("torch"), "B200 environment torch")
    if "sm_100" not in (torch.get("compiled_arch_flags") or []):
        raise RuntimeError("B200 environment torch lacks sm_100")
    libraries = _object(
        receipt.get("native_libraries"),
        "B200 environment native libraries",
    )
    bnb = _object(
        libraries.get("libbitsandbytes_cuda128"),
        "B200 environment bitsandbytes CUDA 12.8 library",
    )
    library_path = _b200_owned_regular_file(
        Path(str(bnb.get("path") or "")),
        "B200 bitsandbytes CUDA 12.8 library",
    )
    _expect(
        "B200 bitsandbytes CUDA 12.8 library name",
        library_path.name,
        "libbitsandbytes_cuda128.so",
    )
    _expect(
        "B200 bitsandbytes CUDA 12.8 library SHA",
        bnb.get("sha256"),
        b200_runtime.sha256_file(library_path),
    )
    return receipt, identity


def _validate_b200_asset_integrity_receipt(
    path: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    resolved, receipt, identity = _b200_receipt(
        path,
        "B200 large-hash asset receipt",
        expected_path=B200_ASSET_INTEGRITY_RECEIPT,
    )
    for key, expected in (
        ("schema", "solar_open2_b200_asset_integrity"),
        ("schema_version", 1),
        ("status", "PASS"),
        ("mode", "full"),
        ("root", str(b200_runtime.OWNED_ROOT)),
        ("revision", BASE_REVISION),
        ("reference_sha256", b200_runtime.REFERENCE_EVALUATOR_SHA256),
        ("large_hashes_checked", True),
        ("b200_production_authorized", False),
        ("user", b200_runtime.EXPECTED_USER),
    ):
        _expect(f"B200 asset receipt {key}", receipt.get(key), expected)
    if "receipt_path" in receipt:
        _expect(
            "B200 asset receipt self path",
            receipt.get("receipt_path"),
            str(resolved),
        )
    for key in ("bf16", "training_nf4", "inference_nf4", "metric_model"):
        entry = _object(receipt.get(key), f"B200 asset receipt {key}")
        _expect(f"B200 asset receipt {key} status", entry.get("status"), "PASS")
    return receipt, identity


def _validate_b200_kernel_probe_receipt(
    path: Path,
    *,
    identities: dict[str, Any],
    environment_identity: dict[str, str],
) -> tuple[dict[str, Any], dict[str, str]]:
    _path, receipt, identity = _b200_receipt(
        path,
        "B200 kernel probe receipt",
        expected_path=B200_KERNEL_PROBE_RECEIPT,
    )
    for key, expected in (
        ("schema", "solar_open2_b200_kernel_probe"),
        ("schema_version", 1),
        ("status", "PASS"),
        ("profile", "b200"),
        ("scope", "environment_and_small_nf4_kernel_only"),
        ("root", str(b200_runtime.OWNED_ROOT)),
        ("user", b200_runtime.EXPECTED_USER),
        ("production_authorized", False),
        ("physical_cuda_visible_devices", "6,7"),
        ("logical_device_mapping", {"cuda:0": 6, "cuda:1": 7}),
        ("config", identities["config"]),
        ("pipeline_code", identities["pipeline_code"]),
    ):
        _expect(f"B200 kernel probe {key}", receipt.get(key), expected)
    environment = _object(
        receipt.get("environment"),
        "B200 kernel probe environment binding",
    )
    for key, expected in environment_identity.items():
        _expect(
            f"B200 kernel probe environment {key}",
            environment.get(key),
            expected,
        )
    _validate_b200_gpu_evidence(receipt.get("hardware"), "B200 kernel hardware")
    runtime = _object(
        receipt.get("bitsandbytes_runtime"),
        "B200 kernel bitsandbytes runtime",
    )
    for key, expected in (
        ("status", "PASS"),
        ("compute_capability", [10, 0]),
        ("manifest_status", "SEALED"),
        ("bitsandbytes_version", "0.49.2"),
        ("native_library", "libbitsandbytes_cuda128.so"),
        ("runtime_profile", "b200_sealed_artifact"),
    ):
        _expect(f"B200 kernel bitsandbytes {key}", runtime.get(key), expected)
    peer = _object(receipt.get("peer_access"), "B200 kernel peer access")
    _expect(
        "B200 kernel peer access",
        peer,
        {
            "status": "PASS",
            "cuda_0_to_1_access": True,
            "cuda_1_to_0_access": True,
            "roundtrip_equal": True,
        },
    )
    devices = receipt.get("devices")
    if not isinstance(devices, list) or len(devices) != 2:
        raise RuntimeError("B200 kernel probe must contain two device probes")
    for index, probe_value in enumerate(devices):
        probe = _object(probe_value, f"B200 kernel device probe {index}")
        for key, expected in (
            ("status", "PASS"),
            ("logical_index", index),
            ("logical_device", f"cuda:{index}"),
            ("nf4_quantize_dequantize_finite", True),
            ("bf16_input_gradient_finite", True),
            ("bf16_input_gradient_nonzero", True),
            ("loss_finite", True),
        ):
            _expect(f"B200 kernel device {index} {key}", probe.get(key), expected)
    return receipt, identity


def _validate_b200_tokenizer_receipt(
    path: Path,
    *,
    identities: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    _path, receipt, identity = _b200_receipt(
        path,
        "B200 tokenizer contract receipt",
        expected_path=B200_TOKENIZER_CONTRACT_RECEIPT,
    )
    for key, expected in (
        ("schema", "solar_open2_b200_tokenizer_contract"),
        ("schema_version", 1),
        ("status", "PASS"),
        ("profile", "b200"),
        ("user", b200_runtime.EXPECTED_USER),
        ("root", str(b200_runtime.OWNED_ROOT)),
        ("config", identities["config"]),
        ("pipeline_code", identities["pipeline_code"]),
        ("production_authorized", False),
    ):
        _expect(f"B200 tokenizer receipt {key}", receipt.get(key), expected)
    source = _object(receipt.get("source"), "B200 tokenizer source")
    _expect(
        "B200 tokenizer source path",
        source.get("path"),
        str(b200_runtime.BF16_SOURCE_DIR),
    )
    data = _object(receipt.get("data"), "B200 tokenizer data")
    files = _object(data.get("files"), "B200 tokenizer data files")
    if set(files) != {"train", "validation"}:
        raise RuntimeError("B200 tokenizer receipt opened test/human-gold data")
    contract = identities["bundle"]["data_contract"]
    for role in ("train", "validation"):
        entry = _object(files.get(role), f"B200 tokenizer {role}")
        configured = contract["files"][role]
        for key, expected in (
            ("path", configured["path"]),
            ("sha256", configured["sha256"]),
            ("rows", configured["rows"]),
        ):
            _expect(f"B200 tokenizer {role} {key}", entry.get(key), expected)
    reference = _object(receipt.get("reference"), "B200 tokenizer reference")
    _expect(
        "B200 tokenizer reference path",
        reference.get("path"),
        str(b200_runtime.REFERENCE_EVALUATOR),
    )
    _expect(
        "B200 tokenizer reference SHA",
        reference.get("sha256"),
        b200_runtime.REFERENCE_EVALUATOR_SHA256,
    )
    output_contract = _object(
        receipt.get("output_only_contract"),
        "B200 tokenizer output-only contract",
    )
    _expect("B200 tokenizer output-only status", output_contract.get("status"), "PASS")
    supervised_eos = output_contract.get("supervised_termination_token_ids")
    generation_eos = output_contract.get("configured_generation_eos_token_ids")
    if (
        not isinstance(supervised_eos, list)
        or len(supervised_eos) != 1
        or not isinstance(generation_eos, list)
        or not set(supervised_eos).issubset(set(generation_eos))
    ):
        raise RuntimeError("B200 tokenizer supervised EOS contract is invalid")
    checks = _object(receipt.get("checks"), "B200 tokenizer checks")
    if not checks or any(value != "PASS" for value in checks.values()):
        raise RuntimeError("B200 tokenizer receipt contains a failed check")
    prompt_budget = _object(
        receipt.get("full_validation_prompt_budget"),
        "B200 full validation prompt budget",
    )
    for key, expected in (
        ("status", "PASS"),
        ("path", contract["files"]["validation"]["path"]),
        ("rows", contract["files"]["validation"]["rows"]),
        ("truncation_count", 0),
    ):
        _expect(
            f"B200 validation prompt budget {key}", prompt_budget.get(key), expected
        )
    validate_b200_input_unicode_inventory(
        prompt_budget.get("input_unicode_inventory"),
        expected_rows=int(contract["files"]["validation"]["rows"]),
        expected_anomalies=B200_VALIDATION_UNICODE_EXPECTED,
    )
    for field in ("invalid_unicode_count", "target_invalid_unicode_count"):
        _expect(
            f"B200 validation prompt budget {field}",
            prompt_budget.get(field),
            0,
        )
    return receipt, identity


def _validate_b200_preflight_receipt(
    path: Path,
    *,
    identities: dict[str, Any],
    environment_identity: dict[str, str],
) -> tuple[dict[str, Any], dict[str, str]]:
    _path, receipt, identity = _b200_receipt(
        path,
        "B200 model preflight receipt",
        expected_path=B200_MODEL_PREFLIGHT_RECEIPT,
    )
    for key, expected in (
        ("schema", "solar_open2_b200_preflight"),
        ("schema_version", 1),
        ("status", "PASS"),
        ("profile", "b200"),
        ("user", b200_runtime.EXPECTED_USER),
        ("root", str(b200_runtime.OWNED_ROOT)),
        ("config", identities["config"]),
        ("pipeline_code", identities["pipeline_code"]),
        ("physical_cuda_visible_devices", "6,7"),
        ("production_authorized", False),
    ):
        _expect(f"B200 preflight {key}", receipt.get(key), expected)
    environment = _object(receipt.get("environment"), "B200 preflight environment")
    for key, expected in environment_identity.items():
        _expect(f"B200 preflight environment {key}", environment.get(key), expected)
    _validate_b200_gpu_evidence(receipt.get("hardware"), "B200 preflight hardware")
    transfer = _object(
        receipt.get("transfer_integrity"),
        "B200 preflight transfer integrity",
    )
    transfer_path = _b200_owned_regular_file(
        Path(str(transfer.get("path") or "")),
        "B200 transfer integrity receipt",
    )
    _expect(
        "B200 transfer integrity SHA",
        transfer.get("sha256"),
        b200_runtime.sha256_file(transfer_path),
    )
    _expect("B200 transfer large hashes", transfer.get("large_hashes_checked"), True)
    bf16 = _object(receipt.get("bf16"), "B200 preflight BF16")
    for key, expected in (
        ("status", "PASS"),
        ("relocated_snapshot_dir", str(b200_runtime.BF16_SOURCE_DIR)),
        ("revision", BASE_REVISION),
        ("indexed_shards", EXPECTED_SOURCE_SHARD_COUNT),
        ("indexed_tensors", EXPECTED_SOURCE_TENSOR_COUNT),
        ("relocation_verified_by_content_hash", True),
    ):
        _expect(f"B200 preflight BF16 {key}", bf16.get(key), expected)
    download_path = _b200_owned_regular_file(
        Path(str(bf16.get("receipt_path") or "")),
        "B200 BF16 download receipt",
    )
    _expect(
        "B200 BF16 receipt SHA",
        bf16.get("receipt_sha256"),
        b200_runtime.sha256_file(download_path),
    )
    for role, expected_path in (
        ("training_nf4", b200_runtime.TRAIN_EXPERT_DIR),
        ("inference_nf4", b200_runtime.INFERENCE_EXPERT_DIR),
    ):
        artifact = _object(receipt.get(role), f"B200 preflight {role}")
        _expect(f"B200 preflight {role} status", artifact.get("status"), "PASS")
        _expect(f"B200 preflight {role} path", artifact.get("path"), str(expected_path))
        for nested in ("manifest", "index"):
            record = _object(artifact.get(nested), f"B200 preflight {role} {nested}")
            artifact_path = _b200_owned_regular_file(
                Path(str(record.get("path") or "")),
                f"B200 preflight {role} {nested}",
            )
            _expect(
                f"B200 preflight {role} {nested} SHA",
                record.get("sha256"),
                b200_runtime.sha256_file(artifact_path),
            )
    data = _object(receipt.get("data"), "B200 preflight data")
    _expect("B200 preflight data status", data.get("status"), "PASS")
    configured_files = identities["bundle"]["data_contract"]["files"]
    data_files = _object(data.get("files"), "B200 preflight data files")
    for role, configured in configured_files.items():
        observed = _object(data_files.get(role), f"B200 preflight data {role}")
        for key, expected in (
            ("path", configured["path"]),
            ("sha256", configured["sha256"]),
            ("rows", configured["rows"]),
        ):
            _expect(f"B200 preflight data {role} {key}", observed.get(key), expected)
    return receipt, identity


def _validate_b200_gradient_checkpointing(value: Any, label: str) -> dict[str, Any]:
    contract = _object(value, label)
    expected_devices = {
        str(layer): ("cuda:0" if layer < 24 else "cuda:1") for layer in range(48)
    }
    for key, expected in (
        ("status", "PASS"),
        ("implementation", "device_aware_non_reentrant"),
        ("use_reentrant", False),
        ("preserve_rng_state", True),
        ("determinism_check", "default"),
        ("early_stop", True),
        ("decoder_layer_count", 48),
        ("boundary_layers", [24]),
        ("layer_input_devices", expected_devices),
    ):
        _expect(f"{label} {key}", contract.get(key), expected)
    return contract


def _validate_b200_training_activation_policy(
    value: Any,
    label: str,
    *,
    recipe: dict[str, Any],
) -> dict[str, Any]:
    performance = _object(
        recipe.get("training_performance"), "B200 training performance recipe"
    )
    _expect(
        "B200 activation policy recipe",
        performance,
        b200_runtime.b200_training_performance_contract(),
    )
    contract = _object(value, label)
    _expect(
        label,
        contract,
        b200_runtime.b200_selective_gradient_checkpointing_contract(),
    )
    return contract


def _validate_b200_inherited_hash_audit(
    value: Any,
    *,
    label: str,
    role: str,
    asset_identity: dict[str, str],
) -> dict[str, Any]:
    audit = _object(value, label)
    if role not in {"training", "inference"}:
        raise RuntimeError(f"{label} has unsupported asset role: {role!r}")
    _boolean(audit.get("hash_verification"), f"{label} hash verification", True)
    _expect(
        f"{label} hash verification mode",
        audit.get("hash_verification_mode"),
        "inherited_large_asset_receipt",
    )
    _boolean(
        audit.get("direct_large_shard_hashes"),
        f"{label} direct large shard hashes",
        False,
    )
    _boolean(
        audit.get("payload_rehash_performed"),
        f"{label} payload rehash performed",
        False,
    )
    _boolean(
        audit.get("production_eligible"),
        f"{label} production eligibility",
        True,
    )
    evidence = _object(
        audit.get("asset_integrity_evidence"),
        f"{label} asset integrity evidence",
    )
    _expect(f"{label} asset integrity status", evidence.get("status"), "PASS")
    _expect(f"{label} asset integrity role", evidence.get("role"), role)
    receipt = _object(
        evidence.get("asset_integrity_receipt"),
        f"{label} asset integrity receipt",
    )
    for key in ("path", "sha256"):
        _expect(
            f"{label} asset integrity receipt {key}",
            receipt.get(key),
            asset_identity[key],
        )
    return audit


def _validate_b200_loader_audit(
    value: Any,
    label: str,
    *,
    role: str,
    asset_identity: dict[str, str],
) -> dict[str, Any]:
    loader = _object(value, label)
    _expect(f"{label} status", loader.get("status"), "PASS")
    _validate_b200_inherited_hash_audit(
        loader,
        label=label,
        role=role,
        asset_identity=asset_identity,
    )
    _validate_b200_gpu_evidence(loader.get("hardware"), f"{label} hardware")
    model = _object(loader.get("model_audit"), f"{label} model audit")
    for key, expected in (
        ("status", "PASS"),
        ("routed_nf4_layers", 48),
        ("lora_target_bf16_linear_count", EXPECTED_LORA_MODULES),
        ("meta_parameter_count", 0),
        ("meta_buffer_count", 0),
        ("base_trainable_parameter_count", 0),
        ("model_parallel_hook_count", 51),
    ):
        _expect(f"{label} model audit {key}", model.get(key), expected)
    _validate_b200_hf_device_map(model.get("device_map"), f"{label} device map")
    placement = _object(model.get("placement_audit"), f"{label} placement audit")
    for key in (
        "cpu_tensor_count",
        "meta_tensor_count",
        "disk_offload_count",
        "forbidden_cuda2_tensor_count",
    ):
        _expect(f"{label} placement {key}", placement.get(key), 0)
    checkpoint_contract = loader.get("gradient_checkpointing_contract")
    if checkpoint_contract is None:
        for key, expected in (
            ("routed_nf4_layout", "per_expert"),
            ("routed_experts_per_layer", 320),
            ("non_cuda_parameter_count", 0),
            ("non_cuda_buffer_count", 0),
            ("gradient_checkpointing", False),
            ("use_cache", True),
        ):
            _expect(f"{label} model audit {key}", model.get(key), expected)
    else:
        for key, expected in (
            ("gradient_checkpointing", True),
            ("device_aware_checkpoint_layer_count", 48),
        ):
            _expect(f"{label} model audit {key}", model.get(key), expected)
        _validate_b200_gradient_checkpointing(
            checkpoint_contract,
            f"{label} gradient checkpointing",
        )
    return loader


def _validate_b200_finite_gradient(value: Any, label: str) -> dict[str, Any]:
    audit = _object(value, label)
    for key in ("parameters_with_gradient", "nonzero_gradient_elements"):
        _integer(audit.get(key), f"{label} {key}", minimum=1)
    _positive_finite(audit.get("l2"), f"{label} l2")
    _boolean(audit.get("finite"), f"{label} finite", True)
    return audit


def _validate_b200_runtime_budget(
    value: Any,
    *,
    runtime_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    budget = _object(value, "B200 runtime budget")
    _expect("B200 runtime budget schema", budget.get("schema_version"), 5)
    measurement = _object(budget.get("measurement"), "B200 runtime measurement")
    for collection_name in (
        "training_bucket_benchmarks",
        "validation_bucket_benchmarks",
    ):
        collection = _object(
            measurement.get(collection_name),
            f"B200 {collection_name}",
        )
        if set(collection) != {"1024", "2048", "4096", "9600"}:
            raise RuntimeError(f"B200 {collection_name} buckets changed")
        for key, item_value in collection.items():
            item = _object(item_value, f"B200 {collection_name} {key}")
            for field, expected in (
                ("warmup_repetitions", 1),
                ("timed_repetitions", 3),
                ("estimator", "median_p50"),
                ("benchmark_state_unchanged", True),
            ):
                _expect(
                    f"B200 {collection_name} {key} {field}", item.get(field), expected
                )
            samples = item.get("seconds_samples")
            if not isinstance(samples, list) or len(samples) != 3:
                raise RuntimeError(f"B200 {collection_name} {key} lacks three samples")
            expected_median = float(
                statistics.median(
                    [
                        _positive_finite(sample, f"B200 {collection_name} {key} sample")
                        for sample in samples
                    ]
                )
            )
            _expect(
                f"B200 {collection_name} {key} median",
                float(item.get("seconds_p50")),
                expected_median,
            )
            _expect(
                f"B200 {collection_name} {key} selected seconds",
                float(item.get("seconds")),
                expected_median,
            )
    update = _object(
        measurement.get("production_update_benchmark"),
        "B200 production update benchmark",
    )
    for key, expected in (
        ("warmup_repetitions", 2),
        ("timed_repetitions", 10),
        ("estimator", "median_p50"),
        ("benchmark_state_unchanged", True),
    ):
        _expect(f"B200 production update {key}", update.get(key), expected)
    samples = update.get("seconds_samples")
    if not isinstance(samples, list) or len(samples) != 10:
        raise RuntimeError("B200 production update requires ten timed samples")
    ordered = sorted(
        _positive_finite(sample, "B200 production update sample") for sample in samples
    )
    _expect(
        "B200 production update p50",
        float(update.get("seconds_p50")),
        float(statistics.median(ordered)),
    )
    _expect("B200 production update p90", float(update.get("seconds_p90")), ordered[8])
    workload = _object(budget.get("production_workload"), "B200 production workload")
    for key, expected in (
        ("optimizer_steps", 9_316),
        ("rows_per_optimizer_step", 16),
        ("training_occurrences", 149_048),
        ("full_validation_steps", [4_658, 9_316]),
        ("full_validation_evaluations", 2),
        ("optimizer_boundaries", 9_316),
    ):
        _expect(f"B200 production workload {key}", workload.get(key), expected)
    projection = _object(budget.get("projection"), "B200 runtime projection")
    _expect("B200 runtime safety factor", float(projection.get("safety_factor")), 1.25)
    if (
        _positive_finite(
            projection.get("raw_validation_seconds"), "B200 validation overhead"
        )
        <= 0
    ):
        raise RuntimeError("B200 validation overhead is absent")
    if (
        _positive_finite(
            projection.get("raw_optimizer_boundary_seconds"),
            "B200 optimizer boundary overhead",
        )
        <= 0
    ):
        raise RuntimeError("B200 AdamW boundary overhead is absent")
    empirical = _object(
        budget.get("empirical_update_projection"),
        "B200 empirical update projection",
    )
    _expect(
        "B200 empirical update estimator",
        empirical.get("estimator"),
        "ten_update_sample_p50_and_p90",
    )
    raw_p90 = _positive_finite(
        empirical.get("raw_p90_total_seconds"), "B200 raw p90 runtime"
    )
    projected_p90 = _positive_finite(
        empirical.get("projected_p90_total_seconds"), "B200 safety-adjusted p90 runtime"
    )
    if not math.isclose(projected_p90, raw_p90 * 1.25, rel_tol=1e-12, abs_tol=1e-6):
        raise RuntimeError("B200 empirical p90 does not apply the 25% safety factor")
    projected_hours = projected_p90 / 3_600.0
    if not math.isclose(
        float(empirical.get("projected_p90_total_hours")),
        projected_hours,
        rel_tol=1e-12,
        abs_tol=1e-9,
    ):
        raise RuntimeError("B200 empirical p90 hour projection changed")
    feasibility = _object(budget.get("feasibility"), "B200 runtime feasibility")
    _expect("B200 runtime feasibility status", feasibility.get("status"), "PASS")
    limitations = _object(budget.get("limitations"), "B200 runtime limitations")
    unmeasured = limitations.get("explicitly_unmeasured_overheads")
    if (
        not isinstance(unmeasured, list)
        or "recovery and candidate checkpoint I/O" not in unmeasured
    ):
        raise RuntimeError("B200 runtime projection lacks checkpoint I/O accounting")
    hard_limit_enforced = True
    if runtime_gate is not None:
        hard_limit_enforced = runtime_gate.get("hard_limit_enforced") is not False
        for key, expected in (
            ("projection_required", True),
            ("safety_factor", 1.25),
            ("recommended_hours", 144.0),
            ("former_absolute_limit_hours", 168.0),
            ("override_scope", "runtime_projection_only_exact_one_epoch"),
            ("non_runtime_gates_overridden", False),
        ):
            _expect(f"B200 runtime gate {key}", runtime_gate.get(key), expected)
    if projected_hours <= B200_RECOMMENDED_HOURS:
        outcome = "PASS_RECOMMENDED"
    elif projected_hours <= B200_ABSOLUTE_LIMIT_HOURS:
        outcome = "RISK_REQUIRES_SEPARATE_APPROVAL"
    elif not hard_limit_enforced:
        outcome = "PASS_EXPLICIT_RUNTIME_LIMIT_OVERRIDE"
    else:
        raise RuntimeError(
            "B200 safety-adjusted p90 runtime exceeds 168 hours; "
            "production is forbidden"
        )
    return {
        "outcome": outcome,
        "projected_p90_total_hours": projected_hours,
        "recommended_limit_hours": B200_RECOMMENDED_HOURS,
        "absolute_limit_hours": B200_ABSOLUTE_LIMIT_HOURS,
        "safety_factor": 1.25,
        "validation_overhead_seconds": float(projection["raw_validation_seconds"]),
        "first_adamw_boundary_seconds": float(
            measurement["optimizer_boundary_seconds"]
        ),
        "checkpoint_overhead_allowance_seconds": projected_p90 - raw_p90,
        "checkpoint_overhead_accounting": "covered_by_25_percent_aggregate_safety_allowance",
        "technical_gate_status": "PASS",
        "hard_limit_enforced": hard_limit_enforced,
        "runtime_override_scope": (
            runtime_gate.get("override_scope") if runtime_gate is not None else None
        ),
        "user_instruction_sha256": (
            runtime_gate.get("user_instruction_sha256")
            if runtime_gate is not None
            else None
        ),
        "non_runtime_gates_overridden": False,
    }


def _validate_b200_training_manifest(
    path: Path,
    *,
    identities: dict[str, Any],
    preflight: dict[str, Any],
    asset_identity: dict[str, str],
) -> tuple[dict[str, Any], dict[str, str], Path, dict[str, Any], dict[str, Any]]:
    resolved, training, identity = _b200_receipt(path, "B200 training manifest")
    recipe = identities["bundle"]["recipe"]
    for key, expected in (
        ("status", "SMOKE_COMPLETED"),
        ("profile", "b200"),
        ("config_identity", identities["config"]),
        ("config", recipe),
        ("completed_updates", 1),
        ("target_updates", 1),
    ):
        _expect(f"B200 training manifest {key}", training.get(key), expected)
    occurrence = _object(
        training.get("occurrence_contract"), "B200 occurrence contract"
    )
    _expect(
        "B200 occurrence contract",
        occurrence,
        {
            "nominal_step_occurrences": 16,
            "total_occurrences": 149_048,
            "optimizer_steps": 9_316,
            "final_step_occurrences": 8,
        },
    )
    state = _object(training.get("state_contract"), "B200 state contract")
    for key, expected in (
        ("format_version", 6),
        ("profile", "b200"),
        ("config_identity", identities["config"]),
        ("source_dir", str(b200_runtime.BF16_SOURCE_DIR)),
        ("expert_nf4_dir", str(b200_runtime.TRAIN_EXPERT_DIR)),
        ("model_contract_sha256", identities["bundle"]["hashes"]["model"]),
        ("recipe_sha256", identities["bundle"]["hashes"]["recipe"]),
        ("data_contract_sha256", identities["bundle"]["hashes"]["data_contract"]),
        ("pipeline_code_sha256", identities["pipeline_code"]["sha256"]),
        ("environment_lock_sha256", identities["environment_lock_sha256"]),
        ("smoke", True),
        (
            "training_performance_sha256",
            canonical_sha256(recipe["training_performance"]),
        ),
        (
            "training_activation_contract_sha256",
            canonical_sha256(
                b200_runtime.b200_selective_gradient_checkpointing_contract()
            ),
        ),
    ):
        _expect(f"B200 state contract {key}", state.get(key), expected)
    _expect(
        "B200 state download receipt SHA",
        state.get("download_receipt_sha256"),
        preflight["bf16"]["receipt_sha256"],
    )
    _expect(
        "B200 state expert manifest SHA",
        state.get("expert_manifest_sha256"),
        preflight["training_nf4"]["manifest"]["sha256"],
    )
    gpu_runtime = _object(state.get("gpu_runtime"), "B200 training GPU runtime")
    for key, expected in (
        ("status", "PASS"),
        ("physical_cuda_visible_devices", "6,7"),
        ("logical_gpu_count", 2),
        ("boundary_layers", [24]),
    ):
        _expect(f"B200 training GPU runtime {key}", gpu_runtime.get(key), expected)
    training_loader = _validate_b200_loader_audit(
        training.get("loader_audit"),
        "B200 training loader",
        role="training",
        asset_identity=asset_identity,
    )
    _expect(
        "B200 training loader config",
        training_loader.get("b200_config"),
        {
            "config_dir": str(identities["bundle"]["config_dir"]),
            "hashes": dict(identities["bundle"]["hashes"]),
        },
    )
    training_source = _object(
        training_loader.get("source_inventory"),
        "B200 training loader source inventory",
    )
    for key, expected in (
        ("source_dir", str(b200_runtime.BF16_SOURCE_DIR)),
        (
            "index_sha256",
            preflight["bf16"]["contract_hashes"]["model.safetensors.index.json"],
        ),
        ("config_sha256", preflight["bf16"]["contract_hashes"]["config.json"]),
    ):
        _expect(
            f"B200 training loader source {key}", training_source.get(key), expected
        )
    training_expert = _object(
        training_loader.get("expert_preflight"),
        "B200 training loader expert preflight",
    )
    for key, expected in (
        ("status", "PASS"),
        ("production_eligible", True),
        ("checkpoint_dir", str(b200_runtime.TRAIN_EXPERT_DIR)),
        ("manifest_sha256", preflight["training_nf4"]["manifest"]["sha256"]),
        ("expert_index_sha256", preflight["training_nf4"]["index"]["sha256"]),
    ):
        _expect(
            f"B200 training loader expert {key}", training_expert.get(key), expected
        )
    device = _object(training.get("device_audit"), "B200 training device audit")
    _expect("B200 training CPU/disk offload", device.get("cpu_disk_offload"), False)
    if set(device.get("hf_device_map_devices") or []) != {"cuda:0", "cuda:1"}:
        raise RuntimeError("B200 training device audit is not exactly cuda:0/cuda:1")
    if any(
        not str(key).startswith("cuda:") or key == "cuda:2"
        for key in (device.get("parameter_devices") or {})
    ):
        raise RuntimeError("B200 training device audit contains offload/cuda:2")
    lora = _object(training.get("lora_module_audit"), "B200 LoRA module audit")
    _expect("B200 LoRA status", lora.get("status"), "PASS")
    _expect("B200 LoRA module count", lora.get("module_count"), EXPECTED_LORA_MODULES)
    _expect(
        "B200 trainable parameters",
        training.get("trainable_parameters"),
        EXPECTED_TRAINABLE_PARAMETERS,
    )
    _validate_b200_training_activation_policy(
        training.get("gradient_checkpointing_contract"),
        "B200 training activation policy",
        recipe=recipe,
    )
    _expect(
        "B200 training performance manifest",
        training.get("training_performance"),
        recipe["training_performance"],
    )
    _expect(
        "B200 training performance SHA",
        training.get("training_performance_sha256"),
        canonical_sha256(recipe["training_performance"]),
    )
    _expect(
        "B200 effective checkpoint contract SHA",
        training.get("gradient_checkpointing_contract_sha256"),
        canonical_sha256(b200_runtime.b200_selective_gradient_checkpointing_contract()),
    )
    split_roles = _object(training.get("split_roles"), "B200 training split roles")
    _expect(
        "B200 full test split role",
        split_roles.get("full_test"),
        "sealed final evaluation only",
    )
    _expect(
        "B200 human gold split role",
        split_roles.get("human_gold"),
        "sealed final evaluation only",
    )
    _expect(
        "B200 quality early stopping",
        recipe["validation"]["early_stopping"]["enabled"],
        False,
    )
    _expect(
        "B200 monitor datasets",
        recipe["validation"]["monitor_datasets"],
        ["validation"],
    )

    smoke = _object(training.get("smoke"), "B200 training smoke")
    longest = _object(
        smoke.get("longest_occurrence_update"), "B200 longest occurrence update"
    )
    for key, expected in (
        ("status", "PASS"),
        ("optimizer_step", 1),
        ("expected_tokens", 9_560),
        ("observed_tokens", 9_560),
        ("occurrences_in_optimizer_group", 16),
        ("longest_row_occurrence_count", 1),
        ("forward_backward_completed", True),
        ("loss_denominator", 16),
        ("gradient_clip_invoked", True),
        ("frozen_base_probe_unchanged", True),
    ):
        _expect(f"B200 longest occurrence {key}", longest.get(key), expected)
    _validate_b200_finite_gradient(
        longest.get("finite_nonzero_gradient"),
        "B200 longest occurrence gradient",
    )
    adamw = _object(longest.get("adamw_update"), "B200 longest occurrence AdamW")
    _expect("B200 longest AdamW changed", adamw.get("changed"), True)
    _expect("B200 longest AdamW step", adamw.get("optimizer_step"), 1)
    _positive_finite(
        adamw.get("learning_rate_used"), "B200 longest AdamW learning rate"
    )
    recompute = _object(
        longest.get("post_update_longest_row_recompute"),
        "B200 post-update longest-row recompute",
    )
    for key, expected in (
        ("status", "PASS"),
        ("observed_tokens", 9_560),
        ("adapter_parameters_unchanged", True),
        ("frozen_parameters_unchanged", True),
    ):
        _expect(f"B200 post-update recompute {key}", recompute.get(key), expected)
    _validate_b200_finite_gradient(
        recompute.get("gradient_audit"),
        "B200 post-update recompute gradient",
    )

    tail = _object(smoke.get("exact_tail_transaction"), "B200 exact-tail transaction")
    _expect("B200 exact-tail mirror", training.get("exact_tail_smoke_audit"), tail)
    for key, expected in (
        ("status", "PASS"),
        ("transactional", True),
        ("optimizer_step", 9_316),
        ("global_positions", list(range(149_040, 149_048))),
        ("canonical_tail_sha256", B200_TAIL_SHA256),
        ("occurrence_identity_duplicate_count", 0),
        ("row_identity_duplicate_count", 0),
        ("nominal_step_occurrences", 16),
        ("actual_step_occurrences", 8),
        ("loss_denominator", 8),
        ("gradient_denominator", 8),
        ("padding_occurrences", 0),
        ("dropped_occurrences", 0),
        ("repeated_row_identities", 0),
        ("frozen_base_probe_unchanged", True),
        ("final_main_smoke_state_restored", True),
    ):
        _expect(f"B200 exact-tail {key}", tail.get(key), expected)
    _expect(
        "B200 exact-tail sampler slice",
        tail.get("sampler_slice"),
        {
            "api": "GlobalAugmentedDeckSampler.occurrences",
            "start": 149_040,
            "count": 8,
            "end_exclusive": 149_048,
        },
    )
    occurrences = tail.get("occurrences")
    if not isinstance(occurrences, list) or len(occurrences) != 8:
        raise RuntimeError("B200 exact-tail must contain eight occurrence records")
    if [
        record.get("global_position")
        for record in occurrences
        if isinstance(record, dict)
    ] != list(range(149_040, 149_048)):
        raise RuntimeError("B200 exact-tail occurrence positions changed")
    plan = _object(tail.get("microbatch_plan"), "B200 exact-tail microbatch plan")
    _expect("B200 exact-tail plan updates", plan.get("optimizer_updates"), 1)
    _expect("B200 exact-tail plan occurrences", plan.get("occurrences"), 8)
    if not isinstance(tail.get("row_losses"), list) or len(tail["row_losses"]) != 8:
        raise RuntimeError("B200 exact-tail must record eight row losses")
    _positive_finite(tail.get("learning_rate_used"), "B200 exact-tail learning rate")
    _validate_b200_finite_gradient(
        tail.get("gradient_before_clip"), "B200 exact-tail pre-clip gradient"
    )
    _validate_b200_finite_gradient(
        tail.get("gradient_after_clip"), "B200 exact-tail post-clip gradient"
    )
    clip = _object(tail.get("gradient_clip"), "B200 exact-tail gradient clip")
    for key, expected in (
        ("invoked", True),
        ("max_norm", 1.0),
        ("finite_nonzero_after_clip", True),
    ):
        _expect(f"B200 exact-tail gradient clip {key}", clip.get(key), expected)
    tail_adamw = _object(tail.get("adamw_state"), "B200 exact-tail AdamW")
    _expect("B200 exact-tail AdamW entries before", tail_adamw.get("entries_before"), 0)
    for key in ("entries_after", "tensor_count_after"):
        _integer(tail_adamw.get(key), f"B200 exact-tail AdamW {key}", minimum=1)
    _expect("B200 exact-tail AdamW allocated", tail_adamw.get("allocated"), True)
    _expect("B200 exact-tail AdamW real update", tail_adamw.get("real_update"), True)
    tail_adapter = _object(tail.get("adapter_update"), "B200 exact-tail adapter update")
    _expect("B200 exact-tail adapter changed", tail_adapter.get("changed"), True)
    if tail_adapter.get("before_sha256") == tail_adapter.get("after_sha256"):
        raise RuntimeError("B200 exact-tail adapter digest did not change")
    payload = _object(
        tail.get("trainable_parameter_payload_save_reload"),
        "B200 exact-tail payload reload",
    )
    for key, expected in (
        ("status", "PASS"),
        ("medium", "cpu_memory"),
        ("distinct_from_final_atomic_disk_adapter_reload", True),
    ):
        _expect(f"B200 exact-tail payload {key}", payload.get(key), expected)
    _expect(
        "B200 exact-tail payload digest",
        payload.get("reloaded_sha256"),
        payload.get("saved_sha256"),
    )

    disk = _object(
        smoke.get("final_adapter_atomic_disk_save"), "B200 atomic disk adapter save"
    )
    for key, expected in (
        ("status", "PASS"),
        ("reload_is_verified_by", "downstream_tuned_inference"),
        ("distinct_from_tail_parameter_payload", True),
    ):
        _expect(f"B200 disk adapter {key}", disk.get(key), expected)
    adapter_dir = b200_runtime.require_owned_path(
        Path(str(disk.get("adapter_dir") or "")),
        must_exist=True,
    )
    success = _b200_owned_regular_file(
        Path(str(disk.get("success_marker") or "")),
        "B200 final adapter success marker",
        expected_path=adapter_dir / "_SUCCESS",
    )
    _expect(
        "B200 final adapter success text", success.read_text(encoding="utf-8"), "PASS\n"
    )
    current_files = _current_file_map(adapter_dir)
    _expect("B200 final adapter files", disk.get("files_sha256"), current_files)
    _expect(
        "B200 smoke adapter files", smoke.get("adapter_files_sha256"), current_files
    )
    adapter = validate_adapter(
        adapter_dir, config_dir=identities["bundle"]["config_dir"]
    )
    runtime = _validate_b200_runtime_budget(
        smoke.get("runtime_budget_projection"),
        runtime_gate=recipe["runtime_gate"],
    )
    _expect(
        "B200 runtime budget mirror",
        training.get("runtime_budget_projection"),
        smoke.get("runtime_budget_projection"),
    )
    return training, identity, adapter_dir, adapter, runtime


def _validate_b200_inference_implementation(
    value: Any,
    *,
    identities: dict[str, Any],
    label: str,
) -> None:
    implementation = _object(value, label)
    expected_files = identities["config"]["files"]
    for key, expected in (
        ("infer_py_sha256", sha256(Path(__file__).with_name("infer.py"))),
        (
            "inference_loader_sha256",
            sha256(Path(__file__).with_name("solar_inference_loader.py")),
        ),
        ("reference_evaluator_sha256", b200_runtime.REFERENCE_EVALUATOR_SHA256),
        ("config_files", expected_files),
        ("model_contract_sha256", expected_files["model"]["sha256"]),
        ("recipe_sha256", expected_files["recipe"]["sha256"]),
        ("data_contract_sha256", expected_files["data_contract"]["sha256"]),
        ("environment_lock_sha256", identities["environment_lock_sha256"]),
        ("pipeline_code_sha256", identities["pipeline_code"]["sha256"]),
    ):
        _expect(f"{label} {key}", implementation.get(key), expected)
    for key in ("transformers_version", "peft_version", "bitsandbytes_version"):
        if not str(implementation.get(key) or "").strip():
            raise RuntimeError(f"{label} lacks {key}")


def _validate_b200_prediction_file(
    path: Path,
    *,
    role: str,
    source: dict[str, Any],
    inference_expert: dict[str, Any],
    adapter: dict[str, Any] | None,
    identities: dict[str, Any],
    asset_identity: dict[str, str],
) -> dict[str, Any]:
    path = _b200_owned_regular_file(path, f"B200 {role} predictions")
    manifest_path = _b200_owned_regular_file(
        path.with_name(path.name + ".manifest.json"),
        f"B200 {role} prediction manifest",
    )
    manifest = b200_runtime.read_json_no_duplicates(manifest_path)
    for key, expected in (
        ("schema", PROVENANCE_SCHEMA),
        ("schema_version", PROVENANCE_SCHEMA_VERSION),
        ("status", "COMPLETE"),
        ("expected_rows", 2),
        ("completed_rows", 2),
        ("output_jsonl_sha256", b200_runtime.sha256_file(path)),
    ):
        _expect(f"B200 {role} manifest {key}", manifest.get(key), expected)
    contract = _object(manifest.get("contract"), f"B200 {role} contract")
    contract_sha256 = canonical_sha256(contract)
    _expect(
        f"B200 {role} contract SHA", manifest.get("contract_sha256"), contract_sha256
    )
    for key, expected in (
        ("schema", PROVENANCE_SCHEMA),
        ("schema_version", PROVENANCE_SCHEMA_VERSION),
        ("profile", "b200"),
        ("inference_mode", B200_VALIDATION_SMOKE_MODE),
        ("model_variant", role),
        ("source", source),
        ("per_expert_nf4", inference_expert),
        ("adapter", adapter),
        ("b200_config", identities["config"]),
        ("terminal_training", None),
    ):
        _expect(f"B200 {role} contract {key}", contract.get(key), expected)
    evaluation = _object(contract.get("evaluation"), f"B200 {role} evaluation")
    validation = identities["bundle"]["data_contract"]["files"]["validation"]
    for key, expected in (
        ("role", "validation"),
        ("path", validation["path"]),
        ("sha256", validation["sha256"]),
        ("rows", validation["rows"]),
        ("by_task", validation["by_task"]),
        ("mode", B200_VALIDATION_SMOKE_MODE),
        ("selection_policy", "first_file_order_row_per_task"),
        ("selected_rows", 2),
        ("selected_by_task", {"translation": 1, "summarization": 1}),
        ("sealed_evaluation_accessed", False),
        ("limit_per_task", 0),
    ):
        _expect(f"B200 {role} evaluation {key}", evaluation.get(key), expected)
    _expect(
        f"B200 {role} prompt",
        contract.get("prompt"),
        {"variant": "no_dictionary_context", **TEMPLATE_KWARGS},
    )
    decoding = _object(contract.get("decoding"), f"B200 {role} decoding")
    expected_decoding = {
        "temperature": 0.0,
        "top_p": 1.0,
        "do_sample": False,
        "num_beams": 1,
        "max_model_len": B200_MAX_MODEL_LEN,
        "translation_max_new_tokens": B200_TASK_MAX_NEW_TOKENS["translation"],
        "summarization_max_new_tokens": B200_TASK_MAX_NEW_TOKENS["summarization"],
        "padding_side": "left",
        "raw_output_policy": "outer strip only; no filtering or repair",
    }
    _expect(f"B200 {role} decoding", decoding, expected_decoding)
    _expect(
        f"B200 {role} decoding SHA",
        contract.get("decoding_sha256"),
        canonical_sha256(decoding),
    )
    runtime = _object(contract.get("runtime"), f"B200 {role} runtime")
    for key, expected in (
        ("strategy", "single_process_two_gpu_model_parallel"),
        ("logical_gpu_count", 2),
        (
            "layer_device_map",
            {
                str(key): value
                for key, value in b200_runtime.EXPECTED_LAYER_DEVICE_MAP.items()
            },
        ),
        ("hf_device_map", dict(b200_runtime.EXPECTED_HF_DEVICE_MAP)),
        ("routed_expert_format", "per_expert_nf4"),
    ):
        _expect(f"B200 {role} runtime {key}", runtime.get(key), expected)
    if (
        _integer(
            runtime.get("generation_batch_size"),
            f"B200 {role} generation batch size",
            minimum=1,
        )
        < 1
    ):
        raise RuntimeError(f"B200 {role} generation batch size is invalid")
    b200_identity = _object(
        runtime.get("b200_identity"), f"B200 {role} runtime identity"
    )
    _expect(
        f"B200 {role} runtime user",
        b200_identity.get("user"),
        b200_runtime.EXPECTED_USER,
    )
    _expect(
        f"B200 {role} runtime Python",
        b200_identity.get("python_environment"),
        str(b200_runtime.VENV_DIR),
    )
    _validate_b200_gpu_evidence(b200_identity.get("gpu"), f"B200 {role} runtime GPU")
    _validate_b200_inference_implementation(
        contract.get("implementation"),
        identities=identities,
        label=f"B200 {role} implementation",
    )

    rows = read_prediction_rows(path)
    if len(rows) != 2:
        raise RuntimeError(f"B200 {role} must contain exactly two predictions")
    keys = [(str(row.get("id") or ""), str(row.get("task") or "")) for row in rows]
    if [task for _row_id, task in keys] != ["translation", "summarization"]:
        raise RuntimeError(f"B200 {role} task order/coverage changed")
    _expect(
        f"B200 {role} selected keys SHA",
        evaluation.get("selected_keys_sha256"),
        canonical_sha256([list(key) for key in keys]),
    )
    invariants = expected_row_provenance(contract, contract_sha256)
    for line_number, row in enumerate(rows, 1):
        for field, expected in invariants.items():
            _expect(
                f"B200 {role} row {line_number} provenance {field}",
                row.get(field),
                expected,
            )
        _expect(
            f"B200 {role} row {line_number} prompt variant",
            row.get("prompt_variant"),
            "no_dictionary_context",
        )
        _expect(
            f"B200 {role} row {line_number} reasoning",
            row.get("reasoning_effort"),
            "none",
        )
        task = str(row["task"])
        violations = generation_integrity_violations(
            row,
            task=task,
            max_new_tokens=B200_TASK_MAX_NEW_TOKENS[task],
        )
        if violations:
            raise RuntimeError(
                f"B200 {role} row {line_number} generation violations: {violations}"
            )
        _expect(
            f"B200 {role} row {line_number} output audit",
            row.get("output_violation_audit"),
            {
                "status": "PASS",
                "violations": [],
                "postprocessing": "outer_strip_only",
            },
        )
    audit = _object(manifest.get("runtime_audit"), f"B200 {role} runtime audit")
    _validate_b200_inherited_hash_audit(
        audit,
        label=f"B200 {role} runtime audit",
        role="inference",
        asset_identity=asset_identity,
    )
    loader = _validate_b200_loader_audit(
        audit.get("loader_receipt"),
        f"B200 {role} inference loader",
        role="inference",
        asset_identity=asset_identity,
    )
    for key in (
        "hash_verification",
        "hash_verification_mode",
        "direct_large_shard_hashes",
        "payload_rehash_performed",
        "production_eligible",
        "asset_integrity_evidence",
    ):
        _expect(
            f"B200 {role} runtime/loader {key} mirror",
            audit.get(key),
            loader.get(key),
        )
    _expect(
        f"B200 {role} loader config",
        loader.get("b200_config"),
        contract["b200_config"],
    )
    _expect(
        f"B200 {role} model audit mirror",
        audit.get("model_audit"),
        loader["model_audit"],
    )
    source_inventory = _object(
        loader.get("source_inventory"), f"B200 {role} loader source inventory"
    )
    _expect(
        f"B200 {role} source inventory mirror",
        audit.get("source_inventory"),
        source_inventory,
    )
    for key, expected in (
        ("source_dir", source["path"]),
        ("index_sha256", source["index_sha256"]),
        ("config_sha256", source["config_sha256"]),
    ):
        _expect(f"B200 {role} loaded source {key}", source_inventory.get(key), expected)
    expert_preflight = _object(
        loader.get("expert_preflight"), f"B200 {role} loader expert preflight"
    )
    _expect(
        f"B200 {role} expert preflight mirror",
        audit.get("expert_preflight"),
        expert_preflight,
    )
    for key, expected in (
        ("status", "PASS"),
        ("production_eligible", True),
        ("checkpoint_dir", inference_expert["path"]),
        ("manifest_sha256", inference_expert["manifest_sha256"]),
        ("expert_index_sha256", inference_expert["expert_index_sha256"]),
    ):
        _expect(f"B200 {role} loaded expert {key}", expert_preflight.get(key), expected)
    adapter_audit = audit.get("adapter_audit")
    if adapter is None:
        _expect(f"B200 {role} adapter audit", adapter_audit, None)
    else:
        adapter_audit = _object(adapter_audit, f"B200 {role} adapter audit")
        for key, expected in (
            ("status", "PASS"),
            ("module_count", EXPECTED_LORA_MODULES),
            ("adapter_tensor_count", EXPECTED_LORA_TENSORS),
            ("adapter_tensor_names_sha256", adapter["serialized_keys_sha256"]),
            ("adapter_parameter_count", EXPECTED_TRAINABLE_PARAMETERS),
            ("adapter_dtypes", {"torch.float32": EXPECTED_LORA_TENSORS}),
            ("trainable_parameter_count", 0),
            ("offloaded_adapter_tensor_count", 0),
        ):
            _expect(
                f"B200 {role} adapter audit {key}", adapter_audit.get(key), expected
            )
    return {
        "path": str(path),
        "sha256": b200_runtime.sha256_file(path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": b200_runtime.sha256_file(manifest_path),
        "contract_sha256": contract_sha256,
        "model_label": contract.get("model_label"),
        "keys": keys,
        "evaluation": evaluation,
        "decoding": decoding,
        "generation_count": len(rows),
    }


def _require_fresh_b200_smoke_output(path: Path | None) -> Path:
    if path is None:
        raise RuntimeError("B200 certification requires an explicit --output")
    raw = path.expanduser()
    if raw.is_symlink():
        raise RuntimeError(f"B200 smoke receipt must not be a symlink: {raw}")
    resolved = b200_runtime.require_owned_path(raw)
    if resolved != B200_SMOKE_RECEIPT.resolve():
        raise RuntimeError(
            f"B200 smoke receipt path mismatch: {resolved} != "
            f"{B200_SMOKE_RECEIPT.resolve()}"
        )
    if resolved.exists() or resolved.is_symlink():
        raise FileExistsError(f"refusing to overwrite B200 smoke receipt: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _certify_b200(
    *,
    config_dir: Path,
    source_dir: Path,
    training_expert_nf4_dir: Path,
    inference_expert_nf4_dir: Path,
    kernel_probe_receipt: Path,
    environment_receipt: Path,
    asset_integrity_receipt: Path,
    tokenizer_contract_receipt: Path,
    preflight_receipt: Path,
    training_manifest: Path,
    base_predictions: Path,
    tuned_predictions: Path,
    output: Path | None,
) -> dict[str, Any]:
    destination = _require_fresh_b200_smoke_output(output)
    b200_runtime.require_expected_user()
    b200_runtime.require_owned_root()
    identities = _b200_current_identities(config_dir)
    for label, supplied, expected in (
        ("B200 source", source_dir, b200_runtime.BF16_SOURCE_DIR),
        ("B200 training NF4", training_expert_nf4_dir, b200_runtime.TRAIN_EXPERT_DIR),
        (
            "B200 inference NF4",
            inference_expert_nf4_dir,
            b200_runtime.INFERENCE_EXPERT_DIR,
        ),
    ):
        resolved = b200_runtime.require_owned_path(supplied, must_exist=True)
        if resolved != expected:
            raise RuntimeError(f"{label} path mismatch: {resolved} != {expected}")
    environment, environment_identity = _validate_b200_environment_receipt(
        environment_receipt
    )
    asset_integrity, asset_identity = _validate_b200_asset_integrity_receipt(
        asset_integrity_receipt
    )
    kernel_probe, kernel_identity = _validate_b200_kernel_probe_receipt(
        kernel_probe_receipt,
        identities=identities,
        environment_identity=environment_identity,
    )
    tokenizer, tokenizer_identity = _validate_b200_tokenizer_receipt(
        tokenizer_contract_receipt,
        identities=identities,
    )
    preflight, preflight_identity = _validate_b200_preflight_receipt(
        preflight_receipt,
        identities=identities,
        environment_identity=environment_identity,
    )
    source = lightweight_source_identity(b200_runtime.BF16_SOURCE_DIR)
    inference_expert = lightweight_expert_identity(
        b200_runtime.INFERENCE_EXPERT_DIR,
        source=source,
    )
    _expect(
        "B200 tokenizer/source config",
        tokenizer["source"]["config_sha256"],
        source["config_sha256"],
    )
    _expect(
        "B200 preflight/source config",
        preflight["bf16"]["contract_hashes"]["config.json"],
        source["config_sha256"],
    )
    _expect(
        "B200 preflight inference manifest",
        preflight["inference_nf4"]["manifest"]["sha256"],
        inference_expert["manifest_sha256"],
    )
    _expect(
        "B200 preflight inference index",
        preflight["inference_nf4"]["index"]["sha256"],
        inference_expert["expert_index_sha256"],
    )
    training, training_identity, adapter_dir, adapter, runtime_outcome = (
        _validate_b200_training_manifest(
            training_manifest,
            identities=identities,
            preflight=preflight,
            asset_identity=asset_identity,
        )
    )
    base = _validate_b200_prediction_file(
        base_predictions,
        role="base",
        source=source,
        inference_expert=inference_expert,
        adapter=None,
        identities=identities,
        asset_identity=asset_identity,
    )
    tuned = _validate_b200_prediction_file(
        tuned_predictions,
        role="tuned",
        source=source,
        inference_expert=inference_expert,
        adapter=adapter,
        identities=identities,
        asset_identity=asset_identity,
    )
    _expect("B200 base/tuned prediction keys", tuned["keys"], base["keys"])
    _expect("B200 base/tuned evaluation", tuned["evaluation"], base["evaluation"])
    _expect("B200 base/tuned decoding", tuned["decoding"], base["decoding"])
    if base["model_label"] == tuned["model_label"]:
        raise RuntimeError("B200 base and tuned model labels must differ")
    if base["generation_count"] + tuned["generation_count"] != 4:
        raise RuntimeError("B200 smoke did not validate exactly four generations")
    smoke = training["smoke"]
    result = {
        "schema": "solar_open2_b200_smoke_certification",
        "schema_version": 2,
        "status": "PASS",
        "technical_production_gate_passed": True,
        "authorization_scope": "technical_smoke_gate_only",
        "human_production_approval_required": True,
        "human_production_approval_recorded": False,
        "production_authorized": False,
        "profile": "b200",
        "root": str(b200_runtime.OWNED_ROOT),
        "user": b200_runtime.EXPECTED_USER,
        "config": identities["config"],
        "pipeline_code": identities["pipeline_code"],
        "config_identity": identities["config"],
        "environment_lock_sha256": identities["environment_lock_sha256"],
        "source_dir": str(b200_runtime.BF16_SOURCE_DIR),
        "expert_nf4_dir": str(b200_runtime.TRAIN_EXPERT_DIR),
        "model_contract_sha256": identities["config"]["files"]["model"]["sha256"],
        "recipe_sha256": identities["config"]["files"]["recipe"]["sha256"],
        "data_contract_sha256": identities["config"]["files"]["data_contract"][
            "sha256"
        ],
        "inputs": {
            "environment_receipt": environment_identity,
            "asset_integrity_receipt": asset_identity,
            "kernel_probe_receipt": kernel_identity,
            "tokenizer_contract_receipt": tokenizer_identity,
            "preflight_receipt": preflight_identity,
            "training_manifest": training_identity,
        },
        "tokenizer_contract_receipt": tokenizer_identity,
        "source": source,
        "training_expert": preflight["training_nf4"],
        "inference_expert": inference_expert,
        "adapter_dir": str(adapter_dir),
        "adapter": adapter,
        "base_predictions": {
            key: value
            for key, value in base.items()
            if key not in {"keys", "evaluation", "decoding"}
        },
        "tuned_predictions": {
            key: value
            for key, value in tuned.items()
            if key not in {"keys", "evaluation", "decoding"}
        },
        "runtime_outcome": runtime_outcome,
        "runtime_budget_projection": smoke["runtime_budget_projection"],
        "training_performance": training["training_performance"],
        "training_performance_sha256": training["training_performance_sha256"],
        "gradient_checkpointing_contract": training["gradient_checkpointing_contract"],
        "gradient_checkpointing_contract_sha256": training[
            "gradient_checkpointing_contract_sha256"
        ],
        "smoke_peak_allocated_bytes": smoke["peak_allocated_bytes"],
        "smoke_peak_reserved_bytes": smoke["peak_reserved_bytes"],
        "longest_occurrence_update": smoke["longest_occurrence_update"],
        "exact_tail_transaction": smoke["exact_tail_transaction"],
        "final_adapter_atomic_disk_save": smoke["final_adapter_atomic_disk_save"],
        "hardware": kernel_probe["hardware"],
        "early_stopping_policy": identities["bundle"]["recipe"]["validation"][
            "early_stopping"
        ],
        "early_stopping_history": training["history"],
        "early_stopping_decision": training["early_stopping_decision"],
        "software": {
            "environment": environment,
            "kernel_bitsandbytes_runtime": kernel_probe["bitsandbytes_runtime"],
        },
        "checks": {
            "fresh_exact_atomic_receipt": "PASS",
            "environment_identity": "PASS",
            "large_hash_asset_integrity": "PASS",
            "b200_kernel_probe": "PASS",
            "tokenizer_output_contract": "PASS",
            "relocated_model_data_nf4_preflight": "PASS",
            "two_gpu_24_24_placement": "PASS",
            "zero_cpu_meta_disk_offload": "PASS",
            "routed_nf4_layers_48": "PASS",
            "lora_528_trainable_133177344": "PASS",
            "longest_9560_real_optimizer_update": "PASS",
            "exact_tail_8_transaction": "PASS",
            "frozen_unchanged_adapter_changed": "PASS",
            "atomic_adapter_save_reload": "PASS",
            "four_eos_before_cap_unicode_outer_strip": "PASS",
            "output_only_violations_zero": "PASS",
            "test_human_input_forbidden": "PASS",
            "timing_warmup_three_median": "PASS",
            "production_like_two_warmup_ten_p50_p90": "PASS",
            "adamw_first_boundary_and_validation_overhead": "PASS",
            "runtime_25_percent_safety": "PASS",
            "model_preflight": "PASS",
            "max_length_backward": "PASS",
            "lora_module_contract": "PASS",
            "no_cpu_disk_offload": "PASS",
            "base_inference": "PASS",
            "lora_inference": "PASS",
            "current_code_and_environment": "PASS",
            "source_and_expert_provenance": "PASS",
            "finite_nonzero_lora_gradient": "PASS",
            "lora_update_and_frozen_base": "PASS",
            "device_aware_gradient_checkpointing": "PASS",
            "training_activation_policy": "PASS",
            "selective_training_activation_policy": "PASS",
            "post_update_checkpoint_recompute": "PASS",
            "output_only_supervision": "PASS",
            "sealed_prediction_contracts": "PASS",
            "runtime_budget_feasibility": "PASS",
            "early_stopping_policy": "PASS",
            "early_stopping_decision_recomputed": "PASS",
            "exact_tail_transaction": "PASS",
            "longest_occurrence_update": "PASS",
        },
    }
    b200_runtime.atomic_json(destination, result)
    return result


def certify(
    *,
    config_dir: Path = CONFIG_DIR,
    source_dir: Path,
    training_expert_nf4_dir: Path,
    inference_expert_nf4_dir: Path,
    kernel_probe_receipt: Path | None = None,
    environment_receipt: Path | None = None,
    asset_integrity_receipt: Path | None = None,
    tokenizer_contract_receipt: Path,
    preflight_receipt: Path,
    training_manifest: Path,
    base_predictions: Path,
    tuned_predictions: Path,
    output: Path | None = None,
    expected_source_index_sha256: str = EXPECTED_SOURCE_INDEX_SHA256,
    expected_evaluation: dict[str, Any] | None = None,
    expected_tokenizer_source_dir: Path = EXPECTED_TOKENIZER_SOURCE_DIR,
    expected_tokenizer_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate every current artifact before authorizing production training."""

    resolved_config_dir = config_dir.expanduser().resolve(strict=True)
    if resolved_config_dir == b200_runtime.B200_CONFIG_DIR:
        missing = [
            name
            for name, value in (
                ("kernel_probe_receipt", kernel_probe_receipt),
                ("environment_receipt", environment_receipt),
                ("asset_integrity_receipt", asset_integrity_receipt),
            )
            if value is None
        ]
        if missing:
            raise RuntimeError(
                "B200 certification requires explicit receipt inputs: "
                + ", ".join(missing)
            )
        return _certify_b200(
            config_dir=resolved_config_dir,
            source_dir=source_dir,
            training_expert_nf4_dir=training_expert_nf4_dir,
            inference_expert_nf4_dir=inference_expert_nf4_dir,
            kernel_probe_receipt=kernel_probe_receipt,
            environment_receipt=environment_receipt,
            asset_integrity_receipt=asset_integrity_receipt,
            tokenizer_contract_receipt=tokenizer_contract_receipt,
            preflight_receipt=preflight_receipt,
            training_manifest=training_manifest,
            base_predictions=base_predictions,
            tuned_predictions=tuned_predictions,
            output=output,
        )
    if resolved_config_dir != CONFIG_DIR.resolve(strict=True):
        raise RuntimeError(
            f"unsupported certification config directory: {resolved_config_dir}"
        )
    source_dir = source_dir.resolve()
    training_expert_nf4_dir = training_expert_nf4_dir.resolve()
    inference_expert_nf4_dir = inference_expert_nf4_dir.resolve()
    preflight_receipt = preflight_receipt.resolve()
    training_manifest = training_manifest.resolve()
    base_predictions = base_predictions.resolve()
    tuned_predictions = tuned_predictions.resolve()
    if source_dir.parent.name != "snapshots" or source_dir.name != BASE_REVISION:
        raise RuntimeError("source-dir is not the pinned Hugging Face snapshot")

    identities = _current_identities()
    source = lightweight_source_identity(source_dir)
    _expect(
        "official source index SHA",
        source["index_sha256"],
        expected_source_index_sha256,
    )
    tokenizer_contract = _validate_tokenizer_contract_receipt(
        tokenizer_contract_receipt,
        source_dir=source_dir,
        identities=identities,
        expected_source_dir=expected_tokenizer_source_dir,
        expected_data=expected_tokenizer_data,
    )
    preflight, training_expert = _validate_preflight(
        preflight_receipt,
        source_dir=source_dir,
        training_expert_dir=training_expert_nf4_dir,
        source=source,
        identities=identities,
    )
    training, adapter_dir, adapter_files = _validate_training(
        training_manifest,
        source_dir=source_dir,
        training_expert_dir=training_expert_nf4_dir,
        source=source,
        preflight=preflight,
        identities=identities,
    )
    inference_expert = lightweight_expert_identity(
        inference_expert_nf4_dir,
        source=source,
    )
    _expect(
        "inference expert source index",
        inference_expert["source_index_sha256"],
        source["index_sha256"],
    )
    _expect(
        "inference expert source config",
        inference_expert["source_config_sha256"],
        source["config_sha256"],
    )
    adapter = validate_adapter(adapter_dir)
    evaluation = (
        _expected_evaluation_identity()
        if expected_evaluation is None
        else expected_evaluation
    )
    base = _validate_prediction_file(
        base_predictions,
        role="base",
        source=source,
        inference_expert=inference_expert,
        adapter=None,
        identities=identities,
        evaluation=evaluation,
    )
    tuned = _validate_prediction_file(
        tuned_predictions,
        role="tuned",
        source=source,
        inference_expert=inference_expert,
        adapter=adapter,
        identities=identities,
        evaluation=evaluation,
    )
    _expect("base/tuned prediction keys", tuned["keys"], base["keys"])
    _expect("base/tuned evaluation contract", tuned["evaluation"], base["evaluation"])
    _expect("base/tuned decoding contract", tuned["decoding"], base["decoding"])
    if base["model_label"] == tuned["model_label"]:
        raise RuntimeError("base and tuned predictions use the same model label")

    smoke = training["smoke"]
    result = {
        "status": "PASS",
        "technical_production_gate_passed": True,
        "authorization_scope": "technical_smoke_gate_only",
        "human_production_approval_required": True,
        "human_production_approval_recorded": False,
        "project": str(PROJECT),
        "source_dir": str(source_dir),
        # train.py consumes this legacy key as the training expert artifact.
        "expert_nf4_dir": str(training_expert_nf4_dir),
        "training_expert_nf4_dir": str(training_expert_nf4_dir),
        "inference_expert_nf4_dir": str(inference_expert_nf4_dir),
        "model_contract_sha256": identities["model_contract_sha256"],
        "recipe_sha256": identities["recipe_sha256"],
        "data_contract_sha256": identities["data_contract_sha256"],
        "pipeline_code": identities["pipeline_code"],
        "environment_lock_sha256": identities["environment_lock_sha256"],
        "source": source,
        "training_expert": training_expert,
        "inference_expert": inference_expert,
        "preflight_receipt": {
            "path": str(preflight_receipt),
            "sha256": sha256(preflight_receipt),
        },
        "tokenizer_contract_receipt": tokenizer_contract,
        "training_manifest": {
            "path": str(training_manifest),
            "sha256": sha256(training_manifest),
        },
        "adapter_dir": str(adapter_dir),
        "adapter_files_sha256": adapter_files,
        "base_predictions": {
            key: value
            for key, value in base.items()
            if key not in {"keys", "evaluation"}
        },
        "tuned_predictions": {
            key: value
            for key, value in tuned.items()
            if key not in {"keys", "evaluation"}
        },
        "base_model_label": base["model_label"],
        "lora_model_label": tuned["model_label"],
        "gradient_checkpointing_contract": training["gradient_checkpointing_contract"],
        "post_update_recompute_audit": training["post_update_recompute_audit"],
        "token_length_contract": training["token_length_contract"],
        "validation_resume_contract": training["validation_resume_contract"],
        "storage_audit": training["storage_audit"],
        "production_training_plan": smoke["production_training_plan"],
        "subset_validation_plan": smoke["subset_validation_plan"],
        "full_validation_plan": smoke["full_validation_plan"],
        "early_stopping_policy": training["config"]["validation"]["early_stopping"],
        "early_stopping_history": training["history"],
        "early_stopping_decision": training["early_stopping_decision"],
        "checks": {
            "model_preflight": "PASS",
            "tokenizer_output_contract": "PASS",
            "max_length_backward": "PASS",
            "absolute_train_max_forward_backward": "PASS",
            "two_epoch_global_deck": "PASS",
            "dynamic_microbatch_target_batches": "PASS",
            "zero_drop_full_length": "PASS",
            "full_validation_candidate_schedule": "PASS",
            "early_stopping_policy": "PASS",
            "early_stopping_decision_recomputed": "PASS",
            "partial_validation_resume": "PASS",
            "storage_capacity": "PASS",
            "lora_module_contract": "PASS",
            "no_cpu_disk_offload": "PASS",
            "base_inference": "PASS",
            "lora_inference": "PASS",
            "current_code_and_environment": "PASS",
            "source_and_expert_provenance": "PASS",
            "finite_nonzero_lora_gradient": "PASS",
            "lora_update_and_frozen_base": "PASS",
            "production_first_update_learning_rate": "PASS",
            "one_time_gradient_audit_projection": "PASS",
            "device_aware_gradient_checkpointing": "PASS",
            "post_update_checkpoint_recompute": "PASS",
            "output_only_supervision": "PASS",
            "sealed_prediction_contracts": "PASS",
            "runtime_budget_feasibility": "PASS",
        },
        "smoke_observed_tokens": smoke["observed_tokens"],
        "smoke_first_update_learning_rate": smoke["adapter_update"][
            "learning_rate_used"
        ],
        "smoke_one_time_gradient_audit_seconds": smoke[
            "one_time_gradient_audit_seconds"
        ],
        "smoke_peak_allocated_bytes": smoke["peak_allocated_bytes"],
        "runtime_budget_projection": training["runtime_budget_projection"],
    }
    if output is not None:
        atomic_json(output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--training-expert-nf4-dir", type=Path, required=True)
    parser.add_argument("--inference-expert-nf4-dir", type=Path, required=True)
    parser.add_argument(
        "--tokenizer-contract-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument("--kernel-probe-receipt", type=Path)
    parser.add_argument("--environment-receipt", type=Path)
    parser.add_argument("--asset-integrity-receipt", type=Path)
    parser.add_argument("--preflight-receipt", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--tuned-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_dir = args.config_dir.expanduser().resolve(strict=True)
    if config_dir == b200_runtime.B200_CONFIG_DIR:
        result = certify(
            config_dir=args.config_dir,
            source_dir=args.source_dir,
            training_expert_nf4_dir=args.training_expert_nf4_dir,
            inference_expert_nf4_dir=args.inference_expert_nf4_dir,
            kernel_probe_receipt=args.kernel_probe_receipt,
            environment_receipt=args.environment_receipt,
            asset_integrity_receipt=args.asset_integrity_receipt,
            tokenizer_contract_receipt=args.tokenizer_contract_receipt,
            preflight_receipt=args.preflight_receipt,
            training_manifest=args.training_manifest,
            base_predictions=args.base_predictions,
            tuned_predictions=args.tuned_predictions,
            output=args.output,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.tokenizer_contract_receipt.is_symlink():
        raise RuntimeError("tokenizer contract receipt must be a regular file")
    result = certify(
        config_dir=config_dir,
        source_dir=require_owned_path(args.source_dir),
        training_expert_nf4_dir=require_owned_path(args.training_expert_nf4_dir),
        inference_expert_nf4_dir=require_owned_path(args.inference_expert_nf4_dir),
        tokenizer_contract_receipt=require_owned_path(args.tokenizer_contract_receipt),
        preflight_receipt=require_owned_path(args.preflight_receipt),
        training_manifest=require_owned_path(args.training_manifest),
        base_predictions=require_owned_path(args.base_predictions),
        tuned_predictions=require_owned_path(args.tuned_predictions),
        output=require_owned_path(args.output, create_parent=True),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
