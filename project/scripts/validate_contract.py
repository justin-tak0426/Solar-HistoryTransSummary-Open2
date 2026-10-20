#!/usr/bin/env python3
"""Validate the immutable model recipe, source data, and evaluation contract."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import b200_runtime

from common import (
    B200_CONFIG_DIR,
    CONFIG_DIR,
    LICENSE_SHA256,
    PROJECT,
    REFERENCE_COPY,
    REFERENCE_SHA256,
    REFERENCE_SOURCE,
    RESULT_LABEL_PREFIX_RE,
    TASKS,
    expected_lora_module_names,
    iter_jsonl,
    load_reference_module,
    output_only_violations,
    read_json,
    sha256,
)
from training_plan import (
    occurrences_in_step,
    occurrences_through_step,
    validate_b200_early_stopping_policy,
    validate_early_stopping_policy,
)


SOURCE_RECIPE_NAME = "solar_open2_hardterm_fancy_v3_earlystop"
B200_RECIPE_NAME = "solar_open2_b200_exact_one_epoch_v1"


def fail(message: str) -> None:
    raise RuntimeError(message)


def validate_evaluator_identity(config_dir: Path) -> dict[str, Any]:
    """Validate only the owned evaluator for B200, retaining source checks otherwise."""
    resolved_config_dir = _resolve_config_dir(config_dir)
    if sha256(REFERENCE_COPY) != REFERENCE_SHA256:
        fail("owned evaluator copy checksum changed")
    evidence: dict[str, Any] = {
        "owned_path": str(REFERENCE_COPY.resolve()),
        "sha256": REFERENCE_SHA256,
        "owned_only": resolved_config_dir == B200_CONFIG_DIR.resolve(),
    }
    if evidence["owned_only"]:
        return evidence
    if sha256(REFERENCE_SOURCE) != REFERENCE_SHA256:
        fail("reference evaluator source checksum changed")
    if REFERENCE_COPY.read_bytes() != REFERENCE_SOURCE.read_bytes():
        fail("owned evaluator copy is not byte-identical to source")
    evidence["source_path"] = str(REFERENCE_SOURCE)
    return evidence


def _values_exact(observed: Any, expected: Any) -> bool:
    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(observed) == set(expected) and all(
            _values_exact(observed[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(observed) == len(expected) and all(
            _values_exact(left, right)
            for left, right in zip(observed, expected, strict=True)
        )
    return observed == expected


def _require_exact_fields(
    observed: Any,
    expected: dict[str, Any],
    *,
    label: str,
    exact_keys: bool = False,
) -> dict[str, Any]:
    if not isinstance(observed, dict):
        fail(f"{label} must be an object")
    if exact_keys and set(observed) != set(expected):
        fail(f"{label} fields changed")
    for key, value in expected.items():
        if key not in observed or not _values_exact(observed[key], value):
            fail(f"{label} {key}={observed.get(key)!r}, expected={value!r}")
    return observed


def normalized_origin_key(row: dict[str, Any]) -> tuple[str, str]:
    normalized = re.sub(r"\s+", " ", str(row.get("origin") or "").strip())
    return str(row.get("task") or ""), normalized


def scan_rows(
    path: Path,
    *,
    expected_rows: int,
    expected_by_task: dict[str, int],
    expected_quality: dict[str, int] | None = None,
) -> dict[str, Any]:
    ids: set[tuple[str, str]] = set()
    origins: set[tuple[str, str]] = set()
    task_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    result_label_ids: set[str] = set()
    for line_number, row in enumerate(iter_jsonl(path), 1):
        task = str(row.get("task") or "")
        row_id = str(row.get("id") or "")
        if task not in TASKS or not row_id:
            fail(f"{path}:{line_number}: invalid id/task")
        key = (row_id, task)
        if key in ids:
            fail(f"{path}:{line_number}: duplicate id/task {key}")
        ids.add(key)
        origin_key = normalized_origin_key(row)
        if origin_key in origins:
            fail(f"{path}:{line_number}: duplicate normalized origin/task")
        origins.add(origin_key)
        task_counts[task] += 1
        if expected_quality is not None:
            quality_counts[str(row.get("quality") or "")] += 1
        if row.get("prompt_variant") != "no_dictionary_context":
            fail(f"{path}:{line_number}: wrong prompt_variant")
        if (
            not str(row.get("origin") or "").strip()
            or not str(row.get("response") or "").strip()
        ):
            fail(f"{path}:{line_number}: empty origin/response")
        if RESULT_LABEL_PREFIX_RE.match(str(row.get("response") or "")):
            result_label_ids.add(row_id)
        normalized_response = RESULT_LABEL_PREFIX_RE.sub(
            "", str(row.get("response") or ""), count=1
        )
        violations = output_only_violations(normalized_response)
        if violations:
            fail(f"{path}:{line_number}: non-body training target {violations}")
    if len(ids) != expected_rows:
        fail(f"{path}: rows={len(ids)} expected={expected_rows}")
    if dict(task_counts) != expected_by_task:
        fail(f"{path}: task counts={dict(task_counts)} expected={expected_by_task}")
    if expected_quality is not None and dict(quality_counts) != expected_quality:
        fail(
            f"{path}: quality counts={dict(quality_counts)} "
            f"expected={expected_quality}"
        )
    return {
        "ids": ids,
        "origins": origins,
        "rows": len(ids),
        "by_task": dict(task_counts),
        "quality": dict(quality_counts),
        "result_label_ids": result_label_ids,
    }


def _validate_source_recipe_contract(recipe: dict[str, Any]) -> dict[str, Any]:
    if recipe.get("recipe_name") != ("solar_open2_hardterm_fancy_v3_earlystop"):
        fail("approved early-stopping recipe name changed")
    fixed = {
        "epochs": 2,
        "optimizer_steps": 18631,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "effective_batch_size": 16,
        "max_length": 9600,
        "learning_rate": 0.0001,
        "minimum_learning_rate_ratio": 0.1,
        "hard_term_target_reading_loss_weight": 2.5,
        "expert_oversampling_factor": 8,
        "neftune_alpha": 5.0,
    }
    for key, expected in fixed.items():
        if recipe.get(key) != expected:
            fail(f"recipe {key}={recipe.get(key)!r}, expected={expected!r}")
    if recipe.get("scheduler") != "cosine" or recipe.get("warmup_ratio") != 0.03:
        fail("cosine scheduler or 3% warmup contract changed")
    if recipe["lora"]["rank"] != 32 or recipe["lora"]["alpha"] != 64:
        fail("LoRA rank/alpha contract changed")

    policy = recipe.get("microbatch_policy")
    if not isinstance(policy, dict):
        fail("dynamic microbatch policy is missing")
    if (
        policy.get("mode") != "dynamic_length_bucket_target_subject_to_gpu_smoke"
        or policy.get("fallback_micro_batch_size") != 1
        or policy.get("fallback_gradient_accumulation_steps") != 16
        or policy.get("optimizer_step_occurrences") != recipe["effective_batch_size"]
    ):
        fail("dynamic microbatch fallback or optimizer grouping changed")
    observed_buckets = [
        (
            bucket.get("min_length_inclusive"),
            bucket.get("max_length_inclusive"),
            bucket.get("target_micro_batch_size"),
        )
        for bucket in policy.get("buckets") or []
        if isinstance(bucket, dict)
    ]
    expected_buckets = [
        (1, 1024, 8),
        (1025, 2048, 4),
        (2049, 4096, 2),
        (4097, 9600, 1),
    ]
    if observed_buckets != expected_buckets:
        fail(f"dynamic microbatch buckets changed: {observed_buckets!r}")

    deck = recipe.get("training_deck")
    if not isinstance(deck, dict):
        fail("training deck contract is missing")
    if (
        deck.get("policy") != "deterministic_global_augmented_epoch_deck"
        or deck.get("shuffle") != "deterministic_per_epoch_seed"
        or deck.get("base_unique_rows") != 142713
        or deck.get("expert_unique_rows") != 905
        or deck.get("expert_occurrences_per_row_per_epoch") != 8
    ):
        fail("deterministic augmented-deck identity changed")
    per_epoch = deck.get("occurrences_per_epoch")
    total = deck.get("total_occurrences")
    expected_per_epoch = {
        "translation": 102224,
        "summarization": 46824,
        "total": 149048,
    }
    expected_total = {
        "translation": 204448,
        "summarization": 93648,
        "total": 298096,
    }
    if per_epoch != expected_per_epoch or total != expected_total:
        fail("approved augmented-deck occurrence counts changed")
    if per_epoch["translation"] + per_epoch["summarization"] != per_epoch["total"]:
        fail("per-epoch task occurrences do not sum to the deck size")
    if any(
        total[task] != recipe["epochs"] * per_epoch[task]
        for task in ("translation", "summarization", "total")
    ):
        fail("two-epoch occurrence totals do not derive from the epoch deck")
    if total["total"] != occurrences_through_step(
        int(recipe["optimizer_steps"]),
        occurrences_per_step=int(recipe["effective_batch_size"]),
        total_occurrences=int(total["total"]),
    ):
        fail("optimizer steps do not exactly consume the two-epoch deck")
    if deck.get("derived_optimizer_steps") != recipe["optimizer_steps"]:
        fail("deck-derived optimizer step count changed")
    natural = deck.get("natural_task_fraction") or {}
    for task in ("translation", "summarization"):
        expected = per_epoch[task] / per_epoch["total"]
        observed = natural.get(task)
        if not isinstance(observed, (int, float)) or not math.isclose(
            float(observed), expected, rel_tol=0.0, abs_tol=1e-15
        ):
            fail(f"natural training-deck fraction changed for {task}")

    validation = recipe.get("validation")
    if not isinstance(validation, dict):
        fail("validation contract is missing")
    if validation.get("eval_every_steps") != 1000:
        fail("fixed-subset validation cadence changed")
    weights = validation.get("selection_weights")
    if weights != {"translation": 0.65, "summarization": 0.35}:
        fail("validation selection weights changed")
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-15):
        fail("validation selection weights do not sum to one")
    full_steps = validation.get("full_validation_steps")
    if full_steps != [9316, 13973, 18631]:
        fail("full-validation candidate steps changed")
    expected_candidates = [
        {
            "target_epoch": 1.0,
            "step": 9316,
            "consumed_occurrences": 149056,
            "target_occurrences": 149048,
            "occurrence_offset": 8,
        },
        {
            "target_epoch": 1.5,
            "step": 13973,
            "consumed_occurrences": 223568,
            "target_occurrences": 223572,
            "occurrence_offset": -4,
        },
        {
            "target_epoch": 2.0,
            "step": 18631,
            "consumed_occurrences": 298096,
            "target_occurrences": 298096,
            "occurrence_offset": 0,
        },
    ]
    if validation.get("full_validation_candidates") != expected_candidates:
        fail("full-validation candidate metadata changed")
    for candidate in expected_candidates:
        if candidate["consumed_occurrences"] != occurrences_through_step(
            int(candidate["step"]),
            occurrences_per_step=int(recipe["effective_batch_size"]),
            total_occurrences=int(total["total"]),
        ):
            fail("candidate consumed-occurrence arithmetic is invalid")
        if candidate["occurrence_offset"] != (
            candidate["consumed_occurrences"] - candidate["target_occurrences"]
        ):
            fail("candidate occurrence offset is invalid")
    if (
        validation.get("selection_policy") != "argmin_full_validation_selection_loss"
        or validation.get("selection_tie_break") != "earlier_optimizer_step"
        or validation.get("selected_step") is not None
    ):
        fail("validation-best selection contract changed")
    try:
        early_stopping = validate_early_stopping_policy(
            validation.get("early_stopping"),
            optimizer_steps=int(recipe["optimizer_steps"]),
            full_validation_steps=[int(step) for step in full_steps],
        )
    except (TypeError, ValueError) as error:
        fail(f"early-stopping policy changed: {error}")

    checkpoint = recipe.get("checkpoint")
    if not isinstance(checkpoint, dict) or any(
        (
            checkpoint.get("save_every_steps") != 100,
            checkpoint.get("save_every_hours") != 4,
            checkpoint.get("recovery_trigger") != "steps_or_hours_whichever_first",
            checkpoint.get("recovery_generations_to_keep") != 2,
            checkpoint.get("candidate_checkpoint_retention") != "retain_all",
            checkpoint.get("final_adapter_dirname") != "final_adapter",
            checkpoint.get("best_adapter_dirname") != "best_adapter",
        )
    ):
        fail("checkpoint/recovery contract changed")

    evaluation = recipe.get("evaluation")
    expected_evaluation = {
        "datasets": ["full_test", "human_gold"],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_model_len": 16384,
        "translation_max_new_tokens": 8192,
        "summarization_max_new_tokens": 6144,
        "bleu_tokenizer": "ko-mecab",
        "bert_model": "xlm-roberta-large",
        "bert_batch_size": 8,
        "metrics": [
            "corpus_bleu",
            "sentence_bleu_mean",
            "rougeL_f1_mean",
            "bertscore_f1_mean",
        ],
    }
    if evaluation != expected_evaluation:
        fail("sealed evaluation contract changed")
    return {
        "epochs": recipe["epochs"],
        "optimizer_steps": recipe["optimizer_steps"],
        "max_length": recipe["max_length"],
        "total_occurrences": total["total"],
        "full_validation_steps": full_steps,
        "selection_policy": validation["selection_policy"],
        "early_stopping_policy": early_stopping,
    }


def _validate_b200_recipe_contract(recipe: dict[str, Any]) -> dict[str, Any]:
    _require_exact_fields(
        recipe,
        {
            "schema": "solar-open2-b200-training-recipe",
            "schema_version": 1,
            "profile": "b200",
            "recipe_name": B200_RECIPE_NAME,
            "epochs": 1,
            "optimizer_steps": 9316,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "effective_batch_size": 16,
            "max_length": 9600,
            "actual_longest_row_tokens": 9560,
            "learning_rate": 0.0001,
            "weight_decay": 0.01,
            "adam_betas": [0.9, 0.95],
            "max_grad_norm": 1.0,
            "scheduler": "positive_cosine",
            "warmup_ratio": 0.03,
            "warmup_updates": 279,
            "minimum_learning_rate_ratio": 0.1,
            "minimum_learning_rate": 0.00001,
            "scheduler_resume_policy": (
                "recompute from durable completed optimizer updates"
            ),
            "seed": 42,
            "hard_term_target_reading_loss_weight": 2.5,
            "expert_quality_label": "gold_expert_corrected",
            "expert_oversampling_factor": 8,
            "neftune_alpha": 5.0,
        },
        label="B200 recipe",
    )

    _require_exact_fields(
        recipe.get("training_performance"),
        b200_runtime.b200_training_performance_contract(),
        label="B200 training performance",
        exact_keys=True,
    )
    _require_exact_fields(
        recipe.get("runtime_gate"),
        {
            "projection_required": True,
            "safety_factor": 1.25,
            "recommended_hours": 144.0,
            "former_absolute_limit_hours": 168.0,
            "hard_limit_enforced": False,
            "override_scope": "runtime_projection_only_exact_one_epoch",
            "user_instruction_sha256": (
                "2bd9df6c227f966e66220029ad927d8021bcc5d3001791d19c2d8d2389f910ab"
            ),
            "non_runtime_gates_overridden": False,
        },
        label="B200 runtime gate",
        exact_keys=True,
    )

    microbatch = _require_exact_fields(
        recipe.get("microbatch_policy"),
        {
            "mode": "dynamic_length_bucket_target_subject_to_b200_smoke",
            "fallback_micro_batch_size": 1,
            "fallback_gradient_accumulation_steps": 16,
            "optimizer_step_occurrences": 16,
            "tail_step_occurrences": 8,
            "deck_order_contract": (
                "consecutive occurrence groups with no cross-step pull-ahead"
            ),
            "qualification_required_before_increase": True,
            "buckets": [
                {
                    "min_length_inclusive": 1,
                    "max_length_inclusive": 1024,
                    "target_micro_batch_size": 8,
                },
                {
                    "min_length_inclusive": 1025,
                    "max_length_inclusive": 2048,
                    "target_micro_batch_size": 4,
                },
                {
                    "min_length_inclusive": 2049,
                    "max_length_inclusive": 4096,
                    "target_micro_batch_size": 2,
                },
                {
                    "min_length_inclusive": 4097,
                    "max_length_inclusive": 9600,
                    "target_micro_batch_size": 1,
                },
            ],
        },
        label="B200 microbatch policy",
        exact_keys=True,
    )
    nominal = int(microbatch["optimizer_step_occurrences"])
    total_occurrences = 149_048
    optimizer_steps = int(recipe["optimizer_steps"])
    if (
        occurrences_through_step(
            optimizer_steps,
            occurrences_per_step=nominal,
            total_occurrences=total_occurrences,
        )
        != total_occurrences
    ):
        fail("B200 optimizer horizon does not consume the exact epoch")

    expected_deck = {
        "policy": "deterministic_global_augmented_epoch_deck",
        "shuffle": "deterministic_per_epoch_seed",
        "base_unique_rows": 142713,
        "expert_unique_rows": 905,
        "expert_occurrences_per_row_per_epoch": 8,
        "occurrences_per_epoch": {
            "translation": 102224,
            "summarization": 46824,
            "total": 149048,
        },
        "natural_task_fraction": {
            "translation": 0.6858461703612259,
            "summarization": 0.3141538296387741,
        },
        "total_occurrences": {
            "translation": 102224,
            "summarization": 46824,
            "total": 149048,
        },
        "derived_optimizer_steps": 9316,
        "occurrence_identity_fields": [
            "id",
            "task",
            "quality",
            "expert_ordinal",
        ],
        "zero_skipped_occurrences_required": True,
        "zero_duplicated_occurrences_required": True,
    }
    deck = _require_exact_fields(
        recipe.get("training_deck"),
        expected_deck,
        label="B200 training deck",
        exact_keys=True,
    )
    if deck["occurrences_per_epoch"] != deck["total_occurrences"]:
        fail("B200 training deck must contain exactly one epoch")

    expected_tail = {
        "total_occurrences": 149048,
        "nominal_step_occurrences": 16,
        "full_steps": 9315,
        "full_step_occurrences": 149040,
        "final_step": 9316,
        "final_step_occurrences": 8,
        "padding_or_duplication_allowed": False,
        "loss_denominator_policy": "occurrences_in_step",
        "gradient_denominator_policy": "occurrences_in_step",
        "occurrences_through_step_formula": "min(step * 16, 149048)",
        "occurrences_in_step_formula": (
            "occurrences_through_step(step) - " "occurrences_through_step(step - 1)"
        ),
        "resume_completed_steps": [9315, 9316],
    }
    tail = _require_exact_fields(
        recipe.get("exact_tail"),
        expected_tail,
        label="B200 exact tail",
        exact_keys=True,
    )
    if occurrences_through_step(
        int(tail["full_steps"]),
        occurrences_per_step=nominal,
        total_occurrences=total_occurrences,
    ) != int(tail["full_step_occurrences"]):
        fail("B200 penultimate resume boundary is invalid")
    if (
        occurrences_through_step(
            int(tail["final_step"]),
            occurrences_per_step=nominal,
            total_occurrences=total_occurrences,
        )
        != total_occurrences
    ):
        fail("B200 terminal resume boundary is invalid")
    if occurrences_in_step(
        int(tail["final_step"]),
        occurrences_per_step=nominal,
        total_occurrences=total_occurrences,
    ) != int(tail["final_step_occurrences"]):
        fail("B200 final-step denominator is invalid")

    _require_exact_fields(
        recipe.get("lora"),
        {
            "rank": 32,
            "alpha": 64,
            "dropout": 0.05,
            "bias": "none",
            "expected_module_count": 528,
            "expected_trainable_parameters": 133177344,
        },
        label="B200 LoRA contract",
    )

    validation = recipe.get("validation")
    expected_candidates = [
        {
            "target_epoch": 0.5,
            "step": 4658,
            "consumed_occurrences": 74528,
            "target_occurrences": 74524,
            "occurrence_offset": 4,
        },
        {
            "target_epoch": 1.0,
            "step": 9316,
            "consumed_occurrences": 149048,
            "target_occurrences": 149048,
            "occurrence_offset": 0,
        },
    ]
    validation = _require_exact_fields(
        validation,
        {
            "monitor_datasets": ["validation"],
            "step_zero_baseline": True,
            "eval_every_steps": 1000,
            "subset_per_task": 512,
            "subset_seed": 42,
            "subset_identity_and_order_sealed": True,
            "selection_loss": ("0.65 * translation_loss + 0.35 * summarization_loss"),
            "selection_weights": {
                "translation": 0.65,
                "summarization": 0.35,
            },
            "validation_loss_hard_term_weight": 1.0,
            "full_validation_steps": [4658, 9316],
            "full_validation_candidates": expected_candidates,
            "candidate_boundary_policy": (
                "nearest optimizer boundary; at an exact half-step choose "
                "the later boundary"
            ),
            "selection_policy": "argmin_full_validation_selection_loss",
            "selection_tie_break": "earlier_optimizer_step",
            "selected_step": None,
        },
        label="B200 validation contract",
    )
    if not math.isclose(
        sum(validation["selection_weights"].values()),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        fail("B200 validation selection weights do not sum to one")
    full_steps = [int(step) for step in validation["full_validation_steps"]]
    for candidate in expected_candidates:
        if int(candidate["consumed_occurrences"]) != occurrences_through_step(
            int(candidate["step"]),
            occurrences_per_step=nominal,
            total_occurrences=total_occurrences,
        ):
            fail("B200 candidate consumed-occurrence arithmetic is invalid")
        if int(candidate["occurrence_offset"]) != (
            int(candidate["consumed_occurrences"])
            - int(candidate["target_occurrences"])
        ):
            fail("B200 candidate occurrence offset is invalid")
    try:
        early_stopping = validate_b200_early_stopping_policy(
            validation.get("early_stopping"),
            optimizer_steps=optimizer_steps,
            full_validation_steps=full_steps,
        )
    except (TypeError, ValueError) as error:
        fail(f"B200 early-stopping policy changed: {error}")

    _require_exact_fields(
        recipe.get("checkpoint"),
        {
            "save_every_steps": 100,
            "save_every_hours": 4,
            "recovery_trigger": "steps_or_hours_whichever_first",
            "recovery_generations_to_keep": 2,
            "candidate_checkpoint_retention": "retain_all",
            "final_adapter_dirname": "final_adapter",
            "best_adapter_dirname": "best_adapter",
            "signal": "SIGUSR1",
            "interrupt_exit_code": 75,
            "resume_requires_exact_occurrence_identity": True,
        },
        label="B200 checkpoint contract",
        exact_keys=True,
    )
    evaluation = _require_exact_fields(
        recipe.get("evaluation"),
        {
            "datasets": ["full_test", "human_gold"],
            "allowed_only_after_terminal_training_success": True,
            "used_for_training_monitor_or_selection": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_model_len": 16384,
            "translation_max_new_tokens": 8192,
            "summarization_max_new_tokens": 6144,
            "bleu_tokenizer": "ko-mecab",
            "bert_model": "xlm-roberta-large",
            "bert_batch_size": 8,
            "metrics": [
                "corpus_bleu",
                "sentence_bleu_mean",
                "rougeL_f1_mean",
                "bertscore_f1_mean",
            ],
        },
        label="B200 sealed evaluation contract",
        exact_keys=True,
    )
    if set(validation["monitor_datasets"]) & set(evaluation["datasets"]):
        fail("B200 test/human-gold datasets cannot be training monitors")

    subset_steps = [0, *range(1000, optimizer_steps + 1, 1000)]
    return {
        "profile": "b200",
        "epochs": 1,
        "optimizer_steps": optimizer_steps,
        "max_length": int(recipe["max_length"]),
        "total_occurrences": total_occurrences,
        "nominal_step_occurrences": nominal,
        "final_step_occurrences": occurrences_in_step(
            optimizer_steps,
            occurrences_per_step=nominal,
            total_occurrences=total_occurrences,
        ),
        "resume_completed_steps": list(tail["resume_completed_steps"]),
        "fixed_subset_validation_steps": subset_steps,
        "full_validation_steps": full_steps,
        "full_validation_consumed_occurrences": [
            int(candidate["consumed_occurrences"]) for candidate in expected_candidates
        ],
        "selection_policy": validation["selection_policy"],
        "early_stopping_policy": early_stopping,
    }


def validate_recipe_contract(recipe: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(recipe, dict):
        fail("recipe must be an object")
    recipe_name = recipe.get("recipe_name")
    if recipe_name == SOURCE_RECIPE_NAME:
        return _validate_source_recipe_contract(recipe)
    if recipe_name == B200_RECIPE_NAME:
        return _validate_b200_recipe_contract(recipe)
    fail(f"approved recipe name changed: {recipe_name!r}")


def _resolve_config_dir(config_dir: str | Path | None) -> Path:
    raw = CONFIG_DIR if config_dir is None else Path(config_dir)
    try:
        resolved = raw.resolve(strict=True)
    except OSError as error:
        fail(f"config directory cannot be resolved: {raw}: {error}")
    if not resolved.is_dir():
        fail(f"config directory is not a directory: {resolved}")
    missing = [
        name
        for name in ("model.json", "recipe.json", "data_contract.json")
        if not (resolved / name).is_file()
    ]
    if missing:
        fail(f"config directory is incomplete: missing {missing}")
    return resolved


def validate_recipe(
    config_dir: str | Path | None = None,
) -> dict[str, Any]:
    resolved_config_dir = _resolve_config_dir(config_dir)
    recipe_path = resolved_config_dir / "recipe.json"
    model_path = resolved_config_dir / "model.json"
    recipe = read_json(recipe_path)
    model = read_json(model_path)
    recipe_evidence = validate_recipe_contract(recipe)
    if recipe["prompt"]["reasoning_effort"] != "none":
        fail("Solar prompt must explicitly disable reasoning")
    output_contract = recipe["output_only_contract"]
    for key in (
        "training_target_prefix_removal",
        "sealed_full_test_reference_exceptions",
    ):
        exception = output_contract[key]
        ids = exception.get("ids") or []
        if (
            exception.get("prefix") != "답변: "
            or int(exception.get("expected_rows", -1)) != len(set(ids))
            or len(ids) != len(set(ids))
        ):
            fail(f"output-only exception contract changed: {key}")
    if model["base"]["revision"] != "8d1dda5ad2af58207c65348d825772d2f4d72481":
        fail("base model revision changed")
    runtime = model["training_runtime"]
    if runtime["expert_artifact_schema"] != (
        "solar-open2-layer-flat-routed-experts-nf4"
    ):
        fail("training must use the audited layer-flat NF4 expert artifact")
    if runtime["required_manifest_status"] != "SEALED":
        fail("training runtime must require a sealed expert artifact")
    if model["base"]["indexed_tensor_count"] != 47019:
        fail("official source tensor count contract changed")
    if model["base"]["indexed_tensor_bytes"] != 500575651328:
        fail("official source indexed byte count contract changed")
    environment = Path(model["software"]["environment_path"])
    if environment != PROJECT / "solar-open-2":
        fail("Solar environment must live inside the executable project")
    if not (environment / "bin/python").is_file():
        fail(f"Solar environment is missing: {environment}")
    if not model["derivative_model_name"].startswith("Solar"):
        fail("derivative model name must begin with Solar")
    if model["attribution"] != "Built with Solar":
        fail("required Solar attribution is missing")
    expected_modules = expected_lora_module_names(model["architecture_contract"])
    if len(expected_modules) != recipe["lora"]["expected_module_count"]:
        fail("LoRA module count does not match architecture")
    if any(
        ".mlp.experts." in name or ".mlp.gate." in name for name in expected_modules
    ):
        fail("routed experts/router entered LoRA target set")
    return {
        "config_dir": str(resolved_config_dir),
        "recipe_path": str(recipe_path),
        "recipe_sha256": sha256(recipe_path),
        "model_path": str(model_path),
        "model_sha256": sha256(model_path),
        "lora_module_count": len(expected_modules),
        **recipe_evidence,
    }


def validate_eval_rows(
    path: Path,
    expected: dict[str, Any],
    prompt_module: Any,
) -> dict[str, Any]:
    keys: set[tuple[str, str]] = set()
    counts: Counter[str] = Counter()
    result_label_ids: set[str] = set()
    for line_number, row in enumerate(iter_jsonl(path), 1):
        key = (str(row.get("id") or ""), str(row.get("task") or ""))
        if not key[0] or key[1] not in TASKS or key in keys:
            fail(f"{path}:{line_number}: invalid or duplicate key {key}")
        keys.add(key)
        counts[key[1]] += 1
        origin = prompt_module.clean_text(row.get("origin"))
        wanted = prompt_module.build_messages_no_dict(key[1], origin)
        if row.get("messages") != wanted:
            fail(f"{path}:{line_number}: non-canonical no-dictionary prompt")
        forbidden = [
            name
            for name in prompt_module.FORBIDDEN_NONEMPTY_FIELDS
            if prompt_module.nonempty(row.get(name))
        ]
        if forbidden:
            fail(f"{path}:{line_number}: dictionary metadata present: {forbidden}")
        if row.get("prompt_variant") != "no_dictionary_context":
            fail(f"{path}:{line_number}: wrong prompt variant")
        if RESULT_LABEL_PREFIX_RE.match(str(row.get("response") or "")):
            result_label_ids.add(key[0])
    if len(keys) != int(expected["rows"]) or dict(counts) != expected["by_task"]:
        fail(f"{path}: eval counts mismatch rows={len(keys)} tasks={dict(counts)}")
    return {
        "keys": keys,
        "rows": len(keys),
        "by_task": dict(counts),
        "result_label_ids": result_label_ids,
    }


def validate_data(
    *,
    skip_large_hashes: bool,
    config_dir: str | Path | None = None,
) -> dict[str, Any]:
    resolved_config_dir = _resolve_config_dir(config_dir)
    contract_path = resolved_config_dir / "data_contract.json"
    contract = read_json(contract_path)
    prepared = Path(contract["prepared_data_dir"])
    raw = Path(contract["raw_data_dir"])
    if (
        sha256(Path(contract["prepared_manifest"]))
        != contract["prepared_manifest_sha256"]
    ):
        fail("prepared data manifest checksum changed")
    if sha256(Path(contract["human_gold_raw"])) != contract["human_gold_raw_sha256"]:
        fail("human gold raw checksum changed")
    for name, expected_hash in contract["raw_files"].items():
        path = raw / name
        if sha256(path) != expected_hash:
            fail(f"raw source checksum changed: {path}")
    if not skip_large_hashes:
        for entry in contract["files"].values():
            path = prepared / entry["name"]
            if sha256(path) != entry["sha256"]:
                fail(f"prepared data checksum changed: {path}")

    train = scan_rows(
        prepared / contract["files"]["train"]["name"],
        expected_rows=contract["files"]["train"]["rows"],
        expected_by_task=contract["files"]["train"]["by_task"],
        expected_quality=contract["files"]["train"]["quality"],
    )
    validation = scan_rows(
        prepared / contract["files"]["validation"]["name"],
        expected_rows=contract["files"]["validation"]["rows"],
        expected_by_task=contract["files"]["validation"]["by_task"],
    )
    prompt_module = load_reference_module(
        config_dir=(
            resolved_config_dir
            if resolved_config_dir == B200_CONFIG_DIR.resolve()
            else None
        )
    )
    full_test = validate_eval_rows(
        prepared / contract["files"]["full_test"]["name"],
        contract["files"]["full_test"],
        prompt_module,
    )
    human_gold = validate_eval_rows(
        prepared / contract["files"]["human_gold"]["name"],
        contract["files"]["human_gold"],
        prompt_module,
    )
    combined = validate_eval_rows(
        prepared / contract["files"]["combined_eval"]["name"],
        contract["files"]["combined_eval"],
        prompt_module,
    )
    output_contract = read_json(resolved_config_dir / "recipe.json")[
        "output_only_contract"
    ]
    expected_train_labels = set(
        output_contract["training_target_prefix_removal"]["ids"]
    )
    expected_full_labels = set(
        output_contract["sealed_full_test_reference_exceptions"]["ids"]
    )
    observed_labels = {
        "train": train["result_label_ids"],
        "validation": validation["result_label_ids"],
        "full_test": full_test["result_label_ids"],
        "human_gold": human_gold["result_label_ids"],
        "combined_eval": combined["result_label_ids"],
    }
    expected_labels = {
        "train": expected_train_labels,
        "validation": set(),
        "full_test": expected_full_labels,
        "human_gold": set(),
        "combined_eval": expected_full_labels,
    }
    if observed_labels != expected_labels:
        fail(
            "output-only target/reference exceptions changed: "
            f"observed={observed_labels} expected={expected_labels}"
        )
    for left_name, left, right_name, right in (
        ("train", train["ids"], "validation", validation["ids"]),
        ("train", train["ids"], "full_test", full_test["keys"]),
        ("validation", validation["ids"], "full_test", full_test["keys"]),
        ("train", train["ids"], "human_gold", human_gold["keys"]),
        ("validation", validation["ids"], "human_gold", human_gold["keys"]),
        ("full_test", full_test["keys"], "human_gold", human_gold["keys"]),
    ):
        overlap = left & right
        if overlap:
            fail(f"{left_name}/{right_name} id-task overlap: {next(iter(overlap))}")
    if combined["keys"] != full_test["keys"] | human_gold["keys"]:
        fail("combined eval is not exact full-test plus human-gold union")
    if train["origins"] & validation["origins"]:
        fail("train/validation normalized-origin overlap")
    return {
        "config_dir": str(resolved_config_dir),
        "data_contract_path": str(contract_path),
        "data_contract_sha256": sha256(contract_path),
        "prepared_data_dir": str(prepared),
        "train_rows": train["rows"],
        "validation_rows": validation["rows"],
        "full_test_rows": full_test["rows"],
        "human_gold_rows": human_gold["rows"],
        "combined_eval_rows": combined["rows"],
        "large_hashes_checked": not skip_large_hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=CONFIG_DIR,
        help=(
            "Explicit directory containing model.json, recipe.json, and "
            "data_contract.json (default: source project/config)."
        ),
    )
    parser.add_argument(
        "--skip-large-hashes",
        action="store_true",
        help="Skip hashes of prepared JSONL files; row and split checks still run.",
    )
    args = parser.parse_args()
    config_dir = _resolve_config_dir(args.config_dir)
    if sha256(PROJECT / "LICENSE") != LICENSE_SHA256:
        fail("LICENSE is not the pinned upstream Upstage Solar License")
    evaluator_identity = validate_evaluator_identity(config_dir)
    notice = (PROJECT / "NOTICE").read_text(encoding="utf-8")
    if "Built with Solar" not in notice:
        fail("NOTICE lacks required attribution")
    result = {
        "status": "PASS",
        "project": str(PROJECT),
        "config_dir": str(config_dir),
        "license_sha256": LICENSE_SHA256,
        "reference_evaluator_sha256": REFERENCE_SHA256,
        "reference_evaluator": evaluator_identity,
        "recipe": validate_recipe(config_dir),
        "data": validate_data(
            skip_large_hashes=args.skip_large_hashes,
            config_dir=config_dir,
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
