#!/usr/bin/env python3
"""Train the fixed Solar Open 2 hard-term LoRA recipe with resumable updates."""

from __future__ import annotations

import argparse
import copy
import getpass
import hashlib
import json
import math
import os
import random
import re
import shutil
import signal
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoTokenizer

from b200_runtime import (
    B200_CONFIG_DIR,
    VENV_DIR as B200_VENV_DIR,
    b200_config_identity,
    b200_runtime_override_evidence,
    b200_selective_gradient_checkpointing_contract,
    b200_training_performance_contract,
    load_b200_config_bundle,
    read_json_no_duplicates,
    require_b200_gpu_runtime,
    require_expected_user,
    require_owned_path as require_b200_owned_path,
    validate_b200_large_asset_receipt,
)
from common import (
    CONFIG_DIR,
    GlobalAugmentedDeckSampler,
    PROJECT,
    RESULT_LABEL_PREFIX_RE,
    TRAINING_RESULT_LABEL_IDS,
    atomic_json,
    atomic_text,
    canonical_sha256,
    encode_row,
    iter_jsonl,
    load_reference_module,
    output_only_violations,
    pipeline_code_identity,
    read_json,
    read_jsonl,
    require_owned_path,
    sha256,
)
from quantization.full_model_loader import (
    configure_model_parallel_gradient_checkpointing,
    configure_selective_model_parallel_gradient_checkpointing,
    disable_model_parallel_gradient_checkpointing,
    load_full_model,
)
from solar_peft_compat import solar_open2_peft_v5_native
from training_plan import (
    early_stopping_decision,
    group_update_rows,
    occurrences_in_step,
    occurrences_through_step,
    planned_microbatch_counts,
    positive_cosine_learning_rate,
    select_best_full_validation,
    token_length_contract,
    validate_b200_early_stopping_policy,
    validate_early_stopping_policy,
    validate_microbatch_buckets,
)


REQUEUE_EXIT_CODE = 75
RUNTIME_SAFETY_FACTOR = 1.25
ALLOCATION_LIMIT_SECONDS = 7 * 24 * 60 * 60
ALLOCATION_REQUEUE_RESERVE_SECONDS = 2 * 60 * 60
USABLE_ALLOCATION_SECONDS = (
    ALLOCATION_LIMIT_SECONDS - ALLOCATION_REQUEUE_RESERVE_SECONDS
)
STATE_CONTRACT_VERSION = 4
B200_STATE_CONTRACT_VERSION = 6
MINIMUM_FREE_STORAGE_BYTES = 16 * 1024**3
SMOKE_BUCKET_TIMED_REPETITIONS = 3
SMOKE_UPDATE_WARMUP_REPETITIONS = 2
SMOKE_UPDATE_TIMED_REPETITIONS = 10
B200_WANDB_ENTITY = "junhyuk-tak-markr"
B200_WANDB_PROJECT = "modern-korean-lm"
B200_PRODUCTION_WANDB_RUN_ID = "solar_open2_b200_1epoch_20260802_v2"
B200_PRODUCTION_WANDB_RUN_NAME = "Solar Open2 250B 2xB200 1epoch len9600 optimized v2"
B200_SMOKE_WANDB_RUN_ID = "solar_open2_b200_smoke_20260802_v2"
B200_SMOKE_WANDB_RUN_NAME = "Solar Open2 250B 2xB200 smoke len9600 optimized v2"


def load_training_config(
    config_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Load one explicitly selected source or sealed B200 config profile."""

    candidate = Path(config_dir).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    resolved = candidate.resolve(strict=True)
    source_config_dir = CONFIG_DIR.resolve(strict=True)
    if resolved == B200_CONFIG_DIR:
        bundle = load_b200_config_bundle(resolved)
        return {
            "profile": "b200",
            "config_dir": bundle["config_dir"],
            "paths": dict(bundle["paths"]),
            "hashes": dict(bundle["hashes"]),
            "identity": b200_config_identity(resolved),
            "model": bundle["model"],
            "recipe": bundle["recipe"],
            "data_contract": bundle["data_contract"],
        }
    if resolved != source_config_dir:
        raise RuntimeError(
            "training config must be explicitly selected as the source or "
            f"B200 profile: {resolved}"
        )
    paths = {
        "model": resolved / "model.json",
        "recipe": resolved / "recipe.json",
        "data_contract": resolved / "data_contract.json",
    }
    if any(not path.is_file() for path in paths.values()):
        raise RuntimeError(f"source config directory is incomplete: {resolved}")
    hashes = {key: sha256(path) for key, path in paths.items()}
    identity = {
        "config_dir": str(resolved),
        "files": {
            key: {"path": str(path), "sha256": hashes[key]}
            for key, path in paths.items()
        },
    }
    return {
        "profile": "source",
        "config_dir": resolved,
        "paths": paths,
        "hashes": hashes,
        "identity": identity,
        "model": read_json(paths["model"]),
        "recipe": read_json(paths["recipe"]),
        "data_contract": read_json(paths["data_contract"]),
    }


def training_pipeline_code_identity(
    training_config: dict[str, Any],
) -> dict[str, Any]:
    profile = training_config.get("profile")
    if profile == "b200":
        config_dir = Path(training_config["config_dir"]).resolve(strict=True)
        if config_dir != B200_CONFIG_DIR:
            raise RuntimeError(
                "B200 pipeline identity received a different config directory"
            )
        return pipeline_code_identity(config_dir=config_dir)
    if profile == "source":
        return pipeline_code_identity()
    raise RuntimeError(f"unsupported training config profile: {profile!r}")


def b200_wandb_identity(*, smoke: bool, mode: str) -> dict[str, str]:
    if mode not in {"online", "offline", "disabled"}:
        raise ValueError(f"invalid W&B mode: {mode!r}")
    identity = {
        "mode": mode,
        "entity": B200_WANDB_ENTITY,
        "project": B200_WANDB_PROJECT,
        "run_id": (B200_SMOKE_WANDB_RUN_ID if smoke else B200_PRODUCTION_WANDB_RUN_ID),
        "run_name": (
            B200_SMOKE_WANDB_RUN_NAME if smoke else B200_PRODUCTION_WANDB_RUN_NAME
        ),
        "resume": "allow",
    }
    if B200_SMOKE_WANDB_RUN_ID == B200_PRODUCTION_WANDB_RUN_ID:
        raise AssertionError("B200 smoke and production W&B IDs must differ")
    return identity


def training_occurrence_contract(recipe: dict[str, Any]) -> dict[str, int]:
    nominal = int(recipe["effective_batch_size"])
    total = int(recipe["training_deck"]["total_occurrences"]["total"])
    optimizer_steps = int(recipe["optimizer_steps"])
    terminal = occurrences_through_step(
        optimizer_steps,
        occurrences_per_step=nominal,
        total_occurrences=total,
    )
    if terminal != total:
        raise RuntimeError("optimizer horizon does not consume the exact deck")
    final_step_occurrences = occurrences_in_step(
        optimizer_steps,
        occurrences_per_step=nominal,
        total_occurrences=total,
    )
    exact_tail = recipe.get("exact_tail")
    if exact_tail is not None:
        expected_tail = {
            "total_occurrences": total,
            "nominal_step_occurrences": nominal,
            "full_steps": optimizer_steps - 1,
            "full_step_occurrences": nominal * (optimizer_steps - 1),
            "final_step": optimizer_steps,
            "final_step_occurrences": final_step_occurrences,
            "padding_or_duplication_allowed": False,
            "loss_denominator_policy": "occurrences_in_step",
            "gradient_denominator_policy": "occurrences_in_step",
        }
        observed_tail = {key: exact_tail.get(key) for key in expected_tail}
        if observed_tail != expected_tail:
            raise RuntimeError(
                "recipe exact-tail metadata differs from derived coverage: "
                f"{observed_tail!r} != {expected_tail!r}"
            )
    return {
        "nominal_step_occurrences": nominal,
        "total_occurrences": total,
        "optimizer_steps": optimizer_steps,
        "final_step_occurrences": final_step_occurrences,
    }


def consumed_occurrences_at_step(
    recipe: dict[str, Any],
    completed_step: int,
) -> int:
    contract = training_occurrence_contract(recipe)
    return occurrences_through_step(
        completed_step,
        occurrences_per_step=contract["nominal_step_occurrences"],
        total_occurrences=contract["total_occurrences"],
    )


def update_occurrences_at_step(recipe: dict[str, Any], step: int) -> int:
    contract = training_occurrence_contract(recipe)
    return occurrences_in_step(
        step,
        occurrences_per_step=contract["nominal_step_occurrences"],
        total_occurrences=contract["total_occurrences"],
    )


def update_occurrence_bounds(
    recipe: dict[str, Any],
    step: int,
) -> tuple[int, int]:
    end = consumed_occurrences_at_step(recipe, step)
    start = consumed_occurrences_at_step(recipe, step - 1)
    if end - start != update_occurrences_at_step(recipe, step):
        raise RuntimeError("optimizer-step occurrence bounds are inconsistent")
    return start, end


def configured_validation_scope(
    recipe: dict[str, Any],
    step: int,
    *,
    smoke: bool,
) -> str | None:
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"invalid validation step: {step!r}")
    if smoke:
        return "fixed_subset" if step == 1 else None
    validation = recipe["validation"]
    if step == 0:
        return "fixed_subset" if validation.get("step_zero_baseline") is True else None
    if step in {int(value) for value in validation["full_validation_steps"]}:
        return "full"
    if step % int(validation["eval_every_steps"]) == 0:
        return "fixed_subset"
    return None


def b200_coverage_decision(
    history: list[dict[str, Any]],
    *,
    completed_step: int,
    recipe: dict[str, Any],
    smoke: bool = False,
) -> dict[str, Any]:
    """Represent disabled quality stopping as an exact-coverage decision."""

    policy = validate_b200_early_stopping_policy(
        recipe["validation"]["early_stopping"],
        optimizer_steps=int(recipe["optimizer_steps"]),
        full_validation_steps=[
            int(step) for step in recipe["validation"]["full_validation_steps"]
        ],
    )
    contract = training_occurrence_contract(recipe)
    if (
        isinstance(completed_step, bool)
        or not isinstance(completed_step, int)
        or not 0 <= completed_step <= contract["optimizer_steps"]
    ):
        raise ValueError(f"invalid completed step: {completed_step!r}")
    identities: list[tuple[int, str]] = []
    full_steps: list[int] = []
    for record in history:
        if not isinstance(record, dict):
            raise ValueError("validation history records must be objects")
        step = record.get("step")
        scope = record.get("scope")
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or not 0 <= step <= completed_step
            or scope not in {"fixed_subset", "full"}
        ):
            raise ValueError(f"invalid B200 validation record: {record!r}")
        identity = (step, str(scope))
        expected_scope = configured_validation_scope(
            recipe,
            step,
            smoke=smoke,
        )
        if scope != expected_scope:
            raise ValueError(
                "B200 validation record differs from its configured scope: "
                f"{identity!r} expected={expected_scope!r}"
            )
        identities.append(identity)
        if scope == "full":
            full_steps.append(step)
    if identities != sorted(identities) or len(set(identities)) != len(identities):
        raise ValueError("B200 validation history must be ordered and unique")
    configured_identities = [
        (step, scope)
        for step in range(completed_step + 1)
        if (
            scope := configured_validation_scope(
                recipe,
                step,
                smoke=smoke,
            )
        )
        is not None
    ]
    missing_prior = [
        identity
        for identity in configured_identities
        if identity[0] < completed_step and identity not in identities
    ]
    if missing_prior:
        raise ValueError(
            "B200 validation history omits durable prior schedule entries: "
            f"{missing_prior}"
        )
    configured_full_steps = [
        int(step) for step in recipe["validation"]["full_validation_steps"]
    ]
    if any(step not in configured_full_steps for step in full_steps):
        raise ValueError("B200 history contains an unconfigured full validation")
    consumed = consumed_occurrences_at_step(recipe, completed_step)
    terminal_full_is_durable = contract["optimizer_steps"] in full_steps
    if completed_step < contract["optimizer_steps"]:
        status = "PENDING_COVERAGE"
        reason = "exact_epoch_coverage_not_reached"
    elif not terminal_full_is_durable:
        status = "PENDING_FINAL_VALIDATION"
        reason = "terminal_full_validation_not_durable"
    else:
        status = "MAXIMUM_REACHED"
        reason = "exact_epoch_coverage_reached"
    return {
        "schema_version": 1,
        "profile": "b200",
        "policy_sha256": canonical_sha256(policy),
        "status": status,
        "evaluated_step": completed_step,
        "triggered": False,
        "reason": reason,
        "quality_early_stopping_enabled": False,
        "completed_candidate_steps": full_steps,
        "completed_occurrences": consumed,
        "completed_epochs": (
            consumed / int(recipe["training_deck"]["occurrences_per_epoch"]["total"])
        ),
    }


def synchronize_cuda_devices() -> None:
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device_index)


def runtime_budget_projection(
    *,
    training_bucket_benchmarks: dict[str, dict[str, Any]],
    validation_bucket_benchmarks: dict[str, dict[str, Any]],
    production_update_benchmark: dict[str, Any],
    validation_row_seconds: float,
    optimizer_boundary_seconds: float,
    one_time_gradient_audit_seconds: float,
    recipe: dict[str, Any],
    validation_row_count: int,
    production_training_plan: dict[str, Any],
    subset_validation_plan: dict[str, Any],
    full_validation_plan: dict[str, Any],
) -> dict[str, Any]:
    for label, value in (
        ("validation_row_seconds", validation_row_seconds),
        ("optimizer_boundary_seconds", optimizer_boundary_seconds),
        ("one_time_gradient_audit_seconds", one_time_gradient_audit_seconds),
    ):
        if not math.isfinite(value) or value <= 0:
            raise RuntimeError(f"{label} must be finite and positive: {value!r}")

    occurrence_contract = training_occurrence_contract(recipe)
    optimizer_steps = occurrence_contract["optimizer_steps"]
    rows_per_update = occurrence_contract["nominal_step_occurrences"]
    eval_every = int(recipe["validation"]["eval_every_steps"])
    subset_rows = 2 * int(recipe["validation"]["subset_per_task"])
    full_steps = {int(step) for step in recipe["validation"]["full_validation_steps"]}
    if not full_steps or max(full_steps) != optimizer_steps:
        raise RuntimeError("final optimizer step must be a full validation candidate")
    subset_steps = [
        *([0] if recipe["validation"].get("step_zero_baseline") is True else []),
        *[
            step
            for step in range(eval_every, optimizer_steps + 1, eval_every)
            if step not in full_steps
        ],
    ]
    subset_events = len(subset_steps)
    total_validation_rows = (
        subset_events * subset_rows + len(full_steps) * validation_row_count
    )

    expected_occurrences = occurrence_contract["total_occurrences"]
    if production_training_plan.get("optimizer_updates") != optimizer_steps:
        raise RuntimeError("training plan optimizer-step count differs from recipe")
    if production_training_plan.get("occurrences") != expected_occurrences:
        raise RuntimeError("training plan occurrence count differs from recipe")
    if expected_occurrences != consumed_occurrences_at_step(recipe, optimizer_steps):
        raise RuntimeError("recipe optimizer steps do not consume the exact deck")
    if subset_validation_plan.get("occurrences") != subset_rows:
        raise RuntimeError("subset validation plan row count differs from recipe")
    if full_validation_plan.get("occurrences") != validation_row_count:
        raise RuntimeError("full validation plan row count differs from data")

    training_benchmark_seconds: dict[str, float] = {}
    validation_benchmark_seconds: dict[str, float] = {}
    expected_bucket_keys = {
        str(bucket["max_length_inclusive"])
        for bucket in recipe["microbatch_policy"]["buckets"]
    }
    for label, benchmarks, destination in (
        (
            "training",
            training_bucket_benchmarks,
            training_benchmark_seconds,
        ),
        (
            "validation",
            validation_bucket_benchmarks,
            validation_benchmark_seconds,
        ),
    ):
        if set(benchmarks) != expected_bucket_keys:
            raise RuntimeError(
                f"{label} benchmark buckets differ from recipe: "
                f"{sorted(benchmarks)} != {sorted(expected_bucket_keys)}"
            )
        for bucket in recipe["microbatch_policy"]["buckets"]:
            key = str(bucket["max_length_inclusive"])
            benchmark = benchmarks[key]
            seconds = float(benchmark["seconds"])
            if not math.isfinite(seconds) or seconds <= 0:
                raise RuntimeError(
                    f"{label} bucket {key} has invalid benchmark time: " f"{seconds}"
                )
            if int(benchmark["batch_size"]) != int(bucket["target_micro_batch_size"]):
                raise RuntimeError(
                    f"{label} bucket {key} benchmark batch size differs " "from recipe"
                )
            destination[key] = seconds

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
        or production_update_benchmark.get("estimator") != "median_p50"
        or production_update_benchmark.get("benchmark_state_unchanged") is not True
    ):
        raise RuntimeError("production update benchmark contract changed")

    bucket_microbatches = production_training_plan["bucket_microbatches"]
    raw_train_seconds = sum(
        int(bucket_microbatches[key]) * training_benchmark_seconds[key]
        for key in expected_bucket_keys
    )
    subset_validation_seconds = sum(
        int(subset_validation_plan["bucket_microbatches"][key])
        * validation_benchmark_seconds[key]
        for key in expected_bucket_keys
    )
    full_validation_seconds = sum(
        int(full_validation_plan["bucket_microbatches"][key])
        * validation_benchmark_seconds[key]
        for key in expected_bucket_keys
    )
    raw_validation_seconds = (
        subset_events * subset_validation_seconds
        + len(full_steps) * full_validation_seconds
    )
    raw_boundary_seconds = optimizer_boundary_seconds * optimizer_steps
    raw_one_time_audit_seconds = one_time_gradient_audit_seconds
    raw_total_seconds = (
        raw_train_seconds
        + raw_validation_seconds
        + raw_boundary_seconds
        + raw_one_time_audit_seconds
    )
    projected_total_seconds = raw_total_seconds * RUNTIME_SAFETY_FACTOR
    projected_full_validation_seconds = full_validation_seconds * RUNTIME_SAFETY_FACTOR
    projected_single_microbatch_seconds = (
        max(training_benchmark_seconds.values()) * RUNTIME_SAFETY_FACTOR
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
    total_validation_microbatches = subset_events * int(
        subset_validation_plan["microbatches"]
    ) + len(full_steps) * int(full_validation_plan["microbatches"])
    if recipe.get("profile") == "b200":
        early_policy = validate_b200_early_stopping_policy(
            recipe["validation"]["early_stopping"],
            optimizer_steps=optimizer_steps,
            full_validation_steps=sorted(full_steps),
        )
        earliest_stop_step = optimizer_steps
    else:
        early_policy = validate_early_stopping_policy(
            recipe["validation"]["early_stopping"],
            optimizer_steps=optimizer_steps,
            full_validation_steps=sorted(full_steps),
        )
        earliest_stop_step = int(early_policy["earliest_stop_step"])
    early_subset_steps = [step for step in subset_steps if step <= earliest_stop_step]
    early_full_steps = [
        step for step in sorted(full_steps) if step <= earliest_stop_step
    ]
    if recipe.get("profile") == "b200":
        if early_full_steps != sorted(full_steps):
            raise RuntimeError("B200 full-validation schedule changed")
    elif early_full_steps != [
        int(early_policy["minimum_step"]),
        int(early_policy["confirmation_step"]),
    ]:
        raise RuntimeError("early-stop full-validation schedule changed")
    early_validation_microbatches = len(early_subset_steps) * int(
        subset_validation_plan["microbatches"]
    ) + len(early_full_steps) * int(full_validation_plan["microbatches"])
    early_validation_seconds = (
        len(early_subset_steps) * subset_validation_seconds
        + len(early_full_steps) * full_validation_seconds
    )
    early_boundary_seconds = optimizer_boundary_seconds * earliest_stop_step
    early_raw_p50_seconds = (
        update_p50_seconds * earliest_stop_step
        + early_validation_seconds
        + early_boundary_seconds
        + raw_one_time_audit_seconds
    )
    early_raw_p90_seconds = (
        update_p90_seconds * earliest_stop_step
        + early_validation_seconds
        + early_boundary_seconds
        + raw_one_time_audit_seconds
    )
    early_projected_p50_seconds = early_raw_p50_seconds * RUNTIME_SAFETY_FACTOR
    early_projected_p90_seconds = early_raw_p90_seconds * RUNTIME_SAFETY_FACTOR
    early_allocation_count = max(
        1,
        math.ceil(early_projected_p50_seconds / USABLE_ALLOCATION_SECONDS),
    )
    early_last_allocation_seconds = early_projected_p50_seconds - (
        (early_allocation_count - 1) * USABLE_ALLOCATION_SECONDS
    )
    saved_optimizer_updates = optimizer_steps - earliest_stop_step
    early_stopping_projection = {
        "enabled": bool(early_policy["enabled"]),
        "earliest_stop_step": earliest_stop_step,
        "maximum_step": optimizer_steps,
        "saved_optimizer_updates": saved_optimizer_updates,
        "maximum_training_fraction_saved": (saved_optimizer_updates / optimizer_steps),
        "subset_validation_steps": early_subset_steps,
        "subset_validation_evaluations": len(early_subset_steps),
        "full_validation_steps": early_full_steps,
        "full_validation_evaluations": len(early_full_steps),
        "projected_validation_microbatches": early_validation_microbatches,
        "raw_validation_seconds": early_validation_seconds,
        "raw_optimizer_boundary_seconds": early_boundary_seconds,
        "raw_one_time_gradient_audit_seconds": raw_one_time_audit_seconds,
        "raw_empirical_p50_total_seconds": early_raw_p50_seconds,
        "raw_empirical_p90_total_seconds": early_raw_p90_seconds,
        "projected_empirical_p50_total_seconds": (early_projected_p50_seconds),
        "projected_empirical_p90_total_seconds": (early_projected_p90_seconds),
        "projected_empirical_p50_total_hours": (early_projected_p50_seconds / 3_600),
        "projected_empirical_p90_total_hours": (early_projected_p90_seconds / 3_600),
        "allocation": {
            "projected_p50_allocation_count": early_allocation_count,
            "projected_p50_last_allocation_seconds": (early_last_allocation_seconds),
            "projected_p50_last_allocation_headroom_seconds": (
                USABLE_ALLOCATION_SECONDS - early_last_allocation_seconds
            ),
        },
    }
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
        "early_stopping_projection": early_stopping_projection,
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


def atomic_torch_save(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def expected_b200_production_approval(
    *,
    smoke_receipt_path: Path,
    smoke_receipt_sha256: str,
    training_config: dict[str, Any],
    pipeline_identity: dict[str, Any],
    output_dir: Path,
    wandb_identity: dict[str, Any],
) -> dict[str, Any]:
    if training_config.get("profile") != "b200":
        raise RuntimeError("production approval is only defined for B200")
    return {
        "schema": "solar-open2-b200-production-approval",
        "schema_version": 2,
        "status": "PASS",
        "authorization_scope": "b200_production_training",
        "human_production_approval_recorded": True,
        "technical_smoke_receipt": {
            "path": str(smoke_receipt_path),
            "sha256": smoke_receipt_sha256,
        },
        "config_identity": training_config["identity"],
        "pipeline_code_sha256": pipeline_identity["sha256"],
        "model_contract_sha256": training_config["hashes"]["model"],
        "recipe_sha256": training_config["hashes"]["recipe"],
        "data_contract_sha256": training_config["hashes"]["data_contract"],
        "training_performance": training_config["recipe"]["training_performance"],
        "training_performance_sha256": canonical_sha256(
            training_config["recipe"]["training_performance"]
        ),
        "gradient_checkpointing_contract_sha256": canonical_sha256(
            b200_selective_gradient_checkpointing_contract()
        ),
        "output_dir": str(output_dir),
        "wandb": wandb_identity,
        "runtime_override": b200_runtime_override_evidence(),
    }


def validate_b200_production_approval_value(
    receipt: dict[str, Any],
    *,
    smoke_receipt_path: Path,
    smoke_receipt_sha256: str,
    training_config: dict[str, Any],
    pipeline_identity: dict[str, Any],
    output_dir: Path,
    wandb_identity: dict[str, Any],
) -> dict[str, Any]:
    expected = expected_b200_production_approval(
        smoke_receipt_path=smoke_receipt_path,
        smoke_receipt_sha256=smoke_receipt_sha256,
        training_config=training_config,
        pipeline_identity=pipeline_identity,
        output_dir=output_dir,
        wandb_identity=wandb_identity,
    )
    if not isinstance(receipt, dict) or set(receipt) != set(expected):
        raise RuntimeError(
            "production approval receipt fields changed: "
            f"{sorted(receipt) if isinstance(receipt, dict) else receipt!r}"
        )
    for key, expected_value in expected.items():
        if receipt.get(key) != expected_value:
            raise RuntimeError(
                "production approval receipt mismatch: "
                f"{key}={receipt.get(key)!r}, expected={expected_value!r}"
            )
    return receipt


def validate_b200_production_approval(
    path: Path,
    *,
    smoke_receipt_path: Path,
    smoke_receipt_sha256: str,
    training_config: dict[str, Any],
    pipeline_identity: dict[str, Any],
    output_dir: Path,
    wandb_identity: dict[str, Any],
) -> dict[str, Any]:
    approval_path = require_b200_owned_path(path, must_exist=True)
    if approval_path == smoke_receipt_path:
        raise RuntimeError(
            "production approval must be separate from the smoke receipt"
        )
    if approval_path.is_symlink() or not approval_path.is_file():
        raise RuntimeError(
            f"production approval must be a regular file: {approval_path}"
        )
    if approval_path.stat().st_mtime_ns <= (smoke_receipt_path.stat().st_mtime_ns):
        raise RuntimeError(
            "production approval receipt is not newer than its smoke receipt"
        )
    receipt = read_json_no_duplicates(approval_path)
    return validate_b200_production_approval_value(
        receipt,
        smoke_receipt_path=smoke_receipt_path,
        smoke_receipt_sha256=smoke_receipt_sha256,
        training_config=training_config,
        pipeline_identity=pipeline_identity,
        output_dir=output_dir,
        wandb_identity=wandb_identity,
    )


def validate_preflight(
    path: Path,
    *,
    source_dir: Path,
    expert_nf4_dir: Path,
    training_config: dict[str, Any],
) -> dict[str, Any]:
    receipt = read_json(path)
    if not isinstance(receipt, dict):
        raise RuntimeError("preflight receipt is not an object")
    b200 = training_config["profile"] == "b200"
    selected_pipeline_identity = training_pipeline_code_identity(training_config)
    if b200:
        model = training_config["model"]
        hardware_contract = model["hardware"]
        wanted = {
            "schema": "solar_open2_b200_preflight",
            "schema_version": 1,
            "status": "PASS",
            "profile": "b200",
            "user": model["expected_user"],
            "root": model["owned_root"],
            "config": training_config["identity"],
            "pipeline_code": selected_pipeline_identity,
            "physical_cuda_visible_devices": hardware_contract[
                "physical_cuda_visible_devices"
            ],
            "production_authorized": False,
        }
        for key, expected in wanted.items():
            if receipt.get(key) != expected:
                raise RuntimeError(
                    f"B200 preflight receipt mismatch: {key}="
                    f"{receipt.get(key)!r}, expected={expected!r}"
                )

        def nested_object(key: str) -> dict[str, Any]:
            value = receipt.get(key)
            if not isinstance(value, dict):
                raise RuntimeError(f"B200 preflight {key} is not an object")
            return value

        bf16 = nested_object("bf16")
        for key, expected in (
            ("status", "PASS"),
            ("relocated_snapshot_dir", str(source_dir)),
            ("revision", model["base"]["revision"]),
            ("indexed_shards", model["base"]["weight_shard_count"]),
            ("indexed_tensors", model["base"]["indexed_tensor_count"]),
            ("relocation_verified_by_content_hash", True),
        ):
            if bf16.get(key) != expected:
                raise RuntimeError(
                    f"B200 preflight BF16 mismatch: {key}={bf16.get(key)!r}, "
                    f"expected={expected!r}"
                )
        training_nf4 = nested_object("training_nf4")
        for key, expected in (
            ("status", "PASS"),
            ("role", "training"),
            ("path", str(expert_nf4_dir)),
            (
                "schema",
                model["training_runtime"]["expert_artifact_schema"],
            ),
            ("payload_file_count", 48),
            ("payload_hashes_inherited_from_full_transfer_receipt", True),
        ):
            if training_nf4.get(key) != expected:
                raise RuntimeError(
                    f"B200 preflight training NF4 mismatch: {key}="
                    f"{training_nf4.get(key)!r}, expected={expected!r}"
                )
        manifest = training_nf4.get("manifest")
        if not isinstance(manifest, dict) or manifest.get("status") != "SEALED":
            raise RuntimeError("B200 preflight training NF4 manifest is not sealed")
        environment = nested_object("environment")
        dependency_lock = environment.get("dependency_lock")
        expected_lock_path = PROJECT / "requirements/solar-open-2.lock.txt"
        if not isinstance(dependency_lock, dict):
            raise RuntimeError("B200 preflight dependency lock is missing")
        for key, expected in (
            ("path", str(expected_lock_path)),
            ("sha256", sha256(expected_lock_path)),
        ):
            if dependency_lock.get(key) != expected:
                raise RuntimeError(
                    f"B200 preflight dependency lock mismatch: {key}="
                    f"{dependency_lock.get(key)!r}, expected={expected!r}"
                )
        transfer = nested_object("transfer_integrity")
        if transfer.get("large_hashes_checked") is not True:
            raise RuntimeError("B200 preflight lacks full transfer hash evidence")
        hardware = nested_object("hardware")
        for key, expected in (
            ("status", "PASS"),
            ("logical_gpu_count", 2),
            (
                "physical_cuda_visible_devices",
                hardware_contract["physical_cuda_visible_devices"],
            ),
            ("boundary_layers", hardware_contract["boundary_layers"]),
        ):
            if hardware.get(key) != expected:
                raise RuntimeError(
                    f"B200 preflight hardware mismatch: {key}="
                    f"{hardware.get(key)!r}, expected={expected!r}"
                )
        devices = hardware.get("logical_to_physical")
        if not isinstance(devices, list) or len(devices) != 2:
            raise RuntimeError("B200 preflight GPU mapping is invalid")
        expected_capability = hardware_contract["required_compute_capability"]
        for logical_index, device in enumerate(devices):
            if not isinstance(device, dict):
                raise RuntimeError("B200 preflight GPU record is invalid")
            for key, expected in (
                ("logical_index", logical_index),
                (
                    "logical_device",
                    hardware_contract["logical_devices"][logical_index],
                ),
                (
                    "physical_index",
                    hardware_contract["physical_gpu_indices"][logical_index],
                ),
                ("name", hardware_contract["required_gpu_name"]),
                (
                    "compute_capability",
                    [expected_capability["major"], expected_capability["minor"]],
                ),
            ):
                if device.get(key) != expected:
                    raise RuntimeError(
                        f"B200 preflight GPU {logical_index} mismatch: {key}="
                        f"{device.get(key)!r}, expected={expected!r}"
                    )
            total_memory = device.get("total_memory_bytes")
            if (
                isinstance(total_memory, bool)
                or not isinstance(total_memory, int)
                or total_memory < int(hardware_contract["minimum_memory_bytes_per_gpu"])
            ):
                raise RuntimeError(
                    f"B200 preflight GPU {logical_index} memory is insufficient"
                )
        return receipt

    wanted = {
        "status": "PASS",
        "user": "mk03",
        "project": str(PROJECT),
        "source_dir": str(source_dir),
        "expert_nf4_dir": str(expert_nf4_dir),
        "model_contract_sha256": training_config["hashes"]["model"],
        "recipe_sha256": training_config["hashes"]["recipe"],
        "data_contract_sha256": training_config["hashes"]["data_contract"],
        "pipeline_code": selected_pipeline_identity,
        "environment_lock_sha256": sha256(
            PROJECT / "requirements/solar-open-2.lock.txt"
        ),
    }
    for key, expected in wanted.items():
        if receipt.get(key) != expected:
            raise RuntimeError(
                f"preflight receipt mismatch: {key}={receipt.get(key)!r}, "
                f"expected={expected!r}"
            )
    expected_gpu_count = 3
    if receipt["hardware"].get("device_count") != expected_gpu_count:
        raise RuntimeError(
            "preflight validated the wrong GPU count: "
            f"{receipt['hardware'].get('device_count')} != {expected_gpu_count}"
        )
    return receipt


def preflight_state_hashes(
    receipt: dict[str, Any],
    *,
    b200: bool,
) -> dict[str, str]:
    if b200:
        bf16 = receipt.get("bf16")
        training_nf4 = receipt.get("training_nf4")
        manifest = (
            training_nf4.get("manifest") if isinstance(training_nf4, dict) else None
        )
        values = {
            "download_receipt_sha256": (
                bf16.get("receipt_sha256") if isinstance(bf16, dict) else None
            ),
            "expert_manifest_sha256": (
                manifest.get("sha256") if isinstance(manifest, dict) else None
            ),
        }
    else:
        download = receipt.get("download")
        expert_artifact = receipt.get("expert_artifact")
        values = {
            "download_receipt_sha256": (
                download.get("sha256") if isinstance(download, dict) else None
            ),
            "expert_manifest_sha256": (
                expert_artifact.get("manifest_sha256")
                if isinstance(expert_artifact, dict)
                else None
            ),
        }
    for key, value in values.items():
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RuntimeError(f"preflight state hash is invalid: {key}={value!r}")
    return values


def validate_production_smoke(
    path: Path,
    *,
    source_dir: Path,
    expert_nf4_dir: Path,
    training_config: dict[str, Any],
) -> dict[str, Any]:
    receipt = read_json(path)
    b200 = training_config["profile"] == "b200"
    selected_pipeline_identity = training_pipeline_code_identity(training_config)
    wanted = {
        "status": "PASS",
        "technical_production_gate_passed": True,
        "authorization_scope": "technical_smoke_gate_only",
        "human_production_approval_required": True,
        "human_production_approval_recorded": False,
        "source_dir": str(source_dir),
        "expert_nf4_dir": str(expert_nf4_dir),
        "model_contract_sha256": training_config["hashes"]["model"],
        "recipe_sha256": training_config["hashes"]["recipe"],
        "data_contract_sha256": training_config["hashes"]["data_contract"],
        "pipeline_code": selected_pipeline_identity,
        "environment_lock_sha256": sha256(
            PROJECT / "requirements/solar-open-2.lock.txt"
        ),
    }
    if b200:
        wanted["config_identity"] = training_config["identity"]
    for key, expected in wanted.items():
        if receipt.get(key) != expected:
            raise RuntimeError(
                f"smoke receipt mismatch: {key}={receipt.get(key)!r}, "
                f"expected={expected!r}"
            )
    checks = receipt.get("checks") or {}
    required = (
        "model_preflight",
        "tokenizer_output_contract",
        "max_length_backward",
        "lora_module_contract",
        "no_cpu_disk_offload",
        "base_inference",
        "lora_inference",
        "current_code_and_environment",
        "source_and_expert_provenance",
        "finite_nonzero_lora_gradient",
        "lora_update_and_frozen_base",
        "device_aware_gradient_checkpointing",
        "post_update_checkpoint_recompute",
        "output_only_supervision",
        "sealed_prediction_contracts",
        "runtime_budget_feasibility",
        "early_stopping_policy",
        "early_stopping_decision_recomputed",
    )
    if b200:
        required += (
            "selective_training_activation_policy",
            "exact_tail_transaction",
            "longest_occurrence_update",
        )
    failed = [name for name in required if checks.get(name) != "PASS"]
    if failed:
        raise RuntimeError(f"smoke receipt lacks production checks: {failed}")
    recipe = training_config["recipe"]
    if b200:
        expected_performance = b200_training_performance_contract()
        expected_activation = b200_selective_gradient_checkpointing_contract()
        if receipt.get("training_performance") != expected_performance:
            raise RuntimeError("smoke receipt training-performance policy changed")
        if receipt.get("training_performance_sha256") != canonical_sha256(
            expected_performance
        ):
            raise RuntimeError("smoke receipt training-performance SHA changed")
        if receipt.get("gradient_checkpointing_contract") != expected_activation:
            raise RuntimeError("smoke receipt effective checkpoint contract changed")
        if receipt.get("gradient_checkpointing_contract_sha256") != canonical_sha256(
            expected_activation
        ):
            raise RuntimeError("smoke receipt effective checkpoint SHA changed")
    if receipt.get("early_stopping_policy") != recipe["validation"]["early_stopping"]:
        raise RuntimeError("smoke receipt early-stopping policy changed")
    receipt_history = receipt.get("early_stopping_history")
    if not isinstance(receipt_history, list) or not receipt_history:
        raise RuntimeError("smoke receipt lacks early-stopping history")
    if b200:
        expected_decision = b200_coverage_decision(
            receipt_history,
            completed_step=1,
            recipe=recipe,
            smoke=True,
        )
    else:
        expected_decision = early_stopping_decision(
            receipt_history,
            completed_step=1,
            policy=recipe["validation"]["early_stopping"],
            optimizer_steps=int(recipe["optimizer_steps"]),
            full_validation_steps=[
                int(step) for step in recipe["validation"]["full_validation_steps"]
            ],
            occurrences_per_step=int(recipe["effective_batch_size"]),
            occurrences_per_epoch=int(
                recipe["training_deck"]["occurrences_per_epoch"]["total"]
            ),
            total_occurrences=int(
                recipe["training_deck"]["total_occurrences"]["total"]
            ),
        )
    if (
        expected_decision["status"]
        != ("PENDING_COVERAGE" if b200 else "PENDING_MINIMUM")
        or expected_decision["triggered"] is not False
        or receipt.get("early_stopping_decision") != expected_decision
    ):
        raise RuntimeError("smoke receipt early-stopping decision changed")
    tokenizer_receipt = receipt.get("tokenizer_contract_receipt")
    if not isinstance(tokenizer_receipt, dict) or set(tokenizer_receipt) != {
        "path",
        "sha256",
    }:
        raise RuntimeError("smoke receipt lacks tokenizer receipt identity")
    tokenizer_path = (
        require_b200_owned_path(
            Path(str(tokenizer_receipt["path"])),
            must_exist=True,
        )
        if b200
        else require_owned_path(Path(str(tokenizer_receipt["path"])))
    )
    if tokenizer_path.is_symlink() or not tokenizer_path.is_file():
        raise RuntimeError("tokenizer contract receipt is missing or symlinked")
    if sha256(tokenizer_path) != tokenizer_receipt["sha256"]:
        raise RuntimeError("tokenizer contract receipt changed after smoke")
    runtime_budget = receipt.get("runtime_budget_projection")
    if (
        not isinstance(runtime_budget, dict)
        or (runtime_budget.get("feasibility") or {}).get("status") != "PASS"
    ):
        raise RuntimeError("smoke receipt lacks a feasible runtime budget projection")
    return receipt


def model_device_audit(model: Any) -> dict[str, Any]:
    device_map = getattr(model, "hf_device_map", None)
    if not isinstance(device_map, dict) or not device_map:
        raise RuntimeError("model loader did not expose hf_device_map")
    forbidden = {
        str(value)
        for value in device_map.values()
        if str(value).lower() in {"cpu", "disk", "meta"}
    }
    if forbidden:
        raise RuntimeError(f"CPU/disk/meta model offload is forbidden: {forbidden}")
    parameter_devices = Counter(
        str(parameter.device) for parameter in model.parameters()
    )
    bad_parameters = {
        device: count
        for device, count in parameter_devices.items()
        if not device.startswith("cuda")
    }
    if bad_parameters:
        raise RuntimeError(f"non-CUDA model parameters found: {bad_parameters}")
    return {
        "hf_device_map_devices": sorted({str(value) for value in device_map.values()}),
        "parameter_devices": dict(parameter_devices),
        "cpu_disk_offload": False,
    }


def lora_module_audit(model: Any, architecture: dict[str, Any]) -> dict[str, Any]:
    from common import expected_lora_module_names

    expected = expected_lora_module_names(architecture)
    modules = dict(model.named_modules())
    missing = sorted(expected - set(modules))
    if missing:
        raise RuntimeError(f"missing LoRA targets: {missing[:8]}")
    wrong_type = {
        name: type(modules[name]).__name__
        for name in expected
        if not isinstance(modules[name], torch.nn.Linear)
    }
    if wrong_type:
        raise RuntimeError(
            "LoRA targets must be uncompressed torch.nn.Linear modules: "
            f"{list(wrong_type.items())[:8]}"
        )
    return {
        "status": "PASS",
        "module_count": len(expected),
        "module_names_sha256": canonical_sha256(sorted(expected)),
        "module_class": "torch.nn.Linear",
    }


def clip_global_grad_norm(
    parameters: list[torch.nn.Parameter], max_norm: float
) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            value = parameter.grad.detach().float().norm(2)
            squared += float(value.item()) ** 2
    total = math.sqrt(squared)
    if not math.isfinite(total) or total <= 0.0:
        raise RuntimeError(f"invalid global LoRA gradient norm: {total!r}")
    coefficient = min(1.0, max_norm / (total + 1e-6))
    if coefficient < 1.0:
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(coefficient)
    return total


@torch.no_grad()
def trainable_parameter_stats(model: Any) -> dict[str, Any]:
    parameter_count = 0
    element_count = 0
    dtype_parameter_counts: Counter[str] = Counter()
    dtype_element_counts: Counter[str] = Counter()
    finite = True
    l1 = 0.0
    squared_l2 = 0.0
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        values = parameter.detach().float()
        parameter_count += 1
        element_count += parameter.numel()
        dtype = str(parameter.dtype)
        dtype_parameter_counts[dtype] += 1
        dtype_element_counts[dtype] += parameter.numel()
        finite = finite and bool(torch.isfinite(values).all().item())
        l1 += float(values.abs().sum().item())
        squared_l2 += float(values.square().sum().item())
    return {
        "parameter_count": parameter_count,
        "element_count": element_count,
        "dtype_parameter_counts": dict(dtype_parameter_counts),
        "dtype_element_counts": dict(dtype_element_counts),
        "finite": finite,
        "l1": l1,
        "l2": math.sqrt(squared_l2),
    }


@torch.no_grad()
def trainable_parameter_digest(model: Any) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        raw = parameter.detach().contiguous().view(torch.uint8).cpu().numpy()
        digest.update(memoryview(raw))
    return digest.hexdigest()


@torch.no_grad()
def frozen_parameter_probe(model: Any, *, limit: int = 32) -> dict[str, Any]:
    values: dict[str, list[float]] = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad or parameter.numel() == 0:
            continue
        flat = parameter.detach().reshape(-1)
        values[name] = [
            float(flat[0].float().item()),
            float(flat[-1].float().item()),
        ]
        if len(values) >= limit:
            break
    if len(values) != limit:
        raise RuntimeError(f"unable to sample {limit} frozen base parameters")
    return {
        "sampled_parameter_count": len(values),
        "values_sha256": canonical_sha256(values),
    }


def gradient_audit(parameters: list[torch.nn.Parameter]) -> dict[str, Any]:
    with_gradient = 0
    nonzero = 0
    finite = True
    squared_l2 = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        with_gradient += 1
        finite = finite and bool(torch.isfinite(gradient).all().item())
        nonzero += int(bool(torch.count_nonzero(gradient).item()))
        squared_l2 += float(gradient.square().sum().item())
    result = {
        "parameters_with_gradient": with_gradient,
        "nonzero_gradient_elements": nonzero,
        "finite": finite,
        "l2": math.sqrt(squared_l2),
    }
    if (
        not finite
        or not math.isfinite(result["l2"])
        or result["l2"] <= 0.0
        or with_gradient == 0
        or nonzero == 0
    ):
        raise RuntimeError(f"invalid LoRA gradient audit: {result}")
    return result


def scheduler_warmup_steps(
    *,
    smoke: bool,
    warmup_ratio: float,
    total_steps: int,
) -> int:
    if total_steps < 1:
        raise ValueError("scheduler total_steps must be positive")
    if smoke:
        return 0
    return max(1, int(warmup_ratio * total_steps))


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    required = {"id", "task", "origin", "response", "quality", "prompt_variant"}
    result_label_ids: set[str] = set()
    for index, row in enumerate(rows, 1):
        missing = required - set(row)
        if missing:
            raise ValueError(f"{path}:{index}: missing fields {sorted(missing)}")
        if row["prompt_variant"] != "no_dictionary_context":
            raise ValueError(f"{path}:{index}: wrong prompt variant")
        if RESULT_LABEL_PREFIX_RE.match(str(row["response"])):
            result_label_ids.add(str(row["id"]))
        normalized_response = RESULT_LABEL_PREFIX_RE.sub(
            "", str(row["response"]), count=1
        )
        violations = output_only_violations(normalized_response)
        if violations:
            raise ValueError(f"{path}:{index}: non-body training target {violations}")
    expected_result_label_ids = (
        set(TRAINING_RESULT_LABEL_IDS) if path.name == "train.jsonl" else set()
    )
    if result_label_ids != expected_result_label_ids:
        raise ValueError(
            f"{path}: output-only target exceptions changed: "
            f"{sorted(result_label_ids)} != {sorted(expected_result_label_ids)}"
        )
    return rows


def stratified_validation_subset(
    rows: list[dict[str, Any]],
    *,
    per_task: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["task"]].append(row)
    selected = []
    for task in ("translation", "summarization"):
        task_rows = list(grouped[task])
        rng.shuffle(task_rows)
        selected.extend(task_rows[:per_task])
    return selected


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--expert-nf4-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-receipt", type=Path, required=True)
    parser.add_argument("--asset-integrity-receipt", type=Path)
    parser.add_argument("--smoke-receipt", type=Path)
    parser.add_argument("--production-approval-receipt", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"))
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()

    training_config = load_training_config(args.config_dir)
    b200 = training_config["profile"] == "b200"
    recipe = training_config["recipe"]
    model_contract = training_config["model"]
    data_contract = training_config["data_contract"]
    occurrence_contract = training_occurrence_contract(recipe)
    selected_pipeline_identity = training_pipeline_code_identity(training_config)

    def resolve_run_path(
        path: str | os.PathLike[str],
        *,
        must_exist: bool = False,
    ) -> Path:
        if b200:
            return require_b200_owned_path(path, must_exist=must_exist)
        resolved_path = require_owned_path(Path(path))
        if must_exist and not resolved_path.exists():
            raise FileNotFoundError(resolved_path)
        return resolved_path

    if b200:
        require_expected_user()
        gpu_runtime_audit = require_b200_gpu_runtime(torch_module=torch)
    else:
        if getpass.getuser() != "mk03":
            raise RuntimeError("Solar source training is restricted to mk03")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 3:
            raise RuntimeError(
                "exactly three visible source GPUs required, "
                f"found={torch.cuda.device_count()}"
            )
        gpu_runtime_audit = {
            "status": "PASS",
            "logical_gpu_count": 3,
            "profile": "source",
        }
    wandb_mode = args.wandb_mode or (
        "disabled" if args.smoke else os.environ.get("WANDB_MODE", "online")
    )
    if b200:
        wandb_identity = b200_wandb_identity(
            smoke=args.smoke,
            mode=wandb_mode,
        )
        expected_wandb_environment = {
            "WANDB_ENTITY": wandb_identity["entity"],
            "WANDB_PROJECT": wandb_identity["project"],
            "WANDB_RUN_ID": wandb_identity["run_id"],
            "WANDB_NAME": wandb_identity["run_name"],
        }
        conflicting_wandb_environment = {
            key: {"observed": os.environ[key], "expected": expected}
            for key, expected in expected_wandb_environment.items()
            if key in os.environ and os.environ[key] != expected
        }
        if conflicting_wandb_environment:
            raise RuntimeError(
                "B200 W&B identity environment changed: "
                f"{conflicting_wandb_environment}"
            )
    else:
        wandb_identity = {
            "mode": wandb_mode,
            "entity": os.environ.get(
                "WANDB_ENTITY",
                "junhyuk-tak-markr",
            ),
            "project": os.environ.get(
                "WANDB_PROJECT",
                "modern-korean-lm",
            ),
            "run_id": os.environ.get(
                "WANDB_RUN_ID",
                "solar_open2_hardterm_2epoch_earlystop_20260727_v2",
            ),
            "run_name": os.environ.get(
                "WANDB_NAME",
                "Solar-Open2-250B hard-term translation-summary "
                "2epoch earlystop step18631 maxlen9600 v2",
            ),
            "resume": "allow",
        }
    if any(
        not isinstance(wandb_identity[key], str) or not wandb_identity[key].strip()
        for key in ("entity", "project", "run_id", "run_name")
    ):
        raise RuntimeError(f"invalid W&B identity: {wandb_identity!r}")
    if not args.smoke and wandb_mode != "online":
        raise RuntimeError("production training requires online W&B reporting")
    expected_environment = (
        B200_VENV_DIR.resolve() if b200 else (PROJECT / "solar-open-2").resolve()
    )
    if Path(sys.prefix).resolve() != expected_environment:
        raise RuntimeError(
            f"Solar training must use {expected_environment}, "
            f"found {Path(sys.prefix).resolve()}"
        )
    if b200:
        if args.asset_integrity_receipt is None:
            raise ValueError("B200 training requires --asset-integrity-receipt")
        source_dir = require_b200_owned_path(args.source_dir, must_exist=True)
        expert_nf4_dir = require_b200_owned_path(args.expert_nf4_dir, must_exist=True)
        data_dir = require_b200_owned_path(args.data_dir, must_exist=True)
        output_dir = require_b200_owned_path(args.output_dir)
        preflight_path = require_b200_owned_path(
            args.preflight_receipt, must_exist=True
        )
        asset_integrity_receipt = require_b200_owned_path(
            args.asset_integrity_receipt,
            must_exist=True,
        )
        validate_b200_large_asset_receipt(
            asset_integrity_receipt,
            role="training",
        )
        configured_runtime = model_contract["training_runtime"]
        if source_dir != Path(configured_runtime["source_dir"]):
            raise RuntimeError("source-dir differs from the B200 config")
        if expert_nf4_dir != Path(configured_runtime["expert_dir"]):
            raise RuntimeError("expert-nf4-dir differs from the B200 config")
    else:
        if args.asset_integrity_receipt is not None:
            raise ValueError(
                "source-profile training does not consume B200 asset evidence"
            )
        asset_integrity_receipt = None
        source_dir = resolve_run_path(args.source_dir, must_exist=True)
        expert_nf4_dir = resolve_run_path(
            args.expert_nf4_dir,
            must_exist=True,
        )
        data_dir = resolve_run_path(args.data_dir, must_exist=True)
        output_dir = resolve_run_path(args.output_dir)
        preflight_path = resolve_run_path(
            args.preflight_receipt,
            must_exist=True,
        )
    preflight = validate_preflight(
        preflight_path,
        source_dir=source_dir,
        expert_nf4_dir=expert_nf4_dir,
        training_config=training_config,
    )
    production_smoke_receipt: dict[str, str] | None = None
    production_approval_receipt: dict[str, str] | None = None
    if args.smoke:
        if args.smoke_receipt is not None:
            raise ValueError("--smoke cannot consume a production smoke receipt")
        if args.production_approval_receipt is not None:
            raise ValueError("--smoke cannot consume a production approval")
    else:
        if args.smoke_receipt is None:
            raise ValueError("production training requires --smoke-receipt")
        smoke_receipt_path = (
            require_b200_owned_path(
                args.smoke_receipt,
                must_exist=True,
            )
            if b200
            else resolve_run_path(args.smoke_receipt, must_exist=True)
        )
        validate_production_smoke(
            smoke_receipt_path,
            source_dir=source_dir,
            expert_nf4_dir=expert_nf4_dir,
            training_config=training_config,
        )
        production_smoke_receipt = {
            "path": str(smoke_receipt_path),
            "sha256": sha256(smoke_receipt_path),
        }
        if b200:
            if args.production_approval_receipt is None:
                raise ValueError(
                    "B200 production training requires " "--production-approval-receipt"
                )
            approval_path = require_b200_owned_path(
                args.production_approval_receipt,
                must_exist=True,
            )
            validate_b200_production_approval(
                approval_path,
                smoke_receipt_path=smoke_receipt_path,
                smoke_receipt_sha256=production_smoke_receipt["sha256"],
                training_config=training_config,
                pipeline_identity=selected_pipeline_identity,
                output_dir=output_dir,
                wandb_identity=wandb_identity,
            )
            production_approval_receipt = {
                "path": str(approval_path),
                "sha256": sha256(approval_path),
            }
        elif args.production_approval_receipt is not None:
            raise ValueError("source profile does not consume B200 production approval")
    validation_recipe = recipe["validation"]
    full_validation_steps = [
        int(step) for step in validation_recipe["full_validation_steps"]
    ]
    if b200:
        early_stopping_policy = validate_b200_early_stopping_policy(
            validation_recipe["early_stopping"],
            optimizer_steps=occurrence_contract["optimizer_steps"],
            full_validation_steps=full_validation_steps,
        )
        early_terminal_step = occurrence_contract["optimizer_steps"]
    else:
        early_stopping_policy = validate_early_stopping_policy(
            validation_recipe["early_stopping"],
            optimizer_steps=occurrence_contract["optimizer_steps"],
            full_validation_steps=full_validation_steps,
        )
        early_terminal_step = int(early_stopping_policy["earliest_stop_step"])

    def derive_early_stopping_decision(
        history_value: list[dict[str, Any]],
        completed_step_value: int,
    ) -> dict[str, Any]:
        if b200:
            return b200_coverage_decision(
                history_value,
                completed_step=completed_step_value,
                recipe=recipe,
                smoke=args.smoke,
            )
        return early_stopping_decision(
            history_value,
            completed_step=completed_step_value,
            policy=early_stopping_policy,
            optimizer_steps=occurrence_contract["optimizer_steps"],
            full_validation_steps=full_validation_steps,
            occurrences_per_step=occurrence_contract["nominal_step_occurrences"],
            occurrences_per_epoch=int(
                recipe["training_deck"]["occurrences_per_epoch"]["total"]
            ),
            total_occurrences=occurrence_contract["total_occurrences"],
        )

    if data_dir != Path(data_contract["prepared_data_dir"]).resolve():
        raise RuntimeError("data-dir is not the pinned prepared dataset")
    for key in ("train", "validation"):
        entry = data_contract["files"][key]
        path = data_dir / entry["name"]
        if sha256(path) != entry["sha256"]:
            raise RuntimeError(f"training data changed: {path}")

    preflight_hashes = preflight_state_hashes(preflight, b200=b200)
    state_contract = {
        "format_version": (
            B200_STATE_CONTRACT_VERSION if b200 else STATE_CONTRACT_VERSION
        ),
        "profile": training_config["profile"],
        "config_identity": training_config["identity"],
        "source_dir": str(source_dir),
        "expert_nf4_dir": str(expert_nf4_dir),
        "download_receipt_sha256": preflight_hashes["download_receipt_sha256"],
        "expert_manifest_sha256": preflight_hashes["expert_manifest_sha256"],
        "model_contract_sha256": training_config["hashes"]["model"],
        "recipe_sha256": training_config["hashes"]["recipe"],
        "data_contract_sha256": training_config["hashes"]["data_contract"],
        "train_sha256": data_contract["files"]["train"]["sha256"],
        "validation_sha256": data_contract["files"]["validation"]["sha256"],
        "pipeline_code_sha256": selected_pipeline_identity["sha256"],
        "environment_lock_sha256": sha256(
            PROJECT / "requirements/solar-open-2.lock.txt"
        ),
        "smoke": args.smoke,
        "gpu_runtime": gpu_runtime_audit,
    }
    if b200:
        state_contract["training_performance_sha256"] = canonical_sha256(
            recipe["training_performance"]
        )
        state_contract["training_activation_contract_sha256"] = canonical_sha256(
            b200_selective_gradient_checkpointing_contract()
        )
    if production_smoke_receipt is not None:
        state_contract["production_smoke_receipt"] = production_smoke_receipt
        state_contract["wandb"] = wandb_identity
    if production_approval_receipt is not None:
        state_contract["production_approval_receipt"] = production_approval_receipt
    state_contract_sha256 = canonical_sha256(state_contract)
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_usage = shutil.disk_usage(output_dir)
    storage_audit = {
        "path": str(output_dir),
        "free_bytes": int(storage_usage.free),
        "minimum_required_bytes": MINIMUM_FREE_STORAGE_BYTES,
        "status": (
            "PASS" if storage_usage.free >= MINIMUM_FREE_STORAGE_BYTES else "FAIL"
        ),
    }
    if storage_audit["status"] != "PASS":
        raise RuntimeError(
            "insufficient free storage for recovery/candidate/final adapters: "
            f"{storage_audit}"
        )
    resume_path = output_dir / "resume_state.pt"
    previous_resume_path = output_dir / "resume_state.previous.pt"
    success_path = output_dir / "_SUCCESS"
    if success_path.exists():
        manifest = read_json(output_dir / "run_manifest.json")
        manifest_steps = int(manifest.get("completed_updates", -1))
        decision = derive_early_stopping_decision(
            list(manifest.get("history") or []),
            manifest_steps,
        )
        wanted = (
            "SMOKE_COMPLETED"
            if args.smoke
            else ("EARLY_STOPPED" if decision["triggered"] else "COMPLETED")
        )
        wanted_steps = (
            1
            if args.smoke
            else (
                early_terminal_step
                if decision["triggered"]
                else int(recipe["optimizer_steps"])
            )
        )
        wanted_target_steps = 1 if args.smoke else int(recipe["optimizer_steps"])
        if (
            manifest.get("status") == wanted
            and manifest.get("state_contract") == state_contract
            and manifest.get("completed_updates") == wanted_steps
            and manifest.get("target_updates") == wanted_target_steps
            and manifest.get("early_stopping_decision") == decision
            and (manifest.get("wandb_reporting") or {}).get("completion_status")
            == ("DISABLED_FOR_SMOKE" if args.smoke else "COMMITTED")
        ):
            for label in ("final_checkpoint", "selected_checkpoint"):
                record = manifest.get(label)
                if not isinstance(record, dict):
                    raise RuntimeError(f"completed manifest lacks {label} evidence")
                adapter_dir = resolve_run_path(
                    Path(str(record.get("adapter_dir") or ""))
                )
                if not (adapter_dir / "_SUCCESS").is_file():
                    raise RuntimeError(
                        f"completed {label} adapter is incomplete: {adapter_dir}"
                    )
                observed_files = {
                    path.name: sha256(path)
                    for path in sorted(adapter_dir.iterdir())
                    if path.is_file()
                }
                if observed_files != record.get("files_sha256"):
                    raise RuntimeError(f"completed {label} adapter hashes changed")
            print(json.dumps({"status": wanted, "already_complete": True}))
            return 0
        raise RuntimeError("success marker exists with stale or invalid run manifest")

    def valid_uncommitted_resume(path: Path) -> dict[str, Any]:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(state, dict):
            raise RuntimeError("resume pointer is not a mapping")
        if state.get("state_contract_sha256") != state_contract_sha256:
            raise RuntimeError("resume pointer state contract changed")
        adapter_dir = resolve_run_path(
            Path(str(state["adapter_dir"])),
            must_exist=True,
        )
        if not (adapter_dir / "_SUCCESS").is_file():
            raise RuntimeError("resume pointer references an incomplete adapter")
        recorded_files = state.get("adapter_files_sha256")
        if not isinstance(recorded_files, dict):
            raise RuntimeError("resume pointer lacks adapter hashes")
        observed_files = {
            child.name: sha256(child)
            for child in sorted(adapter_dir.iterdir())
            if child.is_file()
        }
        if observed_files != recorded_files:
            raise RuntimeError("resume pointer adapter hashes changed")
        trainable_sha256 = str(state.get("trainable_parameter_sha256") or "")
        if (
            len(trainable_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in trainable_sha256
            )
            or (adapter_dir / "TRAINABLE_PARAMETER_SHA256").read_text(encoding="utf-8")
            != f"{trainable_sha256}\n"
        ):
            raise RuntimeError("resume pointer trainable-parameter identity is invalid")
        return state

    def resume_freshness(state: dict[str, Any]) -> tuple[int, ...]:
        return (
            int(state.get("recovery_generation", -1)),
            int(state.get("completed_steps", -1)),
            len(state.get("history") or []),
            len(state.get("wandb_logged_validation_keys") or []),
            len(state.get("candidate_checkpoints") or {}),
        )

    orphan_resume_states: list[tuple[tuple[int, ...], Path]] = []
    for orphan in sorted(output_dir.glob(".resume-next.*.pt")):
        try:
            orphan_state = valid_uncommitted_resume(orphan)
            orphan_freshness = resume_freshness(orphan_state)
        except Exception as error:
            print(
                f"[recovery] discarding invalid uncommitted resume pointer "
                f"{orphan}: {type(error).__name__}: {error}",
                flush=True,
            )
            orphan.unlink()
            continue
        orphan_resume_states.append((orphan_freshness, orphan))
    current_freshness: tuple[int, ...] | None = None
    if resume_path.is_file():
        try:
            current_freshness = resume_freshness(valid_uncommitted_resume(resume_path))
        except Exception as error:
            print(
                f"[recovery] current resume pointer will be rejected by "
                f"full validation: {type(error).__name__}: {error}",
                flush=True,
            )
    previous_freshness: tuple[int, ...] | None = None
    if previous_resume_path.is_file():
        try:
            previous_freshness = resume_freshness(
                valid_uncommitted_resume(previous_resume_path)
            )
        except Exception as error:
            print(
                f"[recovery] previous resume pointer will be rejected by "
                f"full validation: {type(error).__name__}: {error}",
                flush=True,
            )
    if orphan_resume_states:
        orphan_freshness, orphan_path = max(
            orphan_resume_states,
            key=lambda item: (item[0], item[1].stat().st_mtime_ns),
        )
        if (current_freshness is None or orphan_freshness > current_freshness) and (
            current_freshness is not None
            or previous_freshness is None
            or orphan_freshness > previous_freshness
        ):
            if resume_path.is_file():
                if current_freshness is None:
                    resume_path.unlink()
                else:
                    os.replace(resume_path, previous_resume_path)
            os.replace(orphan_path, resume_path)
            current_freshness = orphan_freshness
            print(
                f"[recovery] promoted durable uncommitted resume pointer "
                f"{orphan_path}",
                flush=True,
            )
    for orphan in output_dir.glob(".resume-next.*.pt"):
        orphan.unlink()
    for temporary in output_dir.glob("..resume-next.*.tmp"):
        temporary.unlink()
    descriptor = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

    existing = [path for path in output_dir.iterdir() if path.name != "logs"]
    if existing and not resume_path.is_file() and not previous_resume_path.is_file():
        recovery_root = output_dir / "recovery"
        allowed_initial_recovery = (
            len(existing) == 1
            and existing[0] == recovery_root
            and recovery_root.is_dir()
            and all(
                child.is_dir()
                and (
                    re.fullmatch(
                        r"step-00000-generation-\d+",
                        child.name,
                    )
                    is not None
                    or re.fullmatch(
                        r"\.step-00000-generation-\d+\.\d+\.tmp",
                        child.name,
                    )
                    is not None
                )
                for child in recovery_root.iterdir()
            )
        )
        if not allowed_initial_recovery:
            raise FileExistsError(
                f"non-empty output has no resumable state: {output_dir}"
            )

    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(
            "[signal] SIGUSR1 received; stopping at optimizer/validation boundary",
            flush=True,
        )

    signal.signal(signal.SIGUSR1, request_stop)
    random.seed(recipe["seed"])
    torch.manual_seed(recipe["seed"])
    torch.cuda.manual_seed_all(recipe["seed"])
    torch.set_float32_matmul_precision("high")
    prompt_module = load_reference_module()

    tokenizer = AutoTokenizer.from_pretrained(
        source_dir,
        use_fast=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    chat_template = tokenizer.chat_template
    if not isinstance(chat_template, str):
        raise RuntimeError("Solar tokenizer must expose one chat template")
    for required_text in ("reasoning_effort", "<|think:start|>", "<|think:end|>"):
        if required_text not in chat_template:
            raise RuntimeError(f"Solar chat template lacks {required_text}")

    train_rows = load_rows(data_dir / data_contract["files"]["train"]["name"])
    validation_rows = load_rows(data_dir / data_contract["files"]["validation"]["name"])
    max_length = int(recipe["max_length"])
    raw_buckets = recipe["microbatch_policy"]["buckets"]
    microbatch_buckets = validate_microbatch_buckets(
        [
            {
                "max_length": int(bucket["max_length_inclusive"]),
                "batch_size": int(bucket["target_micro_batch_size"]),
            }
            for bucket in raw_buckets
        ],
        max_length=max_length,
    )

    def scan_token_lengths(
        rows: list[dict[str, Any]],
        *,
        label: str,
    ) -> tuple[list[int], dict[str, Any]]:
        lengths: list[int] = []
        started = time.time()
        for index, row in enumerate(rows, 1):
            item = encode_row(
                tokenizer,
                prompt_module,
                row,
                hard_term_weight=1.0,
            )
            lengths.append(int(item["length"]))
            if index % 10_000 == 0:
                print(
                    f"[token-length-contract] {label} {index}/{len(rows)}",
                    flush=True,
                )
        contract = token_length_contract(
            rows,
            lengths,
            max_length=max_length,
        )
        contract["elapsed_seconds"] = round(time.time() - started, 2)
        if contract["over_max_count"] != 0:
            preview = contract["over_max_rows"][:10]
            raise RuntimeError(
                f"{label} has {contract['over_max_count']} rows over "
                f"max_length={max_length}: {preview}"
            )
        print(
            json.dumps(
                {f"{label}_token_length_contract": contract},
                ensure_ascii=False,
            ),
            flush=True,
        )
        return lengths, contract

    train_lengths, train_length_contract = scan_token_lengths(
        train_rows,
        label="train",
    )
    validation_lengths, validation_length_contract = scan_token_lengths(
        validation_rows,
        label="validation",
    )
    if b200:
        expected_longest_tokens = int(recipe["actual_longest_row_tokens"])
        if expected_longest_tokens != 9_560:
            raise RuntimeError("B200 longest-row contract changed from 9,560")
        if train_length_contract["maximum"] != expected_longest_tokens:
            raise RuntimeError(
                "B200 train token scan did not find the sealed longest row: "
                f"{train_length_contract['maximum']} != "
                f"{expected_longest_tokens}"
            )
    sampler = GlobalAugmentedDeckSampler(
        train_rows,
        expert_label=recipe["expert_quality_label"],
        expert_boost=recipe["expert_oversampling_factor"],
        epochs=int(recipe["epochs"]),
        seed=recipe["seed"],
    )
    deck_contract = recipe["training_deck"]
    if sampler.examples_per_epoch != int(
        deck_contract["occurrences_per_epoch"]["total"]
    ):
        raise RuntimeError("sampler occurrences_per_epoch differs from recipe")
    if sampler.total_examples != int(deck_contract["total_occurrences"]["total"]):
        raise RuntimeError("sampler total occurrences differs from recipe")
    derived_steps = occurrence_contract["optimizer_steps"]
    if consumed_occurrences_at_step(recipe, derived_steps) != sampler.total_examples:
        raise RuntimeError("training deck differs from the exact step horizon")
    if derived_steps != int(recipe["optimizer_steps"]):
        raise RuntimeError(
            f"derived optimizer steps changed: {derived_steps} "
            f"!= {recipe['optimizer_steps']}"
        )
    planning_sampler = GlobalAugmentedDeckSampler(
        train_rows,
        expert_label=recipe["expert_quality_label"],
        expert_boost=recipe["expert_oversampling_factor"],
        epochs=int(recipe["epochs"]),
        seed=recipe["seed"],
    )
    production_training_plan = planned_microbatch_counts(
        (planning_sampler.draw_index() for _ in range(planning_sampler.total_examples)),
        lengths=train_lengths,
        rows_per_update=int(recipe["effective_batch_size"]),
        buckets=microbatch_buckets,
    )
    if production_training_plan["optimizer_updates"] != derived_steps:
        raise RuntimeError("planned microbatch optimizer count differs from deck")
    if production_training_plan["occurrences"] != sampler.total_examples:
        raise RuntimeError("planned microbatch occurrence count differs from deck")
    print(
        json.dumps(
            {"production_training_plan": production_training_plan},
            ensure_ascii=False,
        ),
        flush=True,
    )

    resume_state: dict[str, Any] | None = None
    resume_source: Path | None = None
    recovery_errors: list[str] = []
    initial_sampler_state = copy.deepcopy(sampler.state_dict())
    for candidate in (resume_path, previous_resume_path):
        if not candidate.is_file():
            continue
        try:
            loaded = torch.load(
                candidate,
                map_location="cpu",
                weights_only=False,
            )
            if loaded.get("state_contract_sha256") != state_contract_sha256:
                raise RuntimeError("state contract differs from current run")
            adapter_dir = resolve_run_path(
                Path(str(loaded["adapter_dir"])),
                must_exist=True,
            )
            if not (adapter_dir / "_SUCCESS").is_file():
                raise RuntimeError(f"referenced adapter is incomplete: {adapter_dir}")
            recorded_adapter_files = loaded.get("adapter_files_sha256")
            if not isinstance(recorded_adapter_files, dict):
                raise RuntimeError("recovery state lacks adapter file hashes")
            observed_adapter_files = {
                path.name: sha256(path)
                for path in sorted(adapter_dir.iterdir())
                if path.is_file()
            }
            if observed_adapter_files != recorded_adapter_files:
                raise RuntimeError("recovery adapter file hashes changed")
            trainable_sha256 = str(loaded.get("trainable_parameter_sha256") or "")
            if (
                len(trainable_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in trainable_sha256
                )
                or (adapter_dir / "TRAINABLE_PARAMETER_SHA256").read_text(
                    encoding="utf-8"
                )
                != f"{trainable_sha256}\n"
            ):
                raise RuntimeError("recovery trainable-parameter identity is invalid")
            completed = int(loaded["completed_steps"])
            consumed = int(loaded["consumed_occurrences"])
            maximum_steps = 1 if args.smoke else int(recipe["optimizer_steps"])
            if not 0 <= completed <= maximum_steps:
                raise RuntimeError("recovery completed step is out of range")
            if consumed != consumed_occurrences_at_step(recipe, completed):
                raise RuntimeError(
                    "recovery consumed occurrences differ from completed steps"
                )
            expected_occurrence_boundary = {
                "completed_steps": completed,
                "consumed_occurrences": consumed,
                "next_step_occurrences": (
                    update_occurrences_at_step(recipe, completed + 1)
                    if completed < occurrence_contract["optimizer_steps"]
                    else 0
                ),
            }
            if loaded.get("occurrence_boundary") != expected_occurrence_boundary:
                raise RuntimeError("recovery occurrence boundary changed")
            if int(loaded["backward_passes"]) != consumed:
                raise RuntimeError("recovery backward occurrence count is inconsistent")
            if not isinstance(loaded.get("history"), list):
                raise RuntimeError("recovery validation history is invalid")
            recovery_early_stopping = derive_early_stopping_decision(
                list(loaded["history"]),
                completed,
            )
            if recovery_early_stopping["triggered"] and completed != int(
                early_terminal_step
            ):
                raise RuntimeError(
                    "recovery early-stop decision differs from its completed step"
                )
            if not isinstance(
                loaded.get("wandb_logged_validation_keys"),
                list,
            ):
                raise RuntimeError("recovery W&B validation ledger is invalid")
            if not isinstance(loaded.get("candidate_checkpoints"), dict):
                raise RuntimeError("recovery candidate map is invalid")
            for candidate_key, candidate_record in loaded[
                "candidate_checkpoints"
            ].items():
                if not isinstance(candidate_record, dict):
                    raise RuntimeError(f"candidate record is invalid: {candidate_key}")
                candidate_dir = resolve_run_path(
                    Path(str(candidate_record.get("adapter_dir") or ""))
                )
                if not (candidate_dir / "_SUCCESS").is_file():
                    raise RuntimeError(
                        f"candidate adapter is incomplete: {candidate_dir}"
                    )
                candidate_files = {
                    path.name: sha256(path)
                    for path in sorted(candidate_dir.iterdir())
                    if path.is_file()
                }
                if candidate_files != candidate_record.get("files_sha256"):
                    raise RuntimeError(
                        f"candidate adapter hashes changed: {candidate_dir}"
                    )
            optimizer_state = loaded.get("optimizer")
            if (
                not isinstance(optimizer_state, dict)
                or not isinstance(optimizer_state.get("state"), dict)
                or not isinstance(optimizer_state.get("param_groups"), list)
            ):
                raise RuntimeError("recovery optimizer state is invalid")
            sampler.load_state_dict(loaded["sampler"])
            expected_sampler_cursor = 0 if args.smoke else consumed
            if sampler.cursor != expected_sampler_cursor:
                raise RuntimeError(
                    "recovery sampler cursor differs from consumed occurrences"
                )
            task_counts = Counter(loaded["occurrence_task_counts"])
            quality_counts = Counter(loaded["occurrence_quality_counts"])
            if sum(task_counts.values()) != consumed:
                raise RuntimeError("recovery task occurrence counts are invalid")
            if sum(quality_counts.values()) != consumed:
                raise RuntimeError("recovery quality occurrence counts are invalid")
            resume_state = loaded
            resume_source = candidate
            break
        except Exception as error:
            sampler.load_state_dict(initial_sampler_state)
            recovery_errors.append(f"{candidate}: {type(error).__name__}: {error}")
    if resume_state is None and recovery_errors:
        raise RuntimeError(
            "no valid recovery generation: " + " | ".join(recovery_errors)
        )
    if resume_state is not None:
        sampler.load_state_dict(resume_state["sampler"])
        print(
            json.dumps(
                {
                    "recovery_source": str(resume_source),
                    "completed_steps": resume_state["completed_steps"],
                    "adapter_dir": resume_state["adapter_dir"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    started_load = time.time()
    loader_kwargs: dict[str, Any] = {
        "source_dir": source_dir,
        "expert_checkpoint_dir": expert_nf4_dir,
        "verify_hashes": not b200,
    }
    if b200:
        loader_kwargs["config_dir"] = training_config["config_dir"]
        loader_kwargs["asset_integrity_receipt"] = asset_integrity_receipt
    model, loader_audit = load_full_model(**loader_kwargs)
    model.config.use_cache = False
    device_audit = model_device_audit(model)
    module_audit = lora_module_audit(model, model_contract["architecture_contract"])
    lora_recipe = recipe["lora"]
    lora_config = LoraConfig(
        r=lora_recipe["rank"],
        lora_alpha=lora_recipe["alpha"],
        lora_dropout=lora_recipe["dropout"],
        bias=lora_recipe["bias"],
        task_type="CAUSAL_LM",
        target_modules=lora_recipe["target_regex"],
    )
    if resume_state is None:
        with solar_open2_peft_v5_native():
            model = get_peft_model(model, lora_config)
    else:
        adapter_dir = Path(resume_state["adapter_dir"])
        if not (adapter_dir / "_SUCCESS").is_file():
            raise RuntimeError(f"resume adapter is incomplete: {adapter_dir}")
        with solar_open2_peft_v5_native():
            model = PeftModel.from_pretrained(
                model,
                adapter_dir,
                is_trainable=True,
                local_files_only=True,
            )
    if b200:
        training_performance = recipe["training_performance"]
        if training_performance != b200_training_performance_contract():
            raise RuntimeError("unsupported B200 training-performance policy")
        gradient_checkpointing_contract = (
            configure_selective_model_parallel_gradient_checkpointing(model)
        )
        if (
            gradient_checkpointing_contract
            != b200_selective_gradient_checkpointing_contract()
        ):
            raise RuntimeError(
                "effective B200 activation-checkpointing contract changed"
            )
    else:
        training_performance = {
            "activation_checkpointing": "device_aware_non_reentrant",
            "logit_chunk_size": int(os.environ.get("LOGIT_CHUNK_SIZE", "256")),
            "hidden_environment_override_allowed": True,
        }
        gradient_checkpointing_contract = (
            configure_model_parallel_gradient_checkpointing(model)
        )
    model.enable_input_require_grads()
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    trainable_parameters = sum(parameter.numel() for parameter in parameters)
    if trainable_parameters != lora_recipe["expected_trainable_parameters"]:
        raise RuntimeError(
            f"trainable parameter mismatch: {trainable_parameters} "
            f"!= {lora_recipe['expected_trainable_parameters']}"
        )
    loaded_adapter_stats = trainable_parameter_stats(model)
    if not loaded_adapter_stats["finite"]:
        raise RuntimeError("LoRA parameters are non-finite before training")
    needs_compile_warmup = args.smoke and (
        resume_state is None or int(resume_state["completed_steps"]) == 0
    )
    adapter_digest_at_load = (
        trainable_parameter_digest(model)
        if needs_compile_warmup or resume_state is not None
        else None
    )
    if resume_state is not None and adapter_digest_at_load != str(
        resume_state["trainable_parameter_sha256"]
    ):
        if resume_source == resume_path and previous_resume_path.is_file():
            print(
                "[recovery] loaded current LoRA digest is invalid; "
                "promoting the previous recovery pointer and requesting retry",
                flush=True,
            )
            resume_path.unlink()
            os.replace(previous_resume_path, resume_path)
            descriptor = os.open(
                output_dir,
                os.O_RDONLY | os.O_DIRECTORY,
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            return REQUEUE_EXIT_CODE
        raise RuntimeError(
            "loaded LoRA parameters differ from the sealed recovery digest"
        )
    frozen_probe_before = frozen_parameter_probe(model)
    print(
        json.dumps(
            {
                "model_load_seconds": round(time.time() - started_load, 2),
                "device_audit": device_audit,
                "loader_audit": loader_audit,
                "lora_module_audit": module_audit,
                "trainable_parameters": trainable_parameters,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    embedding = model.get_input_embeddings()

    def neftune_hook(module: Any, _inputs: Any, output: torch.Tensor) -> torch.Tensor:
        alpha = float(recipe["neftune_alpha"])
        if module.training and alpha > 0:
            magnitude = alpha / math.sqrt(output.size(1) * output.size(2))
            return output + torch.empty_like(output).uniform_(-magnitude, magnitude)
        return output

    embedding.register_forward_hook(neftune_hook)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=recipe["learning_rate"],
        weight_decay=recipe["weight_decay"],
        betas=tuple(recipe["adam_betas"]),
        foreach=False,
    )
    total_steps = 1 if args.smoke else int(recipe["optimizer_steps"])
    rows_per_update = int(recipe["effective_batch_size"])
    if rows_per_update != int(
        recipe["microbatch_policy"]["optimizer_step_occurrences"]
    ):
        raise RuntimeError("effective batch differs from optimizer-step occurrences")
    warmup_steps = scheduler_warmup_steps(
        smoke=False,
        warmup_ratio=float(recipe["warmup_ratio"]),
        total_steps=int(recipe["optimizer_steps"]),
    )
    completed_steps = 0
    history: list[dict[str, Any]] = []
    wandb_logged_validation_keys: set[str] = set()
    backward_passes = 0
    consumed_occurrences = 0
    occurrence_task_counts: Counter[str] = Counter()
    occurrence_quality_counts: Counter[str] = Counter()
    max_observed_tokens = 0
    total_hard_term_weighted_tokens = 0
    first_update_gradient_audit: dict[str, Any] | None = None
    first_update_adapter_delta: dict[str, Any] | None = None
    frozen_probe_after_update: dict[str, Any] | None = None
    first_output_only_supervision: dict[str, Any] | None = None
    microbatch_phase_seconds: float | None = None
    validation_row_seconds: float | None = None
    validation_benchmark: dict[str, Any] | None = None
    training_bucket_benchmarks: dict[str, dict[str, Any]] = {}
    validation_bucket_benchmarks: dict[str, dict[str, Any]] = {}
    production_update_benchmark: dict[str, Any] | None = None
    optimizer_boundary_seconds: float | None = None
    one_time_gradient_audit_seconds: float | None = None
    projected_runtime_budget: dict[str, Any] | None = None
    post_update_recompute_audit: dict[str, Any] | None = None
    exact_tail_smoke_audit: dict[str, Any] | None = None
    recovery_generation = 0
    last_recovery_monotonic = time.monotonic()
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        completed_steps = int(resume_state["completed_steps"])
        history = list(resume_state["history"])
        wandb_logged_validation_keys = {
            str(value) for value in resume_state["wandb_logged_validation_keys"]
        }
        backward_passes = int(resume_state["backward_passes"])
        consumed_occurrences = int(resume_state["consumed_occurrences"])
        occurrence_task_counts = Counter(resume_state["occurrence_task_counts"])
        occurrence_quality_counts = Counter(resume_state["occurrence_quality_counts"])
        max_observed_tokens = int(resume_state["max_observed_tokens"])
        total_hard_term_weighted_tokens = int(
            resume_state["total_hard_term_weighted_tokens"]
        )
        first_update_gradient_audit = resume_state["first_update_gradient_audit"]
        first_update_adapter_delta = resume_state["first_update_adapter_delta"]
        frozen_probe_after_update = resume_state["frozen_probe_after_update"]
        first_output_only_supervision = resume_state["first_output_only_supervision"]
        microbatch_phase_seconds = resume_state["microbatch_phase_seconds"]
        validation_row_seconds = resume_state["validation_row_seconds"]
        validation_benchmark = resume_state["validation_benchmark"]
        if validation_benchmark is not None:
            training_bucket_benchmarks = dict(
                validation_benchmark.get("training_buckets") or {}
            )
            validation_bucket_benchmarks = dict(
                validation_benchmark.get("validation_buckets") or {}
            )
            raw_update_benchmark = validation_benchmark.get(
                "production_update_benchmark"
            )
            if isinstance(raw_update_benchmark, dict):
                production_update_benchmark = dict(raw_update_benchmark)
        optimizer_boundary_seconds = resume_state["optimizer_boundary_seconds"]
        one_time_gradient_audit_seconds = resume_state[
            "one_time_gradient_audit_seconds"
        ]
        projected_runtime_budget = resume_state["runtime_budget_projection"]
        post_update_recompute_audit = resume_state["post_update_recompute_audit"]
        exact_tail_smoke_audit = resume_state.get("exact_tail_smoke_audit")
        recovery_generation = int(resume_state["recovery_generation"])
        last_recovery_monotonic = time.monotonic()
        torch.set_rng_state(resume_state["torch_rng"])
        torch.cuda.set_rng_state_all(resume_state["cuda_rng"])
        if (first_update_gradient_audit is None) != (
            first_update_adapter_delta is None
        ):
            raise RuntimeError(
                "resume state has partial first effective-update evidence"
            )

    base_model = model.get_base_model()
    backbone = base_model.model
    lm_head = base_model.lm_head
    if lm_head.weight.requires_grad:
        raise RuntimeError("lm_head must remain frozen")
    input_device = model.get_input_embeddings().weight.device
    lm_device = lm_head.weight.device
    if b200 and "LOGIT_CHUNK_SIZE" in os.environ:
        raise RuntimeError(
            "B200 LOGIT_CHUNK_SIZE is recipe-sealed; hidden override is forbidden"
        )
    logit_chunk_size = int(training_performance["logit_chunk_size"])
    if logit_chunk_size < 1:
        raise RuntimeError("logit chunk size must be positive")

    def encoded(row: dict[str, Any], *, weighted: bool) -> dict[str, Any]:
        item = encode_row(
            tokenizer,
            prompt_module,
            row,
            hard_term_weight=(
                recipe["hard_term_target_reading_loss_weight"] if weighted else 1.0
            ),
        )
        if int(item["length"]) > max_length:
            raise RuntimeError(
                f"row exceeded fail-closed max length after preflight: "
                f"id={row['id']} task={row['task']} "
                f"tokens={item['length']} max={max_length}"
            )
        return item

    smoke_update_indices: list[int] = []
    smoke_recompute_row: dict[str, Any] | None = None
    smoke_validation: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    if args.smoke:
        absolute_longest = max(
            range(len(train_rows)),
            key=lambda index: train_lengths[index],
        )
        hard_term_candidates = sorted(
            (
                index
                for index, row in enumerate(train_rows)
                if row.get("hard_term_readings")
            ),
            key=lambda index: train_lengths[index],
            reverse=True,
        )
        hard_term_index: int | None = None
        for index in hard_term_candidates:
            if (
                encoded(
                    train_rows[index],
                    weighted=True,
                )["hard_term_weighted_tokens"]
                > 0
            ):
                hard_term_index = index
                break
        if hard_term_index is None:
            raise RuntimeError("no training row exercises hard-term token weighting")
        chosen: list[int] = []
        chosen_set: set[int] = set()

        def choose(index: int) -> None:
            if index not in chosen_set and len(chosen) < rows_per_update:
                chosen.append(index)
                chosen_set.add(index)

        choose(absolute_longest)
        choose(hard_term_index)
        for bucket_index_value, (ceiling, capacity) in enumerate(microbatch_buckets):
            lower = (
                1
                if bucket_index_value == 0
                else microbatch_buckets[bucket_index_value - 1][0] + 1
            )
            candidates = sorted(
                (
                    index
                    for index, length in enumerate(train_lengths)
                    if lower <= length <= ceiling
                ),
                key=lambda index: train_lengths[index],
                reverse=True,
            )
            for index in candidates[:capacity]:
                choose(index)
        for index in range(len(train_rows)):
            choose(index)
            if len(chosen) == rows_per_update:
                break
        if len(chosen) != rows_per_update:
            raise RuntimeError("unable to build a full smoke optimizer group")
        smoke_update_indices = chosen
        smoke_recompute_row = copy.deepcopy(train_rows[absolute_longest])
        if train_lengths[absolute_longest] != train_length_contract["maximum"]:
            raise AssertionError("absolute longest smoke row changed")
        for task in ("translation", "summarization"):
            index = max(
                (
                    row_index
                    for row_index, row in enumerate(validation_rows)
                    if row["task"] == task
                ),
                key=lambda row_index: validation_lengths[row_index],
            )
            row = validation_rows[index]
            smoke_validation[task] = (row, encoded(row, weighted=False))

    def next_update_indices() -> list[int]:
        next_step = completed_steps + 1
        expected_update_occurrences = update_occurrences_at_step(
            recipe,
            next_step,
        )
        if args.smoke:
            if completed_steps != 0:
                raise StopIteration("smoke optimizer group is exhausted")
            indices = list(smoke_update_indices)
        else:
            indices = [sampler.draw_index() for _ in range(expected_update_occurrences)]
        if len(indices) != expected_update_occurrences:
            raise RuntimeError(
                "optimizer group differs from its exact occurrence count"
            )
        return indices

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise RuntimeError("tokenizer lacks both pad and EOS token IDs")

    def batch_row_losses(
        items: list[dict[str, Any]],
        *,
        backward_scale: float | None,
    ) -> list[float]:
        if not items:
            raise ValueError("loss batch must not be empty")
        maximum = max(len(item["input_ids"]) for item in items)
        input_rows: list[list[int]] = []
        attention_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        weight_rows: list[list[float]] = []
        for item in items:
            padding = maximum - len(item["input_ids"])
            input_rows.append(item["input_ids"] + [int(pad_token_id)] * padding)
            attention_rows.append([1] * len(item["input_ids"]) + [0] * padding)
            label_rows.append(item["labels"] + [-100] * padding)
            weight_rows.append(item["weights"] + [0.0] * padding)
        ids = torch.tensor(input_rows, dtype=torch.long, device=input_device)
        attention = torch.tensor(
            attention_rows,
            dtype=torch.long,
            device=input_device,
        )
        outputs = backbone(
            input_ids=ids,
            attention_mask=attention,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state[:, :-1, :]
        targets = torch.tensor(
            [row[1:] for row in label_rows],
            dtype=torch.long,
            device=hidden.device,
        )
        token_weights = torch.tensor(
            [row[1:] for row in weight_rows],
            dtype=torch.float32,
            device=hidden.device,
        )
        mask = targets != -100
        supervised_counts = mask.sum(dim=1)
        denominators_local = token_weights.sum(dim=1)
        if bool((supervised_counts <= 0).any().item()):
            raise RuntimeError("loss batch contains a row without target tokens")
        if bool((denominators_local <= 0).any().item()):
            raise RuntimeError("loss batch contains a row without target weight")
        if bool((token_weights[mask] <= 0).any().item()):
            raise RuntimeError("supervised tokens must have positive loss weight")
        if bool((token_weights[~mask] != 0).any().item()):
            raise RuntimeError("ignored tokens must have zero loss weight")
        selected_hidden = hidden[mask].to(lm_device)
        selected_targets = targets[mask].to(lm_device)
        selected_weights = token_weights[mask].to(lm_device)
        batch_indices = (
            torch.arange(len(items), device=hidden.device)[:, None]
            .expand_as(targets)[mask]
            .to(lm_device)
        )
        denominators = denominators_local.to(lm_device)
        normalized_weights = selected_weights / denominators[batch_indices]
        proxy = (
            selected_hidden.detach().requires_grad_(True)
            if backward_scale is not None
            else selected_hidden
        )
        row_values = torch.zeros(
            len(items),
            dtype=torch.float64,
            device=lm_device,
        )
        for start in range(0, selected_targets.numel(), logit_chunk_size):
            end = start + logit_chunk_size
            logits = lm_head(proxy[start:end]).float()
            losses = F.cross_entropy(
                logits, selected_targets[start:end], reduction="none"
            )
            contributions = losses * normalized_weights[start:end]
            row_values.scatter_add_(
                0,
                batch_indices[start:end],
                contributions.detach().to(torch.float64),
            )
            if backward_scale is not None:
                (contributions.sum() * backward_scale).backward()
        if backward_scale is not None:
            if proxy.grad is None:
                raise RuntimeError("chunked loss produced no hidden-state gradient")
            selected_hidden.backward(proxy.grad)
        values = [float(value) for value in row_values.cpu().tolist()]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise RuntimeError(f"batch produced invalid row losses: {values}")
        return values

    if args.smoke and completed_steps == 0:
        if smoke_recompute_row is None:
            raise RuntimeError("smoke training benchmark row is missing")
        if adapter_digest_at_load is None:
            raise RuntimeError("smoke benchmark lacks its adapter identity")
        benchmark_torch_rng = torch.get_rng_state()
        benchmark_cuda_rng = torch.cuda.get_rng_state_all()
        benchmark_optimizer_entries = len(optimizer.state)
        training_bucket_benchmarks: dict[str, dict[str, Any]] = {}

        def benchmark_batch(
            items: list[dict[str, Any]],
            *,
            backward: bool,
        ) -> dict[str, Any]:
            optimizer.zero_grad(set_to_none=True)
            synchronize_cuda_devices()
            warmup_started = time.perf_counter()
            if backward:
                batch_row_losses(
                    items,
                    backward_scale=1.0 / len(items),
                )
            else:
                with torch.no_grad():
                    batch_row_losses(items, backward_scale=None)
            synchronize_cuda_devices()
            warmup_seconds = time.perf_counter() - warmup_started
            optimizer.zero_grad(set_to_none=True)
            repeat_torch_rng = torch.get_rng_state()
            repeat_cuda_rng = torch.cuda.get_rng_state_all()
            seconds_samples: list[float] = []
            for _repeat in range(SMOKE_BUCKET_TIMED_REPETITIONS):
                torch.set_rng_state(repeat_torch_rng)
                torch.cuda.set_rng_state_all(repeat_cuda_rng)
                optimizer.zero_grad(set_to_none=True)
                synchronize_cuda_devices()
                started = time.perf_counter()
                if backward:
                    batch_row_losses(
                        items,
                        backward_scale=1.0 / len(items),
                    )
                else:
                    with torch.no_grad():
                        batch_row_losses(items, backward_scale=None)
                synchronize_cuda_devices()
                seconds_samples.append(time.perf_counter() - started)
                optimizer.zero_grad(set_to_none=True)
            seconds_p50 = float(statistics.median(seconds_samples))
            return {
                "warmup_repetitions": 1,
                "warmup_seconds": warmup_seconds,
                "timed_repetitions": SMOKE_BUCKET_TIMED_REPETITIONS,
                "seconds_samples": seconds_samples,
                "seconds_p50": seconds_p50,
                "seconds": seconds_p50,
                "estimator": "median_p50",
            }

        model.train()
        for bucket_index_value, (ceiling, capacity) in enumerate(microbatch_buckets):
            lower = (
                1
                if bucket_index_value == 0
                else microbatch_buckets[bucket_index_value - 1][0] + 1
            )
            candidates = sorted(
                (
                    index
                    for index, length in enumerate(train_lengths)
                    if lower <= length <= ceiling
                ),
                key=lambda index: train_lengths[index],
                reverse=True,
            )
            if len(candidates) < capacity:
                raise RuntimeError(
                    f"training bucket {lower}-{ceiling} lacks " f"{capacity} smoke rows"
                )
            rows = [train_rows[index] for index in candidates[:capacity]]
            items = [encoded(row, weighted=True) for row in rows]
            training_bucket_benchmarks[str(ceiling)] = {
                "minimum_length": lower,
                "maximum_length": ceiling,
                "batch_size": capacity,
                "row_ids": [str(row["id"]) for row in rows],
                "observed_lengths": [int(item["length"]) for item in items],
                **benchmark_batch(items, backward=True),
            }
        microbatch_phase_seconds = max(
            float(item["seconds"]) for item in training_bucket_benchmarks.values()
        )

        validation_bucket_benchmarks: dict[str, dict[str, Any]] = {}
        model.eval()
        for bucket_index_value, (ceiling, capacity) in enumerate(microbatch_buckets):
            lower = (
                1
                if bucket_index_value == 0
                else microbatch_buckets[bucket_index_value - 1][0] + 1
            )
            candidates = sorted(
                (
                    index
                    for index, length in enumerate(validation_lengths)
                    if lower <= length <= ceiling
                ),
                key=lambda index: validation_lengths[index],
                reverse=True,
            )
            if len(candidates) < capacity:
                raise RuntimeError(
                    f"validation bucket {lower}-{ceiling} lacks "
                    f"{capacity} smoke rows"
                )
            rows = [validation_rows[index] for index in candidates[:capacity]]
            items = [encoded(row, weighted=False) for row in rows]
            validation_bucket_benchmarks[str(ceiling)] = {
                "minimum_length": lower,
                "maximum_length": ceiling,
                "batch_size": capacity,
                "row_ids": [str(row["id"]) for row in rows],
                "observed_lengths": [int(item["length"]) for item in items],
                **benchmark_batch(items, backward=False),
            }

        validation_by_task: dict[str, dict[str, Any]] = {}
        for task in ("translation", "summarization"):
            row, selected_item = smoke_validation[task]
            with torch.no_grad():
                batch_row_losses([selected_item], backward_scale=None)
            synchronize_cuda_devices()
            seconds_samples: list[float] = []
            timed_item: dict[str, Any] | None = None
            for _repeat in range(SMOKE_BUCKET_TIMED_REPETITIONS):
                synchronize_cuda_devices()
                started = time.perf_counter()
                timed_item = encoded(row, weighted=False)
                with torch.no_grad():
                    batch_row_losses([timed_item], backward_scale=None)
                synchronize_cuda_devices()
                seconds_samples.append(time.perf_counter() - started)
            if timed_item is None:
                raise AssertionError("validation timing produced no item")
            seconds_p50 = float(statistics.median(seconds_samples))
            validation_by_task[task] = {
                "row_id": row["id"],
                "tokens": int(timed_item["length"]),
                "warmup_repetitions": 1,
                "timed_repetitions": SMOKE_BUCKET_TIMED_REPETITIONS,
                "seconds_samples": seconds_samples,
                "seconds_p50": seconds_p50,
                "seconds": seconds_p50,
                "estimator": "median_p50",
            }
        model.train()
        validation_row_seconds = max(
            float(item["seconds"]) for item in validation_by_task.values()
        )

        benchmark_deck_sampler = GlobalAugmentedDeckSampler(
            train_rows,
            expert_label=recipe["expert_quality_label"],
            expert_boost=recipe["expert_oversampling_factor"],
            epochs=int(recipe["epochs"]),
            seed=recipe["seed"],
        )
        benchmark_deck = [
            benchmark_deck_sampler.draw_index()
            for _ in range(benchmark_deck_sampler.total_examples)
        ]
        warmup_step_numbers = list(range(1, SMOKE_UPDATE_WARMUP_REPETITIONS + 1))
        timed_step_numbers = [
            min(
                derived_steps,
                max(
                    SMOKE_UPDATE_WARMUP_REPETITIONS + 1,
                    round(
                        (sample_index + 0.5)
                        * derived_steps
                        / SMOKE_UPDATE_TIMED_REPETITIONS
                    ),
                ),
            )
            for sample_index in range(SMOKE_UPDATE_TIMED_REPETITIONS)
        ]
        if len(set(timed_step_numbers)) != SMOKE_UPDATE_TIMED_REPETITIONS:
            raise RuntimeError("production update benchmark steps are not unique")
        update_seconds_samples: list[float] = []
        update_microbatch_samples: list[int] = []
        timed_occurrence_indices: list[int] = []
        timed_row_identities: list[list[Any]] = []

        def benchmark_deck_update(step_number: int, *, timed: bool) -> None:
            start, end = update_occurrence_bounds(recipe, step_number)
            indices = benchmark_deck[start:end]
            expected_update_occurrences = update_occurrences_at_step(
                recipe,
                step_number,
            )
            if len(indices) != expected_update_occurrences:
                raise RuntimeError(f"benchmark deck step {step_number} is incomplete")
            optimizer.zero_grad(set_to_none=True)
            synchronize_cuda_devices()
            started = time.perf_counter()
            groups = group_update_rows(
                indices,
                length_of=lambda index: train_lengths[index],
                buckets=microbatch_buckets,
            )
            for group in groups:
                rows = [train_rows[index] for index in group]
                items = [encoded(row, weighted=True) for row in rows]
                batch_row_losses(
                    items,
                    backward_scale=1.0 / expected_update_occurrences,
                )
            synchronize_cuda_devices()
            seconds = time.perf_counter() - started
            optimizer.zero_grad(set_to_none=True)
            if timed:
                update_seconds_samples.append(seconds)
                update_microbatch_samples.append(len(groups))
                timed_occurrence_indices.extend(indices)
                timed_row_identities.extend(
                    [
                        [
                            str(train_rows[index]["id"]),
                            str(train_rows[index]["task"]),
                            int(train_lengths[index]),
                        ]
                        for index in indices
                    ]
                )

        for step_number in warmup_step_numbers:
            benchmark_deck_update(step_number, timed=False)
        for step_number in timed_step_numbers:
            benchmark_deck_update(step_number, timed=True)
        update_seconds_ordered = sorted(update_seconds_samples)
        update_seconds_p50 = float(statistics.median(update_seconds_samples))
        update_seconds_p90 = float(
            update_seconds_ordered[math.ceil(0.9 * len(update_seconds_ordered)) - 1]
        )
        production_update_benchmark = {
            "warmup_repetitions": SMOKE_UPDATE_WARMUP_REPETITIONS,
            "warmup_step_numbers": warmup_step_numbers,
            "timed_repetitions": SMOKE_UPDATE_TIMED_REPETITIONS,
            "timed_step_numbers": timed_step_numbers,
            "seconds_samples": update_seconds_samples,
            "seconds_p50": update_seconds_p50,
            "seconds_p90": update_seconds_p90,
            "seconds_mean": (sum(update_seconds_samples) / len(update_seconds_samples)),
            "estimator": "median_p50",
            "occurrences_per_update": rows_per_update,
            "microbatch_samples": update_microbatch_samples,
            "timed_plan": planned_microbatch_counts(
                timed_occurrence_indices,
                lengths=train_lengths,
                rows_per_update=rows_per_update,
                buckets=microbatch_buckets,
            ),
            "row_order_sha256": canonical_sha256(timed_row_identities),
        }

        optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(benchmark_torch_rng)
        torch.cuda.set_rng_state_all(benchmark_cuda_rng)
        benchmark_state_unchanged = (
            trainable_parameter_digest(model) == adapter_digest_at_load
            and frozen_parameter_probe(model) == frozen_probe_before
            and len(optimizer.state) == benchmark_optimizer_entries
            and all(parameter.grad is None for parameter in parameters)
            and torch.equal(torch.get_rng_state(), benchmark_torch_rng)
            and all(
                torch.equal(observed, expected)
                for observed, expected in zip(
                    torch.cuda.get_rng_state_all(),
                    benchmark_cuda_rng,
                    strict=True,
                )
            )
        )
        if not benchmark_state_unchanged:
            raise RuntimeError("runtime benchmark changed training state")

        if b200:
            tail_step = occurrence_contract["optimizer_steps"]
            tail_start, tail_end = update_occurrence_bounds(
                recipe,
                tail_step,
            )
            tail_count = tail_end - tail_start
            if (
                tail_step != 9_316
                or tail_start != 149_040
                or tail_end != 149_048
                or tail_count != 8
            ):
                raise RuntimeError(
                    "B200 transactional tail boundary changed: "
                    f"step={tail_step} slice=[{tail_start}, {tail_end})"
                )
            tail_occurrences = sampler.occurrences(
                tail_start,
                tail_count,
            )
            expected_global_positions = list(range(tail_start, tail_end))
            if [
                int(occurrence["global_position"]) for occurrence in tail_occurrences
            ] != expected_global_positions:
                raise RuntimeError(
                    "B200 tail sampler returned non-contiguous positions"
                )
            tail_occurrence_identities = [
                str(occurrence["identity_sha256"]) for occurrence in tail_occurrences
            ]
            occurrence_identity_duplicate_count = len(tail_occurrence_identities) - len(
                set(tail_occurrence_identities)
            )
            tail_row_identities = [
                [str(occurrence["id"]), str(occurrence["task"])]
                for occurrence in tail_occurrences
            ]
            row_identity_duplicate_count = len(tail_row_identities) - len(
                {tuple(identity) for identity in tail_row_identities}
            )
            if occurrence_identity_duplicate_count != 0:
                raise RuntimeError("B200 tail contains duplicate occurrence identities")
            if row_identity_duplicate_count != 0:
                raise RuntimeError(
                    "B200 tail contains duplicate (id, task) row identities"
                )
            tail_indices = [
                int(occurrence["row_index"]) for occurrence in tail_occurrences
            ]
            if tail_indices != benchmark_deck[tail_start:tail_end]:
                raise RuntimeError(
                    "B200 tail sampler identities differ from deck replay"
                )
            tail_occurrence_records = [
                {
                    key: occurrence[key]
                    for key in (
                        "global_position",
                        "epoch",
                        "position",
                        "row_index",
                        "id",
                        "task",
                        "quality",
                        "expert_ordinal",
                        "identity_sha256",
                    )
                }
                for occurrence in tail_occurrences
            ]
            canonical_tail_sha256 = canonical_sha256(tail_occurrence_records)
            if canonical_tail_sha256 != (
                "2d9059e3f0130eb99223b4cab2a22da0" "9090decfd4a4aa920a7528418e7eb14f"
            ):
                raise RuntimeError(
                    "B200 deterministic last-8 identity changed: "
                    f"{canonical_tail_sha256}"
                )
            tail_groups = group_update_rows(
                tail_indices,
                length_of=lambda index: train_lengths[index],
                buckets=microbatch_buckets,
            )
            tail_plan = planned_microbatch_counts(
                tail_indices,
                lengths=train_lengths,
                rows_per_update=tail_count,
                buckets=microbatch_buckets,
            )
            if (
                tail_plan["optimizer_updates"] != 1
                or tail_plan["occurrences"] != tail_count
                or sum(len(group) for group in tail_groups) != tail_count
            ):
                raise RuntimeError(
                    "B200 tail microbatch plan lost or repeated occurrences"
                )

            optimizer.zero_grad(set_to_none=True)
            tail_torch_rng_before = torch.get_rng_state()
            tail_cuda_rng_before = torch.cuda.get_rng_state_all()
            tail_sampler_state_before = copy.deepcopy(sampler.state_dict())
            tail_optimizer_state_before = copy.deepcopy(optimizer.state_dict())
            if tail_optimizer_state_before.get("state") != {}:
                raise RuntimeError(
                    "transactional tail requires an unallocated smoke AdamW"
                )
            tail_optimizer_state_before_sha256 = canonical_sha256(
                tail_optimizer_state_before
            )
            tail_adapter_digest_before = trainable_parameter_digest(model)
            tail_adapter_stats_before = trainable_parameter_stats(model)
            tail_frozen_probe_before = frozen_parameter_probe(model)
            if tail_frozen_probe_before != frozen_probe_before:
                raise RuntimeError("frozen probe changed before transactional tail")
            tail_named_parameters = [
                (name, parameter)
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            ]
            tail_parameter_payload_before = {
                name: parameter.detach().cpu().clone()
                for name, parameter in tail_named_parameters
            }
            tail_parameter_payload_after: dict[str, torch.Tensor] | None = None

            @torch.no_grad()
            def load_tail_parameter_payload(
                payload: dict[str, torch.Tensor],
            ) -> None:
                expected_names = {name for name, _parameter in tail_named_parameters}
                if set(payload) != expected_names:
                    raise RuntimeError("transactional tail parameter payload changed")
                for name, parameter in tail_named_parameters:
                    saved = payload[name]
                    if saved.shape != parameter.shape or saved.dtype != parameter.dtype:
                        raise RuntimeError(
                            f"transactional payload metadata changed: {name}"
                        )
                    parameter.copy_(saved.to(device=parameter.device))

            try:
                tail_learning_rate = positive_cosine_learning_rate(
                    update=tail_step,
                    total_updates=occurrence_contract["optimizer_steps"],
                    warmup_updates=warmup_steps,
                    peak_learning_rate=float(recipe["learning_rate"]),
                    minimum_learning_rate_ratio=float(
                        recipe["minimum_learning_rate_ratio"]
                    ),
                )
                if not math.isfinite(tail_learning_rate) or tail_learning_rate <= 0.0:
                    raise RuntimeError(
                        "B200 tail learning rate is not finite and positive"
                    )
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = tail_learning_rate
                tail_loss_values: list[float] = []
                model.train()
                for group in tail_groups:
                    tail_items = [
                        encoded(train_rows[index], weighted=True) for index in group
                    ]
                    tail_loss_values.extend(
                        batch_row_losses(
                            tail_items,
                            backward_scale=1.0 / tail_count,
                        )
                    )
                if len(tail_loss_values) != tail_count:
                    raise RuntimeError(
                        "B200 tail loss path did not consume exactly 8 rows"
                    )
                tail_loss = sum(tail_loss_values) / tail_count
                if not math.isfinite(tail_loss) or tail_loss < 0.0:
                    raise RuntimeError(
                        f"B200 tail produced an invalid loss: {tail_loss!r}"
                    )
                tail_gradient_before_clip = gradient_audit(parameters)
                tail_gradient_norm_before_clip = clip_global_grad_norm(
                    parameters,
                    float(recipe["max_grad_norm"]),
                )
                tail_gradient_after_clip = gradient_audit(parameters)
                maximum_gradient_norm = float(recipe["max_grad_norm"])
                if tail_gradient_after_clip["l2"] > (maximum_gradient_norm + 1e-4):
                    raise RuntimeError(
                        "B200 tail gradient clipping exceeded its maximum"
                    )
                optimizer.step()
                synchronize_cuda_devices()
                tail_optimizer_entries_after = len(optimizer.state)
                if tail_optimizer_entries_after <= 0:
                    raise RuntimeError(
                        "B200 tail AdamW did not allocate optimizer state"
                    )
                if (
                    tail_optimizer_entries_after
                    != tail_gradient_before_clip["parameters_with_gradient"]
                ):
                    raise RuntimeError(
                        "AdamW allocation differs from gradient-bearing LoRA "
                        "parameters"
                    )
                optimizer_state_tensor_count = 0
                optimizer_step_values: list[float] = []
                for optimizer_entry in optimizer.state.values():
                    for name, value in optimizer_entry.items():
                        if not isinstance(value, torch.Tensor):
                            continue
                        optimizer_state_tensor_count += 1
                        if value.is_floating_point() and not bool(
                            torch.isfinite(value).all().item()
                        ):
                            raise RuntimeError(
                                "B200 tail AdamW state contains NaN or Inf"
                            )
                        if name == "step":
                            optimizer_step_values.append(float(value.item()))
                if (
                    optimizer_state_tensor_count <= 0
                    or not optimizer_step_values
                    or any(value != 1.0 for value in optimizer_step_values)
                ):
                    raise RuntimeError(
                        "B200 tail AdamW allocation lacks first-step evidence"
                    )
                tail_adapter_stats_after = trainable_parameter_stats(model)
                tail_adapter_digest_after = trainable_parameter_digest(model)
                if not tail_adapter_stats_after["finite"]:
                    raise RuntimeError(
                        "B200 tail update produced non-finite LoRA parameters"
                    )
                if tail_adapter_digest_after == tail_adapter_digest_before:
                    raise RuntimeError(
                        "B200 tail AdamW update did not change LoRA parameters"
                    )
                if frozen_parameter_probe(model) != tail_frozen_probe_before:
                    raise RuntimeError(
                        "B200 tail update changed a frozen base parameter"
                    )
                tail_parameter_payload_after = {
                    name: parameter.detach().cpu().clone()
                    for name, parameter in tail_named_parameters
                }
                load_tail_parameter_payload(tail_parameter_payload_before)
                if trainable_parameter_digest(model) != tail_adapter_digest_before:
                    raise RuntimeError("B200 tail pre-update payload did not restore")
                load_tail_parameter_payload(tail_parameter_payload_after)
                tail_payload_reloaded_digest = trainable_parameter_digest(model)
                if tail_payload_reloaded_digest != tail_adapter_digest_after:
                    raise RuntimeError(
                        "B200 tail updated parameter payload did not reload"
                    )
                exact_tail_smoke_audit = {
                    "status": "PASS",
                    "transactional": True,
                    "optimizer_step": tail_step,
                    "sampler_slice": {
                        "api": ("GlobalAugmentedDeckSampler.occurrences"),
                        "start": tail_start,
                        "count": tail_count,
                        "end_exclusive": tail_end,
                    },
                    "global_positions": expected_global_positions,
                    "occurrences": tail_occurrence_records,
                    "occurrence_identity_sha256": canonical_sha256(
                        tail_occurrence_identities
                    ),
                    "occurrence_order_sha256": canonical_sha256(
                        tail_occurrence_records
                    ),
                    "canonical_tail_sha256": canonical_tail_sha256,
                    "row_identity_sha256": canonical_sha256(tail_row_identities),
                    "occurrence_identity_duplicate_count": (
                        occurrence_identity_duplicate_count
                    ),
                    "row_identity_duplicate_count": (row_identity_duplicate_count),
                    "nominal_step_occurrences": rows_per_update,
                    "actual_step_occurrences": tail_count,
                    "loss_denominator": tail_count,
                    "gradient_denominator": tail_count,
                    "padding_occurrences": 0,
                    "dropped_occurrences": 0,
                    "repeated_row_identities": 0,
                    "microbatch_plan": tail_plan,
                    "row_losses": tail_loss_values,
                    "mean_loss": tail_loss,
                    "learning_rate_used": tail_learning_rate,
                    "gradient_before_clip": tail_gradient_before_clip,
                    "gradient_norm_before_clip": (tail_gradient_norm_before_clip),
                    "gradient_after_clip": tail_gradient_after_clip,
                    "gradient_clip": {
                        "invoked": True,
                        "max_norm": maximum_gradient_norm,
                        "finite_nonzero_after_clip": True,
                    },
                    "adamw_state": {
                        "entries_before": 0,
                        "entries_after": tail_optimizer_entries_after,
                        "tensor_count_after": optimizer_state_tensor_count,
                        "step_values_sha256": canonical_sha256(optimizer_step_values),
                        "allocated": True,
                        "real_update": True,
                    },
                    "adapter_update": {
                        "before_sha256": tail_adapter_digest_before,
                        "after_sha256": tail_adapter_digest_after,
                        "changed": True,
                        "before": tail_adapter_stats_before,
                        "after": tail_adapter_stats_after,
                    },
                    "frozen_base_probe_unchanged": True,
                    "trainable_parameter_payload_save_reload": {
                        "status": "PASS",
                        "medium": "cpu_memory",
                        "scope": "transactional_trainable_parameters_only",
                        "saved_sha256": tail_adapter_digest_after,
                        "reloaded_sha256": tail_payload_reloaded_digest,
                        "distinct_from_final_atomic_disk_adapter_reload": True,
                    },
                    "final_main_smoke_state_restored": False,
                }
            finally:
                load_tail_parameter_payload(tail_parameter_payload_before)
                optimizer.load_state_dict(tail_optimizer_state_before)
                sampler.load_state_dict(tail_sampler_state_before)
                torch.set_rng_state(tail_torch_rng_before)
                torch.cuda.set_rng_state_all(tail_cuda_rng_before)
                optimizer.zero_grad(set_to_none=True)
                if tail_parameter_payload_after is not None:
                    del tail_parameter_payload_after
                del tail_parameter_payload_before
                torch.cuda.empty_cache()
            tail_main_state_restored = (
                trainable_parameter_digest(model) == tail_adapter_digest_before
                and frozen_parameter_probe(model) == tail_frozen_probe_before
                and canonical_sha256(optimizer.state_dict())
                == tail_optimizer_state_before_sha256
                and sampler.state_dict() == tail_sampler_state_before
                and torch.equal(torch.get_rng_state(), tail_torch_rng_before)
                and all(
                    torch.equal(observed, expected)
                    for observed, expected in zip(
                        torch.cuda.get_rng_state_all(),
                        tail_cuda_rng_before,
                        strict=True,
                    )
                )
                and all(parameter.grad is None for parameter in parameters)
            )
            if not tail_main_state_restored:
                raise RuntimeError(
                    "transactional B200 tail changed final main smoke state"
                )
            if exact_tail_smoke_audit is None:
                raise RuntimeError("B200 tail smoke produced no audit")
            exact_tail_smoke_audit["final_main_smoke_state_restored"] = True
        for benchmarks in (
            training_bucket_benchmarks,
            validation_bucket_benchmarks,
        ):
            for benchmark in benchmarks.values():
                benchmark["benchmark_state_unchanged"] = True
        for benchmark in validation_by_task.values():
            benchmark["benchmark_state_unchanged"] = True
        production_update_benchmark["benchmark_state_unchanged"] = True

        validation_benchmark = {
            "warmup": ("one excluded shape-matched run before three timed repeats"),
            "selection": "absolute longest validation row per task",
            "by_task": validation_by_task,
            "maximum_seconds": validation_row_seconds,
            "training_buckets": training_bucket_benchmarks,
            "validation_buckets": validation_bucket_benchmarks,
            "production_update_benchmark": production_update_benchmark,
            "benchmark_state_unchanged": True,
        }
        print(
            json.dumps(
                {"validation_runtime_benchmark": validation_benchmark},
                ensure_ascii=False,
            ),
            flush=True,
        )

    production_validation_subset = stratified_validation_subset(
        validation_rows,
        per_task=int(recipe["validation"]["subset_per_task"]),
        seed=recipe["seed"],
    )
    validation_subset = (
        stratified_validation_subset(
            validation_rows,
            per_task=1,
            seed=recipe["seed"],
        )
        if args.smoke
        else production_validation_subset
    )
    validation_index_by_identity = {
        (str(row["id"]), str(row["task"])): index
        for index, row in enumerate(validation_rows)
    }
    production_subset_indices = [
        validation_index_by_identity[(str(row["id"]), str(row["task"]))]
        for row in production_validation_subset
    ]
    subset_validation_plan = planned_microbatch_counts(
        production_subset_indices,
        lengths=validation_lengths,
        rows_per_update=len(production_subset_indices),
        buckets=microbatch_buckets,
    )
    full_validation_plan = planned_microbatch_counts(
        range(len(validation_rows)),
        lengths=validation_lengths,
        rows_per_update=len(validation_rows),
        buckets=microbatch_buckets,
    )
    validation_order_sha256 = canonical_sha256(
        [
            [str(row["id"]), str(row["task"]), validation_lengths[index]]
            for index, row in enumerate(validation_rows)
        ]
    )
    validation_progress_path = output_dir / "validation_progress.json"

    @torch.no_grad()
    def validate(step: int, *, full: bool) -> dict[str, Any] | None:
        model.eval()
        source = validation_rows if full else validation_subset
        indexed_source = [
            (
                validation_index_by_identity[(str(row["id"]), str(row["task"]))],
                row,
            )
            for row in source
        ]
        groups = group_update_rows(
            indexed_source,
            length_of=lambda entry: validation_lengths[entry[0]],
            buckets=microbatch_buckets,
        )
        scope = "full" if full else "fixed_subset"
        if last_recovery_step != step or last_recovery_trainable_sha256 is None:
            raise RuntimeError(
                "validation must be bound to a durable recovery checkpoint"
            )
        progress_identity = {
            "schema_version": 2,
            "step": step,
            "scope": scope,
            "state_contract_sha256": state_contract_sha256,
            "trainable_parameter_sha256": (last_recovery_trainable_sha256),
            "validation_order_sha256": validation_order_sha256,
            "source_rows": len(source),
            "group_count": len(groups),
            "max_length": max_length,
        }
        next_group = 0
        loss_sums = Counter({"translation": 0.0, "summarization": 0.0})
        counts = Counter({"translation": 0, "summarization": 0})
        if validation_progress_path.is_file():
            try:
                progress = read_json(validation_progress_path)
            except Exception as error:
                print(
                    f"[validation] discarding unreadable progress: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
                validation_progress_path.unlink()
            else:
                if progress.get("identity") != progress_identity:
                    print(
                        "[validation] discarding progress from a different "
                        "checkpoint or validation scope",
                        flush=True,
                    )
                    validation_progress_path.unlink()
                else:
                    next_group = int(progress["next_group"])
                    loss_sums = Counter(
                        {
                            task: float(progress["loss_sums"][task])
                            for task in ("translation", "summarization")
                        }
                    )
                    counts = Counter(
                        {
                            task: int(progress["counts"][task])
                            for task in ("translation", "summarization")
                        }
                    )

        def save_validation_progress(position: int) -> None:
            atomic_json(
                validation_progress_path,
                {
                    "identity": progress_identity,
                    "next_group": position,
                    "loss_sums": dict(loss_sums),
                    "counts": dict(counts),
                },
            )

        started = time.time()
        for group_index in range(next_group, len(groups)):
            if stop_requested:
                save_validation_progress(group_index)
                model.train()
                torch.cuda.empty_cache()
                return None
            group = groups[group_index]
            rows = [entry[1] for entry in group]
            items = [encoded(row, weighted=False) for row in rows]
            values = batch_row_losses(items, backward_scale=None)
            for row, value in zip(rows, values, strict=True):
                task = str(row["task"])
                loss_sums[task] += value
                counts[task] += 1
            completed_groups = group_index + 1
            if completed_groups % 25 == 0:
                save_validation_progress(completed_groups)
            completed_rows = sum(counts.values())
            if completed_rows and completed_rows % 250 < len(rows):
                print(
                    f"[validation] step={step} full={full} "
                    f"{completed_rows}/{len(source)}",
                    flush=True,
                )
        if any(counts[task] <= 0 for task in ("translation", "summarization")):
            raise RuntimeError("validation has no eligible rows for one or more tasks")
        means = {
            task: float(loss_sums[task]) / int(counts[task])
            for task in ("translation", "summarization")
        }
        selection_weights = recipe["validation"]["selection_weights"]
        result = {
            "step": step,
            "scope": scope,
            "selection_loss": (
                selection_weights["translation"] * means["translation"]
                + selection_weights["summarization"] * means["summarization"]
            ),
            "translation_loss": means["translation"],
            "summarization_loss": means["summarization"],
            "translation_count": counts["translation"],
            "summarization_count": counts["summarization"],
            "dropped_over_max": {},
            "validation_order_sha256": validation_order_sha256,
            "elapsed_seconds": round(time.time() - started, 2),
        }
        if validation_progress_path.exists():
            validation_progress_path.unlink()
        model.train()
        torch.cuda.empty_cache()
        print(json.dumps({"validation": result}, ensure_ascii=False), flush=True)
        return result

    wandb_run = None
    wandb_completion_status = (
        "DISABLED_FOR_SMOKE" if wandb_mode == "disabled" else "PENDING"
    )

    run_started = time.time()
    last_adapter_dir: Path | None = (
        Path(resume_state["adapter_dir"]) if resume_state is not None else None
    )
    last_recovery_step: int | None = (
        int(resume_state["completed_steps"]) if resume_state is not None else None
    )
    last_recovery_adapter_dir: Path | None = (
        Path(resume_state["adapter_dir"]) if resume_state is not None else None
    )
    last_recovery_adapter_files: dict[str, str] | None = (
        dict(resume_state["adapter_files_sha256"]) if resume_state is not None else None
    )
    last_recovery_trainable_sha256: str | None = (
        str(resume_state["trainable_parameter_sha256"])
        if resume_state is not None
        else None
    )
    candidate_checkpoints: dict[str, dict[str, Any]] = (
        {
            str(step): dict(value)
            for step, value in resume_state.get("candidate_checkpoints", {}).items()
        }
        if resume_state is not None
        else {}
    )
    final_checkpoint_record: dict[str, Any] | None = None
    selection_record_value: dict[str, Any] | None = None
    selected_checkpoint_value: dict[str, Any] | None = None

    def adapter_files_sha256(directory: Path) -> dict[str, str]:
        return {
            path.name: sha256(path)
            for path in sorted(directory.iterdir())
            if path.is_file()
        }

    def publish_adapter(temporary: Path, destination: Path) -> None:
        if destination.exists():
            raise RuntimeError(f"refusing to replace published adapter: {destination}")
        success = temporary / "_SUCCESS"
        if success.exists():
            success.unlink()
        for path in sorted(temporary.rglob("*")):
            if not path.is_file():
                continue
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        with success.open("w", encoding="utf-8") as handle:
            handle.write("PASS\n")
            handle.flush()
            os.fsync(handle.fileno())
        directory_descriptor = os.open(
            temporary,
            os.O_RDONLY | os.O_DIRECTORY,
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        os.replace(temporary, destination)
        parent_descriptor = os.open(
            destination.parent,
            os.O_RDONLY | os.O_DIRECTORY,
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)

    def save_immutable_adapter(
        destination: Path,
        *,
        extra_text_files: dict[str, str] | None = None,
    ) -> dict[str, str]:
        nonlocal last_adapter_dir
        extra_text_files = extra_text_files or {}
        if destination.exists():
            if not (destination / "_SUCCESS").is_file():
                raise RuntimeError(f"immutable adapter is incomplete: {destination}")
            for name, expected_text in extra_text_files.items():
                path = destination / name
                if (
                    not path.is_file()
                    or path.read_text(encoding="utf-8") != expected_text
                ):
                    raise RuntimeError(f"immutable adapter metadata changed: {path}")
            last_adapter_dir = destination
            return adapter_files_sha256(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        model.save_pretrained(temporary, safe_serialization=True)
        tokenizer.save_pretrained(temporary)
        for name, text_value in extra_text_files.items():
            (temporary / name).write_text(text_value, encoding="utf-8")
        publish_adapter(temporary, destination)
        last_adapter_dir = destination
        return adapter_files_sha256(destination)

    def recovery_state(
        step: int,
        adapter_dir: Path,
        adapter_files: dict[str, str],
        trainable_sha256: str,
    ) -> dict[str, Any]:
        return {
            "state_contract_sha256": state_contract_sha256,
            "state_contract": state_contract,
            "completed_steps": step,
            "occurrence_boundary": {
                "completed_steps": step,
                "consumed_occurrences": consumed_occurrences_at_step(
                    recipe,
                    step,
                ),
                "next_step_occurrences": (
                    update_occurrences_at_step(recipe, step + 1)
                    if step < occurrence_contract["optimizer_steps"]
                    else 0
                ),
            },
            "adapter_dir": str(adapter_dir),
            "adapter_files_sha256": adapter_files,
            "trainable_parameter_sha256": trainable_sha256,
            "optimizer": optimizer.state_dict(),
            "sampler": sampler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "history": history,
            "wandb_logged_validation_keys": sorted(wandb_logged_validation_keys),
            "candidate_checkpoints": candidate_checkpoints,
            "backward_passes": backward_passes,
            "consumed_occurrences": consumed_occurrences,
            "occurrence_task_counts": dict(occurrence_task_counts),
            "occurrence_quality_counts": dict(occurrence_quality_counts),
            "max_observed_tokens": max_observed_tokens,
            "total_hard_term_weighted_tokens": total_hard_term_weighted_tokens,
            "first_update_gradient_audit": first_update_gradient_audit,
            "first_update_adapter_delta": first_update_adapter_delta,
            "frozen_probe_after_update": frozen_probe_after_update,
            "first_output_only_supervision": first_output_only_supervision,
            "microbatch_phase_seconds": microbatch_phase_seconds,
            "validation_row_seconds": validation_row_seconds,
            "validation_benchmark": validation_benchmark,
            "optimizer_boundary_seconds": optimizer_boundary_seconds,
            "one_time_gradient_audit_seconds": (one_time_gradient_audit_seconds),
            "runtime_budget_projection": projected_runtime_budget,
            "post_update_recompute_audit": post_update_recompute_audit,
            "exact_tail_smoke_audit": exact_tail_smoke_audit,
            "recovery_generation": recovery_generation,
        }

    def save_checkpoint(step: int) -> Path:
        nonlocal recovery_generation, last_recovery_monotonic
        nonlocal last_recovery_step, last_recovery_adapter_dir
        nonlocal last_recovery_adapter_files
        nonlocal last_recovery_trainable_sha256
        expected_consumed = consumed_occurrences_at_step(recipe, step)
        if consumed_occurrences != expected_consumed:
            raise RuntimeError(
                "checkpoint occurrence ledger differs from its exact step: "
                f"{consumed_occurrences} != {expected_consumed}"
            )
        if backward_passes != expected_consumed:
            raise RuntimeError(
                "checkpoint backward ledger differs from consumed occurrences"
            )
        if not args.smoke and sampler.cursor != expected_consumed:
            raise RuntimeError("checkpoint sampler cursor differs from exact coverage")
        same_model_generation = (
            last_recovery_step == step
            and last_recovery_adapter_dir is not None
            and (last_recovery_adapter_dir / "_SUCCESS").is_file()
            and last_recovery_adapter_files is not None
            and last_recovery_trainable_sha256 is not None
        )
        if same_model_generation:
            destination = last_recovery_adapter_dir
            adapter_files = last_recovery_adapter_files
            trainable_sha256 = last_recovery_trainable_sha256
            if adapter_files_sha256(destination) != adapter_files:
                raise RuntimeError("current recovery adapter hashes changed")
        else:
            adapter_stats = trainable_parameter_stats(model)
            if not adapter_stats["finite"]:
                raise RuntimeError("refusing to checkpoint non-finite LoRA parameters")
            for optimizer_entry in optimizer.state.values():
                for value in optimizer_entry.values():
                    if (
                        isinstance(value, torch.Tensor)
                        and value.is_floating_point()
                        and not bool(torch.isfinite(value).all().item())
                    ):
                        raise RuntimeError(
                            "refusing to checkpoint non-finite optimizer state"
                        )
            trainable_sha256 = trainable_parameter_digest(model)
            recovery_root = output_dir / "recovery"
            on_disk_generations = [recovery_generation]
            if recovery_root.is_dir():
                for path in recovery_root.iterdir():
                    marker = "-generation-"
                    if marker not in path.name or path.name.startswith("."):
                        continue
                    try:
                        on_disk_generations.append(int(path.name.rsplit(marker, 1)[1]))
                    except ValueError:
                        continue
            recovery_generation = max(on_disk_generations) + 1
            destination = (
                recovery_root / f"step-{step:05d}-generation-{recovery_generation:06d}"
            )
            adapter_files = save_immutable_adapter(
                destination,
                extra_text_files={
                    "STATE_CONTRACT_SHA256": (f"{state_contract_sha256}\n"),
                    "COMPLETED_OPTIMIZER_STEP": f"{step}\n",
                    "TRAINABLE_PARAMETER_SHA256": (f"{trainable_sha256}\n"),
                },
            )
            last_recovery_step = step
            last_recovery_adapter_dir = destination
            last_recovery_adapter_files = adapter_files
            last_recovery_trainable_sha256 = trainable_sha256
        state = recovery_state(
            step,
            destination,
            adapter_files,
            trainable_sha256,
        )
        next_state = output_dir / f".resume-next.{os.getpid()}.pt"
        atomic_torch_save(next_state, state)
        if resume_path.is_file() and not same_model_generation:
            os.replace(resume_path, previous_resume_path)
        os.replace(next_state, resume_path)
        descriptor = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        referenced: set[Path] = {destination}
        if previous_resume_path.is_file():
            previous_state = torch.load(
                previous_resume_path,
                map_location="cpu",
                weights_only=False,
            )
            referenced.add(Path(str(previous_state["adapter_dir"])))
        recovery_root = output_dir / "recovery"
        for path in recovery_root.iterdir():
            if path.is_dir() and path not in referenced:
                shutil.rmtree(path)
        last_recovery_monotonic = time.monotonic()
        return destination

    def save_candidate_checkpoint(step: int) -> dict[str, Any]:
        key = str(step)
        destination = output_dir / "candidates" / f"step-{step:05d}"
        parameter_sha256 = trainable_parameter_digest(model)
        files = save_immutable_adapter(
            destination,
            extra_text_files={"TRAINABLE_PARAMETER_SHA256": f"{parameter_sha256}\n"},
        )
        if not any(
            int(item["step"]) == step
            for item in recipe["validation"]["full_validation_candidates"]
        ):
            raise RuntimeError(f"step {step} is not a configured candidate")
        record = {
            "step": step,
            "adapter_dir": str(destination),
            "files_sha256": files,
        }
        previous = candidate_checkpoints.get(key)
        if previous is not None and previous != record:
            raise RuntimeError(f"candidate checkpoint identity changed at step {step}")
        candidate_checkpoints[key] = record
        return record

    def validate_candidate_checkpoints() -> None:
        for key, record in candidate_checkpoints.items():
            if str(record.get("step")) != key:
                raise RuntimeError(f"candidate checkpoint key differs from step: {key}")
            directory = resolve_run_path(Path(str(record.get("adapter_dir") or "")))
            if not (directory / "_SUCCESS").is_file():
                raise RuntimeError(f"candidate checkpoint is incomplete: {directory}")
            if adapter_files_sha256(directory) != record.get("files_sha256"):
                raise RuntimeError(f"candidate checkpoint hashes changed: {directory}")

    def manifest(status: str) -> dict[str, Any]:
        early_stop_evidence = derive_early_stopping_decision(
            history,
            completed_steps,
        )
        production_complete = (
            not args.smoke
            and completed_steps == int(recipe["optimizer_steps"])
            and consumed_occurrences
            == int(recipe["training_deck"]["total_occurrences"]["total"])
        )
        production_early_stopped = (
            not args.smoke
            and early_stop_evidence["triggered"] is True
            and completed_steps == early_terminal_step
        )
        production_terminal = production_complete or production_early_stopped
        expected_occurrences = int(
            recipe["training_deck"]["total_occurrences"]["total"]
        )
        return {
            "status": status,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "model_name": model_contract["derivative_model_name"],
            "attribution": model_contract["attribution"],
            "source_dir": str(source_dir),
            "expert_nf4_dir": str(expert_nf4_dir),
            "output_dir": str(output_dir),
            "profile": training_config["profile"],
            "config_identity": training_config["identity"],
            "config": recipe,
            "occurrence_contract": occurrence_contract,
            "output_only_contract": recipe["output_only_contract"],
            "wandb": wandb_identity,
            "wandb_reporting": {
                "logged_validation_keys": sorted(wandb_logged_validation_keys),
                "completion_status": wandb_completion_status,
            },
            "state_contract": state_contract,
            "completed_updates": completed_steps,
            "target_updates": total_steps,
            "early_stopping_decision": early_stop_evidence,
            "backward_passes": backward_passes,
            "consumed_occurrences": consumed_occurrences,
            "training_completion": {
                "status": (
                    "EARLY_STOPPED"
                    if production_early_stopped
                    else ("COMPLETED" if production_complete else status)
                ),
                "epochs": int(recipe["epochs"]),
                "completed_epochs": early_stop_evidence["completed_epochs"],
                "early_stopping_triggered": production_early_stopped,
                "optimizer_steps": completed_steps,
                "consumed_occurrences": consumed_occurrences,
                "expected_optimizer_steps": int(recipe["optimizer_steps"]),
                "expected_occurrences": expected_occurrences,
            },
            "training_coverage": {
                "total_occurrences": consumed_occurrences,
                "skipped_occurrences": 0,
                "unconsumed_planned_occurrences": (
                    expected_occurrences - consumed_occurrences
                ),
                "unique_raw_rows": (
                    len(train_rows)
                    if production_terminal
                    and consumed_occurrences
                    >= int(recipe["training_deck"]["occurrences_per_epoch"]["total"])
                    else None
                ),
                "task_occurrences": dict(occurrence_task_counts),
                "quality_occurrences": dict(occurrence_quality_counts),
            },
            "token_length_contract": {
                "train": train_length_contract,
                "validation": validation_length_contract,
            },
            "production_training_plan": production_training_plan,
            "training_performance": training_performance,
            "training_performance_sha256": canonical_sha256(training_performance),
            "gradient_checkpointing_contract_sha256": canonical_sha256(
                gradient_checkpointing_contract
            ),
            "subset_validation_plan": subset_validation_plan,
            "full_validation_plan": full_validation_plan,
            "validation_resume_contract": {
                "schema_version": 2,
                "progress_path": str(validation_progress_path),
                "commit_every_groups": 25,
                "bound_to_trainable_parameter_sha256": True,
                "signal_safe": True,
                "zero_drop": True,
            },
            "max_observed_tokens": max_observed_tokens,
            "total_hard_term_weighted_tokens": total_hard_term_weighted_tokens,
            "first_update_gradient_audit": first_update_gradient_audit,
            "first_update_adapter_delta": first_update_adapter_delta,
            "first_output_only_supervision": first_output_only_supervision,
            "microbatch_phase_seconds": microbatch_phase_seconds,
            "validation_row_seconds": validation_row_seconds,
            "validation_benchmark": validation_benchmark,
            "optimizer_boundary_seconds": optimizer_boundary_seconds,
            "one_time_gradient_audit_seconds": (one_time_gradient_audit_seconds),
            "runtime_budget_projection": projected_runtime_budget,
            "gradient_checkpointing_contract": (gradient_checkpointing_contract),
            "post_update_recompute_audit": post_update_recompute_audit,
            "exact_tail_smoke_audit": exact_tail_smoke_audit,
            "frozen_probe_before": frozen_probe_before,
            "frozen_probe_after_update": frozen_probe_after_update,
            "history": history,
            "candidate_checkpoints": candidate_checkpoints,
            "final_checkpoint": final_checkpoint_record,
            "selection_record": selection_record_value,
            "selected_checkpoint": selected_checkpoint_value,
            "split_roles": {
                "train": "optimizer updates",
                "validation": "monitoring and checkpoint selection only",
                "full_test": "sealed final evaluation only",
                "human_gold": "sealed final evaluation only",
            },
            "sampler": sampler.stats(),
            "lora_module_audit": module_audit,
            "trainable_parameters": trainable_parameters,
            "device_audit": device_audit,
            "loader_audit": loader_audit,
            "storage_audit": storage_audit,
            "elapsed_seconds": round(time.time() - run_started, 2),
        }

    if resume_state is None:
        save_checkpoint(0)
        atomic_json(
            output_dir / "run_manifest.json",
            manifest("INITIALIZED"),
        )

    def initialize_wandb_run() -> Any:
        import wandb

        wandb_config_contract: dict[str, Any] = {}
        if b200:
            wandb_config_contract = {
                "b200_wandb_contract_schema": "solar_open2_b200_wandb_training_v2",
                "profile": "b200",
                "recipe_sha256": state_contract["recipe_sha256"],
                "data_contract_sha256": state_contract["data_contract_sha256"],
                "pipeline_code_sha256": state_contract["pipeline_code_sha256"],
                "state_contract_sha256": state_contract_sha256,
                "training_performance_sha256": canonical_sha256(training_performance),
                "gradient_checkpointing_contract_sha256": canonical_sha256(
                    gradient_checkpointing_contract
                ),
                "checkpointed_decoder_layer_count": training_performance[
                    "checkpointed_decoder_layer_count"
                ],
                "uncheckpointed_decoder_layer_count": training_performance[
                    "uncheckpointed_decoder_layer_count"
                ],
                "checkpointed_decoder_layer_indices": training_performance[
                    "checkpointed_decoder_layer_indices"
                ],
                "uncheckpointed_decoder_layer_indices": training_performance[
                    "uncheckpointed_decoder_layer_indices"
                ],
                "runtime_gate_sha256": canonical_sha256(recipe["runtime_gate"]),
                "total_occurrences": occurrence_contract["total_occurrences"],
                "optimizer_steps": occurrence_contract["optimizer_steps"],
                "final_step_occurrences": occurrence_contract["final_step_occurrences"],
                "activation_checkpointing": training_performance[
                    "activation_checkpointing"
                ],
                "logit_chunk_size": int(training_performance["logit_chunk_size"]),
            }
        run = wandb.init(
            entity=wandb_identity["entity"],
            project=wandb_identity["project"],
            id=wandb_identity["run_id"],
            resume=wandb_identity["resume"],
            name=wandb_identity["run_name"],
            mode=wandb_mode,
            dir=str(output_dir / "wandb"),
            tags=["solar-open2", "lora", "translation", "summarization"],
            config={
                "recipe": recipe,
                "model": model_contract,
                "data_contract_sha256": state_contract["data_contract_sha256"],
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                **wandb_config_contract,
            },
        )
        actual_identity = {
            "entity": run.entity,
            "project": run.project,
            "run_id": run.id,
        }
        expected_identity = {
            key: wandb_identity[key] for key in ("entity", "project", "run_id")
        }
        if actual_identity != expected_identity:
            raise RuntimeError(
                "W&B initialized a different run identity: "
                f"{actual_identity!r} != {expected_identity!r}"
            )
        if b200:
            remote_resumed = bool(getattr(run, "resumed", False))
            local_resumed = resume_state is not None
            if remote_resumed != local_resumed:
                try:
                    run.finish(exit_code=1)
                except Exception:
                    pass
                raise RuntimeError(
                    "W&B/local recovery state differs: "
                    f"remote_resumed={remote_resumed}, "
                    f"local_resumed={local_resumed}"
                )
            observed_wandb_contract = {
                key: run.config.get(key) for key in wandb_config_contract
            }
            if observed_wandb_contract != wandb_config_contract:
                try:
                    run.finish(exit_code=1)
                except Exception:
                    pass
                raise RuntimeError(
                    "W&B resumed with a stale B200 training contract: "
                    f"{observed_wandb_contract!r} != {wandb_config_contract!r}"
                )
        run.define_metric("optimizer_step")
        run.define_metric("train/*", step_metric="optimizer_step")
        run.define_metric("validation/*", step_metric="optimizer_step")
        return run

    if wandb_mode != "disabled":
        try:
            wandb_run = initialize_wandb_run()
        except Exception as error:
            print(
                f"[wandb] initialization failed; requesting retry: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            atomic_json(
                output_dir / "run_manifest.json",
                manifest("WANDB_RETRY"),
            )
            return REQUEUE_EXIT_CODE

    eval_every = int(recipe["validation"]["eval_every_steps"])
    save_every = int(recipe["checkpoint"]["save_every_steps"])
    save_every_seconds = float(recipe["checkpoint"]["save_every_hours"]) * 60.0 * 60.0
    full_steps = {int(step) for step in recipe["validation"]["full_validation_steps"]}
    candidate_steps = sorted(full_steps)
    configured_candidate_steps = [
        int(item["step"]) for item in recipe["validation"]["full_validation_candidates"]
    ]
    if candidate_steps != configured_candidate_steps:
        raise RuntimeError("full-validation steps differ from candidate metadata")
    if not args.smoke and candidate_steps[-1] != total_steps:
        raise RuntimeError("final update must be a full-validation candidate")

    def validation_scope(step: int) -> str | None:
        return configured_validation_scope(
            recipe,
            step,
            smoke=args.smoke,
        )

    def log_validation(result: dict[str, Any]) -> None:
        if wandb_run is None:
            return
        scope = str(result["scope"])
        key = f"{int(result['step'])}:{scope}"
        if key in wandb_logged_validation_keys:
            return
        wandb_run.log(
            {
                "optimizer_step": int(result["step"]),
                f"validation/{scope}/selection_loss": result["selection_loss"],
                f"validation/{scope}/translation_loss": result["translation_loss"],
                f"validation/{scope}/summarization_loss": result["summarization_loss"],
            }
        )
        wandb_logged_validation_keys.add(key)

    def finish_wandb_for_retry() -> None:
        if wandb_run is None:
            return
        try:
            wandb_run.finish(exit_code=REQUEUE_EXIT_CODE)
        except Exception as error:
            print(
                f"[wandb] retry finish failed: " f"{type(error).__name__}: {error}",
                flush=True,
            )

    if wandb_run is not None:
        try:
            pending_before = len(wandb_logged_validation_keys)
            for validation_result in history:
                log_validation(validation_result)
            if len(wandb_logged_validation_keys) != pending_before:
                save_checkpoint(completed_steps)
        except Exception as error:
            print(
                f"[wandb] validation replay failed; requesting retry: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            atomic_json(
                output_dir / "run_manifest.json",
                manifest("WANDB_RETRY"),
            )
            return REQUEUE_EXIT_CODE

    model.train()

    # Complete an interrupted validation before advancing to the next update.
    completed_history = {(item["step"], item["scope"]) for item in history}
    pending_scope = validation_scope(completed_steps)
    if pending_scope is not None:
        wanted = (completed_steps, pending_scope)
        if wanted not in completed_history:
            if pending_scope == "full":
                save_checkpoint(completed_steps)
                save_candidate_checkpoint(completed_steps)
            save_checkpoint(completed_steps)
            result = validate(
                completed_steps,
                full=pending_scope == "full",
            )
            if result is None:
                save_checkpoint(completed_steps)
                atomic_json(
                    output_dir / "run_manifest.json",
                    manifest("INTERRUPTED"),
                )
                finish_wandb_for_retry()
                return REQUEUE_EXIT_CODE
            history.append(result)
            save_checkpoint(completed_steps)
            try:
                log_validation(result)
            except Exception as error:
                print(
                    f"[wandb] validation log failed; requesting retry: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
                atomic_json(
                    output_dir / "run_manifest.json",
                    manifest("WANDB_RETRY"),
                )
                return REQUEUE_EXIT_CODE
            if wandb_run is not None:
                save_checkpoint(completed_steps)

    decision_before_training_loop = derive_early_stopping_decision(
        history,
        completed_steps,
    )
    if not args.smoke and decision_before_training_loop["triggered"] is True:
        if completed_steps != early_terminal_step:
            raise RuntimeError(
                "durable early-stop decision is not at the approved boundary"
            )
        atomic_json(
            output_dir / "run_manifest.json",
            manifest("EARLY_STOPPING_DECIDED"),
        )
    training_loop_target = (
        completed_steps
        if (not args.smoke and decision_before_training_loop["triggered"] is True)
        else total_steps
    )

    def save_interrupted_boundary() -> int:
        optimizer.zero_grad(set_to_none=True)
        save_checkpoint(completed_steps)
        atomic_json(output_dir / "run_manifest.json", manifest("INTERRUPTED"))
        finish_wandb_for_retry()
        return REQUEUE_EXIT_CODE

    for step in range(completed_steps + 1, training_loop_target + 1):
        if stop_requested:
            return save_interrupted_boundary()
        rollback_sampler = copy.deepcopy(sampler.state_dict())
        rollback_torch_rng = torch.get_rng_state()
        rollback_cuda_rng = torch.cuda.get_rng_state_all()
        rollback_backward_passes = backward_passes
        rollback_max_observed_tokens = max_observed_tokens
        rollback_first_supervision = copy.deepcopy(first_output_only_supervision)
        update_started = time.time()
        optimizer.zero_grad(set_to_none=True)
        optimizer_lr_before_step = positive_cosine_learning_rate(
            update=step,
            total_updates=int(recipe["optimizer_steps"]),
            warmup_updates=warmup_steps,
            peak_learning_rate=float(recipe["learning_rate"]),
            minimum_learning_rate_ratio=float(recipe["minimum_learning_rate_ratio"]),
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = optimizer_lr_before_step
        update_indices = next_update_indices()
        update_occurrences = len(update_indices)
        expected_update_occurrences = update_occurrences_at_step(recipe, step)
        if update_occurrences != expected_update_occurrences:
            raise RuntimeError("optimizer group occurrence count differs from its step")
        update_groups = group_update_rows(
            update_indices,
            length_of=lambda index: train_lengths[index],
            buckets=microbatch_buckets,
        )
        update_loss_sum = 0.0
        task_counts: Counter[str] = Counter()
        quality_counts: Counter[str] = Counter()
        weighted_tokens = 0
        interrupted_partial_update = False
        for group in update_groups:
            if stop_requested:
                interrupted_partial_update = True
                break
            rows = [train_rows[index] for index in group]
            items = [encoded(row, weighted=True) for row in rows]
            values = batch_row_losses(
                items,
                backward_scale=1.0 / update_occurrences,
            )
            update_loss_sum += sum(values)
            for row, item in zip(rows, items, strict=True):
                if first_output_only_supervision is None:
                    first_output_only_supervision = dict(
                        item["output_only_supervision"]
                    )
                max_observed_tokens = max(
                    max_observed_tokens,
                    int(item["length"]),
                )
                weighted_tokens += int(item["hard_term_weighted_tokens"])
                task_counts[str(row["task"])] += 1
                quality_counts[str(row["quality"])] += 1
                backward_passes += 1
            if stop_requested:
                interrupted_partial_update = True
                break
        if interrupted_partial_update:
            optimizer.zero_grad(set_to_none=True)
            sampler.load_state_dict(rollback_sampler)
            torch.set_rng_state(rollback_torch_rng)
            torch.cuda.set_rng_state_all(rollback_cuda_rng)
            backward_passes = rollback_backward_passes
            max_observed_tokens = rollback_max_observed_tokens
            first_output_only_supervision = rollback_first_supervision
            return save_interrupted_boundary()
        if sum(task_counts.values()) != update_occurrences:
            raise RuntimeError("optimizer group did not consume its exact occurrences")
        update_loss = update_loss_sum / update_occurrences
        if (
            not math.isfinite(optimizer_lr_before_step)
            or optimizer_lr_before_step <= 0.0
        ):
            raise RuntimeError(
                f"invalid optimizer learning rate: {optimizer_lr_before_step!r}"
            )
        capture_effective_update = (
            first_update_adapter_delta is None and optimizer_lr_before_step > 0.0
        )
        effective_update_before: tuple[dict[str, Any], str] | None = None
        if capture_effective_update:
            adapter_stats_before_update = trainable_parameter_stats(model)
            if not adapter_stats_before_update["finite"]:
                raise RuntimeError(
                    "LoRA parameters are non-finite before the first "
                    "positive-LR optimizer step"
                )
            effective_update_before = (
                adapter_stats_before_update,
                trainable_parameter_digest(model),
            )
        if capture_effective_update:
            if args.smoke:
                synchronize_cuda_devices()
                gradient_audit_started = time.perf_counter()
            first_update_gradient_audit = gradient_audit(parameters)
            if args.smoke:
                synchronize_cuda_devices()
                one_time_gradient_audit_seconds = (
                    time.perf_counter() - gradient_audit_started
                )
        measure_boundary = args.smoke and optimizer_boundary_seconds is None
        if measure_boundary:
            synchronize_cuda_devices()
            boundary_started = time.perf_counter()
        grad_norm = clip_global_grad_norm(parameters, float(recipe["max_grad_norm"]))
        optimizer.step()
        synchronize_cuda_devices()
        if measure_boundary:
            optimizer_boundary_seconds = time.perf_counter() - boundary_started
            if microbatch_phase_seconds is None:
                raise RuntimeError("smoke microbatch timing is missing")
            if validation_row_seconds is None:
                raise RuntimeError("smoke validation timing is missing")
            if one_time_gradient_audit_seconds is None:
                raise RuntimeError("smoke gradient audit timing is missing")
            if production_update_benchmark is None:
                raise RuntimeError("smoke production update benchmark is missing")
            projected_runtime_budget = runtime_budget_projection(
                training_bucket_benchmarks=training_bucket_benchmarks,
                validation_bucket_benchmarks=(validation_bucket_benchmarks),
                production_update_benchmark=production_update_benchmark,
                validation_row_seconds=validation_row_seconds,
                optimizer_boundary_seconds=optimizer_boundary_seconds,
                one_time_gradient_audit_seconds=(one_time_gradient_audit_seconds),
                recipe=recipe,
                validation_row_count=len(validation_rows),
                production_training_plan=production_training_plan,
                subset_validation_plan=subset_validation_plan,
                full_validation_plan=full_validation_plan,
            )
        completed_steps = step
        consumed_occurrences = consumed_occurrences_at_step(recipe, step)
        occurrence_task_counts.update(task_counts)
        occurrence_quality_counts.update(quality_counts)
        total_hard_term_weighted_tokens += weighted_tokens
        if capture_effective_update:
            if effective_update_before is None:
                raise RuntimeError("missing positive-LR pre-update LoRA audit")
            adapter_stats_before_update, adapter_digest_before_update = (
                effective_update_before
            )
            adapter_stats_after = trainable_parameter_stats(model)
            adapter_digest_after = trainable_parameter_digest(model)
            if not adapter_stats_after["finite"]:
                raise RuntimeError(
                    "positive-LR optimizer step produced non-finite LoRA "
                    f"parameters: step={step} lr={optimizer_lr_before_step}"
                )
            if adapter_digest_after == adapter_digest_before_update:
                raise RuntimeError(
                    "positive-LR optimizer step did not change LoRA parameters: "
                    f"step={step} lr={optimizer_lr_before_step}"
                )
            first_update_adapter_delta = {
                "before_sha256": adapter_digest_before_update,
                "after_sha256": adapter_digest_after,
                "changed": True,
                "optimizer_step": step,
                "learning_rate_used": optimizer_lr_before_step,
                "before": adapter_stats_before_update,
                "after": adapter_stats_after,
            }
            frozen_probe_after_update = frozen_parameter_probe(model)
            if frozen_probe_after_update != frozen_probe_before:
                raise RuntimeError("frozen base parameter probe changed after update")
        next_learning_rate = (
            positive_cosine_learning_rate(
                update=step + 1,
                total_updates=int(recipe["optimizer_steps"]),
                warmup_updates=warmup_steps,
                peak_learning_rate=float(recipe["learning_rate"]),
                minimum_learning_rate_ratio=float(
                    recipe["minimum_learning_rate_ratio"]
                ),
            )
            if step < int(recipe["optimizer_steps"])
            else 0.0
        )
        update_seconds = time.time() - update_started
        log = {
            "train/loss": update_loss,
            "train/lr": optimizer_lr_before_step,
            "train/next_lr": next_learning_rate,
            "train/grad_norm": grad_norm,
            "train/update_seconds": update_seconds,
            "train/hard_term_weighted_tokens": weighted_tokens,
            "train/translation_occurrences": task_counts["translation"],
            "train/summarization_occurrences": task_counts["summarization"],
            "train/microbatches": len(update_groups),
            "train/update_occurrences": update_occurrences,
            "train/consumed_occurrences": consumed_occurrences,
        }
        if step == 1 or step % 10 == 0 or step == total_steps:
            compute_only_eta_hours = (
                update_seconds * max(0, total_steps - step) / 3600.0
            )
            print(
                json.dumps(
                    {
                        "step": step,
                        "total_steps": total_steps,
                        "loss": update_loss,
                        "grad_norm": grad_norm,
                        "learning_rate": optimizer_lr_before_step,
                        "next_learning_rate": next_learning_rate,
                        "seconds": round(update_seconds, 2),
                        "eta_hours_latest_step_compute_only": round(
                            compute_only_eta_hours,
                            2,
                        ),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        if wandb_run is not None:
            try:
                wandb_run.log({"optimizer_step": step, **log})
            except Exception as error:
                print(
                    f"[wandb] training log failed; requesting retry: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
                save_checkpoint(step)
                atomic_json(
                    output_dir / "run_manifest.json",
                    manifest("WANDB_RETRY"),
                )
                return REQUEUE_EXIT_CODE

        if args.smoke and post_update_recompute_audit is None:
            if first_update_adapter_delta is None or smoke_recompute_row is None:
                raise RuntimeError(
                    "post-update checkpoint smoke lacks its LoRA update or row"
                )
            replay_item = encoded(smoke_recompute_row, weighted=True)
            if int(replay_item["length"]) != int(train_length_contract["maximum"]):
                raise RuntimeError(
                    "post-update smoke did not replay the absolute longest row"
                )
            replay_torch_rng = torch.get_rng_state()
            replay_cuda_rng = torch.cuda.get_rng_state_all()
            replay_adapter_digest = trainable_parameter_digest(model)
            replay_frozen_probe = frozen_parameter_probe(model)
            optimizer.zero_grad(set_to_none=True)
            synchronize_cuda_devices()
            replay_started = time.perf_counter()
            try:
                replay_loss = batch_row_losses(
                    [replay_item],
                    backward_scale=1.0,
                )[0]
                synchronize_cuda_devices()
                replay_seconds = time.perf_counter() - replay_started
                replay_gradient = gradient_audit(parameters)
            finally:
                optimizer.zero_grad(set_to_none=True)
                torch.set_rng_state(replay_torch_rng)
                torch.cuda.set_rng_state_all(replay_cuda_rng)
            if trainable_parameter_digest(model) != replay_adapter_digest:
                raise RuntimeError(
                    "post-update checkpoint replay changed LoRA parameters"
                )
            if frozen_parameter_probe(model) != replay_frozen_probe:
                raise RuntimeError(
                    "post-update checkpoint replay changed frozen parameters"
                )
            post_update_recompute_audit = {
                "status": "PASS",
                "phase": "after_first_positive_lr_optimizer_update",
                "row_id": smoke_recompute_row["id"],
                "observed_tokens": int(replay_item["length"]),
                "hard_term_weighted_tokens": int(
                    replay_item["hard_term_weighted_tokens"]
                ),
                "loss": replay_loss,
                "seconds": replay_seconds,
                "gradient_audit": replay_gradient,
                "adapter_parameters_unchanged": True,
                "frozen_parameters_unchanged": True,
            }

        due_scope = validation_scope(step)
        due_validation = due_scope is not None
        if due_scope == "full":
            save_checkpoint(step)
            atomic_json(
                output_dir / "run_manifest.json",
                manifest("RUNNING"),
            )
            save_candidate_checkpoint(step)
            validate_candidate_checkpoints()
        due_by_time = time.monotonic() - last_recovery_monotonic >= save_every_seconds
        due_save = (
            step % save_every == 0
            or due_by_time
            or step == total_steps
            or stop_requested
            or due_validation
        )
        if due_save:
            save_checkpoint(step)
            atomic_json(output_dir / "run_manifest.json", manifest("RUNNING"))
        if due_validation:
            result = validate(step, full=due_scope == "full")
            if result is None:
                save_checkpoint(step)
                atomic_json(
                    output_dir / "run_manifest.json",
                    manifest("INTERRUPTED"),
                )
                finish_wandb_for_retry()
                return REQUEUE_EXIT_CODE
            history.append(result)
            if wandb_run is not None:
                save_checkpoint(step)
            try:
                log_validation(result)
            except Exception as error:
                print(
                    f"[wandb] validation log failed; requesting retry: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
                atomic_json(
                    output_dir / "run_manifest.json",
                    manifest("WANDB_RETRY"),
                )
                return REQUEUE_EXIT_CODE
            save_checkpoint(step)
        decision_after_validation = derive_early_stopping_decision(
            history,
            completed_steps,
        )
        if stop_requested:
            save_checkpoint(step)
            atomic_json(output_dir / "run_manifest.json", manifest("INTERRUPTED"))
            finish_wandb_for_retry()
            return REQUEUE_EXIT_CODE
        if not args.smoke and decision_after_validation["triggered"] is True:
            if step != early_terminal_step:
                raise RuntimeError(
                    "early stopping triggered outside the approved boundary"
                )
            save_checkpoint(step)
            atomic_json(
                output_dir / "run_manifest.json",
                manifest("EARLY_STOPPING_DECIDED"),
            )
            break

    terminal_early_stopping_decision = derive_early_stopping_decision(
        history,
        completed_steps,
    )
    production_early_stopped = (
        not args.smoke
        and terminal_early_stopping_decision["triggered"] is True
        and completed_steps == early_terminal_step
    )
    production_completed_full_horizon = (
        not args.smoke
        and terminal_early_stopping_decision["triggered"] is False
        and completed_steps == int(recipe["optimizer_steps"])
    )

    if not args.smoke:
        validate_candidate_checkpoints()
        if not (production_early_stopped or production_completed_full_horizon):
            raise RuntimeError("production ended without an approved terminal decision")
        expected_prefix_occurrences = consumed_occurrences_at_step(
            recipe,
            completed_steps,
        )
        if production_early_stopped and expected_prefix_occurrences != 223_568:
            raise RuntimeError("approved early-stop occurrence boundary changed")
        if consumed_occurrences != expected_prefix_occurrences:
            raise RuntimeError(
                "production consumed occurrences differ from its terminal step"
            )
        if sampler.cursor != expected_prefix_occurrences:
            raise RuntimeError(
                "production sampler cursor differs from its terminal prefix"
            )
        if backward_passes != expected_prefix_occurrences:
            raise RuntimeError("production occurrence audit is incomplete")

        prefix_sampler = GlobalAugmentedDeckSampler(
            train_rows,
            expert_label=recipe["expert_quality_label"],
            expert_boost=recipe["expert_oversampling_factor"],
            epochs=int(recipe["epochs"]),
            seed=recipe["seed"],
        )
        expected_task_counts: Counter[str] = Counter()
        expected_quality_counts: Counter[str] = Counter()
        expected_raw_indices: set[int] = set()
        for _ in range(expected_prefix_occurrences):
            row_index = prefix_sampler.draw_index()
            row = train_rows[row_index]
            expected_raw_indices.add(row_index)
            expected_task_counts[str(row["task"])] += 1
            expected_quality_counts[str(row["quality"])] += 1
        if prefix_sampler.cursor != expected_prefix_occurrences:
            raise RuntimeError("prefix replay did not consume its exact boundary")
        if occurrence_task_counts != expected_task_counts:
            raise RuntimeError(
                "production task occurrence counts differ from deterministic "
                f"prefix: {dict(occurrence_task_counts)} != "
                f"{dict(expected_task_counts)}"
            )
        if occurrence_quality_counts != expected_quality_counts:
            raise RuntimeError(
                "production quality occurrence counts differ from "
                f"deterministic prefix: {dict(occurrence_quality_counts)} != "
                f"{dict(expected_quality_counts)}"
            )
        if len(expected_raw_indices) != len(train_rows):
            raise RuntimeError(
                "terminal training prefix did not cover every raw train row"
            )
        expected_completed_candidate_steps = [
            step for step in candidate_steps if step <= completed_steps
        ]
        if (
            terminal_early_stopping_decision["completed_candidate_steps"]
            != expected_completed_candidate_steps
        ):
            raise RuntimeError(
                "completed full-validation history differs from the terminal "
                "candidate schedule"
            )
        if sorted(int(step) for step in candidate_checkpoints) != (
            expected_completed_candidate_steps
        ):
            raise RuntimeError(
                "candidate adapters differ from completed full validations"
            )
        if max_observed_tokens != train_length_contract["maximum"]:
            raise RuntimeError("production did not train on the longest row")

    def materialize_current_adapter(destination: Path) -> dict[str, str]:
        parameter_sha256 = trainable_parameter_digest(model)
        parameter_receipt = f"{parameter_sha256}\n"
        if destination.exists():
            if not (destination / "_SUCCESS").is_file():
                raise RuntimeError(
                    f"published final adapter is incomplete: {destination}"
                )
            receipt_path = destination / "TRAINABLE_PARAMETER_SHA256"
            if (
                not receipt_path.is_file()
                or receipt_path.read_text(encoding="utf-8") != parameter_receipt
            ):
                raise RuntimeError("published final adapter differs from current model")
            return adapter_files_sha256(destination)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        model.save_pretrained(temporary, safe_serialization=True)
        tokenizer.save_pretrained(temporary)
        (temporary / "LICENSE").write_bytes((PROJECT / "LICENSE").read_bytes())
        (temporary / "NOTICE").write_bytes((PROJECT / "NOTICE").read_bytes())
        (temporary / "TRAINABLE_PARAMETER_SHA256").write_text(
            parameter_receipt,
            encoding="utf-8",
        )
        publish_adapter(temporary, destination)
        return adapter_files_sha256(destination)

    def materialize_selected_adapter(
        source: Path,
        destination: Path,
    ) -> dict[str, str]:
        if not (source / "_SUCCESS").is_file():
            raise RuntimeError(f"selected source adapter is incomplete: {source}")
        source_files = adapter_files_sha256(source)
        if destination.exists():
            if not (destination / "_SUCCESS").is_file():
                raise RuntimeError(
                    f"published best adapter is incomplete: {destination}"
                )
            destination_files = adapter_files_sha256(destination)
            for name, digest in source_files.items():
                if name == "_SUCCESS":
                    continue
                if destination_files.get(name) != digest:
                    raise RuntimeError(
                        f"published best adapter differs from source: {name}"
                    )
            for name in ("LICENSE", "NOTICE"):
                if destination_files.get(name) != sha256(PROJECT / name):
                    raise RuntimeError(
                        f"published best adapter attribution changed: {name}"
                    )
            return destination_files
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        if temporary.exists():
            shutil.rmtree(temporary)
        shutil.copytree(source, temporary)
        success = temporary / "_SUCCESS"
        if success.exists():
            success.unlink()
        (temporary / "LICENSE").write_bytes((PROJECT / "LICENSE").read_bytes())
        (temporary / "NOTICE").write_bytes((PROJECT / "NOTICE").read_bytes())
        publish_adapter(temporary, destination)
        return adapter_files_sha256(destination)

    final_dir = output_dir / recipe["checkpoint"]["final_adapter_dirname"]
    final_files = materialize_current_adapter(final_dir)
    last_adapter_dir = final_dir
    final_checkpoint_record = {
        "step": completed_steps,
        "adapter_dir": str(final_dir),
        "files_sha256": final_files,
    }

    best_dir = output_dir / recipe["checkpoint"]["best_adapter_dirname"]
    if args.smoke:
        selected_source = final_dir
        best_files = materialize_selected_adapter(selected_source, best_dir)
        selection_record_value = {
            "candidate_steps": [completed_steps],
            "candidates": [
                {
                    "step": completed_steps,
                    "selection_loss": history[-1]["selection_loss"],
                    "translation_loss": history[-1]["translation_loss"],
                    "summarization_loss": history[-1]["summarization_loss"],
                    "adapter_dir": str(final_dir),
                    "files_sha256": final_files,
                }
            ],
            "selected_step": completed_steps,
            "selected_selection_loss": history[-1]["selection_loss"],
            "policy": "smoke_final_step",
            "tie_break": "not_applicable",
        }
        selected_checkpoint_value = {
            "policy": "smoke_final_step",
            "tie_break": "not_applicable",
            "step": completed_steps,
            "selection_loss": history[-1]["selection_loss"],
            "source_checkpoint_dir": str(final_dir),
            "adapter_dir": str(best_dir),
            "files_sha256": best_files,
        }
    else:
        weights = recipe["validation"]["selection_weights"]
        completed_candidate_steps = [
            int(step)
            for step in terminal_early_stopping_decision["completed_candidate_steps"]
        ]
        if not completed_candidate_steps:
            raise RuntimeError(
                "terminal production run has no completed full-validation " "candidate"
            )
        selected_validation = select_best_full_validation(
            history,
            candidate_steps=completed_candidate_steps,
            translation_weight=float(weights["translation"]),
            summarization_weight=float(weights["summarization"]),
        )
        candidate_records: list[dict[str, Any]] = []
        full_history = {
            int(item["step"]): item for item in history if item["scope"] == "full"
        }
        for candidate_step in completed_candidate_steps:
            checkpoint = candidate_checkpoints.get(str(candidate_step))
            if checkpoint is None:
                raise RuntimeError(
                    f"missing candidate adapter at step {candidate_step}"
                )
            validation = full_history[candidate_step]
            candidate_records.append(
                {
                    "step": candidate_step,
                    "selection_loss": validation["selection_loss"],
                    "translation_loss": validation["translation_loss"],
                    "summarization_loss": validation["summarization_loss"],
                    "adapter_dir": checkpoint["adapter_dir"],
                    "files_sha256": checkpoint["files_sha256"],
                }
            )
        selected_step = int(selected_validation["step"])
        selected_candidate = candidate_checkpoints[str(selected_step)]
        selected_source = Path(selected_candidate["adapter_dir"])
        best_files = materialize_selected_adapter(
            selected_source,
            best_dir,
        )
        selection_record_value = {
            "candidate_steps": completed_candidate_steps,
            "candidates": candidate_records,
            "selected_step": selected_step,
            "selected_selection_loss": selected_validation["selection_loss"],
            "policy": recipe["validation"]["selection_policy"],
            "tie_break": recipe["validation"]["selection_tie_break"],
        }
        selected_checkpoint_value = {
            "policy": recipe["validation"]["selection_policy"],
            "tie_break": recipe["validation"]["selection_tie_break"],
            "step": selected_step,
            "selection_loss": selected_validation["selection_loss"],
            "source_checkpoint_dir": str(selected_source),
            "adapter_dir": str(best_dir),
            "files_sha256": best_files,
        }

    status = (
        "SMOKE_COMPLETED"
        if args.smoke
        else ("EARLY_STOPPED" if production_early_stopped else "COMPLETED")
    )
    smoke_manifest_value: dict[str, Any] | None = None
    if args.smoke:
        if (
            first_update_gradient_audit is None
            or first_update_adapter_delta is None
            or frozen_probe_after_update != frozen_probe_before
            or total_hard_term_weighted_tokens <= 0
            or first_output_only_supervision is None
            or microbatch_phase_seconds is None
            or validation_row_seconds is None
            or validation_benchmark is None
            or optimizer_boundary_seconds is None
            or one_time_gradient_audit_seconds is None
            or projected_runtime_budget is None
            or post_update_recompute_audit is None
            or len(training_bucket_benchmarks) != len(microbatch_buckets)
            or len(validation_bucket_benchmarks) != len(microbatch_buckets)
            or production_update_benchmark is None
            or (b200 and exact_tail_smoke_audit is None)
        ):
            raise RuntimeError("smoke training evidence is incomplete")
        smoke_update_plan = planned_microbatch_counts(
            smoke_update_indices,
            lengths=train_lengths,
            rows_per_update=rows_per_update,
            buckets=microbatch_buckets,
        )
        if smoke_recompute_row is None:
            raise RuntimeError("smoke longest-row identity is missing")
        longest_row_index = next(
            (
                index
                for index, row in enumerate(train_rows)
                if str(row["id"]) == str(smoke_recompute_row["id"])
                and str(row["task"]) == str(smoke_recompute_row["task"])
            ),
            None,
        )
        if longest_row_index is None:
            raise RuntimeError("smoke longest row is absent from training data")
        longest_occurrence_count = smoke_update_indices.count(longest_row_index)
        if (
            longest_occurrence_count != 1
            or train_lengths[longest_row_index] != train_length_contract["maximum"]
            or max_observed_tokens != train_length_contract["maximum"]
        ):
            raise RuntimeError(
                "smoke update did not consume the absolute longest row once"
            )
        longest_occurrence_update = {
            "status": "PASS",
            "optimizer_step": 1,
            "row_index": longest_row_index,
            "row_id": str(smoke_recompute_row["id"]),
            "task": str(smoke_recompute_row["task"]),
            "expected_tokens": (
                int(recipe["actual_longest_row_tokens"])
                if b200
                else train_length_contract["maximum"]
            ),
            "observed_tokens": train_lengths[longest_row_index],
            "occurrences_in_optimizer_group": rows_per_update,
            "longest_row_occurrence_count": longest_occurrence_count,
            "forward_backward_completed": True,
            "loss_denominator": rows_per_update,
            "finite_nonzero_gradient": first_update_gradient_audit,
            "gradient_clip_invoked": True,
            "adamw_update": first_update_adapter_delta,
            "frozen_base_probe_unchanged": True,
            "post_update_longest_row_recompute": (post_update_recompute_audit),
        }
        smoke_manifest_value = {
            "max_length_contract": max_length,
            "absolute_longest_required_tokens": train_length_contract["maximum"],
            "observed_tokens": max_observed_tokens,
            "backward_passes": backward_passes,
            "consumed_occurrences": consumed_occurrences,
            "update_microbatch_plan": smoke_update_plan,
            "production_training_plan": production_training_plan,
            "subset_validation_plan": subset_validation_plan,
            "full_validation_plan": full_validation_plan,
            "hard_term_weighted_tokens": total_hard_term_weighted_tokens,
            "gradient_audit": first_update_gradient_audit,
            "adapter_update": first_update_adapter_delta,
            "longest_occurrence_update": longest_occurrence_update,
            "exact_tail_transaction": exact_tail_smoke_audit,
            "frozen_base_probe_unchanged": True,
            "adapter_files_sha256": final_files,
            "final_adapter_atomic_disk_save": {
                "status": "PASS",
                "adapter_dir": str(final_dir),
                "success_marker": str(final_dir / "_SUCCESS"),
                "files_sha256": final_files,
                "reload_is_verified_by": "downstream_tuned_inference",
                "distinct_from_tail_parameter_payload": True,
            },
            "output_only_supervision": {
                **first_output_only_supervision,
                "translation_or_summarization_body_only": True,
            },
            "gpu_count": torch.cuda.device_count(),
            "peak_allocated_bytes": {
                str(index): torch.cuda.max_memory_allocated(index)
                for index in range(torch.cuda.device_count())
            },
            "peak_reserved_bytes": {
                str(index): torch.cuda.max_memory_reserved(index)
                for index in range(torch.cuda.device_count())
            },
            "microbatch_phase_seconds": microbatch_phase_seconds,
            "validation_row_seconds": validation_row_seconds,
            "validation_benchmark": validation_benchmark,
            "production_update_benchmark": (production_update_benchmark),
            "optimizer_boundary_seconds": optimizer_boundary_seconds,
            "one_time_gradient_audit_seconds": (one_time_gradient_audit_seconds),
            "runtime_budget_projection": projected_runtime_budget,
            "post_update_recompute_audit": post_update_recompute_audit,
            "early_stopping_decision": terminal_early_stopping_decision,
        }

    if wandb_run is not None:
        atomic_json(
            output_dir / "run_manifest.json",
            manifest("FINALIZING_WANDB"),
        )
        try:
            wandb_run.summary["training/status"] = status
            wandb_run.summary["training/completed_updates"] = completed_steps
            wandb_run.summary["training/final_step"] = completed_steps
            wandb_run.summary["training/selected_step"] = selected_checkpoint_value[
                "step"
            ]
            wandb_run.summary["training/selected_validation_loss"] = (
                selected_checkpoint_value["selection_loss"]
            )
            wandb_run.summary["training/consumed_occurrences"] = consumed_occurrences
            wandb_run.summary["training/completed_epochs"] = (
                terminal_early_stopping_decision["completed_epochs"]
            )
            wandb_run.summary["training/early_stopping_triggered"] = (
                terminal_early_stopping_decision["triggered"]
            )
            wandb_run.summary["training/early_stopping_reason"] = (
                terminal_early_stopping_decision["reason"]
            )
            wandb_run.summary["training/activation_checkpointing"] = (
                training_performance["activation_checkpointing"]
            )
            wandb_run.summary["training/checkpointed_decoder_layer_count"] = (
                training_performance["checkpointed_decoder_layer_count"]
            )
            wandb_run.summary["training/uncheckpointed_decoder_layer_count"] = (
                training_performance["uncheckpointed_decoder_layer_count"]
            )
            wandb_run.summary["training/training_performance_sha256"] = (
                canonical_sha256(training_performance)
            )
            wandb_run.summary["training/gradient_checkpointing_contract_sha256"] = (
                canonical_sha256(gradient_checkpointing_contract)
            )
            for device_index in range(torch.cuda.device_count()):
                wandb_run.summary[
                    f"training/peak_allocated_bytes_gpu{device_index}"
                ] = torch.cuda.max_memory_allocated(device_index)
                wandb_run.summary[f"training/peak_reserved_bytes_gpu{device_index}"] = (
                    torch.cuda.max_memory_reserved(device_index)
                )
            wandb_run.finish(exit_code=0)
        except Exception as error:
            print(
                f"[wandb] final commit failed; requesting retry: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            atomic_json(
                output_dir / "run_manifest.json",
                manifest("WANDB_RETRY"),
            )
            return REQUEUE_EXIT_CODE
        wandb_completion_status = "COMMITTED"

    final_manifest = manifest(status)
    if smoke_manifest_value is not None:
        final_manifest["smoke"] = smoke_manifest_value
    atomic_json(output_dir / "run_manifest.json", final_manifest)
    atomic_text(output_dir / "_SUCCESS", "PASS\n")
    print(json.dumps(final_manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
