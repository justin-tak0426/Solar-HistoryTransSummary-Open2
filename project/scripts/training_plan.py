#!/usr/bin/env python3
"""Pure planning helpers for the Solar training contracts."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypeVar


T = TypeVar("T")


def _strict_positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer: {value!r}")
    return value


def _optimizer_steps_for_occurrences(
    *,
    occurrences_per_step: int,
    total_occurrences: int,
) -> int:
    per_step = _strict_positive_int(
        occurrences_per_step,
        label="occurrences_per_step",
    )
    total = _strict_positive_int(
        total_occurrences,
        label="total_occurrences",
    )
    return math.ceil(total / per_step)


def occurrences_through_step(
    step: int,
    *,
    occurrences_per_step: int,
    total_occurrences: int,
) -> int:
    """Return exact cumulative coverage through a completed optimizer step.

    ``step`` is zero for a pre-training/resume boundary and otherwise cannot
    exceed the exact optimizer horizon. The final step may therefore contain
    fewer occurrences than the nominal optimizer grouping, without padding.
    """

    optimizer_steps = _optimizer_steps_for_occurrences(
        occurrences_per_step=occurrences_per_step,
        total_occurrences=total_occurrences,
    )
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or not 0 <= step <= optimizer_steps
    ):
        raise ValueError(
            f"step must be an integer in [0, {optimizer_steps}]: {step!r}"
        )
    return min(step * occurrences_per_step, total_occurrences)


def occurrences_in_step(
    step: int,
    *,
    occurrences_per_step: int,
    total_occurrences: int,
) -> int:
    """Return the exact denominator/draw count for one optimizer step."""

    optimizer_steps = _optimizer_steps_for_occurrences(
        occurrences_per_step=occurrences_per_step,
        total_occurrences=total_occurrences,
    )
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or not 1 <= step <= optimizer_steps
    ):
        raise ValueError(
            f"step must be an integer in [1, {optimizer_steps}]: {step!r}"
        )
    return occurrences_through_step(
        step,
        occurrences_per_step=occurrences_per_step,
        total_occurrences=total_occurrences,
    ) - occurrences_through_step(
        step - 1,
        occurrences_per_step=occurrences_per_step,
        total_occurrences=total_occurrences,
    )


EARLY_STOPPING_POLICY_FIELDS = {
    "enabled",
    "monitor",
    "mode",
    "minimum_epochs",
    "minimum_step",
    "subset_baseline_policy",
    "subset_patience_evaluations",
    "subset_min_delta",
    "full_validation_confirmation_required",
    "confirmation_step",
    "full_min_delta",
    "continue_if_either_task_materially_improves",
    "earliest_stop_step",
    "latest_stop_step",
    "restore_best_full_validation",
    "sealed_evaluation_not_monitor",
}
EARLY_STOPPING_APPROVED_VALUES = {
    "enabled": True,
    "monitor": "selection_loss",
    "mode": "min",
    "minimum_epochs": 1.0,
    "minimum_step": 9316,
    "subset_baseline_policy": "last_fixed_subset_before_minimum_step",
    "subset_patience_evaluations": 3,
    "full_validation_confirmation_required": True,
    "confirmation_step": 13973,
    "continue_if_either_task_materially_improves": True,
    "earliest_stop_step": 13973,
    "latest_stop_step": 18631,
    "restore_best_full_validation": True,
    "sealed_evaluation_not_monitor": True,
}
EARLY_STOPPING_APPROVED_DELTAS = {
    "subset_min_delta": {
        "absolute": 0.001,
        "relative": 0.0025,
        "combination": "max",
    },
    "full_min_delta": {
        "absolute": 0.0005,
        "relative": 0.001,
        "combination": "max",
    },
}

B200_EARLY_STOPPING_POLICY_FIELDS = {
    "enabled",
    "monitor",
    "mode",
    "minimum_coverage_epochs",
    "earliest_terminal_step",
    "latest_terminal_step",
    "subset_patience_used_for_stopping",
    "subset_diagnostic_min_delta",
    "full_diagnostic_min_delta",
    "continue_if_either_task_materially_improves",
    "restore_best_completed_full_validation",
    "sealed_evaluation_not_monitor",
}
B200_EARLY_STOPPING_APPROVED_VALUES = {
    "enabled": False,
    "monitor": "selection_loss",
    "mode": "min",
    "minimum_coverage_epochs": 1.0,
    "earliest_terminal_step": 9316,
    "latest_terminal_step": 9316,
    "subset_patience_used_for_stopping": False,
    "continue_if_either_task_materially_improves": True,
    "restore_best_completed_full_validation": True,
    "sealed_evaluation_not_monitor": True,
}
B200_EARLY_STOPPING_APPROVED_DELTAS = {
    "subset_diagnostic_min_delta": {
        "absolute": 0.001,
        "relative": 0.0025,
        "combination": "max",
    },
    "full_diagnostic_min_delta": {
        "absolute": 0.0005,
        "relative": 0.001,
        "combination": "max",
    },
}


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_early_stopping_policy(
    policy: dict[str, Any],
    *,
    optimizer_steps: int,
    full_validation_steps: Sequence[int],
) -> dict[str, Any]:
    """Fail closed unless the approved early-stopping policy is unchanged."""

    if not isinstance(policy, dict):
        raise ValueError("early-stopping policy must be an object")
    if set(policy) != EARLY_STOPPING_POLICY_FIELDS:
        raise ValueError(
            "early-stopping policy fields changed: "
            f"{sorted(policy)} != {sorted(EARLY_STOPPING_POLICY_FIELDS)}"
        )
    for key, expected in EARLY_STOPPING_APPROVED_VALUES.items():
        observed = policy.get(key)
        if type(observed) is not type(expected) or observed != expected:
            raise ValueError(
                f"early-stopping {key} changed: "
                f"{observed!r} != {expected!r}"
            )
    for key, expected in EARLY_STOPPING_APPROVED_DELTAS.items():
        value = policy.get(key)
        if (
            not isinstance(value, dict)
            or set(value) != set(expected)
            or any(
                type(value[field]) is not type(expected_value)
                or value[field] != expected_value
                for field, expected_value in expected.items()
            )
        ):
            raise ValueError(
                f"early-stopping {key} changed: {value!r} != {expected!r}"
            )
    if optimizer_steps != int(policy["latest_stop_step"]):
        raise ValueError("early-stopping latest step differs from optimizer horizon")
    candidates = list(full_validation_steps)
    if candidates != sorted(set(candidates)):
        raise ValueError("full-validation steps must be sorted and unique")
    required_candidates = {
        int(policy["minimum_step"]),
        int(policy["confirmation_step"]),
        int(policy["latest_stop_step"]),
    }
    if not required_candidates.issubset(candidates):
        raise ValueError(
            "early-stopping boundaries are not full-validation candidates: "
            f"{sorted(required_candidates - set(candidates))}"
        )
    if int(policy["earliest_stop_step"]) != int(policy["confirmation_step"]):
        raise ValueError("earliest stop must equal the confirmation step")
    return dict(policy)


def validate_b200_early_stopping_policy(
    policy: dict[str, Any],
    *,
    optimizer_steps: int,
    full_validation_steps: Sequence[int],
) -> dict[str, Any]:
    """Fail closed unless B200 quality early stopping stays disabled."""

    if not isinstance(policy, dict):
        raise ValueError("B200 early-stopping policy must be an object")
    if set(policy) != B200_EARLY_STOPPING_POLICY_FIELDS:
        raise ValueError(
            "B200 early-stopping policy fields changed: "
            f"{sorted(policy)} != {sorted(B200_EARLY_STOPPING_POLICY_FIELDS)}"
        )
    for key, expected in B200_EARLY_STOPPING_APPROVED_VALUES.items():
        observed = policy.get(key)
        if type(observed) is not type(expected) or observed != expected:
            raise ValueError(
                f"B200 early-stopping {key} changed: "
                f"{observed!r} != {expected!r}"
            )
    for key, expected in B200_EARLY_STOPPING_APPROVED_DELTAS.items():
        value = policy.get(key)
        if (
            not isinstance(value, dict)
            or set(value) != set(expected)
            or any(
                type(value[field]) is not type(expected_value)
                or value[field] != expected_value
                for field, expected_value in expected.items()
            )
        ):
            raise ValueError(
                f"B200 early-stopping {key} changed: "
                f"{value!r} != {expected!r}"
            )
    if optimizer_steps != 9316:
        raise ValueError("B200 optimizer horizon must be exactly 9316 steps")
    if list(full_validation_steps) != [4658, 9316]:
        raise ValueError("B200 full-validation steps must be [4658, 9316]")
    if (
        int(policy["earliest_terminal_step"]) != optimizer_steps
        or int(policy["latest_terminal_step"]) != optimizer_steps
    ):
        raise ValueError("B200 terminal boundary must equal the exact epoch horizon")
    return dict(policy)


def _validated_loss_record(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("validation history records must be objects")
    step = record.get("step")
    scope = record.get("scope")
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise ValueError(f"validation history has invalid step: {step!r}")
    if scope not in {"fixed_subset", "full"}:
        raise ValueError(f"validation history has invalid scope: {scope!r}")
    result = {"step": step, "scope": scope}
    for key in (
        "selection_loss",
        "translation_loss",
        "summarization_loss",
    ):
        value = record.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"validation history has invalid {key}: {value!r}")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"validation history has invalid {key}: {value!r}")
        result[key] = value
    return result


def early_stopping_decision(
    history: Sequence[dict[str, Any]],
    *,
    completed_step: int,
    policy: dict[str, Any],
    optimizer_steps: int,
    full_validation_steps: Sequence[int],
    occurrences_per_step: int,
    occurrences_per_epoch: int,
    total_occurrences: int | None = None,
) -> dict[str, Any]:
    """Recompute the terminal decision from durable validation history only."""

    validated_policy = validate_early_stopping_policy(
        policy,
        optimizer_steps=optimizer_steps,
        full_validation_steps=full_validation_steps,
    )
    if (
        isinstance(completed_step, bool)
        or not isinstance(completed_step, int)
        or not 0 <= completed_step <= optimizer_steps
    ):
        raise ValueError(f"invalid completed step: {completed_step!r}")
    per_step = _strict_positive_int(
        occurrences_per_step,
        label="occurrences_per_step",
    )
    per_epoch = _strict_positive_int(
        occurrences_per_epoch,
        label="occurrences_per_epoch",
    )
    if total_occurrences is None:
        total_occurrences = optimizer_steps * per_step
    total = _strict_positive_int(
        total_occurrences,
        label="total_occurrences",
    )
    expected_optimizer_steps = _optimizer_steps_for_occurrences(
        occurrences_per_step=per_step,
        total_occurrences=total,
    )
    if optimizer_steps != expected_optimizer_steps:
        raise ValueError(
            "optimizer horizon does not exactly cover total occurrences: "
            f"{optimizer_steps} != {expected_optimizer_steps}"
        )

    records = [_validated_loss_record(record) for record in history]
    identities = [(record["step"], record["scope"]) for record in records]
    if identities != sorted(identities):
        raise ValueError("validation history must be ordered by step and scope")
    if len(set(identities)) != len(identities):
        raise ValueError("validation history contains duplicate step/scope records")
    if any(int(record["step"]) > completed_step for record in records):
        raise ValueError("validation history extends beyond the completed step")

    minimum_step = int(validated_policy["minimum_step"])
    confirmation_step = int(validated_policy["confirmation_step"])
    latest_step = int(validated_policy["latest_stop_step"])
    baseline_candidates = [
        record
        for record in records
        if record["scope"] == "fixed_subset"
        and int(record["step"]) < minimum_step
    ]
    baseline = baseline_candidates[-1] if baseline_candidates else None
    subset_records = [
        record
        for record in records
        if record["scope"] == "fixed_subset"
        and minimum_step < int(record["step"]) < confirmation_step
    ]
    subset_best_step = int(baseline["step"]) if baseline is not None else None
    subset_best_loss = (
        float(baseline["selection_loss"]) if baseline is not None else None
    )
    consecutive = 0
    subset_evaluations: list[dict[str, Any]] = []
    subset_delta = validated_policy["subset_min_delta"]
    for record in subset_records:
        if subset_best_loss is None or subset_best_step is None:
            raise ValueError(
                "post-minimum subset history exists without its pre-minimum baseline"
            )
        threshold = max(
            float(subset_delta["absolute"]),
            float(subset_delta["relative"]) * subset_best_loss,
        )
        improved = subset_best_loss - float(record["selection_loss"]) >= threshold
        before_step = subset_best_step
        before_loss = subset_best_loss
        if improved:
            subset_best_step = int(record["step"])
            subset_best_loss = float(record["selection_loss"])
            consecutive = 0
        else:
            consecutive += 1
        subset_evaluations.append(
            {
                "step": int(record["step"]),
                "loss": float(record["selection_loss"]),
                "best_step_before": before_step,
                "best_loss_before": before_loss,
                "threshold": threshold,
                "material_improvement": improved,
                "consecutive_non_improvements": consecutive,
            }
        )
    patience = int(validated_policy["subset_patience_evaluations"])
    patience_reached = consecutive >= patience

    full_records = [record for record in records if record["scope"] == "full"]
    completed_candidate_steps = [int(record["step"]) for record in full_records]
    if any(step not in set(full_validation_steps) for step in completed_candidate_steps):
        raise ValueError("history contains an unconfigured full-validation candidate")
    current_full = next(
        (
            record
            for record in full_records
            if int(record["step"]) == confirmation_step
        ),
        None,
    )
    prior_full_records = [
        record for record in full_records if int(record["step"]) < confirmation_step
    ]
    prior_full = (
        min(
            prior_full_records,
            key=lambda record: (
                float(record["selection_loss"]),
                int(record["step"]),
            ),
        )
        if prior_full_records
        else None
    )
    loss_fields = (
        "selection_loss",
        "translation_loss",
        "summarization_loss",
    )
    prior_full_loss = (
        {key: float(prior_full[key]) for key in loss_fields}
        if prior_full is not None
        else None
    )
    confirmation_loss = (
        {key: float(current_full[key]) for key in loss_fields}
        if current_full is not None
        else None
    )
    confirmation_thresholds: dict[str, float] | None = None
    material_improvements: dict[str, bool] | None = None
    if current_full is not None:
        if prior_full is None:
            raise ValueError(
                "confirmation full validation lacks a prior full candidate"
            )
        full_delta = validated_policy["full_min_delta"]
        confirmation_thresholds = {
            key: max(
                float(full_delta["absolute"]),
                float(full_delta["relative"]) * float(prior_full[key]),
            )
            for key in loss_fields
        }
        material_improvements = {
            key: (
                float(prior_full[key]) - float(current_full[key])
                >= confirmation_thresholds[key]
            )
            for key in loss_fields
        }

    triggered = False
    if completed_step < minimum_step:
        status = "PENDING_MINIMUM"
        reason = "minimum_step_not_reached"
    elif baseline is None:
        raise ValueError(
            "minimum step was reached without a fixed-subset baseline"
        )
    elif completed_step < confirmation_step:
        status = (
            "PENDING_CONFIRMATION"
            if patience_reached
            else "PENDING_SUBSET_PATIENCE"
        )
        reason = (
            "awaiting_full_validation_confirmation"
            if patience_reached
            else "subset_patience_not_reached"
        )
    elif current_full is None:
        if completed_step > confirmation_step:
            raise ValueError(
                "training advanced beyond confirmation without full validation"
            )
        status = "PENDING_CONFIRMATION"
        reason = "awaiting_full_validation_confirmation"
    else:
        any_material = bool(
            material_improvements
            and any(material_improvements.values())
        )
        triggered = patience_reached and not any_material
        if triggered:
            status = "TRIGGERED"
            reason = "no_material_improvement_confirmed"
        elif completed_step >= latest_step:
            status = "MAXIMUM_REACHED"
            reason = "maximum_step_reached"
        else:
            status = "CONTINUE"
            reason = (
                "material_improvement_confirmed"
                if any_material
                else "subset_patience_not_reached"
            )
    if triggered and completed_step > confirmation_step:
        raise ValueError("training advanced beyond a triggered early stop")

    return {
        "schema_version": 1,
        "policy_sha256": _canonical_sha256(validated_policy),
        "status": status,
        "evaluated_step": completed_step,
        "triggered": triggered,
        "reason": reason,
        "subset_baseline_step": (
            int(baseline["step"]) if baseline is not None else None
        ),
        "subset_baseline_loss": (
            float(baseline["selection_loss"]) if baseline is not None else None
        ),
        "considered_subset_steps": [
            int(record["step"]) for record in subset_records
        ],
        "subset_evaluations": subset_evaluations,
        "subset_best_step": subset_best_step,
        "subset_best_loss": subset_best_loss,
        "consecutive_non_improvements": consecutive,
        "patience_reached": patience_reached,
        "prior_full_step": (
            int(prior_full["step"]) if prior_full is not None else None
        ),
        "prior_full_loss": prior_full_loss,
        "confirmation_step": confirmation_step,
        "confirmation_loss": confirmation_loss,
        "confirmation_thresholds": confirmation_thresholds,
        "material_improvements": material_improvements,
        "completed_candidate_steps": completed_candidate_steps,
        "completed_occurrences": occurrences_through_step(
            completed_step,
            occurrences_per_step=per_step,
            total_occurrences=total,
        ),
        "completed_epochs": (
            occurrences_through_step(
                completed_step,
                occurrences_per_step=per_step,
                total_occurrences=total,
            )
            / per_epoch
        ),
    }


def positive_cosine_learning_rate(
    *,
    update: int,
    total_updates: int,
    warmup_updates: int,
    peak_learning_rate: float,
    minimum_learning_rate_ratio: float,
) -> float:
    """Return the LR used by a one-indexed optimizer update.

    Every real optimizer update receives a positive learning rate. The
    conceptual value after ``total_updates`` is zero, but that value is never
    used for a data-bearing update.
    """

    if total_updates < 1:
        raise ValueError("total_updates must be positive")
    if not 1 <= update <= total_updates:
        raise ValueError(
            f"update must be in [1, {total_updates}], got {update}"
        )
    if not 0 <= warmup_updates < total_updates:
        raise ValueError("warmup_updates must be in [0, total_updates)")
    if not math.isfinite(peak_learning_rate) or peak_learning_rate <= 0:
        raise ValueError("peak_learning_rate must be finite and positive")
    if (
        not math.isfinite(minimum_learning_rate_ratio)
        or not 0.0 < minimum_learning_rate_ratio < 1.0
    ):
        raise ValueError(
            "minimum_learning_rate_ratio must be finite and in (0, 1)"
        )

    if warmup_updates and update <= warmup_updates:
        return peak_learning_rate * update / warmup_updates

    decay_updates = total_updates - warmup_updates
    decay_index = update - warmup_updates - 1
    progress = decay_index / decay_updates
    multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
    decay_multiplier = minimum_learning_rate_ratio + (
        1.0 - minimum_learning_rate_ratio
    ) * multiplier
    result = peak_learning_rate * decay_multiplier
    if not math.isfinite(result) or result <= 0:
        raise RuntimeError(f"data-bearing update received invalid LR: {result}")
    return result


def validate_microbatch_buckets(
    buckets: Sequence[dict[str, Any]],
    *,
    max_length: int,
) -> tuple[tuple[int, int], ...]:
    if max_length < 1:
        raise ValueError("max_length must be positive")
    normalized: list[tuple[int, int]] = []
    previous = 0
    for index, raw in enumerate(buckets):
        if not isinstance(raw, dict):
            raise ValueError(f"microbatch bucket {index} must be an object")
        ceiling = raw.get("max_length")
        batch_size = raw.get("batch_size")
        if (
            isinstance(ceiling, bool)
            or not isinstance(ceiling, int)
            or ceiling <= previous
        ):
            raise ValueError(
                f"microbatch bucket {index} has invalid max_length: {ceiling}"
            )
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError(
                f"microbatch bucket {index} has invalid batch_size: {batch_size}"
            )
        normalized.append((ceiling, batch_size))
        previous = ceiling
    if not normalized or normalized[-1][0] != max_length:
        raise ValueError(
            "final microbatch bucket must end exactly at training max_length"
        )
    return tuple(normalized)


def bucket_index(
    length: int,
    buckets: Sequence[tuple[int, int]],
) -> int:
    if length < 1:
        raise ValueError(f"sequence length must be positive: {length}")
    for index, (ceiling, _batch_size) in enumerate(buckets):
        if length <= ceiling:
            return index
    raise ValueError(
        f"sequence length {length} exceeds final bucket {buckets[-1][0]}"
    )


def group_update_rows(
    rows: Sequence[T],
    *,
    length_of: Callable[[T], int],
    buckets: Sequence[tuple[int, int]],
) -> list[list[T]]:
    """Make deterministic, length-bucketed microbatches for one update."""

    grouped: dict[int, list[tuple[int, T]]] = defaultdict(list)
    for ordinal, row in enumerate(rows):
        grouped[bucket_index(length_of(row), buckets)].append((ordinal, row))

    result: list[list[T]] = []
    for index in sorted(grouped):
        capacity = buckets[index][1]
        # Longest first makes the memory benchmark and padding behavior stable.
        ordered = sorted(
            grouped[index],
            key=lambda item: (-length_of(item[1]), item[0]),
        )
        for start in range(0, len(ordered), capacity):
            result.append([row for _ordinal, row in ordered[start : start + capacity]])
    if sum(len(group) for group in result) != len(rows):
        raise AssertionError("microbatch grouping lost rows")
    return result


def planned_microbatch_counts(
    row_indices: Iterable[int],
    *,
    lengths: Sequence[int],
    rows_per_update: int,
    buckets: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    if rows_per_update < 1:
        raise ValueError("rows_per_update must be positive")
    pending: list[int] = []
    bucket_microbatches: Counter[int] = Counter()
    bucket_rows: Counter[int] = Counter()
    updates = 0
    occurrences = 0

    def consume(update_rows: list[int]) -> None:
        nonlocal updates, occurrences
        if not update_rows:
            return
        updates += 1
        occurrences += len(update_rows)
        for group in group_update_rows(
            update_rows,
            length_of=lambda index: lengths[index],
            buckets=buckets,
        ):
            index = bucket_index(lengths[group[0]], buckets)
            bucket_microbatches[index] += 1
            bucket_rows[index] += len(group)

    for row_index in row_indices:
        if not 0 <= row_index < len(lengths):
            raise ValueError(f"row index is out of range: {row_index}")
        pending.append(row_index)
        if len(pending) == rows_per_update:
            consume(pending)
            pending = []
    consume(pending)
    return {
        "optimizer_updates": updates,
        "occurrences": occurrences,
        "microbatches": sum(bucket_microbatches.values()),
        "bucket_microbatches": {
            str(buckets[index][0]): bucket_microbatches[index]
            for index in range(len(buckets))
        },
        "bucket_rows": {
            str(buckets[index][0]): bucket_rows[index]
            for index in range(len(buckets))
        },
    }


def nearest_rank(values: Sequence[int], quantile: float) -> int:
    if not values:
        raise ValueError("cannot calculate a quantile of an empty sequence")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


def token_length_contract(
    rows: Sequence[dict[str, Any]],
    lengths: Sequence[int],
    *,
    max_length: int,
) -> dict[str, Any]:
    if len(rows) != len(lengths) or not rows:
        raise ValueError("rows and lengths must be non-empty and aligned")
    over = [
        {
            "row_index": index,
            "id": str(row["id"]),
            "task": str(row["task"]),
            "quality": str(row.get("quality") or ""),
            "tokens": int(length),
        }
        for index, (row, length) in enumerate(zip(rows, lengths, strict=True))
        if length > max_length
    ]
    by_task = Counter(str(row["task"]) for row in rows)
    return {
        "rows": len(rows),
        "by_task": dict(by_task),
        "max_length": max_length,
        "over_max_count": len(over),
        "over_max_rows": over,
        "minimum": min(lengths),
        "p50": nearest_rank(lengths, 0.5),
        "p90": nearest_rank(lengths, 0.9),
        "p95": nearest_rank(lengths, 0.95),
        "p99": nearest_rank(lengths, 0.99),
        "p99_9": nearest_rank(lengths, 0.999),
        "maximum": max(lengths),
    }


def select_best_full_validation(
    history: Sequence[dict[str, Any]],
    *,
    candidate_steps: Sequence[int],
    translation_weight: float,
    summarization_weight: float,
) -> dict[str, Any]:
    if (
        not math.isfinite(translation_weight)
        or not math.isfinite(summarization_weight)
        or translation_weight <= 0
        or summarization_weight <= 0
        or not math.isclose(
            translation_weight + summarization_weight,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("validation selection weights must be positive and sum to 1")
    wanted = list(candidate_steps)
    if not wanted or wanted != sorted(set(wanted)):
        raise ValueError("candidate_steps must be sorted and unique")
    by_step: dict[int, dict[str, Any]] = {}
    for record in history:
        if record.get("scope") != "full":
            continue
        step = int(record["step"])
        if step not in wanted:
            continue
        if step in by_step:
            raise ValueError(f"duplicate full validation record at step {step}")
        translation = float(record["translation_loss"])
        summarization = float(record["summarization_loss"])
        score = (
            translation_weight * translation
            + summarization_weight * summarization
        )
        recorded_score = float(record["selection_loss"])
        for label, value in (
            ("translation_loss", translation),
            ("summarization_loss", summarization),
            ("selection_loss", recorded_score),
            ("recomputed_selection_loss", score),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{label} is invalid at step {step}: {value}")
        if not math.isclose(
            score,
            recorded_score,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"selection loss mismatch at step {step}: "
                f"{recorded_score} != {score}"
            )
        by_step[step] = dict(record)
    if sorted(by_step) != wanted:
        raise ValueError(
            f"full validation candidates are incomplete: "
            f"{sorted(by_step)} != {wanted}"
        )
    return min(
        (by_step[step] for step in wanted),
        key=lambda record: (float(record["selection_loss"]), int(record["step"])),
    )
