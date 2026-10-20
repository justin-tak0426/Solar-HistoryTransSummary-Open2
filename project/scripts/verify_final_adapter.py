#!/usr/bin/env python3
"""Fail closed unless evaluation uses the exact sealed production adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from common import (
    CONFIG_DIR,
    GlobalAugmentedDeckSampler,
    PROJECT,
    atomic_json,
    canonical_sha256,
    expected_lora_module_names,
    pipeline_code_identity,
    read_jsonl,
    require_owned_path,
    sha256,
)
from b200_runtime import (
    B200_CONFIG_DIR,
    BF16_SOURCE_DIR,
    EXPECTED_HF_DEVICE_MAP,
    TRAIN_EXPERT_DIR,
    b200_config_identity,
    b200_full_gradient_checkpointing_contract,
    b200_runtime_override_evidence,
    b200_selective_gradient_checkpointing_contract,
    b200_training_performance_contract,
    load_b200_config_bundle,
    require_expected_user,
    require_owned_path as require_b200_owned_path,
)
from infer import (
    lightweight_source_identity,
    validate_adapter,
    validate_terminal_training_success,
)
from quantization.contract import (
    BASE_MODEL_ID,
    BASE_REVISION,
    EXPECTED_SOURCE_INDEX_SHA256,
    EXPECTED_SOURCE_SHARD_COUNT,
    EXPECTED_SOURCE_TENSOR_COUNT,
    EXPECTED_SOURCE_TOTAL_SIZE,
    EXPERT_INDEX_FILENAME,
    LAYER_FLAT_SCHEMA_NAME,
    MANIFEST_FILENAME,
    SCHEMA_VERSION,
)
from quantization.output_checkpoint import read_json_no_duplicates
from training_plan import early_stopping_decision


TASKS = ("translation", "summarization")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_SPLIT_ROLES = {
    "train": "optimizer updates",
    "validation": "monitoring and checkpoint selection only",
    "full_test": "sealed final evaluation only",
    "human_gold": "sealed final evaluation only",
}
PRODUCTION_SMOKE_CHECKS = (
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
EXPECTED_PRODUCTION_WANDB = {
    "mode": "online",
    "entity": "junhyuk-tak-markr",
    "project": "modern-korean-lm",
    "run_id": "solar_open2_hardterm_2epoch_earlystop_20260727_v2",
    "run_name": (
        "Solar-Open2-250B hard-term translation-summary 2epoch earlystop "
        "step18631 maxlen9600 v2"
    ),
    "resume": "allow",
}
B200_PRODUCTION_WANDB = {
    "mode": "online",
    "entity": "junhyuk-tak-markr",
    "project": "modern-korean-lm",
    "run_id": "solar_open2_b200_1epoch_20260802_v2",
    "run_name": "Solar Open2 250B 2xB200 1epoch len9600 optimized v2",
    "resume": "allow",
}
B200_FINAL_STEP = 9_316
B200_TOTAL_OCCURRENCES = 149_048
B200_LOGICAL_DEVICES = {"cuda:0", "cuda:1"}


def _fail(label: str, actual: Any, expected: Any) -> None:
    raise RuntimeError(f"{label} mismatch: actual={actual!r}, expected={expected!r}")


def _expect(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        _fail(label, actual, expected)


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be an integer: {value!r}")
    if minimum is not None and value < minimum:
        raise RuntimeError(f"{label} must be >= {minimum}: {value!r}")
    return value


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be numeric: {value!r}")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise RuntimeError(f"{label} is invalid: {value!r}")
    return result


def _sha256_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise RuntimeError(f"{label} is not a SHA-256 digest: {value!r}")
    return value


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is missing or symlinked: {path}")
    return path


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
    for key, expected in (
        ("use_reentrant", False),
        ("preserve_rng_state", True),
        ("early_stop", True),
    ):
        value = contract.get(key)
        if not isinstance(value, bool) or value is not expected:
            _fail(f"{label} {key}", value, expected)
    _expect(
        f"{label} determinism_check",
        contract.get("determinism_check"),
        "default",
    )
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
    expected_devices = {str(layer): f"cuda:{layer // 16}" for layer in range(48)}
    _expect(
        f"{label} layer_input_devices",
        _object(
            contract.get("layer_input_devices"),
            f"{label} layer input devices",
        ),
        expected_devices,
    )
    return contract


def current_file_hashes(directory: Path) -> dict[str, str]:
    if directory.is_symlink() or not directory.is_dir():
        raise RuntimeError(f"adapter directory is missing or symlinked: {directory}")
    result: dict[str, str] = {}
    for path in sorted(directory.iterdir()):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"adapter contains a non-regular entry: {path}")
        result[path.name] = sha256(path)
    return result


def _scan_jsonl(
    path: Path,
    entry: dict[str, Any],
    *,
    collect_quality: bool,
) -> dict[str, Any]:
    _regular_file(path, f"training data {path.name}")
    digest = hashlib.sha256()
    rows = 0
    by_task: Counter[str] = Counter()
    quality_by_task: dict[str, Counter[str]] = {task: Counter() for task in TASKS}
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RuntimeError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_number}: expected JSON object")
            task = row.get("task")
            if task not in TASKS:
                raise RuntimeError(f"{path}:{line_number}: invalid task {task!r}")
            rows += 1
            by_task[task] += 1
            if collect_quality:
                quality = row.get("quality")
                if not isinstance(quality, str) or not quality:
                    raise RuntimeError(
                        f"{path}:{line_number}: invalid quality {quality!r}"
                    )
                quality_by_task[task][quality] += 1
    actual_sha256 = digest.hexdigest()
    _expect(f"{path.name} SHA-256", actual_sha256, entry.get("sha256"))
    _expect(f"{path.name} row count", rows, entry.get("rows"))
    _expect(f"{path.name} task counts", dict(by_task), entry.get("by_task"))
    if collect_quality:
        aggregate = Counter()
        for counts in quality_by_task.values():
            aggregate.update(counts)
        _expect(f"{path.name} quality counts", dict(aggregate), entry.get("quality"))
    return {
        "path": str(path),
        "sha256": actual_sha256,
        "rows": rows,
        "by_task": dict(by_task),
        "quality_by_task": {task: dict(quality_by_task[task]) for task in TASKS},
    }


def current_data_evidence(data_contract: dict[str, Any]) -> dict[str, Any]:
    data_dir = Path(str(data_contract.get("prepared_data_dir") or "")).resolve()
    files = _object(data_contract.get("files"), "data contract files")
    train_entry = _object(files.get("train"), "train data contract")
    validation_entry = _object(files.get("validation"), "validation data contract")
    train = _scan_jsonl(
        data_dir / str(train_entry.get("name") or ""),
        train_entry,
        collect_quality=True,
    )
    validation = _scan_jsonl(
        data_dir / str(validation_entry.get("name") or ""),
        validation_entry,
        collect_quality=False,
    )
    return {"train": train, "validation": validation}


def _validate_source_shards(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != EXPECTED_SOURCE_SHARD_COUNT:
        raise RuntimeError(
            f"{label} must seal exactly {EXPECTED_SOURCE_SHARD_COUNT} BF16 shards"
        )
    expected_names = [
        f"model-{index:05d}-of-{EXPECTED_SOURCE_SHARD_COUNT:05d}.safetensors"
        for index in range(1, EXPECTED_SOURCE_SHARD_COUNT + 1)
    ]
    required_fields = {
        "filename",
        "file_bytes",
        "payload_bytes",
        "tensor_count",
        "sha256",
    }
    observed_names: list[str] = []
    total_payload_bytes = 0
    total_tensors = 0
    for index, item in enumerate(value):
        record = _object(item, f"{label} shard {index + 1}")
        _expect(f"{label} shard fields", set(record), required_fields)
        filename = record.get("filename")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise RuntimeError(f"{label} has an unsafe shard name: {filename!r}")
        observed_names.append(filename)
        file_bytes = _integer(
            record.get("file_bytes"),
            f"{label} {filename} file bytes",
            minimum=1,
        )
        payload_bytes = _integer(
            record.get("payload_bytes"),
            f"{label} {filename} payload bytes",
            minimum=1,
        )
        if payload_bytes >= file_bytes:
            raise RuntimeError(
                f"{label} {filename} payload is not smaller than its file"
            )
        total_payload_bytes += payload_bytes
        total_tensors += _integer(
            record.get("tensor_count"),
            f"{label} {filename} tensor count",
            minimum=1,
        )
        _sha256_text(record.get("sha256"), f"{label} {filename} SHA")
    _expect(f"{label} shard coverage", observed_names, expected_names)
    _expect(
        f"{label} tensor coverage",
        total_tensors,
        EXPECTED_SOURCE_TENSOR_COUNT,
    )
    _expect(
        f"{label} payload coverage",
        total_payload_bytes,
        EXPECTED_SOURCE_TOTAL_SIZE,
    )
    return value


def _training_expert_identity(
    expert_dir: Path,
    *,
    source: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = _regular_file(
        expert_dir / MANIFEST_FILENAME, "training expert manifest"
    )
    index_path = _regular_file(
        expert_dir / EXPERT_INDEX_FILENAME, "training expert index"
    )
    manifest = read_json_no_duplicates(manifest_path)
    for key, expected in (
        ("schema", LAYER_FLAT_SCHEMA_NAME),
        ("schema_version", SCHEMA_VERSION),
        ("status", "SEALED"),
        ("complete_model_checkpoint", False),
        ("base_model_id", BASE_MODEL_ID),
        ("base_revision", BASE_REVISION),
    ):
        _expect(f"training expert {key}", manifest.get(key), expected)
    sealed_source = _object(manifest.get("source"), "training expert source provenance")
    for key, expected in (
        ("source_dir", source["path"]),
        ("index_sha256", source["index_sha256"]),
        ("config_sha256", source["config_sha256"]),
    ):
        _expect(f"training expert source {key}", sealed_source.get(key), expected)
    source_shards = _validate_source_shards(
        sealed_source.get("shards"), "training expert source"
    )
    index_sha256 = sha256(index_path)
    _expect(
        "training expert sealed index SHA",
        manifest.get("expert_index_sha256"),
        index_sha256,
    )
    return {
        "path": str(expert_dir),
        "manifest_sha256": sha256(manifest_path),
        "expert_index_sha256": index_sha256,
        "source_shards": source_shards,
    }


def _validate_production_smoke_receipt(
    value: Any,
    *,
    source_dir: Path,
    expert_dir: Path,
    pipeline: dict[str, Any],
) -> dict[str, str]:
    summary = _object(value, "production smoke receipt identity")
    _expect(
        "production smoke receipt identity fields",
        set(summary),
        {"path", "sha256"},
    )
    path = Path(str(summary.get("path") or "")).resolve()
    _expect("production smoke receipt path", summary.get("path"), str(path))
    _regular_file(path, "production smoke receipt")
    actual_sha256 = sha256(path)
    _expect(
        "production smoke receipt SHA",
        summary.get("sha256"),
        actual_sha256,
    )
    receipt = read_json_no_duplicates(path)
    wanted = {
        "status": "PASS",
        "technical_production_gate_passed": True,
        "authorization_scope": "technical_smoke_gate_only",
        "human_production_approval_required": True,
        "human_production_approval_recorded": False,
        "source_dir": str(source_dir),
        "expert_nf4_dir": str(expert_dir),
        "model_contract_sha256": sha256(CONFIG_DIR / "model.json"),
        "recipe_sha256": sha256(CONFIG_DIR / "recipe.json"),
        "data_contract_sha256": sha256(CONFIG_DIR / "data_contract.json"),
        "pipeline_code": pipeline,
        "environment_lock_sha256": sha256(
            PROJECT / "requirements/solar-open-2.lock.txt"
        ),
    }
    for key, expected in wanted.items():
        _expect(f"production smoke receipt {key}", receipt.get(key), expected)
    tokenizer_summary = _object(
        receipt.get("tokenizer_contract_receipt"),
        "production smoke tokenizer contract receipt",
    )
    _expect(
        "production smoke tokenizer receipt identity fields",
        set(tokenizer_summary),
        {"path", "sha256"},
    )
    tokenizer_path_value = tokenizer_summary.get("path")
    if not isinstance(tokenizer_path_value, str) or not tokenizer_path_value:
        raise RuntimeError("production smoke tokenizer receipt path is invalid")
    tokenizer_unresolved = Path(tokenizer_path_value)
    if tokenizer_unresolved.is_symlink():
        raise RuntimeError("production smoke tokenizer contract receipt is symlinked")
    tokenizer_path = tokenizer_unresolved.resolve()
    _expect(
        "production smoke tokenizer receipt path",
        tokenizer_path_value,
        str(tokenizer_path),
    )
    _regular_file(tokenizer_path, "production smoke tokenizer contract receipt")
    _expect(
        "production smoke tokenizer receipt SHA",
        tokenizer_summary.get("sha256"),
        sha256(tokenizer_path),
    )
    checks = _object(receipt.get("checks"), "production smoke receipt checks")
    failed = [name for name in PRODUCTION_SMOKE_CHECKS if checks.get(name) != "PASS"]
    if failed:
        raise RuntimeError(
            f"production smoke receipt lacks required PASS checks: {failed}"
        )
    runtime_budget = _object(
        receipt.get("runtime_budget_projection"),
        "production smoke runtime budget",
    )
    feasibility = _object(
        runtime_budget.get("feasibility"),
        "production smoke runtime feasibility",
    )
    _expect(
        "production smoke runtime feasibility status",
        feasibility.get("status"),
        "PASS",
    )
    recipe = read_json_no_duplicates(CONFIG_DIR / "recipe.json")
    validation = _object(recipe.get("validation"), "validation recipe")
    policy = _object(validation.get("early_stopping"), "early-stopping policy")
    _expect(
        "production smoke early-stopping policy",
        receipt.get("early_stopping_policy"),
        policy,
    )
    decision = _object(
        receipt.get("early_stopping_decision"),
        "production smoke early-stopping decision",
    )
    _expect(
        "production smoke early-stopping status",
        decision.get("status"),
        "PENDING_MINIMUM",
    )
    _expect(
        "production smoke early-stopping trigger",
        decision.get("triggered"),
        False,
    )
    _expect(
        "production smoke early-stopping evaluated step",
        decision.get("evaluated_step"),
        1,
    )
    smoke_history = receipt.get("early_stopping_history")
    if not isinstance(smoke_history, list):
        raise RuntimeError("production smoke early-stopping history is missing")
    _expect(
        "production smoke early-stopping history schedule",
        [
            (item.get("step"), item.get("scope"))
            for item in smoke_history
            if isinstance(item, dict)
        ],
        [(1, "fixed_subset")],
    )
    expected_decision = early_stopping_decision(
        smoke_history,
        completed_step=1,
        policy=policy,
        optimizer_steps=int(recipe["optimizer_steps"]),
        full_validation_steps=[
            int(step) for step in validation["full_validation_steps"]
        ],
        occurrences_per_step=int(
            recipe["microbatch_policy"]["optimizer_step_occurrences"]
        ),
        occurrences_per_epoch=int(
            recipe["training_deck"]["occurrences_per_epoch"]["total"]
        ),
    )
    _expect(
        "production smoke recomputed early-stopping decision",
        decision,
        expected_decision,
    )
    return {"path": str(path), "sha256": actual_sha256}


def _validate_state_contract(
    manifest: dict[str, Any],
    *,
    source_dir: Path,
    expert_dir: Path,
    expert: dict[str, Any],
    data_contract: dict[str, Any],
    pipeline: dict[str, Any],
) -> dict[str, Any]:
    state = _object(manifest.get("state_contract"), "training state_contract")
    download_sha256 = _sha256_text(
        state.get("download_receipt_sha256"), "download receipt SHA"
    )
    production_smoke_receipt = _validate_production_smoke_receipt(
        state.get("production_smoke_receipt"),
        source_dir=source_dir,
        expert_dir=expert_dir,
        pipeline=pipeline,
    )
    expected = {
        "format_version": 4,
        "source_dir": str(source_dir),
        "expert_nf4_dir": str(expert_dir),
        "download_receipt_sha256": download_sha256,
        "expert_manifest_sha256": expert["manifest_sha256"],
        "model_contract_sha256": sha256(CONFIG_DIR / "model.json"),
        "recipe_sha256": sha256(CONFIG_DIR / "recipe.json"),
        "data_contract_sha256": sha256(CONFIG_DIR / "data_contract.json"),
        "train_sha256": data_contract["files"]["train"]["sha256"],
        "validation_sha256": data_contract["files"]["validation"]["sha256"],
        "pipeline_code_sha256": pipeline["sha256"],
        "environment_lock_sha256": sha256(
            PROJECT / "requirements/solar-open-2.lock.txt"
        ),
        "smoke": False,
        "production_smoke_receipt": production_smoke_receipt,
        "wandb": EXPECTED_PRODUCTION_WANDB,
    }
    _expect("training state contract", state, expected)
    return state


def _validate_loader_evidence(
    manifest: dict[str, Any],
    *,
    source_dir: Path,
    expert_dir: Path,
    source: dict[str, Any],
    expert: dict[str, Any],
) -> None:
    loader = _object(manifest.get("loader_audit"), "training loader audit")
    _expect("training loader status", loader.get("status"), "PASS")
    _expect("training loader base revision", loader.get("base_revision"), BASE_REVISION)
    hardware = _object(loader.get("hardware"), "training loader hardware")
    _expect("training loader GPU count", hardware.get("device_count"), 3)
    inventory = _object(
        loader.get("source_inventory"), "training loader source inventory"
    )
    for key, expected in (
        ("source_dir", str(source_dir)),
        ("index_sha256", source["index_sha256"]),
        ("config_sha256", source["config_sha256"]),
        ("tensor_count", EXPECTED_SOURCE_TENSOR_COUNT),
        ("shard_count", EXPECTED_SOURCE_SHARD_COUNT),
    ):
        _expect(f"training loader source {key}", inventory.get(key), expected)
    loader_shards = _validate_source_shards(
        inventory.get("shards"), "training loader source"
    )
    _expect(
        "training source shard payload seal",
        loader_shards,
        expert["source_shards"],
    )
    expert_preflight = _object(
        loader.get("expert_preflight"), "training loader expert preflight"
    )
    for key, expected in (
        ("status", "PASS"),
        ("production_eligible", True),
        ("checkpoint_dir", str(expert_dir)),
        ("manifest_sha256", expert["manifest_sha256"]),
        ("expert_index_sha256", expert["expert_index_sha256"]),
    ):
        _expect(
            f"training loader expert {key}",
            expert_preflight.get(key),
            expected,
        )
    model_audit = _object(loader.get("model_audit"), "training loaded-model audit")
    for key, expected in (
        ("status", "PASS"),
        ("lora_target_bf16_linear_count", 528),
        ("meta_parameter_count", 0),
        ("meta_buffer_count", 0),
        ("base_trainable_parameter_count", 0),
        ("model_parallel_hook_count", 51),
        ("gradient_checkpointing", True),
        ("device_aware_checkpoint_layer_count", 48),
    ):
        _expect(f"training loaded-model {key}", model_audit.get(key), expected)
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
        model_audit.get("device_map"),
        "training loaded-model device map",
    )
    for layer in range(48):
        _expect(
            f"training layer {layer} checkpoint/device-map binding",
            loader_checkpointing["layer_input_devices"][str(layer)],
            loaded_device_map.get(f"model.layers.{layer}"),
        )
    training_checkpointing = _validate_gradient_checkpointing_contract(
        manifest.get("gradient_checkpointing_contract"),
        "training final gradient-checkpointing contract",
    )
    _expect(
        "loader/final gradient-checkpointing contract",
        training_checkpointing,
        loader_checkpointing,
    )


def _validate_optimization_evidence(
    manifest: dict[str, Any],
    *,
    recipe: dict[str, Any],
    model_contract: dict[str, Any],
) -> None:
    lora = recipe["lora"]
    expected_parameters = int(lora["expected_trainable_parameters"])
    expected_modules = int(lora["expected_module_count"])
    _expect(
        "training trainable parameters",
        manifest.get("trainable_parameters"),
        expected_parameters,
    )
    module_audit = _object(manifest.get("lora_module_audit"), "LoRA module audit")
    expected_names = expected_lora_module_names(model_contract["architecture_contract"])
    for key, expected in (
        ("status", "PASS"),
        ("module_count", expected_modules),
        ("module_names_sha256", canonical_sha256(sorted(expected_names))),
        ("module_class", "torch.nn.Linear"),
    ):
        _expect(f"LoRA module {key}", module_audit.get(key), expected)

    gradient = _object(
        manifest.get("first_update_gradient_audit"), "first gradient audit"
    )
    _expect("first gradient finite", gradient.get("finite"), True)
    with_gradient = _integer(
        gradient.get("parameters_with_gradient"),
        "first gradient parameter count",
        minimum=1,
    )
    if with_gradient > expected_modules * 2:
        raise RuntimeError("first gradient parameter count exceeds LoRA tensors")
    _integer(
        gradient.get("nonzero_gradient_elements"),
        "first nonzero gradient elements",
        minimum=1,
    )
    _finite(gradient.get("l2"), "first gradient l2", positive=True)

    delta = _object(manifest.get("first_update_adapter_delta"), "first adapter update")
    before_sha256 = _sha256_text(delta.get("before_sha256"), "pre-update adapter SHA")
    after_sha256 = _sha256_text(delta.get("after_sha256"), "post-update adapter SHA")
    if delta.get("changed") is not True:
        raise RuntimeError("first adapter update changed must be true")
    if before_sha256 == after_sha256:
        raise RuntimeError("first adapter update did not change the LoRA digest")
    _expect(
        "first effective adapter optimizer step",
        _integer(
            delta.get("optimizer_step"),
            "first effective adapter optimizer step",
            minimum=1,
        ),
        1,
    )
    warmup_steps = max(
        1,
        int(float(recipe["warmup_ratio"]) * int(recipe["optimizer_steps"])),
    )
    _expect(
        "first effective adapter learning rate",
        _finite(
            delta.get("learning_rate_used"),
            "first effective adapter learning rate",
            positive=True,
        ),
        float(recipe["learning_rate"]) / warmup_steps,
    )
    for side in ("before", "after"):
        stats = _object(delta.get(side), f"first adapter {side} stats")
        for key, expected in (
            ("parameter_count", expected_modules * 2),
            ("element_count", expected_parameters),
            ("finite", True),
        ):
            _expect(f"first adapter {side} {key}", stats.get(key), expected)
        _expect(
            f"first adapter {side} dtype parameter counts",
            stats.get("dtype_parameter_counts"),
            {"torch.float32": expected_modules * 2},
        )
        _expect(
            f"first adapter {side} dtype element counts",
            stats.get("dtype_element_counts"),
            {"torch.float32": expected_parameters},
        )
        _finite(stats.get("l1"), f"first adapter {side} l1")
        _finite(stats.get("l2"), f"first adapter {side} l2")

    frozen_before = _object(manifest.get("frozen_probe_before"), "frozen probe before")
    frozen_after = _object(
        manifest.get("frozen_probe_after_update"), "frozen probe after"
    )
    _expect("frozen base probe", frozen_after, frozen_before)
    _expect(
        "frozen base sampled parameter count",
        frozen_before.get("sampled_parameter_count"),
        32,
    )
    _sha256_text(frozen_before.get("values_sha256"), "frozen base values SHA")

    supervision = _object(
        manifest.get("first_output_only_supervision"),
        "first output-only supervision",
    )
    expected_keys = {
        "prompt_token_count",
        "answer_token_count",
        "assistant_termination_token_count",
        "assistant_termination_token_id",
        "template_suffix_token_count",
        "template_suffix_starts_with_eos",
        "supervised_sequence_ends_at_eos",
        "prompt_labels_all_masked",
        "prompt_weights_all_zero",
        "supervised_prefix_is_answer_body",
        "reasoning_effort",
        "answer_label_prefix_removed",
    }
    _expect("output-only supervision fields", set(supervision), expected_keys)
    for key in ("prompt_token_count", "answer_token_count"):
        _integer(supervision.get(key), f"output-only {key}", minimum=1)
    termination_count = _integer(
        supervision.get("assistant_termination_token_count"),
        "output-only assistant termination tokens",
        minimum=1,
    )
    _expect(
        "output-only assistant termination token count",
        termination_count,
        1,
    )
    _integer(
        supervision.get("assistant_termination_token_id"),
        "output-only assistant termination token id",
        minimum=0,
    )
    _integer(
        supervision.get("template_suffix_token_count"),
        "output-only template suffix token count",
        minimum=1,
    )
    for key, expected in (
        ("prompt_labels_all_masked", True),
        ("prompt_weights_all_zero", True),
        ("supervised_prefix_is_answer_body", True),
        ("template_suffix_starts_with_eos", True),
        ("supervised_sequence_ends_at_eos", True),
        ("reasoning_effort", "none"),
    ):
        _expect(f"output-only {key}", supervision.get(key), expected)
    if not isinstance(supervision.get("answer_label_prefix_removed"), bool):
        raise RuntimeError("output-only prefix-removal evidence must be boolean")

    max_observed = _integer(
        manifest.get("max_observed_tokens"),
        "maximum observed tokens",
        minimum=1,
    )
    if max_observed > int(recipe["max_length"]):
        raise RuntimeError("training observed a row beyond the max-length contract")
    _integer(
        manifest.get("total_hard_term_weighted_tokens"),
        "hard-term weighted token total",
        minimum=1,
    )

    device = _object(manifest.get("device_audit"), "training device audit")
    _expect("CPU/disk model offload", device.get("cpu_disk_offload"), False)
    mapped = {str(value) for value in device.get("hf_device_map_devices") or []}
    _expect("model-parallel device map", mapped, {"0", "1", "2"})
    parameter_devices = _object(
        device.get("parameter_devices"), "parameter device counts"
    )
    if set(parameter_devices) != {"cuda:0", "cuda:1", "cuda:2"}:
        raise RuntimeError(f"unexpected parameter devices: {sorted(parameter_devices)}")
    for key, count in parameter_devices.items():
        _integer(count, f"{key} parameter count", minimum=1)


def _expected_sampler(
    data_evidence: dict[str, Any],
    recipe: dict[str, Any],
) -> dict[str, Any]:
    expert_label = str(recipe["expert_quality_label"])
    boost = int(recipe["expert_oversampling_factor"])
    epochs = int(recipe["epochs"])
    base_task_counts = {
        task: int(data_evidence["train"]["by_task"][task]) for task in TASKS
    }
    expert_task_counts: dict[str, int] = {}
    for task in TASKS:
        qualities = Counter(data_evidence["train"]["quality_by_task"][task])
        expert_count = qualities.get(expert_label, 0)
        if expert_count <= 0:
            raise RuntimeError(f"training data has no expert rows for {task}")
        expert_task_counts[task] = expert_count
    task_examples_per_epoch = {
        task: (base_task_counts[task] + (boost - 1) * expert_task_counts[task])
        for task in TASKS
    }
    examples_per_epoch = sum(task_examples_per_epoch.values())
    result = {
        "base_rows": int(data_evidence["train"]["rows"]),
        "expert_rows": sum(expert_task_counts.values()),
        "expert_boost": boost,
        "epochs": epochs,
        "examples_per_epoch": examples_per_epoch,
        "total_examples": examples_per_epoch * epochs,
        "base_task_counts": base_task_counts,
        "expert_task_counts": expert_task_counts,
        "task_examples_per_epoch": task_examples_per_epoch,
        "task_total_examples": {
            task: count * epochs for task, count in task_examples_per_epoch.items()
        },
    }
    deck = _object(recipe.get("training_deck"), "training deck recipe")
    _expect(
        "training deck base rows",
        result["base_rows"],
        deck.get("base_unique_rows"),
    )
    _expect(
        "training deck expert rows",
        result["expert_rows"],
        deck.get("expert_unique_rows"),
    )
    _expect(
        "training deck examples per epoch",
        result["examples_per_epoch"],
        _object(
            deck.get("occurrences_per_epoch"),
            "training deck per-epoch occurrences",
        ).get("total"),
    )
    _expect(
        "training deck total examples",
        result["total_examples"],
        _object(
            deck.get("total_occurrences"),
            "training deck total occurrences",
        ).get("total"),
    )
    _expect(
        "training deck task total examples",
        result["task_total_examples"],
        {task: deck["total_occurrences"][task] for task in TASKS},
    )
    return result


def _expected_training_exposure(
    *,
    recipe: dict[str, Any],
    data_evidence: dict[str, Any],
    consumed_occurrences: int,
) -> tuple[dict[str, int], dict[str, int]]:
    deck = _object(recipe.get("training_deck"), "training deck recipe")
    total_occurrences = _object(
        deck.get("total_occurrences"), "training deck total occurrences"
    )
    expected_total = _integer(
        total_occurrences.get("total"),
        "training deck total occurrences",
        minimum=1,
    )
    if consumed_occurrences == expected_total:
        task_occurrences = {
            task: _integer(
                total_occurrences.get(task),
                f"training deck {task} total occurrences",
                minimum=1,
            )
            for task in TASKS
        }
        quality_counts = Counter()
        for task in TASKS:
            quality_counts.update(data_evidence["train"]["quality_by_task"][task])
        expert_label = str(recipe["expert_quality_label"])
        quality_occurrences = {
            quality: (
                int(count)
                * int(recipe["epochs"])
                * (
                    int(recipe["expert_oversampling_factor"])
                    if quality == expert_label
                    else 1
                )
            )
            for quality, count in quality_counts.items()
        }
        return task_occurrences, quality_occurrences

    train_rows = read_jsonl(Path(data_evidence["train"]["path"]))
    sampler = GlobalAugmentedDeckSampler(
        train_rows,
        expert_label=str(recipe["expert_quality_label"]),
        expert_boost=int(recipe["expert_oversampling_factor"]),
        epochs=int(recipe["epochs"]),
        seed=int(recipe["seed"]),
    )
    if not 0 < consumed_occurrences <= len(sampler):
        raise RuntimeError(
            "training consumed occurrence count is outside the sampler deck"
        )
    task_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    for _ in range(consumed_occurrences):
        row = train_rows[sampler.draw_index()]
        task_counts[str(row["task"])] += 1
        quality_counts[str(row.get("quality") or "")] += 1
    return (
        {task: int(task_counts[task]) for task in TASKS},
        {quality: int(count) for quality, count in quality_counts.items()},
    )


def _validate_training_completion(
    manifest: dict[str, Any],
    *,
    recipe: dict[str, Any],
    data_evidence: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    deck = _object(recipe.get("training_deck"), "training deck recipe")
    total_occurrences = _object(
        deck.get("total_occurrences"), "training deck total occurrences"
    )
    expected_total = _integer(
        total_occurrences.get("total"),
        "training deck total occurrences",
        minimum=1,
    )
    expected_steps = _integer(
        recipe.get("optimizer_steps"), "recipe optimizer steps", minimum=1
    )
    occurrences_per_step = _integer(
        _object(recipe.get("microbatch_policy"), "recipe microbatch policy").get(
            "optimizer_step_occurrences"
        ),
        "recipe optimizer-step occurrences",
        minimum=1,
    )
    _expect(
        "optimizer-step occurrence coverage",
        expected_steps * occurrences_per_step,
        expected_total,
    )
    _expect(
        "derived optimizer steps",
        deck.get("derived_optimizer_steps"),
        expected_steps,
    )
    status = manifest.get("status")
    if status not in {"COMPLETED", "EARLY_STOPPED"}:
        raise RuntimeError(f"unsupported training terminal status: {status!r}")
    early_policy = _object(
        _object(recipe.get("validation"), "validation recipe").get("early_stopping"),
        "early-stopping policy",
    )
    actual_steps = _integer(
        manifest.get("completed_updates"),
        "training completed updates",
        minimum=1,
    )
    if status == "COMPLETED":
        wanted_steps = expected_steps
    else:
        wanted_steps = _integer(
            early_policy.get("confirmation_step"),
            "early-stopping confirmation step",
            minimum=1,
        )
        _expect(
            "early-stopping exact terminal step",
            wanted_steps,
            _integer(
                early_policy.get("earliest_stop_step"),
                "early-stopping earliest step",
                minimum=1,
            ),
        )
        if wanted_steps >= expected_steps:
            raise RuntimeError(
                "early-stopping confirmation must precede the full horizon"
            )
    _expect("training actual terminal step", actual_steps, wanted_steps)
    _expect(
        "training target update count",
        manifest.get("target_updates"),
        expected_steps,
    )
    actual_occurrences = actual_steps * occurrences_per_step
    _expect(
        "training consumed occurrences",
        manifest.get("consumed_occurrences"),
        actual_occurrences,
    )
    examples_per_epoch = _integer(
        _object(
            deck.get("occurrences_per_epoch"),
            "training deck per-epoch occurrences",
        ).get("total"),
        "training deck occurrences per epoch",
        minimum=1,
    )
    completed_epochs = actual_occurrences / examples_per_epoch
    if status == "EARLY_STOPPED" and completed_epochs < float(
        early_policy.get("minimum_epochs", 0.0)
    ):
        raise RuntimeError("early-stopped run did not complete the minimum epoch")

    completion = _object(manifest.get("training_completion"), "training completion")
    _expect(
        "training completion fields",
        set(completion),
        {
            "status",
            "epochs",
            "completed_epochs",
            "optimizer_steps",
            "consumed_occurrences",
            "expected_optimizer_steps",
            "expected_occurrences",
            "early_stopping_triggered",
        },
    )
    for key, expected in (
        ("status", status),
        ("epochs", recipe["epochs"]),
        ("optimizer_steps", actual_steps),
        ("consumed_occurrences", actual_occurrences),
        ("expected_optimizer_steps", expected_steps),
        ("expected_occurrences", expected_total),
        ("early_stopping_triggered", status == "EARLY_STOPPED"),
    ):
        _expect(f"training completion {key}", completion.get(key), expected)
    recorded_completed_epochs = _finite(
        completion.get("completed_epochs"),
        "training completion completed epochs",
    )
    if not math.isclose(
        recorded_completed_epochs,
        completed_epochs,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        _fail(
            "training completion completed epochs",
            recorded_completed_epochs,
            completed_epochs,
        )

    coverage = _object(manifest.get("training_coverage"), "training coverage")
    required_fields = {
        "total_occurrences",
        "skipped_occurrences",
        "unconsumed_planned_occurrences",
        "unique_raw_rows",
        "task_occurrences",
        "quality_occurrences",
    }
    missing = required_fields - set(coverage)
    if missing:
        raise RuntimeError(
            f"training coverage lacks required fields: {sorted(missing)}"
        )
    for key, expected in (
        ("total_occurrences", actual_occurrences),
        ("skipped_occurrences", 0),
        (
            "unconsumed_planned_occurrences",
            expected_total - actual_occurrences,
        ),
        ("unique_raw_rows", deck["base_unique_rows"]),
    ):
        _expect(
            f"training coverage {key}",
            _integer(
                coverage.get(key),
                f"training coverage {key}",
                minimum=0,
            ),
            expected,
        )
    _expect(
        "training coverage unique raw/data rows",
        coverage["unique_raw_rows"],
        data_evidence["train"]["rows"],
    )
    task_occurrences = _object(
        coverage.get("task_occurrences"),
        "training coverage task occurrences",
    )
    _expect("training coverage task set", set(task_occurrences), set(TASKS))
    expected_tasks, expected_quality_occurrences = _expected_training_exposure(
        recipe=recipe,
        data_evidence=data_evidence,
        consumed_occurrences=actual_occurrences,
    )
    _expect("training task exposure", task_occurrences, expected_tasks)
    _expect(
        "training task exposure sum",
        sum(task_occurrences.values()),
        actual_occurrences,
    )
    quality_occurrences = _object(
        coverage.get("quality_occurrences"),
        "training coverage quality occurrences",
    )
    _expect(
        "training quality exposure",
        quality_occurrences,
        expected_quality_occurrences,
    )
    _expect(
        "training quality exposure sum",
        sum(quality_occurrences.values()),
        actual_occurrences,
    )
    return (
        completion,
        coverage,
        {
            "status": status,
            "completed_updates": actual_steps,
            "target_updates": expected_steps,
            "consumed_occurrences": actual_occurrences,
            "completed_epochs": completed_epochs,
        },
    )


def _validate_token_length_contract(
    manifest: dict[str, Any],
    *,
    recipe: dict[str, Any],
    data_evidence: dict[str, Any],
) -> dict[str, Any]:
    contract = _object(manifest.get("token_length_contract"), "token length contract")
    _expect("token length contract splits", set(contract), {"train", "validation"})
    required_fields = {
        "rows",
        "by_task",
        "max_length",
        "over_max_count",
        "over_max_rows",
        "minimum",
        "p50",
        "p90",
        "p95",
        "p99",
        "p99_9",
        "maximum",
    }
    max_length = int(recipe["max_length"])
    for split in ("train", "validation"):
        evidence = _object(contract.get(split), f"{split} token length contract")
        _expect(
            f"{split} token length contract fields",
            set(evidence),
            required_fields,
        )
        for key, expected in (
            ("rows", data_evidence[split]["rows"]),
            ("by_task", data_evidence[split]["by_task"]),
            ("max_length", max_length),
            ("over_max_count", 0),
            ("over_max_rows", []),
        ):
            _expect(f"{split} token length {key}", evidence.get(key), expected)
        ordered = [
            _integer(
                evidence.get(key),
                f"{split} token length {key}",
                minimum=1,
            )
            for key in ("minimum", "p50", "p90", "p95", "p99", "p99_9", "maximum")
        ]
        if ordered != sorted(ordered):
            raise RuntimeError(
                f"{split} token length quantiles are not monotonic: {ordered}"
            )
        if ordered[-1] > max_length:
            raise RuntimeError(
                f"{split} token length maximum exceeds recipe: {ordered[-1]}"
            )
    return contract


def _validate_validation_history(
    history: Any,
    *,
    recipe: dict[str, Any],
    data_evidence: dict[str, Any],
    completed_updates: int,
) -> list[dict[str, Any]]:
    if not isinstance(history, list):
        raise RuntimeError("validation history must be a JSON array")
    validation = recipe["validation"]
    target_steps = int(recipe["optimizer_steps"])
    if not 1 <= completed_updates <= target_steps:
        raise RuntimeError("validation history terminal step is invalid")
    every = int(validation["eval_every_steps"])
    full_steps = {
        int(step)
        for step in validation["full_validation_steps"]
        if int(step) <= completed_updates
    }
    scheduled_steps = set(range(every, completed_updates + 1, every)) | full_steps
    expected_schedule = [
        (
            step,
            "full" if step in full_steps else "fixed_subset",
        )
        for step in sorted(scheduled_steps)
    ]
    actual_schedule = [
        (
            _integer(_object(item, "validation record").get("step"), "validation step"),
            item.get("scope"),
        )
        for item in history
    ]
    _expect("validation schedule", actual_schedule, expected_schedule)
    required_fields = {
        "step",
        "scope",
        "selection_loss",
        "translation_loss",
        "summarization_loss",
        "translation_count",
        "summarization_count",
        "dropped_over_max",
        "validation_order_sha256",
        "elapsed_seconds",
    }
    selection_weights = _object(
        validation.get("selection_weights"),
        "validation selection weights",
    )
    _expect("validation selection weight tasks", set(selection_weights), set(TASKS))
    translation_weight = _finite(
        selection_weights.get("translation"),
        "validation translation selection weight",
    )
    summarization_weight = _finite(
        selection_weights.get("summarization"),
        "validation summarization selection weight",
    )
    if not math.isclose(
        translation_weight + summarization_weight,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("validation selection weights must sum to one")
    subset_per_task = int(validation["subset_per_task"])
    validation_counts = data_evidence["validation"]["by_task"]
    for item, (_step, scope) in zip(history, expected_schedule, strict=True):
        _expect("validation record fields", set(item), required_fields)
        _sha256_text(
            item.get("validation_order_sha256"),
            f"step {item['step']} validation order SHA",
        )
        translation_loss = _finite(
            item.get("translation_loss"),
            f"step {item['step']} translation loss",
        )
        summarization_loss = _finite(
            item.get("summarization_loss"),
            f"step {item['step']} summarization loss",
        )
        selection_loss = _finite(
            item.get("selection_loss"),
            f"step {item['step']} selection loss",
        )
        if min(translation_loss, summarization_loss, selection_loss) < 0:
            raise RuntimeError(
                f"step {item['step']} validation loss cannot be negative"
            )
        expected_selection = (
            translation_weight * translation_loss
            + summarization_weight * summarization_loss
        )
        if not math.isclose(
            selection_loss, expected_selection, rel_tol=1e-12, abs_tol=1e-12
        ):
            _fail(
                f"step {item['step']} selection loss",
                selection_loss,
                expected_selection,
            )
        dropped = _object(
            item.get("dropped_over_max"),
            f"step {item['step']} dropped counts",
        )
        if not set(dropped).issubset(TASKS):
            raise RuntimeError(
                f"step {item['step']} has unexpected dropped-count tasks"
            )
        for task in TASKS:
            kept = _integer(
                item.get(f"{task}_count"),
                f"step {item['step']} {task} validation count",
                minimum=1,
            )
            dropped_count = _integer(
                dropped.get(task, 0),
                f"step {item['step']} dropped {task} count",
                minimum=0,
            )
            _expect(
                f"step {item['step']} skipped {task} validation rows",
                dropped_count,
                0,
            )
            source_count = (
                int(validation_counts[task]) if scope == "full" else subset_per_task
            )
            _expect(
                f"step {item['step']} {task} validation coverage",
                kept + dropped_count,
                source_count,
            )
        elapsed = _finite(
            item.get("elapsed_seconds"),
            f"step {item['step']} validation elapsed seconds",
        )
        if elapsed < 0:
            raise RuntimeError("validation elapsed seconds cannot be negative")
    _expect(
        "full validation history steps",
        [item["step"] for item in history if item["scope"] == "full"],
        sorted(full_steps),
    )
    order_hashes = {item["validation_order_sha256"] for item in history}
    if len(order_hashes) != 1:
        raise RuntimeError(
            f"validation history order identity changed: {sorted(order_hashes)}"
        )
    return history


def _validate_early_stopping_evidence(
    manifest: dict[str, Any],
    *,
    recipe: dict[str, Any],
    history: list[dict[str, Any]],
    horizon: dict[str, Any],
) -> dict[str, Any]:
    validation = _object(recipe.get("validation"), "validation recipe")
    policy = _object(validation.get("early_stopping"), "early-stopping policy")
    for key, expected in (
        ("enabled", True),
        ("monitor", "selection_loss"),
        ("mode", "min"),
        ("full_validation_confirmation_required", True),
        ("continue_if_either_task_materially_improves", True),
        ("restore_best_full_validation", True),
        ("sealed_evaluation_not_monitor", True),
    ):
        _expect(f"early-stopping policy {key}", policy.get(key), expected)
    if manifest.get("split_roles") != EXPECTED_SPLIT_ROLES:
        raise RuntimeError(
            "sealed test/human roles cannot participate in early stopping"
        )

    expected = early_stopping_decision(
        history,
        completed_step=int(horizon["completed_updates"]),
        policy=policy,
        optimizer_steps=int(recipe["optimizer_steps"]),
        full_validation_steps=[
            int(step) for step in validation["full_validation_steps"]
        ],
        occurrences_per_step=int(
            recipe["microbatch_policy"]["optimizer_step_occurrences"]
        ),
        occurrences_per_epoch=int(
            recipe["training_deck"]["occurrences_per_epoch"]["total"]
        ),
    )
    evidence = _object(
        manifest.get("early_stopping_decision"), "early-stopping evidence"
    )
    _expect("early-stopping evidence", evidence, expected)
    completion = _object(manifest.get("training_completion"), "training completion")
    _expect(
        "training completion early-stopping trigger",
        completion.get("early_stopping_triggered"),
        horizon["status"] == "EARLY_STOPPED",
    )
    completed_updates = int(horizon["completed_updates"])
    _expect(
        "early-stopping evaluated step",
        evidence.get("evaluated_step"),
        completed_updates,
    )
    realized_candidates = [
        int(step)
        for step in validation["full_validation_steps"]
        if int(step) <= completed_updates
    ]
    _expect(
        "early-stopping completed candidates",
        evidence.get("completed_candidate_steps"),
        realized_candidates,
    )
    recorded_epochs = _finite(
        evidence.get("completed_epochs"), "early-stopping completed epochs"
    )
    if not math.isclose(
        recorded_epochs,
        float(horizon["completed_epochs"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        _fail(
            "early-stopping completed epochs",
            recorded_epochs,
            horizon["completed_epochs"],
        )

    if horizon["status"] == "EARLY_STOPPED":
        _expect("early-stopping decision status", evidence.get("status"), "TRIGGERED")
        _expect("early-stopping trigger", evidence.get("triggered"), True)
        _expect(
            "early-stopping terminal confirmation",
            completed_updates,
            int(policy["confirmation_step"]),
        )
    else:
        _expect(
            "completed-run early-stopping status",
            evidence.get("status"),
            "MAXIMUM_REACHED",
        )
        _expect(
            "completed-run early-stopping trigger",
            evidence.get("triggered"),
            False,
        )
        _expect(
            "completed-run terminal horizon",
            completed_updates,
            int(policy["latest_stop_step"]),
        )
    return evidence


def _validate_sealed_adapter(
    directory: Path,
    *,
    label: str,
    expected_files: Any,
    require_attribution: bool,
) -> tuple[dict[str, str], dict[str, Any]]:
    success = _regular_file(directory / "_SUCCESS", f"{label} success marker")
    _expect(
        f"{label} success marker",
        success.read_text(encoding="utf-8").strip(),
        "PASS",
    )
    files = current_file_hashes(directory)
    expected = _object(expected_files, f"{label} recorded file hashes")
    for name, digest in expected.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise RuntimeError(f"{label} has an unsafe recorded filename: {name!r}")
        _sha256_text(digest, f"{label} {name} recorded SHA")
    _expect(f"{label} file hashes", files, expected)
    if require_attribution:
        for name in ("LICENSE", "NOTICE"):
            _expect(
                f"{label} {name}",
                files.get(name),
                sha256(PROJECT / name),
            )
    adapter = validate_adapter(directory)
    for name, digest in adapter["adapter_file_sha256"].items():
        if files.get(name) != digest:
            raise RuntimeError(
                f"{label} inference identity differs from sealed files: {name}"
            )
    return files, adapter


def _validate_checkpoint_selection(
    manifest: dict[str, Any],
    *,
    recipe: dict[str, Any],
    history: list[dict[str, Any]],
    completed_updates: int,
    output_dir: Path,
    evaluation_adapter_dir: Path,
) -> dict[str, Any]:
    validation = _object(recipe.get("validation"), "validation recipe")
    checkpoint = _object(recipe.get("checkpoint"), "checkpoint recipe")
    full_steps = [
        int(step)
        for step in validation["full_validation_steps"]
        if int(step) <= completed_updates
    ]
    policy = str(validation["selection_policy"])
    tie_break = str(validation["selection_tie_break"])
    _expect(
        "validation selection policy",
        policy,
        "argmin_full_validation_selection_loss",
    )
    _expect("validation selection tie break", tie_break, "earlier_optimizer_step")

    full_history = {
        int(item["step"]): item for item in history if item["scope"] == "full"
    }
    _expect("full-validation selection coverage", list(full_history), full_steps)

    record = _object(manifest.get("selection_record"), "selection record")
    required_record_fields = {
        "policy",
        "tie_break",
        "candidate_steps",
        "candidates",
        "selected_step",
        "selected_selection_loss",
    }
    _expect("selection record fields", set(record), required_record_fields)
    for key, expected in (
        ("policy", policy),
        ("tie_break", tie_break),
        ("candidate_steps", full_steps),
    ):
        _expect(f"selection record {key}", record.get(key), expected)
    candidates = record.get("candidates")
    if not isinstance(candidates, list):
        raise RuntimeError("selection record candidates must be a JSON array")
    _expect("selection candidate count", len(candidates), len(full_steps))
    candidate_by_step: dict[int, dict[str, Any]] = {}
    candidate_artifacts: dict[int, dict[str, Any]] = {}
    candidate_fields = {
        "step",
        "selection_loss",
        "translation_loss",
        "summarization_loss",
        "adapter_dir",
        "files_sha256",
    }
    for wanted_step, value in zip(full_steps, candidates, strict=True):
        candidate = _object(value, f"selection candidate step {wanted_step}")
        _expect(
            f"selection candidate step {wanted_step} fields",
            set(candidate),
            candidate_fields,
        )
        step = _integer(
            candidate.get("step"),
            f"selection candidate step {wanted_step}",
            minimum=1,
        )
        _expect(f"selection candidate order at {wanted_step}", step, wanted_step)
        history_item = full_history[step]
        for key in (
            "selection_loss",
            "translation_loss",
            "summarization_loss",
        ):
            loss = _finite(candidate.get(key), f"selection candidate {step} {key}")
            if loss < 0:
                raise RuntimeError(
                    f"selection candidate {step} {key} cannot be negative"
                )
            _expect(
                f"selection candidate/history {step} {key}",
                loss,
                float(history_item[key]),
            )
        expected_dir = output_dir / "candidates" / f"step-{step:05d}"
        candidate_dir = Path(str(candidate.get("adapter_dir") or "")).resolve()
        _expect(f"selection candidate {step} path", candidate_dir, expected_dir)
        files, adapter = _validate_sealed_adapter(
            candidate_dir,
            label=f"selection candidate {step}",
            expected_files=candidate.get("files_sha256"),
            require_attribution=False,
        )
        candidate_by_step[step] = candidate
        candidate_artifacts[step] = {
            "adapter_dir": str(candidate_dir),
            "files_sha256": files,
            "adapter_inference_identity": adapter,
        }
    checkpoint_records = _object(
        manifest.get("candidate_checkpoints"), "candidate checkpoints"
    )
    _expect(
        "candidate checkpoint step keys",
        {int(step) for step in checkpoint_records},
        set(full_steps),
    )
    for step in full_steps:
        checkpoint_record = _object(
            checkpoint_records[str(step)],
            f"candidate checkpoint {step}",
        )
        _expect(
            f"candidate checkpoint {step} fields",
            set(checkpoint_record),
            {"step", "adapter_dir", "files_sha256"},
        )
        selected_record = candidate_by_step[step]
        _expect(
            f"candidate checkpoint/selection record {step}",
            checkpoint_record,
            {
                "step": selected_record["step"],
                "adapter_dir": selected_record["adapter_dir"],
                "files_sha256": selected_record["files_sha256"],
            },
        )

    selected_candidate = min(
        candidates,
        key=lambda item: (float(item["selection_loss"]), int(item["step"])),
    )
    selected_step = int(selected_candidate["step"])
    selected_loss = float(selected_candidate["selection_loss"])
    _expect(
        "selection record selected step",
        record.get("selected_step"),
        selected_step,
    )
    _expect(
        "selection record selected loss",
        _finite(
            record.get("selected_selection_loss"),
            "selection record selected loss",
        ),
        selected_loss,
    )

    best_dir = output_dir / str(checkpoint["best_adapter_dirname"])
    final_dir = output_dir / str(checkpoint["final_adapter_dirname"])
    if best_dir == final_dir:
        raise RuntimeError("best and final adapter paths must be distinct")
    _expect("evaluation adapter path", evaluation_adapter_dir, best_dir)

    selected = _object(manifest.get("selected_checkpoint"), "selected checkpoint")
    selected_fields = {
        "policy",
        "tie_break",
        "step",
        "selection_loss",
        "source_checkpoint_dir",
        "adapter_dir",
        "files_sha256",
    }
    _expect("selected checkpoint fields", set(selected), selected_fields)
    selected_source = Path(str(selected.get("source_checkpoint_dir") or "")).resolve()
    for key, expected in (
        ("policy", policy),
        ("tie_break", tie_break),
        ("step", selected_step),
        ("selection_loss", selected_loss),
        (
            "source_checkpoint_dir",
            str(Path(selected_candidate["adapter_dir"]).resolve()),
        ),
        ("adapter_dir", str(best_dir)),
    ):
        actual = selected.get(key)
        if key in {"source_checkpoint_dir", "adapter_dir"}:
            actual = str(Path(str(actual or "")).resolve())
        _expect(f"selected checkpoint {key}", actual, expected)
    _expect(
        "selected checkpoint source path",
        selected_source,
        output_dir / "candidates" / f"step-{selected_step:05d}",
    )
    best_files, best_adapter = _validate_sealed_adapter(
        best_dir,
        label="best adapter",
        expected_files=selected.get("files_sha256"),
        require_attribution=True,
    )
    _expect(
        "best/selected-candidate adapter identity",
        best_adapter["adapter_file_sha256"],
        candidate_artifacts[selected_step]["adapter_inference_identity"][
            "adapter_file_sha256"
        ],
    )

    final = _object(manifest.get("final_checkpoint"), "final checkpoint")
    final_fields = {"step", "adapter_dir", "files_sha256"}
    _expect("final checkpoint fields", set(final), final_fields)
    _expect(
        "final checkpoint step",
        _integer(final.get("step"), "final checkpoint step", minimum=1),
        completed_updates,
    )
    _expect(
        "final checkpoint adapter path",
        Path(str(final.get("adapter_dir") or "")).resolve(),
        final_dir,
    )
    final_files, final_adapter = _validate_sealed_adapter(
        final_dir,
        label="final adapter",
        expected_files=final.get("files_sha256"),
        require_attribution=True,
    )
    _expect(
        "final/terminal-candidate adapter identity",
        final_adapter["adapter_file_sha256"],
        candidate_artifacts[completed_updates]["adapter_inference_identity"][
            "adapter_file_sha256"
        ],
    )
    return {
        "selection_record": record,
        "selected_checkpoint": selected,
        "selected_step": selected_step,
        "selected_selection_loss": selected_loss,
        "candidate_artifacts": candidate_artifacts,
        "best_adapter_dir": str(best_dir),
        "best_adapter_files_sha256": best_files,
        "best_adapter_inference_identity": best_adapter,
        "final_adapter_dir": str(final_dir),
        "final_adapter_files_sha256": final_files,
        "final_adapter_inference_identity": final_adapter,
    }


def validate_b200_manifest_contract(
    manifest: dict[str, Any],
    *,
    config_identity: dict[str, Any],
    recipe: dict[str, Any],
    model_contract: dict[str, Any],
) -> None:
    """Validate the CPU-safe, exact B200 terminal and placement contract."""

    exact = {
        "status": "COMPLETED",
        "profile": "b200",
        "config_identity": config_identity,
        "model_name": model_contract["derivative_model_name"],
        "attribution": model_contract["attribution"],
        "config": recipe,
        "output_only_contract": recipe["output_only_contract"],
        "completed_updates": B200_FINAL_STEP,
        "target_updates": B200_FINAL_STEP,
        "consumed_occurrences": B200_TOTAL_OCCURRENCES,
        "backward_passes": B200_TOTAL_OCCURRENCES,
        "split_roles": EXPECTED_SPLIT_ROLES,
        "wandb": B200_PRODUCTION_WANDB,
    }
    for field, expected in exact.items():
        _expect(f"B200 training manifest {field}", manifest.get(field), expected)
    if "smoke" in manifest:
        raise RuntimeError("B200 production manifest contains smoke evidence")

    occurrence = _object(
        manifest.get("occurrence_contract"), "B200 occurrence contract"
    )
    _expect(
        "B200 occurrence contract",
        occurrence,
        {
            "nominal_step_occurrences": 16,
            "total_occurrences": B200_TOTAL_OCCURRENCES,
            "optimizer_steps": B200_FINAL_STEP,
            "final_step_occurrences": 8,
        },
    )
    tail = _object(recipe.get("exact_tail"), "B200 exact-tail recipe")
    for field, expected in (
        ("total_occurrences", B200_TOTAL_OCCURRENCES),
        ("nominal_step_occurrences", 16),
        ("full_steps", 9_315),
        ("full_step_occurrences", 149_040),
        ("final_step", B200_FINAL_STEP),
        ("final_step_occurrences", 8),
        ("padding_or_duplication_allowed", False),
    ):
        _expect(f"B200 exact tail {field}", tail.get(field), expected)

    state = _object(manifest.get("state_contract"), "B200 state contract")
    data_contract = read_json_no_duplicates(B200_CONFIG_DIR / "data_contract.json")
    current_pipeline = pipeline_code_identity(config_dir=B200_CONFIG_DIR)
    for field, expected in (
        ("format_version", 6),
        ("profile", "b200"),
        ("config_identity", config_identity),
        ("source_dir", str(BF16_SOURCE_DIR)),
        ("expert_nf4_dir", str(TRAIN_EXPERT_DIR)),
        ("model_contract_sha256", config_identity["files"]["model"]["sha256"]),
        ("recipe_sha256", config_identity["files"]["recipe"]["sha256"]),
        (
            "data_contract_sha256",
            config_identity["files"]["data_contract"]["sha256"],
        ),
        ("train_sha256", data_contract["files"]["train"]["sha256"]),
        ("validation_sha256", data_contract["files"]["validation"]["sha256"]),
        ("pipeline_code_sha256", current_pipeline["sha256"]),
        (
            "environment_lock_sha256",
            sha256(PROJECT / "requirements/solar-open-2.lock.txt"),
        ),
        ("smoke", False),
        ("wandb", B200_PRODUCTION_WANDB),
        (
            "training_performance_sha256",
            canonical_sha256(b200_training_performance_contract()),
        ),
        (
            "training_activation_contract_sha256",
            canonical_sha256(b200_selective_gradient_checkpointing_contract()),
        ),
    ):
        _expect(f"B200 state contract {field}", state.get(field), expected)
    receipt_values: dict[str, dict[str, Any]] = {}
    receipt_identities: dict[str, dict[str, Any]] = {}
    for receipt_name in (
        "production_smoke_receipt",
        "production_approval_receipt",
    ):
        identity = _object(state.get(receipt_name), f"B200 {receipt_name}")
        receipt_path = require_b200_owned_path(
            Path(str(identity.get("path") or "")), must_exist=True
        )
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise RuntimeError(f"B200 {receipt_name} is missing or symlinked")
        _expect(
            f"B200 {receipt_name} SHA-256",
            identity.get("sha256"),
            sha256(receipt_path),
        )
        receipt_values[receipt_name] = read_json_no_duplicates(receipt_path)
        receipt_identities[receipt_name] = identity
    smoke_value = receipt_values["production_smoke_receipt"]
    for field, expected in (
        ("schema", "solar_open2_b200_smoke_certification"),
        ("schema_version", 2),
        ("status", "PASS"),
        ("technical_production_gate_passed", True),
        ("authorization_scope", "technical_smoke_gate_only"),
        ("human_production_approval_required", True),
        ("human_production_approval_recorded", False),
    ):
        _expect(f"B200 technical smoke {field}", smoke_value.get(field), expected)
    _expect(
        "B200 technical smoke training performance",
        smoke_value.get("training_performance"),
        b200_training_performance_contract(),
    )
    _expect(
        "B200 technical smoke activation contract",
        smoke_value.get("gradient_checkpointing_contract"),
        b200_selective_gradient_checkpointing_contract(),
    )
    smoke_config = smoke_value.get("config_identity", smoke_value.get("config"))
    _expect("B200 technical smoke config", smoke_config, config_identity)
    approval_value = receipt_values["production_approval_receipt"]
    expected_approval = {
        "schema": "solar-open2-b200-production-approval",
        "schema_version": 2,
        "status": "PASS",
        "authorization_scope": "b200_production_training",
        "human_production_approval_recorded": True,
        "technical_smoke_receipt": receipt_identities["production_smoke_receipt"],
        "config_identity": config_identity,
        "pipeline_code_sha256": current_pipeline["sha256"],
        "model_contract_sha256": config_identity["files"]["model"]["sha256"],
        "recipe_sha256": config_identity["files"]["recipe"]["sha256"],
        "data_contract_sha256": config_identity["files"]["data_contract"]["sha256"],
        "training_performance": b200_training_performance_contract(),
        "training_performance_sha256": canonical_sha256(
            b200_training_performance_contract()
        ),
        "gradient_checkpointing_contract_sha256": canonical_sha256(
            b200_selective_gradient_checkpointing_contract()
        ),
        "output_dir": str(manifest.get("output_dir") or ""),
        "wandb": B200_PRODUCTION_WANDB,
        "runtime_override": b200_runtime_override_evidence(),
    }
    _expect("B200 production approval receipt", approval_value, expected_approval)

    gpu_runtime = _object(state.get("gpu_runtime"), "B200 GPU runtime receipt")
    _expect("B200 logical GPU count", gpu_runtime.get("logical_gpu_count"), 2)
    _expect("B200 GPU boundary", gpu_runtime.get("boundary_layers"), [24])
    logical = gpu_runtime.get("logical_to_physical")
    if (
        not isinstance(logical, list)
        or {
            str(value.get("logical_device"))
            for value in logical
            if isinstance(value, dict)
        }
        != B200_LOGICAL_DEVICES
    ):
        raise RuntimeError("B200 GPU runtime does not bind exactly cuda:0/cuda:1")

    device = _object(manifest.get("device_audit"), "B200 device audit")
    _expect(
        "B200 hf_device_map device set",
        set(device.get("hf_device_map_devices") or []),
        B200_LOGICAL_DEVICES,
    )
    if device.get("cpu_disk_offload") is not False:
        raise RuntimeError("B200 device audit permits CPU/disk offload")
    parameter_devices = _object(
        device.get("parameter_devices"), "B200 parameter devices"
    )
    if set(parameter_devices) != B200_LOGICAL_DEVICES or any(
        not isinstance(count, int) or count <= 0 for count in parameter_devices.values()
    ):
        raise RuntimeError("B200 parameters are not exclusively on cuda:0/cuda:1")

    loader = _object(manifest.get("loader_audit"), "B200 loader audit")
    _expect("B200 loader status", loader.get("status"), "PASS")
    _expect("B200 loader config", loader.get("b200_config"), config_identity)
    hardware = _object(loader.get("hardware"), "B200 loader hardware")
    _expect("B200 loader GPU count", hardware.get("logical_gpu_count"), 2)
    _expect("B200 loader boundary", hardware.get("boundary_layers"), [24])
    _expect(
        "B200 loader physical visibility",
        hardware.get("physical_cuda_visible_devices"),
        "6,7",
    )
    bnb_runtime = _object(
        loader.get("bitsandbytes_runtime"), "B200 bitsandbytes runtime"
    )
    for field, expected in (
        ("status", "PASS"),
        ("compute_capability", [10, 0]),
        ("manifest_status", "SEALED"),
        ("bitsandbytes_version", "0.49.2"),
        ("native_library", "libbitsandbytes_cuda128.so"),
    ):
        _expect(f"B200 bitsandbytes runtime {field}", bnb_runtime.get(field), expected)
    source_inventory = _object(loader.get("source_inventory"), "B200 source inventory")
    if Path(str(source_inventory.get("source_dir") or "")).resolve() != BF16_SOURCE_DIR:
        raise RuntimeError("B200 loader source inventory path changed")
    expert_preflight = _object(loader.get("expert_preflight"), "B200 expert preflight")
    if (
        Path(str(expert_preflight.get("checkpoint_dir") or "")).resolve()
        != TRAIN_EXPERT_DIR
        or expert_preflight.get("status") != "PASS"
        or expert_preflight.get("production_eligible") is not True
    ):
        raise RuntimeError("B200 loader expert preflight changed")
    model_audit = _object(loader.get("model_audit"), "B200 loader model audit")
    _expect(
        "B200 loader device map",
        model_audit.get("device_map"),
        dict(EXPECTED_HF_DEVICE_MAP),
    )
    placement = _object(
        model_audit.get("placement_audit"), "B200 loader placement audit"
    )
    for field in (
        "cpu_tensor_count",
        "meta_tensor_count",
        "disk_offload_count",
        "forbidden_cuda2_tensor_count",
    ):
        _expect(f"B200 placement {field}", placement.get(field), 0)
    _expect(
        "B200 loader full checkpoint contract",
        loader.get("gradient_checkpointing_contract"),
        b200_full_gradient_checkpointing_contract(),
    )
    _expect(
        "B200 training performance",
        manifest.get("training_performance"),
        b200_training_performance_contract(),
    )
    _expect(
        "B200 training performance SHA",
        manifest.get("training_performance_sha256"),
        canonical_sha256(b200_training_performance_contract()),
    )
    _expect(
        "B200 effective selective checkpoint contract",
        manifest.get("gradient_checkpointing_contract"),
        b200_selective_gradient_checkpointing_contract(),
    )
    _expect(
        "B200 effective selective checkpoint contract SHA",
        manifest.get("gradient_checkpointing_contract_sha256"),
        canonical_sha256(b200_selective_gradient_checkpointing_contract()),
    )

    early = _object(manifest.get("early_stopping_decision"), "B200 coverage decision")
    for field, expected in (
        ("profile", "b200"),
        ("status", "MAXIMUM_REACHED"),
        ("evaluated_step", B200_FINAL_STEP),
        ("triggered", False),
        ("reason", "exact_epoch_coverage_reached"),
        ("quality_early_stopping_enabled", False),
        ("completed_candidate_steps", [4_658, B200_FINAL_STEP]),
        ("completed_occurrences", B200_TOTAL_OCCURRENCES),
        ("completed_epochs", 1.0),
    ):
        _expect(f"B200 coverage decision {field}", early.get(field), expected)
    coverage = _object(manifest.get("training_coverage"), "B200 coverage")
    for field, expected in (
        ("total_occurrences", B200_TOTAL_OCCURRENCES),
        ("skipped_occurrences", 0),
        ("unconsumed_planned_occurrences", 0),
        ("unique_raw_rows", 142_713),
        (
            "task_occurrences",
            {"translation": 102_224, "summarization": 46_824},
        ),
    ):
        _expect(f"B200 training coverage {field}", coverage.get(field), expected)


def _verify_b200(
    training_manifest: Path,
    adapter_dir: Path,
    *,
    config_dir: Path,
    receipt: Path | None,
) -> dict[str, Any]:
    bundle = load_b200_config_bundle(config_dir)
    identity = b200_config_identity(config_dir)
    training_manifest = require_b200_owned_path(training_manifest, must_exist=True)
    adapter_dir = require_b200_owned_path(adapter_dir, must_exist=True)
    if training_manifest.is_symlink() or not training_manifest.is_file():
        raise RuntimeError("B200 training manifest is missing or symlinked")
    manifest = read_json_no_duplicates(training_manifest)
    output_dir = require_b200_owned_path(
        Path(str(manifest.get("output_dir") or "")), must_exist=True
    )
    _expect(
        "B200 training manifest path",
        training_manifest,
        output_dir / "run_manifest.json",
    )
    _expect("B200 source directory", manifest.get("source_dir"), str(BF16_SOURCE_DIR))
    _expect(
        "B200 training expert directory",
        manifest.get("expert_nf4_dir"),
        str(TRAIN_EXPERT_DIR),
    )
    validate_b200_manifest_contract(
        manifest,
        config_identity=identity,
        recipe=bundle["recipe"],
        model_contract=bundle["model"],
    )
    # Import locally so the standalone source verifier remains independent;
    # this binds B200 evaluation to the exact full-validation argmin.
    from finalize import validate_training_selection

    selection_gate = validate_training_selection(
        manifest,
        recipe=bundle["recipe"],
        config_dir=config_dir,
    )
    terminal = validate_terminal_training_success(
        output_dir,
        config_identity=identity,
        recipe=bundle["recipe"],
    )
    selected = _object(manifest.get("selected_checkpoint"), "B200 selected checkpoint")
    _expect(
        "B200 evaluation adapter path",
        adapter_dir,
        Path(str(selected.get("adapter_dir") or "")).resolve(strict=True),
    )
    adapter = validate_adapter(adapter_dir, config_dir=config_dir)
    recorded_files = selected.get("files_sha256")
    _expect(
        "B200 selected adapter files", current_file_hashes(adapter_dir), recorded_files
    )
    result = {
        "status": "PASS",
        "profile": "b200",
        "b200_config": identity,
        "training_manifest": {
            "path": str(training_manifest),
            "sha256": sha256(training_manifest),
        },
        "terminal_training": terminal,
        "training_status": "COMPLETED",
        "completed_updates": B200_FINAL_STEP,
        "target_updates": B200_FINAL_STEP,
        "training_completion": manifest["training_completion"],
        "early_stopping_decision": manifest["early_stopping_decision"],
        "training_coverage": manifest["training_coverage"],
        "selection_record": selection_gate["selection_record"],
        "selected_checkpoint": selected,
        "selected_step": selection_gate["selected_step"],
        "selected_selection_loss": selection_gate["selected_selection_loss"],
        "adapter_dir": str(adapter_dir),
        "adapter_files_sha256": recorded_files,
        "adapter_inference_identity": adapter,
    }
    if receipt is not None:
        receipt = require_b200_owned_path(receipt)
        atomic_json(receipt, result)
    return result


def verify(
    training_manifest: Path,
    adapter_dir: Path,
    *,
    receipt: Path | None = None,
    config_dir: Path = CONFIG_DIR,
) -> dict[str, Any]:
    resolved_config = config_dir.resolve(strict=True)
    if resolved_config == B200_CONFIG_DIR:
        return _verify_b200(
            training_manifest,
            adapter_dir,
            config_dir=resolved_config,
            receipt=receipt,
        )
    if resolved_config != CONFIG_DIR.resolve(strict=True):
        raise RuntimeError(f"unsupported config directory: {resolved_config}")
    training_manifest = training_manifest.resolve()
    adapter_dir = adapter_dir.resolve()
    _regular_file(training_manifest, "training manifest")
    manifest = read_json_no_duplicates(training_manifest)
    recipe = read_json_no_duplicates(CONFIG_DIR / "recipe.json")
    model_contract = read_json_no_duplicates(CONFIG_DIR / "model.json")
    data_contract = read_json_no_duplicates(CONFIG_DIR / "data_contract.json")
    pipeline = pipeline_code_identity()

    expected_updates = int(recipe["optimizer_steps"])
    for key, expected in (
        ("model_name", model_contract["derivative_model_name"]),
        ("attribution", model_contract["attribution"]),
        ("target_updates", expected_updates),
        ("config", recipe),
        ("output_only_contract", recipe["output_only_contract"]),
        ("wandb", EXPECTED_PRODUCTION_WANDB),
        ("split_roles", EXPECTED_SPLIT_ROLES),
    ):
        _expect(f"training manifest {key}", manifest.get(key), expected)
    slurm_job_id = manifest.get("slurm_job_id")
    if not isinstance(slurm_job_id, str) or not slurm_job_id.isdigit():
        raise RuntimeError(f"invalid production Slurm job ID: {slurm_job_id!r}")
    _finite(
        manifest.get("elapsed_seconds"),
        "production training elapsed seconds",
        positive=True,
    )
    if "smoke" in manifest:
        raise RuntimeError("production training manifest contains smoke evidence")
    backward_passes = _integer(
        manifest.get("backward_passes"),
        "production training backward passes",
        minimum=1,
    )
    state = _object(manifest.get("state_contract"), "training state contract")
    _expect(
        "top-level/state W&B identity",
        manifest["wandb"],
        state.get("wandb"),
    )
    for key in (
        "microbatch_phase_seconds",
        "optimizer_boundary_seconds",
        "runtime_budget_projection",
        "post_update_recompute_audit",
    ):
        if manifest.get(key) is not None:
            raise RuntimeError(f"production training {key} must be null")

    output_dir = Path(str(manifest.get("output_dir") or "")).resolve()
    _expect(
        "training manifest path",
        training_manifest,
        output_dir / "run_manifest.json",
    )
    output_success = _regular_file(output_dir / "_SUCCESS", "training success marker")
    _expect(
        "training success marker",
        output_success.read_text(encoding="utf-8").strip(),
        "PASS",
    )

    source_dir = Path(str(manifest.get("source_dir") or "")).resolve()
    expert_dir = Path(str(manifest.get("expert_nf4_dir") or "")).resolve()
    source = lightweight_source_identity(source_dir)
    _expect(
        "official Solar source index SHA",
        source["index_sha256"],
        EXPECTED_SOURCE_INDEX_SHA256,
    )
    expert = _training_expert_identity(expert_dir, source=source)
    _validate_state_contract(
        manifest,
        source_dir=source_dir,
        expert_dir=expert_dir,
        expert=expert,
        data_contract=data_contract,
        pipeline=pipeline,
    )
    data_evidence = current_data_evidence(data_contract)
    _expect(
        "training sampler",
        manifest.get("sampler"),
        _expected_sampler(data_evidence, recipe),
    )
    training_completion, training_coverage, horizon = _validate_training_completion(
        manifest,
        recipe=recipe,
        data_evidence=data_evidence,
    )
    _expect(
        "production occurrence audit count",
        backward_passes,
        horizon["consumed_occurrences"],
    )
    token_length_contract = _validate_token_length_contract(
        manifest,
        recipe=recipe,
        data_evidence=data_evidence,
    )
    history = _validate_validation_history(
        manifest.get("history"),
        recipe=recipe,
        data_evidence=data_evidence,
        completed_updates=horizon["completed_updates"],
    )
    early_stopping = _validate_early_stopping_evidence(
        manifest,
        recipe=recipe,
        history=history,
        horizon=horizon,
    )
    _validate_loader_evidence(
        manifest,
        source_dir=source_dir,
        expert_dir=expert_dir,
        source=source,
        expert=expert,
    )
    _validate_optimization_evidence(
        manifest,
        recipe=recipe,
        model_contract=model_contract,
    )
    selection = _validate_checkpoint_selection(
        manifest,
        recipe=recipe,
        history=history,
        completed_updates=horizon["completed_updates"],
        output_dir=output_dir,
        evaluation_adapter_dir=adapter_dir,
    )
    result = {
        "status": "PASS",
        "training_manifest": {
            "path": str(training_manifest),
            "sha256": sha256(training_manifest),
        },
        "production_state_contract": manifest["state_contract"],
        "wandb": manifest["wandb"],
        "training_status": horizon["status"],
        "completed_updates": horizon["completed_updates"],
        "target_updates": horizon["target_updates"],
        "backward_passes": backward_passes,
        "training_completion": training_completion,
        "early_stopping_decision": early_stopping,
        "training_coverage": training_coverage,
        "token_length_contract": token_length_contract,
        "validation_schedule": [
            {"step": item["step"], "scope": item["scope"]} for item in history
        ],
        "data": data_evidence,
        "sampler": manifest["sampler"],
        "selection_policy": recipe["validation"]["selection_policy"],
        "selection_tie_break": recipe["validation"]["selection_tie_break"],
        **selection,
        "adapter_dir": selection["best_adapter_dir"],
        "adapter_files_sha256": selection["best_adapter_files_sha256"],
        "adapter_inference_identity": selection["best_adapter_inference_identity"],
    }
    if receipt is not None:
        atomic_json(receipt, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    config_dir = args.config_dir.resolve(strict=True)
    if config_dir == B200_CONFIG_DIR:
        require_expected_user()
        result = verify(
            args.training_manifest,
            args.adapter_dir,
            receipt=args.receipt,
            config_dir=config_dir,
        )
    elif config_dir == CONFIG_DIR.resolve(strict=True):
        result = verify(
            require_owned_path(args.training_manifest),
            require_owned_path(args.adapter_dir),
            receipt=require_owned_path(args.receipt, create_parent=True),
            config_dir=config_dir,
        )
    else:
        raise RuntimeError(f"unsupported config directory: {config_dir}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
