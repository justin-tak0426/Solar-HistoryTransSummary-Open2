#!/usr/bin/env python3
"""Run provenance-bound Solar base or LoRA inference without vLLM."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import signal
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    CONFIG_DIR,
    PROJECT,
    REFERENCE_SHA256,
    SOLAR_OUTPUT_ONLY_GENERATION_PREAMBLE,
    atomic_json,
    audit_lora_modules,
    canonical_sha256,
    expected_lora_module_names,
    load_reference_module,
    output_only_violations,
    pipeline_code_identity,
    read_json,
    read_jsonl,
    require_owned_path,
    sha256,
)
from b200_runtime import (
    B200_CONFIG_DIR,
    BF16_SOURCE_DIR,
    EXPECTED_HF_DEVICE_MAP,
    EXPECTED_LAYER_DEVICE_MAP,
    EXPECTED_LOGICAL_GPU_COUNT,
    INFERENCE_EXPERT_DIR,
    INHERITED_LARGE_HASH_MODE,
    VENV_DIR,
    b200_config_identity,
    load_b200_config_bundle,
    require_b200_gpu_runtime,
    require_expected_user,
    require_owned_path as require_b200_owned_path,
    require_owned_root,
    validate_b200_large_asset_receipt,
)
from quantization.contract import (
    BASE_MODEL_ID,
    BASE_REVISION,
    EXPERT_INDEX_FILENAME,
    MANIFEST_FILENAME,
    SCHEMA_NAME,
    SCHEMA_VERSION,
)
from quantization.output_checkpoint import read_json_no_duplicates
from solar_peft_compat import solar_open2_peft_v5_native


PROVENANCE_SCHEMA = "solar-open2-inference-jsonl"
PROVENANCE_SCHEMA_VERSION = 2
REQUEUE_EXIT_CODE = 75
TEMPLATE_KWARGS = {
    "reasoning_effort": "none",
    "think_render_option": "preserved",
    "provider_system_prompt": False,
}
SOURCE_IDENTITY_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
)
PINNED_COMBINED_EVAL_SHA256 = (
    "6ee48a2ad2a884ca3f661ee30624052110d7f46bbb053fc7ce35d6243a3e1eec"
)
PINNED_COMBINED_EVAL_ROWS = 19_778
PINNED_COMBINED_EVAL_TASK_COUNTS = {
    "translation": 14_012,
    "summarization": 5_766,
}
B200_VALIDATION_SMOKE_MODE = "b200-validation-smoke"
B200_PRODUCTION_EVALUATION_MODE = "production-evaluation"
B200_INFERENCE_MODES = (
    B200_VALIDATION_SMOKE_MODE,
    B200_PRODUCTION_EVALUATION_MODE,
)
B200_MAX_MODEL_LEN = 16_384
B200_TASK_MAX_NEW_TOKENS = {
    "translation": 8_192,
    "summarization": 6_144,
}
B200_TENSOR_PARALLEL_SIZE = 2
B200_TASK_ORDER = ("translation", "summarization")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("id") or ""), str(row.get("task") or "")


def provenance_path(output_jsonl: Path) -> Path:
    return output_jsonl.with_name(output_jsonl.name + ".manifest.json")


def select_rows(
    rows: list[dict[str, Any]],
    *,
    limit_per_task: int,
) -> list[dict[str, Any]]:
    if limit_per_task <= 0:
        return rows
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row in rows:
        task = str(row["task"])
        if counts[task] < limit_per_task:
            selected.append(row)
            counts[task] += 1
    return selected


def _b200_dataset_path(
    config_bundle: dict[str, Any],
    role: str,
) -> Path:
    data_contract = config_bundle["data_contract"]
    entry = data_contract["files"][role]
    configured = Path(str(entry.get("path") or ""))
    if not configured.is_absolute():
        raise RuntimeError(f"B200 {role} path is not absolute: {configured}")
    resolved = require_b200_owned_path(configured, must_exist=True)
    expected = (Path(data_contract["prepared_data_dir"]) / str(entry["name"])).resolve(
        strict=True
    )
    if resolved != expected:
        raise RuntimeError(
            f"B200 {role} path differs from prepared-data contract: "
            f"{resolved} != {expected}"
        )
    if configured.is_symlink() or not resolved.is_file():
        raise RuntimeError(f"B200 {role} must be a regular non-symlink file")
    return resolved


def _validate_b200_dataset(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    role: str,
    config_bundle: dict[str, Any],
) -> dict[str, Any]:
    expected_path = _b200_dataset_path(config_bundle, role)
    if path != expected_path:
        raise RuntimeError(f"B200 {role} path mismatch: {path} != {expected_path}")
    entry = config_bundle["data_contract"]["files"][role]
    actual_sha256 = sha256(path)
    if actual_sha256 != entry.get("sha256"):
        raise RuntimeError(
            f"B200 {role} SHA-256 mismatch: {actual_sha256} "
            f"!= {entry.get('sha256')}"
        )
    if len(rows) != entry.get("rows"):
        raise RuntimeError(
            f"B200 {role} row count mismatch: {len(rows)} " f"!= {entry.get('rows')}"
        )
    counts: Counter[str] = Counter()
    keys: set[tuple[str, str]] = set()
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"B200 {role} row {position} is not an object")
        key = row_key(row)
        if not key[0] or key[1] not in B200_TASK_ORDER:
            raise RuntimeError(
                f"B200 {role} row {position} has invalid identity: {key}"
            )
        if key in keys:
            raise RuntimeError(f"B200 {role} has duplicate id/task: {key}")
        if not isinstance(row.get("origin"), str) or not isinstance(
            row.get("response"), str
        ):
            raise RuntimeError(
                f"B200 {role} row {position} lacks string origin/response"
            )
        keys.add(key)
        counts[key[1]] += 1
    if dict(counts) != entry.get("by_task"):
        raise RuntimeError(
            f"B200 {role} task counts mismatch: {dict(counts)} "
            f"!= {entry.get('by_task')}"
        )
    return {
        "role": role,
        "path": str(path),
        "sha256": actual_sha256,
        "rows": len(rows),
        "by_task": dict(counts),
        "row_order_sha256": canonical_sha256(
            [[row["id"], row["task"]] for row in rows]
        ),
    }


def reject_b200_smoke_sealed_path(
    eval_path: Path,
    config_bundle: dict[str, Any],
) -> None:
    """Fail before reading any sealed evaluation payload in smoke mode."""

    sealed_roles = ("full_test", "human_gold", "combined_eval")
    sealed_paths = {
        role: _b200_dataset_path(config_bundle, role) for role in sealed_roles
    }
    matches = [role for role, path in sealed_paths.items() if eval_path == path]
    if matches:
        raise RuntimeError(
            "B200 validation smoke forbids sealed evaluation path: "
            f"role={matches[0]} path={eval_path}"
        )


def deterministic_validation_smoke_rows(
    rows: list[dict[str, Any]],
    reference: Any,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Select exactly the first file-order row for each task."""

    selected: list[dict[str, Any]] = []
    positions: list[int] = []
    seen: set[str] = set()
    for position, raw_row in enumerate(rows):
        task = str(raw_row.get("task") or "")
        if task not in B200_TASK_ORDER or task in seen:
            continue
        if raw_row.get("source_split") != "validation":
            raise RuntimeError(
                f"validation smoke row has wrong source_split: {row_key(raw_row)}"
            )
        row = dict(raw_row)
        messages = reference.build_messages_no_dict(
            task,
            reference.clean_text(row["origin"]),
        )
        existing_messages = row.get("messages")
        if existing_messages is not None and existing_messages != messages:
            raise RuntimeError(
                f"validation smoke row has noncanonical messages: {row_key(row)}"
            )
        row["messages"] = messages
        if row.get("prompt_variant") != "no_dictionary_context":
            raise RuntimeError(
                f"validation smoke prompt variant changed: {row_key(row)}"
            )
        selected.append(row)
        positions.append(position)
        seen.add(task)
        if seen == set(B200_TASK_ORDER):
            break
    if [row["task"] for row in selected] != list(B200_TASK_ORDER):
        raise RuntimeError(
            "B200 validation smoke requires exactly one translation and "
            "one summarization row"
        )
    reference.validate_eval_rows(selected, "B200 deterministic validation smoke")
    return selected, positions


def _evaluation_row_identity(
    row: dict[str, Any],
    *,
    split: str,
    position: int,
    dataset_sha256: str,
) -> dict[str, Any]:
    return {
        "split": split,
        "position": position,
        "id": str(row["id"]),
        "task": str(row["task"]),
        "dataset_sha256": dataset_sha256,
        "row_sha256": canonical_sha256(row),
    }


def prepare_b200_evaluation(
    *,
    mode: str,
    eval_path: Path,
    config_bundle: dict[str, Any],
    reference: Any,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[tuple[str, str], dict[str, Any]],
]:
    """Load either the two-row validation smoke or the sealed full union."""

    if mode not in B200_INFERENCE_MODES:
        raise RuntimeError(f"unsupported B200 inference mode: {mode!r}")
    if mode == B200_VALIDATION_SMOKE_MODE:
        reject_b200_smoke_sealed_path(eval_path, config_bundle)
        validation_path = _b200_dataset_path(config_bundle, "validation")
        if eval_path != validation_path:
            raise RuntimeError(
                "B200 validation smoke accepts only the pinned validation file: "
                f"{eval_path} != {validation_path}"
            )
        all_rows = read_jsonl(eval_path)
        dataset = _validate_b200_dataset(
            eval_path,
            all_rows,
            role="validation",
            config_bundle=config_bundle,
        )
        rows, positions = deterministic_validation_smoke_rows(
            all_rows,
            reference,
        )
        identities = {
            row_key(row): _evaluation_row_identity(
                all_rows[position],
                split="validation",
                position=position,
                dataset_sha256=dataset["sha256"],
            )
            for row, position in zip(rows, positions, strict=True)
        }
        selected_keys = [list(row_key(row)) for row in rows]
        return (
            rows,
            {
                **dataset,
                "mode": mode,
                "selection_policy": "first_file_order_row_per_task",
                "selected_rows": 2,
                "selected_by_task": {task: 1 for task in B200_TASK_ORDER},
                "selected_keys_sha256": canonical_sha256(selected_keys),
                "selected_row_identities": [identities[row_key(row)] for row in rows],
                "sealed_evaluation_accessed": False,
            },
            identities,
        )

    combined_path = _b200_dataset_path(config_bundle, "combined_eval")
    if eval_path != combined_path:
        raise RuntimeError(
            "production evaluation accepts only the pinned combined union: "
            f"{eval_path} != {combined_path}"
        )
    rows = read_jsonl(eval_path)
    combined = _validate_b200_dataset(
        eval_path,
        rows,
        role="combined_eval",
        config_bundle=config_bundle,
    )
    reference.validate_eval_rows(rows, str(eval_path))
    components: dict[str, dict[str, Any]] = {}
    component_rows: dict[str, list[dict[str, Any]]] = {}
    for role in ("full_test", "human_gold"):
        path = _b200_dataset_path(config_bundle, role)
        values = read_jsonl(path)
        reference.validate_eval_rows(values, str(path))
        components[role] = _validate_b200_dataset(
            path,
            values,
            role=role,
            config_bundle=config_bundle,
        )
        component_rows[role] = values
    full_keys = {row_key(row) for row in component_rows["full_test"]}
    gold_keys = {row_key(row) for row in component_rows["human_gold"]}
    if full_keys & gold_keys:
        raise RuntimeError("sealed full_test and human_gold identities overlap")
    expected_union = [
        *component_rows["full_test"],
        *component_rows["human_gold"],
    ]
    if rows != expected_union:
        raise RuntimeError(
            "combined evaluation is not the exact ordered full_test+human_gold union"
        )
    identities: dict[tuple[str, str], dict[str, Any]] = {}
    for split in ("full_test", "human_gold"):
        dataset_sha256 = components[split]["sha256"]
        for position, row in enumerate(component_rows[split]):
            identities[row_key(row)] = _evaluation_row_identity(
                row,
                split=split,
                position=position,
                dataset_sha256=dataset_sha256,
            )
    selected_keys = [list(row_key(row)) for row in rows]
    return (
        rows,
        {
            **combined,
            "role": "full_test_plus_human_gold_sealed_union",
            "mode": mode,
            "components": components,
            "ordered_union": ["full_test", "human_gold"],
            "selected_rows": len(rows),
            "selected_keys_sha256": canonical_sha256(selected_keys),
            "sealed_evaluation_accessed": True,
        },
        identities,
    )


def validate_pinned_combined_eval(
    path: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Permit only the sealed full-test plus human-gold evaluation union."""

    data_contract = read_json(CONFIG_DIR / "data_contract.json")
    entry = data_contract["files"]["combined_eval"]
    expected_path = (Path(data_contract["prepared_data_dir"]) / entry["name"]).resolve()
    if path != expected_path:
        raise RuntimeError(
            f"eval-jsonl is not the pinned combined evaluation: "
            f"{path} != {expected_path}"
        )
    if entry.get("sha256") != PINNED_COMBINED_EVAL_SHA256:
        raise RuntimeError("data contract combined-eval SHA changed")
    if entry.get("rows") != PINNED_COMBINED_EVAL_ROWS:
        raise RuntimeError("data contract combined-eval row count changed")
    if entry.get("by_task") != PINNED_COMBINED_EVAL_TASK_COUNTS:
        raise RuntimeError("data contract combined-eval task counts changed")
    actual_sha256 = sha256(path)
    if actual_sha256 != PINNED_COMBINED_EVAL_SHA256:
        raise RuntimeError(
            f"combined evaluation changed: {actual_sha256} "
            f"!= {PINNED_COMBINED_EVAL_SHA256}"
        )
    counts = Counter(str(row.get("task") or "") for row in rows)
    if len(rows) != PINNED_COMBINED_EVAL_ROWS:
        raise RuntimeError(
            f"combined evaluation row count changed: {len(rows)} "
            f"!= {PINNED_COMBINED_EVAL_ROWS}"
        )
    if dict(counts) != PINNED_COMBINED_EVAL_TASK_COUNTS:
        raise RuntimeError(f"combined evaluation task counts changed: {dict(counts)}")
    return {
        "role": "full_test_plus_human_gold_sealed_union",
        "path": str(path),
        "sha256": actual_sha256,
        "rows": len(rows),
        "by_task": dict(counts),
    }


def validate_terminal_training_success(
    training_dir: Path,
    *,
    config_identity: dict[str, Any],
    recipe: dict[str, Any],
) -> dict[str, Any]:
    """Require the exact completed production run before sealed evaluation."""

    training_dir = require_b200_owned_path(training_dir, must_exist=True)
    if not training_dir.is_dir():
        raise RuntimeError(f"training terminal path is not a directory: {training_dir}")
    success_path = training_dir / "_SUCCESS"
    manifest_path = training_dir / "run_manifest.json"
    for label, path in (
        ("success marker", success_path),
        ("run manifest", manifest_path),
    ):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"training {label} is missing or symlinked: {path}")
    if success_path.read_text(encoding="utf-8") != "PASS\n":
        raise RuntimeError("training success marker is not exact PASS")
    manifest = read_json_no_duplicates(manifest_path)
    expected_updates = int(recipe["optimizer_steps"])
    expected_occurrences = int(recipe["training_deck"]["total_occurrences"]["total"])
    exact = {
        "status": "COMPLETED",
        "profile": "b200",
        "config_identity": config_identity,
        "completed_updates": expected_updates,
        "target_updates": expected_updates,
        "consumed_occurrences": expected_occurrences,
    }
    for field, expected in exact.items():
        if manifest.get(field) != expected:
            raise RuntimeError(
                f"terminal training manifest {field} changed: "
                f"{manifest.get(field)!r} != {expected!r}"
            )
    state_contract = manifest.get("state_contract")
    if not isinstance(state_contract, dict):
        raise RuntimeError("terminal training manifest lacks state_contract")
    if state_contract.get("profile") != "b200":
        raise RuntimeError("terminal training state profile is not B200")
    if state_contract.get("smoke") is not False:
        raise RuntimeError("sealed evaluation cannot follow a smoke training run")
    if state_contract.get("config_identity") != config_identity:
        raise RuntimeError("terminal training state/config identity changed")
    completion = manifest.get("training_completion")
    expected_completion = {
        "status": "COMPLETED",
        "epochs": 1,
        "completed_epochs": 1.0,
        "early_stopping_triggered": False,
        "optimizer_steps": expected_updates,
        "consumed_occurrences": expected_occurrences,
        "expected_optimizer_steps": expected_updates,
        "expected_occurrences": expected_occurrences,
    }
    if completion != expected_completion:
        raise RuntimeError(f"terminal training completion changed: {completion!r}")
    coverage = manifest.get("training_coverage")
    if not isinstance(coverage, dict):
        raise RuntimeError("terminal training manifest lacks coverage")
    expected_task_occurrences = recipe["training_deck"]["total_occurrences"]
    for field, expected in (
        ("total_occurrences", expected_occurrences),
        ("skipped_occurrences", 0),
        ("unconsumed_planned_occurrences", 0),
        ("unique_raw_rows", recipe["training_deck"]["base_unique_rows"]),
        (
            "task_occurrences",
            {task: int(expected_task_occurrences[task]) for task in B200_TASK_ORDER},
        ),
    ):
        if coverage.get(field) != expected:
            raise RuntimeError(
                f"terminal training coverage {field} changed: "
                f"{coverage.get(field)!r} != {expected!r}"
            )
    if (manifest.get("wandb_reporting") or {}).get("completion_status") != "COMMITTED":
        raise RuntimeError("terminal training W&B completion is not committed")
    checkpoints: dict[str, Any] = {}
    for label, dirname in (
        ("final_checkpoint", recipe["checkpoint"]["final_adapter_dirname"]),
        ("selected_checkpoint", recipe["checkpoint"]["best_adapter_dirname"]),
    ):
        record = manifest.get(label)
        if not isinstance(record, dict):
            raise RuntimeError(f"terminal training manifest lacks {label}")
        adapter_dir = require_b200_owned_path(
            Path(str(record.get("adapter_dir") or "")),
            must_exist=True,
        )
        expected_dir = (training_dir / str(dirname)).resolve(strict=True)
        if adapter_dir != expected_dir:
            raise RuntimeError(
                f"terminal {label} path changed: {adapter_dir} != {expected_dir}"
            )
        marker = adapter_dir / "_SUCCESS"
        if marker.is_symlink() or not marker.is_file():
            raise RuntimeError(f"terminal {label} is incomplete: {adapter_dir}")
        observed_files = {
            child.name: sha256(child)
            for child in sorted(adapter_dir.iterdir())
            if child.is_file()
        }
        if observed_files != record.get("files_sha256"):
            raise RuntimeError(f"terminal {label} file hashes changed")
        checkpoints[label] = record
    selection = manifest.get("selection_record")
    selected = checkpoints["selected_checkpoint"]
    if not isinstance(selection, dict):
        raise RuntimeError("terminal training manifest lacks selection record")
    if selection.get("selected_step") != selected.get("step"):
        raise RuntimeError("selected checkpoint step differs from selection record")
    return {
        "status": "PASS",
        "training_dir": str(training_dir),
        "run_manifest": {
            "path": str(manifest_path),
            "sha256": sha256(manifest_path),
        },
        "success_marker": {
            "path": str(success_path),
            "sha256": sha256(success_path),
        },
        "config_identity": config_identity,
        "completed_updates": expected_updates,
        "consumed_occurrences": expected_occurrences,
        "selected_checkpoint": selected,
        "final_checkpoint": checkpoints["final_checkpoint"],
        "selection_record_sha256": canonical_sha256(selection),
    }


def validate_adapter(
    adapter_dir: Path,
    *,
    config_dir: Path = CONFIG_DIR,
) -> dict[str, Any]:
    success = adapter_dir / "_SUCCESS"
    config_path = adapter_dir / "adapter_config.json"
    if (
        success.is_symlink()
        or config_path.is_symlink()
        or not success.is_file()
        or success.read_text(encoding="utf-8").strip() != "PASS"
    ):
        raise RuntimeError(f"adapter is incomplete: {adapter_dir}")
    config = read_json_no_duplicates(config_path)
    recipe = read_json(config_dir / "recipe.json")
    lora = recipe["lora"]
    expected = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": lora["rank"],
        "lora_alpha": lora["alpha"],
        "lora_dropout": lora["dropout"],
        "bias": lora["bias"],
        "target_modules": lora["target_regex"],
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(
                f"adapter recipe mismatch: {key}={config.get(key)!r}, "
                f"expected={value!r}"
            )
    if config.get("modules_to_save") not in (None, []):
        raise RuntimeError("adapter unexpectedly saves full base-model modules")
    weight_files = sorted(adapter_dir.glob("adapter_model*.safetensors"))
    if len(weight_files) != 1 or any(path.is_symlink() for path in weight_files):
        raise RuntimeError(
            "adapter must contain exactly one regular adapter_model safetensors file"
        )
    weights = [
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in weight_files
    ]
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("safetensors is required for adapter validation") from error
    architecture = read_json(config_dir / "model.json")["architecture_contract"]
    target_names = expected_lora_module_names(architecture)
    expected_keys = {
        f"base_model.model.{name}.lora_{side}.weight"
        for name in target_names
        for side in ("A", "B")
    }
    with safe_open(weight_files[0], framework="pt", device="cpu") as handle:
        actual_keys = set(handle.keys())
        if actual_keys != expected_keys:
            raise RuntimeError(
                "adapter tensor coverage changed: "
                f"missing={sorted(expected_keys - actual_keys)[:8]} "
                f"unexpected={sorted(actual_keys - expected_keys)[:8]}"
            )
        serialized_dtypes = Counter(
            str(handle.get_slice(key).get_dtype()) for key in actual_keys
        )
    if dict(serialized_dtypes) != {"F32": len(expected_keys)}:
        raise RuntimeError(
            f"adapter serialized dtype changed: {dict(serialized_dtypes)}"
        )
    file_sha256 = {
        "adapter_config.json": sha256(config_path),
        **{record["name"]: record["sha256"] for record in weights},
    }
    return {
        "path": str(adapter_dir),
        "adapter_config_sha256": sha256(config_path),
        "adapter_weights": weights,
        "adapter_weights_sha256": canonical_sha256(weights),
        "adapter_file_sha256": file_sha256,
        "serialized_tensor_count": len(expected_keys),
        "serialized_keys_sha256": canonical_sha256(sorted(expected_keys)),
        "serialized_dtypes": dict(serialized_dtypes),
        "rank": int(config["r"]),
        "target_modules": config["target_modules"],
    }


def lightweight_source_identity(source_dir: Path) -> dict[str, Any]:
    if source_dir.name != BASE_REVISION:
        raise RuntimeError(f"source path is not the pinned revision: {source_dir.name}")
    config_path = source_dir / "config.json"
    index_path = source_dir / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise FileNotFoundError("official Solar config/index is incomplete")
    config = read_json_no_duplicates(config_path)
    if config.get("model_type") != "solar_open2":
        raise RuntimeError(f"wrong source model_type: {config.get('model_type')}")
    if "SolarOpen2ForCausalLM" not in (config.get("architectures") or []):
        raise RuntimeError("source config does not declare SolarOpen2ForCausalLM")
    resource_hashes = {
        name: sha256(source_dir / name)
        for name in SOURCE_IDENTITY_FILES
        if (source_dir / name).is_file()
    }
    if "tokenizer_config.json" not in resource_hashes or not {
        "tokenizer.json",
        "tokenizer.model",
    } & set(resource_hashes):
        raise RuntimeError("official Solar tokenizer resources are incomplete")
    return {
        "path": str(source_dir),
        "model_id": BASE_MODEL_ID,
        "revision": BASE_REVISION,
        "config_sha256": sha256(config_path),
        "index_sha256": sha256(index_path),
        "tokenizer_and_generation_files": resource_hashes,
    }


def lightweight_expert_identity(
    expert_dir: Path,
    *,
    source: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = expert_dir / MANIFEST_FILENAME
    index_path = expert_dir / EXPERT_INDEX_FILENAME
    if not manifest_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(f"per-expert NF4 sidecar is not sealed: {expert_dir}")
    manifest = read_json_no_duplicates(manifest_path)
    expected = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "SEALED",
        "complete_model_checkpoint": False,
        "base_model_id": BASE_MODEL_ID,
        "base_revision": BASE_REVISION,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"per-expert manifest mismatch: {key}={manifest.get(key)!r}, "
                f"expected={value!r}"
            )
    sealed_source = manifest.get("source")
    if not isinstance(sealed_source, dict):
        raise RuntimeError("per-expert manifest lacks source provenance")
    for key in ("index_sha256", "config_sha256"):
        if sealed_source.get(key) != source[key]:
            raise RuntimeError(
                f"per-expert/source mismatch: {key}="
                f"{sealed_source.get(key)!r}, expected={source[key]!r}"
            )
    index_sha256 = sha256(index_path)
    if manifest.get("expert_index_sha256") != index_sha256:
        raise RuntimeError("per-expert index changed after manifest sealing")
    return {
        "path": str(expert_dir),
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "manifest_sha256": sha256(manifest_path),
        "expert_index_sha256": index_sha256,
        "source_index_sha256": sealed_source["index_sha256"],
        "source_config_sha256": sealed_source["config_sha256"],
    }


def build_inference_contract(
    *,
    source_dir: Path,
    expert_dir: Path,
    adapter_dir: Path | None,
    model_label: str,
    eval_jsonl: Path,
    selected_rows: list[dict[str, Any]],
    max_model_len: int,
    translation_max_new_tokens: int,
    summarization_max_new_tokens: int,
    generation_batch_size: int,
    limit_per_task: int,
    config_dir: Path = CONFIG_DIR,
    config_identity: dict[str, Any] | None = None,
    evaluation_identity: dict[str, Any] | None = None,
    inference_mode: str | None = None,
    terminal_training: dict[str, Any] | None = None,
    b200_runtime_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source = lightweight_source_identity(source_dir)
    expert = lightweight_expert_identity(expert_dir, source=source)
    adapter = (
        validate_adapter(adapter_dir, config_dir=config_dir)
        if adapter_dir is not None
        else None
    )
    selected_keys = [list(row_key(row)) for row in selected_rows]
    if len(set(map(tuple, selected_keys))) != len(selected_keys):
        raise RuntimeError("selected evaluation rows contain duplicate id/task keys")
    decoding = {
        "temperature": 0.0,
        "top_p": 1.0,
        "do_sample": False,
        "num_beams": 1,
        "max_model_len": max_model_len,
        "translation_max_new_tokens": translation_max_new_tokens,
        "summarization_max_new_tokens": summarization_max_new_tokens,
        "padding_side": "left",
        "raw_output_policy": "outer strip only; no filtering or repair",
    }
    b200 = config_dir.resolve() == B200_CONFIG_DIR.resolve()
    if b200 and config_identity is None:
        config_identity = b200_config_identity(config_dir)
    if b200 and inference_mode not in B200_INFERENCE_MODES:
        raise RuntimeError("B200 inference contract requires an explicit mode")
    if b200 and evaluation_identity is None:
        raise RuntimeError("B200 inference contract lacks evaluation identity")
    if b200 and b200_runtime_identity is None:
        raise RuntimeError("B200 inference contract lacks runtime identity")
    if b200 and (
        max_model_len != B200_MAX_MODEL_LEN
        or translation_max_new_tokens != B200_TASK_MAX_NEW_TOKENS["translation"]
        or summarization_max_new_tokens != B200_TASK_MAX_NEW_TOKENS["summarization"]
        or limit_per_task != 0
    ):
        raise RuntimeError("B200 inference decoding/selection contract changed")
    if b200 and inference_mode == B200_VALIDATION_SMOKE_MODE:
        if terminal_training is not None or len(selected_rows) != 2:
            raise RuntimeError("B200 validation smoke contract is not isolated")
    if b200 and inference_mode == B200_PRODUCTION_EVALUATION_MODE:
        if terminal_training is None or len(selected_rows) != PINNED_COMBINED_EVAL_ROWS:
            raise RuntimeError(
                "B200 production evaluation lacks terminal/full coverage"
            )
    pipeline_code = pipeline_code_identity(config_dir=config_dir if b200 else None)
    config_files = (
        config_identity["files"]
        if b200 and config_identity is not None
        else {
            "model": {
                "path": str(config_dir / "model.json"),
                "sha256": sha256(config_dir / "model.json"),
            },
            "recipe": {
                "path": str(config_dir / "recipe.json"),
                "sha256": sha256(config_dir / "recipe.json"),
            },
            "data_contract": {
                "path": str(config_dir / "data_contract.json"),
                "sha256": sha256(config_dir / "data_contract.json"),
            },
        }
    )
    implementation = {
        "infer_py_sha256": sha256(Path(__file__)),
        "inference_loader_sha256": sha256(
            Path(__file__).with_name("solar_inference_loader.py")
        ),
        "reference_evaluator_sha256": REFERENCE_SHA256,
        "config_files": config_files,
        "model_contract_sha256": config_files["model"]["sha256"],
        "recipe_sha256": config_files["recipe"]["sha256"],
        "data_contract_sha256": config_files["data_contract"]["sha256"],
        "environment_lock_sha256": sha256(
            PROJECT / "requirements/solar-open-2.lock.txt"
        ),
        "pipeline_code_sha256": pipeline_code["sha256"],
        "transformers_version": importlib.metadata.version("transformers"),
        "peft_version": importlib.metadata.version("peft"),
        "bitsandbytes_version": importlib.metadata.version("bitsandbytes"),
    }
    evaluation = (
        {
            **evaluation_identity,
            "limit_per_task": limit_per_task,
        }
        if evaluation_identity is not None
        else {
            "role": "full_test_plus_human_gold_sealed_union",
            "path": str(eval_jsonl),
            "sha256": sha256(eval_jsonl),
            "pinned_total_rows": PINNED_COMBINED_EVAL_ROWS,
            "pinned_by_task": PINNED_COMBINED_EVAL_TASK_COUNTS,
            "selected_rows": len(selected_rows),
            "selected_keys_sha256": canonical_sha256(selected_keys),
            "limit_per_task": limit_per_task,
        }
    )
    result = {
        "schema": PROVENANCE_SCHEMA,
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "profile": "b200" if b200 else "source",
        "inference_mode": inference_mode,
        "model_variant": "tuned" if adapter is not None else "base",
        "model_label": model_label,
        "source": source,
        "per_expert_nf4": expert,
        "adapter": adapter,
        "evaluation": evaluation,
        "prompt": {
            "variant": "no_dictionary_context",
            **TEMPLATE_KWARGS,
        },
        "decoding": decoding,
        "decoding_sha256": canonical_sha256(decoding),
        "runtime": {
            "strategy": (
                "single_process_two_gpu_model_parallel"
                if b200
                else "single_process_three_gpu_model_parallel"
            ),
            "logical_gpu_count": 2 if b200 else 3,
            "layer_device_map": (
                {str(key): value for key, value in EXPECTED_LAYER_DEVICE_MAP.items()}
                if b200
                else None
            ),
            "hf_device_map": dict(EXPECTED_HF_DEVICE_MAP) if b200 else None,
            "b200_identity": b200_runtime_identity if b200 else None,
            "routed_expert_format": "per_expert_nf4",
            "generation_batch_size": generation_batch_size,
        },
        "implementation": implementation,
    }
    if b200:
        result["b200_config"] = config_identity
        result["terminal_training"] = terminal_training
    return result


def read_prediction_rows(path: Path) -> list[dict[str, Any]]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise RuntimeError(
                    f"duplicate JSON key {key!r} in prediction file: {path}"
                )
            value[key] = item
        return value

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line, object_pairs_hook=reject_duplicates)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"truncated/corrupt prediction JSONL at {path}:{line_number}"
                ) from error
            if not isinstance(value, dict):
                raise RuntimeError(
                    f"prediction row is not an object: {path}:{line_number}"
                )
            rows.append(value)
    return rows


def expected_row_provenance(
    contract: dict[str, Any],
    contract_sha256: str,
) -> dict[str, Any]:
    adapter = contract["adapter"]
    provenance = {
        "source": {
            "model_id": contract["source"]["model_id"],
            "revision": contract["source"]["revision"],
            "index_sha256": contract["source"]["index_sha256"],
            "config_sha256": contract["source"]["config_sha256"],
        },
        "per_expert_nf4": {
            "schema": contract["per_expert_nf4"]["schema"],
            "manifest_sha256": contract["per_expert_nf4"]["manifest_sha256"],
            "expert_index_sha256": contract["per_expert_nf4"]["expert_index_sha256"],
        },
        "adapter_file_sha256": (
            adapter["adapter_file_sha256"] if adapter is not None else None
        ),
        "eval_jsonl_sha256": contract["evaluation"]["sha256"],
        "model_label": contract["model_label"],
        "decoding": contract["decoding"],
        "profile": contract.get("profile"),
        "inference_mode": contract.get("inference_mode"),
        "b200_config": contract.get("b200_config"),
        "evaluation_contract_sha256": canonical_sha256(contract["evaluation"]),
        "terminal_training": contract.get("terminal_training"),
    }
    return {
        "model_label": contract["model_label"],
        "profile": contract.get("profile"),
        "inference_mode": contract.get("inference_mode"),
        "inference_contract_sha256": contract_sha256,
        "provenance": provenance,
        "eval_jsonl_sha256": contract["evaluation"]["sha256"],
        "evaluation_contract_sha256": canonical_sha256(contract["evaluation"]),
        "b200_config": contract.get("b200_config"),
        "terminal_training": contract.get("terminal_training"),
        "source_revision": contract["source"]["revision"],
        "source_index_sha256": contract["source"]["index_sha256"],
        "source_config_sha256": contract["source"]["config_sha256"],
        "expert_manifest_sha256": contract["per_expert_nf4"]["manifest_sha256"],
        "adapter_config_sha256": (
            adapter["adapter_config_sha256"] if adapter is not None else None
        ),
        "adapter_weights_sha256": (
            adapter["adapter_weights_sha256"] if adapter is not None else None
        ),
        "adapter_file_sha256": (
            adapter["adapter_file_sha256"] if adapter is not None else None
        ),
        "decoding": contract["decoding"],
        "decoding_contract_sha256": contract["decoding_sha256"],
    }


def validate_previous_rows(
    previous: list[dict[str, Any]],
    *,
    selected_rows: list[dict[str, Any]],
    contract: dict[str, Any],
    contract_sha256: str,
    evaluation_row_identities: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> set[tuple[str, str]]:
    selected = {row_key(row): row for row in selected_rows}
    invariants = expected_row_provenance(contract, contract_sha256)
    done: set[tuple[str, str]] = set()
    for line_number, row in enumerate(previous, 1):
        key = row_key(row)
        if key in done:
            raise RuntimeError(f"resume file contains duplicate key: {key}")
        if key not in selected:
            raise RuntimeError(f"resume file contains an unexpected key: {key}")
        wanted = selected[key]
        for field, value in invariants.items():
            if row.get(field) != value:
                raise RuntimeError(
                    f"resume provenance mismatch at row {line_number}: "
                    f"{field}={row.get(field)!r}, expected={value!r}"
                )
        for field, value in (
            ("origin", wanted["origin"]),
            ("reference", wanted["response"]),
            ("reference_source", wanted.get("reference_source")),
            ("prompt_variant", "no_dictionary_context"),
            ("reasoning_effort", "none"),
        ):
            if row.get(field) != value:
                raise RuntimeError(
                    f"resume evaluation mismatch at row {line_number}: {field}"
                )
        if row.get("prediction") is None:
            raise RuntimeError(f"resume row lacks a prediction: {key}")
        if contract.get("profile") == "b200":
            if (
                evaluation_row_identities is None
                or key not in evaluation_row_identities
            ):
                raise RuntimeError(
                    f"resume lacks expected B200 evaluation row identity: {key}"
                )
            identity = evaluation_row_identities[key]
            if row.get("evaluation_row_identity") != identity:
                raise RuntimeError(f"resume evaluation row identity changed: {key}")
            if row.get("evaluation_split") != identity["split"]:
                raise RuntimeError(f"resume evaluation split changed: {key}")
            maximum = contract["decoding"][f"{key[1]}_max_new_tokens"]
            validate_generation_record(
                row,
                task=key[1],
                max_new_tokens=int(maximum),
            )
        done.add(key)
    return done


def initialize_or_resume(
    *,
    output_jsonl: Path,
    contract: dict[str, Any],
    selected_rows: list[dict[str, Any]],
    resume: bool,
    evaluation_row_identities: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]], set[tuple[str, str]]]:
    manifest_path = provenance_path(output_jsonl)
    contract_sha256 = canonical_sha256(contract)
    if not resume and (output_jsonl.exists() or manifest_path.exists()):
        raise FileExistsError(
            f"output/provenance exists; pass --resume: {output_jsonl}"
        )
    if output_jsonl.exists() and not manifest_path.is_file():
        raise RuntimeError("prediction JSONL exists without its provenance manifest")

    if manifest_path.is_file():
        manifest = read_json_no_duplicates(manifest_path)
        if manifest.get("schema") != PROVENANCE_SCHEMA:
            raise RuntimeError("resume provenance schema changed")
        if manifest.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
            raise RuntimeError("resume provenance schema version changed")
        if manifest.get("contract_sha256") != contract_sha256:
            raise RuntimeError("resume inference contract digest differs")
        if manifest.get("contract") != contract:
            raise RuntimeError("resume inference contract content differs")
        if manifest.get("status") not in {"IN_PROGRESS", "COMPLETE"}:
            raise RuntimeError("resume provenance status is invalid")
    else:
        manifest = {
            "schema": PROVENANCE_SCHEMA,
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "status": "IN_PROGRESS",
            "created_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
            "contract_sha256": contract_sha256,
            "contract": contract,
            "expected_rows": len(selected_rows),
            "completed_rows": 0,
        }
        atomic_json(manifest_path, manifest)

    previous = read_prediction_rows(output_jsonl) if output_jsonl.is_file() else []
    done = validate_previous_rows(
        previous,
        selected_rows=selected_rows,
        contract=contract,
        contract_sha256=contract_sha256,
        evaluation_row_identities=evaluation_row_identities,
    )
    if manifest["status"] == "COMPLETE":
        if len(done) != len(selected_rows):
            raise RuntimeError("COMPLETE provenance has incomplete predictions")
        if manifest.get("output_jsonl_sha256") != sha256(output_jsonl):
            raise RuntimeError("completed prediction JSONL changed after sealing")
        runtime_audit = manifest.get("runtime_audit")
        if not isinstance(runtime_audit, dict):
            raise RuntimeError("COMPLETE provenance lacks a runtime audit")
        model_audit = runtime_audit.get("model_audit")
        if not isinstance(model_audit, dict) or model_audit.get("status") != "PASS":
            raise RuntimeError("COMPLETE provenance model audit did not pass")
        if (
            runtime_audit.get("hash_verification") is not True
            or runtime_audit.get("production_eligible") is not True
        ):
            raise RuntimeError("COMPLETE provenance lacks verified runtime hashes")
        verify_recorded_runtime_identity(runtime_audit, contract)
        adapter_audit = runtime_audit.get("adapter_audit")
        if contract.get("adapter") is None:
            if adapter_audit is not None:
                raise RuntimeError(
                    "COMPLETE base runtime unexpectedly has an adapter audit"
                )
        elif (
            not isinstance(adapter_audit, dict)
            or adapter_audit.get("status") != "PASS"
            or adapter_audit.get("module_count") != 528
            or adapter_audit.get("adapter_parameter_count") != 133_177_344
        ):
            raise RuntimeError("COMPLETE tuned runtime adapter audit changed")
    return manifest_path, manifest, previous, done


def update_progress_manifest(
    path: Path,
    manifest: dict[str, Any],
    *,
    completed_rows: int,
) -> None:
    manifest["status"] = "IN_PROGRESS"
    manifest["completed_rows"] = completed_rows
    manifest["updated_at_utc"] = utc_now()
    atomic_json(path, manifest)


def record_runtime_audit(
    path: Path,
    manifest: dict[str, Any],
    *,
    loader_receipt: dict[str, Any],
    adapter_audit: dict[str, Any] | None,
) -> None:
    model_audit = loader_receipt.get("model_audit")
    if not isinstance(model_audit, dict) or model_audit.get("status") != "PASS":
        raise RuntimeError("loader receipt lacks a passing model audit")
    if (
        loader_receipt.get("hash_verification") is not True
        or loader_receipt.get("production_eligible") is not True
    ):
        raise RuntimeError("runtime audit requires production hash verification")
    if adapter_audit is not None:
        if adapter_audit.get("status") != "PASS":
            raise RuntimeError("adapter runtime audit did not pass")
        if adapter_audit.get("module_count") != 528:
            raise RuntimeError("adapter runtime audit module count changed")
        if adapter_audit.get("adapter_parameter_count") != 133_177_344:
            raise RuntimeError("adapter runtime audit parameter count changed")
    contract = manifest.get("contract")
    if not isinstance(contract, dict):
        raise RuntimeError("prediction manifest lacks its inference contract")
    source_inventory = loader_receipt.get("source_inventory")
    expert_preflight = loader_receipt.get("expert_preflight")
    if contract.get("source", {}).get("path"):
        verify_loader_receipt(loader_receipt, contract)
    manifest["runtime_audit"] = {
        "loaded_at_utc": utc_now(),
        "loader_receipt": loader_receipt,
        "model_audit": model_audit,
        "adapter_audit": adapter_audit,
        "source_inventory": source_inventory,
        "expert_preflight": expert_preflight,
        "hash_verification": loader_receipt.get("hash_verification"),
        "hash_verification_mode": loader_receipt.get("hash_verification_mode"),
        "direct_large_shard_hashes": loader_receipt.get("direct_large_shard_hashes"),
        "payload_rehash_performed": loader_receipt.get("payload_rehash_performed"),
        "asset_integrity_evidence": loader_receipt.get("asset_integrity_evidence"),
        "production_eligible": loader_receipt.get("production_eligible"),
    }
    manifest["updated_at_utc"] = utc_now()
    atomic_json(path, manifest)


def seal_predictions(
    path: Path,
    manifest: dict[str, Any],
    *,
    output_jsonl: Path,
    completed_rows: int,
) -> None:
    if completed_rows != manifest.get("expected_rows"):
        raise RuntimeError(
            f"cannot seal incomplete predictions: {completed_rows} "
            f"!= {manifest.get('expected_rows')}"
        )
    runtime_audit = manifest.get("runtime_audit")
    if not isinstance(runtime_audit, dict):
        raise RuntimeError("cannot seal predictions without a runtime audit")
    model_audit = runtime_audit.get("model_audit")
    if not isinstance(model_audit, dict) or model_audit.get("status") != "PASS":
        raise RuntimeError("cannot seal predictions with a failed model audit")
    if (
        runtime_audit.get("hash_verification") is not True
        or runtime_audit.get("production_eligible") is not True
    ):
        raise RuntimeError("cannot seal predictions without verified runtime hashes")
    verify_recorded_runtime_identity(runtime_audit, manifest["contract"])
    adapter_expected = manifest["contract"].get("adapter") is not None
    adapter_audit = runtime_audit.get("adapter_audit")
    if adapter_expected:
        if (
            not isinstance(adapter_audit, dict)
            or adapter_audit.get("status") != "PASS"
            or adapter_audit.get("module_count") != 528
            or adapter_audit.get("adapter_parameter_count") != 133_177_344
        ):
            raise RuntimeError("cannot seal tuned predictions without adapter audit")
    elif adapter_audit is not None:
        raise RuntimeError("base prediction runtime unexpectedly has an adapter audit")
    manifest["status"] = "COMPLETE"
    manifest["completed_rows"] = completed_rows
    manifest["updated_at_utc"] = utc_now()
    manifest["completed_at_utc"] = utc_now()
    manifest["output_jsonl_sha256"] = sha256(output_jsonl)
    atomic_json(path, manifest)


def verify_loader_receipt(
    receipt: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    if receipt.get("status") != "PASS":
        raise RuntimeError("inference loader receipt status is not PASS")
    if contract.get("profile") == "b200":
        if receipt.get("b200_config") != contract.get("b200_config"):
            raise RuntimeError("loader B200 config identity differs from inference")
        hardware = receipt.get("hardware")
        if not isinstance(hardware, dict):
            raise RuntimeError("loader receipt lacks B200 hardware audit")
        if hardware.get("logical_gpu_count") != EXPECTED_LOGICAL_GPU_COUNT:
            raise RuntimeError("loader did not qualify exactly two logical GPUs")
        if hardware.get("boundary_layers") != [24]:
            raise RuntimeError("loader B200 boundary layers changed")
        model_audit = receipt.get("model_audit")
        if not isinstance(model_audit, dict):
            raise RuntimeError("loader receipt lacks model placement audit")
        if model_audit.get("device_map") != dict(EXPECTED_HF_DEVICE_MAP):
            raise RuntimeError("loader model device map differs from B200 24/24 map")
        for field in (
            "meta_parameter_count",
            "meta_buffer_count",
            "non_cuda_parameter_count",
            "non_cuda_buffer_count",
        ):
            if model_audit.get(field) != 0:
                raise RuntimeError(f"loader model audit {field} is not zero")
        if "cuda:2" in model_audit["device_map"].values():
            raise RuntimeError("loader receipt unexpectedly references cuda:2")
        bnb_runtime = receipt.get("bitsandbytes_runtime")
        if not isinstance(bnb_runtime, dict) or bnb_runtime.get("status") != "PASS":
            raise RuntimeError("loader sealed bitsandbytes runtime did not pass")
        for field, expected in (
            ("hash_verification", True),
            ("hash_verification_mode", INHERITED_LARGE_HASH_MODE),
            ("direct_large_shard_hashes", False),
            ("payload_rehash_performed", False),
            ("production_eligible", True),
        ):
            if receipt.get(field) != expected:
                raise RuntimeError(
                    f"B200 inference loader {field} mismatch: "
                    f"{receipt.get(field)!r} != {expected!r}"
                )
        evidence = receipt.get("asset_integrity_evidence")
        if not isinstance(evidence, dict) or evidence.get("role") != "inference":
            raise RuntimeError("B200 inference loader lacks inference asset evidence")
        identity = evidence.get("asset_integrity_receipt")
        if not isinstance(identity, dict) or not identity.get("path"):
            raise RuntimeError("B200 inference loader lacks asset receipt identity")
        current_evidence = validate_b200_large_asset_receipt(
            Path(str(identity["path"])),
            role="inference",
        )
        if evidence != current_evidence:
            raise RuntimeError("B200 inference asset evidence changed after loading")
    source = receipt["source_inventory"]
    expert = receipt["expert_preflight"]
    wanted = (
        (source["index_sha256"], contract["source"]["index_sha256"], "source index"),
        (source["config_sha256"], contract["source"]["config_sha256"], "source config"),
        (
            expert["manifest_sha256"],
            contract["per_expert_nf4"]["manifest_sha256"],
            "expert manifest",
        ),
        (
            expert["expert_index_sha256"],
            contract["per_expert_nf4"]["expert_index_sha256"],
            "expert index",
        ),
    )
    for actual, expected, label in wanted:
        if actual != expected:
            raise RuntimeError(
                f"loaded {label} identity differs: {actual} != {expected}"
            )
    expert_manifest_path = Path(contract["per_expert_nf4"]["path"]) / MANIFEST_FILENAME
    expert_manifest = read_json_no_duplicates(expert_manifest_path)
    sealed_source = expert_manifest.get("source")
    if not isinstance(sealed_source, dict):
        raise RuntimeError("sealed expert manifest lacks source payload identity")
    for field in (
        "index_sha256",
        "config_sha256",
        "weight_map_sha256",
        "shards",
    ):
        if source.get(field) != sealed_source.get(field):
            raise RuntimeError(
                f"loaded source payload differs from expert seal: {field}"
            )
    shards = source.get("shards")
    if (
        not isinstance(shards, list)
        or not shards
        or any(
            not isinstance(item, dict)
            or not isinstance(item.get("sha256"), str)
            or len(item["sha256"]) != 64
            for item in shards
        )
    ):
        raise RuntimeError("loaded source receipt lacks complete shard digests")
    if expert.get("status") != "PASS" or not expert.get("production_eligible"):
        raise RuntimeError("loaded expert preflight is not production eligible")
    if not receipt.get("production_eligible"):
        raise RuntimeError("inference loader did not perform production hash checks")


def verify_recorded_runtime_identity(
    runtime_audit: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    """Revalidate the payload identities persisted beside sealed predictions."""

    source_contract = contract.get("source")
    expert_contract = contract.get("per_expert_nf4")
    if not (
        isinstance(source_contract, dict)
        and source_contract.get("path")
        and isinstance(expert_contract, dict)
        and expert_contract.get("path")
    ):
        return
    full_receipt = runtime_audit.get("loader_receipt")
    if isinstance(full_receipt, dict):
        receipt = full_receipt
    else:
        if contract.get("profile") == "b200":
            raise RuntimeError("recorded B200 runtime lacks the full loader receipt")
        receipt = {
            "status": "PASS",
            "source_inventory": runtime_audit.get("source_inventory"),
            "expert_preflight": runtime_audit.get("expert_preflight"),
            "production_eligible": runtime_audit.get("production_eligible"),
        }
    if not isinstance(receipt["source_inventory"], dict) or not isinstance(
        receipt["expert_preflight"],
        dict,
    ):
        raise RuntimeError("recorded runtime payload identities are missing")
    verify_loader_receipt(receipt, contract)


def attach_adapter(
    model: Any,
    *,
    adapter_dir: Path,
    model_contract: dict[str, Any],
    recipe: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    import torch
    from peft import PeftModel

    device_map = dict(model.hf_device_map)
    with solar_open2_peft_v5_native():
        model = PeftModel.from_pretrained(
            model,
            adapter_dir,
            is_trainable=False,
            local_files_only=True,
            autocast_adapter_dtype=True,
            low_cpu_mem_usage=False,
        )
    model.hf_device_map = device_map
    model.config.use_cache = True
    model.eval()
    module_audit = audit_lora_modules(
        model,
        model_contract["architecture_contract"],
    )
    lora_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    }
    serialized_parameter_names = {
        name.replace(".default.", "."): parameter
        for name, parameter in lora_parameters.items()
    }
    adapter_weights = sorted(adapter_dir.glob("adapter_model*.safetensors"))
    if len(adapter_weights) != 1:
        raise RuntimeError("loaded adapter weight file count changed")
    from safetensors import safe_open

    with safe_open(adapter_weights[0], framework="pt", device="cpu") as handle:
        serialized_keys = set(handle.keys())
        if serialized_keys != set(serialized_parameter_names):
            raise RuntimeError(
                "loaded adapter tensor coverage differs from safetensors: "
                f"file_only={sorted(serialized_keys - set(serialized_parameter_names))[:8]} "
                f"model_only={sorted(set(serialized_parameter_names) - serialized_keys)[:8]}"
            )
        wrong_shapes = {
            key: (
                tuple(handle.get_slice(key).get_shape()),
                tuple(serialized_parameter_names[key].shape),
            )
            for key in serialized_keys
            if tuple(handle.get_slice(key).get_shape())
            != tuple(serialized_parameter_names[key].shape)
        }
    if wrong_shapes:
        raise RuntimeError(
            f"loaded adapter shapes differ: {list(wrong_shapes.items())[:8]}"
        )
    parameter_count = sum(parameter.numel() for parameter in lora_parameters.values())
    expected_count = recipe["lora"]["expected_trainable_parameters"]
    if parameter_count != expected_count:
        raise RuntimeError(
            f"loaded LoRA parameter count changed: {parameter_count} "
            f"!= {expected_count}"
        )
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("inference adapter left trainable parameters enabled")
    invalid_devices = {
        name: str(parameter.device)
        for name, parameter in lora_parameters.items()
        if parameter.device.type != "cuda" or parameter.is_meta
    }
    if invalid_devices:
        raise RuntimeError(
            f"adapter tensors are offloaded/meta: "
            f"{list(invalid_devices.items())[:8]}"
        )
    dtypes = Counter(str(parameter.dtype) for parameter in lora_parameters.values())
    if dict(dtypes) != {"torch.float32": len(lora_parameters)}:
        raise RuntimeError(f"inference adapter dtype changed: {dict(dtypes)}")
    return model, {
        "status": "PASS",
        "module_count": module_audit["module_count"],
        "module_audit": module_audit,
        "adapter_tensor_count": len(lora_parameters),
        "adapter_tensor_names_sha256": canonical_sha256(
            sorted(serialized_parameter_names)
        ),
        "adapter_parameter_count": parameter_count,
        "adapter_dtypes": dict(dtypes),
        "trainable_parameter_count": 0,
        "offloaded_adapter_tensor_count": 0,
    }


def render_prompt_ids(
    tokenizer: Any,
    reference: Any,
    row: dict[str, Any],
    *,
    max_prompt_tokens: int,
) -> list[int]:
    messages = reference.build_messages_no_dict(
        row["task"],
        reference.clean_text(row["origin"]),
    )
    if messages != row["messages"]:
        raise RuntimeError(f"prompt reconstruction changed: {row_key(row)}")
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **TEMPLATE_KWARGS,
    )
    if not prompt.endswith(SOLAR_OUTPUT_ONLY_GENERATION_PREAMBLE):
        raise RuntimeError(
            "Solar output-only generation preamble changed: " f"{row_key(row)}"
        )
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if not ids:
        raise RuntimeError(f"rendered prompt is empty: {row_key(row)}")
    if len(ids) > max_prompt_tokens:
        raise RuntimeError(
            "rendered prompt exceeds the B200 context budget; truncation is "
            f"forbidden: row={row_key(row)} tokens={len(ids)} "
            f"limit={max_prompt_tokens}"
        )
    return [int(value) for value in ids]


def left_pad(
    prompt_ids: list[list[int]],
    *,
    pad_token_id: int,
) -> tuple[Any, Any]:
    import torch

    width = max(len(ids) for ids in prompt_ids)
    input_ids = torch.full(
        (len(prompt_ids), width),
        pad_token_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(input_ids)
    for index, ids in enumerate(prompt_ids):
        values = torch.tensor(ids, dtype=torch.long)
        input_ids[index, -len(ids) :] = values
        attention_mask[index, -len(ids) :] = 1
    return input_ids, attention_mask


def trim_generated_ids(
    token_ids: list[int],
    *,
    eos_token_ids: set[int],
    pad_token_id: int,
) -> list[int]:
    result: list[int] = []
    for token_id in token_ids:
        if token_id in eos_token_ids:
            break
        result.append(token_id)
    while result and result[-1] == pad_token_id:
        result.pop()
    return result


def classify_generation_stop(
    token_ids: list[int],
    *,
    eos_token_ids: set[int],
    max_new_tokens: int,
) -> str:
    if any(token_id in eos_token_ids for token_id in token_ids):
        return "eos"
    if len(token_ids) >= max_new_tokens:
        return "max_new_tokens"
    return "stopped_without_eos"


def decode_raw_continuation(tokenizer: Any, token_ids: list[int]) -> str:
    """Decode every non-EOS/pad continuation token and trim only outer space."""

    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ).strip()


def generation_integrity_violations(
    record: dict[str, Any],
    *,
    task: str,
    max_new_tokens: int,
) -> list[str]:
    """Audit one raw continuation without filtering or repairing it."""

    violations: list[str] = []
    prediction = record.get("prediction")
    raw_continuation = record.get("raw_continuation")
    if not isinstance(prediction, str):
        return ["prediction_not_string"]
    if not isinstance(raw_continuation, str):
        violations.append("raw_continuation_not_string")
    else:
        if raw_continuation.strip() != prediction:
            violations.append("postprocessing_not_outer_strip_only")
        if "\ufffd" in raw_continuation:
            violations.append("raw_unicode_replacement_character")
        if any(0xD800 <= ord(character) <= 0xDFFF for character in raw_continuation):
            violations.append("raw_unicode_surrogate")
        if any(
            unicodedata.category(character) == "Cc" and character not in {"\n", "\t"}
            for character in raw_continuation
        ):
            violations.append("raw_unicode_control_character")
        violations.extend(output_only_violations(raw_continuation))
    if not prediction:
        violations.append("empty_prediction")
    if "\ufffd" in prediction:
        violations.append("unicode_replacement_character")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in prediction):
        violations.append("unicode_surrogate")
    if any(
        unicodedata.category(character) == "Cc" and character not in {"\n", "\t"}
        for character in prediction
    ):
        violations.append("unicode_control_character")
    try:
        if prediction.encode("utf-8").decode("utf-8") != prediction:
            violations.append("unicode_utf8_roundtrip_mismatch")
    except UnicodeError:
        violations.append("unicode_utf8_roundtrip_error")
    violations.extend(output_only_violations(prediction))

    prompt_ids = record.get("prompt_token_ids")
    raw_ids = record.get("raw_generation_token_ids")
    prediction_ids = record.get("prediction_token_ids")
    for field, values in (
        ("prompt_token_ids", prompt_ids),
        ("raw_generation_token_ids", raw_ids),
        ("prediction_token_ids", prediction_ids),
    ):
        if not isinstance(values, list) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            violations.append(f"invalid_{field}")
    if isinstance(prompt_ids, list) and not prompt_ids:
        violations.append("empty_prompt_token_ids")
    if isinstance(prediction_ids, list):
        if not prediction_ids:
            violations.append("empty_prediction_token_ids")
        if len(prediction_ids) >= max_new_tokens:
            violations.append("generation_limit_exhausted")
        if record.get("prediction_tokens") != len(prediction_ids):
            violations.append("prediction_token_count_mismatch")
    if isinstance(prompt_ids, list) and record.get("prompt_tokens") != len(prompt_ids):
        violations.append("prompt_token_count_mismatch")
    if record.get("generation_stop_reason") != "eos":
        violations.append("generation_did_not_stop_at_eos")
    eos_ids = record.get("eos_token_ids")
    if (
        not isinstance(eos_ids, list)
        or not eos_ids
        or any(
            isinstance(value, bool) or not isinstance(value, int) for value in eos_ids
        )
    ):
        violations.append("invalid_eos_token_ids")
    if record.get("special_token_ids_in_prediction") != []:
        violations.append("special_token_in_prediction")
    stop_position = record.get("eos_position")
    if (
        isinstance(stop_position, bool)
        or not isinstance(stop_position, int)
        or not 0 <= stop_position < max_new_tokens
    ):
        violations.append("invalid_eos_position")
    elif isinstance(raw_ids, list) and isinstance(eos_ids, list):
        if stop_position >= len(raw_ids) or raw_ids[stop_position] not in eos_ids:
            violations.append("eos_position_token_mismatch")
        elif isinstance(prediction_ids, list) and (
            prediction_ids != raw_ids[:stop_position]
        ):
            violations.append("prediction_token_slice_mismatch")
    if record.get("tokenizer_roundtrip_exact") is not True:
        violations.append("tokenizer_roundtrip_mismatch")
    if record.get("unicode_roundtrip_exact") is not True:
        violations.append("unicode_roundtrip_mismatch")
    if task not in B200_TASK_ORDER:
        violations.append("invalid_task")
    return sorted(set(violations))


def validate_generation_record(
    record: dict[str, Any],
    *,
    task: str,
    max_new_tokens: int,
) -> None:
    violations = generation_integrity_violations(
        record,
        task=task,
        max_new_tokens=max_new_tokens,
    )
    if violations:
        raise RuntimeError(
            f"raw generation violates B200 output contract: {violations}"
        )
    if record.get("output_violation_audit") != {
        "status": "PASS",
        "violations": [],
        "postprocessing": "outer_strip_only",
    }:
        raise RuntimeError("generation output audit receipt is missing or changed")


def generate_batch(
    model: Any,
    tokenizer: Any,
    reference: Any,
    rows: list[dict[str, Any]],
    *,
    max_model_len: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    import torch

    max_prompt_tokens = max_model_len - max_new_tokens
    if max_prompt_tokens < 1:
        raise ValueError("max_new_tokens must be smaller than max_model_len")
    prompt_ids = [
        render_prompt_ids(
            tokenizer,
            reference,
            row,
            max_prompt_tokens=max_prompt_tokens,
        )
        for row in rows
    ]
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise RuntimeError("Solar tokenizer has neither a pad nor EOS token")
    input_ids, attention_mask = left_pad(
        prompt_ids,
        pad_token_id=int(pad_token_id),
    )
    input_width = input_ids.shape[1]
    input_ids = input_ids.to("cuda:0", non_blocking=True)
    attention_mask = attention_mask.to("cuda:0", non_blocking=True)
    with torch.inference_mode():
        sequences = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            top_p=1.0,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=int(pad_token_id),
            return_dict_in_generate=False,
        )
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    if isinstance(eos, int):
        eos_token_ids = {eos}
    else:
        eos_token_ids = {int(value) for value in (eos or [])}
    if not eos_token_ids:
        raise RuntimeError("Solar generation has no EOS token contract")
    special_token_ids = {
        int(value) for value in (getattr(tokenizer, "all_special_ids", None) or [])
    }
    results = []
    for index, ids in enumerate(prompt_ids):
        raw_generated = [
            int(value)
            for value in sequences[index, input_width:].detach().to("cpu").tolist()
        ]
        stop_reason = classify_generation_stop(
            raw_generated,
            eos_token_ids=eos_token_ids,
            max_new_tokens=max_new_tokens,
        )
        generated = trim_generated_ids(
            raw_generated,
            eos_token_ids=eos_token_ids,
            pad_token_id=int(pad_token_id),
        )
        raw_continuation = tokenizer.decode(
            generated,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        prediction = raw_continuation.strip()
        roundtrip_ids = tokenizer.encode(
            prediction,
            add_special_tokens=False,
        )
        roundtrip_text = tokenizer.decode(
            roundtrip_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        special_in_prediction = sorted(set(generated) & special_token_ids)
        eos_position = next(
            (
                position
                for position, token_id in enumerate(raw_generated)
                if token_id in eos_token_ids
            ),
            None,
        )
        result = {
            "prediction": prediction,
            "raw_continuation": raw_continuation,
            "prompt_token_ids": list(ids),
            "raw_generation_token_ids": raw_generated,
            "prediction_token_ids": generated,
            "prompt_tokens": len(ids),
            "prediction_tokens": len(generated),
            "generation_stop_reason": stop_reason,
            "eos_position": eos_position,
            "eos_token_ids": sorted(eos_token_ids),
            "special_token_ids_in_prediction": special_in_prediction,
            "tokenizer_roundtrip_exact": roundtrip_text == prediction,
            "unicode_roundtrip_exact": (
                prediction.encode("utf-8").decode("utf-8") == prediction
            ),
            "output_violation_audit": {
                "status": "PASS",
                "violations": [],
                "postprocessing": "outer_strip_only",
            },
        }
        if special_in_prediction:
            raise RuntimeError(
                "raw generation contains tokenizer special tokens before EOS: "
                f"{special_in_prediction}"
            )
        validate_generation_record(
            result,
            task=str(rows[index]["task"]),
            max_new_tokens=max_new_tokens,
        )
        results.append(result)
    return results


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=B200_INFERENCE_MODES, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--expert-nf4-dir", type=Path, required=True)
    parser.add_argument("--asset-integrity-receipt", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--terminal-training-dir", type=Path)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=B200_TENSOR_PARALLEL_SIZE,
    )
    parser.add_argument("--max-model-len", type=int, default=B200_MAX_MODEL_LEN)
    parser.add_argument(
        "--translation-max-new-tokens",
        type=int,
        default=B200_TASK_MAX_NEW_TOKENS["translation"],
    )
    parser.add_argument(
        "--summarization-max-new-tokens",
        type=int,
        default=B200_TASK_MAX_NEW_TOKENS["summarization"],
    )
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--limit-per-task", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser


def validate_b200_arguments(
    args: argparse.Namespace,
    config_bundle: dict[str, Any],
) -> None:
    evaluation = config_bundle["recipe"]["evaluation"]
    expected = {
        "tensor_parallel_size": B200_TENSOR_PARALLEL_SIZE,
        "max_model_len": int(evaluation["max_model_len"]),
        "translation_max_new_tokens": int(evaluation["translation_max_new_tokens"]),
        "summarization_max_new_tokens": int(evaluation["summarization_max_new_tokens"]),
        "limit_per_task": 0,
    }
    for field, wanted in expected.items():
        if getattr(args, field) != wanted:
            raise ValueError(
                f"B200 inference requires {field}={wanted}, "
                f"found {getattr(args, field)!r}"
            )
    if evaluation.get("temperature") != 0.0 or evaluation.get("top_p") != 1.0:
        raise RuntimeError("B200 recipe decoding is not temperature=0/top_p=1")
    if args.generation_batch_size < 1:
        raise ValueError("generation-batch-size must be positive")
    if args.mode == B200_VALIDATION_SMOKE_MODE:
        if args.terminal_training_dir is not None:
            raise ValueError(
                "validation smoke must not receive --terminal-training-dir"
            )
    elif args.terminal_training_dir is None:
        raise ValueError("production-evaluation requires --terminal-training-dir")


def main() -> int:
    args = build_argument_parser().parse_args()

    require_owned_root()
    runtime_user = require_expected_user()
    config_dir = require_b200_owned_path(args.config_dir, must_exist=True)
    config_bundle = load_b200_config_bundle(config_dir)
    config_identity = b200_config_identity(config_dir)
    validate_b200_arguments(args, config_bundle)
    if Path(sys.prefix).resolve() != VENV_DIR:
        raise RuntimeError(
            f"Solar inference must use {VENV_DIR}, "
            f"found {Path(sys.prefix).resolve()}"
        )
    source_dir = require_b200_owned_path(args.source_dir, must_exist=True)
    expert_dir = require_b200_owned_path(args.expert_nf4_dir, must_exist=True)
    asset_integrity_receipt = require_b200_owned_path(
        args.asset_integrity_receipt,
        must_exist=True,
    )
    validate_b200_large_asset_receipt(
        asset_integrity_receipt,
        role="inference",
    )
    if source_dir != BF16_SOURCE_DIR:
        raise RuntimeError(f"B200 inference source path changed: {source_dir}")
    if expert_dir != INFERENCE_EXPERT_DIR:
        raise RuntimeError(f"B200 inference expert path changed: {expert_dir}")
    eval_jsonl = require_b200_owned_path(args.eval_jsonl, must_exist=True)
    output_jsonl = require_b200_owned_path(args.output_jsonl)
    if output_jsonl.is_symlink() or output_jsonl.is_dir():
        raise RuntimeError(f"invalid B200 output path: {output_jsonl}")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    adapter_dir = (
        require_b200_owned_path(args.adapter_dir, must_exist=True)
        if args.adapter_dir is not None
        else None
    )
    terminal_training = None
    if args.mode == B200_PRODUCTION_EVALUATION_MODE:
        terminal_training = validate_terminal_training_success(
            require_b200_owned_path(
                args.terminal_training_dir,
                must_exist=True,
            ),
            config_identity=config_identity,
            recipe=config_bundle["recipe"],
        )
        if adapter_dir is not None and adapter_dir != Path(
            terminal_training["selected_checkpoint"]["adapter_dir"]
        ):
            raise RuntimeError(
                "production tuned inference requires the validation-selected "
                "best adapter"
            )
    reference = load_reference_module(config_dir=config_dir)
    rows, evaluation_identity, evaluation_row_identities = prepare_b200_evaluation(
        mode=args.mode,
        eval_path=eval_jsonl,
        config_bundle=config_bundle,
        reference=reference,
    )
    if not rows:
        raise RuntimeError("no evaluation rows selected")
    gpu_runtime = require_b200_gpu_runtime()
    b200_runtime_identity = {
        "user": runtime_user,
        "python_environment": str(Path(sys.prefix).resolve()),
        "gpu": gpu_runtime,
    }

    contract = build_inference_contract(
        source_dir=source_dir,
        expert_dir=expert_dir,
        adapter_dir=adapter_dir,
        model_label=args.model_label,
        eval_jsonl=eval_jsonl,
        selected_rows=rows,
        max_model_len=args.max_model_len,
        translation_max_new_tokens=args.translation_max_new_tokens,
        summarization_max_new_tokens=args.summarization_max_new_tokens,
        generation_batch_size=args.generation_batch_size,
        limit_per_task=args.limit_per_task,
        config_dir=config_dir,
        config_identity=config_identity,
        evaluation_identity=evaluation_identity,
        inference_mode=args.mode,
        terminal_training=terminal_training,
        b200_runtime_identity=b200_runtime_identity,
    )
    contract_sha256 = canonical_sha256(contract)
    manifest_path, manifest, previous, done = initialize_or_resume(
        output_jsonl=output_jsonl,
        contract=contract,
        selected_rows=rows,
        resume=args.resume,
        evaluation_row_identities=evaluation_row_identities,
    )
    if len(done) == len(rows):
        if manifest["status"] != "COMPLETE":
            seal_predictions(
                manifest_path,
                manifest,
                output_jsonl=output_jsonl,
                completed_rows=len(done),
            )
        print(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "model_label": args.model_label,
                    "already_done": len(done),
                    "contract_sha256": contract_sha256,
                }
            )
        )
        return 0

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        source_dir,
        use_fast=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    from solar_inference_loader import (
        default_layer_device_map,
        load_inference_model,
    )

    layer_device_map = default_layer_device_map()
    if layer_device_map != dict(EXPECTED_LAYER_DEVICE_MAP):
        raise RuntimeError("inference loader B200 layer map changed")
    if "cuda:2" in layer_device_map.values():
        raise RuntimeError("inference loader map unexpectedly references cuda:2")

    model, loader_receipt = load_inference_model(
        source_dir=source_dir,
        expert_checkpoint_dir=expert_dir,
        config_dir=config_dir,
        layer_device_map=layer_device_map,
        verify_hashes=False,
        asset_integrity_receipt=asset_integrity_receipt,
    )
    verify_loader_receipt(loader_receipt, contract)
    adapter_audit = None
    if adapter_dir is not None:
        model, adapter_audit = attach_adapter(
            model,
            adapter_dir=adapter_dir,
            model_contract=config_bundle["model"],
            recipe=config_bundle["recipe"],
        )
        if validate_adapter(adapter_dir, config_dir=config_dir) != contract["adapter"]:
            raise RuntimeError("adapter artifact changed while the model was loading")
    if lightweight_source_identity(source_dir) != contract["source"]:
        raise RuntimeError("source/tokenizer artifact changed while loading")
    if (
        lightweight_expert_identity(
            expert_dir,
            source=contract["source"],
        )
        != contract["per_expert_nf4"]
    ):
        raise RuntimeError("per-expert NF4 artifact changed while loading")
    if sha256(eval_jsonl) != contract["evaluation"]["sha256"]:
        raise RuntimeError("evaluation artifact changed while loading")
    for component in contract["evaluation"].get("components", {}).values():
        if sha256(Path(component["path"])) != component["sha256"]:
            raise RuntimeError("sealed evaluation component changed while loading")
    if b200_config_identity(config_dir) != contract["b200_config"]:
        raise RuntimeError("B200 config bundle changed while loading")
    if terminal_training is not None:
        refreshed_terminal = validate_terminal_training_success(
            Path(terminal_training["training_dir"]),
            config_identity=config_identity,
            recipe=config_bundle["recipe"],
        )
        if refreshed_terminal != terminal_training:
            raise RuntimeError("terminal training evidence changed while loading")
    record_runtime_audit(
        manifest_path,
        manifest,
        loader_receipt=loader_receipt,
        adapter_audit=adapter_audit,
    )
    print(
        json.dumps(
            {
                "status": "MODEL_READY",
                "model_label": args.model_label,
                "remaining": len(rows) - len(done),
                "already_done": len(done),
                "contract_sha256": contract_sha256,
                "loader_audit": loader_receipt,
                "adapter_audit": adapter_audit,
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(
            "[signal] SIGUSR1 received; stopping after the current batch",
            flush=True,
        )

    signal.signal(signal.SIGUSR1, request_stop)
    invariants = expected_row_provenance(contract, contract_sha256)
    remaining = [row for row in rows if row_key(row) not in done]
    mode = "a" if output_jsonl.exists() else "w"
    generated = 0
    with output_jsonl.open(mode, encoding="utf-8") as output:
        for task, max_new_tokens in (
            ("translation", args.translation_max_new_tokens),
            ("summarization", args.summarization_max_new_tokens),
        ):
            task_rows = [row for row in remaining if row["task"] == task]
            for start in range(0, len(task_rows), args.generation_batch_size):
                batch = task_rows[start : start + args.generation_batch_size]
                results = generate_batch(
                    model,
                    tokenizer,
                    reference,
                    batch,
                    max_model_len=args.max_model_len,
                    max_new_tokens=max_new_tokens,
                )
                for row, result in zip(batch, results, strict=True):
                    evaluation_row_identity = evaluation_row_identities[row_key(row)]
                    output.write(
                        json.dumps(
                            {
                                "id": row["id"],
                                "task": task,
                                **invariants,
                                "model_dir": str(source_dir),
                                "source_dir": str(source_dir),
                                "expert_nf4_dir": str(expert_dir),
                                "adapter_dir": (
                                    str(adapter_dir)
                                    if adapter_dir is not None
                                    else None
                                ),
                                "origin": row["origin"],
                                "reference": row["response"],
                                "reference_source": row.get("reference_source"),
                                "prompt_variant": "no_dictionary_context",
                                "reasoning_effort": "none",
                                "evaluation_split": evaluation_row_identity["split"],
                                "evaluation_row_identity": (evaluation_row_identity),
                                **result,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    done.add(row_key(row))
                    generated += 1
                output.flush()
                os.fsync(output.fileno())
                update_progress_manifest(
                    manifest_path,
                    manifest,
                    completed_rows=len(done),
                )
                print(
                    f"generated={generated}/{len(remaining)} "
                    f"total={len(done)}/{len(rows)} model={args.model_label}",
                    flush=True,
                )
                if stop_requested:
                    return REQUEUE_EXIT_CODE

    if len(done) != len(rows):
        raise RuntimeError(f"generation coverage mismatch: {len(done)} != {len(rows)}")
    seal_predictions(
        manifest_path,
        manifest,
        output_jsonl=output_jsonl,
        completed_rows=len(done),
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "model_label": args.model_label,
                "generated": generated,
                "already_done": len(previous),
                "total": len(done),
                "contract_sha256": contract_sha256,
                "reference_evaluator_sha256": REFERENCE_SHA256,
                "provenance_manifest": str(manifest_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
