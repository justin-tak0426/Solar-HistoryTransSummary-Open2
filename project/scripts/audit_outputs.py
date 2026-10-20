#!/usr/bin/env python3
"""Audit raw Solar predictions without filtering or repairing model output."""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import (
    CONFIG_DIR,
    OUTPUT_ONLY_FORBIDDEN_PATTERNS,
    PROJECT,
    REFERENCE_SHA256,
    RESULT_LABEL_PREFIX_RE,
    atomic_json,
    atomic_text,
    canonical_sha256,
    load_reference_module,
    iter_jsonl,
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
    INFERENCE_EXPERT_DIR,
    VENV_DIR,
    b200_config_identity,
    load_b200_config_bundle,
    require_expected_user,
    require_owned_path as require_b200_owned_path,
)
from infer import (
    B200_MAX_MODEL_LEN,
    B200_PRODUCTION_EVALUATION_MODE,
    B200_TASK_MAX_NEW_TOKENS,
    PINNED_COMBINED_EVAL_ROWS,
    PINNED_COMBINED_EVAL_SHA256,
    PINNED_COMBINED_EVAL_TASK_COUNTS,
    PROVENANCE_SCHEMA,
    PROVENANCE_SCHEMA_VERSION,
    TEMPLATE_KWARGS,
    expected_row_provenance,
    lightweight_expert_identity,
    lightweight_source_identity,
    read_prediction_rows,
    validate_adapter,
    validate_generation_record,
    verify_recorded_runtime_identity,
)
from quantization.output_checkpoint import read_json_no_duplicates
from verify_final_adapter import verify as verify_final_adapter


def resolve_config_dir(config_dir: Path) -> Path:
    resolved = config_dir.resolve(strict=True)
    if resolved == B200_CONFIG_DIR:
        load_b200_config_bundle(resolved)
        return resolved
    if resolved == CONFIG_DIR.resolve(strict=True):
        return resolved
    raise RuntimeError(f"unsupported config directory: {resolved}")


def is_b200_config(config_dir: Path) -> bool:
    return resolve_config_dir(config_dir) == B200_CONFIG_DIR


def selected_owned_path(
    path: Path,
    *,
    config_dir: Path,
    create_parent: bool = False,
    must_exist: bool = False,
) -> Path:
    if is_b200_config(config_dir):
        resolved = require_b200_owned_path(path, must_exist=must_exist)
        if create_parent:
            resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved
    return require_owned_path(path, create_parent=create_parent)


def detect_output_signals(prediction: str) -> dict[str, str]:
    signals: dict[str, str] = {}
    if not prediction.strip():
        signals["empty"] = ""
    for name in output_only_violations(prediction):
        pattern = OUTPUT_ONLY_FORBIDDEN_PATTERNS[name]
        match = pattern.search(prediction)
        if match is not None:
            signals[name] = match.group(0)
    return signals


def generation_integrity_failures(row: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    prediction = row.get("prediction")
    if not isinstance(prediction, str):
        return ["prediction_not_string"]
    if "\ufffd" in prediction:
        failures.append("unicode_replacement_character")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in prediction):
        failures.append("unicode_surrogate")
    if any(
        unicodedata.category(character) == "Cc" and character not in {"\n", "\t"}
        for character in prediction
    ):
        failures.append("unicode_control_character")

    for field in ("prompt_tokens", "prediction_tokens"):
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            failures.append(f"invalid_{field}")

    task = row.get("task")
    decoding = row.get("decoding")
    maximum: Any = None
    if isinstance(task, str) and isinstance(decoding, dict):
        maximum = decoding.get(f"{task}_max_new_tokens")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
        failures.append("invalid_generation_limit")
    else:
        prediction_tokens = row.get("prediction_tokens")
        if (
            isinstance(prediction_tokens, int)
            and not isinstance(prediction_tokens, bool)
            and prediction_tokens >= maximum
        ):
            failures.append("generation_limit_exhausted")
    if row.get("generation_stop_reason") != "eos":
        failures.append("generation_did_not_stop_at_eos")
    if row.get("tokenizer_roundtrip_exact") is not True:
        failures.append("tokenizer_roundtrip_mismatch")
    return failures


def audit_status(
    models: list[dict[str, Any]],
    integrity_failures: list[dict[str, Any]],
    output_failures: list[dict[str, Any]],
) -> dict[str, str]:
    clean = all(model["confirmed_output_failures"]["count"] == 0 for model in models)
    return {
        "status": (
            "PASS"
            if not integrity_failures and not output_failures and clean
            else "FAIL"
        ),
        "evaluation_integrity_status": ("PASS" if not integrity_failures else "FAIL"),
        "output_cleanliness_status": "PASS" if clean else "FAIL",
    }


def scan_reference_outputs(
    config_dir: Path = CONFIG_DIR,
) -> dict[str, Any]:
    """Audit targets from the explicitly selected config profile."""

    data_contract = read_json(config_dir / "data_contract.json")
    output_contract = read_json(config_dir / "recipe.json")["output_only_contract"]
    exception_ids = {
        "train": set(output_contract["training_target_prefix_removal"]["ids"]),
        "validation": set(),
        "full_test": set(
            output_contract["sealed_full_test_reference_exceptions"]["ids"]
        ),
        "human_gold": set(),
    }
    roles: dict[str, Any] = {}
    for role in ("train", "validation", "full_test", "human_gold"):
        file_contract = data_contract["files"][role]
        path = (
            Path(data_contract["prepared_data_dir"]) / file_contract["name"]
        ).resolve()
        if sha256(path) != file_contract["sha256"]:
            raise RuntimeError(f"{role} changed before reference output scan")
        seen_exceptions: set[str] = set()
        task_counts: Counter[str] = Counter()
        category_counts: Counter[str] = Counter()
        failures: list[dict[str, Any]] = []
        row_count = 0
        for row_count, row in enumerate(iter_jsonl(path), 1):
            row_id = str(row.get("id") or "")
            task = str(row.get("task") or "")
            task_counts[task] += 1
            response = str(row.get("response") or "")
            prefix_match = RESULT_LABEL_PREFIX_RE.match(response)
            if row_id in exception_ids[role]:
                if prefix_match is None:
                    raise RuntimeError(
                        f"{role} pinned reference prefix disappeared: {row_id}"
                    )
                seen_exceptions.add(row_id)
                response = response[prefix_match.end() :]
            elif prefix_match is not None:
                raise RuntimeError(f"{role} gained an unpinned answer label: {row_id}")
            signals = detect_output_signals(response)
            if signals:
                category_counts.update(signals)
                if len(failures) < 10:
                    failures.append(
                        {
                            "id": row_id,
                            "task": task,
                            "signals": signals,
                            "response": response[:360],
                        }
                    )
        if row_count != int(file_contract["rows"]):
            raise RuntimeError(f"{role} reference scan row count changed")
        if dict(task_counts) != file_contract["by_task"]:
            raise RuntimeError(f"{role} reference scan task counts changed")
        if seen_exceptions != exception_ids[role]:
            raise RuntimeError(f"{role} reference prefix exceptions changed")
        if category_counts:
            raise RuntimeError(
                f"{role} reference output false positives: "
                f"counts={dict(category_counts)} examples={failures}"
            )
        roles[role] = {
            "path": str(path),
            "sha256": file_contract["sha256"],
            "rows": row_count,
            "by_task": dict(task_counts),
            "normalized_answer_prefix_count": len(seen_exceptions),
            "normalized_answer_prefix_ids": sorted(seen_exceptions),
            "output_signal_rows": 0,
            "output_signal_categories": {},
        }
    return {
        "status": "PASS",
        "detector": "same detect_output_signals used for raw predictions",
        "prediction_normalization": "none",
        "reference_normalization": ("pinned answer-prefix exceptions only"),
        "roles": roles,
    }


def key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("id") or ""), str(row.get("task") or "")


def validate_prediction_seal(
    path: Path,
    *,
    label: str,
    expected_adapter: dict[str, Any] | None,
    expected_eval_path: Path,
    expected_eval_sha256: str,
    expected_eval_rows: list[dict[str, Any]],
    config_dir: Path = CONFIG_DIR,
    expected_terminal_training: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config_dir = resolve_config_dir(config_dir)
    b200 = is_b200_config(config_dir)
    manifest_path = path.with_name(path.name + ".manifest.json")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError(
            f"prediction provenance manifest is missing: {manifest_path}"
        )
    manifest = read_json_no_duplicates(manifest_path)
    rows = read_prediction_rows(path)
    wanted_manifest = {
        "schema": PROVENANCE_SCHEMA,
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "status": "COMPLETE",
        "expected_rows": PINNED_COMBINED_EVAL_ROWS,
        "completed_rows": PINNED_COMBINED_EVAL_ROWS,
        "output_jsonl_sha256": sha256(path),
    }
    for field, expected in wanted_manifest.items():
        if manifest.get(field) != expected:
            raise RuntimeError(
                f"{label} prediction manifest {field} changed: "
                f"{manifest.get(field)!r} != {expected!r}"
            )
    if len(rows) != PINNED_COMBINED_EVAL_ROWS:
        raise RuntimeError(f"{label} sealed prediction row count changed")
    contract = manifest.get("contract")
    if not isinstance(contract, dict):
        raise RuntimeError(f"{label} prediction manifest lacks a contract")
    contract_sha256 = canonical_sha256(contract)
    if manifest.get("contract_sha256") != contract_sha256:
        raise RuntimeError(f"{label} inference contract digest changed")
    for field, expected in (
        ("schema", PROVENANCE_SCHEMA),
        ("schema_version", PROVENANCE_SCHEMA_VERSION),
        ("model_label", label),
        ("adapter", expected_adapter),
        ("prompt", {"variant": "no_dictionary_context", **TEMPLATE_KWARGS}),
    ):
        if contract.get(field) != expected:
            raise RuntimeError(f"{label} inference contract {field} changed")
    if b200:
        identity = b200_config_identity(config_dir)
        for field, expected in (
            ("profile", "b200"),
            ("inference_mode", B200_PRODUCTION_EVALUATION_MODE),
            ("b200_config", identity),
            ("terminal_training", expected_terminal_training),
        ):
            if contract.get(field) != expected:
                raise RuntimeError(f"{label} B200 inference {field} changed")
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
        if contract.get("decoding") != expected_decoding or contract.get(
            "decoding_sha256"
        ) != canonical_sha256(expected_decoding):
            raise RuntimeError(f"{label} B200 decoding contract changed")
        if contract.get("model_variant") != (
            "tuned" if expected_adapter is not None else "base"
        ):
            raise RuntimeError(f"{label} B200 model variant changed")

    evaluation = contract.get("evaluation")
    expected_keys = [[row["id"], row["task"]] for row in expected_eval_rows]
    if b200:
        data_contract = read_json(config_dir / "data_contract.json")
        if not isinstance(evaluation, dict):
            raise RuntimeError(f"{label} B200 evaluation contract is missing")
        combined = data_contract["files"]["combined_eval"]
        for field, expected in (
            ("role", "full_test_plus_human_gold_sealed_union"),
            ("path", str(expected_eval_path)),
            ("sha256", expected_eval_sha256),
            ("rows", PINNED_COMBINED_EVAL_ROWS),
            ("by_task", PINNED_COMBINED_EVAL_TASK_COUNTS),
            ("mode", B200_PRODUCTION_EVALUATION_MODE),
            ("ordered_union", ["full_test", "human_gold"]),
            ("selected_rows", PINNED_COMBINED_EVAL_ROWS),
            ("selected_keys_sha256", canonical_sha256(expected_keys)),
            ("sealed_evaluation_accessed", True),
            ("limit_per_task", 0),
        ):
            if evaluation.get(field) != expected:
                raise RuntimeError(f"{label} B200 evaluation contract {field} changed")
        if combined.get("sha256") != expected_eval_sha256:
            raise RuntimeError("B200 combined evaluation config changed")
        components = evaluation.get("components")
        if not isinstance(components, dict):
            raise RuntimeError("B200 evaluation components are missing")
        for split in ("full_test", "human_gold"):
            entry = data_contract["files"][split]
            expected_path = str(
                (Path(data_contract["prepared_data_dir"]) / entry["name"]).resolve(
                    strict=True
                )
            )
            component = components.get(split)
            if not isinstance(component, dict) or any(
                component.get(field) != expected
                for field, expected in (
                    ("role", split),
                    ("path", expected_path),
                    ("sha256", entry["sha256"]),
                    ("rows", entry["rows"]),
                    ("by_task", entry["by_task"]),
                )
            ):
                raise RuntimeError(f"{label} B200 {split} component changed")
    else:
        wanted_evaluation = {
            "role": "full_test_plus_human_gold_sealed_union",
            "path": str(expected_eval_path),
            "sha256": expected_eval_sha256,
            "pinned_total_rows": PINNED_COMBINED_EVAL_ROWS,
            "pinned_by_task": PINNED_COMBINED_EVAL_TASK_COUNTS,
            "selected_rows": PINNED_COMBINED_EVAL_ROWS,
            "selected_keys_sha256": canonical_sha256(expected_keys),
            "limit_per_task": 0,
        }
        if evaluation != wanted_evaluation:
            raise RuntimeError(f"{label} sealed evaluation contract changed")

    source_contract = contract.get("source")
    if not isinstance(source_contract, dict):
        raise RuntimeError(f"{label} source contract is missing")
    source_dir = selected_owned_path(
        Path(str(source_contract.get("path") or "")),
        config_dir=config_dir,
        must_exist=True,
    )
    if b200 and source_dir != BF16_SOURCE_DIR:
        raise RuntimeError(f"{label} B200 source directory changed")
    if lightweight_source_identity(source_dir) != source_contract:
        raise RuntimeError(f"{label} official source changed after inference")
    expert_contract = contract.get("per_expert_nf4")
    if not isinstance(expert_contract, dict):
        raise RuntimeError(f"{label} per-expert contract is missing")
    expert_dir = selected_owned_path(
        Path(str(expert_contract.get("path") or "")),
        config_dir=config_dir,
        must_exist=True,
    )
    if b200 and expert_dir != INFERENCE_EXPERT_DIR:
        raise RuntimeError(f"{label} B200 inference expert directory changed")
    if (
        lightweight_expert_identity(expert_dir, source=source_contract)
        != expert_contract
    ):
        raise RuntimeError(f"{label} NF4 expert artifact changed after inference")
    if expected_adapter is not None:
        adapter_dir = selected_owned_path(
            Path(str(expected_adapter.get("path") or "")),
            config_dir=config_dir,
            must_exist=True,
        )
        if validate_adapter(adapter_dir, config_dir=config_dir) != expected_adapter:
            raise RuntimeError(f"{label} adapter changed after inference")

    implementation = contract.get("implementation")
    if not isinstance(implementation, dict):
        raise RuntimeError(f"{label} implementation contract is missing")
    wanted_implementation = {
        "infer_py_sha256": sha256(PROJECT / "scripts/infer.py"),
        "inference_loader_sha256": sha256(
            PROJECT / "scripts/solar_inference_loader.py"
        ),
        "reference_evaluator_sha256": REFERENCE_SHA256,
        "model_contract_sha256": sha256(config_dir / "model.json"),
        "recipe_sha256": sha256(config_dir / "recipe.json"),
        "data_contract_sha256": sha256(config_dir / "data_contract.json"),
        "environment_lock_sha256": sha256(
            PROJECT / "requirements/solar-open-2.lock.txt"
        ),
        "pipeline_code_sha256": pipeline_code_identity(
            config_dir=config_dir if b200 else None
        )["sha256"],
    }
    for field, expected in wanted_implementation.items():
        if implementation.get(field) != expected:
            raise RuntimeError(f"{label} inference implementation {field} changed")
    if (
        b200
        and implementation.get("config_files")
        != b200_config_identity(config_dir)["files"]
    ):
        raise RuntimeError(f"{label} inference config file identities changed")

    runtime_audit = manifest.get("runtime_audit")
    if not isinstance(runtime_audit, dict):
        raise RuntimeError(f"{label} runtime audit is missing")
    model_audit = runtime_audit.get("model_audit")
    if (
        not isinstance(model_audit, dict)
        or model_audit.get("status") != "PASS"
        or runtime_audit.get("hash_verification") is not True
        or runtime_audit.get("production_eligible") is not True
    ):
        raise RuntimeError(f"{label} runtime audit is not production eligible")
    adapter_audit = runtime_audit.get("adapter_audit")
    if expected_adapter is None:
        if adapter_audit is not None:
            raise RuntimeError(f"{label} base runtime unexpectedly loaded an adapter")
    elif not isinstance(adapter_audit, dict) or adapter_audit.get("status") != "PASS":
        raise RuntimeError(f"{label} tuned runtime adapter audit failed")
    verify_recorded_runtime_identity(runtime_audit, contract)
    if b200:
        runtime = contract.get("runtime")
        loader_receipt = runtime_audit.get("loader_receipt")
        if (
            not isinstance(runtime, dict)
            or not isinstance(loader_receipt, dict)
            or runtime.get("logical_gpu_count") != 2
            or runtime.get("hf_device_map") != dict(EXPECTED_HF_DEVICE_MAP)
            or "cuda:2" in runtime.get("hf_device_map", {}).values()
            or runtime.get("b200_identity")
            != {
                "user": "e202",
                "python_environment": str(VENV_DIR),
                "gpu": loader_receipt.get("hardware"),
            }
        ):
            raise RuntimeError(f"{label} B200 runtime placement changed")

    invariants = expected_row_provenance(contract, contract_sha256)
    for line_number, row in enumerate(rows, 1):
        for field, expected in invariants.items():
            if row.get(field) != expected:
                raise RuntimeError(
                    f"{label} row {line_number} provenance {field} changed"
                )
        if b200:
            split = "full_test" if line_number <= 18_557 else "human_gold"
            position = line_number - 1 if split == "full_test" else line_number - 18_558
            split_entry = data_contract["files"][split]
            expected_identity = {
                "split": split,
                "position": position,
                "id": str(expected_eval_rows[line_number - 1]["id"]),
                "task": str(expected_eval_rows[line_number - 1]["task"]),
                "dataset_sha256": split_entry["sha256"],
                "row_sha256": canonical_sha256(expected_eval_rows[line_number - 1]),
            }
            if row.get("evaluation_split") != split:
                raise RuntimeError(f"{label} row {line_number} split changed")
            if row.get("evaluation_row_identity") != expected_identity:
                raise RuntimeError(f"{label} row {line_number} identity changed")
            task = str(row.get("task") or "")
            validate_generation_record(
                row,
                task=task,
                max_new_tokens=B200_TASK_MAX_NEW_TOKENS[task],
            )
    return rows, {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "output_jsonl_sha256": sha256(path),
        "contract_sha256": contract_sha256,
        "source_dir": str(source_dir),
        "expert_nf4_dir": str(expert_dir),
        "adapter": expected_adapter,
        "profile": "b200" if b200 else "source",
        "b200_config": b200_config_identity(config_dir) if b200 else None,
        "terminal_training": expected_terminal_training if b200 else None,
        "source_shards_sha256": canonical_sha256(
            runtime_audit["source_inventory"]["shards"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        nargs=2,
        action="append",
        metavar=("LABEL", "PREDICTIONS"),
        required=True,
    )
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--tuned-label", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    config_dir = resolve_config_dir(args.config_dir)
    if is_b200_config(config_dir):
        require_expected_user()
    eval_path = selected_owned_path(
        args.eval_jsonl, config_dir=config_dir, must_exist=True
    )
    output_json = selected_owned_path(
        args.output_json, config_dir=config_dir, create_parent=True
    )
    output_md = selected_owned_path(
        args.output_md, config_dir=config_dir, create_parent=True
    )
    reference_module = load_reference_module(
        config_dir=config_dir if is_b200_config(config_dir) else None
    )
    reference_output_scan = scan_reference_outputs(config_dir)
    training_manifest = selected_owned_path(
        args.training_manifest, config_dir=config_dir, must_exist=True
    )
    training = read_json(training_manifest)
    selected = training.get("selected_checkpoint") or {}
    selected_adapter_dir = selected_owned_path(
        Path(str(selected.get("adapter_dir") or "")),
        config_dir=config_dir,
        must_exist=True,
    )
    adapter_gate = verify_final_adapter(
        training_manifest,
        selected_adapter_dir,
        config_dir=config_dir,
    )
    tuned_adapter_files = adapter_gate["adapter_inference_identity"][
        "adapter_file_sha256"
    ]
    expected_rows = read_jsonl(eval_path)
    data_contract = read_json(config_dir / "data_contract.json")
    combined = data_contract["files"]["combined_eval"]
    if (
        combined.get("sha256") != PINNED_COMBINED_EVAL_SHA256
        or combined.get("rows") != PINNED_COMBINED_EVAL_ROWS
        or combined.get("by_task") != PINNED_COMBINED_EVAL_TASK_COUNTS
    ):
        raise RuntimeError("audit data contract differs from the pinned eval union")
    expected_eval_path = (
        Path(data_contract["prepared_data_dir"]) / combined["name"]
    ).resolve()
    if eval_path != expected_eval_path:
        raise RuntimeError("audit eval path differs from pinned combined eval")
    if sha256(eval_path) != combined["sha256"]:
        raise RuntimeError("audit eval checksum differs from pinned contract")
    if len(expected_rows) != int(combined["rows"]):
        raise RuntimeError("audit eval row count differs from pinned contract")
    reference_module.validate_eval_rows(expected_rows, str(eval_path))
    expected = {key(row): row for row in expected_rows}
    models = []
    integrity_failures = []
    output_failures = []
    labels = [str(label) for label, _path in args.model]
    if len(labels) != 2 or len(set(labels)) != 2 or args.tuned_label not in labels:
        raise RuntimeError(
            "audit requires exactly one base and one tuned prediction label"
        )
    for label, raw_path in args.model:
        path = selected_owned_path(
            Path(raw_path), config_dir=config_dir, must_exist=True
        )
        expected_adapter = (
            adapter_gate["adapter_inference_identity"]
            if label == args.tuned_label
            else None
        )
        rows, prediction_seal = validate_prediction_seal(
            path,
            label=label,
            expected_adapter=expected_adapter,
            expected_eval_path=eval_path,
            expected_eval_sha256=combined["sha256"],
            expected_eval_rows=expected_rows,
            config_dir=config_dir,
            expected_terminal_training=(
                adapter_gate.get("terminal_training")
                if is_b200_config(config_dir)
                else None
            ),
        )
        observed = {key(row): row for row in rows}
        structural = Counter()
        categories = Counter()
        output_failure_keys: set[tuple[str, str]] = set()
        examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
        if len(observed) != len(rows):
            structural["duplicate_key"] += len(rows) - len(observed)
        missing = set(expected) - set(observed)
        unexpected = set(observed) - set(expected)
        if missing:
            structural["missing_key"] += len(missing)
        if unexpected:
            structural["unexpected_key"] += len(unexpected)
        for row_key in set(expected) & set(observed):
            row = observed[row_key]
            wanted = expected[row_key]
            for field, value in (
                ("model_label", label),
                ("origin", wanted["origin"]),
                ("reference", wanted["response"]),
                ("reference_source", wanted.get("reference_source")),
                ("prompt_variant", "no_dictionary_context"),
                ("reasoning_effort", "none"),
            ):
                if row.get(field) != value:
                    structural[f"{field}_mismatch"] += 1
            provenance = row.get("provenance")
            provenance_adapter = (
                provenance.get("adapter_file_sha256")
                if isinstance(provenance, dict)
                else "__missing__"
            )
            if label == args.tuned_label:
                if row.get("adapter_dir") != str(selected_adapter_dir):
                    structural["selected_adapter_dir_mismatch"] += 1
                if row.get("adapter_file_sha256") != tuned_adapter_files:
                    structural["selected_adapter_files_mismatch"] += 1
                if provenance_adapter != tuned_adapter_files:
                    structural["provenance_adapter_files_mismatch"] += 1
            else:
                if row.get("adapter_dir") is not None:
                    structural["base_adapter_dir_not_null"] += 1
                if row.get("adapter_file_sha256") is not None:
                    structural["base_adapter_files_not_null"] += 1
                if provenance_adapter is not None:
                    structural["base_provenance_adapter_not_null"] += 1
            for failure in generation_integrity_failures(row):
                structural[failure] += 1
            prediction_value = row.get("prediction")
            prediction = prediction_value if isinstance(prediction_value, str) else ""
            for name, match_text in detect_output_signals(prediction).items():
                categories[name] += 1
                output_failure_keys.add(row_key)
                if len(examples[name]) < 10:
                    examples[name].append(
                        {
                            "id": row_key[0],
                            "task": row_key[1],
                            "match": match_text,
                            "prediction": prediction[:360],
                        }
                    )
        structural = +structural
        if structural:
            integrity_failures.append({"model": label, "failures": dict(structural)})
        confirmed = len(output_failure_keys)
        if confirmed:
            output_failures.append(
                {
                    "model": label,
                    "violating_rows": confirmed,
                    "signals": dict(categories),
                }
            )
        models.append(
            {
                "model_label": label,
                "prediction_path": str(path),
                "prediction_seal": prediction_seal,
                "rows": len(rows),
                "structural_failures": dict(structural),
                "output_categories": dict(categories),
                "confirmed_output_failures": {
                    "count": confirmed,
                    "rate": confirmed / len(rows) if rows else 1.0,
                    "signal_occurrences": sum(categories.values()),
                },
                "examples": dict(examples),
            }
        )
    statuses = audit_status(models, integrity_failures, output_failures)
    failures = [*integrity_failures, *output_failures]
    result = {
        **statuses,
        "raw_scoring_policy": "outer strip only; no output filtering or repair",
        "pipeline_code_sha256": pipeline_code_identity(
            config_dir=config_dir if is_b200_config(config_dir) else None
        )["sha256"],
        "reference_evaluator_sha256": REFERENCE_SHA256,
        "eval_jsonl": str(eval_path),
        "eval_sha256": combined["sha256"],
        "expected_rows": len(expected),
        "training_adapter_gate": adapter_gate,
        "output_only_contract": read_json(config_dir / "recipe.json")[
            "output_only_contract"
        ],
        "reference_output_only_scan": reference_output_scan,
        "models": models,
        "integrity_failures": integrity_failures,
        "output_failures": output_failures,
    }
    if is_b200_config(config_dir):
        result["profile"] = "b200"
        result["b200_config"] = b200_config_identity(config_dir)
        result["terminal_training"] = adapter_gate["terminal_training"]
    lines = [
        "# Solar raw-output audit",
        "",
        "**Built with Solar**",
        "",
        f"- Evaluation integrity: `{result['evaluation_integrity_status']}`",
        f"- Output cleanliness: `{result['output_cleanliness_status']}`",
        "- Scoring policy: raw output, outer strip only, no repair",
        "",
        "| Model | Rows | Structural failures | Output signals |",
        "| --- | ---: | ---: | ---: |",
    ]
    for model in models:
        lines.append(
            f"| {model['model_label']} | {model['rows']} | "
            f"{sum(model['structural_failures'].values())} | "
            f"{model['confirmed_output_failures']['count']} |"
        )
    atomic_text(output_md, "\n".join(lines) + "\n")
    result["artifacts"] = {
        "markdown": {
            "path": str(output_md),
            "sha256": sha256(output_md),
            "bytes": output_md.stat().st_size,
        }
    }
    atomic_json(output_json, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
