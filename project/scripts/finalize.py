#!/usr/bin/env python3
"""Split Solar predictions and build base-vs-tuned test/gold reports."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

from common import (
    CONFIG_DIR,
    REFERENCE_COPY,
    REFERENCE_SHA256,
    atomic_json,
    atomic_text,
    canonical_sha256,
    pipeline_code_identity,
    read_json,
    read_jsonl,
    require_owned_path,
    sha256,
)
from b200_runtime import (
    B200_CONFIG_DIR,
    b200_config_identity,
    load_b200_config_bundle,
    require_owned_path as require_b200_owned_path,
)
from training_plan import early_stopping_decision


METRICS = (
    "corpus_bleu",
    "sentence_bleu_mean",
    "rougeL_f1_mean",
    "bertscore_f1_mean",
)
TASKS = ("translation", "summarization")
DATASETS = ("full_test", "human_gold")
EXPECTED_SPLIT_ROLES = {
    "train": "optimizer updates",
    "validation": "monitoring and checkpoint selection only",
    "full_test": "sealed final evaluation only",
    "human_gold": "sealed final evaluation only",
}
SPLIT_RECEIPT_SCHEMA = "solar-open2-evaluation-split-receipt"
METRICS_RECEIPT_SCHEMA = "solar-open2-exact-metrics-receipt"
RECEIPT_SCHEMA_VERSION = 1
EVALUATION_ENTRYPOINT = Path(__file__).with_name("evaluation.py").resolve(strict=True)
METRIC_ROW_FIELDS = (
    "sentence_bleu",
    "rougeL_f1",
    "bertscore_precision",
    "bertscore_recall",
    "bertscore_f1",
)


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


def selected_pipeline_identity(config_dir: Path) -> dict[str, Any]:
    return pipeline_code_identity(
        config_dir=config_dir if is_b200_config(config_dir) else None
    )


def selected_evaluator(config_dir: Path) -> Path:
    if is_b200_config(config_dir):
        if EVALUATION_ENTRYPOINT.is_symlink() or not EVALUATION_ENTRYPOINT.is_file():
            raise RuntimeError(
                f"B200 evaluation entrypoint is invalid: {EVALUATION_ENTRYPOINT}"
            )
        if sha256(EVALUATION_ENTRYPOINT) != REFERENCE_SHA256:
            raise RuntimeError("B200 evaluation.py methodology changed")
        return EVALUATION_ENTRYPOINT
    return REFERENCE_COPY


def file_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def require_file_identity(
    value: Any,
    *,
    label: str,
    expected_path: Path | None = None,
    config_dir: Path = CONFIG_DIR,
) -> Path:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} identity is missing")
    path = selected_owned_path(
        Path(str(value.get("path") or "")),
        config_dir=config_dir,
        must_exist=True,
    )
    if expected_path is not None and path != expected_path:
        raise RuntimeError(f"{label} path changed: {path} != {expected_path}")
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label} is not a regular file: {path}")
    if value.get("sha256") != sha256(path):
        raise RuntimeError(f"{label} checksum changed")
    if value.get("bytes") not in (None, path.stat().st_size):
        raise RuntimeError(f"{label} size changed")
    return path


def read_passing_audit(
    path: Path,
    *,
    config_dir: Path = CONFIG_DIR,
) -> dict[str, Any]:
    audit = read_json(path)
    if (
        audit.get("status") != "PASS"
        or audit.get("evaluation_integrity_status") != "PASS"
        or audit.get("output_cleanliness_status") != "PASS"
        or audit.get("integrity_failures")
        or audit.get("output_failures")
    ):
        raise RuntimeError("raw prediction audit is not a clean PASS")
    if (
        audit.get("pipeline_code_sha256")
        != selected_pipeline_identity(config_dir)["sha256"]
    ):
        raise RuntimeError("raw prediction audit was made by stale pipeline code")
    validate_reference_scan_receipt(audit, config_dir=config_dir)
    if is_b200_config(config_dir):
        if audit.get("profile") != "b200":
            raise RuntimeError("raw prediction audit is not a B200 receipt")
        if audit.get("b200_config") != b200_config_identity(config_dir):
            raise RuntimeError("raw prediction audit B200 config changed")
    return audit


def validate_reference_scan_receipt(
    audit: dict[str, Any],
    *,
    config_dir: Path = CONFIG_DIR,
) -> None:
    scan = audit.get("reference_output_only_scan")
    if (
        not isinstance(scan, dict)
        or set(scan)
        != {
            "status",
            "detector",
            "prediction_normalization",
            "reference_normalization",
            "roles",
        }
        or scan.get("status") != "PASS"
        or scan.get("detector") != "same detect_output_signals used for raw predictions"
        or scan.get("prediction_normalization") != "none"
        or scan.get("reference_normalization") != "pinned answer-prefix exceptions only"
    ):
        raise RuntimeError("reference output-only scan is not a clean PASS")
    data_contract = read_json(config_dir / "data_contract.json")
    output_contract = read_json(config_dir / "recipe.json")["output_only_contract"]
    expected_exceptions = {
        "train": sorted(output_contract["training_target_prefix_removal"]["ids"]),
        "validation": [],
        "full_test": sorted(
            output_contract["sealed_full_test_reference_exceptions"]["ids"]
        ),
        "human_gold": [],
    }
    roles = scan.get("roles")
    if not isinstance(roles, dict) or set(roles) != set(expected_exceptions):
        raise RuntimeError("reference output-only scan roles changed")
    for role, exceptions in expected_exceptions.items():
        value = roles.get(role)
        file_contract = data_contract["files"][role]
        wanted_path = str(
            (Path(data_contract["prepared_data_dir"]) / file_contract["name"]).resolve()
        )
        wanted = {
            "path": wanted_path,
            "sha256": file_contract["sha256"],
            "rows": file_contract["rows"],
            "by_task": file_contract["by_task"],
            "normalized_answer_prefix_count": len(exceptions),
            "normalized_answer_prefix_ids": exceptions,
            "output_signal_rows": 0,
            "output_signal_categories": {},
        }
        if value != wanted:
            raise RuntimeError(f"reference output-only scan receipt changed for {role}")


def audited_model(
    audit: dict[str, Any],
    *,
    model_label: str,
    prediction_path: Path,
    config_dir: Path = CONFIG_DIR,
) -> dict[str, Any]:
    matches = [
        model
        for model in audit.get("models") or []
        if model.get("model_label") == model_label
    ]
    if len(matches) != 1:
        raise RuntimeError(f"audit does not contain exactly one model {model_label}")
    model = matches[0]
    audited_path = selected_owned_path(
        Path(str(model.get("prediction_path") or "")),
        config_dir=config_dir,
        must_exist=True,
    )
    if audited_path != prediction_path:
        raise RuntimeError(f"audited prediction path changed for {model_label}")
    seal = model.get("prediction_seal")
    if not isinstance(seal, dict):
        raise RuntimeError(f"audit prediction seal is missing for {model_label}")
    if seal.get("output_jsonl_sha256") != sha256(prediction_path):
        raise RuntimeError(f"audited combined predictions changed for {model_label}")
    require_file_identity(
        {
            "path": seal.get("manifest_path"),
            "sha256": seal.get("manifest_sha256"),
        },
        label=f"{model_label} inference manifest",
        config_dir=config_dir,
    )
    failures = model.get("confirmed_output_failures") or {}
    if failures.get("count") != 0 or model.get("structural_failures"):
        raise RuntimeError(f"audited model is not output-only clean: {model_label}")
    return model


def validate_eval_contract(
    path: Path,
    *,
    role: str,
    rows: list[dict[str, Any]],
    config_dir: Path = CONFIG_DIR,
) -> dict[str, Any]:
    contract = read_json(config_dir / "data_contract.json")
    entry = contract["files"][role]
    expected_path = Path(contract["prepared_data_dir"]) / entry["name"]
    if path != expected_path.resolve():
        raise RuntimeError(f"{role} path differs from pinned data contract")
    if sha256(path) != entry["sha256"]:
        raise RuntimeError(f"{role} checksum differs from pinned data contract")
    if len(rows) != int(entry["rows"]):
        raise RuntimeError(f"{role} row count differs from pinned data contract")
    counts = {task: sum(row.get("task") == task for row in rows) for task in TASKS}
    if counts != entry["by_task"]:
        raise RuntimeError(f"{role} task counts differ from pinned data contract")
    return {
        "path": str(path),
        "sha256": entry["sha256"],
        "rows": entry["rows"],
        "by_task": entry["by_task"],
    }


def key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("id") or ""), str(row.get("task") or "")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def split_predictions(args: argparse.Namespace) -> None:
    config_dir = resolve_config_dir(getattr(args, "config_dir", CONFIG_DIR))
    combined_path = selected_owned_path(
        args.combined_predictions, config_dir=config_dir, must_exist=True
    )
    full_path = selected_owned_path(
        args.full_eval, config_dir=config_dir, must_exist=True
    )
    gold_path = selected_owned_path(
        args.human_gold_eval, config_dir=config_dir, must_exist=True
    )
    full_output = selected_owned_path(
        args.full_predictions, config_dir=config_dir, create_parent=True
    )
    gold_output = selected_owned_path(
        args.gold_predictions, config_dir=config_dir, create_parent=True
    )
    audit_path = selected_owned_path(
        args.audit_json, config_dir=config_dir, must_exist=True
    )
    receipt_path = selected_owned_path(
        args.receipt, config_dir=config_dir, create_parent=True
    )
    audit = read_passing_audit(audit_path, config_dir=config_dir)
    model_audit = audited_model(
        audit,
        model_label=args.model_label,
        prediction_path=combined_path,
        config_dir=config_dir,
    )
    predictions = read_jsonl(combined_path)
    full_rows = read_jsonl(full_path)
    gold_rows = read_jsonl(gold_path)
    full_contract = validate_eval_contract(
        full_path, role="full_test", rows=full_rows, config_dir=config_dir
    )
    gold_contract = validate_eval_contract(
        gold_path, role="human_gold", rows=gold_rows, config_dir=config_dir
    )
    prediction_map = {key(row): row for row in predictions}
    if len(prediction_map) != len(predictions):
        raise RuntimeError("combined predictions contain duplicate keys")
    full_keys = [key(row) for row in full_rows]
    gold_keys = [key(row) for row in gold_rows]
    if set(full_keys) & set(gold_keys):
        raise RuntimeError("full test and human gold overlap")
    if set(prediction_map) != set(full_keys) | set(gold_keys):
        raise RuntimeError("combined prediction keys do not match eval union")
    if is_b200_config(config_dir) and [key(row) for row in predictions] != [
        *full_keys,
        *gold_keys,
    ]:
        raise RuntimeError("B200 combined predictions changed ordered split union")
    for reference_row in [*full_rows, *gold_rows]:
        prediction = prediction_map[key(reference_row)]
        if prediction.get("reference") != reference_row.get("response"):
            raise RuntimeError(f"reference mismatch: {key(reference_row)}")
        if prediction.get("origin") != reference_row.get("origin"):
            raise RuntimeError(f"origin mismatch: {key(reference_row)}")
    if is_b200_config(config_dir):
        for split, references, contract in (
            ("full_test", full_rows, full_contract),
            ("human_gold", gold_rows, gold_contract),
        ):
            for position, reference_row in enumerate(references):
                prediction = prediction_map[key(reference_row)]
                identity = {
                    "split": split,
                    "position": position,
                    "id": str(reference_row["id"]),
                    "task": str(reference_row["task"]),
                    "dataset_sha256": contract["sha256"],
                    "row_sha256": canonical_sha256(reference_row),
                }
                if prediction.get("evaluation_split") != split:
                    raise RuntimeError("B200 prediction split identity changed")
                if prediction.get("evaluation_row_identity") != identity:
                    raise RuntimeError("B200 prediction row identity changed")
    write_jsonl(full_output, [prediction_map[item] for item in full_keys])
    write_jsonl(gold_output, [prediction_map[item] for item in gold_keys])
    prediction_seal = model_audit["prediction_seal"]
    receipt = {
        "schema": SPLIT_RECEIPT_SCHEMA,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "PASS",
        "model_label": args.model_label,
        "audit": file_identity(audit_path),
        "combined_predictions": {
            **file_identity(combined_path),
            "inference_manifest": {
                "path": prediction_seal["manifest_path"],
                "sha256": prediction_seal["manifest_sha256"],
            },
            "inference_contract_sha256": prediction_seal["contract_sha256"],
        },
        "datasets": {
            "full_test": full_contract,
            "human_gold": gold_contract,
        },
        "outputs": {
            "full_test": {
                **file_identity(full_output),
                "rows": len(full_keys),
                "by_task": full_contract["by_task"],
            },
            "human_gold": {
                **file_identity(gold_output),
                "rows": len(gold_keys),
                "by_task": gold_contract["by_task"],
            },
        },
    }
    if is_b200_config(config_dir):
        receipt["profile"] = "b200"
        receipt["b200_config"] = b200_config_identity(config_dir)
    atomic_json(receipt_path, receipt)
    validate_split_receipt(
        receipt_path,
        expected_audit_path=audit_path,
        expected_model_label=args.model_label,
        config_dir=config_dir,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "model_label": args.model_label,
                "full_test": len(full_keys),
                "human_gold": len(gold_keys),
                "receipt": str(receipt_path),
                "contracts": {
                    "full_test": full_contract,
                    "human_gold": gold_contract,
                },
            }
        )
    )


def validate_split_receipt(
    path: Path,
    *,
    expected_audit_path: Path | None = None,
    expected_model_label: str | None = None,
    config_dir: Path = CONFIG_DIR,
) -> dict[str, Any]:
    config_dir = resolve_config_dir(config_dir)
    path = selected_owned_path(path, config_dir=config_dir, must_exist=True)
    receipt = read_json(path)
    if (
        receipt.get("schema") != SPLIT_RECEIPT_SCHEMA
        or receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != "PASS"
    ):
        raise RuntimeError(f"invalid split receipt: {path}")
    if is_b200_config(config_dir) and (
        receipt.get("profile") != "b200"
        or receipt.get("b200_config") != b200_config_identity(config_dir)
    ):
        raise RuntimeError("split receipt B200 config identity changed")
    model_label = str(receipt.get("model_label") or "")
    if expected_model_label is not None and model_label != expected_model_label:
        raise RuntimeError("split receipt model label changed")
    audit_path = require_file_identity(
        receipt.get("audit"),
        label="split audit",
        expected_path=expected_audit_path,
        config_dir=config_dir,
    )
    audit = read_passing_audit(audit_path, config_dir=config_dir)
    combined_identity = receipt.get("combined_predictions")
    combined_path = require_file_identity(
        combined_identity,
        label="combined predictions",
        config_dir=config_dir,
    )
    model_audit = audited_model(
        audit,
        model_label=model_label,
        prediction_path=combined_path,
        config_dir=config_dir,
    )
    prediction_seal = model_audit["prediction_seal"]
    if (
        not isinstance(combined_identity, dict)
        or combined_identity.get("inference_contract_sha256")
        != prediction_seal["contract_sha256"]
    ):
        raise RuntimeError("split inference contract changed")
    require_file_identity(
        combined_identity.get("inference_manifest"),
        label="split inference manifest",
        expected_path=Path(prediction_seal["manifest_path"]).resolve(),
        config_dir=config_dir,
    )

    data_contract = read_json(config_dir / "data_contract.json")
    combined_rows = read_jsonl(combined_path)
    combined_map = {key(row): row for row in combined_rows}
    if len(combined_map) != len(combined_rows):
        raise RuntimeError("split receipt combined predictions have duplicate keys")
    for role in DATASETS:
        dataset_entry = data_contract["files"][role]
        dataset_path = (
            Path(data_contract["prepared_data_dir"]) / dataset_entry["name"]
        ).resolve()
        reference_rows = read_jsonl(dataset_path)
        contract = validate_eval_contract(
            dataset_path,
            role=role,
            rows=reference_rows,
            config_dir=config_dir,
        )
        if receipt.get("datasets", {}).get(role) != contract:
            raise RuntimeError(f"split receipt {role} data contract changed")
        output_identity = receipt.get("outputs", {}).get(role)
        output_path = require_file_identity(
            output_identity,
            label=f"{role} split predictions",
            config_dir=config_dir,
        )
        output_rows = read_jsonl(output_path)
        expected_rows = [combined_map[key(row)] for row in reference_rows]
        if output_rows != expected_rows:
            raise RuntimeError(f"{role} split is not the exact audited subset")
        if is_b200_config(config_dir):
            for position, (reference_row, prediction) in enumerate(
                zip(reference_rows, output_rows, strict=True)
            ):
                identity = {
                    "split": role,
                    "position": position,
                    "id": str(reference_row["id"]),
                    "task": str(reference_row["task"]),
                    "dataset_sha256": contract["sha256"],
                    "row_sha256": canonical_sha256(reference_row),
                }
                if prediction.get("evaluation_split") != role:
                    raise RuntimeError("B200 split label changed")
                if prediction.get("evaluation_row_identity") != identity:
                    raise RuntimeError("B200 split row identity changed")
        wanted_output = {
            **file_identity(output_path),
            "rows": len(reference_rows),
            "by_task": contract["by_task"],
        }
        if output_identity != wanted_output:
            raise RuntimeError(f"{role} split receipt identity changed")
    return receipt


def validate_scorer_outputs(
    *,
    prediction_paths: list[Path],
    summary_path: Path,
    rows_path: Path,
    expected_model_labels: list[str],
    config_dir: Path = CONFIG_DIR,
) -> dict[str, Any]:
    if len(prediction_paths) != 2 or len(set(expected_model_labels)) != 2:
        raise RuntimeError("exact scorer validation requires base and tuned inputs")
    expected_rows = [
        row
        for prediction_path in prediction_paths
        for row in read_jsonl(prediction_path)
    ]
    metric_rows = read_jsonl(rows_path)
    if len(metric_rows) != len(expected_rows):
        raise RuntimeError("scorer row count differs from the sealed predictions")
    recipe = read_json(config_dir / "recipe.json")["evaluation"]
    bleu_name = str(recipe["bleu_tokenizer"])
    bert_model = str(recipe["bert_model"])
    row_metric_ranges = {
        "sentence_bleu": (0.0, 100.0),
        "rougeL_f1": (0.0, 1.0),
        "bertscore_precision": (-1.0, 1.0),
        "bertscore_recall": (-1.0, 1.0),
        "bertscore_f1": (-1.0, 1.0),
    }

    def strict_metric(value: Any, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"invalid {label}")
        result = float(value)
        if not math.isfinite(result):
            raise RuntimeError(f"invalid {label}")
        return result

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for index, (expected, observed) in enumerate(
        zip(expected_rows, metric_rows),
        1,
    ):
        if set(observed) != set(expected) | set(METRIC_ROW_FIELDS):
            raise RuntimeError(f"scorer row fields changed at row {index}")
        for name, value in expected.items():
            if observed.get(name) != value:
                raise RuntimeError(
                    f"scorer changed prediction input field {name} at row {index}"
                )
        for field, (minimum, maximum) in row_metric_ranges.items():
            value = strict_metric(observed[field], f"{field} at scorer row {index}")
            if value < minimum - 1e-6 or value > maximum + 1e-6:
                raise RuntimeError(f"invalid {field} at scorer row {index}")
        group_key = (str(observed["model_label"]), str(observed["task"]))
        groups.setdefault(group_key, []).append(observed)

    wanted_group_keys = {
        (model_label, task) for model_label in expected_model_labels for task in TASKS
    }
    if set(groups) != wanted_group_keys:
        raise RuntimeError("scorer model/task groups changed")
    summary = read_json(summary_path)
    if set(summary) != {"bleu_tokenizer", "bert_model", "by_model_task"}:
        raise RuntimeError("scorer summary fields changed")
    if summary["bleu_tokenizer"] != bleu_name or summary["bert_model"] != bert_model:
        raise RuntimeError("scorer metric configuration changed")
    by_model_task = summary.get("by_model_task")
    if not isinstance(by_model_task, dict):
        raise RuntimeError("scorer summary groups are missing")
    if set(by_model_task) != {f"{model}/{task}" for model, task in wanted_group_keys}:
        raise RuntimeError("scorer summary model/task keys changed")
    summary_metric_ranges = {
        "corpus_bleu": (0.0, 100.0),
        "sentence_bleu_mean": (0.0, 100.0),
        "rougeL_f1_mean": (0.0, 1.0),
        "bertscore_f1_mean": (-1.0, 1.0),
    }
    for (model_label, task), group in groups.items():
        values = by_model_task[f"{model_label}/{task}"]
        if set(values) != {"count", *METRICS}:
            raise RuntimeError(f"scorer summary fields changed: {model_label}/{task}")
        if type(values["count"]) is not int or values["count"] != len(group):
            raise RuntimeError(f"scorer count changed: {model_label}/{task}")
        for metric, (minimum, maximum) in summary_metric_ranges.items():
            value = strict_metric(values[metric], f"{model_label}/{task}/{metric}")
            if value < minimum - 1e-6 or value > maximum + 1e-6:
                raise RuntimeError(f"invalid {model_label}/{task}/{metric}")
    return summary


def resolve_metric_inputs(
    *,
    dataset: str,
    audit_path: Path,
    split_receipt_paths: list[Path],
    config_dir: Path = CONFIG_DIR,
) -> dict[str, Any]:
    if dataset not in DATASETS:
        raise RuntimeError(f"unsupported dataset role: {dataset}")
    config_dir = resolve_config_dir(config_dir)
    audit_path = selected_owned_path(audit_path, config_dir=config_dir, must_exist=True)
    audit = read_passing_audit(audit_path, config_dir=config_dir)
    split_receipt_paths = [
        selected_owned_path(path, config_dir=config_dir, must_exist=True)
        for path in split_receipt_paths
    ]
    if len(split_receipt_paths) != 2:
        raise RuntimeError("exactly two split receipts are required")
    split_receipts = [
        validate_split_receipt(
            path,
            expected_audit_path=audit_path,
            config_dir=config_dir,
        )
        for path in split_receipt_paths
    ]
    labels = [str(receipt["model_label"]) for receipt in split_receipts]
    audited_labels = {str(model["model_label"]) for model in audit["models"]}
    if len(set(labels)) != 2 or set(labels) != audited_labels:
        raise RuntimeError("metric split labels differ from the clean audit")
    audit_by_label = {str(model["model_label"]): model for model in audit["models"]}
    model_roles = [
        (
            "base"
            if audit_by_label[label]["prediction_seal"].get("adapter") is None
            else "tuned"
        )
        for label in labels
    ]
    if model_roles != ["base", "tuned"]:
        raise RuntimeError("metric inputs must be ordered as base then tuned")
    prediction_paths = [
        require_file_identity(
            receipt["outputs"][dataset],
            label=f"{dataset}/{receipt['model_label']} split",
            config_dir=config_dir,
        )
        for receipt in split_receipts
    ]
    return {
        "audit": audit,
        "audit_path": audit_path,
        "split_receipt_paths": split_receipt_paths,
        "split_receipts": split_receipts,
        "labels": labels,
        "model_roles": model_roles,
        "prediction_paths": prediction_paths,
    }


def scorer_execution_identity(
    *,
    prediction_paths: list[Path],
    model_labels: list[str],
    model_roles: list[str],
    out_dir: Path,
    config_dir: Path = CONFIG_DIR,
) -> dict[str, Any]:
    recipe = read_json(config_dir / "recipe.json")["evaluation"]
    evaluator = selected_evaluator(config_dir)
    return {
        "method": "external_project_evaluation_py_score",
        "command": [
            str(Path(sys.executable).resolve()),
            str(evaluator),
            "score",
            "--prediction_jsonl",
            *(str(path) for path in prediction_paths),
            "--out_dir",
            str(out_dir),
            "--bert_model",
            str(recipe["bert_model"]),
            "--bert_batch_size",
            str(recipe["bert_batch_size"]),
            "--device",
            "cuda",
            "--bleu_tokenizer",
            str(recipe["bleu_tokenizer"]),
        ],
        "reference_evaluator": {
            "path": str(evaluator),
            "sha256": REFERENCE_SHA256,
        },
        "ordered_prediction_jsonl": [
            {
                **file_identity(path),
                "model_label": label,
                "model_role": role,
            }
            for path, label, role in zip(
                prediction_paths,
                model_labels,
                model_roles,
            )
        ],
        "arguments": {
            "out_dir": str(out_dir),
            "bert_model": recipe["bert_model"],
            "bert_batch_size": recipe["bert_batch_size"],
            "device": "cuda",
            "bleu_tokenizer": recipe["bleu_tokenizer"],
        },
        "expected_outputs": {
            "summary": str(out_dir / "metrics.summary.json"),
            "rows": str(out_dir / "metrics.rows.jsonl"),
        },
    }


def seal_metrics(
    args: argparse.Namespace,
) -> None:
    config_dir = resolve_config_dir(getattr(args, "config_dir", CONFIG_DIR))
    inputs = resolve_metric_inputs(
        dataset=args.dataset,
        audit_path=args.audit_json,
        split_receipt_paths=args.split_receipt,
        config_dir=config_dir,
    )
    audit_path = inputs["audit_path"]
    split_receipt_paths = inputs["split_receipt_paths"]
    split_receipts = inputs["split_receipts"]
    labels = inputs["labels"]
    model_roles = inputs["model_roles"]
    prediction_paths = inputs["prediction_paths"]
    summary_path = selected_owned_path(
        args.summary, config_dir=config_dir, must_exist=True
    )
    rows_path = selected_owned_path(args.rows, config_dir=config_dir, must_exist=True)
    if (
        summary_path.name != "metrics.summary.json"
        or rows_path.name != "metrics.rows.jsonl"
        or summary_path.parent != rows_path.parent
    ):
        raise RuntimeError("evaluation.py scorer output paths changed")
    summary = validate_scorer_outputs(
        prediction_paths=prediction_paths,
        summary_path=summary_path,
        rows_path=rows_path,
        expected_model_labels=labels,
        config_dir=config_dir,
    )
    scorer_execution = scorer_execution_identity(
        prediction_paths=prediction_paths,
        model_labels=labels,
        model_roles=model_roles,
        out_dir=summary_path.parent,
        config_dir=config_dir,
    )
    recipe = read_json(config_dir / "recipe.json")["evaluation"]
    receipt_path = selected_owned_path(
        args.receipt, config_dir=config_dir, create_parent=True
    )
    evaluator = selected_evaluator(config_dir)
    receipt = {
        "schema": METRICS_RECEIPT_SCHEMA,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "PASS",
        "dataset": args.dataset,
        "audit": file_identity(audit_path),
        "reference_evaluator": {
            "path": str(evaluator),
            "sha256": REFERENCE_SHA256,
        },
        "metric_contract": {
            "bleu_tokenizer": recipe["bleu_tokenizer"],
            "bert_model": recipe["bert_model"],
            "bert_batch_size": recipe["bert_batch_size"],
            "device": "cuda",
            "metrics": list(METRICS),
        },
        "scorer_execution": scorer_execution,
        "split_receipts": [
            {
                **file_identity(path),
                "model_label": split_receipt["model_label"],
                "model_role": role,
            }
            for path, split_receipt, role in zip(
                split_receipt_paths,
                split_receipts,
                model_roles,
            )
        ],
        "prediction_inputs": [
            {
                **file_identity(path),
                "model_label": label,
                "model_role": role,
            }
            for path, label, role in zip(
                prediction_paths,
                labels,
                model_roles,
            )
        ],
        "scorer_outputs": {
            "summary": file_identity(summary_path),
            "rows": file_identity(rows_path),
        },
        "summary_groups": sorted(summary["by_model_task"]),
    }
    if is_b200_config(config_dir):
        receipt["profile"] = "b200"
        receipt["b200_config"] = b200_config_identity(config_dir)
    atomic_json(receipt_path, receipt)
    validate_metrics_receipt(
        receipt_path,
        expected_dataset=args.dataset,
        expected_audit_path=audit_path,
        expected_model_labels=labels,
        config_dir=config_dir,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "dataset": args.dataset,
                "receipt": str(receipt_path),
            }
        )
    )


def validate_metrics_receipt(
    path: Path,
    *,
    expected_dataset: str,
    expected_audit_path: Path,
    expected_model_labels: list[str],
    expected_summary_path: Path | None = None,
    config_dir: Path = CONFIG_DIR,
) -> dict[str, Any]:
    config_dir = resolve_config_dir(config_dir)
    path = selected_owned_path(path, config_dir=config_dir, must_exist=True)
    receipt = read_json(path)
    if (
        receipt.get("schema") != METRICS_RECEIPT_SCHEMA
        or receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != "PASS"
        or receipt.get("dataset") != expected_dataset
    ):
        raise RuntimeError(f"invalid exact-metrics receipt: {path}")
    if is_b200_config(config_dir) and (
        receipt.get("profile") != "b200"
        or receipt.get("b200_config") != b200_config_identity(config_dir)
    ):
        raise RuntimeError("metrics receipt B200 config identity changed")
    audit_path = require_file_identity(
        receipt.get("audit"),
        label=f"{expected_dataset} metrics audit",
        expected_path=expected_audit_path,
        config_dir=config_dir,
    )
    audit = read_passing_audit(audit_path, config_dir=config_dir)
    if {str(model["model_label"]) for model in audit["models"]} != set(
        expected_model_labels
    ):
        raise RuntimeError("metrics receipt model labels differ from audit")
    evaluator = receipt.get("reference_evaluator")
    evaluator_path = selected_evaluator(config_dir)
    if (
        not isinstance(evaluator, dict)
        or evaluator.get("path") != str(evaluator_path)
        or evaluator.get("sha256") != REFERENCE_SHA256
        or sha256(evaluator_path) != REFERENCE_SHA256
    ):
        raise RuntimeError("metrics receipt evaluator identity changed")
    recipe = read_json(config_dir / "recipe.json")["evaluation"]
    wanted_metric_contract = {
        "bleu_tokenizer": recipe["bleu_tokenizer"],
        "bert_model": recipe["bert_model"],
        "bert_batch_size": recipe["bert_batch_size"],
        "device": "cuda",
        "metrics": list(METRICS),
    }
    if receipt.get("metric_contract") != wanted_metric_contract:
        raise RuntimeError("metrics receipt metric contract changed")

    split_identities = receipt.get("split_receipts")
    if not isinstance(split_identities, list) or len(split_identities) != 2:
        raise RuntimeError("metrics receipt split identities changed")
    split_receipts = []
    for position, identity in enumerate(split_identities):
        if identity.get("model_role") != ("base", "tuned")[position]:
            raise RuntimeError("metrics receipt split role/order changed")
        split_path = require_file_identity(
            identity,
            label=f"{expected_dataset} split receipt",
            config_dir=config_dir,
        )
        split_receipt = validate_split_receipt(
            split_path,
            expected_audit_path=audit_path,
            expected_model_label=str(identity.get("model_label") or ""),
            config_dir=config_dir,
        )
        split_receipts.append(split_receipt)
    labels = [str(item["model_label"]) for item in split_receipts]
    if labels != expected_model_labels or len(set(labels)) != 2:
        raise RuntimeError("metrics receipt split order/labels changed")
    prediction_paths = [
        require_file_identity(
            split_receipt["outputs"][expected_dataset],
            label=f"{expected_dataset}/{label} split input",
            config_dir=config_dir,
        )
        for split_receipt, label in zip(split_receipts, labels)
    ]
    prediction_identities = receipt.get("prediction_inputs")
    if not isinstance(prediction_identities, list) or len(prediction_identities) != 2:
        raise RuntimeError("metrics receipt prediction identities changed")
    for identity, prediction_path, label in zip(
        prediction_identities,
        prediction_paths,
        labels,
    ):
        if identity.get("model_label") != label:
            raise RuntimeError("metrics receipt prediction label changed")
        position = labels.index(label)
        if identity.get("model_role") != ("base", "tuned")[position]:
            raise RuntimeError("metrics receipt prediction role changed")
        require_file_identity(
            identity,
            label=f"{expected_dataset}/{label} metric input",
            expected_path=prediction_path,
            config_dir=config_dir,
        )
    outputs = receipt.get("scorer_outputs")
    if not isinstance(outputs, dict):
        raise RuntimeError("metrics receipt scorer outputs are missing")
    summary_path = require_file_identity(
        outputs.get("summary"),
        label=f"{expected_dataset} scorer summary",
        expected_path=expected_summary_path,
        config_dir=config_dir,
    )
    rows_path = require_file_identity(
        outputs.get("rows"),
        label=f"{expected_dataset} scorer rows",
        config_dir=config_dir,
    )
    wanted_execution = scorer_execution_identity(
        prediction_paths=prediction_paths,
        model_labels=labels,
        model_roles=["base", "tuned"],
        out_dir=summary_path.parent,
        config_dir=config_dir,
    )
    if receipt.get("scorer_execution") != wanted_execution:
        raise RuntimeError("metrics receipt scorer execution changed")
    summary = validate_scorer_outputs(
        prediction_paths=prediction_paths,
        summary_path=summary_path,
        rows_path=rows_path,
        expected_model_labels=labels,
        config_dir=config_dir,
    )
    if receipt.get("summary_groups") != sorted(summary["by_model_task"]):
        raise RuntimeError("metrics receipt summary groups changed")
    return {
        "receipt": receipt,
        "receipt_path": path,
        "summary": summary,
        "summary_path": summary_path,
        "rows_path": rows_path,
        "split_receipts": split_receipts,
        "prediction_paths": prediction_paths,
    }


def task_stats(summary: dict[str, Any], model_label: str, task: str) -> dict[str, Any]:
    values = summary["by_model_task"][f"{model_label}/{task}"]
    return {
        "count": int(values["count"]),
        **{metric: float(values[metric]) for metric in METRICS},
    }


def delta(tuned: dict[str, Any], base: dict[str, Any]) -> dict[str, float]:
    if tuned["count"] != base["count"]:
        raise RuntimeError("base and tuned counts differ")
    return {metric: tuned[metric] - base[metric] for metric in METRICS}


def _validate_b200_training_selection(
    training: dict[str, Any],
    *,
    recipe: dict[str, Any],
    config_identity: dict[str, Any],
) -> dict[str, Any]:
    final_step = 9_316
    total_occurrences = 149_048
    if training.get("status") != "COMPLETED":
        raise RuntimeError("B200 training has no exact terminal completion")
    for field, expected in (
        ("profile", "b200"),
        ("config_identity", config_identity),
        ("config", recipe),
        ("split_roles", EXPECTED_SPLIT_ROLES),
        ("completed_updates", final_step),
        ("target_updates", final_step),
        ("consumed_occurrences", total_occurrences),
    ):
        if training.get(field) != expected:
            raise RuntimeError(f"B200 training {field} changed")
    if recipe.get("epochs") != 1 or recipe.get("optimizer_steps") != final_step:
        raise RuntimeError("B200 one-epoch optimizer horizon changed")
    tail = recipe.get("exact_tail")
    if not isinstance(tail, dict) or any(
        tail.get(field) != expected
        for field, expected in (
            ("total_occurrences", total_occurrences),
            ("full_steps", 9_315),
            ("full_step_occurrences", 149_040),
            ("final_step", final_step),
            ("final_step_occurrences", 8),
            ("padding_or_duplication_allowed", False),
        )
    ):
        raise RuntimeError("B200 exact-tail recipe changed")
    completion = training.get("training_completion")
    wanted_completion = {
        "status": "COMPLETED",
        "epochs": 1,
        "completed_epochs": 1.0,
        "early_stopping_triggered": False,
        "optimizer_steps": final_step,
        "consumed_occurrences": total_occurrences,
        "expected_optimizer_steps": final_step,
        "expected_occurrences": total_occurrences,
    }
    if completion != wanted_completion:
        raise RuntimeError("B200 training completion record changed")
    coverage = training.get("training_coverage")
    if not isinstance(coverage, dict):
        raise RuntimeError("B200 training coverage is missing")
    for field, expected in (
        ("total_occurrences", total_occurrences),
        ("skipped_occurrences", 0),
        ("unconsumed_planned_occurrences", 0),
        ("unique_raw_rows", 142_713),
        (
            "task_occurrences",
            {"translation": 102_224, "summarization": 46_824},
        ),
    ):
        if coverage.get(field) != expected:
            raise RuntimeError(f"B200 training coverage {field} changed")
    quality = coverage.get("quality_occurrences")
    if (
        not isinstance(quality, dict)
        or not quality
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in quality.values()
        )
        or sum(quality.values()) != total_occurrences
    ):
        raise RuntimeError("B200 quality occurrence coverage changed")

    validation = recipe["validation"]
    if validation.get("full_validation_steps") != [4_658, final_step]:
        raise RuntimeError("B200 full-validation schedule changed")
    early_policy = validation.get("early_stopping")
    if not isinstance(early_policy, dict) or any(
        early_policy.get(field) != expected
        for field, expected in (
            ("enabled", False),
            ("earliest_terminal_step", final_step),
            ("latest_terminal_step", final_step),
            ("sealed_evaluation_not_monitor", True),
        )
    ):
        raise RuntimeError("B200 disabled early-stopping policy changed")
    history = training.get("history")
    if not isinstance(history, list):
        raise RuntimeError("B200 validation history is missing")
    expected_schedule = [
        (0, "fixed_subset"),
        (1_000, "fixed_subset"),
        (2_000, "fixed_subset"),
        (3_000, "fixed_subset"),
        (4_000, "fixed_subset"),
        (4_658, "full"),
        (5_000, "fixed_subset"),
        (6_000, "fixed_subset"),
        (7_000, "fixed_subset"),
        (8_000, "fixed_subset"),
        (9_000, "fixed_subset"),
        (final_step, "full"),
    ]
    actual_schedule = [
        (item.get("step"), item.get("scope"))
        for item in history
        if isinstance(item, dict)
    ]
    if len(actual_schedule) != len(history) or actual_schedule != expected_schedule:
        raise RuntimeError("B200 validation schedule changed")
    full_history = {
        int(item["step"]): item for item in history if item["scope"] == "full"
    }
    early = training.get("early_stopping_decision")
    if not isinstance(early, dict) or any(
        early.get(field) != expected
        for field, expected in (
            ("profile", "b200"),
            ("status", "MAXIMUM_REACHED"),
            ("evaluated_step", final_step),
            ("triggered", False),
            ("reason", "exact_epoch_coverage_reached"),
            ("quality_early_stopping_enabled", False),
            ("completed_candidate_steps", [4_658, final_step]),
            ("completed_occurrences", total_occurrences),
            ("completed_epochs", 1.0),
        )
    ):
        raise RuntimeError("B200 terminal coverage decision changed")

    policy = "argmin_full_validation_selection_loss"
    tie_break = "earlier_optimizer_step"
    record = training.get("selection_record")
    selected = training.get("selected_checkpoint")
    if not isinstance(record, dict) or not isinstance(selected, dict):
        raise RuntimeError("B200 checkpoint selection is missing")
    if any(
        record.get(field) != expected
        for field, expected in (
            ("policy", policy),
            ("tie_break", tie_break),
            ("candidate_steps", [4_658, final_step]),
        )
    ):
        raise RuntimeError("B200 selection policy changed")
    candidates = record.get("candidates")
    if not isinstance(candidates, list) or [
        item.get("step") for item in candidates if isinstance(item, dict)
    ] != [4_658, final_step]:
        raise RuntimeError("B200 selection candidates changed")
    weights = validation["selection_weights"]
    normalized: list[tuple[float, int, dict[str, Any]]] = []
    for candidate in candidates:
        step = int(candidate["step"])
        history_item = full_history.get(step)
        if history_item is None:
            raise RuntimeError("B200 candidate lacks full validation history")
        for field in (
            "selection_loss",
            "translation_loss",
            "summarization_loss",
        ):
            value = candidate.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
                or value != history_item.get(field)
            ):
                raise RuntimeError("B200 candidate validation loss changed")
        recomputed = float(weights["translation"]) * float(
            candidate["translation_loss"]
        ) + float(weights["summarization"]) * float(candidate["summarization_loss"])
        if not math.isclose(
            float(candidate["selection_loss"]),
            recomputed,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("B200 candidate weighted loss changed")
        normalized.append((float(candidate["selection_loss"]), step, candidate))
    selected_loss, selected_step, selected_candidate = min(normalized)
    if (
        record.get("selected_step") != selected_step
        or record.get("selected_selection_loss") != selected_loss
        or selected.get("policy") != policy
        or selected.get("tie_break") != tie_break
        or selected.get("step") != selected_step
        or selected.get("selection_loss") != selected_loss
        or selected.get("source_checkpoint_dir")
        != selected_candidate.get("adapter_dir")
        or Path(str(selected.get("adapter_dir") or "")).name
        != recipe["checkpoint"]["best_adapter_dirname"]
    ):
        raise RuntimeError("B200 selected checkpoint is not the validated argmin")
    final = training.get("final_checkpoint")
    if (
        not isinstance(final, dict)
        or final.get("step") != final_step
        or Path(str(final.get("adapter_dir") or "")).name
        != recipe["checkpoint"]["final_adapter_dirname"]
    ):
        raise RuntimeError("B200 final checkpoint changed")
    return {
        "training_status": "COMPLETED",
        "final_step": final_step,
        "target_step": final_step,
        "completed_epochs": 1.0,
        "selected_step": selected_step,
        "selected_selection_loss": selected_loss,
        "candidate_steps": [4_658, final_step],
        "selection_record": record,
        "selected_checkpoint": selected,
        "training_completion": completion,
        "early_stopping_decision": early,
    }


def validate_training_selection(
    training: dict[str, Any],
    *,
    recipe: dict[str, Any] | None = None,
    config_dir: Path = CONFIG_DIR,
) -> dict[str, Any]:
    """Bind report/upload inputs to an approved terminal validation argmin."""

    config_dir = resolve_config_dir(config_dir)
    expected_recipe = recipe or read_json(config_dir / "recipe.json")
    if expected_recipe.get("profile") == "b200":
        if not is_b200_config(config_dir):
            raise RuntimeError(
                "B200 recipe requires the explicit B200 config directory"
            )
        return _validate_b200_training_selection(
            training,
            recipe=expected_recipe,
            config_identity=b200_config_identity(config_dir),
        )
    status = training.get("status")
    if status not in {"COMPLETED", "EARLY_STOPPED"}:
        raise RuntimeError("training has no approved terminal status")
    if training.get("config") != expected_recipe:
        raise RuntimeError("training manifest recipe differs from the pinned recipe")
    if training.get("split_roles") != EXPECTED_SPLIT_ROLES:
        raise RuntimeError(
            "sealed test/human roles cannot participate in model selection"
        )

    target_step = int(expected_recipe["optimizer_steps"])
    expected_occurrences = int(
        expected_recipe["training_deck"]["total_occurrences"]["total"]
    )
    occurrences_per_step = int(
        expected_recipe["microbatch_policy"]["optimizer_step_occurrences"]
    )
    occurrences_per_epoch = int(
        expected_recipe["training_deck"]["occurrences_per_epoch"]["total"]
    )
    validation = expected_recipe["validation"]
    early_policy = validation.get("early_stopping")
    if not isinstance(early_policy, dict):
        raise RuntimeError("early-stopping policy is missing")
    for key, expected in (
        ("enabled", True),
        ("monitor", "selection_loss"),
        ("mode", "min"),
        ("full_validation_confirmation_required", True),
        ("restore_best_full_validation", True),
        ("sealed_evaluation_not_monitor", True),
    ):
        if early_policy.get(key) != expected:
            raise RuntimeError(f"early-stopping policy {key} changed")

    final_step = training.get("completed_updates")
    if isinstance(final_step, bool) or not isinstance(final_step, int):
        raise RuntimeError("training completed update count is invalid")
    if status == "COMPLETED":
        wanted_final_step = target_step
    else:
        wanted_final_step = int(early_policy["confirmation_step"])
        if wanted_final_step != int(early_policy["earliest_stop_step"]):
            raise RuntimeError("approved early stop is not the confirmation boundary")
        if wanted_final_step >= target_step:
            raise RuntimeError("early stop does not precede the target horizon")
    if final_step != wanted_final_step:
        raise RuntimeError(
            f"training completion step changed: "
            f"{final_step!r} != {wanted_final_step}"
        )
    if training.get("target_updates") != target_step:
        raise RuntimeError("training target update count changed")
    consumed_occurrences = final_step * occurrences_per_step
    if training.get("consumed_occurrences") != consumed_occurrences:
        raise RuntimeError("training occurrence completion changed")
    completed_epochs = consumed_occurrences / occurrences_per_epoch
    if status == "EARLY_STOPPED" and completed_epochs < float(
        early_policy["minimum_epochs"]
    ):
        raise RuntimeError("early-stopped run did not complete the minimum epoch")

    configured_candidate_steps = [
        int(step) for step in expected_recipe["validation"]["full_validation_steps"]
    ]
    candidate_metadata_steps = [
        int(item["step"])
        for item in expected_recipe["validation"]["full_validation_candidates"]
    ]
    if (
        not configured_candidate_steps
        or configured_candidate_steps != sorted(set(configured_candidate_steps))
        or configured_candidate_steps != candidate_metadata_steps
        or configured_candidate_steps[-1] != target_step
    ):
        raise RuntimeError("recipe full-validation candidate schedule is invalid")
    candidate_steps = [
        step for step in configured_candidate_steps if step <= final_step
    ]
    if not candidate_steps or candidate_steps[-1] != final_step:
        raise RuntimeError("terminal step is not a completed full-validation candidate")

    completion = training.get("training_completion")
    if not isinstance(completion, dict):
        raise RuntimeError("training completion record is missing")
    if set(completion) != {
        "status",
        "epochs",
        "completed_epochs",
        "optimizer_steps",
        "consumed_occurrences",
        "expected_optimizer_steps",
        "expected_occurrences",
        "early_stopping_triggered",
    }:
        raise RuntimeError("training completion record changed")
    for key, expected in (
        ("status", status),
        ("epochs", expected_recipe["epochs"]),
        ("completed_epochs", completed_epochs),
        ("optimizer_steps", final_step),
        ("consumed_occurrences", consumed_occurrences),
        ("expected_optimizer_steps", target_step),
        ("expected_occurrences", expected_occurrences),
        ("early_stopping_triggered", status == "EARLY_STOPPED"),
    ):
        if completion.get(key) != expected:
            raise RuntimeError(f"training completion {key} changed")
    recorded_epochs = completion.get("completed_epochs")
    if (
        isinstance(recorded_epochs, bool)
        or not isinstance(recorded_epochs, (int, float))
        or not math.isfinite(float(recorded_epochs))
        or not math.isclose(
            float(recorded_epochs),
            completed_epochs,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError("training completion completed epochs changed")
    coverage = training.get("training_coverage")
    if not isinstance(coverage, dict) or set(coverage) != {
        "total_occurrences",
        "skipped_occurrences",
        "unconsumed_planned_occurrences",
        "unique_raw_rows",
        "task_occurrences",
        "quality_occurrences",
    }:
        raise RuntimeError("training coverage record changed")
    for key, expected in (
        ("total_occurrences", consumed_occurrences),
        ("skipped_occurrences", 0),
        (
            "unconsumed_planned_occurrences",
            expected_occurrences - consumed_occurrences,
        ),
        (
            "unique_raw_rows",
            expected_recipe["training_deck"]["base_unique_rows"],
        ),
    ):
        if coverage.get(key) != expected:
            raise RuntimeError(f"training coverage {key} changed")
    task_occurrences = coverage.get("task_occurrences")
    quality_occurrences = coverage.get("quality_occurrences")
    if (
        not isinstance(task_occurrences, dict)
        or set(task_occurrences) != set(TASKS)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in task_occurrences.values()
        )
        or sum(task_occurrences.values()) != consumed_occurrences
    ):
        raise RuntimeError("training task occurrence coverage changed")
    if (
        not isinstance(quality_occurrences, dict)
        or not quality_occurrences
        or any(
            not isinstance(key, str)
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
            for key, value in quality_occurrences.items()
        )
        or sum(quality_occurrences.values()) != consumed_occurrences
    ):
        raise RuntimeError("training quality occurrence coverage changed")

    policy = validation["selection_policy"]
    tie_break = validation["selection_tie_break"]
    if (
        policy != "argmin_full_validation_selection_loss"
        or tie_break != "earlier_optimizer_step"
    ):
        raise RuntimeError("validation-best selection recipe changed")
    weights = validation["selection_weights"]
    translation_weight = float(weights["translation"])
    summarization_weight = float(weights["summarization"])

    history = training.get("history")
    if not isinstance(history, list):
        raise RuntimeError("training validation history is missing")
    every = int(validation["eval_every_steps"])
    scheduled_steps = set(range(every, final_step + 1, every)) | set(candidate_steps)
    expected_schedule = [
        (
            step,
            "full" if step in candidate_steps else "fixed_subset",
        )
        for step in sorted(scheduled_steps)
    ]
    actual_schedule: list[tuple[int, str]] = []
    for item in history:
        if not isinstance(item, dict):
            raise RuntimeError("training validation history record is invalid")
        step = item.get("step")
        scope = item.get("scope")
        if isinstance(step, bool) or not isinstance(step, int):
            raise RuntimeError("training validation history step is invalid")
        if scope not in {"fixed_subset", "full"}:
            raise RuntimeError("test/human data cannot be an early-stopping monitor")
        actual_schedule.append((step, scope))
    if actual_schedule != expected_schedule:
        raise RuntimeError("training validation schedule changed")

    expected_early_stopping = early_stopping_decision(
        history,
        completed_step=final_step,
        policy=early_policy,
        optimizer_steps=target_step,
        full_validation_steps=configured_candidate_steps,
        occurrences_per_step=occurrences_per_step,
        occurrences_per_epoch=occurrences_per_epoch,
    )
    early_stopping = training.get("early_stopping_decision")
    if (
        not isinstance(early_stopping, dict)
        or early_stopping != expected_early_stopping
    ):
        raise RuntimeError("early-stopping evidence changed")
    if completion.get("early_stopping_triggered") is not bool(
        early_stopping["triggered"]
    ):
        raise RuntimeError("training completion stopping trigger changed")
    if early_stopping.get("evaluated_step") != final_step:
        raise RuntimeError("early-stopping evidence has the wrong terminal step")
    if early_stopping.get("completed_candidate_steps") != candidate_steps:
        raise RuntimeError("early-stopping evidence has the wrong completed candidates")
    evidence_epochs = early_stopping.get("completed_epochs")
    if (
        isinstance(evidence_epochs, bool)
        or not isinstance(evidence_epochs, (int, float))
        or not math.isclose(
            float(evidence_epochs),
            completed_epochs,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError("early-stopping evidence has wrong epoch coverage")
    if status == "EARLY_STOPPED":
        if (
            early_stopping.get("status") != "TRIGGERED"
            or early_stopping.get("triggered") is not True
        ):
            raise RuntimeError("early-stopped run lacks a policy trigger")
    elif (
        early_stopping.get("status") != "MAXIMUM_REACHED"
        or early_stopping.get("triggered") is not False
    ):
        raise RuntimeError("completed run has an invalid stopping decision")

    full_history: dict[int, dict[str, Any]] = {}
    for item in history:
        if not isinstance(item, dict) or item.get("scope") != "full":
            continue
        step = item.get("step")
        if isinstance(step, bool) or not isinstance(step, int):
            raise RuntimeError("full-validation history step is invalid")
        if step in full_history:
            raise RuntimeError(f"duplicate full-validation history step: {step}")
        full_history[step] = item
    if sorted(full_history) != candidate_steps:
        raise RuntimeError(
            "full-validation history does not cover the candidate schedule"
        )

    record = training.get("selection_record")
    selected = training.get("selected_checkpoint")
    if not isinstance(record, dict) or not isinstance(selected, dict):
        raise RuntimeError("training checkpoint selection record is missing")
    if (
        record.get("policy") != policy
        or record.get("tie_break") != tie_break
        or record.get("candidate_steps") != candidate_steps
    ):
        raise RuntimeError("selection record policy or candidates changed")
    candidates = record.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(candidate_steps):
        raise RuntimeError("selection record candidate list changed")

    normalized_candidates: list[dict[str, Any]] = []
    for expected_step, candidate in zip(candidate_steps, candidates, strict=True):
        if not isinstance(candidate, dict) or candidate.get("step") != expected_step:
            raise RuntimeError("selection candidate order or step changed")
        history_item = full_history[expected_step]
        normalized: dict[str, Any] = {"step": expected_step}
        for key in (
            "selection_loss",
            "translation_loss",
            "summarization_loss",
        ):
            value = candidate.get(key)
            history_value = history_item.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise RuntimeError(
                    f"selection candidate {expected_step} has invalid {key}"
                )
            if (
                isinstance(history_value, bool)
                or not isinstance(history_value, (int, float))
                or not math.isclose(
                    float(value),
                    float(history_value),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise RuntimeError(
                    f"selection candidate/history mismatch at {expected_step}: {key}"
                )
            normalized[key] = float(value)
        recomputed = (
            translation_weight * normalized["translation_loss"]
            + summarization_weight * normalized["summarization_loss"]
        )
        if not math.isclose(
            normalized["selection_loss"],
            recomputed,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"selection loss does not match task weights at {expected_step}"
            )
        adapter_dir = candidate.get("adapter_dir")
        if not isinstance(adapter_dir, str) or not adapter_dir:
            raise RuntimeError(
                f"selection candidate {expected_step} adapter path is missing"
            )
        normalized["adapter_dir"] = adapter_dir
        normalized_candidates.append(normalized)

    expected_selected = min(
        normalized_candidates,
        key=lambda item: (item["selection_loss"], item["step"]),
    )
    selected_step = expected_selected["step"]
    selected_loss = expected_selected["selection_loss"]
    recorded_selected_loss = record.get("selected_selection_loss")
    if (
        record.get("selected_step") != selected_step
        or isinstance(recorded_selected_loss, bool)
        or not isinstance(recorded_selected_loss, (int, float))
        or not math.isclose(
            float(recorded_selected_loss),
            selected_loss,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError("selection record is not the candidate argmin")
    if selected_step not in candidate_steps:
        raise RuntimeError("selected checkpoint is not a full-validation candidate")
    selected_checkpoint_loss = selected.get("selection_loss")
    if (
        selected.get("policy") != policy
        or selected.get("tie_break") != tie_break
        or selected.get("step") != selected_step
        or isinstance(selected_checkpoint_loss, bool)
        or not isinstance(selected_checkpoint_loss, (int, float))
        or not math.isclose(
            float(selected_checkpoint_loss),
            selected_loss,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError("selected checkpoint differs from the validated argmin")
    if selected.get("source_checkpoint_dir") != expected_selected["adapter_dir"]:
        raise RuntimeError("selected checkpoint source differs from its candidate")
    best_dirname = expected_recipe["checkpoint"]["best_adapter_dirname"]
    if Path(str(selected.get("adapter_dir") or "")).name != best_dirname:
        raise RuntimeError("selected checkpoint is not the best adapter")

    final = training.get("final_checkpoint")
    if not isinstance(final, dict) or final.get("step") != final_step:
        raise RuntimeError("final checkpoint does not prove training completion")
    final_dirname = expected_recipe["checkpoint"]["final_adapter_dirname"]
    if Path(str(final.get("adapter_dir") or "")).name != final_dirname:
        raise RuntimeError("final checkpoint adapter path changed")
    return {
        "training_status": status,
        "final_step": final_step,
        "target_step": target_step,
        "completed_epochs": completed_epochs,
        "selected_step": selected_step,
        "selected_selection_loss": selected_loss,
        "candidate_steps": candidate_steps,
        "selection_record": record,
        "selected_checkpoint": selected,
        "training_completion": completion,
        "early_stopping_decision": early_stopping,
    }


def report(args: argparse.Namespace) -> None:
    config_dir = resolve_config_dir(getattr(args, "config_dir", CONFIG_DIR))
    full_summary_path = selected_owned_path(
        args.full_summary, config_dir=config_dir, must_exist=True
    )
    gold_summary_path = selected_owned_path(
        args.gold_summary, config_dir=config_dir, must_exist=True
    )
    full_receipt_path = selected_owned_path(
        args.full_metrics_receipt, config_dir=config_dir, must_exist=True
    )
    gold_receipt_path = selected_owned_path(
        args.gold_metrics_receipt, config_dir=config_dir, must_exist=True
    )
    training_path = selected_owned_path(
        args.training_manifest, config_dir=config_dir, must_exist=True
    )
    audit_path = selected_owned_path(
        args.audit_json, config_dir=config_dir, must_exist=True
    )
    output_json = selected_owned_path(
        args.output_json, config_dir=config_dir, create_parent=True
    )
    output_md = selected_owned_path(
        args.output_md, config_dir=config_dir, create_parent=True
    )
    training = read_json(training_path)
    audit = read_passing_audit(audit_path, config_dir=config_dir)
    require_file_identity(
        (audit.get("training_adapter_gate") or {}).get("training_manifest"),
        label="audited training manifest",
        expected_path=training_path,
        config_dir=config_dir,
    )
    audit_markdown_path = require_file_identity(
        audit.get("artifacts", {}).get("markdown"),
        label="audit markdown",
        config_dir=config_dir,
    )
    selection = validate_training_selection(training, config_dir=config_dir)
    labels_in_order = [args.base_label, args.tuned_label]
    full_metrics = validate_metrics_receipt(
        full_receipt_path,
        expected_dataset="full_test",
        expected_audit_path=audit_path,
        expected_model_labels=labels_in_order,
        expected_summary_path=full_summary_path,
        config_dir=config_dir,
    )
    gold_metrics = validate_metrics_receipt(
        gold_receipt_path,
        expected_dataset="human_gold",
        expected_audit_path=audit_path,
        expected_model_labels=labels_in_order,
        expected_summary_path=gold_summary_path,
        config_dir=config_dir,
    )
    full_summary = full_metrics["summary"]
    gold_summary = gold_metrics["summary"]
    summaries = {"full_test": full_summary, "human_gold": gold_summary}
    data_contract = read_json(config_dir / "data_contract.json")
    evaluator = selected_evaluator(config_dir)
    models = {}
    for label in (args.base_label, args.tuned_label):
        models[label] = {
            dataset: {
                task: task_stats(summaries[dataset], label, task) for task in TASKS
            }
            for dataset in DATASETS
        }
    deltas = {
        dataset: {
            task: delta(
                models[args.tuned_label][dataset][task],
                models[args.base_label][dataset][task],
            )
            for task in TASKS
        }
        for dataset in DATASETS
    }
    for dataset in DATASETS:
        for task in TASKS:
            expected = int(data_contract["files"][dataset]["by_task"][task])
            for label in (args.base_label, args.tuned_label):
                if models[label][dataset][task]["count"] != expected:
                    raise RuntimeError(
                        f"metric row count mismatch: {dataset}/{task}/{label}"
                    )
    result = {
        "status": "PASS",
        "model_name": training["model_name"],
        "attribution": "Built with Solar",
        "base_model_label": args.base_label,
        "tuned_model_label": args.tuned_label,
        "training_completion": selection["training_completion"],
        "early_stopping_decision": selection["early_stopping_decision"],
        "selection_record": selection["selection_record"],
        "selected_checkpoint": selection["selected_checkpoint"],
        "split_roles": training["split_roles"],
        "models": models,
        "deltas_tuned_minus_base": deltas,
        "training": {
            "status": selection["training_status"],
            "completed_updates": selection["final_step"],
            "target_updates": selection["target_step"],
            "completed_epochs": selection["completed_epochs"],
            "selected_step": selection["selected_step"],
            "selected_selection_loss": selection["selected_selection_loss"],
            "config": training["config"],
            "validation_history": training["history"],
            "trainable_parameters": training["trainable_parameters"],
            "lora_module_audit": training["lora_module_audit"],
        },
        "metric_contract": {
            "reference_evaluator": str(evaluator),
            "reference_evaluator_sha256": REFERENCE_SHA256,
            "bleu_tokenizer": full_summary["bleu_tokenizer"],
            "bert_model": full_summary["bert_model"],
            "metrics": list(METRICS),
            "raw_output_policy": "outer strip only; no output filtering or repair",
        },
        "output_only_audit": {
            "status": audit["output_cleanliness_status"],
            "models": [
                {
                    "model_label": model["model_label"],
                    "confirmed_output_failures": model["confirmed_output_failures"],
                    "output_categories": model["output_categories"],
                }
                for model in audit["models"]
            ],
            "contract": audit["output_only_contract"],
            "reference_scan": audit["reference_output_only_scan"],
        },
        "data_contract": {
            "path": str(config_dir / "data_contract.json"),
            "sha256": sha256(config_dir / "data_contract.json"),
            "full_test": data_contract["files"]["full_test"],
            "human_gold": data_contract["files"]["human_gold"],
        },
        "artifacts": {
            "full_summary": {
                "path": str(full_summary_path),
                "sha256": sha256(full_summary_path),
            },
            "gold_summary": {
                "path": str(gold_summary_path),
                "sha256": sha256(gold_summary_path),
            },
            "full_metrics_rows": file_identity(full_metrics["rows_path"]),
            "gold_metrics_rows": file_identity(gold_metrics["rows_path"]),
            "full_metrics_receipt": file_identity(full_receipt_path),
            "gold_metrics_receipt": file_identity(gold_receipt_path),
            "training_manifest": file_identity(training_path),
            "audit": file_identity(audit_path),
            "audit_markdown": file_identity(audit_markdown_path),
        },
        "provenance_chain": {
            "audit_sha256": sha256(audit_path),
            "combined_predictions": {
                model["model_label"]: model["prediction_seal"]["output_jsonl_sha256"]
                for model in audit["models"]
            },
            "split_predictions": {
                dataset: {
                    receipt["model_label"]: receipt["outputs"][dataset]["sha256"]
                    for receipt in metrics["split_receipts"]
                }
                for dataset, metrics in (
                    ("full_test", full_metrics),
                    ("human_gold", gold_metrics),
                )
            },
            "metric_receipts": {
                "full_test": sha256(full_receipt_path),
                "human_gold": sha256(gold_receipt_path),
            },
        },
    }
    if is_b200_config(config_dir):
        result["profile"] = "b200"
        result["b200_config"] = b200_config_identity(config_dir)
        if (
            audit.get("training_adapter_gate", {}).get("b200_config")
            != result["b200_config"]
        ):
            raise RuntimeError("report adapter gate B200 config changed")
    labels = {"full_test": "Full test", "human_gold": "Human gold"}
    terminal_label = (
        "early stopped"
        if selection["training_status"] == "EARLY_STOPPED"
        else "completed"
    )
    lines = [
        "# Solar Open 2 역사 번역·요약 튜닝 결과",
        "",
        "**Built with Solar**",
        "",
        f"- Base: `{args.base_label}`",
        f"- Tuned: `{args.tuned_label}`",
        "- Training terminal status: "
        f"`{selection['training_status']}` ({terminal_label} at step "
        f"`{selection['final_step']}` / target `{selection['target_step']}`, "
        f"{selection['completed_epochs']:.6f} epochs)",
        "- Selected checkpoint: validation-best step "
        f"`{selection['selected_step']}` "
        f"(selection loss {selection['selected_selection_loss']:.8f})",
        "- Test and human gold were evaluated and reported separately.",
        f"- Output-only audit: `{audit['output_cleanliness_status']}`",
        "- Reference output detector scan: "
        f"`{audit['reference_output_only_scan']['status']}` "
        "(train/validation/full test/human gold)",
        "- Metrics use raw model output with outer whitespace stripping only.",
        "- Three sealed full-test references retain their original `답변:` prefix; they are not rewritten.",
        "",
        "## 절대 성능",
        "",
        "| Dataset | Task | Model | n | Corpus BLEU | Sentence BLEU | ROUGE-L F1 | BERTScore F1 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in DATASETS:
        for task in TASKS:
            for label in (args.base_label, args.tuned_label):
                values = models[label][dataset][task]
                lines.append(
                    f"| {labels[dataset]} | {task} | {label} | "
                    f"{values['count']} | {values['corpus_bleu']:.4f} | "
                    f"{values['sentence_bleu_mean']:.4f} | "
                    f"{values['rougeL_f1_mean']:.6f} | "
                    f"{values['bertscore_f1_mean']:.6f} |"
                )
    lines.extend(
        [
            "",
            "## 학습 전 대비 변화",
            "",
            "| Dataset | Task | Delta Corpus BLEU | Delta Sentence BLEU | Delta ROUGE-L F1 | Delta BERTScore F1 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for dataset in DATASETS:
        for task in TASKS:
            values = deltas[dataset][task]
            lines.append(
                f"| {labels[dataset]} | {task} | "
                f"{values['corpus_bleu']:+.4f} | "
                f"{values['sentence_bleu_mean']:+.4f} | "
                f"{values['rougeL_f1_mean']:+.6f} | "
                f"{values['bertscore_f1_mean']:+.6f} |"
            )
    atomic_text(output_md, "\n".join(lines) + "\n")
    result["artifacts"]["report_markdown"] = file_identity(output_md)
    atomic_json(output_json, result)
    print(json.dumps({"status": "PASS", "report": str(output_json)}))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    split = commands.add_parser("split")
    split.add_argument("--config-dir", type=Path, required=True)
    split.add_argument("--combined-predictions", type=Path, required=True)
    split.add_argument("--model-label", required=True)
    split.add_argument("--audit-json", type=Path, required=True)
    split.add_argument("--full-eval", type=Path, required=True)
    split.add_argument("--human-gold-eval", type=Path, required=True)
    split.add_argument("--full-predictions", type=Path, required=True)
    split.add_argument("--gold-predictions", type=Path, required=True)
    split.add_argument("--receipt", type=Path, required=True)
    seal = commands.add_parser("seal-metrics")
    seal.add_argument("--config-dir", type=Path, required=True)
    seal.add_argument("--dataset", choices=DATASETS, required=True)
    seal.add_argument(
        "--split-receipt",
        type=Path,
        action="append",
        required=True,
    )
    seal.add_argument("--audit-json", type=Path, required=True)
    seal.add_argument("--summary", type=Path, required=True)
    seal.add_argument("--rows", type=Path, required=True)
    seal.add_argument("--receipt", type=Path, required=True)
    build = commands.add_parser("report")
    build.add_argument("--config-dir", type=Path, required=True)
    build.add_argument("--full-summary", type=Path, required=True)
    build.add_argument("--gold-summary", type=Path, required=True)
    build.add_argument("--full-metrics-receipt", type=Path, required=True)
    build.add_argument("--gold-metrics-receipt", type=Path, required=True)
    build.add_argument("--training-manifest", type=Path, required=True)
    build.add_argument("--audit-json", type=Path, required=True)
    build.add_argument("--base-label", required=True)
    build.add_argument("--tuned-label", required=True)
    build.add_argument("--output-json", type=Path, required=True)
    build.add_argument("--output-md", type=Path, required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command == "split":
        split_predictions(args)
    elif args.command == "seal-metrics":
        seal_metrics(args)
    else:
        report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
