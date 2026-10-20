#!/usr/bin/env python3
"""Validate and upload Solar base-vs-tuned training/evaluation results to W&B."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from common import (
    CONFIG_DIR,
    atomic_json,
    atomic_text,
    read_json,
    require_owned_path,
    sha256,
)
from b200_runtime import (
    B200_CONFIG_DIR,
    b200_config_identity,
    require_expected_user,
)
from finalize import (
    require_file_identity,
    resolve_config_dir,
    selected_owned_path,
    task_stats,
    validate_metrics_receipt,
    validate_training_selection,
)


METRICS = (
    "corpus_bleu",
    "sentence_bleu_mean",
    "rougeL_f1_mean",
    "bertscore_f1_mean",
)
DATASETS = ("full_test", "human_gold")
TASKS = ("translation", "summarization")
EXPECTED_PRODUCTION_TRAINING_RUN_ID = (
    "solar_open2_hardterm_2epoch_earlystop_20260727_v2"
)
EXPECTED_PRODUCTION_TRAINING_RUN_NAME = (
    "Solar-Open2-250B hard-term translation-summary 2epoch earlystop "
    "step18631 maxlen9600 v2"
)
B200_PRODUCTION_TRAINING_RUN_ID = "solar_open2_b200_1epoch_20260802_v2"
B200_PRODUCTION_TRAINING_RUN_NAME = (
    "Solar Open2 250B 2xB200 1epoch len9600 optimized v2"
)


def validate_training_wandb_identity(
    training: dict[str, Any],
    *,
    entity: str,
    project: str,
    training_run_id: str,
    expected_run_id: str | None = None,
    expected_run_name: str | None = None,
) -> dict[str, str]:
    if expected_run_id is not None and training_run_id != expected_run_id:
        raise RuntimeError(
            f"training W&B run_id changed: "
            f"{training_run_id!r} != {expected_run_id!r}"
        )
    identity = training.get("wandb")
    if not isinstance(identity, dict):
        raise RuntimeError("training manifest lacks its W&B run identity")
    wanted = {
        "mode": "online",
        "entity": entity,
        "project": project,
        "run_id": training_run_id,
        "resume": "allow",
    }
    for field, expected in wanted.items():
        if identity.get(field) != expected:
            raise RuntimeError(
                f"training W&B {field} changed: "
                f"{identity.get(field)!r} != {expected!r}"
            )
    run_name = identity.get("run_name")
    if not isinstance(run_name, str) or not run_name.strip():
        raise RuntimeError("training W&B run name is missing")
    if expected_run_name is not None and run_name != expected_run_name:
        raise RuntimeError(
            f"training W&B run_name changed: " f"{run_name!r} != {expected_run_name!r}"
        )
    state_identity = (training.get("state_contract") or {}).get("wandb")
    if state_identity != identity:
        raise RuntimeError("training W&B identity differs from state contract")
    return {str(key): str(value) for key, value in identity.items()}


def validate(args: argparse.Namespace) -> dict[str, Any]:
    config_dir = resolve_config_dir(getattr(args, "config_dir", CONFIG_DIR))
    b200 = config_dir == B200_CONFIG_DIR
    report_path = selected_owned_path(
        args.report_json, config_dir=config_dir, must_exist=True
    )
    report_md_path = selected_owned_path(
        args.report_md, config_dir=config_dir, must_exist=True
    )
    training_path = selected_owned_path(
        args.training_manifest, config_dir=config_dir, must_exist=True
    )
    audit_path = selected_owned_path(
        args.audit_json, config_dir=config_dir, must_exist=True
    )
    audit_md_path = selected_owned_path(
        args.audit_md, config_dir=config_dir, must_exist=True
    )
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
    report = read_json(report_path)
    training = read_json(training_path)
    audit = read_json(audit_path)
    if report.get("status") != "PASS":
        raise RuntimeError("evaluation report is incomplete")
    if report.get("attribution") != "Built with Solar":
        raise RuntimeError("Solar attribution missing from report")
    selection = validate_training_selection(training, config_dir=config_dir)
    if report.get("training_completion") != selection["training_completion"]:
        raise RuntimeError("report training completion differs from the manifest")
    if report.get("early_stopping_decision") != selection["early_stopping_decision"]:
        raise RuntimeError("report early-stopping evidence differs from the manifest")
    if report.get("selection_record") != selection["selection_record"]:
        raise RuntimeError("report selection record differs from the manifest")
    if report.get("selected_checkpoint") != selection["selected_checkpoint"]:
        raise RuntimeError("report selected checkpoint differs from the manifest")
    report_training = report.get("training")
    if not isinstance(report_training, dict) or any(
        (
            report_training.get("status") != selection["training_status"],
            report_training.get("completed_updates") != selection["final_step"],
            report_training.get("target_updates") != selection["target_step"],
            report_training.get("completed_epochs") != selection["completed_epochs"],
            report_training.get("selected_step") != selection["selected_step"],
            report_training.get("selected_selection_loss")
            != selection["selected_selection_loss"],
        )
    ):
        raise RuntimeError("report training horizon or selected step changed")
    training_wandb = validate_training_wandb_identity(
        training,
        entity=args.entity,
        project=args.project,
        training_run_id=args.training_run_id,
        expected_run_id=(
            B200_PRODUCTION_TRAINING_RUN_ID
            if b200
            else EXPECTED_PRODUCTION_TRAINING_RUN_ID
        ),
        expected_run_name=(
            B200_PRODUCTION_TRAINING_RUN_NAME
            if b200
            else EXPECTED_PRODUCTION_TRAINING_RUN_NAME
        ),
    )
    if b200:
        identity = b200_config_identity(config_dir)
        if report.get("profile") != "b200" or report.get("b200_config") != identity:
            raise RuntimeError("evaluation report B200 config identity changed")
        if audit.get("profile") != "b200" or audit.get("b200_config") != identity:
            raise RuntimeError("raw prediction audit B200 config identity changed")
        if training.get("config_identity") != identity:
            raise RuntimeError("training B200 config identity changed")
    if (
        audit.get("status") != "PASS"
        or audit.get("evaluation_integrity_status") != "PASS"
        or audit.get("output_cleanliness_status") != "PASS"
        or audit.get("integrity_failures")
        or audit.get("output_failures")
    ):
        raise RuntimeError("raw prediction audit is not a clean PASS")
    if (
        audit.get("raw_scoring_policy")
        != "outer strip only; no output filtering or repair"
    ):
        raise RuntimeError("raw-output scoring policy changed")
    require_file_identity(
        (audit.get("training_adapter_gate") or {}).get("training_manifest"),
        label="audited training manifest",
        expected_path=training_path,
        config_dir=config_dir,
    )
    data_contract = read_json(config_dir / "data_contract.json")
    data_contract_sha256 = sha256(config_dir / "data_contract.json")
    if report.get("data_contract", {}).get("sha256") != data_contract_sha256:
        raise RuntimeError("report data contract identity changed")
    if audit.get("eval_sha256") != data_contract["files"]["combined_eval"]["sha256"]:
        raise RuntimeError("audit did not use the pinned combined eval")
    base_label = report["base_model_label"]
    tuned_label = report["tuned_model_label"]
    labels = [base_label, tuned_label]
    full_metrics = validate_metrics_receipt(
        full_receipt_path,
        expected_dataset="full_test",
        expected_audit_path=audit_path,
        expected_model_labels=labels,
        expected_summary_path=full_summary_path,
        config_dir=config_dir,
    )
    gold_metrics = validate_metrics_receipt(
        gold_receipt_path,
        expected_dataset="human_gold",
        expected_audit_path=audit_path,
        expected_model_labels=labels,
        expected_summary_path=gold_summary_path,
        config_dir=config_dir,
    )
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("report artifact identities are missing")
    artifact_checks = (
        ("training_manifest", training_path),
        ("audit", audit_path),
        ("full_summary", full_summary_path),
        ("gold_summary", gold_summary_path),
        ("full_metrics_receipt", full_receipt_path),
        ("gold_metrics_receipt", gold_receipt_path),
        ("full_metrics_rows", full_metrics["rows_path"]),
        ("gold_metrics_rows", gold_metrics["rows_path"]),
        ("report_markdown", report_md_path),
        ("audit_markdown", audit_md_path),
    )
    for name, expected_path in artifact_checks:
        require_file_identity(
            artifacts.get(name),
            label=f"report artifact {name}",
            expected_path=expected_path,
            config_dir=config_dir,
        )
    require_file_identity(
        audit.get("artifacts", {}).get("markdown"),
        label="audit markdown artifact",
        expected_path=audit_md_path,
        config_dir=config_dir,
    )
    if report.get("output_only_audit", {}).get("status") != "PASS":
        raise RuntimeError("report does not certify output-only predictions")
    if report.get("output_only_audit", {}).get("reference_scan") != audit.get(
        "reference_output_only_scan"
    ):
        raise RuntimeError("report reference output-only scan changed")
    expected_contract = read_json(config_dir / "recipe.json")["evaluation"]
    metric_contract = report.get("metric_contract") or {}
    if (
        metric_contract.get("bleu_tokenizer") != expected_contract["bleu_tokenizer"]
        or metric_contract.get("bert_model") != expected_contract["bert_model"]
        or metric_contract.get("metrics") != list(METRICS)
        or metric_contract.get("raw_output_policy")
        != "outer strip only; no output filtering or repair"
    ):
        raise RuntimeError("report metric contract changed")
    summaries = {
        "full_test": full_metrics["summary"],
        "human_gold": gold_metrics["summary"],
    }
    for dataset in DATASETS:
        for task in TASKS:
            base = report["models"][base_label][dataset][task]
            tuned = report["models"][tuned_label][dataset][task]
            expected_count = int(data_contract["files"][dataset]["by_task"][task])
            if int(base["count"]) != expected_count:
                raise RuntimeError(f"pinned count mismatch: {dataset}/{task}")
            if int(base["count"]) != int(tuned["count"]):
                raise RuntimeError(f"count mismatch: {dataset}/{task}")
            for label, observed in (
                (base_label, base),
                (tuned_label, tuned),
            ):
                wanted = task_stats(summaries[dataset], label, task)
                if observed != wanted:
                    raise RuntimeError(
                        f"report metrics changed: {dataset}/{task}/{label}"
                    )
            for metric in METRICS:
                observed = float(tuned[metric]) - float(base[metric])
                recorded = float(
                    report["deltas_tuned_minus_base"][dataset][task][metric]
                )
                if abs(observed - recorded) > 1e-10:
                    raise RuntimeError(f"delta mismatch: {dataset}/{task}/{metric}")
    chain = report.get("provenance_chain") or {}
    if chain.get("audit_sha256") != sha256(audit_path):
        raise RuntimeError("report provenance audit digest changed")
    if chain.get("metric_receipts") != {
        "full_test": sha256(full_receipt_path),
        "human_gold": sha256(gold_receipt_path),
    }:
        raise RuntimeError("report provenance metric receipts changed")
    return {
        "report": report,
        "training": training,
        "audit": audit,
        "selection": selection,
        "training_wandb": training_wandb,
        "full_metrics": full_metrics,
        "gold_metrics": gold_metrics,
        "config_dir": config_dir,
        "b200_config": b200_config_identity(config_dir) if b200 else None,
    }


def metric_rows(report: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for label in (report["base_model_label"], report["tuned_model_label"]):
        for dataset in DATASETS:
            for task in TASKS:
                values = report["models"][label][dataset][task]
                rows.append(
                    [
                        label,
                        dataset,
                        task,
                        int(values["count"]),
                        *(float(values[metric]) for metric in METRICS),
                    ]
                )
    return rows


def delta_rows(report: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for dataset in DATASETS:
        for task in TASKS:
            values = report["deltas_tuned_minus_base"][dataset][task]
            rows.append(
                [
                    dataset,
                    task,
                    *(float(values[metric]) for metric in METRICS),
                ]
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--audit-md", type=Path, required=True)
    parser.add_argument("--full-summary", type=Path, required=True)
    parser.add_argument("--gold-summary", type=Path, required=True)
    parser.add_argument("--full-metrics-receipt", type=Path, required=True)
    parser.add_argument("--gold-metrics-receipt", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--success-marker", type=Path, required=True)
    parser.add_argument("--wandb-dir", type=Path, required=True)
    parser.add_argument("--entity", default="junhyuk-tak-markr")
    parser.add_argument("--project", default="modern-korean-lm")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config_dir = resolve_config_dir(args.config_dir)
    if config_dir == B200_CONFIG_DIR:
        require_expected_user()
    for name in (
        "report_json",
        "report_md",
        "training_manifest",
        "audit_json",
        "audit_md",
        "full_summary",
        "gold_summary",
        "full_metrics_receipt",
        "gold_metrics_receipt",
    ):
        selected_owned_path(getattr(args, name), config_dir=config_dir, must_exist=True)
    if args.run_id == args.training_run_id:
        raise RuntimeError("evaluation and training must use separate W&B run IDs")
    receipt_path = selected_owned_path(
        args.receipt, config_dir=config_dir, create_parent=True
    )
    success_path = selected_owned_path(
        args.success_marker, config_dir=config_dir, create_parent=True
    )
    wandb_dir = selected_owned_path(
        args.wandb_dir, config_dir=config_dir, create_parent=True
    )
    values = validate(args)
    report = values["report"]
    training = values["training"]
    audit = values["audit"]
    selection = values["selection"]
    training_wandb = values["training_wandb"]
    identity = {
        "entity": args.entity,
        "project": args.project,
        "run_id": args.run_id,
        "run_name": args.run_name,
        "report_sha256": sha256(args.report_json),
        "report_md_sha256": sha256(args.report_md),
        "training_manifest_sha256": sha256(args.training_manifest),
        "audit_sha256": sha256(args.audit_json),
        "audit_md_sha256": sha256(args.audit_md),
        "full_summary_sha256": sha256(args.full_summary),
        "gold_summary_sha256": sha256(args.gold_summary),
        "full_metrics_receipt_sha256": sha256(args.full_metrics_receipt),
        "gold_metrics_receipt_sha256": sha256(args.gold_metrics_receipt),
        "full_metrics_rows_sha256": sha256(values["full_metrics"]["rows_path"]),
        "gold_metrics_rows_sha256": sha256(values["gold_metrics"]["rows_path"]),
        "training_run_id": args.training_run_id,
        "training_run_name": training_wandb["run_name"],
        "training_status": selection["training_status"],
        "training_final_step": selection["final_step"],
        "training_target_step": selection["target_step"],
        "training_selected_step": selection["selected_step"],
        "training_early_stopping_status": selection["early_stopping_decision"][
            "status"
        ],
    }
    if config_dir == B200_CONFIG_DIR:
        identity.update(
            {
                "profile": "b200",
                "b200_config": values["b200_config"],
            }
        )
    if args.validate_only:
        print(json.dumps({"status": "VALID", **identity}, sort_keys=True))
        return 0
    if receipt_path.is_file() and success_path.is_file():
        previous = read_json(receipt_path)
        if previous.get("status") == "ONLINE_SYNCED" and all(
            previous.get(key) == value for key, value in identity.items()
        ):
            print(json.dumps(previous, ensure_ascii=False, sort_keys=True))
            return 0

    wandb_dir.mkdir(parents=True, exist_ok=True)
    success_path.unlink(missing_ok=True)
    os.environ.update(
        {
            "WANDB_MODE": "online",
            "WANDB_ENTITY": args.entity,
            "WANDB_PROJECT": args.project,
            "WANDB_DIR": str(wandb_dir),
            "WANDB_INIT_TIMEOUT": "180",
            "WANDB_CONSOLE": "off",
        }
    )
    last_error: Exception | None = None
    for attempt in range(1, 4):
        run = None
        try:
            import wandb

            run = wandb.init(
                entity=args.entity,
                project=args.project,
                id=args.run_id,
                resume="allow",
                name=args.run_name,
                job_type="evaluation",
                tags=[
                    "solar-open2",
                    "lora",
                    "translation",
                    "summarization",
                    "human-gold",
                ],
                dir=str(wandb_dir),
                config={
                    "model_name": report["model_name"],
                    "attribution": "Built with Solar",
                    "base_model_label": report["base_model_label"],
                    "tuned_model_label": report["tuned_model_label"],
                    "training": training["config"],
                    "training_completion": selection["training_completion"],
                    "early_stopping_decision": selection["early_stopping_decision"],
                    "selection_record": selection["selection_record"],
                    "selected_checkpoint": selection["selected_checkpoint"],
                    "metric_contract": report["metric_contract"],
                    "training_wandb_run": {
                        **training_wandb,
                    },
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    **(
                        {"b200_config": values["b200_config"]}
                        if config_dir == B200_CONFIG_DIR
                        else {}
                    ),
                },
            )
            if run is None or run.offline:
                raise RuntimeError("W&B did not create an online run")
            run.log(
                {
                    "evaluation/base_vs_tuned": wandb.Table(
                        columns=[
                            "model",
                            "dataset",
                            "task",
                            "count",
                            *METRICS,
                        ],
                        data=metric_rows(report),
                    ),
                    "evaluation/tuned_minus_base": wandb.Table(
                        columns=["dataset", "task", *METRICS],
                        data=delta_rows(report),
                    ),
                    "training/validation_history": wandb.Table(
                        columns=[
                            "step",
                            "scope",
                            "selection_loss",
                            "translation_loss",
                            "summarization_loss",
                            "translation_count",
                            "summarization_count",
                        ],
                        data=[
                            [
                                row["step"],
                                row["scope"],
                                row["selection_loss"],
                                row["translation_loss"],
                                row["summarization_loss"],
                                row["translation_count"],
                                row["summarization_count"],
                            ]
                            for row in training["history"]
                        ],
                    ),
                }
            )
            run.summary["training/completed_updates"] = selection["final_step"]
            run.summary["training/target_updates"] = selection["target_step"]
            run.summary["training/status"] = selection["training_status"]
            run.summary["training/completed_epochs"] = selection["completed_epochs"]
            run.summary["training/early_stopping_status"] = selection[
                "early_stopping_decision"
            ]["status"]
            run.summary["training/early_stopping_triggered"] = selection[
                "early_stopping_decision"
            ]["triggered"]
            run.summary["training/early_stopping_reason"] = selection[
                "early_stopping_decision"
            ]["reason"]
            run.summary["training/selected_step"] = selection["selected_step"]
            run.summary["training/selected_selection_loss"] = selection[
                "selected_selection_loss"
            ]
            run.summary["training/trainable_parameters"] = training[
                "trainable_parameters"
            ]
            run.summary["audit/evaluation_integrity"] = audit[
                "evaluation_integrity_status"
            ]
            run.summary["audit/output_cleanliness"] = audit["output_cleanliness_status"]
            for role, scan in audit["reference_output_only_scan"]["roles"].items():
                run.summary[f"audit/reference_scan/{role}/rows"] = scan["rows"]
                run.summary[f"audit/reference_scan/{role}/normalized_prefixes"] = scan[
                    "normalized_answer_prefix_count"
                ]
                run.summary[f"audit/reference_scan/{role}/output_signal_rows"] = scan[
                    "output_signal_rows"
                ]
            tuned_label = report["tuned_model_label"]
            for dataset in DATASETS:
                for task in TASKS:
                    current = report["models"][tuned_label][dataset][task]
                    deltas = report["deltas_tuned_minus_base"][dataset][task]
                    for metric in METRICS:
                        run.summary[f"eval/tuned/{dataset}/{task}/{metric}"] = current[
                            metric
                        ]
                        run.summary[
                            f"delta/tuned_vs_base/{dataset}/{task}/{metric}"
                        ] = deltas[metric]
            artifact = wandb.Artifact(
                name=f"{args.run_id}-report",
                type="training-evaluation-report",
                metadata={
                    **identity,
                    "training_status": selection["training_status"],
                    "final_step": selection["final_step"],
                    "target_step": selection["target_step"],
                    "early_stopping_decision": selection["early_stopping_decision"],
                    "selected_step": selection["selected_step"],
                    "selected_selection_loss": selection["selected_selection_loss"],
                },
            )
            for path, name in (
                (args.report_json, "report.json"),
                (args.report_md, "report.md"),
                (args.training_manifest, "run_manifest.json"),
                (args.audit_json, "output_audit.json"),
                (args.audit_md, "output_audit.md"),
                (args.full_summary, "full_test.metrics.summary.json"),
                (args.gold_summary, "human_gold.metrics.summary.json"),
                (
                    args.full_metrics_receipt,
                    "full_test.metrics.receipt.json",
                ),
                (
                    args.gold_metrics_receipt,
                    "human_gold.metrics.receipt.json",
                ),
                (
                    values["full_metrics"]["rows_path"],
                    "full_test.metrics.rows.jsonl",
                ),
                (
                    values["gold_metrics"]["rows_path"],
                    "human_gold.metrics.rows.jsonl",
                ),
            ):
                artifact.add_file(str(path), name=name)
            run.log_artifact(artifact)
            run_url = run.url
            run.finish(exit_code=0)
            run = None
            result = {
                "status": "ONLINE_SYNCED",
                **identity,
                "entity": args.entity,
                "project": args.project,
                "run_url": run_url,
                "attempt": attempt,
            }
            atomic_json(receipt_path, result)
            atomic_text(success_path, "PASS\n")
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            return 0
        except Exception as error:  # noqa: BLE001
            last_error = error
            if run is not None:
                run.finish(exit_code=1)
            if attempt < 3:
                time.sleep(15 * attempt)
    raise RuntimeError("W&B online upload failed after three attempts") from last_error


if __name__ == "__main__":
    raise SystemExit(main())
