#!/usr/bin/env python3
"""CPU-only fixture tests for fail-closed smoke certification."""

from __future__ import annotations

import copy
import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
from safetensors.numpy import save_file


sys.path.insert(0, str(Path(__file__).resolve().parent))

import certify_smoke  # noqa: E402
import infer  # noqa: E402
import train  # noqa: E402
from common import (  # noqa: E402
    CONFIG_DIR,
    PROJECT,
    REFERENCE_SHA256,
    TRAINING_RESULT_LABEL_IDS,
    canonical_sha256,
    expected_lora_module_names,
    sha256,
)
from quantization.contract import (  # noqa: E402
    BASE_MODEL_ID,
    BASE_REVISION,
    EXPECTED_SOURCE_SHARD_COUNT,
    EXPECTED_SOURCE_TENSOR_COUNT,
    LAYER_FLAT_SCHEMA_NAME,
    SCHEMA_NAME,
    SCHEMA_VERSION,
)


TEST_TEMP_ROOT = Path(__file__).resolve().parents[2] / "runtime/tmp/tests/certify_smoke"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


class TrainingSchedulerContractTest(unittest.TestCase):
    def test_smoke_replays_production_first_warmup_learning_rate(self) -> None:
        learning_rate = 1e-4
        total_steps = 18_631
        warmup_steps = train.scheduler_warmup_steps(
            smoke=False,
            warmup_ratio=0.03,
            total_steps=total_steps,
        )
        self.assertEqual(warmup_steps, 558)
        self.assertEqual(
            train.positive_cosine_learning_rate(
                update=1,
                total_updates=total_steps,
                warmup_updates=warmup_steps,
                peak_learning_rate=learning_rate,
                minimum_learning_rate_ratio=0.1,
            ),
            learning_rate / warmup_steps,
        )

    def test_every_production_update_uses_a_positive_learning_rate(self) -> None:
        learning_rate = 1e-4
        total_steps = 18_631
        warmup_steps = train.scheduler_warmup_steps(
            smoke=False,
            warmup_ratio=0.03,
            total_steps=total_steps,
        )
        self.assertEqual(warmup_steps, 558)
        first = train.positive_cosine_learning_rate(
            update=1,
            total_updates=total_steps,
            warmup_updates=warmup_steps,
            peak_learning_rate=learning_rate,
            minimum_learning_rate_ratio=0.1,
        )
        last = train.positive_cosine_learning_rate(
            update=total_steps,
            total_updates=total_steps,
            warmup_updates=warmup_steps,
            peak_learning_rate=learning_rate,
            minimum_learning_rate_ratio=0.1,
        )
        self.assertEqual(
            first,
            learning_rate / warmup_steps,
        )
        self.assertGreaterEqual(last, learning_rate * 0.1)


class CertificationFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.source_dir = root / "hf_cache/model/snapshots" / BASE_REVISION
        self.training_expert_dir = root / "training-experts"
        self.inference_expert_dir = root / "inference-experts"
        self.final_adapter_dir = root / "train/final_adapter"
        self.adapter_dir = root / "train/best_adapter"
        self.tokenizer_contract_path = root / "tokenizer-output-contract.json"
        self.preflight_path = root / "preflight.json"
        self.training_path = root / "train/run_manifest.json"
        self.download_path = root / "download.json"
        self.eval_path = root / "combined-eval.jsonl"
        self.base_path = root / "base.jsonl"
        self.tuned_path = root / "tuned.jsonl"
        self._build()

    @property
    def kwargs(self) -> dict[str, Any]:
        return {
            "source_dir": self.source_dir,
            "training_expert_nf4_dir": self.training_expert_dir,
            "inference_expert_nf4_dir": self.inference_expert_dir,
            "tokenizer_contract_receipt": self.tokenizer_contract_path,
            "preflight_receipt": self.preflight_path,
            "training_manifest": self.training_path,
            "base_predictions": self.base_path,
            "tuned_predictions": self.tuned_path,
            "expected_source_index_sha256": self.source["index_sha256"],
            "expected_evaluation": self.evaluation,
            "expected_tokenizer_source_dir": self.source_dir,
            "expected_tokenizer_data": self.tokenizer_data,
        }

    def rewrite_manifest(self, path: Path, mutate: Any) -> None:
        value = json.loads(path.read_text(encoding="utf-8"))
        mutate(value)
        write_json(path, value)

    def _build_source(self) -> None:
        self.source_dir.mkdir(parents=True)
        write_json(
            self.source_dir / "config.json",
            {
                "model_type": "solar_open2",
                "architectures": ["SolarOpen2ForCausalLM"],
            },
        )
        write_json(
            self.source_dir / "model.safetensors.index.json",
            {
                "metadata": {"total_size": 1},
                "weight_map": {"weight": "model.safetensors"},
            },
        )
        write_json(
            self.source_dir / "tokenizer_config.json",
            {"model_max_length": 3072},
        )
        write_json(self.source_dir / "tokenizer.json", {"version": "1.0"})
        (self.source_dir / "chat_template.jinja").write_text(
            "fixture template\n",
            encoding="utf-8",
        )
        write_json(
            self.source_dir / "generation_config.json",
            {"eos_token_id": [2, 129]},
        )
        (self.source_dir / "LICENSE").write_text("fixture license\n", encoding="utf-8")
        self.source = infer.lightweight_source_identity(self.source_dir)
        self.source_weight_map_sha256 = canonical_sha256(
            {"weight": "model.safetensors"}
        )
        self.source_shards = [
            {
                "name": "model.safetensors",
                "file_bytes": 1,
                "sha256": "a" * 64,
            }
        ]

    def _build_tokenizer_contract(self) -> None:
        data_contract_path = CONFIG_DIR / "data_contract.json"
        data_contract = json.loads(data_contract_path.read_text(encoding="utf-8"))
        data_dir = Path(data_contract["prepared_data_dir"]).resolve()
        manifest_path = Path(data_contract["prepared_manifest"]).resolve()
        self.tokenizer_data = {
            "path": str(data_dir),
            "data_contract_sha256": sha256(data_contract_path),
            "prepared_manifest": {
                "path": str(manifest_path),
                "sha256": data_contract["prepared_manifest_sha256"],
            },
            "files": {
                role: {
                    "path": str(data_dir / entry["name"]),
                    "sha256": entry["sha256"],
                    "rows": entry["rows"],
                    "by_task": entry["by_task"],
                }
                for role, entry in data_contract["files"].items()
            },
        }
        ordinary = [
            ("train-translation", "translation", "train"),
            ("train-summarization", "summarization", "train"),
            (
                "validation-translation",
                "translation",
                "validation",
            ),
            (
                "validation-summarization",
                "summarization",
                "validation",
            ),
        ]
        row_specs = [
            *ordinary,
            *(
                (row_id, "translation", "train")
                for row_id in sorted(TRAINING_RESULT_LABEL_IDS)
            ),
        ]
        rows = []
        for row_id, task, split in row_specs:
            pinned = row_id in TRAINING_RESULT_LABEL_IDS
            rows.append(
                {
                    "id": row_id,
                    "task": task,
                    "coverage_role": (
                        "pinned_answer_prefix"
                        if pinned
                        else f"{split}_{task}_representative"
                    ),
                    "split": split,
                    "body_sha256": canonical_sha256(f"body-{row_id}"),
                    "prompt_ids_sha256": canonical_sha256(
                        [ord(character) for character in row_id]
                    ),
                    "prompt_token_count": 100,
                    "answer_token_count": 20,
                    "assistant_termination_token_count": 1,
                    "assistant_termination_token_id": 2,
                    "template_suffix_token_count": 2,
                    "supervised_labels_are_body_plus_single_eos": True,
                    "answer_label_prefix_removed": pinned,
                    "forbidden_body_framing": [],
                    "train_inference_prompt_identical": True,
                    "inference_requires_body_label": False,
                }
            )
        counts: dict[str, int] = {
            task: sum(row["task"] == task for row in rows)
            for task in ("translation", "summarization")
        }
        source = certify_smoke._current_tokenizer_source_identity(
            self.source_dir,
            expected_source_dir=self.source_dir,
        )
        self.tokenizer_contract = {
            "status": "PASS",
            "user": "mk03",
            "project": str(PROJECT),
            "source": source,
            "data": self.tokenizer_data,
            "reference": {
                "path": str(PROJECT / "scripts/reference/shared_nodict_evaluation.py"),
                "sha256": REFERENCE_SHA256,
            },
            "tokenizer": {
                "class": "FixtureSolarTokenizerFast",
                "name_or_path": str(self.source_dir),
                "is_fast": True,
            },
            "output_only_contract": {
                "status": "PASS",
                "template_kwargs": {
                    "reasoning_effort": "none",
                    "think_render_option": "preserved",
                    "provider_system_prompt": False,
                },
                "training_add_generation_prompt": True,
                "training_full_sequence_add_generation_prompt": False,
                "inference_add_generation_prompt": True,
                "inference_requires_response_field": False,
                "inference_requires_body_label": False,
                "rows": rows,
                "coverage": {
                    "rows": 7,
                    "by_task": counts,
                    "pinned_answer_prefix_ids": sorted(TRAINING_RESULT_LABEL_IDS),
                    "ordinary_task_representatives": {
                        "train": ["summarization", "translation"],
                        "validation": ["summarization", "translation"],
                    },
                },
                "supervised_termination_token_ids": [2],
                "configured_generation_eos_token_ids": [2, 129],
            },
            "combined_eval_prompt_budget": {
                "status": "PASS",
                "path": self.tokenizer_data["files"]["combined_eval"]["path"],
                "rows": infer.PINNED_COMBINED_EVAL_ROWS,
                "by_task": infer.PINNED_COMBINED_EVAL_TASK_COUNTS,
                "prompt_budget_tokens": {
                    "translation": 8_192,
                    "summarization": 10_240,
                },
                "truncation_count": 0,
                "exact_output_only_preamble_count": (infer.PINNED_COMBINED_EVAL_ROWS),
                "prompt_tokens": {
                    "translation": {
                        "minimum": 100,
                        "maximum": 200,
                        "maximum_row": {
                            "id": "translation-maximum",
                            "tokens": 200,
                        },
                    },
                    "summarization": {
                        "minimum": 120,
                        "maximum": 240,
                        "maximum_row": {
                            "id": "summarization-maximum",
                            "tokens": 240,
                        },
                    },
                },
            },
            "pipeline_code": certify_smoke._current_identities()["pipeline_code"],
            "checks": {
                name: "PASS" for name in certify_smoke.TOKENIZER_CONTRACT_CHECKS
            },
        }
        write_json(self.tokenizer_contract_path, self.tokenizer_contract)

    def _expert_manifest(self, schema: str, index_sha256: str) -> dict[str, Any]:
        return {
            "schema": schema,
            "schema_version": SCHEMA_VERSION,
            "status": "SEALED",
            "complete_model_checkpoint": False,
            "base_model_id": BASE_MODEL_ID,
            "base_revision": BASE_REVISION,
            "source": {
                "source_dir": str(self.source_dir),
                "index_sha256": self.source["index_sha256"],
                "config_sha256": self.source["config_sha256"],
                "weight_map_sha256": self.source_weight_map_sha256,
                "shards": self.source_shards,
            },
            "expert_index_sha256": index_sha256,
        }

    def _build_experts(self) -> None:
        for directory, schema in (
            (self.training_expert_dir, LAYER_FLAT_SCHEMA_NAME),
            (self.inference_expert_dir, SCHEMA_NAME),
        ):
            directory.mkdir(parents=True)
            write_json(
                directory / "expert_nf4.safetensors.index.json",
                {
                    "metadata": {"total_size": 1},
                    "weight_map": {"weight": "fixture.safetensors"},
                },
            )
            index_sha256 = sha256(directory / "expert_nf4.safetensors.index.json")
            write_json(
                directory / "expert_nf4_manifest.json",
                self._expert_manifest(schema, index_sha256),
            )
        self.inference_expert = infer.lightweight_expert_identity(
            self.inference_expert_dir,
            source=self.source,
        )

    def _build_adapter(self) -> None:
        self.final_adapter_dir.mkdir(parents=True)
        lora = json.loads((CONFIG_DIR / "recipe.json").read_text(encoding="utf-8"))[
            "lora"
        ]
        write_json(
            self.final_adapter_dir / "adapter_config.json",
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": lora["rank"],
                "lora_alpha": lora["alpha"],
                "lora_dropout": lora["dropout"],
                "bias": lora["bias"],
                "target_modules": lora["target_regex"],
                "modules_to_save": None,
            },
        )
        architecture = json.loads(
            (CONFIG_DIR / "model.json").read_text(encoding="utf-8")
        )["architecture_contract"]
        keys = {
            f"base_model.model.{name}.lora_{side}.weight"
            for name in expected_lora_module_names(architecture)
            for side in ("A", "B")
        }
        save_file(
            {key: np.zeros((1,), dtype=np.float32) for key in sorted(keys)},
            self.final_adapter_dir / "adapter_model.safetensors",
        )
        (self.final_adapter_dir / "_SUCCESS").write_text(
            "PASS\n",
            encoding="utf-8",
        )
        shutil.copytree(self.final_adapter_dir, self.adapter_dir)
        self.adapter = infer.validate_adapter(self.adapter_dir)
        self.final_adapter_files = {
            path.name: sha256(path)
            for path in sorted(self.final_adapter_dir.iterdir())
            if path.is_file()
        }
        self.adapter_files = {
            path.name: sha256(path)
            for path in sorted(self.adapter_dir.iterdir())
            if path.is_file()
        }

    def _build_preflight(self) -> None:
        write_json(
            self.download_path,
            {
                "status": "PASS",
                "repo_id": BASE_MODEL_ID,
                "requested_revision": BASE_REVISION,
                "repo_sha": BASE_REVISION,
                "snapshot_dir": str(self.source_dir),
                "indexed_shards": EXPECTED_SOURCE_SHARD_COUNT,
                "indexed_tensors": EXPECTED_SOURCE_TENSOR_COUNT,
                "local_dir_copy": False,
                "resumable_hf_cache": True,
                "verified_tensor_bytes": 1,
                "cache_used_bytes": 1,
                "contract_hashes": {
                    "config.json": self.source["config_sha256"],
                    "model.safetensors.index.json": self.source["index_sha256"],
                    "LICENSE": sha256(self.source_dir / "LICENSE"),
                },
            },
        )
        identities = certify_smoke._current_identities()
        training_manifest_path = self.training_expert_dir / "expert_nf4_manifest.json"
        training_index_path = (
            self.training_expert_dir / "expert_nf4.safetensors.index.json"
        )
        self.preflight = {
            "status": "PASS",
            "user": "mk03",
            "project": str(PROJECT),
            "source_dir": str(self.source_dir),
            "expert_nf4_dir": str(self.training_expert_dir),
            **identities,
            "download": {
                "path": str(self.download_path),
                "sha256": sha256(self.download_path),
                "repo_sha": BASE_REVISION,
                "weight_bytes": 1,
                "cache_used_bytes": 1,
            },
            "expert_artifact": {
                "status": "PASS",
                "production_eligible": True,
                "checkpoint_dir": str(self.training_expert_dir),
                "manifest_sha256": sha256(training_manifest_path),
                "expert_index_sha256": sha256(training_index_path),
            },
            "hardware": {"device_count": 3},
        }
        write_json(self.preflight_path, self.preflight)

    def _build_training(self) -> None:
        identities = certify_smoke._current_identities()
        data_contract = json.loads(
            (CONFIG_DIR / "data_contract.json").read_text(encoding="utf-8")
        )
        recipe = json.loads((CONFIG_DIR / "recipe.json").read_text(encoding="utf-8"))
        state_contract = {
            "format_version": 4,
            "source_dir": str(self.source_dir),
            "expert_nf4_dir": str(self.training_expert_dir),
            "download_receipt_sha256": self.preflight["download"]["sha256"],
            "expert_manifest_sha256": self.preflight["expert_artifact"][
                "manifest_sha256"
            ],
            "model_contract_sha256": identities["model_contract_sha256"],
            "recipe_sha256": identities["recipe_sha256"],
            "data_contract_sha256": identities["data_contract_sha256"],
            "train_sha256": data_contract["files"]["train"]["sha256"],
            "validation_sha256": data_contract["files"]["validation"]["sha256"],
            "pipeline_code_sha256": identities["pipeline_code"]["sha256"],
            "environment_lock_sha256": identities["environment_lock_sha256"],
            "smoke": True,
        }
        stats = {
            "parameter_count": certify_smoke.EXPECTED_LORA_TENSORS,
            "element_count": certify_smoke.EXPECTED_TRAINABLE_PARAMETERS,
            "dtype_parameter_counts": {
                "torch.float32": certify_smoke.EXPECTED_LORA_TENSORS
            },
            "dtype_element_counts": {
                "torch.float32": certify_smoke.EXPECTED_TRAINABLE_PARAMETERS
            },
            "finite": True,
            "l1": 1.0,
            "l2": 1.0,
        }
        gradient = {
            "parameters_with_gradient": 1,
            "nonzero_gradient_elements": 1,
            "finite": True,
            "l2": 1.0,
        }
        update = {
            "before_sha256": "1" * 64,
            "after_sha256": "2" * 64,
            "changed": True,
            "optimizer_step": 1,
            "learning_rate_used": 1e-4 / 558,
            "before": stats,
            "after": stats,
        }
        probe = {"sampled_parameter_count": 32, "values_sha256": "3" * 64}
        output_only = {
            "prompt_token_count": 100,
            "answer_token_count": 20,
            "assistant_termination_token_count": 1,
            "assistant_termination_token_id": 2,
            "template_suffix_token_count": 2,
            "template_suffix_starts_with_eos": True,
            "supervised_sequence_ends_at_eos": True,
            "prompt_labels_all_masked": True,
            "prompt_weights_all_zero": True,
            "supervised_prefix_is_answer_body": True,
            "reasoning_effort": "none",
        }
        production_training_plan = {
            "optimizer_updates": 18_631,
            "occurrences": 298_096,
            "microbatches": 58_096,
            "bucket_microbatches": {
                "1024": 25_000,
                "2048": 17_500,
                "4096": 12_500,
                "9600": 3_096,
            },
            "bucket_rows": {
                "1024": 200_000,
                "2048": 70_000,
                "4096": 25_000,
                "9600": 3_096,
            },
        }
        subset_validation_plan = {
            "optimizer_updates": 1,
            "occurrences": 1_024,
            "microbatches": 174,
            "bucket_microbatches": {
                "1024": 100,
                "2048": 40,
                "4096": 30,
                "9600": 4,
            },
            "bucket_rows": {
                "1024": 800,
                "2048": 160,
                "4096": 60,
                "9600": 4,
            },
        }
        full_validation_plan = {
            "optimizer_updates": 1,
            "occurrences": 17_850,
            "microbatches": 3_450,
            "bucket_microbatches": {
                "1024": 1_500,
                "2048": 1_000,
                "4096": 900,
                "9600": 50,
            },
            "bucket_rows": {
                "1024": 12_000,
                "2048": 4_000,
                "4096": 1_800,
                "9600": 50,
            },
        }

        def timing_evidence(seconds: float) -> dict[str, object]:
            return {
                "warmup_repetitions": 1,
                "warmup_seconds": seconds * 1.5,
                "timed_repetitions": 3,
                "seconds_samples": [
                    seconds * 1.1,
                    seconds,
                    seconds * 0.9,
                ],
                "seconds_p50": seconds,
                "seconds": seconds,
                "estimator": "median_p50",
                "benchmark_state_unchanged": True,
            }

        def max_context_timing_evidence(
            seconds: float,
        ) -> dict[str, object]:
            evidence = timing_evidence(seconds)
            evidence.pop("warmup_seconds")
            return evidence

        training_bucket_benchmarks = {
            "1024": {
                "minimum_length": 1,
                "maximum_length": 1_024,
                "batch_size": 8,
                "row_ids": [f"bucket-1024-{index}" for index in range(8)],
                "observed_lengths": [1_024] * 8,
                **timing_evidence(1.0),
            },
            "2048": {
                "minimum_length": 1_025,
                "maximum_length": 2_048,
                "batch_size": 4,
                "row_ids": [f"bucket-2048-{index}" for index in range(4)],
                "observed_lengths": [2_048] * 4,
                **timing_evidence(2.0),
            },
            "4096": {
                "minimum_length": 2_049,
                "maximum_length": 4_096,
                "batch_size": 2,
                "row_ids": ["bucket-4096-0", "bucket-4096-1"],
                "observed_lengths": [4_096, 4_000],
                **timing_evidence(4.0),
            },
            "9600": {
                "minimum_length": 4_097,
                "maximum_length": 9_600,
                "batch_size": 1,
                "row_ids": ["train-longest"],
                "observed_lengths": [9_560],
                **timing_evidence(10.0),
            },
        }
        validation_bucket_benchmarks = {
            "1024": {
                "minimum_length": 1,
                "maximum_length": 1_024,
                "batch_size": 8,
                "row_ids": [f"validation-bucket-1024-{index}" for index in range(8)],
                "observed_lengths": [1_024] * 8,
                **timing_evidence(0.5),
            },
            "2048": {
                "minimum_length": 1_025,
                "maximum_length": 2_048,
                "batch_size": 4,
                "row_ids": [f"validation-bucket-2048-{index}" for index in range(4)],
                "observed_lengths": [2_048] * 4,
                **timing_evidence(1.0),
            },
            "4096": {
                "minimum_length": 2_049,
                "maximum_length": 4_096,
                "batch_size": 2,
                "row_ids": [
                    "validation-bucket-4096-0",
                    "validation-bucket-4096-1",
                ],
                "observed_lengths": [4_096, 4_000],
                **timing_evidence(2.0),
            },
            "9600": {
                "minimum_length": 4_097,
                "maximum_length": 9_600,
                "batch_size": 1,
                "row_ids": ["validation-longest"],
                "observed_lengths": [7_000],
                **timing_evidence(5.0),
            },
        }
        update_samples = [float(value) for value in range(10, 20)]
        update_timed_steps = [
            min(
                18_631,
                max(
                    3,
                    round((index + 0.5) * 18_631 / 10),
                ),
            )
            for index in range(10)
        ]
        production_update_benchmark = {
            "warmup_repetitions": 2,
            "warmup_step_numbers": [1, 2],
            "timed_repetitions": 10,
            "timed_step_numbers": update_timed_steps,
            "seconds_samples": update_samples,
            "seconds_p50": 14.5,
            "seconds_p90": 18.0,
            "seconds_mean": 14.5,
            "estimator": "median_p50",
            "occurrences_per_update": 16,
            "microbatch_samples": [5] * 10,
            "timed_plan": {
                "optimizer_updates": 10,
                "occurrences": 160,
                "microbatches": 50,
                "bucket_microbatches": {
                    "1024": 10,
                    "2048": 10,
                    "4096": 10,
                    "9600": 20,
                },
                "bucket_rows": {
                    "1024": 80,
                    "2048": 40,
                    "4096": 20,
                    "9600": 20,
                },
            },
            "row_order_sha256": "f" * 64,
            "benchmark_state_unchanged": True,
        }
        microbatch_phase_seconds = 10.0
        validation_row_seconds = 5.0
        optimizer_boundary_seconds = 5.0
        one_time_gradient_audit_seconds = 3.0
        runtime_budget = certify_smoke._runtime_budget_projection(
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
        validation_benchmark = {
            "warmup": ("one excluded shape-matched run before three timed repeats"),
            "selection": "absolute longest validation row per task",
            "by_task": {
                "translation": {
                    "row_id": "validation-translation",
                    "tokens": 6_800,
                    **max_context_timing_evidence(4.0),
                },
                "summarization": {
                    "row_id": "validation-summarization",
                    "tokens": 7_000,
                    **max_context_timing_evidence(validation_row_seconds),
                },
            },
            "maximum_seconds": validation_row_seconds,
            "training_buckets": training_bucket_benchmarks,
            "validation_buckets": validation_bucket_benchmarks,
            "production_update_benchmark": production_update_benchmark,
            "benchmark_state_unchanged": True,
        }
        gradient_checkpointing_contract = {
            "status": "PASS",
            "implementation": "device_aware_non_reentrant",
            "use_reentrant": False,
            "preserve_rng_state": True,
            "determinism_check": "default",
            "early_stop": True,
            "decoder_layer_count": 48,
            "boundary_layers": [16, 32],
            "layer_input_devices": {
                str(layer): f"cuda:{layer // 16}" for layer in range(48)
            },
        }
        post_update_recompute_audit = {
            "status": "PASS",
            "phase": "after_first_positive_lr_optimizer_update",
            "row_id": "train-longest",
            "observed_tokens": 9_560,
            "hard_term_weighted_tokens": 0,
            "loss": 1.25,
            "seconds": 13.0,
            "gradient_audit": gradient,
            "adapter_parameters_unchanged": True,
            "frozen_parameters_unchanged": True,
        }
        smoke_history = [
            {
                "step": 1,
                "scope": "fixed_subset",
                "selection_loss": 1.175,
                "translation_loss": 1.0,
                "summarization_loss": 1.5,
                "translation_count": 1,
                "summarization_count": 1,
                "dropped_over_max": {},
                "validation_order_sha256": "4" * 64,
                "elapsed_seconds": 1.0,
            }
        ]
        early_stopping_evidence = certify_smoke.early_stopping_decision(
            smoke_history,
            completed_step=1,
            policy=recipe["validation"]["early_stopping"],
            optimizer_steps=18_631,
            full_validation_steps=[9_316, 13_973, 18_631],
            occurrences_per_step=16,
            occurrences_per_epoch=149_048,
        )
        smoke = {
            "max_length_contract": 9_600,
            "absolute_longest_required_tokens": 9_560,
            "observed_tokens": 9_560,
            "backward_passes": 16,
            "consumed_occurrences": 16,
            "update_microbatch_plan": {
                "optimizer_updates": 1,
                "occurrences": 16,
                "microbatches": 5,
                "bucket_microbatches": {
                    "1024": 1,
                    "2048": 1,
                    "4096": 1,
                    "9600": 2,
                },
                "bucket_rows": {
                    "1024": 8,
                    "2048": 4,
                    "4096": 2,
                    "9600": 2,
                },
            },
            "hard_term_weighted_tokens": 2,
            "gradient_audit": gradient,
            "adapter_update": update,
            "frozen_base_probe_unchanged": True,
            "adapter_files_sha256": self.adapter_files,
            "gpu_count": 3,
            "peak_allocated_bytes": {"0": 1, "1": 1, "2": 1},
            "output_only_supervision": {
                **output_only,
                "translation_or_summarization_body_only": True,
            },
            "microbatch_phase_seconds": microbatch_phase_seconds,
            "validation_row_seconds": validation_row_seconds,
            "validation_benchmark": validation_benchmark,
            "production_update_benchmark": production_update_benchmark,
            "optimizer_boundary_seconds": optimizer_boundary_seconds,
            "one_time_gradient_audit_seconds": (one_time_gradient_audit_seconds),
            "runtime_budget_projection": runtime_budget,
            "post_update_recompute_audit": post_update_recompute_audit,
            "production_training_plan": production_training_plan,
            "subset_validation_plan": subset_validation_plan,
            "full_validation_plan": full_validation_plan,
            "early_stopping_decision": early_stopping_evidence,
        }
        self.training = {
            "status": "SMOKE_COMPLETED",
            "source_dir": str(self.source_dir),
            "expert_nf4_dir": str(self.training_expert_dir),
            "output_dir": str(self.adapter_dir.parent),
            "config": recipe,
            "state_contract": state_contract,
            "completed_updates": 1,
            "target_updates": 1,
            "backward_passes": 16,
            "consumed_occurrences": 16,
            "training_completion": {
                "status": "SMOKE_COMPLETED",
                "epochs": 2,
                "optimizer_steps": 1,
                "consumed_occurrences": 16,
                "expected_optimizer_steps": 18_631,
                "expected_occurrences": 298_096,
                "completed_epochs": early_stopping_evidence["completed_epochs"],
                "early_stopping_triggered": False,
            },
            "training_coverage": {
                "total_occurrences": 16,
                "skipped_occurrences": 0,
                "unconsumed_planned_occurrences": 298_080,
                "unique_raw_rows": None,
                "task_occurrences": {
                    "translation": 10,
                    "summarization": 6,
                },
                "quality_occurrences": {
                    "silver_model_generated": 15,
                    "gold_expert_corrected": 1,
                },
            },
            "token_length_contract": {
                "train": {
                    "rows": 142_713,
                    "by_task": {
                        "translation": 97_044,
                        "summarization": 45_669,
                    },
                    "max_length": 9_600,
                    "over_max_count": 0,
                    "over_max_rows": [],
                    "minimum": 10,
                    "p50": 512,
                    "p90": 1_500,
                    "p95": 2_000,
                    "p99": 3_500,
                    "p99_9": 7_500,
                    "maximum": 9_560,
                },
                "validation": {
                    "rows": 17_850,
                    "by_task": {
                        "translation": 12_133,
                        "summarization": 5_717,
                    },
                    "max_length": 9_600,
                    "over_max_count": 0,
                    "over_max_rows": [],
                    "minimum": 10,
                    "p50": 500,
                    "p90": 1_400,
                    "p95": 1_900,
                    "p99": 3_200,
                    "p99_9": 6_000,
                    "maximum": 7_000,
                },
            },
            "production_training_plan": production_training_plan,
            "subset_validation_plan": subset_validation_plan,
            "full_validation_plan": full_validation_plan,
            "validation_resume_contract": {
                "schema_version": 2,
                "progress_path": str(
                    self.adapter_dir.parent / "validation_progress.json"
                ),
                "commit_every_groups": 25,
                "bound_to_trainable_parameter_sha256": True,
                "signal_safe": True,
                "zero_drop": True,
            },
            "storage_audit": {
                "path": str(self.adapter_dir.parent),
                "free_bytes": 32 * 1024**3,
                "minimum_required_bytes": 16 * 1024**3,
                "status": "PASS",
            },
            "max_observed_tokens": 9_560,
            "total_hard_term_weighted_tokens": 2,
            "first_update_gradient_audit": gradient,
            "first_update_adapter_delta": update,
            "first_output_only_supervision": output_only,
            "microbatch_phase_seconds": microbatch_phase_seconds,
            "validation_row_seconds": validation_row_seconds,
            "validation_benchmark": validation_benchmark,
            "optimizer_boundary_seconds": optimizer_boundary_seconds,
            "one_time_gradient_audit_seconds": (one_time_gradient_audit_seconds),
            "runtime_budget_projection": runtime_budget,
            "gradient_checkpointing_contract": (gradient_checkpointing_contract),
            "post_update_recompute_audit": post_update_recompute_audit,
            "output_only_contract": recipe["output_only_contract"],
            "history": smoke_history,
            "early_stopping_decision": early_stopping_evidence,
            "frozen_probe_before": probe,
            "frozen_probe_after_update": probe,
            "final_checkpoint": {
                "step": 1,
                "adapter_dir": str(self.final_adapter_dir),
                "files_sha256": self.final_adapter_files,
            },
            "selection_record": {
                "candidate_steps": [1],
                "candidates": [
                    {
                        "step": 1,
                        "selection_loss": 1.175,
                        "translation_loss": 1.0,
                        "summarization_loss": 1.5,
                        "adapter_dir": str(self.final_adapter_dir),
                        "files_sha256": self.final_adapter_files,
                    }
                ],
                "selected_step": 1,
                "selected_selection_loss": 1.175,
                "policy": "smoke_final_step",
                "tie_break": "not_applicable",
            },
            "selected_checkpoint": {
                "policy": "smoke_final_step",
                "tie_break": "not_applicable",
                "step": 1,
                "selection_loss": 1.175,
                "source_checkpoint_dir": str(self.final_adapter_dir),
                "adapter_dir": str(self.adapter_dir),
                "files_sha256": self.adapter_files,
            },
            "lora_module_audit": {
                "status": "PASS",
                "module_count": certify_smoke.EXPECTED_LORA_MODULES,
                "module_class": "torch.nn.Linear",
            },
            "loader_audit": {
                "status": "PASS",
                "base_revision": BASE_REVISION,
                "hardware": {"device_count": 3},
                "source_inventory": {
                    "source_dir": str(self.source_dir),
                    "index_sha256": self.source["index_sha256"],
                    "config_sha256": self.source["config_sha256"],
                    "tensor_count": EXPECTED_SOURCE_TENSOR_COUNT,
                    "shard_count": EXPECTED_SOURCE_SHARD_COUNT,
                },
                "expert_preflight": {
                    "status": "PASS",
                    "production_eligible": True,
                    "checkpoint_dir": str(self.training_expert_dir),
                    "manifest_sha256": self.preflight["expert_artifact"][
                        "manifest_sha256"
                    ],
                    "expert_index_sha256": self.preflight["expert_artifact"][
                        "expert_index_sha256"
                    ],
                },
                "model_audit": {
                    "status": "PASS",
                    "lora_target_bf16_linear_count": (
                        certify_smoke.EXPECTED_LORA_MODULES
                    ),
                    "meta_parameter_count": 0,
                    "meta_buffer_count": 0,
                    "base_trainable_parameter_count": 0,
                    "model_parallel_hook_count": 51,
                    "gradient_checkpointing": True,
                    "device_aware_checkpoint_layer_count": 48,
                    "device_map": {
                        "model.embed_tokens": "cuda:0",
                        **{
                            f"model.layers.{layer}": f"cuda:{layer // 16}"
                            for layer in range(48)
                        },
                        "model.norm": "cuda:2",
                        "lm_head": "cuda:0",
                    },
                },
                "gradient_checkpointing": True,
                "gradient_checkpointing_contract": (gradient_checkpointing_contract),
            },
            "trainable_parameters": certify_smoke.EXPECTED_TRAINABLE_PARAMETERS,
            "device_audit": {
                "hf_device_map_devices": ["0", "1", "2"],
                "parameter_devices": {"cuda:0": 1, "cuda:1": 1, "cuda:2": 1},
                "cpu_disk_offload": False,
            },
            "sampler": {
                "base_rows": 142_713,
                "expert_rows": 905,
                "expert_boost": 8,
                "epochs": 2,
                "examples_per_epoch": 149_048,
                "total_examples": 298_096,
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
            },
            "smoke": smoke,
        }
        write_json(self.training_path, self.training)

    def _implementation(self) -> dict[str, Any]:
        identities = certify_smoke._current_identities()
        return {
            "infer_py_sha256": sha256(Path(infer.__file__)),
            "inference_loader_sha256": sha256(
                Path(infer.__file__).with_name("solar_inference_loader.py")
            ),
            "reference_evaluator_sha256": REFERENCE_SHA256,
            "model_contract_sha256": identities["model_contract_sha256"],
            "recipe_sha256": identities["recipe_sha256"],
            "data_contract_sha256": identities["data_contract_sha256"],
            "environment_lock_sha256": identities["environment_lock_sha256"],
            "pipeline_code_sha256": identities["pipeline_code"]["sha256"],
            "transformers_version": "fixture",
            "peft_version": "fixture",
            "bitsandbytes_version": "fixture",
        }

    def _contract(
        self,
        *,
        model_label: str,
        adapter: dict[str, Any] | None,
        keys: list[list[str]],
    ) -> dict[str, Any]:
        decoding = {
            "temperature": 0.0,
            "top_p": 1.0,
            "do_sample": False,
            "num_beams": 1,
            "max_model_len": 16384,
            "translation_max_new_tokens": 512,
            "summarization_max_new_tokens": 512,
            "padding_side": "left",
            "raw_output_policy": "outer strip only; no filtering or repair",
        }
        return {
            "schema": infer.PROVENANCE_SCHEMA,
            "schema_version": infer.PROVENANCE_SCHEMA_VERSION,
            "model_label": model_label,
            "source": self.source,
            "per_expert_nf4": self.inference_expert,
            "adapter": adapter,
            "evaluation": {
                "role": "full_test_plus_human_gold_sealed_union",
                "path": str(self.eval_path),
                "sha256": sha256(self.eval_path),
                "pinned_total_rows": infer.PINNED_COMBINED_EVAL_ROWS,
                "pinned_by_task": infer.PINNED_COMBINED_EVAL_TASK_COUNTS,
                "selected_rows": 2,
                "selected_keys_sha256": canonical_sha256(keys),
                "limit_per_task": 1,
            },
            "prompt": {
                "variant": "no_dictionary_context",
                **infer.TEMPLATE_KWARGS,
            },
            "decoding": decoding,
            "decoding_sha256": canonical_sha256(decoding),
            "runtime": {
                "strategy": "single_process_three_gpu_model_parallel",
                "routed_expert_format": "per_expert_nf4",
                "generation_batch_size": 2,
            },
            "implementation": self._implementation(),
        }

    def _write_predictions(
        self,
        path: Path,
        *,
        model_label: str,
        adapter: dict[str, Any] | None,
    ) -> None:
        sources = [
            {
                "id": "translation-1",
                "task": "translation",
                "origin": "옛말",
                "response": "modern text",
                "reference_source": "fixture",
            },
            {
                "id": "summarization-1",
                "task": "summarization",
                "origin": "long source",
                "response": "short reference",
                "reference_source": "fixture",
            },
        ]
        keys = [[row["id"], row["task"]] for row in sources]
        contract = self._contract(
            model_label=model_label,
            adapter=adapter,
            keys=keys,
        )
        contract_sha256 = canonical_sha256(contract)
        invariants = infer.expected_row_provenance(contract, contract_sha256)
        rows = [
            {
                "id": row["id"],
                "task": row["task"],
                **invariants,
                "origin": row["origin"],
                "reference": row["response"],
                "reference_source": row["reference_source"],
                "prompt_variant": "no_dictionary_context",
                "reasoning_effort": "none",
                "prediction": (
                    "옛말을 현대 한국어로 옮겼다."
                    if row["task"] == "translation"
                    else "긴 역사 기록의 핵심을 짧게 요약했다."
                ),
                "prompt_tokens": 8,
                "prediction_tokens": 3,
                "generation_stop_reason": "eos",
                "tokenizer_roundtrip_exact": True,
            }
            for row in sources
        ]
        write_jsonl(path, rows)
        adapter_audit = None
        if adapter is not None:
            adapter_audit = {
                "status": "PASS",
                "module_count": certify_smoke.EXPECTED_LORA_MODULES,
                "adapter_tensor_count": certify_smoke.EXPECTED_LORA_TENSORS,
                "adapter_tensor_names_sha256": adapter["serialized_keys_sha256"],
                "adapter_parameter_count": (
                    certify_smoke.EXPECTED_TRAINABLE_PARAMETERS
                ),
                "adapter_dtypes": {
                    "torch.float32": certify_smoke.EXPECTED_LORA_TENSORS
                },
                "trainable_parameter_count": 0,
                "offloaded_adapter_tensor_count": 0,
            }
        manifest = {
            "schema": infer.PROVENANCE_SCHEMA,
            "schema_version": infer.PROVENANCE_SCHEMA_VERSION,
            "status": "COMPLETE",
            "contract_sha256": contract_sha256,
            "contract": contract,
            "expected_rows": 2,
            "completed_rows": 2,
            "runtime_audit": {
                "model_audit": {"status": "PASS"},
                "adapter_audit": adapter_audit,
                "source_inventory": {
                    "source_dir": str(self.source_dir),
                    "index_sha256": self.source["index_sha256"],
                    "config_sha256": self.source["config_sha256"],
                    "weight_map_sha256": self.source_weight_map_sha256,
                    "shards": self.source_shards,
                },
                "expert_preflight": {
                    "status": "PASS",
                    "production_eligible": True,
                    "manifest_sha256": self.inference_expert["manifest_sha256"],
                    "expert_index_sha256": self.inference_expert["expert_index_sha256"],
                },
                "hash_verification": True,
                "production_eligible": True,
            },
            "output_jsonl_sha256": sha256(path),
        }
        write_json(path.with_name(path.name + ".manifest.json"), manifest)

    def _build_predictions(self) -> None:
        evaluation_rows = [
            {"id": "translation-1", "task": "translation"},
            {"id": "summarization-1", "task": "summarization"},
        ]
        write_jsonl(self.eval_path, evaluation_rows)
        self.evaluation = {
            "path": str(self.eval_path),
            "sha256": sha256(self.eval_path),
            "rows": infer.PINNED_COMBINED_EVAL_ROWS,
            "by_task": infer.PINNED_COMBINED_EVAL_TASK_COUNTS,
        }
        self._write_predictions(
            self.base_path,
            model_label="solar_open2_base_smoke",
            adapter=None,
        )
        self._write_predictions(
            self.tuned_path,
            model_label="solar_open2_tuned_smoke",
            adapter=self.adapter,
        )

    def _build(self) -> None:
        self._build_source()
        self._build_tokenizer_contract()
        self._build_experts()
        self._build_adapter()
        self._build_preflight()
        self._build_training()
        self._build_predictions()


class CertifySmokeTest(unittest.TestCase):
    def test_real_pinned_evaluation_identity_path(self) -> None:
        data_contract = json.loads(
            (CONFIG_DIR / "data_contract.json").read_text(encoding="utf-8")
        )
        entry = data_contract["files"]["combined_eval"]
        configured_path = (
            Path(data_contract["prepared_data_dir"]) / str(entry["name"])
        ).resolve()
        if not configured_path.exists() and not configured_path.is_symlink():
            self.skipTest(
                "source-profile pinned evaluation asset is not mounted in the "
                "owned B200 workspace"
            )
        identity = certify_smoke._expected_evaluation_identity()
        self.assertEqual(identity["sha256"], infer.PINNED_COMBINED_EVAL_SHA256)
        self.assertEqual(identity["rows"], infer.PINNED_COMBINED_EVAL_ROWS)
        self.assertEqual(
            identity["by_task"],
            infer.PINNED_COMBINED_EVAL_TASK_COUNTS,
        )

    @classmethod
    def setUpClass(cls) -> None:
        TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            dir=TEST_TEMP_ROOT,
            prefix="certify-smoke-",
        )
        self.fixture = CertificationFixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_valid_fixture_and_train_receipt_compatibility(self) -> None:
        output = Path(self.temporary.name) / "certification.json"
        result = certify_smoke.certify(**self.fixture.kwargs, output=output)
        self.assertEqual(result["status"], "PASS")
        self.assertNotIn("production_authorized", result)
        self.assertTrue(result["technical_production_gate_passed"])
        self.assertEqual(
            result["authorization_scope"],
            "technical_smoke_gate_only",
        )
        self.assertTrue(result["human_production_approval_required"])
        self.assertFalse(result["human_production_approval_recorded"])
        self.assertEqual(
            result["early_stopping_policy"],
            certify_smoke.EXPECTED_EARLY_STOPPING_POLICY,
        )
        self.assertEqual(
            result["early_stopping_decision"],
            self.fixture.training["early_stopping_decision"],
        )
        self.assertEqual(
            result["early_stopping_history"],
            self.fixture.training["history"],
        )
        self.assertEqual(
            result["early_stopping_decision"]["status"],
            "PENDING_MINIMUM",
        )
        self.assertFalse(result["early_stopping_decision"]["triggered"])
        self.assertEqual(
            result["expert_nf4_dir"],
            str(self.fixture.training_expert_dir),
        )
        self.assertEqual(
            result["pipeline_code"],
            certify_smoke.pipeline_code_identity(),
        )
        self.assertEqual(
            result["tokenizer_contract_receipt"],
            {
                "path": str(self.fixture.tokenizer_contract_path),
                "sha256": sha256(self.fixture.tokenizer_contract_path),
            },
        )
        for name in (
            "model_preflight",
            "tokenizer_output_contract",
            "max_length_backward",
            "lora_module_contract",
            "no_cpu_disk_offload",
            "base_inference",
            "lora_inference",
            "device_aware_gradient_checkpointing",
            "post_update_checkpoint_recompute",
            "runtime_budget_feasibility",
            "early_stopping_policy",
            "early_stopping_decision_recomputed",
            "production_first_update_learning_rate",
            "one_time_gradient_audit_projection",
        ):
            self.assertEqual(result["checks"][name], "PASS")
        self.assertEqual(
            result["smoke_first_update_learning_rate"],
            1e-4 / 558,
        )
        self.assertEqual(
            result["smoke_one_time_gradient_audit_seconds"],
            3.0,
        )
        self.assertEqual(
            result["gradient_checkpointing_contract"],
            self.fixture.training["gradient_checkpointing_contract"],
        )
        self.assertEqual(
            result["post_update_recompute_audit"],
            self.fixture.training["post_update_recompute_audit"],
        )
        self.assertEqual(
            result["validation_resume_contract"],
            {
                "schema_version": 2,
                "progress_path": str(
                    self.fixture.adapter_dir.parent / "validation_progress.json"
                ),
                "commit_every_groups": 25,
                "bound_to_trainable_parameter_sha256": True,
                "signal_safe": True,
                "zero_drop": True,
            },
        )
        budget = result["runtime_budget_projection"]
        self.assertEqual(budget["schema_version"], 5)
        self.assertEqual(
            budget["production_workload"]["train_microbatches"],
            58_096,
        )
        self.assertEqual(
            budget["production_workload"]["training_occurrences"],
            298_096,
        )
        self.assertEqual(
            budget["production_workload"]["total_validation_rows"],
            18 * 1_024 + 3 * 17_850,
        )
        self.assertEqual(
            budget["production_workload"]["optimizer_boundaries"],
            18_631,
        )
        self.assertEqual(
            budget["production_workload"]["full_gradient_audits"],
            1,
        )
        self.assertEqual(
            budget["production_workload"]["full_validation_steps"],
            [9_316, 13_973, 18_631],
        )
        early_projection = budget["early_stopping_projection"]
        self.assertEqual(early_projection["earliest_stop_step"], 13_973)
        self.assertEqual(early_projection["maximum_step"], 18_631)
        self.assertEqual(
            early_projection["saved_optimizer_updates"],
            4_658,
        )
        self.assertEqual(
            early_projection["maximum_training_fraction_saved"],
            4_658 / 18_631,
        )
        self.assertEqual(
            early_projection["subset_validation_steps"],
            list(range(1_000, 14_000, 1_000)),
        )
        self.assertEqual(
            early_projection["subset_validation_evaluations"],
            13,
        )
        self.assertEqual(
            early_projection["full_validation_steps"],
            [9_316, 13_973],
        )
        self.assertEqual(
            early_projection["full_validation_evaluations"],
            2,
        )
        self.assertEqual(
            early_projection["projected_validation_microbatches"],
            13 * self.fixture.training["subset_validation_plan"]["microbatches"]
            + 2 * self.fixture.training["full_validation_plan"]["microbatches"],
        )
        self.assertEqual(
            early_projection["raw_optimizer_boundary_seconds"],
            5.0 * 13_973,
        )
        self.assertEqual(
            early_projection["raw_one_time_gradient_audit_seconds"],
            3.0,
        )
        self.assertEqual(
            early_projection["raw_empirical_p50_total_seconds"],
            14.5 * 13_973
            + early_projection["raw_validation_seconds"]
            + 5.0 * 13_973
            + 3.0,
        )
        self.assertEqual(
            early_projection["raw_empirical_p90_total_seconds"],
            18.0 * 13_973
            + early_projection["raw_validation_seconds"]
            + 5.0 * 13_973
            + 3.0,
        )
        self.assertEqual(
            set(budget["measurement"]["training_bucket_benchmarks"]),
            {"1024", "2048", "4096", "9600"},
        )
        self.assertEqual(budget["projection"]["safety_factor"], 1.25)
        self.assertEqual(
            budget["projection"]["raw_optimizer_boundary_seconds"],
            5.0 * 18_631,
        )
        self.assertEqual(
            budget["projection"]["raw_one_time_gradient_audit_seconds"],
            3.0,
        )
        self.assertIn(
            "wall-clock completion guarantee",
            budget["limitations"]["interpretation"],
        )
        self.assertEqual(
            set(budget["measurement"]["validation_bucket_benchmarks"]),
            {"1024", "2048", "4096", "9600"},
        )

        self.assertEqual(
            budget["measurement"]["production_update_benchmark"],
            self.fixture.training["smoke"]["production_update_benchmark"],
        )
        self.assertEqual(
            budget["empirical_update_projection"]["estimator"],
            "ten_update_sample_p50_and_p90",
        )
        self.assertGreater(
            budget["allocation"]["projected_allocation_count"],
            0,
        )
        self.assertGreaterEqual(
            budget["allocation"]["projected_allocation_headroom_seconds"],
            0,
        )
        self.assertTrue(budget["allocation"]["explicit_requeue"])
        self.assertTrue(
            budget["allocation"]["max_batch_requeue_is_not_total_allocation_limit"]
        )
        self.assertEqual(
            json.loads(output.read_text(encoding="utf-8")),
            result,
        )
        self.assertEqual(
            train.validate_production_smoke(
                output,
                source_dir=self.fixture.source_dir,
                expert_nf4_dir=self.fixture.training_expert_dir,
                training_config=train.load_training_config(CONFIG_DIR),
            ),
            result,
        )

    def test_recipe_early_stopping_policy_is_exact_and_fail_closed(self) -> None:
        recipe = json.loads((CONFIG_DIR / "recipe.json").read_text(encoding="utf-8"))
        data_contract = json.loads(
            (CONFIG_DIR / "data_contract.json").read_text(encoding="utf-8")
        )
        certify_smoke._validate_recipe_v3(recipe, data_contract)
        mutations = {
            "disabled": lambda policy: policy.update({"enabled": False}),
            "sealed-monitor": lambda policy: policy.update(
                {"monitor": "human_gold_loss"}
            ),
            "early-minimum": lambda policy: policy.update({"minimum_step": 9_315}),
            "different-baseline": lambda policy: policy.update(
                {"subset_baseline_policy": "running_best"}
            ),
            "short-patience": lambda policy: policy.update(
                {"subset_patience_evaluations": 2}
            ),
            "weaker-subset-delta": lambda policy: policy["subset_min_delta"].update(
                {"absolute": 0.0}
            ),
            "no-full-confirmation": lambda policy: policy.update(
                {"full_validation_confirmation_required": False}
            ),
            "later-confirmation": lambda policy: policy.update(
                {"confirmation_step": 18_631}
            ),
            "weaker-full-delta": lambda policy: policy["full_min_delta"].update(
                {"relative": 0.0}
            ),
            "ignore-task-improvement": lambda policy: policy.update(
                {"continue_if_either_task_materially_improves": False}
            ),
            "premature-stop": lambda policy: policy.update(
                {"earliest_stop_step": 9_316}
            ),
            "unbounded-stop": lambda policy: policy.update(
                {"latest_stop_step": 20_000}
            ),
            "no-best-restore": lambda policy: policy.update(
                {"restore_best_full_validation": False}
            ),
            "sealed-eval-monitor": lambda policy: policy.update(
                {"sealed_evaluation_not_monitor": False}
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                changed = copy.deepcopy(recipe)
                mutate(changed["validation"]["early_stopping"])
                with self.assertRaisesRegex(
                    RuntimeError,
                    "recipe early-stopping policy",
                ):
                    certify_smoke._validate_recipe_v3(
                        changed,
                        data_contract,
                    )

    def test_rejects_tampered_early_stopping_decision_evidence(self) -> None:
        self.fixture.rewrite_manifest(
            self.fixture.training_path,
            lambda value: value["early_stopping_decision"].update(
                {"status": "CONTINUE"}
            ),
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "training early-stopping decision",
        ):
            certify_smoke.certify(**self.fixture.kwargs)

    def test_rejects_divergent_smoke_early_stopping_mirror(self) -> None:
        self.fixture.rewrite_manifest(
            self.fixture.training_path,
            lambda value: value["smoke"]["early_stopping_decision"].update(
                {"reason": "tampered"}
            ),
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "smoke early-stopping decision mirror",
        ):
            certify_smoke.certify(**self.fixture.kwargs)

    def test_train_rejects_tokenizer_receipt_changed_after_certification(
        self,
    ) -> None:
        output = Path(self.temporary.name) / "certification.json"
        certify_smoke.certify(**self.fixture.kwargs, output=output)
        with self.fixture.tokenizer_contract_path.open("a", encoding="utf-8") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(
            RuntimeError,
            "tokenizer contract receipt changed after smoke",
        ):
            train.validate_production_smoke(
                output,
                source_dir=self.fixture.source_dir,
                expert_nf4_dir=self.fixture.training_expert_dir,
                training_config=train.load_training_config(CONFIG_DIR),
            )

    def test_train_rejects_tampered_early_stopping_policy_seal(self) -> None:
        output = Path(self.temporary.name) / "certification.json"
        result = certify_smoke.certify(
            **self.fixture.kwargs,
            output=output,
        )
        result["early_stopping_policy"]["earliest_stop_step"] = 9_316
        write_json(output, result)
        with self.assertRaisesRegex(
            RuntimeError,
            "early-stopping policy changed",
        ):
            train.validate_production_smoke(
                output,
                source_dir=self.fixture.source_dir,
                expert_nf4_dir=self.fixture.training_expert_dir,
                training_config=train.load_training_config(CONFIG_DIR),
            )

    def test_train_recomputes_sealed_early_stopping_history(self) -> None:
        output = Path(self.temporary.name) / "certification.json"
        result = certify_smoke.certify(
            **self.fixture.kwargs,
            output=output,
        )
        result["early_stopping_history"][0]["selection_loss"] = 9.0
        write_json(output, result)
        with self.assertRaisesRegex(
            RuntimeError,
            "early-stopping decision changed",
        ):
            train.validate_production_smoke(
                output,
                source_dir=self.fixture.source_dir,
                expert_nf4_dir=self.fixture.training_expert_dir,
                training_config=train.load_training_config(CONFIG_DIR),
            )

    def test_rejects_incomplete_dynamic_bucket_benchmark(self) -> None:
        cases = (
            (
                "missing bucket",
                lambda benchmark: benchmark.pop("2048"),
                "bucket coverage",
            ),
            (
                "wrong target batch",
                lambda benchmark: benchmark["1024"].update({"batch_size": 7}),
                "batch_size",
            ),
            (
                "missing measured row",
                lambda benchmark: benchmark["4096"].update(
                    {"observed_lengths": [4_096]}
                ),
                "target-batch token lengths",
            ),
        )
        for label, mutate, message in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory(
                dir=TEST_TEMP_ROOT,
                prefix="certify-smoke-bucket-contract-",
            ) as root:
                fixture = CertificationFixture(Path(root))

                def mutate_manifest(value: dict[str, Any]) -> None:
                    mutate(value["validation_benchmark"]["training_buckets"])
                    mutate(value["smoke"]["validation_benchmark"]["training_buckets"])

                fixture.rewrite_manifest(
                    fixture.training_path,
                    mutate_manifest,
                )
                with self.assertRaisesRegex(RuntimeError, message):
                    certify_smoke.certify(**fixture.kwargs)

    def test_rejects_unstable_bucket_timing_evidence(self) -> None:
        cases = (
            (
                "missing warmup",
                lambda value: value.update({"warmup_repetitions": 0}),
                "warmup_repetitions",
            ),
            (
                "too few repeats",
                lambda value: value.update({"timed_repetitions": 2}),
                "timed_repetitions",
            ),
            (
                "invalid warmup time",
                lambda value: value.update({"warmup_seconds": 0.0}),
                "warmup seconds",
            ),
            (
                "too few samples",
                lambda value: value.update(
                    {"seconds_samples": value["seconds_samples"][:2]}
                ),
                "exactly 3 timed samples",
            ),
            (
                "incorrect p50",
                lambda value: value.update({"seconds_p50": 123.0}),
                "recomputed p50",
            ),
            (
                "projection alias differs",
                lambda value: value.update({"seconds": 123.0}),
                "projection timing alias",
            ),
            (
                "wrong estimator",
                lambda value: value.update({"estimator": "mean"}),
                "estimator",
            ),
            (
                "state changed",
                lambda value: value.update({"benchmark_state_unchanged": False}),
                "benchmark state",
            ),
            (
                "nonboolean state",
                lambda value: value.update({"benchmark_state_unchanged": 1}),
                "benchmark state",
            ),
        )
        for bucket_kind in ("training_buckets", "validation_buckets"):
            for label, mutate, message in cases:
                with self.subTest(
                    bucket_kind=bucket_kind,
                    label=label,
                ), tempfile.TemporaryDirectory(
                    dir=TEST_TEMP_ROOT,
                    prefix="certify-smoke-stable-bucket-",
                ) as root:
                    fixture = CertificationFixture(Path(root))

                    def mutate_manifest(value: dict[str, Any]) -> None:
                        mutate(value["validation_benchmark"][bucket_kind]["1024"])
                        mutate(
                            value["smoke"]["validation_benchmark"][bucket_kind]["1024"]
                        )

                    fixture.rewrite_manifest(
                        fixture.training_path,
                        mutate_manifest,
                    )
                    with self.assertRaisesRegex(RuntimeError, message):
                        certify_smoke.certify(**fixture.kwargs)

    def test_rejects_unstable_max_context_timing_evidence(self) -> None:
        cases = (
            (
                "single timed sample",
                lambda value: value.update(
                    {
                        "timed_repetitions": 1,
                        "seconds_samples": [value["seconds"]],
                    }
                ),
                "timed_repetitions",
            ),
            (
                "incorrect p50",
                lambda value: value.update({"seconds_p50": 123.0}),
                "recomputed p50",
            ),
            (
                "state changed",
                lambda value: value.update({"benchmark_state_unchanged": False}),
                "benchmark state",
            ),
        )
        for label, mutate, message in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory(
                dir=TEST_TEMP_ROOT,
                prefix="certify-smoke-stable-max-context-",
            ) as root:
                fixture = CertificationFixture(Path(root))

                def mutate_manifest(value: dict[str, Any]) -> None:
                    mutate(value["validation_benchmark"]["by_task"]["translation"])
                    mutate(
                        value["smoke"]["validation_benchmark"]["by_task"]["translation"]
                    )

                fixture.rewrite_manifest(
                    fixture.training_path,
                    mutate_manifest,
                )
                with self.assertRaisesRegex(RuntimeError, message):
                    certify_smoke.certify(**fixture.kwargs)

    def test_rejects_invalid_production_update_benchmark(self) -> None:
        cases = (
            (
                "wrong warmup count",
                lambda value: value.update({"warmup_repetitions": 1}),
                "warmup_repetitions",
            ),
            (
                "changed sampled steps",
                lambda value: value["timed_step_numbers"].__setitem__(0, 3),
                "timed steps",
            ),
            (
                "missing timed sample",
                lambda value: value.update(
                    {"seconds_samples": value["seconds_samples"][:-1]}
                ),
                "exactly 10 timed samples",
            ),
            (
                "incorrect p50",
                lambda value: value.update({"seconds_p50": 123.0}),
                "recomputed p50",
            ),
            (
                "incorrect p90",
                lambda value: value.update({"seconds_p90": 123.0}),
                "recomputed p90",
            ),
            (
                "incorrect mean",
                lambda value: value.update({"seconds_mean": 123.0}),
                "recomputed mean",
            ),
            (
                "changed state",
                lambda value: value.update({"benchmark_state_unchanged": False}),
                "benchmark state",
            ),
            (
                "invalid microbatch samples",
                lambda value: value["microbatch_samples"].__setitem__(0, 0),
                "invalid microbatch samples",
            ),
        )
        for label, mutate, message in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory(
                dir=TEST_TEMP_ROOT,
                prefix="certify-smoke-production-update-",
            ) as root:
                fixture = CertificationFixture(Path(root))

                def mutate_manifest(value: dict[str, Any]) -> None:
                    for benchmark in (
                        value["validation_benchmark"]["production_update_benchmark"],
                        value["smoke"]["validation_benchmark"][
                            "production_update_benchmark"
                        ],
                        value["smoke"]["production_update_benchmark"],
                    ):
                        mutate(benchmark)

                fixture.rewrite_manifest(
                    fixture.training_path,
                    mutate_manifest,
                )
                with self.assertRaisesRegex(RuntimeError, message):
                    certify_smoke.certify(**fixture.kwargs)

    def test_rejects_missing_absolute_longest_training_evidence(self) -> None:
        cases = (
            (
                "token scan",
                lambda value: value["token_length_contract"]["train"].update(
                    {"maximum": 9_559}
                ),
                "absolute maximum",
            ),
            (
                "first update",
                lambda value: value["smoke"].update({"observed_tokens": 9_559}),
                "absolute longest",
            ),
            (
                "bucket F+B",
                lambda value: (
                    value["validation_benchmark"]["training_buckets"]["9600"].update(
                        {"observed_lengths": [9_559]}
                    ),
                    value["smoke"]["validation_benchmark"]["training_buckets"][
                        "9600"
                    ].update({"observed_lengths": [9_559]}),
                ),
                "absolute longest training row benchmark",
            ),
        )
        for label, mutate, message in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory(
                dir=TEST_TEMP_ROOT,
                prefix="certify-smoke-longest-contract-",
            ) as root:
                fixture = CertificationFixture(Path(root))
                fixture.rewrite_manifest(fixture.training_path, mutate)
                with self.assertRaisesRegex(RuntimeError, message):
                    certify_smoke.certify(**fixture.kwargs)

    def test_rejects_drop_or_nonresumable_validation_evidence(self) -> None:
        cases = (
            (
                "training occurrence drop",
                lambda value: value["training_coverage"].update(
                    {"skipped_occurrences": 1}
                ),
                "skipped_occurrences",
            ),
            (
                "obsolete validation resume schema",
                lambda value: value["validation_resume_contract"].update(
                    {"schema_version": 1}
                ),
                "schema_version",
            ),
            (
                "noninteger validation resume schema",
                lambda value: value["validation_resume_contract"].update(
                    {"schema_version": 2.0}
                ),
                "schema_version",
            ),
            (
                "validation progress not bound to model state",
                lambda value: value["validation_resume_contract"].update(
                    {"bound_to_trainable_parameter_sha256": False}
                ),
                "bound_to_trainable_parameter_sha256",
            ),
            (
                "nonboolean validation progress binding",
                lambda value: value["validation_resume_contract"].update(
                    {"bound_to_trainable_parameter_sha256": 1}
                ),
                "bound_to_trainable_parameter_sha256",
            ),
            (
                "validation zero-drop flag",
                lambda value: value["validation_resume_contract"].update(
                    {"zero_drop": False}
                ),
                "zero_drop",
            ),
            (
                "partial resume signal safety",
                lambda value: value["validation_resume_contract"].update(
                    {"signal_safe": False}
                ),
                "signal_safe",
            ),
            (
                "validation history drop",
                lambda value: value["history"][0].update(
                    {"dropped_over_max": {"translation": 1}}
                ),
                "zero-drop",
            ),
        )
        for label, mutate, message in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory(
                dir=TEST_TEMP_ROOT,
                prefix="certify-smoke-zero-drop-",
            ) as root:
                fixture = CertificationFixture(Path(root))
                fixture.rewrite_manifest(fixture.training_path, mutate)
                with self.assertRaisesRegex(RuntimeError, message):
                    certify_smoke.certify(**fixture.kwargs)

    def test_rejects_inexact_two_epoch_production_plan(self) -> None:
        def mutate(value: dict[str, Any]) -> None:
            value["smoke"]["production_training_plan"]["occurrences"] = 298_095

        self.fixture.rewrite_manifest(self.fixture.training_path, mutate)
        with self.assertRaisesRegex(RuntimeError, "occurrences"):
            certify_smoke.certify(**self.fixture.kwargs)

    def test_rejects_insufficient_or_unsealed_storage_audit(self) -> None:
        cases = (
            (
                "failed status",
                lambda audit: audit.update({"status": "FAIL"}),
                "storage audit status",
            ),
            (
                "below minimum",
                lambda audit: audit.update(
                    {"free_bytes": audit["minimum_required_bytes"] - 1}
                ),
                "insufficient free bytes",
            ),
            (
                "weak minimum",
                lambda audit: audit.update({"minimum_required_bytes": 8 * 1024**3}),
                "16-GiB",
            ),
        )
        for label, mutate, message in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory(
                dir=TEST_TEMP_ROOT,
                prefix="certify-smoke-storage-",
            ) as root:
                fixture = CertificationFixture(Path(root))

                def mutate_manifest(value: dict[str, Any]) -> None:
                    mutate(value["storage_audit"])

                fixture.rewrite_manifest(
                    fixture.training_path,
                    mutate_manifest,
                )
                with self.assertRaisesRegex(RuntimeError, message):
                    certify_smoke.certify(**fixture.kwargs)

    def test_runtime_projection_separates_train_and_validation_costs(self) -> None:
        microbatch_seconds = 30.0
        validation_seconds = 5.0
        boundary_seconds = 2.0
        audit_seconds = 7.0
        recipe = json.loads((CONFIG_DIR / "recipe.json").read_text(encoding="utf-8"))
        data_contract = json.loads(
            (CONFIG_DIR / "data_contract.json").read_text(encoding="utf-8")
        )
        training_projection = train.runtime_budget_projection(
            training_bucket_benchmarks={
                key: {
                    **value,
                    "seconds": microbatch_seconds * (index + 1) / 4,
                }
                for index, (key, value) in enumerate(
                    self.fixture.training["validation_benchmark"][
                        "training_buckets"
                    ].items()
                )
            },
            validation_bucket_benchmarks=self.fixture.training["validation_benchmark"][
                "validation_buckets"
            ],
            production_update_benchmark=self.fixture.training["validation_benchmark"][
                "production_update_benchmark"
            ],
            validation_row_seconds=validation_seconds,
            optimizer_boundary_seconds=boundary_seconds,
            one_time_gradient_audit_seconds=audit_seconds,
            recipe=recipe,
            validation_row_count=data_contract["files"]["validation"]["rows"],
            production_training_plan=self.fixture.training["smoke"][
                "production_training_plan"
            ],
            subset_validation_plan=self.fixture.training["smoke"][
                "subset_validation_plan"
            ],
            full_validation_plan=self.fixture.training["smoke"]["full_validation_plan"],
        )
        benchmarks = training_projection["measurement"]["training_bucket_benchmarks"]
        certification_projection = certify_smoke._runtime_budget_projection(
            benchmarks,
            self.fixture.training["validation_benchmark"]["validation_buckets"],
            self.fixture.training["validation_benchmark"][
                "production_update_benchmark"
            ],
            validation_seconds,
            boundary_seconds,
            audit_seconds,
            self.fixture.training["smoke"]["production_training_plan"],
            self.fixture.training["smoke"]["subset_validation_plan"],
            self.fixture.training["smoke"]["full_validation_plan"],
        )
        self.assertEqual(training_projection, certification_projection)
        validation_benchmarks = self.fixture.training["validation_benchmark"][
            "validation_buckets"
        ]
        subset_plan = self.fixture.training["smoke"]["subset_validation_plan"]
        full_plan = self.fixture.training["smoke"]["full_validation_plan"]
        single_subset_seconds = sum(
            subset_plan["bucket_microbatches"][key]
            * validation_benchmarks[key]["seconds"]
            for key in validation_benchmarks
        )
        single_full_seconds = sum(
            full_plan["bucket_microbatches"][key]
            * validation_benchmarks[key]["seconds"]
            for key in validation_benchmarks
        )
        expected_subset_seconds = 18 * single_subset_seconds
        expected_full_seconds = 3 * single_full_seconds
        self.assertEqual(
            training_projection["projection"]["raw_validation_seconds"],
            expected_subset_seconds + expected_full_seconds,
        )
        early_projection = training_projection["early_stopping_projection"]
        expected_early_validation_seconds = (
            13 * single_subset_seconds + 2 * single_full_seconds
        )
        self.assertEqual(
            early_projection["raw_validation_seconds"],
            expected_early_validation_seconds,
        )
        self.assertEqual(
            early_projection["subset_validation_steps"],
            list(range(1_000, 14_000, 1_000)),
        )
        self.assertEqual(
            early_projection["full_validation_steps"],
            [9_316, 13_973],
        )
        self.assertEqual(
            early_projection["saved_optimizer_updates"],
            4_658,
        )
        self.assertEqual(
            early_projection["maximum_training_fraction_saved"],
            4_658 / 18_631,
        )
        update_benchmark = self.fixture.training["validation_benchmark"][
            "production_update_benchmark"
        ]
        for estimator in ("p50", "p90"):
            expected_raw = (
                update_benchmark[f"seconds_{estimator}"] * 13_973
                + expected_early_validation_seconds
                + boundary_seconds * 13_973
                + audit_seconds
            )
            self.assertEqual(
                early_projection[f"raw_empirical_{estimator}_total_seconds"],
                expected_raw,
            )
            self.assertEqual(
                early_projection[f"projected_empirical_{estimator}_total_seconds"],
                expected_raw * certify_smoke.RUNTIME_SAFETY_FACTOR,
            )
        self.assertEqual(
            training_projection["projection"]["raw_optimizer_boundary_seconds"],
            boundary_seconds * 18_631,
        )
        self.assertEqual(
            training_projection["projection"]["raw_one_time_gradient_audit_seconds"],
            audit_seconds,
        )
        self.assertEqual(
            training_projection["production_workload"]["full_gradient_audits"],
            1,
        )
        projection = training_projection["projection"]
        self.assertEqual(
            projection["raw_total_seconds"],
            projection["raw_train_microbatch_seconds"]
            + projection["raw_validation_seconds"]
            + projection["raw_optimizer_boundary_seconds"]
            + audit_seconds,
        )
        self.assertEqual(
            training_projection["measurement"]["optimizer_boundary_scope"],
            "first real gradient clipping and AdamW step; includes initial "
            "optimizer-state allocation",
        )
        self.assertTrue(
            training_projection["feasibility"]["validation_progress_is_resumable"]
        )
        empirical = training_projection["empirical_update_projection"]
        allocation = training_projection["allocation"]
        self.assertEqual(
            allocation["projection_basis"],
            "empirical_update_projection.projected_p50_total_seconds",
        )
        self.assertEqual(
            allocation["projected_allocation_count"],
            max(
                1,
                math.ceil(
                    empirical["projected_p50_total_seconds"]
                    / certify_smoke.USABLE_ALLOCATION_SECONDS
                ),
            ),
        )
        self.assertEqual(
            allocation["projected_allocation_headroom_seconds"],
            allocation["projected_allocation_count"]
            * certify_smoke.USABLE_ALLOCATION_SECONDS
            - empirical["projected_p50_total_seconds"],
        )

    def test_rejects_tampered_earliest_stop_runtime_projection(self) -> None:
        def mutate(value: dict[str, Any]) -> None:
            for runtime in (
                value["runtime_budget_projection"],
                value["smoke"]["runtime_budget_projection"],
            ):
                runtime["early_stopping_projection"]["saved_optimizer_updates"] = 4_657

        self.fixture.rewrite_manifest(self.fixture.training_path, mutate)
        with self.assertRaisesRegex(
            RuntimeError,
            "smoke runtime budget projection",
        ):
            certify_smoke.certify(**self.fixture.kwargs)

    def test_rejects_nonpositive_one_time_gradient_audit(self) -> None:
        for invalid in (0.0, -1.0, False, None):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory(
                dir=TEST_TEMP_ROOT,
                prefix="certify-smoke-gradient-audit-time-",
            ) as root:
                fixture = CertificationFixture(Path(root))

                def mutate(value: dict[str, Any]) -> None:
                    value["one_time_gradient_audit_seconds"] = invalid
                    value["smoke"]["one_time_gradient_audit_seconds"] = invalid

                fixture.rewrite_manifest(fixture.training_path, mutate)
                with self.assertRaisesRegex(
                    RuntimeError,
                    "one_time_gradient_audit_seconds",
                ):
                    certify_smoke.certify(**fixture.kwargs)

    def test_train_rejects_tampered_tokenizer_receipt_seal(self) -> None:
        cases = {
            "identity": lambda value: value["tokenizer_contract_receipt"].update(
                {"sha256": "0" * 64}
            ),
            "required check": lambda value: value["checks"].update(
                {"tokenizer_output_contract": "FAIL"}
            ),
        }
        for label, mutate in cases.items():
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory(
                    dir=TEST_TEMP_ROOT,
                    prefix="train-tokenizer-receipt-",
                ) as root:
                    fixture = CertificationFixture(Path(root))
                    output = Path(root) / "certification.json"
                    certify_smoke.certify(**fixture.kwargs, output=output)
                    fixture.rewrite_manifest(output, mutate)
                    with self.assertRaises(RuntimeError):
                        train.validate_production_smoke(
                            output,
                            source_dir=fixture.source_dir,
                            expert_nf4_dir=fixture.training_expert_dir,
                            training_config=train.load_training_config(CONFIG_DIR),
                        )

    def test_rejects_stale_preflight_pipeline(self) -> None:
        self.fixture.rewrite_manifest(
            self.fixture.preflight_path,
            lambda value: value["pipeline_code"].update({"sha256": "0" * 64}),
        )
        with self.assertRaisesRegex(RuntimeError, "preflight pipeline_code"):
            certify_smoke.certify(**self.fixture.kwargs)

    def test_rejects_tampered_tokenizer_contract_receipt(self) -> None:
        cases = {
            "status": lambda value: value.update({"status": "FAIL"}),
            "user": lambda value: value.update({"user": "mk01"}),
            "source revision": lambda value: value["source"].update(
                {"revision": "wrong"}
            ),
            "source path": lambda value: value["source"].update({"path": "/tmp/wrong"}),
            "source tokenizer hash": lambda value: value["source"][
                "tokenizer_files_sha256"
            ].update({"tokenizer.json": "0" * 64}),
            "source generation EOS": lambda value: value["source"].update(
                {"generation_eos_token_ids": [129]}
            ),
            "data contract hash": lambda value: value["data"].update(
                {"data_contract_sha256": "0" * 64}
            ),
            "combined data hash": lambda value: value["data"]["files"][
                "combined_eval"
            ].update({"sha256": "0" * 64}),
            "combined data count": lambda value: value["data"]["files"][
                "combined_eval"
            ].update({"rows": 19_777}),
            "reference hash": lambda value: value["reference"].update(
                {"sha256": "0" * 64}
            ),
            "supervised termination": lambda value: value[
                "output_only_contract"
            ].update({"supervised_termination_token_ids": [129]}),
            "configured generation EOS": lambda value: value[
                "output_only_contract"
            ].update({"configured_generation_eos_token_ids": [2]}),
            "template boolean type": lambda value: value["output_only_contract"][
                "template_kwargs"
            ].update({"provider_system_prompt": 0}),
            "row termination count": lambda value: value["output_only_contract"][
                "rows"
            ][0].update({"assistant_termination_token_count": 2}),
            "row EOS supervision": lambda value: value["output_only_contract"]["rows"][
                0
            ].update({"supervised_labels_are_body_plus_single_eos": False}),
            "row EOS boolean type": lambda value: value["output_only_contract"]["rows"][
                0
            ].update({"supervised_labels_are_body_plus_single_eos": 1}),
            "combined truncation": lambda value: value[
                "combined_eval_prompt_budget"
            ].update({"truncation_count": 1}),
            "combined truncation boolean": lambda value: value[
                "combined_eval_prompt_budget"
            ].update({"truncation_count": False}),
            "combined preamble count": lambda value: value[
                "combined_eval_prompt_budget"
            ].update({"exact_output_only_preamble_count": 19_777}),
            "pipeline": lambda value: value["pipeline_code"].update(
                {"sha256": "0" * 64}
            ),
            "required EOS check": lambda value: value["checks"].update(
                {"termination_is_configured_generation_eos": "FAIL"}
            ),
        }
        for label, mutate in cases.items():
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory(
                    dir=TEST_TEMP_ROOT,
                    prefix="certify-smoke-tokenizer-contract-",
                ) as root:
                    fixture = CertificationFixture(Path(root))
                    fixture.rewrite_manifest(
                        fixture.tokenizer_contract_path,
                        mutate,
                    )
                    with self.assertRaises(RuntimeError):
                        certify_smoke.certify(**fixture.kwargs)

    def test_rejects_symlinked_tokenizer_contract_receipt(self) -> None:
        symlink = Path(self.temporary.name) / "tokenizer-contract-link.json"
        symlink.symlink_to(self.fixture.tokenizer_contract_path)
        kwargs = {**self.fixture.kwargs, "tokenizer_contract_receipt": symlink}
        with self.assertRaisesRegex(RuntimeError, "missing or symlinked"):
            certify_smoke.certify(**kwargs)

    def test_rejects_adapter_changed_after_training(self) -> None:
        adapter_weights = self.fixture.adapter_dir / "adapter_model.safetensors"
        with adapter_weights.open("ab") as handle:
            handle.write(b"tamper")
        with self.assertRaisesRegex(RuntimeError, "selected adapter file hashes"):
            certify_smoke.certify(**self.fixture.kwargs)

    def test_rejects_invalid_output_only_termination_evidence(self) -> None:
        cases = {
            "multiple termination tokens": (
                "assistant_termination_token_count",
                2,
            ),
            "boolean termination token count": (
                "assistant_termination_token_count",
                True,
            ),
            "negative termination token id": (
                "assistant_termination_token_id",
                -1,
            ),
            "boolean termination token id": (
                "assistant_termination_token_id",
                True,
            ),
            "empty template suffix": ("template_suffix_token_count", 0),
            "boolean template suffix count": (
                "template_suffix_token_count",
                True,
            ),
            "template suffix without EOS": (
                "template_suffix_starts_with_eos",
                False,
            ),
            "supervised sequence without final EOS": (
                "supervised_sequence_ends_at_eos",
                False,
            ),
        }
        for label, (field, invalid_value) in cases.items():
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory(
                    dir=TEST_TEMP_ROOT,
                    prefix="certify-smoke-output-only-",
                ) as root:
                    fixture = CertificationFixture(Path(root))

                    def mutate(value: dict[str, Any]) -> None:
                        value["first_output_only_supervision"][field] = invalid_value
                        value["smoke"]["output_only_supervision"][field] = invalid_value

                    fixture.rewrite_manifest(fixture.training_path, mutate)
                    with self.assertRaises(RuntimeError):
                        certify_smoke.certify(**fixture.kwargs)

    def test_rejects_prediction_changed_after_sealing(self) -> None:
        with self.fixture.base_path.open("a", encoding="utf-8") as handle:
            handle.write("{}\n")
        with self.assertRaisesRegex(RuntimeError, "output_jsonl_sha256"):
            certify_smoke.certify(**self.fixture.kwargs)

    def test_rejects_framed_smoke_prediction(self) -> None:
        rows = [
            json.loads(line)
            for line in self.fixture.base_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        rows[0]["prediction"] = "최종 답변: 현대 한국어 본문"
        write_jsonl(self.fixture.base_path, rows)
        manifest = self.fixture.base_path.with_name(
            self.fixture.base_path.name + ".manifest.json"
        )
        self.fixture.rewrite_manifest(
            manifest,
            lambda value: value.update(
                {"output_jsonl_sha256": sha256(self.fixture.base_path)}
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "non-body prediction"):
            certify_smoke.certify(**self.fixture.kwargs)

    def test_rejects_corrupt_or_unterminated_smoke_prediction(self) -> None:
        cases = {
            "generation stop reason": ("generation_stop_reason", "max_new_tokens"),
            "tokenizer roundtrip.*False": ("tokenizer_roundtrip_exact", False),
            "tokenizer roundtrip.*1": ("tokenizer_roundtrip_exact", 1),
            "exhausted its generation cap": ("prediction_tokens", 512),
            "invalid Unicode": ("prediction", "깨진 문자 \ufffd"),
        }
        for message, (field, value) in cases.items():
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory(
                    dir=TEST_TEMP_ROOT,
                    prefix="certify-smoke-prediction-integrity-",
                ) as root:
                    fixture = CertificationFixture(Path(root))
                    rows = [
                        json.loads(line)
                        for line in fixture.base_path.read_text(
                            encoding="utf-8"
                        ).splitlines()
                        if line
                    ]
                    rows[0][field] = value
                    write_jsonl(fixture.base_path, rows)
                    manifest = fixture.base_path.with_name(
                        fixture.base_path.name + ".manifest.json"
                    )
                    fixture.rewrite_manifest(
                        manifest,
                        lambda item: item.update(
                            {"output_jsonl_sha256": sha256(fixture.base_path)}
                        ),
                    )
                    with self.assertRaisesRegex(RuntimeError, message):
                        certify_smoke.certify(**fixture.kwargs)

    def test_rejects_changed_smoke_decoding_limits(self) -> None:
        manifest = self.fixture.base_path.with_name(
            self.fixture.base_path.name + ".manifest.json"
        )

        def mutate(value: dict[str, Any]) -> None:
            contract = value["contract"]
            contract["decoding"]["translation_max_new_tokens"] = 511
            contract["decoding_sha256"] = canonical_sha256(contract["decoding"])
            value["contract_sha256"] = canonical_sha256(contract)

        self.fixture.rewrite_manifest(manifest, mutate)
        with self.assertRaisesRegex(
            RuntimeError,
            "base decoding translation_max_new_tokens",
        ):
            certify_smoke.certify(**self.fixture.kwargs)

    def test_rejects_changed_smoke_generation_batch_size(self) -> None:
        manifest = self.fixture.base_path.with_name(
            self.fixture.base_path.name + ".manifest.json"
        )

        def mutate(value: dict[str, Any]) -> None:
            contract = value["contract"]
            contract["runtime"]["generation_batch_size"] = 1
            value["contract_sha256"] = canonical_sha256(contract)

        self.fixture.rewrite_manifest(manifest, mutate)
        with self.assertRaisesRegex(RuntimeError, "generation batch size"):
            certify_smoke.certify(**self.fixture.kwargs)

    def test_rejects_invalid_effective_optimizer_update_evidence(self) -> None:
        cases = (
            ("optimizer_step", 2, "effective update optimizer step"),
            ("learning_rate_used", 0.0, "effective update learning rate"),
            ("learning_rate_used", 1e-4, "effective update learning rate"),
            ("changed", 1, "update changed flag"),
        )
        for field, invalid, message in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory(
                dir=TEST_TEMP_ROOT,
                prefix="certify-smoke-update-evidence-",
            ) as root:
                fixture = CertificationFixture(Path(root))

                def mutate(value: dict[str, Any]) -> None:
                    value["smoke"]["adapter_update"][field] = invalid

                fixture.rewrite_manifest(fixture.training_path, mutate)
                with self.assertRaisesRegex(RuntimeError, message):
                    certify_smoke.certify(**fixture.kwargs)

    def test_rejects_invalid_device_aware_checkpoint_contract(self) -> None:
        cases = (
            ("use_reentrant", True, "use_reentrant"),
            ("preserve_rng_state", 1, "preserve_rng_state"),
            ("early_stop", False, "early_stop"),
            ("boundary_layers", [16, 31], "boundary_layers"),
            ("decoder_layer_count", True, "decoder_layer_count"),
        )
        for field, invalid, message in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory(
                dir=TEST_TEMP_ROOT,
                prefix="certify-smoke-checkpoint-contract-",
            ) as root:
                fixture = CertificationFixture(Path(root))

                def mutate(value: dict[str, Any]) -> None:
                    value["gradient_checkpointing_contract"][field] = invalid
                    value["loader_audit"]["gradient_checkpointing_contract"][
                        field
                    ] = invalid

                fixture.rewrite_manifest(fixture.training_path, mutate)
                with self.assertRaisesRegex(RuntimeError, message):
                    certify_smoke.certify(**fixture.kwargs)

    def test_rejects_invalid_post_update_recompute_evidence(self) -> None:
        cases = (
            ("observed_tokens", True, "observed tokens"),
            ("observed_tokens", 9_559, "absolute-longest"),
            ("observed_tokens", 9_601, "absolute-longest"),
            ("hard_term_weighted_tokens", -1, "hard-term weighted tokens"),
            ("row_id", "different-row", "absolute-longest row"),
            ("loss", float("nan"), "recompute loss"),
            ("seconds", 0.0, "recompute seconds"),
            ("adapter_parameters_unchanged", 1, "adapter unchanged"),
            ("frozen_parameters_unchanged", False, "frozen parameters"),
        )
        for field, invalid, message in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory(
                dir=TEST_TEMP_ROOT,
                prefix="certify-smoke-post-update-",
            ) as root:
                fixture = CertificationFixture(Path(root))

                def mutate(value: dict[str, Any]) -> None:
                    value["post_update_recompute_audit"][field] = invalid
                    value["smoke"]["post_update_recompute_audit"][field] = invalid

                fixture.rewrite_manifest(fixture.training_path, mutate)
                with self.assertRaisesRegex(RuntimeError, message):
                    certify_smoke.certify(**fixture.kwargs)

    def test_rejects_invalid_post_update_recompute_gradient(self) -> None:
        cases = (
            ("finite", 1, "gradient finite"),
            ("finite", False, "gradient finite"),
            ("parameters_with_gradient", 0, "parameters with gradient"),
            ("nonzero_gradient_elements", 0, "nonzero gradient elements"),
            ("l2", 0.0, "gradient l2"),
        )
        for field, invalid, message in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory(
                dir=TEST_TEMP_ROOT,
                prefix="certify-smoke-post-update-gradient-",
            ) as root:
                fixture = CertificationFixture(Path(root))

                def mutate(value: dict[str, Any]) -> None:
                    for replay in (
                        value["post_update_recompute_audit"],
                        value["smoke"]["post_update_recompute_audit"],
                    ):
                        replay["gradient_audit"][field] = invalid

                fixture.rewrite_manifest(fixture.training_path, mutate)
                with self.assertRaisesRegex(RuntimeError, message):
                    certify_smoke.certify(**fixture.kwargs)

    def test_rejects_missing_prediction_runtime_identity(self) -> None:
        manifest = self.fixture.base_path.with_name(
            self.fixture.base_path.name + ".manifest.json"
        )
        self.fixture.rewrite_manifest(
            manifest,
            lambda value: value["runtime_audit"].pop("source_inventory"),
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "recorded runtime payload identities are missing",
        ):
            certify_smoke.certify(**self.fixture.kwargs)

    def test_rejects_non_boolean_prediction_runtime_flags(self) -> None:
        for field, message in (
            ("hash_verification", "runtime hash verification"),
            ("production_eligible", "runtime production eligibility"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory(
                dir=TEST_TEMP_ROOT,
                prefix="certify-smoke-runtime-flags-",
            ) as root:
                fixture = CertificationFixture(Path(root))
                manifest = fixture.base_path.with_name(
                    fixture.base_path.name + ".manifest.json"
                )
                fixture.rewrite_manifest(
                    manifest,
                    lambda value: value["runtime_audit"].update({field: 1}),
                )
                with self.assertRaisesRegex(RuntimeError, message):
                    certify_smoke.certify(**fixture.kwargs)

    def test_rejects_tuned_runtime_adapter_audit_mismatch(self) -> None:
        manifest = self.fixture.tuned_path.with_name(
            self.fixture.tuned_path.name + ".manifest.json"
        )
        self.fixture.rewrite_manifest(
            manifest,
            lambda value: value["runtime_audit"]["adapter_audit"].update(
                {"adapter_parameter_count": 1}
            ),
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "tuned adapter audit adapter_parameter_count",
        ):
            certify_smoke.certify(**self.fixture.kwargs)

    def test_full_validation_can_cross_allocation_with_partial_resume(self) -> None:
        full_microbatches = self.fixture.training["smoke"]["full_validation_plan"][
            "microbatches"
        ]
        oversized = (
            certify_smoke.USABLE_ALLOCATION_SECONDS
            / (certify_smoke.RUNTIME_SAFETY_FACTOR * full_microbatches)
            * 1.01
        )

        def mutate(value: dict[str, Any]) -> None:
            for location in (
                value["validation_benchmark"]["validation_buckets"],
                value["smoke"]["validation_benchmark"]["validation_buckets"],
            ):
                for benchmark in location.values():
                    benchmark["seconds_samples"] = [oversized] * 3
                    benchmark["seconds_p50"] = oversized
                    benchmark["seconds"] = oversized
            runtime = certify_smoke._runtime_budget_projection(
                value["smoke"]["validation_benchmark"]["training_buckets"],
                value["smoke"]["validation_benchmark"]["validation_buckets"],
                value["smoke"]["production_update_benchmark"],
                value["validation_row_seconds"],
                value["optimizer_boundary_seconds"],
                value["one_time_gradient_audit_seconds"],
                value["smoke"]["production_training_plan"],
                value["smoke"]["subset_validation_plan"],
                value["smoke"]["full_validation_plan"],
            )
            value["runtime_budget_projection"] = runtime
            value["smoke"]["runtime_budget_projection"] = runtime

        self.fixture.rewrite_manifest(self.fixture.training_path, mutate)
        result = certify_smoke.certify(**self.fixture.kwargs)
        self.assertTrue(
            result["runtime_budget_projection"]["projection"][
                "full_validation_requires_partial_resume"
            ]
        )
        self.assertTrue(
            result["runtime_budget_projection"]["feasibility"][
                "validation_progress_is_resumable"
            ]
        )

    def test_rejects_optimizer_boundary_over_usable_allocation(self) -> None:
        oversized = (
            certify_smoke.USABLE_ALLOCATION_SECONDS
            / certify_smoke.RUNTIME_SAFETY_FACTOR
            * 1.01
        )

        def mutate(value: dict[str, Any]) -> None:
            value["optimizer_boundary_seconds"] = oversized
            value["smoke"]["optimizer_boundary_seconds"] = oversized

        self.fixture.rewrite_manifest(self.fixture.training_path, mutate)
        with self.assertRaisesRegex(
            RuntimeError,
            "single_optimizer_boundary",
        ):
            certify_smoke.certify(**self.fixture.kwargs)

    def test_cli_exposes_separate_training_and_inference_artifacts(self) -> None:
        option_strings = {
            option
            for action in certify_smoke.build_parser()._actions
            for option in action.option_strings
        }
        self.assertTrue(
            {
                "--source-dir",
                "--training-expert-nf4-dir",
                "--inference-expert-nf4-dir",
                "--tokenizer-contract-receipt",
                "--preflight-receipt",
                "--training-manifest",
                "--base-predictions",
                "--tuned-predictions",
                "--output",
            }.issubset(option_strings)
        )


class B200CertifierContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _gpu_evidence() -> dict[str, Any]:
        return {
            "status": "PASS",
            "physical_cuda_visible_devices": "6,7",
            "logical_gpu_count": 2,
            "boundary_layers": [24],
            "torch_arch_list": ["sm_100"],
            "logical_to_physical": [
                {
                    "logical_index": index,
                    "logical_device": f"cuda:{index}",
                    "physical_index": 6 + index,
                    "compute_capability": [10, 0],
                    "name": "NVIDIA B200",
                    "total_memory_bytes": (
                        certify_smoke.b200_runtime.MINIMUM_MEMORY_BYTES_PER_GPU
                    ),
                }
                for index in range(2)
            ],
        }

    @staticmethod
    def _checkpoint_contract() -> dict[str, Any]:
        return {
            "status": "PASS",
            "implementation": "device_aware_non_reentrant",
            "use_reentrant": False,
            "preserve_rng_state": True,
            "determinism_check": "default",
            "early_stop": True,
            "decoder_layer_count": 48,
            "boundary_layers": [24],
            "layer_input_devices": {
                str(layer): "cuda:0" if layer < 24 else "cuda:1" for layer in range(48)
            },
        }

    @staticmethod
    def _asset_identity() -> dict[str, str]:
        return {
            "path": str(certify_smoke.B200_ASSET_INTEGRITY_RECEIPT),
            "sha256": "a" * 64,
        }

    @classmethod
    def _inherited_hash_audit(cls, role: str) -> dict[str, Any]:
        return {
            "hash_verification": True,
            "hash_verification_mode": "inherited_large_asset_receipt",
            "direct_large_shard_hashes": False,
            "payload_rehash_performed": False,
            "production_eligible": True,
            "asset_integrity_evidence": {
                "status": "PASS",
                "role": role,
                "asset_integrity_receipt": cls._asset_identity(),
            },
        }

    @staticmethod
    def _runtime_budget(projected_hours: float) -> dict[str, Any]:
        bucket = {
            "warmup_repetitions": 1,
            "timed_repetitions": 3,
            "estimator": "median_p50",
            "benchmark_state_unchanged": True,
            "seconds_samples": [1.0, 2.0, 3.0],
            "seconds_p50": 2.0,
            "seconds": 2.0,
        }
        update_samples = [float(index) for index in range(1, 11)]
        projected_seconds = projected_hours * 3_600.0
        raw_seconds = projected_seconds / 1.25
        return {
            "schema_version": 5,
            "measurement": {
                "training_bucket_benchmarks": {
                    key: copy.deepcopy(bucket)
                    for key in ("1024", "2048", "4096", "9600")
                },
                "validation_bucket_benchmarks": {
                    key: copy.deepcopy(bucket)
                    for key in ("1024", "2048", "4096", "9600")
                },
                "production_update_benchmark": {
                    "warmup_repetitions": 2,
                    "timed_repetitions": 10,
                    "estimator": "median_p50",
                    "benchmark_state_unchanged": True,
                    "seconds_samples": update_samples,
                    "seconds_p50": 5.5,
                    "seconds_p90": 9.0,
                },
                "optimizer_boundary_seconds": 2.0,
            },
            "production_workload": {
                "optimizer_steps": 9_316,
                "rows_per_optimizer_step": 16,
                "training_occurrences": 149_048,
                "full_validation_steps": [4_658, 9_316],
                "full_validation_evaluations": 2,
                "optimizer_boundaries": 9_316,
            },
            "projection": {
                "safety_factor": 1.25,
                "raw_validation_seconds": 10.0,
                "raw_optimizer_boundary_seconds": 20.0,
            },
            "empirical_update_projection": {
                "estimator": "ten_update_sample_p50_and_p90",
                "raw_p90_total_seconds": raw_seconds,
                "projected_p90_total_seconds": projected_seconds,
                "projected_p90_total_hours": projected_hours,
            },
            "feasibility": {"status": "PASS"},
            "limitations": {
                "explicitly_unmeasured_overheads": [
                    "recovery and candidate checkpoint I/O"
                ]
            },
        }

    def test_exact_two_b200_hardware_and_24_24_device_map(self) -> None:
        evidence = self._gpu_evidence()
        self.assertIs(
            certify_smoke._validate_b200_gpu_evidence(evidence, "synthetic"),
            evidence,
        )
        device_map = dict(certify_smoke.b200_runtime.EXPECTED_HF_DEVICE_MAP)
        self.assertEqual(
            certify_smoke._validate_b200_hf_device_map(device_map, "synthetic map"),
            device_map,
        )
        changed = copy.deepcopy(evidence)
        changed["boundary_layers"] = [23]
        with self.assertRaisesRegex(RuntimeError, "boundary_layers"):
            certify_smoke._validate_b200_gpu_evidence(changed, "synthetic")
        changed_map = dict(device_map)
        changed_map["model.layers.24"] = "cuda:0"
        with self.assertRaisesRegex(RuntimeError, "synthetic map"):
            certify_smoke._validate_b200_hf_device_map(changed_map, "synthetic map")

    def test_device_aware_checkpointing_boundary_is_exact(self) -> None:
        contract = self._checkpoint_contract()
        self.assertIs(
            certify_smoke._validate_b200_gradient_checkpointing(
                contract, "synthetic checkpoint"
            ),
            contract,
        )
        changed = copy.deepcopy(contract)
        changed["layer_input_devices"]["24"] = "cuda:0"
        with self.assertRaisesRegex(RuntimeError, "layer_input_devices"):
            certify_smoke._validate_b200_gradient_checkpointing(
                changed, "synthetic checkpoint"
            )

    def test_selective_training_checkpoint_policy_is_exact_and_fail_closed(
        self,
    ) -> None:
        recipe = {
            "training_performance": (
                certify_smoke.b200_runtime.b200_training_performance_contract()
            )
        }
        contract = (
            certify_smoke.b200_runtime.b200_selective_gradient_checkpointing_contract()
        )
        self.assertIs(
            certify_smoke._validate_b200_training_activation_policy(
                contract,
                "synthetic selective checkpoint",
                recipe=recipe,
            ),
            contract,
        )
        changed = copy.deepcopy(contract)
        changed["uncheckpointed_decoder_layers"][0] = 2
        with self.assertRaisesRegex(RuntimeError, "synthetic selective checkpoint"):
            certify_smoke._validate_b200_training_activation_policy(
                changed,
                "synthetic selective checkpoint",
                recipe=recipe,
            )
        changed_recipe = copy.deepcopy(recipe)
        changed_recipe["training_performance"]["checkpointed_decoder_layer_count"] = 35
        with self.assertRaisesRegex(RuntimeError, "activation policy recipe"):
            certify_smoke._validate_b200_training_activation_policy(
                contract,
                "synthetic selective checkpoint",
                recipe=changed_recipe,
            )

    def test_finite_gradient_schema_matches_training_producer(self) -> None:
        audit = {
            "parameters_with_gradient": 528,
            "nonzero_gradient_elements": 1,
            "finite": True,
            "l2": 0.5,
        }
        self.assertIs(
            certify_smoke._validate_b200_finite_gradient(audit, "gradient"),
            audit,
        )
        changed = dict(audit, nonzero_gradient_elements=0)
        with self.assertRaises(RuntimeError):
            certify_smoke._validate_b200_finite_gradient(changed, "gradient")

    def test_training_and_inference_loader_audits_are_distinct(self) -> None:
        common_model = {
            "status": "PASS",
            "routed_nf4_layers": 48,
            "lora_target_bf16_linear_count": 528,
            "meta_parameter_count": 0,
            "meta_buffer_count": 0,
            "base_trainable_parameter_count": 0,
            "model_parallel_hook_count": 51,
            "device_map": dict(certify_smoke.b200_runtime.EXPECTED_HF_DEVICE_MAP),
            "placement_audit": {
                "cpu_tensor_count": 0,
                "meta_tensor_count": 0,
                "disk_offload_count": 0,
                "forbidden_cuda2_tensor_count": 0,
            },
        }
        training = {
            "status": "PASS",
            **self._inherited_hash_audit("training"),
            "hardware": self._gpu_evidence(),
            "model_audit": {
                **common_model,
                "gradient_checkpointing": True,
                "device_aware_checkpoint_layer_count": 48,
            },
            "gradient_checkpointing_contract": self._checkpoint_contract(),
        }
        inference = {
            "status": "PASS",
            **self._inherited_hash_audit("inference"),
            "hardware": self._gpu_evidence(),
            "model_audit": {
                **common_model,
                "routed_nf4_layout": "per_expert",
                "routed_experts_per_layer": 320,
                "non_cuda_parameter_count": 0,
                "non_cuda_buffer_count": 0,
                "gradient_checkpointing": False,
                "use_cache": True,
            },
        }
        self.assertIs(
            certify_smoke._validate_b200_loader_audit(
                training,
                "training",
                role="training",
                asset_identity=self._asset_identity(),
            ),
            training,
        )
        self.assertIs(
            certify_smoke._validate_b200_loader_audit(
                inference,
                "inference",
                role="inference",
                asset_identity=self._asset_identity(),
            ),
            inference,
        )
        changed = copy.deepcopy(inference)
        changed["model_audit"]["use_cache"] = False
        with self.assertRaisesRegex(RuntimeError, "use_cache"):
            certify_smoke._validate_b200_loader_audit(
                changed,
                "inference",
                role="inference",
                asset_identity=self._asset_identity(),
            )

    def test_b200_loader_requires_exact_inherited_asset_receipt_evidence(self) -> None:
        valid = self._inherited_hash_audit("inference")
        self.assertIs(
            certify_smoke._validate_b200_inherited_hash_audit(
                valid,
                label="synthetic inference loader",
                role="inference",
                asset_identity=self._asset_identity(),
            ),
            valid,
        )

        def missing_evidence(value: dict[str, Any]) -> None:
            value.pop("asset_integrity_evidence")

        def direct_hashes(value: dict[str, Any]) -> None:
            value["direct_large_shard_hashes"] = True

        def payload_rehash(value: dict[str, Any]) -> None:
            value["payload_rehash_performed"] = True

        def wrong_role(value: dict[str, Any]) -> None:
            value["asset_integrity_evidence"]["role"] = "training"

        def wrong_path(value: dict[str, Any]) -> None:
            value["asset_integrity_evidence"]["asset_integrity_receipt"][
                "path"
            ] = "/tampered/receipt.json"

        def wrong_sha(value: dict[str, Any]) -> None:
            value["asset_integrity_evidence"]["asset_integrity_receipt"]["sha256"] = (
                "b" * 64
            )

        cases = (
            (missing_evidence, "asset integrity evidence"),
            (direct_hashes, "direct large shard hashes"),
            (payload_rehash, "payload rehash performed"),
            (wrong_role, "asset integrity role"),
            (wrong_path, "asset integrity receipt path"),
            (wrong_sha, "asset integrity receipt sha256"),
        )
        for mutate, message in cases:
            with self.subTest(message=message):
                changed = copy.deepcopy(valid)
                mutate(changed)
                with self.assertRaisesRegex(RuntimeError, message):
                    certify_smoke._validate_b200_inherited_hash_audit(
                        changed,
                        label="synthetic inference loader",
                        role="inference",
                        asset_identity=self._asset_identity(),
                    )

    def test_runtime_projection_144_and_168_hour_gates(self) -> None:
        recommended = certify_smoke._validate_b200_runtime_budget(
            self._runtime_budget(144.0)
        )
        self.assertEqual(recommended["outcome"], "PASS_RECOMMENDED")
        risk = certify_smoke._validate_b200_runtime_budget(self._runtime_budget(168.0))
        self.assertEqual(risk["outcome"], "RISK_REQUIRES_SEPARATE_APPROVAL")
        self.assertEqual(risk["technical_gate_status"], "PASS")
        with self.assertRaisesRegex(RuntimeError, "exceeds 168 hours"):
            certify_smoke._validate_b200_runtime_budget(self._runtime_budget(168.001))
        override_gate = {
            "projection_required": True,
            "safety_factor": 1.25,
            "recommended_hours": 144.0,
            "former_absolute_limit_hours": 168.0,
            "hard_limit_enforced": False,
            "override_scope": "runtime_projection_only_exact_one_epoch",
            "user_instruction_sha256": "a" * 64,
            "non_runtime_gates_overridden": False,
        }
        overridden = certify_smoke._validate_b200_runtime_budget(
            self._runtime_budget(268.0), runtime_gate=override_gate
        )
        self.assertEqual(overridden["outcome"], "PASS_EXPLICIT_RUNTIME_LIMIT_OVERRIDE")
        self.assertFalse(overridden["hard_limit_enforced"])
        changed = copy.deepcopy(override_gate)
        changed["non_runtime_gates_overridden"] = True
        with self.assertRaisesRegex(RuntimeError, "non_runtime_gates_overridden"):
            certify_smoke._validate_b200_runtime_budget(
                self._runtime_budget(268.0), runtime_gate=changed
            )

    def test_b200_receipt_must_be_exact_fresh_and_not_symlinked(self) -> None:
        with tempfile.TemporaryDirectory(
            dir=TEST_TEMP_ROOT, prefix="b200-certifier-fresh-"
        ) as root:
            directory = Path(root)
            receipt = directory / "b200_smoke_2gpu_len9600_v1.json"
            with mock.patch.object(certify_smoke, "B200_SMOKE_RECEIPT", receipt):
                self.assertEqual(
                    certify_smoke._require_fresh_b200_smoke_output(receipt), receipt
                )
                with self.assertRaisesRegex(RuntimeError, "path mismatch"):
                    certify_smoke._require_fresh_b200_smoke_output(
                        directory / "wrong.json"
                    )
                receipt.touch()
                with self.assertRaises(FileExistsError):
                    certify_smoke._require_fresh_b200_smoke_output(receipt)

            target = directory / "target.json"
            target.touch()
            link = directory / "linked_receipt.json"
            link.symlink_to(target)
            with mock.patch.object(certify_smoke, "B200_SMOKE_RECEIPT", link):
                with self.assertRaisesRegex(RuntimeError, "symlink"):
                    certify_smoke._require_fresh_b200_smoke_output(link)

    def test_b200_cli_exposes_explicit_config_and_receipt_inputs(self) -> None:
        parser = certify_smoke.build_parser()
        actions = {
            option: action
            for action in parser._actions
            for option in action.option_strings
        }
        self.assertTrue(actions["--config-dir"].required)
        self.assertTrue(
            {
                "--kernel-probe-receipt",
                "--environment-receipt",
                "--asset-integrity-receipt",
                "--tokenizer-contract-receipt",
                "--preflight-receipt",
                "--training-manifest",
                "--base-predictions",
                "--tuned-predictions",
                "--output",
            }.issubset(actions)
        )


if __name__ == "__main__":
    unittest.main()
