#!/usr/bin/env python3
"""CPU tests for the fail-closed production adapter provenance gate."""

from __future__ import annotations

import copy
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from common import (
    CONFIG_DIR,
    PROJECT,
    canonical_sha256,
    expected_lora_module_names,
    pipeline_code_identity,
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
)
from quantization.contract import (
    EXPECTED_SOURCE_SHARD_COUNT,
    EXPECTED_SOURCE_TENSOR_COUNT,
    EXPECTED_SOURCE_TOTAL_SIZE,
)
from quantization.output_checkpoint import read_json_no_duplicates
from test_certify_smoke import CertificationFixture, write_json
from training_plan import early_stopping_decision
from verify_final_adapter import (
    B200_PRODUCTION_WANDB,
    EXPECTED_PRODUCTION_WANDB,
    EXPECTED_SPLIT_ROLES,
    PRODUCTION_SMOKE_CHECKS,
    _expected_sampler,
    _expected_training_exposure,
    current_data_evidence,
    current_file_hashes,
    validate_b200_manifest_contract,
    verify,
)


class FinalAdapterGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.recipe = read_json_no_duplicates(CONFIG_DIR / "recipe.json")
        cls.model_contract = read_json_no_duplicates(CONFIG_DIR / "model.json")
        # The source profile remains under test, while its original absolute
        # data root is intentionally unavailable on B200.  The owned B200 copy
        # is byte-identical and supplies the live evidence scan.
        cls.data_contract = read_json_no_duplicates(
            B200_CONFIG_DIR / "data_contract.json"
        )
        cls.data_evidence = current_data_evidence(cls.data_contract)

    @staticmethod
    def source_shards() -> list[dict]:
        payload_base, payload_remainder = divmod(
            EXPECTED_SOURCE_TOTAL_SIZE, EXPECTED_SOURCE_SHARD_COUNT
        )
        tensor_base, tensor_remainder = divmod(
            EXPECTED_SOURCE_TENSOR_COUNT, EXPECTED_SOURCE_SHARD_COUNT
        )
        return [
            {
                "filename": (
                    f"model-{index:05d}-of-"
                    f"{EXPECTED_SOURCE_SHARD_COUNT:05d}.safetensors"
                ),
                "file_bytes": (payload_base + int(index <= payload_remainder) + 1_024),
                "payload_bytes": (payload_base + int(index <= payload_remainder)),
                "tensor_count": (tensor_base + int(index <= tensor_remainder)),
                "sha256": f"{index:064x}",
            }
            for index in range(1, EXPECTED_SOURCE_SHARD_COUNT + 1)
        ]

    @staticmethod
    def certification_fixture(root: Path) -> CertificationFixture:
        # This suite tests the final production gate, not the smoke runtime
        # projection. The shared smoke fixture still encodes the superseded
        # 4,000-step cadence and cannot derive a v2 projection.
        with patch(
            "certify_smoke._runtime_budget_projection",
            return_value={"feasibility": {"status": "PASS"}},
        ):
            return CertificationFixture(root)

    def production_fixture(
        self,
        root: Path,
        *,
        early_stopped: bool = False,
    ) -> CertificationFixture:
        fixture = self.certification_fixture(root)
        for directory in (fixture.adapter_dir, fixture.final_adapter_dir):
            for name in ("LICENSE", "NOTICE"):
                (directory / name).write_bytes((PROJECT / name).read_bytes())
        fixture.adapter_files = current_file_hashes(fixture.final_adapter_dir)
        configured_full_steps = [
            int(step) for step in self.recipe["validation"]["full_validation_steps"]
        ]
        terminal_step = (
            int(self.recipe["validation"]["early_stopping"]["confirmation_step"])
            if early_stopped
            else int(self.recipe["optimizer_steps"])
        )
        full_steps = [step for step in configured_full_steps if step <= terminal_step]
        fixture.candidate_dirs = {
            step: fixture.adapter_dir.parent / "candidates" / f"step-{step:05d}"
            for step in full_steps
        }
        fixture.candidate_files = {}
        for step, directory in fixture.candidate_dirs.items():
            directory.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(fixture.final_adapter_dir, directory)
            fixture.candidate_files[step] = current_file_hashes(directory)
        fixture.fixture_selected_step = (
            full_steps[0] if early_stopped else full_steps[1]
        )
        fixture.best_adapter_dir = fixture.adapter_dir.parent / str(
            self.recipe["checkpoint"]["best_adapter_dirname"]
        )
        shutil.rmtree(fixture.best_adapter_dir)
        shutil.copytree(
            fixture.candidate_dirs[fixture.fixture_selected_step],
            fixture.best_adapter_dir,
        )
        fixture.best_adapter_files = current_file_hashes(fixture.best_adapter_dir)

        source_shards = self.source_shards()
        expert_manifest_path = fixture.training_expert_dir / "expert_nf4_manifest.json"
        expert_manifest = read_json_no_duplicates(expert_manifest_path)
        expert_manifest["source"]["shards"] = source_shards
        write_json(expert_manifest_path, expert_manifest)
        expert_manifest_sha256 = sha256(expert_manifest_path)

        manifest = copy.deepcopy(fixture.training)
        pipeline = pipeline_code_identity()
        fixture.production_smoke_path = root / "gpu_smoke.json"
        write_json(
            fixture.production_smoke_path,
            {
                "status": "PASS",
                "technical_production_gate_passed": True,
                "authorization_scope": "technical_smoke_gate_only",
                "human_production_approval_required": True,
                "human_production_approval_recorded": False,
                "source_dir": str(fixture.source_dir),
                "expert_nf4_dir": str(fixture.training_expert_dir),
                "model_contract_sha256": sha256(CONFIG_DIR / "model.json"),
                "recipe_sha256": sha256(CONFIG_DIR / "recipe.json"),
                "data_contract_sha256": sha256(CONFIG_DIR / "data_contract.json"),
                "pipeline_code": pipeline,
                "environment_lock_sha256": sha256(
                    PROJECT / "requirements/solar-open-2.lock.txt"
                ),
                "tokenizer_contract_receipt": {
                    "path": str(fixture.tokenizer_contract_path),
                    "sha256": sha256(fixture.tokenizer_contract_path),
                },
                "early_stopping_policy": copy.deepcopy(
                    self.recipe["validation"]["early_stopping"]
                ),
                "early_stopping_decision": copy.deepcopy(
                    fixture.training["early_stopping_decision"]
                ),
                "early_stopping_history": copy.deepcopy(fixture.training["history"]),
                "checks": {name: "PASS" for name in PRODUCTION_SMOKE_CHECKS},
                "runtime_budget_projection": {"feasibility": {"status": "PASS"}},
            },
        )
        state = manifest["state_contract"]
        state.update(
            {
                "model_contract_sha256": sha256(CONFIG_DIR / "model.json"),
                "recipe_sha256": sha256(CONFIG_DIR / "recipe.json"),
                "data_contract_sha256": sha256(CONFIG_DIR / "data_contract.json"),
                "train_sha256": self.data_contract["files"]["train"]["sha256"],
                "validation_sha256": self.data_contract["files"]["validation"][
                    "sha256"
                ],
                "pipeline_code_sha256": pipeline["sha256"],
                "environment_lock_sha256": sha256(
                    PROJECT / "requirements/solar-open-2.lock.txt"
                ),
                "expert_manifest_sha256": expert_manifest_sha256,
                "smoke": False,
                "production_smoke_receipt": {
                    "path": str(fixture.production_smoke_path),
                    "sha256": sha256(fixture.production_smoke_path),
                },
                "wandb": copy.deepcopy(EXPECTED_PRODUCTION_WANDB),
            }
        )
        lora = self.recipe["lora"]
        stats = {
            "parameter_count": int(lora["expected_module_count"]) * 2,
            "element_count": lora["expected_trainable_parameters"],
            "dtype_parameter_counts": {
                "torch.float32": int(lora["expected_module_count"]) * 2
            },
            "dtype_element_counts": {
                "torch.float32": lora["expected_trainable_parameters"]
            },
            "finite": True,
            "l1": 1.0,
            "l2": 1.0,
        }
        gradient = {
            "parameters_with_gradient": lora["expected_module_count"],
            "nonzero_gradient_elements": 1,
            "finite": True,
            "l2": 1.0,
        }
        delta = {
            "before_sha256": "1" * 64,
            "after_sha256": "2" * 64,
            "changed": True,
            "optimizer_step": 1,
            "learning_rate_used": (
                float(self.recipe["learning_rate"])
                / max(
                    1,
                    int(
                        float(self.recipe["warmup_ratio"])
                        * int(self.recipe["optimizer_steps"])
                    ),
                )
            ),
            "before": stats,
            "after": stats,
        }
        probe = {"sampled_parameter_count": 32, "values_sha256": "3" * 64}
        supervision = {
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
            "answer_label_prefix_removed": False,
        }
        history = []
        validation_recipe = self.recipe["validation"]
        selection_weights = validation_recipe["selection_weights"]
        configured_full_steps = [
            int(step) for step in validation_recipe["full_validation_steps"]
        ]
        full_steps = [step for step in configured_full_steps if step <= terminal_step]
        candidate_losses = (
            {
                configured_full_steps[0]: 0.75,
                configured_full_steps[1]: 1.0,
            }
            if early_stopped
            else {
                configured_full_steps[0]: 1.25,
                configured_full_steps[1]: 0.75,
                configured_full_steps[2]: 1.0,
            }
        )
        scheduled_steps = sorted(
            set(
                range(
                    int(validation_recipe["eval_every_steps"]),
                    terminal_step + 1,
                    int(validation_recipe["eval_every_steps"]),
                )
            )
            | set(full_steps)
        )
        for step in scheduled_steps:
            scope = "full" if step in full_steps else "fixed_subset"
            translation_loss = candidate_losses.get(step, 2.0 + step / 100_000)
            summarization_loss = candidate_losses.get(step, 3.0 + step / 100_000)
            counts = (
                self.data_evidence["validation"]["by_task"]
                if scope == "full"
                else {
                    task: self.recipe["validation"]["subset_per_task"]
                    for task in ("translation", "summarization")
                }
            )
            history.append(
                {
                    "step": step,
                    "scope": scope,
                    "selection_loss": (
                        selection_weights["translation"] * translation_loss
                        + selection_weights["summarization"] * summarization_loss
                    ),
                    "translation_loss": translation_loss,
                    "summarization_loss": summarization_loss,
                    "translation_count": counts["translation"],
                    "summarization_count": counts["summarization"],
                    "dropped_over_max": {},
                    "validation_order_sha256": "4" * 64,
                    "elapsed_seconds": 1.0,
                }
            )
        full_history = {
            item["step"]: item for item in history if item["scope"] == "full"
        }
        candidates = [
            {
                "step": step,
                "selection_loss": full_history[step]["selection_loss"],
                "translation_loss": full_history[step]["translation_loss"],
                "summarization_loss": full_history[step]["summarization_loss"],
                "adapter_dir": str(fixture.candidate_dirs[step]),
                "files_sha256": fixture.candidate_files[step],
            }
            for step in full_steps
        ]
        selected = min(
            candidates,
            key=lambda item: (item["selection_loss"], item["step"]),
        )
        deck = self.recipe["training_deck"]
        total_occurrences = deck["total_occurrences"]
        occurrences_per_step = int(
            self.recipe["microbatch_policy"]["optimizer_step_occurrences"]
        )
        occurrences_per_epoch = int(deck["occurrences_per_epoch"]["total"])
        actual_occurrences = terminal_step * occurrences_per_step
        completed_epochs = actual_occurrences / occurrences_per_epoch
        status = "EARLY_STOPPED" if early_stopped else "COMPLETED"
        early_stopping = early_stopping_decision(
            history,
            completed_step=terminal_step,
            policy=validation_recipe["early_stopping"],
            optimizer_steps=int(self.recipe["optimizer_steps"]),
            full_validation_steps=configured_full_steps,
            occurrences_per_step=occurrences_per_step,
            occurrences_per_epoch=occurrences_per_epoch,
        )
        task_occurrences, quality_occurrences = _expected_training_exposure(
            recipe=self.recipe,
            data_evidence=self.data_evidence,
            consumed_occurrences=actual_occurrences,
        )
        expected_names = expected_lora_module_names(
            self.model_contract["architecture_contract"]
        )
        manifest.update(
            {
                "status": status,
                "slurm_job_id": "12345",
                "model_name": self.model_contract["derivative_model_name"],
                "attribution": self.model_contract["attribution"],
                "output_dir": str(fixture.adapter_dir.parent),
                "config": self.recipe,
                "output_only_contract": self.recipe["output_only_contract"],
                "wandb": copy.deepcopy(EXPECTED_PRODUCTION_WANDB),
                "completed_updates": terminal_step,
                "target_updates": self.recipe["optimizer_steps"],
                "backward_passes": actual_occurrences,
                "consumed_occurrences": actual_occurrences,
                "training_completion": {
                    "status": status,
                    "epochs": self.recipe["epochs"],
                    "completed_epochs": completed_epochs,
                    "optimizer_steps": terminal_step,
                    "consumed_occurrences": actual_occurrences,
                    "expected_optimizer_steps": self.recipe["optimizer_steps"],
                    "expected_occurrences": total_occurrences["total"],
                    "early_stopping_triggered": early_stopping["triggered"],
                },
                "early_stopping_decision": early_stopping,
                "training_coverage": {
                    "total_occurrences": actual_occurrences,
                    "skipped_occurrences": 0,
                    "unconsumed_planned_occurrences": (
                        total_occurrences["total"] - actual_occurrences
                    ),
                    "unique_raw_rows": deck["base_unique_rows"],
                    "task_occurrences": task_occurrences,
                    "quality_occurrences": quality_occurrences,
                },
                "token_length_contract": {
                    split: {
                        "rows": self.data_evidence[split]["rows"],
                        "by_task": self.data_evidence[split]["by_task"],
                        "max_length": self.recipe["max_length"],
                        "over_max_count": 0,
                        "over_max_rows": [],
                        "minimum": 1,
                        "p50": 500,
                        "p90": 2_000,
                        "p95": 3_000,
                        "p99": 5_000,
                        "p99_9": 8_000,
                        "maximum": self.recipe["max_length"],
                    }
                    for split in ("train", "validation")
                },
                "max_observed_tokens": self.recipe["max_length"],
                "total_hard_term_weighted_tokens": 100,
                "first_update_gradient_audit": gradient,
                "first_update_adapter_delta": delta,
                "first_output_only_supervision": supervision,
                "frozen_probe_before": probe,
                "frozen_probe_after_update": probe,
                "microbatch_phase_seconds": None,
                "optimizer_boundary_seconds": None,
                "runtime_budget_projection": None,
                "post_update_recompute_audit": None,
                "elapsed_seconds": 10.0,
                "history": history,
                "split_roles": copy.deepcopy(EXPECTED_SPLIT_ROLES),
                "sampler": _expected_sampler(self.data_evidence, self.recipe),
                "candidate_checkpoints": {
                    str(step): {
                        "step": step,
                        "adapter_dir": str(fixture.candidate_dirs[step]),
                        "files_sha256": fixture.candidate_files[step],
                    }
                    for step in full_steps
                },
                "selection_record": {
                    "policy": validation_recipe["selection_policy"],
                    "tie_break": validation_recipe["selection_tie_break"],
                    "candidate_steps": full_steps,
                    "candidates": candidates,
                    "selected_step": selected["step"],
                    "selected_selection_loss": selected["selection_loss"],
                },
                "selected_checkpoint": {
                    "policy": validation_recipe["selection_policy"],
                    "tie_break": validation_recipe["selection_tie_break"],
                    "step": selected["step"],
                    "selection_loss": selected["selection_loss"],
                    "source_checkpoint_dir": selected["adapter_dir"],
                    "adapter_dir": str(fixture.best_adapter_dir),
                    "files_sha256": fixture.best_adapter_files,
                },
                "final_checkpoint": {
                    "step": terminal_step,
                    "adapter_dir": str(fixture.final_adapter_dir),
                    "files_sha256": fixture.adapter_files,
                },
                "lora_module_audit": {
                    "status": "PASS",
                    "module_count": lora["expected_module_count"],
                    "module_names_sha256": canonical_sha256(sorted(expected_names)),
                    "module_class": "torch.nn.Linear",
                },
                "trainable_parameters": lora["expected_trainable_parameters"],
                "device_audit": {
                    "hf_device_map_devices": ["0", "1", "2"],
                    "parameter_devices": {
                        "cuda:0": 1,
                        "cuda:1": 1,
                        "cuda:2": 1,
                    },
                    "cpu_disk_offload": False,
                },
            }
        )
        manifest.pop("smoke", None)
        manifest["loader_audit"].update(
            {
                "status": "PASS",
                "hardware": {"device_count": 3},
                "source_inventory": {
                    "source_dir": str(fixture.source_dir),
                    "index_sha256": fixture.source["index_sha256"],
                    "config_sha256": fixture.source["config_sha256"],
                    "tensor_count": EXPECTED_SOURCE_TENSOR_COUNT,
                    "shard_count": EXPECTED_SOURCE_SHARD_COUNT,
                    "shards": source_shards,
                },
                "expert_preflight": {
                    "status": "PASS",
                    "production_eligible": True,
                    "checkpoint_dir": str(fixture.training_expert_dir),
                    "manifest_sha256": expert_manifest_sha256,
                    "expert_index_sha256": sha256(
                        fixture.training_expert_dir
                        / "expert_nf4.safetensors.index.json"
                    ),
                },
                "model_audit": {
                    "status": "PASS",
                    "lora_target_bf16_linear_count": 528,
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
            }
        )
        write_json(fixture.training_path, manifest)
        (fixture.training_path.parent / "_SUCCESS").write_text(
            "PASS\n", encoding="utf-8"
        )
        fixture.training = manifest
        return fixture

    def verified(self, fixture: CertificationFixture) -> dict:
        with (
            patch(
                "verify_final_adapter.current_data_evidence",
                return_value=self.data_evidence,
            ),
            patch(
                "verify_final_adapter.EXPECTED_SOURCE_INDEX_SHA256",
                fixture.source["index_sha256"],
            ),
        ):
            return verify(
                fixture.training_path,
                getattr(fixture, "best_adapter_dir", fixture.adapter_dir),
            )

    def refresh_early_stopping_evidence(self, manifest: dict) -> None:
        decision = early_stopping_decision(
            manifest["history"],
            completed_step=int(manifest["completed_updates"]),
            policy=self.recipe["validation"]["early_stopping"],
            optimizer_steps=int(self.recipe["optimizer_steps"]),
            full_validation_steps=self.recipe["validation"]["full_validation_steps"],
            occurrences_per_step=int(
                self.recipe["microbatch_policy"]["optimizer_step_occurrences"]
            ),
            occurrences_per_epoch=int(
                self.recipe["training_deck"]["occurrences_per_epoch"]["total"]
            ),
        )
        manifest["early_stopping_decision"] = decision
        manifest["training_completion"]["early_stopping_triggered"] = decision[
            "triggered"
        ]

    def test_current_training_data_matches_contract(self) -> None:
        self.assertEqual(
            self.data_evidence["train"]["sha256"],
            self.data_contract["files"]["train"]["sha256"],
        )
        self.assertEqual(
            self.data_evidence["validation"]["by_task"],
            self.data_contract["files"]["validation"]["by_task"],
        )

    def test_exact_production_best_adapter_passes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="solar-final-adapter-") as temporary:
            fixture = self.production_fixture(Path(temporary))
            result = self.verified(fixture)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["selected_step"], fixture.fixture_selected_step)
            self.assertEqual(
                result["training_coverage"]["total_occurrences"],
                self.recipe["training_deck"]["total_occurrences"]["total"],
            )
            self.assertEqual(
                [
                    item["step"]
                    for item in result["validation_schedule"]
                    if item["scope"] == "full"
                ],
                self.recipe["validation"]["full_validation_steps"],
            )
            self.assertNotEqual(result["best_adapter_dir"], result["final_adapter_dir"])

    def test_exact_approved_early_stopped_adapter_passes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="solar-final-adapter-") as temporary:
            fixture = self.production_fixture(
                Path(temporary),
                early_stopped=True,
            )
            result = self.verified(fixture)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["training_status"], "EARLY_STOPPED")
            self.assertEqual(result["completed_updates"], 13973)
            self.assertEqual(result["target_updates"], 18631)
            self.assertEqual(result["backward_passes"], 223568)
            self.assertEqual(
                result["early_stopping_decision"]["status"],
                "TRIGGERED",
            )
            self.assertEqual(
                [
                    item["step"]
                    for item in result["validation_schedule"]
                    if item["scope"] == "full"
                ],
                [9316, 13973],
            )
            self.assertEqual(
                sorted(result["candidate_artifacts"]),
                [9316, 13973],
            )
            self.assertEqual(result["selected_step"], 9316)

    def test_early_stop_evidence_and_terminal_step_are_fail_closed(self) -> None:
        for label, mutate in (
            (
                "trigger",
                lambda manifest: manifest["early_stopping_decision"].update(
                    {"triggered": False}
                ),
            ),
            (
                "terminal",
                lambda manifest: manifest.update({"completed_updates": 13972}),
            ),
            (
                "sealed monitor",
                lambda manifest: manifest["history"][0].update({"scope": "human_gold"}),
            ),
        ):
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory(
                    prefix="solar-final-adapter-"
                ) as temporary:
                    fixture = self.production_fixture(
                        Path(temporary),
                        early_stopped=True,
                    )
                    manifest = copy.deepcopy(fixture.training)
                    mutate(manifest)
                    write_json(fixture.training_path, manifest)
                    with self.assertRaises(RuntimeError):
                        self.verified(fixture)

    def test_relabeled_smoke_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="solar-final-adapter-") as temporary:
            fixture = self.certification_fixture(Path(temporary))
            for name in ("LICENSE", "NOTICE"):
                (fixture.adapter_dir / name).write_bytes((PROJECT / name).read_bytes())
            files = current_file_hashes(fixture.adapter_dir)
            manifest = fixture.training
            manifest.update(
                {
                    "status": "COMPLETED",
                    "slurm_job_id": "12345",
                    "model_name": self.model_contract["derivative_model_name"],
                    "attribution": self.model_contract["attribution"],
                    "output_dir": str(fixture.adapter_dir.parent),
                    "config": self.recipe,
                    "wandb": copy.deepcopy(EXPECTED_PRODUCTION_WANDB),
                    "completed_updates": self.recipe["optimizer_steps"],
                    "target_updates": self.recipe["optimizer_steps"],
                    "backward_passes": self.recipe["training_deck"][
                        "total_occurrences"
                    ]["total"],
                    "consumed_occurrences": self.recipe["training_deck"][
                        "total_occurrences"
                    ]["total"],
                    "elapsed_seconds": 10.0,
                    "split_roles": copy.deepcopy(EXPECTED_SPLIT_ROLES),
                }
            )
            manifest["selected_checkpoint"].update(
                {
                    "step": self.recipe["optimizer_steps"],
                    "files_sha256": files,
                }
            )
            write_json(fixture.training_path, manifest)
            (fixture.training_path.parent / "_SUCCESS").write_text(
                "PASS\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "smoke"):
                self.verified(fixture)

    def test_changed_adapter_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="solar-final-adapter-") as temporary:
            fixture = self.production_fixture(Path(temporary))
            with (fixture.adapter_dir / "_SUCCESS").open("a", encoding="utf-8") as out:
                out.write("changed\n")
            with self.assertRaisesRegex(RuntimeError, "success marker"):
                self.verified(fixture)

    def test_production_evidence_tampering_is_rejected(self) -> None:
        cases = {
            "state smoke": lambda value: value["state_contract"].update(
                {"smoke": True}
            ),
            "pipeline identity": lambda value: value["state_contract"].update(
                {"pipeline_code_sha256": "0" * 64}
            ),
            "recipe": lambda value: value["config"].update({"learning_rate": 2e-4}),
            "output-only contract": lambda value: value["output_only_contract"].update(
                {"labels_notes_explanations_markdown_reasoning_framing_allowed": (True)}
            ),
            "backward passes": lambda value: value.update({"backward_passes": 0}),
            "completion occurrences": lambda value: value["training_completion"].update(
                {"consumed_occurrences": 1}
            ),
            "skipped training occurrence": lambda value: value[
                "training_coverage"
            ].update({"skipped_occurrences": 1}),
            "unique raw coverage": lambda value: value["training_coverage"].update(
                {"unique_raw_rows": 1}
            ),
            "task exposure": lambda value: value["training_coverage"][
                "task_occurrences"
            ].update({"translation": 1}),
            "token length skip": lambda value: value["token_length_contract"][
                "train"
            ].update({"over_max_count": 1}),
            "validation scope": lambda value: value["history"][0].update(
                {"scope": "full"}
            ),
            "validation count": lambda value: value["history"][7].update(
                {"translation_count": 1}
            ),
            "sampler": lambda value: value["sampler"].update({"total_examples": 1}),
            "selection loss": lambda value: value["selection_record"].update(
                {"selected_selection_loss": 99.0}
            ),
            "selected step": lambda value: value["selected_checkpoint"].update(
                {"step": value["selection_record"]["candidate_steps"][0]}
            ),
            "final checkpoint hash": lambda value: value["final_checkpoint"][
                "files_sha256"
            ].update({"_SUCCESS": "0" * 64}),
            "gradient": lambda value: value["first_update_gradient_audit"].update(
                {"finite": False}
            ),
            "output-only evidence": lambda value: value[
                "first_output_only_supervision"
            ].update({"prompt_labels_all_masked": False}),
            "multiple termination tokens": lambda value: value[
                "first_output_only_supervision"
            ].update({"assistant_termination_token_count": 2}),
            "boolean termination token count": lambda value: value[
                "first_output_only_supervision"
            ].update({"assistant_termination_token_count": True}),
            "negative termination token id": lambda value: value[
                "first_output_only_supervision"
            ].update({"assistant_termination_token_id": -1}),
            "boolean termination token id": lambda value: value[
                "first_output_only_supervision"
            ].update({"assistant_termination_token_id": True}),
            "empty template suffix": lambda value: value[
                "first_output_only_supervision"
            ].update({"template_suffix_token_count": 0}),
            "boolean template suffix count": lambda value: value[
                "first_output_only_supervision"
            ].update({"template_suffix_token_count": True}),
            "template suffix without EOS": lambda value: value[
                "first_output_only_supervision"
            ].update({"template_suffix_starts_with_eos": False}),
            "supervised sequence without final EOS": lambda value: value[
                "first_output_only_supervision"
            ].update({"supervised_sequence_ends_at_eos": False}),
            "expert identity": lambda value: value["state_contract"].update(
                {"expert_manifest_sha256": "0" * 64}
            ),
            "smoke receipt identity": lambda value: value["state_contract"][
                "production_smoke_receipt"
            ].update({"sha256": "0" * 64}),
            "top-level W&B": lambda value: value["wandb"].update({"run_id": "wrong"}),
            "state W&B": lambda value: value["state_contract"]["wandb"].update(
                {"run_name": "wrong"}
            ),
            "source payload digest": lambda value: value["loader_audit"][
                "source_inventory"
            ]["shards"][0].update({"sha256": None}),
            "source payload coverage": lambda value: value["loader_audit"][
                "source_inventory"
            ]["shards"].pop(),
            "source payload cross-seal": lambda value: value["loader_audit"][
                "source_inventory"
            ]["shards"][0].update({"sha256": "f" * 64}),
        }
        for label, mutate in cases.items():
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory(
                    prefix="solar-final-adapter-"
                ) as temporary:
                    fixture = self.production_fixture(Path(temporary))
                    manifest = copy.deepcopy(fixture.training)
                    mutate(manifest)
                    write_json(fixture.training_path, manifest)
                    with self.assertRaises(RuntimeError):
                        self.verified(fixture)

    def test_full_validation_losses_must_be_finite(self) -> None:
        with tempfile.TemporaryDirectory(prefix="solar-final-adapter-") as temporary:
            fixture = self.production_fixture(Path(temporary))
            manifest = copy.deepcopy(fixture.training)
            step = self.recipe["validation"]["full_validation_steps"][0]
            history = next(item for item in manifest["history"] if item["step"] == step)
            candidate = next(
                item
                for item in manifest["selection_record"]["candidates"]
                if item["step"] == step
            )
            history["translation_loss"] = float("nan")
            candidate["translation_loss"] = float("nan")
            write_json(fixture.training_path, manifest)
            with self.assertRaisesRegex(RuntimeError, "translation loss"):
                self.verified(fixture)

    def test_selection_tie_uses_earlier_full_validation_step(self) -> None:
        with tempfile.TemporaryDirectory(prefix="solar-final-adapter-") as temporary:
            fixture = self.production_fixture(Path(temporary))
            manifest = copy.deepcopy(fixture.training)
            first, second = self.recipe["validation"]["full_validation_steps"][:2]
            tied_loss = 0.5
            for item in manifest["history"]:
                if item["step"] in {first, second}:
                    item["translation_loss"] = tied_loss
                    item["summarization_loss"] = tied_loss
                    item["selection_loss"] = tied_loss
                elif (
                    item["scope"] == "fixed_subset"
                    and item["step"]
                    > self.recipe["validation"]["early_stopping"]["minimum_step"]
                ):
                    improving_loss = 2.0 - item["step"] / 10_000
                    item["translation_loss"] = improving_loss
                    item["summarization_loss"] = improving_loss
                    item["selection_loss"] = improving_loss
            candidates = manifest["selection_record"]["candidates"]
            for item in candidates:
                if item["step"] in {first, second}:
                    item["translation_loss"] = tied_loss
                    item["summarization_loss"] = tied_loss
                    item["selection_loss"] = tied_loss
            first_candidate = next(item for item in candidates if item["step"] == first)
            manifest["selection_record"].update(
                {
                    "selected_step": first,
                    "selected_selection_loss": tied_loss,
                }
            )
            manifest["selected_checkpoint"].update(
                {
                    "step": first,
                    "selection_loss": tied_loss,
                    "source_checkpoint_dir": first_candidate["adapter_dir"],
                }
            )
            self.refresh_early_stopping_evidence(manifest)
            write_json(fixture.training_path, manifest)
            self.assertEqual(self.verified(fixture)["selected_step"], first)

            manifest["selection_record"]["selected_step"] = second
            manifest["selected_checkpoint"].update(
                {
                    "step": second,
                    "source_checkpoint_dir": str(fixture.candidate_dirs[second]),
                }
            )
            write_json(fixture.training_path, manifest)
            with self.assertRaisesRegex(RuntimeError, "selected step"):
                self.verified(fixture)

    def test_changed_candidate_adapter_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="solar-final-adapter-") as temporary:
            fixture = self.production_fixture(Path(temporary))
            candidate = fixture.candidate_dirs[fixture.fixture_selected_step]
            with (candidate / "_SUCCESS").open("a", encoding="utf-8") as out:
                out.write("changed\n")
            with self.assertRaises(RuntimeError):
                self.verified(fixture)

    def test_final_adapter_must_be_separate_from_best_adapter(self) -> None:
        with tempfile.TemporaryDirectory(prefix="solar-final-adapter-") as temporary:
            fixture = self.production_fixture(Path(temporary))
            manifest = copy.deepcopy(fixture.training)
            manifest["final_checkpoint"]["adapter_dir"] = str(fixture.best_adapter_dir)
            write_json(fixture.training_path, manifest)
            with self.assertRaisesRegex(RuntimeError, "final checkpoint adapter"):
                self.verified(fixture)

    def test_evaluation_cannot_use_final_adapter(self) -> None:
        with tempfile.TemporaryDirectory(prefix="solar-final-adapter-") as temporary:
            fixture = self.production_fixture(Path(temporary))
            with (
                patch(
                    "verify_final_adapter.current_data_evidence",
                    return_value=self.data_evidence,
                ),
                patch(
                    "verify_final_adapter.EXPECTED_SOURCE_INDEX_SHA256",
                    fixture.source["index_sha256"],
                ),
                self.assertRaisesRegex(RuntimeError, "evaluation adapter path"),
            ):
                verify(fixture.training_path, fixture.final_adapter_dir)

    def test_production_smoke_receipt_tampering_is_rejected(self) -> None:
        cases = {
            "status": lambda value: value.update({"status": "FAIL"}),
            "technical gate": lambda value: value.update(
                {"technical_production_gate_passed": False}
            ),
            "authorization scope": lambda value: value.update(
                {"authorization_scope": "production_launch"}
            ),
            "human approval semantics": lambda value: value.update(
                {"human_production_approval_recorded": True}
            ),
            "output-only check": lambda value: value["checks"].update(
                {"output_only_supervision": "FAIL"}
            ),
            "tokenizer contract check": lambda value: value["checks"].update(
                {"tokenizer_output_contract": "FAIL"}
            ),
            "tokenizer receipt identity": lambda value: value[
                "tokenizer_contract_receipt"
            ].update({"sha256": "0" * 64}),
            "runtime check": lambda value: value["checks"].update(
                {"runtime_budget_feasibility": "FAIL"}
            ),
            "early-stopping check": lambda value: value["checks"].update(
                {"early_stopping_policy": "FAIL"}
            ),
            "early-stopping policy": lambda value: value[
                "early_stopping_policy"
            ].update({"minimum_step": 1}),
            "early-stopping decision": lambda value: value[
                "early_stopping_decision"
            ].update({"triggered": True}),
            "early-stopping history": lambda value: value["early_stopping_history"][
                0
            ].update({"selection_loss": 99.0}),
            "runtime feasibility": lambda value: value["runtime_budget_projection"][
                "feasibility"
            ].update({"status": "FAIL"}),
        }
        for label, mutate in cases.items():
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory(
                    prefix="solar-final-adapter-"
                ) as temporary:
                    fixture = self.production_fixture(Path(temporary))
                    receipt = read_json_no_duplicates(fixture.production_smoke_path)
                    mutate(receipt)
                    write_json(fixture.production_smoke_path, receipt)
                    manifest = copy.deepcopy(fixture.training)
                    manifest["state_contract"]["production_smoke_receipt"]["sha256"] = (
                        sha256(fixture.production_smoke_path)
                    )
                    write_json(fixture.training_path, manifest)
                    with self.assertRaises(RuntimeError):
                        self.verified(fixture)

    def test_changed_nested_tokenizer_receipt_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="solar-final-adapter-") as temporary:
            fixture = self.production_fixture(Path(temporary))
            tokenizer_receipt = read_json_no_duplicates(fixture.tokenizer_contract_path)
            tokenizer_receipt["status"] = "FAIL"
            write_json(fixture.tokenizer_contract_path, tokenizer_receipt)
            with self.assertRaisesRegex(RuntimeError, "tokenizer receipt SHA"):
                self.verified(fixture)


class B200FinalAdapterContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_b200_config_bundle(B200_CONFIG_DIR)
        cls.identity = b200_config_identity(B200_CONFIG_DIR)

    def manifest(self, root: Path) -> dict:
        smoke = root / "smoke.json"
        approval = root / "approval.json"
        output_dir = root / "run"
        pipeline = pipeline_code_identity(config_dir=B200_CONFIG_DIR)
        write_json(
            smoke,
            {
                "schema": "solar_open2_b200_smoke_certification",
                "schema_version": 2,
                "status": "PASS",
                "technical_production_gate_passed": True,
                "authorization_scope": "technical_smoke_gate_only",
                "human_production_approval_required": True,
                "human_production_approval_recorded": False,
                "config": self.identity,
                "training_performance": b200_training_performance_contract(),
                "gradient_checkpointing_contract": (
                    b200_selective_gradient_checkpointing_contract()
                ),
            },
        )
        smoke_identity = {"path": str(smoke), "sha256": sha256(smoke)}
        write_json(
            approval,
            {
                "schema": "solar-open2-b200-production-approval",
                "schema_version": 2,
                "status": "PASS",
                "authorization_scope": "b200_production_training",
                "human_production_approval_recorded": True,
                "technical_smoke_receipt": smoke_identity,
                "config_identity": self.identity,
                "pipeline_code_sha256": pipeline["sha256"],
                "model_contract_sha256": self.identity["files"]["model"]["sha256"],
                "recipe_sha256": self.identity["files"]["recipe"]["sha256"],
                "data_contract_sha256": self.identity["files"]["data_contract"][
                    "sha256"
                ],
                "training_performance": b200_training_performance_contract(),
                "training_performance_sha256": canonical_sha256(
                    b200_training_performance_contract()
                ),
                "gradient_checkpointing_contract_sha256": canonical_sha256(
                    b200_selective_gradient_checkpointing_contract()
                ),
                "output_dir": str(output_dir),
                "wandb": B200_PRODUCTION_WANDB,
                "runtime_override": b200_runtime_override_evidence(),
            },
        )
        return {
            "status": "COMPLETED",
            "profile": "b200",
            "config_identity": self.identity,
            "model_name": self.bundle["model"]["derivative_model_name"],
            "attribution": self.bundle["model"]["attribution"],
            "config": self.bundle["recipe"],
            "output_only_contract": self.bundle["recipe"]["output_only_contract"],
            "completed_updates": 9_316,
            "target_updates": 9_316,
            "consumed_occurrences": 149_048,
            "backward_passes": 149_048,
            "split_roles": EXPECTED_SPLIT_ROLES,
            "wandb": B200_PRODUCTION_WANDB,
            "training_performance": b200_training_performance_contract(),
            "training_performance_sha256": canonical_sha256(
                b200_training_performance_contract()
            ),
            "gradient_checkpointing_contract": (
                b200_selective_gradient_checkpointing_contract()
            ),
            "gradient_checkpointing_contract_sha256": canonical_sha256(
                b200_selective_gradient_checkpointing_contract()
            ),
            "output_dir": str(output_dir),
            "occurrence_contract": {
                "nominal_step_occurrences": 16,
                "total_occurrences": 149_048,
                "optimizer_steps": 9_316,
                "final_step_occurrences": 8,
            },
            "state_contract": {
                "format_version": 6,
                "profile": "b200",
                "config_identity": self.identity,
                "source_dir": str(BF16_SOURCE_DIR),
                "expert_nf4_dir": str(TRAIN_EXPERT_DIR),
                "model_contract_sha256": self.identity["files"]["model"]["sha256"],
                "recipe_sha256": self.identity["files"]["recipe"]["sha256"],
                "data_contract_sha256": self.identity["files"]["data_contract"][
                    "sha256"
                ],
                "train_sha256": self.bundle["data_contract"]["files"]["train"][
                    "sha256"
                ],
                "validation_sha256": self.bundle["data_contract"]["files"][
                    "validation"
                ]["sha256"],
                "pipeline_code_sha256": pipeline["sha256"],
                "environment_lock_sha256": sha256(
                    PROJECT / "requirements/solar-open-2.lock.txt"
                ),
                "smoke": False,
                "wandb": B200_PRODUCTION_WANDB,
                "training_performance_sha256": canonical_sha256(
                    b200_training_performance_contract()
                ),
                "training_activation_contract_sha256": canonical_sha256(
                    b200_selective_gradient_checkpointing_contract()
                ),
                "production_smoke_receipt": {
                    **smoke_identity,
                },
                "production_approval_receipt": {
                    "path": str(approval),
                    "sha256": sha256(approval),
                },
                "gpu_runtime": {
                    "logical_gpu_count": 2,
                    "boundary_layers": [24],
                    "logical_to_physical": [
                        {"logical_device": "cuda:0", "physical_index": 6},
                        {"logical_device": "cuda:1", "physical_index": 7},
                    ],
                },
            },
            "device_audit": {
                "hf_device_map_devices": ["cuda:0", "cuda:1"],
                "parameter_devices": {"cuda:0": 1, "cuda:1": 1},
                "cpu_disk_offload": False,
            },
            "loader_audit": {
                "status": "PASS",
                "b200_config": self.identity,
                "gradient_checkpointing_contract": (
                    b200_full_gradient_checkpointing_contract()
                ),
                "hardware": {
                    "logical_gpu_count": 2,
                    "boundary_layers": [24],
                    "physical_cuda_visible_devices": "6,7",
                },
                "bitsandbytes_runtime": {
                    "status": "PASS",
                    "compute_capability": [10, 0],
                    "manifest_status": "SEALED",
                    "bitsandbytes_version": "0.49.2",
                    "native_library": "libbitsandbytes_cuda128.so",
                },
                "source_inventory": {
                    "source_dir": str(BF16_SOURCE_DIR),
                },
                "expert_preflight": {
                    "checkpoint_dir": str(TRAIN_EXPERT_DIR),
                    "status": "PASS",
                    "production_eligible": True,
                },
                "model_audit": {
                    "device_map": dict(EXPECTED_HF_DEVICE_MAP),
                    "placement_audit": {
                        "cpu_tensor_count": 0,
                        "meta_tensor_count": 0,
                        "disk_offload_count": 0,
                        "forbidden_cuda2_tensor_count": 0,
                    },
                },
            },
            "early_stopping_decision": {
                "profile": "b200",
                "status": "MAXIMUM_REACHED",
                "evaluated_step": 9_316,
                "triggered": False,
                "reason": "exact_epoch_coverage_reached",
                "quality_early_stopping_enabled": False,
                "completed_candidate_steps": [4_658, 9_316],
                "completed_occurrences": 149_048,
                "completed_epochs": 1.0,
            },
            "training_coverage": {
                "total_occurrences": 149_048,
                "skipped_occurrences": 0,
                "unconsumed_planned_occurrences": 0,
                "unique_raw_rows": 142_713,
                "task_occurrences": {
                    "translation": 102_224,
                    "summarization": 46_824,
                },
            },
        }

    def test_exact_b200_terminal_manifest_passes_cpu_gate(self) -> None:
        test_root = PROJECT.parent / "runtime/tmp/tests/b200-final-adapter"
        test_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_root) as temporary:
            manifest = self.manifest(Path(temporary))
            validate_b200_manifest_contract(
                manifest,
                config_identity=self.identity,
                recipe=self.bundle["recipe"],
                model_contract=self.bundle["model"],
            )

    def test_b200_tail_and_cuda2_mutations_fail_closed(self) -> None:
        test_root = PROJECT.parent / "runtime/tmp/tests/b200-final-adapter"
        test_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_root) as temporary:
            original = self.manifest(Path(temporary))
            mutations = (
                lambda value: value.update({"completed_updates": 9_315}),
                lambda value: value["device_audit"].update(
                    {"parameter_devices": {"cuda:0": 1, "cuda:2": 1}}
                ),
                lambda value: value["training_coverage"].update(
                    {"unconsumed_planned_occurrences": 8}
                ),
                lambda value: value["state_contract"][
                    "production_approval_receipt"
                ].update({"sha256": "0" * 64}),
            )
            for mutate in mutations:
                changed = copy.deepcopy(original)
                mutate(changed)
                with self.assertRaises(RuntimeError):
                    validate_b200_manifest_contract(
                        changed,
                        config_identity=self.identity,
                        recipe=self.bundle["recipe"],
                        model_contract=self.bundle["model"],
                    )


if __name__ == "__main__":
    unittest.main()
