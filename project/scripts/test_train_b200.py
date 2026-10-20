#!/usr/bin/env python3
"""CPU-only integration contracts for the B200 training entrypoint."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

import train  # noqa: E402
from b200_runtime import B200_ASSET_INTEGRITY_RECEIPT, B200_CONFIG_DIR  # noqa: E402
from common import (  # noqa: E402
    CONFIG_DIR,
    GlobalAugmentedDeckSampler,
    PROJECT,
    canonical_sha256,
)


OWNED_RUNTIME_TMP = B200_CONFIG_DIR.parents[2] / "runtime" / "tmp"


class B200TrainContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.b200_config = train.load_training_config(B200_CONFIG_DIR)
        cls.source_config = train.load_training_config(CONFIG_DIR)
        cls.b200_recipe = cls.b200_config["recipe"]
        cls.source_recipe = cls.source_config["recipe"]

    def b200_history_through(
        self,
        completed_step: int,
        *,
        include_current: bool = True,
    ) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for step in range(completed_step + 1):
            scope = train.configured_validation_scope(
                self.b200_recipe,
                step,
                smoke=False,
            )
            if scope is None or (step == completed_step and not include_current):
                continue
            result.append({"step": step, "scope": scope})
        return result

    def b200_preflight_receipt(self) -> dict[str, object]:
        config = self.b200_config
        model = config["model"]
        hardware = model["hardware"]
        source_dir = model["training_runtime"]["source_dir"]
        expert_dir = model["training_runtime"]["expert_dir"]
        return {
            "schema": "solar_open2_b200_preflight",
            "schema_version": 1,
            "status": "PASS",
            "profile": "b200",
            "user": model["expected_user"],
            "root": model["owned_root"],
            "config": config["identity"],
            "pipeline_code": train.training_pipeline_code_identity(config),
            "physical_cuda_visible_devices": hardware["physical_cuda_visible_devices"],
            "production_authorized": False,
            "bf16": {
                "status": "PASS",
                "relocated_snapshot_dir": source_dir,
                "revision": model["base"]["revision"],
                "indexed_shards": model["base"]["weight_shard_count"],
                "indexed_tensors": model["base"]["indexed_tensor_count"],
                "relocation_verified_by_content_hash": True,
                "receipt_sha256": "a" * 64,
            },
            "training_nf4": {
                "status": "PASS",
                "role": "training",
                "path": expert_dir,
                "schema": model["training_runtime"]["expert_artifact_schema"],
                "payload_file_count": 48,
                "payload_hashes_inherited_from_full_transfer_receipt": True,
                "manifest": {
                    "status": "SEALED",
                    "sha256": "b" * 64,
                },
            },
            "environment": {
                "dependency_lock": {
                    "path": str(PROJECT / "requirements/solar-open-2.lock.txt"),
                    "sha256": train.sha256(
                        PROJECT / "requirements/solar-open-2.lock.txt"
                    ),
                }
            },
            "transfer_integrity": {"large_hashes_checked": True},
            "hardware": {
                "status": "PASS",
                "logical_gpu_count": 2,
                "physical_cuda_visible_devices": hardware[
                    "physical_cuda_visible_devices"
                ],
                "boundary_layers": hardware["boundary_layers"],
                "logical_to_physical": [
                    {
                        "logical_index": index,
                        "logical_device": hardware["logical_devices"][index],
                        "physical_index": hardware["physical_gpu_indices"][index],
                        "name": hardware["required_gpu_name"],
                        "compute_capability": [
                            hardware["required_compute_capability"]["major"],
                            hardware["required_compute_capability"]["minor"],
                        ],
                        "total_memory_bytes": hardware["minimum_memory_bytes_per_gpu"],
                    }
                    for index in range(2)
                ],
            },
        }

    def test_cli_requires_explicit_config_directory(self) -> None:
        required_without_config = [
            "--source-dir",
            str(OWNED_RUNTIME_TMP / "source"),
            "--expert-nf4-dir",
            str(OWNED_RUNTIME_TMP / "expert"),
            "--data-dir",
            str(OWNED_RUNTIME_TMP / "data"),
            "--output-dir",
            str(OWNED_RUNTIME_TMP / "output"),
            "--asset-integrity-receipt",
            str(B200_ASSET_INTEGRITY_RECEIPT),
            "--preflight-receipt",
            str(OWNED_RUNTIME_TMP / "preflight.json"),
        ]
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                train.build_argument_parser().parse_args(required_without_config)
        parsed = train.build_argument_parser().parse_args(
            ["--config-dir", str(B200_CONFIG_DIR), *required_without_config]
        )
        self.assertEqual(parsed.config_dir, B200_CONFIG_DIR)
        self.assertEqual(
            parsed.asset_integrity_receipt,
            B200_ASSET_INTEGRITY_RECEIPT,
        )
        self.assertIsNone(parsed.production_approval_receipt)

    def test_b200_loader_call_uses_receipt_backed_hash_mode(self) -> None:
        source = Path(train.__file__).read_text(encoding="utf-8")
        self.assertIn('"verify_hashes": not b200', source)
        self.assertIn(
            'loader_kwargs["asset_integrity_receipt"] = asset_integrity_receipt',
            source,
        )

    def test_only_explicit_source_or_b200_config_is_accepted(self) -> None:
        self.assertEqual(self.b200_config["profile"], "b200")
        self.assertEqual(
            self.b200_config["config_dir"],
            B200_CONFIG_DIR.resolve(),
        )
        self.assertEqual(self.source_config["profile"], "source")
        self.assertEqual(
            self.source_config["config_dir"],
            CONFIG_DIR.resolve(),
        )
        for config in (self.b200_config, self.source_config):
            self.assertEqual(
                set(config["hashes"]),
                {"model", "recipe", "data_contract"},
            )
            self.assertEqual(
                set(config["identity"]["files"]),
                {"model", "recipe", "data_contract"},
            )
        with self.assertRaises(RuntimeError):
            train.load_training_config(CONFIG_DIR.parent)

    def test_b200_pipeline_identity_binds_porting_assets(self) -> None:
        b200_identity = train.training_pipeline_code_identity(self.b200_config)
        source_identity = train.training_pipeline_code_identity(self.source_config)
        self.assertIn("config/b200/recipe.json", b200_identity["files"])
        if (PROJECT / "launch" / "b200").is_dir():
            self.assertTrue(
                any(path.startswith("launch/b200/") for path in b200_identity["files"])
            )
        self.assertFalse(any("__pycache__" in path for path in b200_identity["files"]))
        self.assertFalse(any(path.endswith(".pyc") for path in b200_identity["files"]))
        self.assertTrue(
            any(path.startswith("migration/b200/") for path in b200_identity["files"])
        )
        self.assertNotIn("config/b200/recipe.json", source_identity["files"])
        changed = dict(self.b200_config)
        changed["config_dir"] = CONFIG_DIR
        with self.assertRaises(RuntimeError):
            train.training_pipeline_code_identity(changed)

    def test_b200_wandb_identities_are_pinned_and_separate(self) -> None:
        production = train.b200_wandb_identity(
            smoke=False,
            mode="online",
        )
        smoke = train.b200_wandb_identity(
            smoke=True,
            mode="disabled",
        )
        self.assertEqual(
            production,
            {
                "mode": "online",
                "entity": "junhyuk-tak-markr",
                "project": "modern-korean-lm",
                "run_id": "solar_open2_b200_1epoch_20260802_v2",
                "run_name": ("Solar Open2 250B 2xB200 1epoch len9600 optimized v2"),
                "resume": "allow",
            },
        )
        self.assertEqual(
            smoke["run_id"],
            "solar_open2_b200_smoke_20260802_v2",
        )
        self.assertEqual(smoke["mode"], "disabled")
        self.assertNotEqual(smoke["run_id"], production["run_id"])
        self.assertNotEqual(smoke["run_name"], production["run_name"])

    def test_b200_preflight_consumer_uses_nested_receipt_schema(self) -> None:
        receipt = self.b200_preflight_receipt()
        model = self.b200_config["model"]
        with tempfile.TemporaryDirectory(dir=OWNED_RUNTIME_TMP) as temporary:
            path = Path(temporary) / "preflight.json"
            path.write_text(
                json.dumps(receipt, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                train.validate_preflight(
                    path,
                    source_dir=Path(model["training_runtime"]["source_dir"]),
                    expert_nf4_dir=Path(model["training_runtime"]["expert_dir"]),
                    training_config=self.b200_config,
                ),
                receipt,
            )
            receipt["hardware"]["logical_to_physical"][1]["physical_index"] = 5
            path.write_text(
                json.dumps(receipt, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "physical_index"):
                train.validate_preflight(
                    path,
                    source_dir=Path(model["training_runtime"]["source_dir"]),
                    expert_nf4_dir=Path(model["training_runtime"]["expert_dir"]),
                    training_config=self.b200_config,
                )

    def test_b200_preflight_state_hashes_use_nested_evidence(self) -> None:
        receipt = self.b200_preflight_receipt()
        self.assertEqual(
            train.preflight_state_hashes(receipt, b200=True),
            {
                "download_receipt_sha256": "a" * 64,
                "expert_manifest_sha256": "b" * 64,
            },
        )
        receipt["bf16"]["receipt_sha256"] = "not-a-digest"
        with self.assertRaisesRegex(RuntimeError, "download_receipt_sha256"):
            train.preflight_state_hashes(receipt, b200=True)

    def test_production_approval_value_is_exact_and_fail_closed(self) -> None:
        smoke_receipt_path = (
            B200_CONFIG_DIR.parents[2]
            / "runtime"
            / "receipts"
            / "b200_smoke_2gpu_len9600_v1.json"
        )
        output_dir = (
            B200_CONFIG_DIR.parents[2] / "runtime" / "runs" / "b200_1epoch_len9600_v1"
        )
        pipeline_identity = train.training_pipeline_code_identity(self.b200_config)
        wandb_identity = train.b200_wandb_identity(
            smoke=False,
            mode="online",
        )
        approval = train.expected_b200_production_approval(
            smoke_receipt_path=smoke_receipt_path,
            smoke_receipt_sha256="a" * 64,
            training_config=self.b200_config,
            pipeline_identity=pipeline_identity,
            output_dir=output_dir,
            wandb_identity=wandb_identity,
        )
        self.assertIs(
            train.validate_b200_production_approval_value(
                approval,
                smoke_receipt_path=smoke_receipt_path,
                smoke_receipt_sha256="a" * 64,
                training_config=self.b200_config,
                pipeline_identity=pipeline_identity,
                output_dir=output_dir,
                wandb_identity=wandb_identity,
            ),
            approval,
        )
        mutations = {
            "status": lambda value: value.update({"status": "PENDING"}),
            "human approval": lambda value: value.update(
                {"human_production_approval_recorded": False}
            ),
            "smoke hash": lambda value: value["technical_smoke_receipt"].update(
                {"sha256": "b" * 64}
            ),
            "config": lambda value: value.update({"config_identity": {}}),
            "code": lambda value: value.update({"pipeline_code_sha256": "c" * 64}),
            "extra field": lambda value: value.update({"extra": True}),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                changed = copy.deepcopy(approval)
                mutate(changed)
                with self.assertRaises(RuntimeError):
                    train.validate_b200_production_approval_value(
                        changed,
                        smoke_receipt_path=smoke_receipt_path,
                        smoke_receipt_sha256="a" * 64,
                        training_config=self.b200_config,
                        pipeline_identity=pipeline_identity,
                        output_dir=output_dir,
                        wandb_identity=wandb_identity,
                    )

    def test_b200_exact_tail_boundaries_and_denominators(self) -> None:
        self.assertEqual(
            train.training_occurrence_contract(self.b200_recipe),
            {
                "nominal_step_occurrences": 16,
                "total_occurrences": 149_048,
                "optimizer_steps": 9_316,
                "final_step_occurrences": 8,
            },
        )
        self.assertEqual(
            train.update_occurrence_bounds(self.b200_recipe, 1),
            (0, 16),
        )
        self.assertEqual(
            train.update_occurrence_bounds(self.b200_recipe, 9_315),
            (149_024, 149_040),
        )
        self.assertEqual(
            train.update_occurrence_bounds(self.b200_recipe, 9_316),
            (149_040, 149_048),
        )
        self.assertEqual(
            train.update_occurrences_at_step(self.b200_recipe, 9_315),
            16,
        )
        self.assertEqual(
            train.update_occurrences_at_step(self.b200_recipe, 9_316),
            8,
        )
        self.assertEqual(
            train.consumed_occurrences_at_step(self.b200_recipe, 9_315),
            149_040,
        )
        self.assertEqual(
            train.consumed_occurrences_at_step(self.b200_recipe, 9_316),
            149_048,
        )
        with self.assertRaises(ValueError):
            train.update_occurrence_bounds(self.b200_recipe, 9_317)

        invalid = copy.deepcopy(self.b200_recipe)
        invalid["exact_tail"]["final_step_occurrences"] = 16
        with self.assertRaises(RuntimeError):
            train.training_occurrence_contract(invalid)

    def test_source_config_retains_full_sixteen_occurrence_updates(self) -> None:
        self.assertEqual(
            train.training_occurrence_contract(self.source_recipe),
            {
                "nominal_step_occurrences": 16,
                "total_occurrences": 298_096,
                "optimizer_steps": 18_631,
                "final_step_occurrences": 16,
            },
        )
        self.assertEqual(
            train.update_occurrence_bounds(self.source_recipe, 18_631),
            (298_080, 298_096),
        )

    def test_b200_validation_schedule_and_source_compatibility(self) -> None:
        self.assertEqual(
            train.configured_validation_scope(
                self.b200_recipe,
                0,
                smoke=False,
            ),
            "fixed_subset",
        )
        self.assertEqual(
            train.configured_validation_scope(
                self.b200_recipe,
                1_000,
                smoke=False,
            ),
            "fixed_subset",
        )
        self.assertEqual(
            train.configured_validation_scope(
                self.b200_recipe,
                4_658,
                smoke=False,
            ),
            "full",
        )
        self.assertEqual(
            train.configured_validation_scope(
                self.b200_recipe,
                9_316,
                smoke=False,
            ),
            "full",
        )
        self.assertIsNone(
            train.configured_validation_scope(
                self.b200_recipe,
                999,
                smoke=False,
            )
        )
        self.assertIsNone(
            train.configured_validation_scope(
                self.source_recipe,
                0,
                smoke=False,
            )
        )
        self.assertEqual(
            train.configured_validation_scope(
                self.source_recipe,
                9_316,
                smoke=False,
            ),
            "full",
        )
        self.assertIsNone(
            train.configured_validation_scope(
                self.b200_recipe,
                0,
                smoke=True,
            )
        )
        self.assertEqual(
            train.configured_validation_scope(
                self.b200_recipe,
                1,
                smoke=True,
            ),
            "fixed_subset",
        )

    def test_b200_coverage_decision_never_early_stops(self) -> None:
        pending_history = self.b200_history_through(9_315)
        pending = train.b200_coverage_decision(
            pending_history,
            completed_step=9_315,
            recipe=self.b200_recipe,
        )
        self.assertEqual(pending["status"], "PENDING_COVERAGE")
        self.assertFalse(pending["triggered"])
        self.assertEqual(pending["completed_occurrences"], 149_040)

        awaiting_history = self.b200_history_through(
            9_316,
            include_current=False,
        )
        awaiting_validation = train.b200_coverage_decision(
            awaiting_history,
            completed_step=9_316,
            recipe=self.b200_recipe,
        )
        self.assertEqual(
            awaiting_validation["status"],
            "PENDING_FINAL_VALIDATION",
        )
        self.assertFalse(awaiting_validation["triggered"])

        completed_history = self.b200_history_through(9_316)
        completed = train.b200_coverage_decision(
            completed_history,
            completed_step=9_316,
            recipe=self.b200_recipe,
        )
        self.assertEqual(completed["status"], "MAXIMUM_REACHED")
        self.assertFalse(completed["triggered"])
        self.assertEqual(completed["completed_occurrences"], 149_048)
        self.assertEqual(completed["completed_epochs"], 1.0)
        self.assertEqual(
            completed["completed_candidate_steps"],
            [4_658, 9_316],
        )

        invalid_policy = copy.deepcopy(self.b200_recipe)
        invalid_policy["validation"]["early_stopping"]["enabled"] = True
        with self.assertRaises(ValueError):
            train.b200_coverage_decision(
                [],
                completed_step=0,
                recipe=invalid_policy,
            )
        with self.assertRaises(ValueError):
            train.b200_coverage_decision(
                [{"step": 0, "scope": "full_test"}],
                completed_step=0,
                recipe=self.b200_recipe,
            )
        with self.assertRaises(ValueError):
            train.b200_coverage_decision(
                [{"step": 0, "scope": "fixed_subset"}],
                completed_step=2_000,
                recipe=self.b200_recipe,
            )
        smoke = train.b200_coverage_decision(
            [{"step": 1, "scope": "fixed_subset"}],
            completed_step=1,
            recipe=self.b200_recipe,
            smoke=True,
        )
        self.assertEqual(smoke["status"], "PENDING_COVERAGE")
        with self.assertRaises(ValueError):
            train.b200_coverage_decision(
                [{"step": 1, "scope": "fixed_subset"}],
                completed_step=1,
                recipe=self.b200_recipe,
            )

    def test_actual_last_eight_sampler_identities_are_sealed(self) -> None:
        data_path = (
            Path(self.b200_config["data_contract"]["prepared_data_dir"])
            / self.b200_config["data_contract"]["files"]["train"]["name"]
        )
        rows: list[dict[str, str]] = []
        with data_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                rows.append(
                    {
                        "id": str(row["id"]),
                        "task": str(row["task"]),
                        "quality": str(row["quality"]),
                    }
                )
        sampler = GlobalAugmentedDeckSampler(
            rows,
            expert_label=self.b200_recipe["expert_quality_label"],
            expert_boost=self.b200_recipe["expert_oversampling_factor"],
            epochs=self.b200_recipe["epochs"],
            seed=self.b200_recipe["seed"],
        )
        tail = sampler.occurrences(149_040, 8)
        self.assertEqual(sampler.cursor, 0)
        self.assertEqual(
            [item["global_position"] for item in tail],
            list(range(149_040, 149_048)),
        )
        self.assertEqual(len({item["identity_sha256"] for item in tail}), 8)
        self.assertEqual(
            len({(item["id"], item["task"]) for item in tail}),
            8,
        )
        self.assertEqual(
            canonical_sha256(tail),
            "2d9059e3f0130eb99223b4cab2a22da0" "9090decfd4a4aa920a7528418e7eb14f",
        )

    def test_b200_monitoring_never_uses_sealed_evaluation_data(self) -> None:
        validation = self.b200_recipe["validation"]
        evaluation = self.b200_recipe["evaluation"]
        self.assertEqual(validation["monitor_datasets"], ["validation"])
        self.assertFalse(evaluation["used_for_training_monitor_or_selection"])
        self.assertEqual(
            set(evaluation["datasets"]),
            {"full_test", "human_gold"},
        )
        source = Path(train.__file__).read_text(encoding="utf-8")
        self.assertNotIn("cuda:" + "2", source)


if __name__ == "__main__":
    unittest.main()
