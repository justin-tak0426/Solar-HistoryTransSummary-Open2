#!/usr/bin/env python3
"""CPU-only tests for the relocated B200 preflight."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import b200_runtime  # noqa: E402
import preflight_model as preflight  # noqa: E402


def fake_hardware() -> dict[str, Any]:
    return {
        "status": "PASS",
        "physical_cuda_visible_devices": "6,7",
        "logical_gpu_count": 2,
        "torch_arch_list": ["sm_90", "sm_100"],
        "boundary_layers": [24],
        "logical_to_physical": [
            {
                "logical_index": index,
                "logical_device": f"cuda:{index}",
                "physical_index": 6 + index,
                "name": "NVIDIA B200",
                "compute_capability": [10, 0],
                "total_memory_bytes": 183359 * 1024**2,
            }
            for index in range(2)
        ],
    }


class B200PreflightTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = b200_runtime.load_b200_config_bundle(b200_runtime.B200_CONFIG_DIR)
        cls.transfer = preflight.validate_b200_transfer_receipt(
            preflight.B200_TRANSFER_RECEIPT
        )
        cls.bf16 = preflight.validate_b200_relocated_bf16(
            preflight.B200_DOWNLOAD_RECEIPT,
            b200_runtime.BF16_SOURCE_DIR,
            cls.transfer,
        )

    def test_relocated_bf16_receipt_config_and_index(self) -> None:
        self.assertEqual(self.bf16["status"], "PASS")
        self.assertEqual(
            self.bf16["revision"], self.bundle["model"]["base"]["revision"]
        )
        self.assertEqual(self.bf16["indexed_shards"], 94)
        self.assertEqual(self.bf16["indexed_tensors"], 47019)
        self.assertTrue(self.bf16["relocation_verified_by_content_hash"])
        self.assertEqual(
            self.bf16["relocated_snapshot_dir"],
            str(b200_runtime.BF16_SOURCE_DIR),
        )

    def test_both_nf4_manifest_and_index_identities(self) -> None:
        source_config_sha256 = self.bf16["contract_hashes"]["config.json"]
        cases = (
            (
                "training",
                b200_runtime.TRAIN_EXPERT_DIR,
                "solar-open2-layer-flat-routed-experts-nf4",
                48,
                2880,
            ),
            (
                "inference",
                b200_runtime.INFERENCE_EXPERT_DIR,
                "solar-open2-routed-experts-nf4",
                960,
                276480,
            ),
        )
        for role, path, schema, files, tensors in cases:
            with self.subTest(role=role):
                result = preflight.validate_b200_nf4_artifact(
                    path,
                    role=role,
                    expected_path=path,
                    expected_schema=schema,
                    expected_payload_files=files,
                    transfer=self.transfer,
                    source_config_sha256=source_config_sha256,
                )
                self.assertEqual(result["status"], "PASS")
                self.assertEqual(result["payload_file_count"], files)
                self.assertEqual(result["index"]["serialized_tensor_count"], tensors)

    def test_environment_receipt_identity(self) -> None:
        result = preflight.validate_b200_environment_receipt(
            preflight.B200_ENVIRONMENT_RECEIPT
        )
        self.assertEqual(result["status"], "PASS")
        self.assertIn("sm_100", result["torch"]["compiled_arch_flags"])
        self.assertEqual(
            Path(result["bitsandbytes_cuda128"]["path"]).name,
            "libbitsandbytes_cuda128.so",
        )

    def test_raw_and_prepared_human_gold_are_distinct(self) -> None:
        contract = self.bundle["data_contract"]
        raw = Path(contract["human_gold_raw"])
        prepared = Path(contract["files"]["human_gold"]["path"])
        self.assertNotEqual(raw, prepared)
        self.assertEqual(
            contract["human_gold_raw_sha256"],
            "996a094e9abca028bf89a647a86645715b9a6b05331f85dead4f3f5197eb531b",
        )
        self.assertEqual(
            contract["files"]["human_gold"]["sha256"],
            "d841fa159461e86b3be84b3c3f1fecf32b457f18221ea3e94f8cfd67c8fa3bf5",
        )

    def test_b200_receipt_refuses_overwrite_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory(
            dir=b200_runtime.OWNED_ROOT / "runtime/tmp"
        ) as temporary:
            directory = Path(temporary)
            existing = directory / "existing.json"
            existing.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                preflight.require_fresh_b200_receipt(existing)
            target = directory / "target.json"
            link = directory / "link.json"
            link.symlink_to(target)
            with self.assertRaisesRegex(RuntimeError, "symlink"):
                preflight.require_fresh_b200_receipt(link)

    def test_b200_run_binds_all_configs_and_mocked_gpu_contract(self) -> None:
        data = {
            "status": "PASS",
            "raw_human_gold": {
                "path": self.bundle["data_contract"]["human_gold_raw"],
                "sha256": self.bundle["data_contract"]["human_gold_raw_sha256"],
            },
            "files": {
                "human_gold": {
                    "path": self.bundle["data_contract"]["files"]["human_gold"]["path"],
                    "sha256": self.bundle["data_contract"]["files"]["human_gold"][
                        "sha256"
                    ],
                }
            },
        }
        software = {
            "status": "PASS",
            "bitsandbytes_runtime": {
                "native_library": "libbitsandbytes_cuda128.so",
            },
        }
        result = preflight.run_b200_preflight(
            config_dir=b200_runtime.B200_CONFIG_DIR,
            source_dir=b200_runtime.BF16_SOURCE_DIR,
            training_expert_dir=b200_runtime.TRAIN_EXPERT_DIR,
            inference_expert_dir=b200_runtime.INFERENCE_EXPERT_DIR,
            download_receipt=preflight.B200_DOWNLOAD_RECEIPT,
            environment_receipt=preflight.B200_ENVIRONMENT_RECEIPT,
            transfer_receipt=preflight.B200_TRANSFER_RECEIPT,
            hardware_validator=fake_hardware,
            software_validator=lambda: software,
            data_validator=lambda _bundle: data,
        )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["profile"], "b200")
        self.assertEqual(result["physical_cuda_visible_devices"], "6,7")
        self.assertEqual(
            set(result["config"]["files"]),
            {
                "model",
                "recipe",
                "data_contract",
            },
        )
        for entry in result["config"]["files"].values():
            self.assertTrue(Path(entry["path"]).is_absolute())
            self.assertEqual(len(entry["sha256"]), 64)
        self.assertNotEqual(
            result["data"]["raw_human_gold"]["path"],
            result["data"]["files"]["human_gold"]["path"],
        )
        self.assertFalse(result["production_authorized"])

    def test_mocked_gpu_contract_is_still_fail_closed(self) -> None:
        hardware = fake_hardware()
        hardware["logical_to_physical"][1]["compute_capability"] = [9, 0]
        dummy_transfer = {
            "path": "transfer",
            "sha256": "0" * 64,
            "receipt": {},
        }
        dummy_bf16 = {
            "contract_hashes": {"config.json": "1" * 64},
        }
        with mock.patch.object(
            preflight,
            "validate_b200_transfer_receipt",
            return_value=dummy_transfer,
        ), mock.patch.object(
            preflight,
            "validate_b200_relocated_bf16",
            return_value=dummy_bf16,
        ), mock.patch.object(
            preflight,
            "validate_b200_nf4_artifact",
            return_value={"status": "PASS"},
        ), mock.patch.object(
            preflight,
            "validate_b200_environment_receipt",
            return_value={"status": "PASS"},
        ):
            with self.assertRaisesRegex(RuntimeError, "compute_capability"):
                preflight.run_b200_preflight(
                    config_dir=b200_runtime.B200_CONFIG_DIR,
                    source_dir=b200_runtime.BF16_SOURCE_DIR,
                    training_expert_dir=b200_runtime.TRAIN_EXPERT_DIR,
                    inference_expert_dir=b200_runtime.INFERENCE_EXPERT_DIR,
                    download_receipt=preflight.B200_DOWNLOAD_RECEIPT,
                    environment_receipt=preflight.B200_ENVIRONMENT_RECEIPT,
                    transfer_receipt=preflight.B200_TRANSFER_RECEIPT,
                    hardware_validator=lambda: hardware,
                    software_validator=lambda: {"status": "PASS"},
                    data_validator=lambda _bundle: {"status": "PASS"},
                )

    def test_cli_has_explicit_b200_inputs(self) -> None:
        options = {
            option
            for action in preflight.build_parser()._actions
            for option in action.option_strings
        }
        self.assertTrue(
            {
                "--config-dir",
                "--source-dir",
                "--expert-nf4-dir",
                "--inference-expert-nf4-dir",
                "--download-receipt",
                "--environment-receipt",
                "--transfer-integrity-receipt",
                "--receipt",
            }.issubset(options)
        )


if __name__ == "__main__":
    unittest.main()
