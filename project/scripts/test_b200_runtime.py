#!/usr/bin/env python3
"""CPU-only tests for the B200 config and runtime contracts."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import b200_runtime as runtime  # noqa: E402


class _FakeCuda:
    def __init__(
        self,
        *,
        count: int = 2,
        name: str = "NVIDIA B200",
        capability: tuple[int, int] = (10, 0),
        memory: int = 183359 * 1024**2,
        arch_list: list[str] | None = None,
    ) -> None:
        self._count = count
        self._name = name
        self._capability = capability
        self._memory = memory
        self._arch_list = ["sm_90", "sm_100"] if arch_list is None else arch_list

    @staticmethod
    def is_available() -> bool:
        return True

    def device_count(self) -> int:
        return self._count

    def get_arch_list(self) -> list[str]:
        return list(self._arch_list)

    def get_device_properties(self, index: int) -> SimpleNamespace:
        if not 0 <= index < self._count:
            raise IndexError(index)
        return SimpleNamespace(name=self._name, total_memory=self._memory)

    def get_device_capability(self, index: int) -> tuple[int, int]:
        if not 0 <= index < self._count:
            raise IndexError(index)
        return self._capability


class B200RuntimeContractTest(unittest.TestCase):
    def test_large_asset_receipt_reuses_sealed_hashes_without_payload_reads(
        self,
    ) -> None:
        real_sha256_file = runtime.sha256_file
        hashed_paths: list[Path] = []

        def reject_payload_hash(path: str | os.PathLike[str]) -> str:
            candidate = Path(path)
            if candidate.name.endswith(".safetensors"):
                raise AssertionError(f"payload shard was rehashed: {candidate}")
            hashed_paths.append(candidate)
            return real_sha256_file(candidate)

        with mock.patch.object(
            runtime,
            "sha256_file",
            side_effect=reject_payload_hash,
        ):
            evidence = {
                role: runtime.validate_b200_large_asset_receipt(
                    runtime.B200_ASSET_INTEGRITY_RECEIPT,
                    role=role,
                )
                for role in ("training", "inference")
            }

        self.assertTrue(hashed_paths)
        for role, result in evidence.items():
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["role"], role)
            self.assertEqual(
                result["hash_verification_mode"],
                runtime.INHERITED_LARGE_HASH_MODE,
            )
            self.assertFalse(result["payload_rehash_performed"])
            self.assertEqual(len(result["source"]["shards"]), 94)

    def test_large_asset_receipt_rejects_wrong_path_and_role(self) -> None:
        with self.assertRaises(runtime.B200ContractError):
            runtime.validate_b200_large_asset_receipt(
                runtime.B200_BF16_DOWNLOAD_RECEIPT,
                role="training",
            )
        with self.assertRaisesRegex(runtime.B200ContractError, "unknown.*role"):
            runtime.validate_b200_large_asset_receipt(
                runtime.B200_ASSET_INTEGRITY_RECEIPT,
                role="evaluation",
            )

    def test_inherited_source_digests_are_exact_and_fail_closed(self) -> None:
        def fixtures() -> tuple[dict[str, object], dict[str, object]]:
            current_shards: list[dict[str, object]] = []
            sealed_shards: list[dict[str, object]] = []
            for index in range(1, 95):
                record = {
                    "filename": f"model-{index:05d}-of-00094.safetensors",
                    "file_bytes": 1_000 + index,
                    "payload_bytes": 900 + index,
                    "tensor_count": index,
                }
                current_shards.append({**record, "sha256": None})
                sealed_shards.append({**record, "sha256": f"{index:064x}"})
            shard_file_bytes = sum(
                int(record["file_bytes"]) for record in current_shards
            )
            return (
                {
                    "shards": current_shards,
                    "shard_file_bytes": shard_file_bytes,
                    "preserved": "value",
                },
                {
                    "status": "PASS",
                    "hash_verification_mode": runtime.INHERITED_LARGE_HASH_MODE,
                    "payload_rehash_performed": False,
                    "source": {
                        "shards": sealed_shards,
                        "shard_file_bytes": shard_file_bytes,
                    },
                },
            )

        inventory, evidence = fixtures()
        hydrated = runtime.inherit_b200_source_shard_digests(inventory, evidence)
        self.assertEqual(
            [record["sha256"] for record in hydrated["shards"]],
            [record["sha256"] for record in evidence["source"]["shards"]],
        )
        self.assertTrue(all(record["sha256"] is None for record in inventory["shards"]))
        self.assertEqual(
            hydrated["shard_hash_provenance"],
            runtime.INHERITED_LARGE_HASH_MODE,
        )
        self.assertFalse(hydrated["payload_rehash_performed"])
        self.assertEqual(hydrated["preserved"], "value")

        inventory, evidence = fixtures()
        inventory["shards"][0]["sha256"] = "f" * 64
        with self.assertRaisesRegex(runtime.B200ContractError, "recomputed"):
            runtime.inherit_b200_source_shard_digests(inventory, evidence)

        inventory, evidence = fixtures()
        evidence["source"]["shards"][0]["file_bytes"] += 1
        with self.assertRaisesRegex(runtime.B200ContractError, "file_bytes"):
            runtime.inherit_b200_source_shard_digests(inventory, evidence)

        inventory, evidence = fixtures()
        evidence["status"] = "FAIL"
        with self.assertRaisesRegex(runtime.B200ContractError, "invalid.*evidence"):
            runtime.inherit_b200_source_shard_digests(inventory, evidence)

    def test_config_bundle_is_exact_b200_profile(self) -> None:
        bundle = runtime.load_b200_config_bundle(runtime.B200_CONFIG_DIR)
        self.assertEqual(bundle["model"]["profile"], "b200")
        self.assertEqual(bundle["recipe"]["epochs"], 1)
        self.assertEqual(bundle["recipe"]["optimizer_steps"], 9316)
        self.assertEqual(
            bundle["data_contract"]["prepared_data_dir"],
            str(runtime.OWNED_ROOT / "data"),
        )
        self.assertEqual(
            bundle["data_contract"]["human_gold_raw"],
            str(
                runtime.OWNED_ROOT
                / "data/raw/human_gold/test-human-infer-vocab-final.jsonl"
            ),
        )
        self.assertNotEqual(
            bundle["data_contract"]["human_gold_raw"],
            bundle["data_contract"]["files"]["human_gold"]["path"],
        )
        self.assertEqual(
            set(bundle["hashes"]),
            {
                "model",
                "recipe",
                "data_contract",
            },
        )
        self.assertTrue(all(len(value) == 64 for value in bundle["hashes"].values()))

    def test_config_dir_must_be_explicit_exact_and_owned(self) -> None:
        with self.assertRaises(TypeError):
            runtime.load_b200_config_bundle()  # type: ignore[call-arg]
        with self.assertRaises(runtime.B200ContractError):
            runtime.resolve_b200_config_dir(runtime.PROJECT_DIR / "config")
        with self.assertRaises(runtime.B200ContractError):
            runtime.resolve_b200_config_dir(Path("/var/tmp"))

    def test_user_and_root_are_fail_closed(self) -> None:
        self.assertEqual(runtime.require_expected_user("e202"), "e202")
        with self.assertRaises(runtime.B200ContractError):
            runtime.require_expected_user("unexpected")
        self.assertEqual(
            runtime.require_owned_root(runtime.OWNED_ROOT),
            runtime.OWNED_ROOT,
        )
        with self.assertRaises(runtime.B200ContractError):
            runtime.require_owned_root(runtime.PROJECT_DIR)
        with self.assertRaises(runtime.B200ContractError):
            runtime.require_owned_path(Path("/var/tmp/not-owned"))

    def test_owned_evaluator_has_exact_identity(self) -> None:
        evaluator = runtime.require_owned_evaluator()
        self.assertEqual(evaluator, runtime.REFERENCE_EVALUATOR)
        self.assertEqual(
            runtime.sha256_file(evaluator),
            runtime.REFERENCE_EVALUATOR_SHA256,
        )
        with self.assertRaises(runtime.B200ContractError):
            runtime.require_owned_evaluator(runtime.B200_CONFIG_DIR / "model.json")

    def test_map_is_24_by_24_with_exact_endpoints(self) -> None:
        mapping = dict(runtime.EXPECTED_LAYER_DEVICE_MAP)
        self.assertEqual(runtime.BOUNDARY_LAYERS, (24,))
        self.assertEqual(len(mapping), 48)
        self.assertEqual(
            [layer for layer, device in mapping.items() if device == "cuda:0"],
            list(range(24)),
        )
        self.assertEqual(
            [layer for layer, device in mapping.items() if device == "cuda:1"],
            list(range(24, 48)),
        )
        self.assertEqual(runtime.EXPECTED_HF_DEVICE_MAP["model.embed_tokens"], "cuda:0")
        self.assertEqual(runtime.EXPECTED_HF_DEVICE_MAP["model.norm"], "cuda:1")
        self.assertEqual(runtime.EXPECTED_HF_DEVICE_MAP["lm_head"], "cuda:1")
        prohibited = f"cuda:{runtime.EXPECTED_LOGICAL_GPU_COUNT}"
        self.assertNotIn(prohibited, runtime.EXPECTED_HF_DEVICE_MAP.values())

    def test_physical_visibility_requires_exact_string(self) -> None:
        expected = {"CUDA_VISIBLE_DEVICES": "6,7"}
        self.assertEqual(runtime.require_physical_visibility(expected), "6,7")
        for value in ("7,6", "6, 7", "0,1", "", None):
            environment = {} if value is None else {"CUDA_VISIBLE_DEVICES": value}
            with self.subTest(value=value):
                with self.assertRaises(runtime.B200ContractError):
                    runtime.require_physical_visibility(environment)

    def test_gpu_gate_is_strict_but_injectable_for_cpu_test(self) -> None:
        fake_torch = SimpleNamespace(cuda=_FakeCuda())
        result = runtime.require_b200_gpu_runtime(
            fake_torch,
            environ={"CUDA_VISIBLE_DEVICES": "6,7"},
        )
        self.assertEqual(result["logical_gpu_count"], 2)
        self.assertEqual(
            [item["physical_index"] for item in result["logical_to_physical"]],
            [6, 7],
        )
        for bad_cuda in (
            _FakeCuda(count=1),
            _FakeCuda(name="NVIDIA Other"),
            _FakeCuda(capability=(9, 0)),
            _FakeCuda(memory=runtime.MINIMUM_MEMORY_BYTES_PER_GPU - 1),
            _FakeCuda(arch_list=["sm_90"]),
        ):
            with self.subTest(cuda=bad_cuda):
                with self.assertRaises(runtime.B200ContractError):
                    runtime.require_b200_gpu_runtime(
                        SimpleNamespace(cuda=bad_cuda),
                        environ={"CUDA_VISIBLE_DEVICES": "6,7"},
                    )

    def test_cpu_helpers_do_not_import_torch(self) -> None:
        real_import = runtime.importlib.import_module

        def guarded_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "torch":
                raise AssertionError("CPU-safe helper imported torch")
            return real_import(name, *args, **kwargs)

        with mock.patch.object(
            runtime.importlib, "import_module", side_effect=guarded_import
        ):
            runtime.load_b200_config_bundle(runtime.B200_CONFIG_DIR)
            runtime.require_owned_evaluator()
            runtime.require_physical_visibility({"CUDA_VISIBLE_DEVICES": "6,7"})

    def test_sealed_nf4_runtime_allows_only_pinned_b200_contract(self) -> None:
        result = runtime.require_sealed_nf4_runtime(
            capability=(10, 0),
            manifest_status="SEALED",
            bitsandbytes_version="0.49.2",
            native_library="/owned/lib/libbitsandbytes_cuda128.so",
        )
        self.assertEqual(result["status"], "PASS")
        failures = (
            {"capability": (9, 0)},
            {"manifest_status": "BUILDING"},
            {"bitsandbytes_version": "0.49.1"},
            {"native_library": "/owned/lib/libbitsandbytes_cpu.so"},
        )
        baseline = {
            "capability": (10, 0),
            "manifest_status": "SEALED",
            "bitsandbytes_version": "0.49.2",
            "native_library": "/owned/lib/libbitsandbytes_cuda128.so",
        }
        for change in failures:
            with self.subTest(change=change):
                arguments = {**baseline, **change}
                with self.assertRaises(runtime.B200ContractError):
                    runtime.require_sealed_nf4_runtime(**arguments)

    def test_atomic_json_is_owned_atomic_and_duplicate_safe(self) -> None:
        with tempfile.TemporaryDirectory(
            dir=runtime.OWNED_ROOT / "runtime/tmp"
        ) as directory:
            path = Path(directory) / "receipt.json"
            returned = runtime.atomic_json(path, {"status": "PASS", "value": 1})
            self.assertEqual(returned, path.resolve())
            self.assertEqual(
                runtime.read_json_no_duplicates(path),
                {"status": "PASS", "value": 1},
            )
            self.assertFalse(any(Path(directory).glob(".*.tmp")))
            duplicate = Path(directory) / "duplicate.json"
            duplicate.write_text('{"key": 1, "key": 2}\\n', encoding="utf-8")
            with self.assertRaises(runtime.B200ContractError):
                runtime.read_json_no_duplicates(duplicate)
        with self.assertRaises(runtime.B200ContractError):
            runtime.atomic_json(Path("/var/tmp/not-owned.json"), {"bad": True})

    def test_config_contains_no_source_paths_or_unapproved_device(self) -> None:
        legacy_source_prefix = "/" + "data" + "/" + "home"
        prohibited_device = f"cuda:{runtime.EXPECTED_LOGICAL_GPU_COUNT}"
        for path in runtime.resolve_b200_config_paths(runtime.B200_CONFIG_DIR).values():
            text = path.read_text(encoding="utf-8")
            self.assertNotIn(legacy_source_prefix, text)
            self.assertNotIn(prohibited_device, text)
            keys = {key.lower() for key in self._json_keys(json.loads(text))}
            self.assertTrue(
                keys.isdisjoint(
                    {
                        "api_key",
                        "access_key",
                        "secret",
                        "password",
                        "credential",
                    }
                )
            )

    @classmethod
    def _json_keys(cls, value: object) -> list[str]:
        if isinstance(value, dict):
            keys = list(value)
            for item in value.values():
                keys.extend(cls._json_keys(item))
            return keys
        if isinstance(value, list):
            keys: list[str] = []
            for item in value:
                keys.extend(cls._json_keys(item))
            return keys
        return []


if __name__ == "__main__":
    unittest.main()
