#!/usr/bin/env python3
"""Fail-closed official-BF16 plus custom routed-expert NF4 preflight."""

from __future__ import annotations

import argparse
import getpass
import importlib
import importlib.metadata as importlib_metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any, Callable

import b200_runtime

from common import (
    CONFIG_DIR,
    PROJECT,
    atomic_json,
    pipeline_code_identity,
    read_json,
    require_owned_path,
    sha256,
)
from quantization.bnb_nf4 import (
    require_b200_sealed_bitsandbytes,
    require_bitsandbytes,
)
from quantization.contract import (
    BASE_MODEL_ID,
    BASE_REVISION,
    EXPECTED_SOURCE_SHARD_COUNT,
    EXPECTED_SOURCE_TENSOR_COUNT,
)
from quantization.layer_flat_checkpoint import preflight_layer_flat_checkpoint


SOLAR_ENV = (PROJECT / "solar-open-2").resolve()
B200_DOWNLOAD_RECEIPT = (
    b200_runtime.OWNED_ROOT / "runtime/receipts/source_bf16_download.json"
)
B200_ENVIRONMENT_RECEIPT = (
    b200_runtime.OWNED_ROOT / "runtime/receipts/b200_environment_v1.json"
)
B200_TRANSFER_RECEIPT = (
    b200_runtime.OWNED_ROOT / "runtime/receipts/b200_transfer_integrity_full.json"
)
NF4_MANIFEST = "expert_nf4_manifest.json"
NF4_INDEX = "expert_nf4.safetensors.index.json"


def validate_download_receipt(path: Path, source_dir: Path) -> dict[str, Any]:
    receipt = read_json(path)
    wanted = {
        "status": "PASS",
        "repo_id": BASE_MODEL_ID,
        "requested_revision": BASE_REVISION,
        "repo_sha": BASE_REVISION,
        "snapshot_dir": str(source_dir),
        "indexed_shards": EXPECTED_SOURCE_SHARD_COUNT,
        "indexed_tensors": EXPECTED_SOURCE_TENSOR_COUNT,
        "local_dir_copy": False,
        "resumable_hf_cache": True,
    }
    for key, expected in wanted.items():
        if receipt.get(key) != expected:
            raise RuntimeError(
                f"download receipt mismatch: {key}={receipt.get(key)!r}, "
                f"expected={expected!r}"
            )
    hashes = receipt.get("contract_hashes") or {}
    for name in ("config.json", "model.safetensors.index.json", "LICENSE"):
        if hashes.get(name) != sha256(source_dir / name):
            raise RuntimeError(f"downloaded source changed after receipt: {name}")
    source_shards = receipt.get("source_shards")
    if not isinstance(source_shards, list) or len(source_shards) != (
        EXPECTED_SOURCE_SHARD_COUNT
    ):
        raise RuntimeError("download receipt lacks all BF16 shard digests")
    expected_names = {
        f"model-{index:05d}-of-{EXPECTED_SOURCE_SHARD_COUNT:05d}.safetensors"
        for index in range(1, EXPECTED_SOURCE_SHARD_COUNT + 1)
    }
    observed_names = set()
    for record in source_shards:
        if not isinstance(record, dict):
            raise RuntimeError("download shard receipt is not an object")
        name = record.get("filename")
        digest = record.get("sha256")
        if (
            name in observed_names
            or name not in expected_names
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or int(record.get("file_bytes", 0)) <= 0
            or int(record.get("payload_bytes", 0)) <= 0
            or int(record.get("tensor_count", 0)) <= 0
        ):
            raise RuntimeError(f"invalid download shard receipt: {record!r}")
        observed_names.add(name)
    if observed_names != expected_names:
        raise RuntimeError("download shard receipt coverage changed")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "repo_sha": receipt["repo_sha"],
        "weight_bytes": receipt["verified_tensor_bytes"],
        "cache_used_bytes": receipt["cache_used_bytes"],
        "source_shards": source_shards,
    }


def validate_expert_artifact(path: Path) -> dict[str, Any]:
    result = preflight_layer_flat_checkpoint(path, hash_shards=True)
    if result.get("status") != "PASS" or not result.get("production_eligible"):
        raise RuntimeError("layer-flat NF4 artifact is not production eligible")
    return result


def validate_software() -> dict[str, Any]:
    if (
        Path(sys.prefix).resolve() != SOLAR_ENV
        or Path(sys.base_prefix).resolve() != SOLAR_ENV
    ):
        raise RuntimeError(
            f"Solar commands must use the self-contained {SOLAR_ENV}, "
            f"found prefix={Path(sys.prefix).resolve()} "
            f"base_prefix={Path(sys.base_prefix).resolve()}"
        )
    torch, _bnb = require_bitsandbytes(require_cuda=True)
    transformers = importlib.import_module("transformers")
    peft = importlib.import_module("peft")
    fla = importlib.import_module("fla")
    version = str(transformers.__version__)
    if not version.startswith("5.14.1"):
        raise RuntimeError(
            "training requires Upstage Transformers v5.14.1-solar-open2, "
            f"found={version}"
        )
    solar = importlib.import_module(
        "transformers.models.solar_open2.modeling_solar_open2"
    )
    if not hasattr(solar, "SolarOpen2ForCausalLM"):
        raise RuntimeError("native SolarOpen2ForCausalLM is unavailable")
    modules = {
        "torch": torch,
        "transformers": transformers,
        "peft": peft,
        "fla": fla,
        "bitsandbytes": _bnb,
        "accelerate": importlib.import_module("accelerate"),
    }
    module_paths = {}
    for name, module in modules.items():
        path = Path(str(module.__file__)).resolve()
        if SOLAR_ENV not in path.parents:
            raise RuntimeError(f"external Solar dependency is forbidden: {name}={path}")
        module_paths[name] = str(path)
    return {
        "torch": str(torch.__version__),
        "transformers": version,
        "peft": str(peft.__version__),
        "fla": str(getattr(fla, "__version__", "unknown")),
        "bitsandbytes": importlib_metadata.version("bitsandbytes"),
        "accelerate": importlib_metadata.version("accelerate"),
        "environment": str(SOLAR_ENV),
        "module_paths": module_paths,
    }


def validate_hardware(contract: dict[str, Any]) -> dict[str, Any]:
    import torch

    required = int(contract["required_gpus"])
    if not torch.cuda.is_available() or torch.cuda.device_count() != required:
        raise RuntimeError(
            f"exactly {required} visible GPUs required, found={torch.cuda.device_count()}"
        )
    devices = []
    for index in range(required):
        props = torch.cuda.get_device_properties(index)
        capability = torch.cuda.get_device_capability(index)
        if props.total_memory < int(contract["minimum_memory_bytes_per_gpu"]):
            raise RuntimeError(f"GPU {index} memory too small")
        if capability[0] < int(contract["minimum_cuda_capability_major"]):
            raise RuntimeError(f"GPU {index} capability too old: {capability}")
        devices.append(
            {
                "index": index,
                "name": props.name,
                "total_memory": props.total_memory,
                "capability": list(capability),
            }
        )
    return {
        "cuda_version": torch.version.cuda,
        "device_count": required,
        "devices": devices,
    }


def _expect(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise RuntimeError(f"{label} mismatch: expected={expected!r} actual={actual!r}")


def _owned_hf_file(path: Path) -> Path:
    """Resolve an expected HF-cache symlink to its owned blob."""

    if not path.exists():
        raise FileNotFoundError(path)
    resolved = path.resolve(strict=True)
    b200_runtime.require_owned_path(resolved, must_exist=True)
    if not resolved.is_file():
        raise RuntimeError(f"expected regular HF asset: {resolved}")
    return resolved


def _sha256_hf_file(path: Path) -> str:
    return b200_runtime.sha256_file(_owned_hf_file(path))


def require_fresh_b200_receipt(path: Path) -> Path:
    """Reject receipt overwrite and every symlink target before publication."""

    raw = path.expanduser()
    if raw.is_symlink():
        raise RuntimeError(f"B200 receipt must not be a symlink: {raw}")
    resolved = b200_runtime.require_owned_path(raw)
    if resolved.exists() or resolved.is_symlink():
        raise FileExistsError(f"refusing to overwrite B200 receipt: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def write_fresh_b200_receipt(path: Path, value: dict[str, Any]) -> Path:
    destination = require_fresh_b200_receipt(path)
    return b200_runtime.atomic_json(destination, value)


def validate_b200_transfer_receipt(path: Path) -> dict[str, Any]:
    resolved = b200_runtime.require_owned_path(path, must_exist=True)
    _expect("transfer receipt path", resolved, B200_TRANSFER_RECEIPT)
    receipt = b200_runtime.read_json_no_duplicates(resolved)
    for key, expected in (
        ("status", "PASS"),
        ("root", str(b200_runtime.OWNED_ROOT)),
        ("revision", BASE_REVISION),
        ("mode", "full"),
        ("large_hashes_checked", True),
        ("b200_production_authorized", False),
        ("reference_sha256", b200_runtime.REFERENCE_EVALUATOR_SHA256),
    ):
        _expect(f"transfer receipt {key}", receipt.get(key), expected)
    return {
        "path": str(resolved),
        "sha256": b200_runtime.sha256_file(resolved),
        "receipt": receipt,
    }


def validate_b200_relocated_bf16(
    receipt_path: Path,
    source_dir: Path,
    transfer: dict[str, Any],
) -> dict[str, Any]:
    source = b200_runtime.require_owned_path(source_dir, must_exist=True)
    _expect("relocated BF16 source", source, b200_runtime.BF16_SOURCE_DIR)
    if source.name != BASE_REVISION or source.parent.name != "snapshots":
        raise RuntimeError("relocated BF16 source is not the pinned snapshot")
    receipt_path = b200_runtime.require_owned_path(receipt_path, must_exist=True)
    _expect("BF16 receipt path", receipt_path, B200_DOWNLOAD_RECEIPT)
    receipt_sha256 = b200_runtime.sha256_file(receipt_path)
    _expect(
        "BF16 receipt transfer binding",
        transfer["receipt"]["bf16"].get("receipt_sha256"),
        receipt_sha256,
    )
    receipt = b200_runtime.read_json_no_duplicates(receipt_path)
    for key, expected in (
        ("status", "PASS"),
        ("repo_id", BASE_MODEL_ID),
        ("requested_revision", BASE_REVISION),
        ("repo_sha", BASE_REVISION),
        ("indexed_shards", EXPECTED_SOURCE_SHARD_COUNT),
        ("indexed_tensors", EXPECTED_SOURCE_TENSOR_COUNT),
        ("local_dir_copy", False),
        ("resumable_hf_cache", True),
    ):
        _expect(f"BF16 download receipt {key}", receipt.get(key), expected)
    contract_hashes = receipt.get("contract_hashes")
    if not isinstance(contract_hashes, dict):
        raise RuntimeError("BF16 receipt contract_hashes must be an object")
    actual_hashes = {
        name: _sha256_hf_file(source / name)
        for name in ("config.json", "model.safetensors.index.json", "LICENSE")
    }
    for name, actual in actual_hashes.items():
        _expect(f"relocated BF16 {name}", actual, contract_hashes.get(name))
    config = b200_runtime.read_json_no_duplicates(
        _owned_hf_file(source / "config.json")
    )
    _expect("relocated BF16 model_type", config.get("model_type"), "solar_open2")
    if "SolarOpen2ForCausalLM" not in (config.get("architectures") or []):
        raise RuntimeError("relocated BF16 architecture changed")
    index = b200_runtime.read_json_no_duplicates(
        _owned_hf_file(source / "model.safetensors.index.json")
    )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError("relocated BF16 index weight_map is invalid")
    _expect("relocated BF16 indexed tensors", len(weight_map), 47019)
    shard_names = {
        str(value) for value in weight_map.values() if isinstance(value, str)
    }
    _expect("relocated BF16 shard count", len(shard_names), 94)
    if any(
        not (source / name).exists()
        or b200_runtime.OWNED_ROOT not in (source / name).resolve(strict=True).parents
        for name in shard_names
    ):
        raise RuntimeError("relocated BF16 index references a missing/outside shard")
    return {
        "status": "PASS",
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha256,
        "recorded_original_snapshot_dir": receipt.get("snapshot_dir"),
        "relocated_snapshot_dir": str(source),
        "relocation_verified_by_content_hash": True,
        "revision": BASE_REVISION,
        "indexed_shards": len(shard_names),
        "indexed_tensors": len(weight_map),
        "contract_hashes": actual_hashes,
    }


def validate_b200_nf4_artifact(
    path: Path,
    *,
    role: str,
    expected_path: Path,
    expected_schema: str,
    expected_payload_files: int,
    transfer: dict[str, Any],
    source_config_sha256: str,
) -> dict[str, Any]:
    directory = b200_runtime.require_owned_path(path, must_exist=True)
    _expect(f"{role} NF4 directory", directory, expected_path)
    manifest_path = directory / NF4_MANIFEST
    index_path = directory / NF4_INDEX
    manifest = b200_runtime.read_json_no_duplicates(manifest_path)
    index = b200_runtime.read_json_no_duplicates(index_path)
    for key, expected in (
        ("status", "SEALED"),
        ("schema", expected_schema),
        ("schema_version", 1),
        ("base_model_id", BASE_MODEL_ID),
        ("base_revision", BASE_REVISION),
        ("complete_model_checkpoint", False),
    ):
        _expect(f"{role} NF4 {key}", manifest.get(key), expected)
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise RuntimeError(f"{role} NF4 source identity is missing")
    _expect(
        f"{role} NF4 source config",
        source.get("config_sha256"),
        source_config_sha256,
    )
    files = manifest.get("files")
    if not isinstance(files, list):
        raise RuntimeError(f"{role} NF4 files must be a list")
    _expect(f"{role} NF4 payload file count", len(files), expected_payload_files)
    filenames = [record.get("filename") for record in files if isinstance(record, dict)]
    if (
        len(filenames) != expected_payload_files
        or len(set(filenames)) != expected_payload_files
        or any(not isinstance(name, str) or not name for name in filenames)
    ):
        raise RuntimeError(f"{role} NF4 payload filenames are invalid")
    for name in filenames:
        payload = directory / str(name)
        if payload.is_symlink() or not payload.is_file():
            raise RuntimeError(f"{role} NF4 payload is missing/symlinked: {payload}")
    manifest_sha256 = b200_runtime.sha256_file(manifest_path)
    index_sha256 = b200_runtime.sha256_file(index_path)
    transfer_entry = transfer["receipt"][f"{role}_nf4"]
    _expect(
        f"{role} NF4 manifest transfer binding",
        transfer_entry.get("manifest_sha256"),
        manifest_sha256,
    )
    _expect(
        f"{role} NF4 index transfer binding",
        transfer_entry.get("index_sha256"),
        index_sha256,
    )
    _expect(
        f"{role} NF4 internal index binding",
        manifest.get("expert_index_sha256"),
        index_sha256,
    )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError(f"{role} NF4 index weight_map is invalid")
    _expect(
        f"{role} NF4 serialized tensor count",
        len(weight_map),
        manifest.get("serialized_tensor_count"),
    )
    if set(weight_map.values()) != set(filenames):
        raise RuntimeError(f"{role} NF4 index payload coverage changed")
    return {
        "status": "PASS",
        "role": role,
        "path": str(directory),
        "schema": expected_schema,
        "manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha256,
            "status": "SEALED",
        },
        "index": {
            "path": str(index_path),
            "sha256": index_sha256,
            "serialized_tensor_count": len(weight_map),
        },
        "payload_file_count": len(filenames),
        "payload_hashes_inherited_from_full_transfer_receipt": True,
    }


def validate_b200_environment_receipt(path: Path) -> dict[str, Any]:
    resolved = b200_runtime.require_owned_path(path, must_exist=True)
    _expect("environment receipt path", resolved, B200_ENVIRONMENT_RECEIPT)
    receipt = b200_runtime.read_json_no_duplicates(resolved)
    for key, expected in (
        ("status", "PASS"),
        ("schema", "solar_open2_b200_environment"),
        ("schema_version", 1),
        ("root", str(b200_runtime.OWNED_ROOT)),
        ("user", b200_runtime.EXPECTED_USER),
        ("secrets_recorded", False),
        ("b200_production_authorized", False),
    ):
        _expect(f"environment receipt {key}", receipt.get(key), expected)
    python = receipt.get("python")
    lock = receipt.get("dependency_lock")
    torch = receipt.get("torch")
    libraries = receipt.get("native_libraries")
    if not all(isinstance(value, dict) for value in (python, lock, torch, libraries)):
        raise RuntimeError("environment receipt nested contract is invalid")
    _expect("environment sys.prefix", python.get("sys_prefix"), str(SOLAR_ENV))
    lock_path = Path(str(lock.get("path")))
    _expect(
        "environment lock path",
        lock_path,
        PROJECT / "requirements/solar-open-2.lock.txt",
    )
    _expect(
        "environment lock hash",
        lock.get("sha256"),
        b200_runtime.sha256_file(lock_path),
    )
    if "sm_100" not in (torch.get("compiled_arch_flags") or []):
        raise RuntimeError("environment receipt lacks sm_100")
    cuda128 = libraries.get("libbitsandbytes_cuda128")
    if not isinstance(cuda128, dict):
        raise RuntimeError("environment receipt lacks CUDA 12.8 bnb library")
    library_path = Path(str(cuda128.get("path")))
    _expect(
        "environment bnb library name",
        library_path.name,
        "libbitsandbytes_cuda128.so",
    )
    _expect(
        "environment bnb library hash",
        cuda128.get("sha256"),
        b200_runtime.sha256_file(library_path),
    )
    return {
        "status": "PASS",
        "path": str(resolved),
        "sha256": b200_runtime.sha256_file(resolved),
        "python": {
            "version": python.get("version"),
            "sys_prefix": python.get("sys_prefix"),
        },
        "dependency_lock": {
            "path": str(lock_path),
            "sha256": lock.get("sha256"),
        },
        "torch": {
            "version": torch.get("version"),
            "compiled_cuda": torch.get("compiled_cuda"),
            "compiled_arch_flags": torch.get("compiled_arch_flags"),
        },
        "bitsandbytes_cuda128": {
            "path": str(library_path),
            "sha256": cuda128.get("sha256"),
        },
    }


def validate_b200_data_assets(bundle: dict[str, Any]) -> dict[str, Any]:
    contract = bundle["data_contract"]
    manifest_path = Path(str(contract["prepared_manifest"]))
    manifest_sha256 = b200_runtime.sha256_file(manifest_path)
    _expect(
        "prepared data manifest",
        manifest_sha256,
        contract["prepared_manifest_sha256"],
    )
    files: dict[str, Any] = {}
    for role, entry in contract["files"].items():
        path = Path(str(entry["path"]))
        actual = b200_runtime.sha256_file(path)
        _expect(f"{role} data hash", actual, entry["sha256"])
        files[role] = {
            "path": str(path),
            "sha256": actual,
            "rows": entry["rows"],
        }
    raw_human_gold = Path(str(contract["human_gold_raw"]))
    raw_human_gold_sha256 = b200_runtime.sha256_file(raw_human_gold)
    _expect(
        "raw human-gold hash",
        raw_human_gold_sha256,
        contract["human_gold_raw_sha256"],
    )
    evaluator = b200_runtime.require_owned_evaluator()
    return {
        "status": "PASS",
        "prepared_manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha256,
        },
        "files": files,
        "raw_human_gold": {
            "path": str(raw_human_gold),
            "sha256": raw_human_gold_sha256,
        },
        "evaluator": {
            "path": str(evaluator),
            "sha256": b200_runtime.REFERENCE_EVALUATOR_SHA256,
        },
    }


def validate_b200_software() -> dict[str, Any]:
    prefix = Path(sys.prefix).resolve()
    base_prefix = Path(sys.base_prefix).resolve()
    _expect("B200 Python sys.prefix", prefix, SOLAR_ENV)
    b200_runtime.require_owned_path(base_prefix, must_exist=True)
    torch, bnb, bnb_contract = require_b200_sealed_bitsandbytes(
        manifest_status="SEALED"
    )
    modules = {
        "torch": torch,
        "bitsandbytes": bnb,
        "transformers": importlib.import_module("transformers"),
        "peft": importlib.import_module("peft"),
        "accelerate": importlib.import_module("accelerate"),
        "fla": importlib.import_module("fla"),
    }
    module_paths = {}
    for name, module in modules.items():
        path = Path(str(module.__file__)).resolve(strict=True)
        if SOLAR_ENV not in path.parents:
            raise RuntimeError(f"external B200 dependency is forbidden: {name}={path}")
        module_paths[name] = str(path)
    transformers = modules["transformers"]
    if not str(transformers.__version__).startswith("5.14.1"):
        raise RuntimeError("B200 requires the pinned Solar Transformers build")
    solar = importlib.import_module(
        "transformers.models.solar_open2.modeling_solar_open2"
    )
    if not hasattr(solar, "SolarOpen2ForCausalLM"):
        raise RuntimeError("native SolarOpen2ForCausalLM is unavailable")
    versions = {
        "torch": str(torch.__version__),
        "bitsandbytes": importlib_metadata.version("bitsandbytes"),
        "transformers": str(transformers.__version__),
        "peft": importlib_metadata.version("peft"),
        "accelerate": importlib_metadata.version("accelerate"),
        "fla-core": importlib_metadata.version("fla-core"),
    }
    return {
        "status": "PASS",
        "environment": str(SOLAR_ENV),
        "base_prefix": str(base_prefix),
        "module_paths": module_paths,
        "versions": versions,
        "bitsandbytes_runtime": bnb_contract,
    }


def run_b200_preflight(
    *,
    config_dir: Path,
    source_dir: Path,
    training_expert_dir: Path,
    inference_expert_dir: Path,
    download_receipt: Path,
    environment_receipt: Path,
    transfer_receipt: Path,
    hardware_validator: Callable[[], dict[str, Any]] | None = None,
    software_validator: Callable[[], dict[str, Any]] | None = None,
    data_validator: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    b200_runtime.require_expected_user()
    b200_runtime.require_owned_root()
    bundle = b200_runtime.load_b200_config_bundle(config_dir)
    model = bundle["model"]
    _expect(
        "B200 training expert CLI path",
        training_expert_dir.resolve(),
        b200_runtime.TRAIN_EXPERT_DIR,
    )
    _expect(
        "B200 inference expert CLI path",
        inference_expert_dir.resolve(),
        b200_runtime.INFERENCE_EXPERT_DIR,
    )
    transfer = validate_b200_transfer_receipt(transfer_receipt)
    bf16 = validate_b200_relocated_bf16(download_receipt, source_dir, transfer)
    source_config_sha256 = bf16["contract_hashes"]["config.json"]
    training_nf4 = validate_b200_nf4_artifact(
        training_expert_dir,
        role="training",
        expected_path=b200_runtime.TRAIN_EXPERT_DIR,
        expected_schema=model["training_runtime"]["expert_artifact_schema"],
        expected_payload_files=48,
        transfer=transfer,
        source_config_sha256=source_config_sha256,
    )
    inference_nf4 = validate_b200_nf4_artifact(
        inference_expert_dir,
        role="inference",
        expected_path=b200_runtime.INFERENCE_EXPERT_DIR,
        expected_schema=model["inference_runtime"]["expert_artifact_schema"],
        expected_payload_files=960,
        transfer=transfer,
        source_config_sha256=source_config_sha256,
    )
    environment = validate_b200_environment_receipt(environment_receipt)
    data = (data_validator or validate_b200_data_assets)(bundle)
    software = (software_validator or validate_b200_software)()
    hardware = (hardware_validator or b200_runtime.require_b200_gpu_runtime)()
    for key, expected in (
        ("status", "PASS"),
        ("logical_gpu_count", 2),
        ("boundary_layers", [24]),
    ):
        _expect(f"B200 hardware {key}", hardware.get(key), expected)
    physical_visibility = hardware.get("physical_cuda_visible_devices")
    _expect(
        "B200 preflight physical visibility",
        physical_visibility,
        b200_runtime.EXPECTED_CUDA_VISIBLE_DEVICES,
    )
    logical_to_physical = hardware.get("logical_to_physical")
    if not isinstance(logical_to_physical, list) or len(logical_to_physical) != 2:
        raise RuntimeError("B200 logical-to-physical evidence is invalid")
    for logical_index, device in enumerate(logical_to_physical):
        if not isinstance(device, dict):
            raise RuntimeError("B200 hardware device evidence is invalid")
        for key, expected in (
            ("logical_index", logical_index),
            ("logical_device", f"cuda:{logical_index}"),
            ("physical_index", 6 + logical_index),
            ("compute_capability", [10, 0]),
        ):
            _expect(
                f"B200 hardware device {logical_index} {key}",
                device.get(key),
                expected,
            )
        if int(device.get("total_memory_bytes", 0)) < (
            b200_runtime.MINIMUM_MEMORY_BYTES_PER_GPU
        ):
            raise RuntimeError(
                f"B200 hardware device {logical_index} has insufficient memory"
            )
    return {
        "schema": "solar_open2_b200_preflight",
        "schema_version": 1,
        "status": "PASS",
        "profile": "b200",
        "user": b200_runtime.EXPECTED_USER,
        "host": platform.node(),
        "root": str(b200_runtime.OWNED_ROOT),
        "config": b200_runtime.b200_config_identity(config_dir),
        "pipeline_code": pipeline_code_identity(config_dir=bundle["config_dir"]),
        "transfer_integrity": {
            "path": transfer["path"],
            "sha256": transfer["sha256"],
            "large_hashes_checked": True,
        },
        "bf16": bf16,
        "training_nf4": training_nf4,
        "inference_nf4": inference_nf4,
        "data": data,
        "environment": environment,
        "software": software,
        "hardware": hardware,
        "physical_cuda_visible_devices": physical_visibility,
        "production_authorized": False,
        "next_gate": "tokenizer contract then two-GPU full-loader smoke",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--expert-nf4-dir", type=Path, required=True)
    parser.add_argument("--inference-expert-nf4-dir", type=Path)
    parser.add_argument("--download-receipt", type=Path, required=True)
    parser.add_argument("--environment-receipt", type=Path)
    parser.add_argument("--transfer-integrity-receipt", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.config_dir is not None:
        missing = [
            name
            for name, value in (
                ("--inference-expert-nf4-dir", args.inference_expert_nf4_dir),
                ("--environment-receipt", args.environment_receipt),
                (
                    "--transfer-integrity-receipt",
                    args.transfer_integrity_receipt,
                ),
            )
            if value is None
        ]
        if missing:
            raise RuntimeError(
                f"B200 preflight requires explicit inputs: {', '.join(missing)}"
            )
        receipt = require_fresh_b200_receipt(args.receipt)
        result = run_b200_preflight(
            config_dir=args.config_dir,
            source_dir=args.source_dir,
            training_expert_dir=args.expert_nf4_dir,
            inference_expert_dir=args.inference_expert_nf4_dir,
            download_receipt=args.download_receipt,
            environment_receipt=args.environment_receipt,
            transfer_receipt=args.transfer_integrity_receipt,
        )
        b200_runtime.atomic_json(receipt, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if getpass.getuser() != "mk03":
        raise RuntimeError("Solar preflight is restricted to mk03")
    source_dir = require_owned_path(args.source_dir)
    expert_dir = require_owned_path(args.expert_nf4_dir)
    download_receipt = require_owned_path(args.download_receipt)
    receipt = require_owned_path(args.receipt, create_parent=True)
    if source_dir.name != BASE_REVISION or source_dir.parent.name != "snapshots":
        raise RuntimeError("source-dir is not the pinned Hugging Face snapshot")
    contract = read_json(CONFIG_DIR / "model.json")
    result = {
        "status": "PASS",
        "user": "mk03",
        "host": platform.node(),
        "project": str(PROJECT),
        "source_dir": str(source_dir),
        "expert_nf4_dir": str(expert_dir),
        "model_contract_sha256": sha256(CONFIG_DIR / "model.json"),
        "recipe_sha256": sha256(CONFIG_DIR / "recipe.json"),
        "data_contract_sha256": sha256(CONFIG_DIR / "data_contract.json"),
        "pipeline_code": pipeline_code_identity(),
        "environment_lock_sha256": sha256(
            PROJECT / "requirements/solar-open-2.lock.txt"
        ),
        "download": validate_download_receipt(download_receipt, source_dir),
        "expert_artifact": validate_expert_artifact(expert_dir),
        "software": validate_software(),
        "hardware": validate_hardware(contract["hardware"]),
        "production_authorized": False,
        "next_gate": "one-update max-length backward and base/LoRA inference",
    }
    atomic_json(receipt, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
