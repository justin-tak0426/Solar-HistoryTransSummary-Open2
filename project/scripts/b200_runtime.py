#!/usr/bin/env python3
"""Fail-closed, dependency-light runtime contracts for the B200 profile.

Configuration, path, hash, and JSON helpers in this module are safe to import
and use with CUDA hidden.  ``require_b200_gpu_runtime`` is the only helper that
imports PyTorch or queries CUDA when a caller does not inject a test double.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import pwd
import re
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


OWNED_ROOT = Path("/NHNHOME/WORKSPACE/26mfds002_B/jhtak/solar-open-2")
PROJECT_DIR = OWNED_ROOT / "project"
B200_CONFIG_DIR = PROJECT_DIR / "config/b200"
VENV_DIR = PROJECT_DIR / "solar-open-2"
DATA_DIR = OWNED_ROOT / "data"
BF16_SOURCE_DIR = (
    OWNED_ROOT
    / "runtime/hf_cache/models--upstage--Solar-Open2-250B/snapshots"
    / "8d1dda5ad2af58207c65348d825772d2f4d72481"
)
TRAIN_EXPERT_DIR = OWNED_ROOT / "runtime/models/Solar-Open2-250B-NF4-layer-flat-experts"
INFERENCE_EXPERT_DIR = (
    OWNED_ROOT / "runtime/models/Solar-Open2-250B-NF4-per-expert-fallback"
)
B200_ASSET_INTEGRITY_RECEIPT = (
    OWNED_ROOT / "runtime/receipts/b200_asset_integrity_large_v1.json"
)
B200_BF16_DOWNLOAD_RECEIPT = OWNED_ROOT / "runtime/receipts/source_bf16_download.json"
B200_ASSET_VERIFIER = PROJECT_DIR / "migration/b200/verify_assets.py"
REFERENCE_EVALUATOR = PROJECT_DIR / "scripts/reference/shared_nodict_evaluation.py"
REFERENCE_EVALUATOR_SHA256 = (
    "a6e68d29f3db569ff78283f15b10680c11c2ff9f6472453dc8fd6ef9a7320b23"
)

EXPECTED_USER = "e202"
EXPECTED_CUDA_VISIBLE_DEVICES = "6,7"
EXPECTED_PHYSICAL_GPU_INDICES = (6, 7)
EXPECTED_LOGICAL_GPU_COUNT = 2
EXPECTED_GPU_NAME_TOKEN = "B200"
EXPECTED_COMPUTE_CAPABILITY = (10, 0)
MINIMUM_MEMORY_BYTES_PER_GPU = 160 * 1024**3
REQUIRED_TORCH_ARCH = "sm_100"
REQUIRED_BITSANDBYTES_VERSION = "0.49.2"
REQUIRED_BITSANDBYTES_LIBRARY = "libbitsandbytes_cuda128.so"
BOUNDARY_LAYERS = (24,)
RUNTIME_OVERRIDE_USER_INSTRUCTION = (
    "1 epoch 학습시키는데도 그렇게 오래걸려? 일단 하드 시간 제약은 걸지말고, "
    "최대 1epoch 돌아가는거 내에서 효율적으로 학습해서 최대 효율을 뽑으면서 "
    "학습시간을 단축시켜봐"
)
RUNTIME_OVERRIDE_USER_INSTRUCTION_SHA256 = (
    "2bd9df6c227f966e66220029ad927d8021bcc5d3001791d19c2d8d2389f910ab"
)

_layer_device_map = {
    layer: ("cuda:0" if layer < BOUNDARY_LAYERS[0] else "cuda:1") for layer in range(48)
}
EXPECTED_LAYER_DEVICE_MAP: Mapping[int, str] = MappingProxyType(_layer_device_map)
EXPECTED_HF_DEVICE_MAP: Mapping[str, str] = MappingProxyType(
    {
        "model.embed_tokens": "cuda:0",
        **{
            f"model.layers.{layer}": device
            for layer, device in _layer_device_map.items()
        },
        "model.norm": "cuda:1",
        "lm_head": "cuda:1",
    }
)
EXPECTED_CHECKPOINTED_DECODER_LAYER_INDICES = tuple(
    layer for layer in range(48) if layer % 4 != 3
)
EXPECTED_UNCHECKPOINTED_DECODER_LAYER_INDICES = tuple(
    layer for layer in range(48) if layer % 4 == 3
)
CONFIG_FILENAMES = ("model.json", "recipe.json", "data_contract.json")
INHERITED_LARGE_HASH_MODE = "inherited_large_asset_receipt"
_SOURCE_SHARD_RE = re.compile(r"model-(\d{5})-of-00094\.safetensors")


class B200ContractError(RuntimeError):
    """Raised when a B200 runtime invariant is not satisfied."""


def b200_training_performance_contract() -> dict[str, Any]:
    """Return the exact sealed B200 throughput policy."""

    return {
        "schema": "solar-open2-b200-training-performance",
        "schema_version": 2,
        "activation_checkpointing": (
            "selective_device_aware_non_reentrant_subject_to_b200_smoke"
        ),
        "decoder_layer_count": 48,
        "checkpointed_decoder_layer_count": 36,
        "checkpointed_decoder_layer_indices": list(
            EXPECTED_CHECKPOINTED_DECODER_LAYER_INDICES
        ),
        "uncheckpointed_decoder_layer_count": 12,
        "uncheckpointed_decoder_layer_indices": list(
            EXPECTED_UNCHECKPOINTED_DECODER_LAYER_INDICES
        ),
        "per_device_checkpointed_layer_count": {"cuda:0": 18, "cuda:1": 18},
        "per_device_uncheckpointed_layer_count": {"cuda:0": 6, "cuda:1": 6},
        "uncheckpoint_stride": 4,
        "uncheckpoint_offset": 3,
        "logit_chunk_size": 2048,
        "hidden_environment_override_allowed": False,
    }


def b200_selective_gradient_checkpointing_contract() -> dict[str, Any]:
    """Return the exact effective checkpoint contract expected after PEFT wrap."""

    return {
        "status": "PASS",
        "implementation": "device_aware_non_reentrant_selective",
        "enabled": True,
        "use_reentrant": False,
        "preserve_rng_state": True,
        "determinism_check": "default",
        "early_stop": True,
        "decoder_layer_count": 48,
        "checkpointed_decoder_layer_count": 36,
        "uncheckpointed_decoder_layer_count": 12,
        "uncheckpoint_stride": 4,
        "uncheckpoint_offset": 3,
        "checkpointed_decoder_layers": list(
            EXPECTED_CHECKPOINTED_DECODER_LAYER_INDICES
        ),
        "uncheckpointed_decoder_layers": list(
            EXPECTED_UNCHECKPOINTED_DECODER_LAYER_INDICES
        ),
        "boundary_layers": [24],
        "layer_input_devices": {
            str(layer): device for layer, device in EXPECTED_LAYER_DEVICE_MAP.items()
        },
        "qualification": "longest_row_and_all_bucket_backward_smoke_required",
    }


def b200_full_gradient_checkpointing_contract() -> dict[str, Any]:
    """Return the exact loader-time all-layer checkpoint contract."""

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
            str(layer): device for layer, device in EXPECTED_LAYER_DEVICE_MAP.items()
        },
    }


def _absolute_candidate(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate


def require_owned_root(
    root: str | os.PathLike[str] = OWNED_ROOT,
) -> Path:
    """Require the one exact, existing project root from the execution contract."""

    candidate = _absolute_candidate(root)
    resolved = candidate.resolve(strict=True)
    if resolved != OWNED_ROOT:
        raise B200ContractError(
            f"owned root mismatch: expected={OWNED_ROOT} actual={resolved}"
        )
    if candidate.is_symlink():
        raise B200ContractError(f"owned root must not be a symlink: {candidate}")
    return resolved


def require_owned_path(
    path: str | os.PathLike[str],
    *,
    root: str | os.PathLike[str] = OWNED_ROOT,
    must_exist: bool = False,
) -> Path:
    """Resolve a path and reject any target outside the exact owned root."""

    owned_root = require_owned_root(root)
    candidate = _absolute_candidate(path)
    if candidate.is_symlink():
        raise B200ContractError(f"owned path must not be a symlink: {candidate}")
    resolved = candidate.resolve(strict=must_exist)
    if resolved != owned_root and owned_root not in resolved.parents:
        raise B200ContractError(f"path is outside owned root: {resolved}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def require_expected_user(observed_user: str | None = None) -> str:
    """Verify the effective OS user, or an explicitly supplied observation."""

    actual = (
        pwd.getpwuid(os.geteuid()).pw_name
        if observed_user is None
        else str(observed_user)
    )
    if actual != EXPECTED_USER:
        raise B200ContractError(
            f"user mismatch: expected={EXPECTED_USER} actual={actual}"
        )
    return actual


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return the SHA-256 of an owned regular file."""

    resolved = require_owned_path(path, must_exist=True)
    if not resolved.is_file():
        raise B200ContractError(f"expected regular file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file_sha256(
    path: str | os.PathLike[str],
    expected_sha256: str,
) -> Path:
    """Require a lower-case SHA-256 identity for an owned regular file."""

    expected = str(expected_sha256)
    if (
        len(expected) != 64
        or expected != expected.lower()
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise B200ContractError(f"invalid expected SHA-256: {expected!r}")
    resolved = require_owned_path(path, must_exist=True)
    actual = sha256_file(resolved)
    if actual != expected:
        raise B200ContractError(
            f"file hash mismatch: path={resolved} expected={expected} "
            f"actual={actual}"
        )
    return resolved


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise B200ContractError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def read_json_no_duplicates(
    path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Read an owned JSON object while rejecting duplicate object keys."""

    resolved = require_owned_path(path, must_exist=True)
    try:
        value = json.loads(
            resolved.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise B200ContractError(f"invalid JSON: {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise B200ContractError(f"expected JSON object: {resolved}")
    return value


def _require_sha256(label: str, value: Any) -> str:
    digest = str(value)
    if (
        len(digest) != 64
        or digest != digest.lower()
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise B200ContractError(f"{label} is not a lower-case SHA-256: {value!r}")
    return digest


def _require_fixed_regular_file(
    path: str | os.PathLike[str],
    expected: Path,
    label: str,
) -> Path:
    candidate = _absolute_candidate(path)
    resolved = require_owned_path(candidate, must_exist=True)
    if resolved != expected or candidate.is_symlink() or not resolved.is_file():
        raise B200ContractError(
            f"{label} must be the fixed regular non-symlink file: "
            f"expected={expected} actual={resolved}"
        )
    return resolved


def _validate_b200_source_download_receipt(
    *,
    receipt_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = _require_fixed_regular_file(
        B200_BF16_DOWNLOAD_RECEIPT,
        B200_BF16_DOWNLOAD_RECEIPT,
        "B200 BF16 download receipt",
    )
    _expect(
        "B200 BF16 download receipt SHA-256",
        sha256_file(path),
        _require_sha256("asset BF16 receipt SHA-256", receipt_sha256),
    )
    receipt = read_json_no_duplicates(path)
    revision = BF16_SOURCE_DIR.name
    for key, expected in (
        ("status", "PASS"),
        ("repo_id", "upstage/Solar-Open2-250B"),
        ("requested_revision", revision),
        ("repo_sha", revision),
        ("indexed_shards", 94),
        ("indexed_tensors", 47_019),
        ("local_dir_copy", False),
        ("resumable_hf_cache", True),
    ):
        _expect(f"B200 BF16 download receipt {key}", receipt.get(key), expected)

    contract_hashes = receipt.get("contract_hashes")
    expected_contract_files = {
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "LICENSE",
    }
    if not isinstance(contract_hashes, dict) or set(contract_hashes) != (
        expected_contract_files
    ):
        raise B200ContractError("B200 BF16 contract file set changed")
    for filename, digest_value in contract_hashes.items():
        if Path(filename).name != filename:
            raise B200ContractError(f"unsafe B200 BF16 contract filename: {filename!r}")
        candidate = BF16_SOURCE_DIR / filename
        if not candidate.is_file():
            raise B200ContractError(f"missing B200 BF16 contract file: {candidate}")
        target = candidate.resolve(strict=True)
        if OWNED_ROOT not in target.parents:
            raise B200ContractError(
                f"B200 BF16 contract file escaped owned root: {candidate} -> {target}"
            )
        _expect(
            f"B200 BF16 contract file {filename}",
            sha256_file(target),
            _require_sha256(f"B200 BF16 contract file {filename}", digest_value),
        )

    raw_shards = receipt.get("source_shards")
    if not isinstance(raw_shards, list) or len(raw_shards) != 94:
        raise B200ContractError("B200 BF16 receipt must list exactly 94 shards")
    expected_fields = {
        "filename",
        "file_bytes",
        "payload_bytes",
        "tensor_count",
        "sha256",
    }
    source_shards: list[dict[str, Any]] = []
    observed_numbers: set[int] = set()
    for raw_record in raw_shards:
        if not isinstance(raw_record, dict) or set(raw_record) != expected_fields:
            raise B200ContractError("B200 BF16 shard receipt schema changed")
        filename = raw_record.get("filename")
        match = (
            _SOURCE_SHARD_RE.fullmatch(filename) if isinstance(filename, str) else None
        )
        if match is None:
            raise B200ContractError(f"unsafe B200 BF16 shard name: {filename!r}")
        number = int(match.group(1))
        if number in observed_numbers:
            raise B200ContractError(f"duplicate B200 BF16 shard number: {number}")
        observed_numbers.add(number)
        file_bytes = raw_record.get("file_bytes")
        payload_bytes = raw_record.get("payload_bytes")
        tensor_count = raw_record.get("tensor_count")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (file_bytes, payload_bytes, tensor_count)
        ):
            raise B200ContractError(f"invalid B200 BF16 shard accounting: {filename}")
        digest = _require_sha256(
            f"B200 BF16 shard {filename}",
            raw_record.get("sha256"),
        )
        candidate = BF16_SOURCE_DIR / filename
        if not candidate.is_file():
            raise B200ContractError(f"missing B200 BF16 shard: {candidate}")
        target = candidate.resolve(strict=True)
        if OWNED_ROOT not in target.parents or not target.is_file():
            raise B200ContractError(
                f"B200 BF16 shard escaped owned root: {candidate} -> {target}"
            )
        _expect(
            f"B200 BF16 shard size {filename}",
            candidate.stat().st_size,
            file_bytes,
        )
        source_shards.append(
            {
                "filename": filename,
                "file_bytes": file_bytes,
                "payload_bytes": payload_bytes,
                "tensor_count": tensor_count,
                "sha256": digest,
            }
        )
    if observed_numbers != set(range(1, 95)):
        raise B200ContractError("B200 BF16 shard sequence changed")
    source_shards.sort(key=lambda record: str(record["filename"]))
    return receipt, source_shards


def _validate_b200_nf4_asset(
    asset_receipt: Mapping[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    contracts = {
        "training": {
            "asset_key": "training_nf4",
            "directory": TRAIN_EXPERT_DIR,
            "schema": "solar-open2-layer-flat-routed-experts-nf4",
            "files": 48,
        },
        "inference": {
            "asset_key": "inference_nf4",
            "directory": INFERENCE_EXPERT_DIR,
            "schema": "solar-open2-routed-experts-nf4",
            "files": 960,
        },
    }
    if role not in contracts:
        raise B200ContractError(f"unknown B200 asset role: {role!r}")
    contract = contracts[role]
    expected_directory = Path(str(contract["directory"]))
    directory = require_owned_path(expected_directory, must_exist=True)
    if directory != expected_directory or not directory.is_dir():
        raise B200ContractError(f"B200 {role} NF4 directory changed: {directory}")
    manifest_path = directory / "expert_nf4_manifest.json"
    index_path = directory / "expert_nf4.safetensors.index.json"
    for candidate, label in ((manifest_path, "manifest"), (index_path, "index")):
        if candidate.is_symlink() or not candidate.is_file():
            raise B200ContractError(
                f"B200 {role} NF4 {label} is missing/symlinked: {candidate}"
            )
    manifest_sha256 = sha256_file(manifest_path)
    index_sha256 = sha256_file(index_path)
    manifest = read_json_no_duplicates(manifest_path)
    for key, expected in (
        ("status", "SEALED"),
        ("schema", contract["schema"]),
        ("schema_version", 1),
        ("base_revision", BF16_SOURCE_DIR.name),
        ("complete_model_checkpoint", False),
    ):
        _expect(f"B200 {role} NF4 manifest {key}", manifest.get(key), expected)
    _expect(
        f"B200 {role} NF4 internal index SHA-256",
        manifest.get("expert_index_sha256"),
        index_sha256,
    )
    records = manifest.get("files")
    if not isinstance(records, list) or len(records) != contract["files"]:
        raise B200ContractError(
            f"B200 {role} NF4 manifest file count changed: "
            f"{len(records) if isinstance(records, list) else records!r}"
        )
    payload_names: set[str] = set()
    payload_file_bytes = 0
    for record in records:
        if not isinstance(record, dict):
            raise B200ContractError(f"B200 {role} NF4 file record is invalid")
        filename = record.get("filename")
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
            or filename in payload_names
        ):
            raise B200ContractError(
                f"unsafe/duplicate B200 {role} NF4 filename: {filename!r}"
            )
        payload_names.add(filename)
        file_bytes = record.get("file_bytes")
        if (
            isinstance(file_bytes, bool)
            or not isinstance(file_bytes, int)
            or file_bytes <= 0
        ):
            raise B200ContractError(f"invalid B200 {role} NF4 file size: {filename}")
        _require_sha256(
            f"B200 {role} NF4 shard {filename}",
            record.get("sha256"),
        )
        payload = directory / filename
        if payload.is_symlink() or not payload.is_file():
            raise B200ContractError(
                f"B200 {role} NF4 payload is missing/symlinked: {payload}"
            )
        _expect(
            f"B200 {role} NF4 payload size {filename}",
            payload.stat().st_size,
            file_bytes,
        )
        payload_file_bytes += file_bytes
    expected_names = payload_names | {manifest_path.name, index_path.name}
    entries = list(directory.iterdir())
    observed_names = {entry.name for entry in entries}
    if observed_names != expected_names or any(
        entry.is_symlink() or not entry.is_file() for entry in entries
    ):
        raise B200ContractError(f"B200 {role} NF4 asset tree changed")

    asset_entry = asset_receipt.get(str(contract["asset_key"]))
    if not isinstance(asset_entry, Mapping):
        raise B200ContractError(f"B200 asset receipt lacks {contract['asset_key']}")
    for key, expected in (
        ("status", "PASS"),
        ("large_hashes_checked", True),
        ("files", contract["files"]),
        ("payload_file_bytes", payload_file_bytes),
        ("manifest_sha256", manifest_sha256),
        ("index_sha256", index_sha256),
        (
            "tree_inventory",
            {
                "directories": 0,
                "regular_files": int(contract["files"]) + 2,
                "symbolic_links": 0,
            },
        ),
    ):
        _expect(f"B200 asset receipt {role} {key}", asset_entry.get(key), expected)
    return {
        "path": str(directory),
        "manifest": {"path": str(manifest_path), "sha256": manifest_sha256},
        "index": {"path": str(index_path), "sha256": index_sha256},
        "payload_file_count": contract["files"],
        "payload_file_bytes": payload_file_bytes,
    }


def validate_b200_large_asset_receipt(
    path: str | os.PathLike[str],
    *,
    role: str,
) -> dict[str, Any]:
    """Bind one-time large-payload hashes without rereading payload bytes."""

    receipt_path = _require_fixed_regular_file(
        path,
        B200_ASSET_INTEGRITY_RECEIPT,
        "B200 large-hash asset receipt",
    )
    receipt = read_json_no_duplicates(receipt_path)
    for key, expected in (
        ("schema", "solar_open2_b200_asset_integrity"),
        ("schema_version", 1),
        ("status", "PASS"),
        ("scope", "transfer_integrity_not_b200_production_authorization"),
        ("mode", "full"),
        ("root", str(OWNED_ROOT)),
        ("revision", BF16_SOURCE_DIR.name),
        ("reference_sha256", REFERENCE_EVALUATOR_SHA256),
        ("large_hashes_checked", True),
        ("b200_production_authorized", False),
        ("user", EXPECTED_USER),
        ("receipt_path", str(B200_ASSET_INTEGRITY_RECEIPT)),
    ):
        _expect(f"B200 asset receipt {key}", receipt.get(key), expected)
    verifier = receipt.get("verifier")
    if not isinstance(verifier, Mapping):
        raise B200ContractError("B200 asset receipt lacks verifier identity")
    _expect("B200 asset verifier path", verifier.get("path"), str(B200_ASSET_VERIFIER))
    _expect(
        "B200 asset verifier SHA-256",
        verifier.get("sha256"),
        sha256_file(B200_ASSET_VERIFIER),
    )
    bf16 = receipt.get("bf16")
    if not isinstance(bf16, Mapping):
        raise B200ContractError("B200 asset receipt lacks BF16 evidence")
    for key, expected in (("status", "PASS"), ("source_shards", 94)):
        _expect(f"B200 asset receipt BF16 {key}", bf16.get(key), expected)
    _download, source_shards = _validate_b200_source_download_receipt(
        receipt_sha256=str(bf16.get("receipt_sha256")),
    )
    source_shard_bytes = sum(int(record["file_bytes"]) for record in source_shards)
    _expect(
        "B200 asset receipt BF16 source bytes",
        bf16.get("source_shard_bytes"),
        source_shard_bytes,
    )
    _expect(
        "B200 asset receipt BF16 tree inventory",
        bf16.get("tree_inventory"),
        {"directories": 3, "regular_files": 102, "symbolic_links": 102},
    )
    expert = _validate_b200_nf4_asset(receipt, role=role)
    return {
        "status": "PASS",
        "role": role,
        "hash_verification_mode": INHERITED_LARGE_HASH_MODE,
        "asset_integrity_receipt": {
            "path": str(receipt_path),
            "sha256": sha256_file(receipt_path),
        },
        "source_download_receipt": {
            "path": str(B200_BF16_DOWNLOAD_RECEIPT),
            "sha256": sha256_file(B200_BF16_DOWNLOAD_RECEIPT),
        },
        "source": {
            "path": str(BF16_SOURCE_DIR),
            "revision": BF16_SOURCE_DIR.name,
            "shard_count": len(source_shards),
            "shard_file_bytes": source_shard_bytes,
            "shards": source_shards,
        },
        "expert": expert,
        "payload_rehash_performed": False,
    }


def inherit_b200_source_shard_digests(
    inventory_summary: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind current header/size inventory to the sealed 94-shard digest ledger."""

    if (
        evidence.get("status") != "PASS"
        or evidence.get("hash_verification_mode") != INHERITED_LARGE_HASH_MODE
        or evidence.get("payload_rehash_performed") is not False
    ):
        raise B200ContractError("invalid B200 inherited large-hash evidence")
    source = evidence.get("source")
    if not isinstance(source, Mapping):
        raise B200ContractError("B200 inherited evidence lacks source metadata")
    current_shards = inventory_summary.get("shards")
    sealed_shards = source.get("shards")
    if not isinstance(current_shards, list) or not isinstance(sealed_shards, list):
        raise B200ContractError("B200 source shard inventories are invalid")
    if len(current_shards) != 94 or len(sealed_shards) != 94:
        raise B200ContractError("B200 source shard inventory count changed")
    hydrated_shards: list[dict[str, Any]] = []
    for current, sealed in zip(current_shards, sealed_shards, strict=True):
        if not isinstance(current, Mapping) or not isinstance(sealed, Mapping):
            raise B200ContractError("B200 source shard record is invalid")
        for key in ("filename", "file_bytes", "payload_bytes", "tensor_count"):
            _expect(
                f"B200 current/sealed source shard {key}",
                current.get(key),
                sealed.get(key),
            )
        if current.get("sha256") is not None:
            raise B200ContractError(
                "B200 inherited mode unexpectedly recomputed a source shard hash"
            )
        hydrated_shards.append(
            {
                "filename": current["filename"],
                "file_bytes": current["file_bytes"],
                "payload_bytes": current["payload_bytes"],
                "tensor_count": current["tensor_count"],
                "sha256": _require_sha256(
                    f"B200 inherited source shard {current['filename']}",
                    sealed.get("sha256"),
                ),
            }
        )
    _expect(
        "B200 source shard byte accounting",
        inventory_summary.get("shard_file_bytes"),
        source.get("shard_file_bytes"),
    )
    hydrated = dict(inventory_summary)
    hydrated["shards"] = hydrated_shards
    hydrated["shard_hash_provenance"] = INHERITED_LARGE_HASH_MODE
    hydrated["payload_rehash_performed"] = False
    return hydrated


def atomic_json(
    path: str | os.PathLike[str],
    value: Any,
    *,
    root: str | os.PathLike[str] = OWNED_ROOT,
) -> Path:
    """Durably replace an owned JSON file without exposing a partial receipt."""

    destination = require_owned_path(path, root=root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    payload = (
        json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def resolve_b200_config_dir(
    config_dir: str | os.PathLike[str],
) -> Path:
    """Resolve an explicitly supplied config directory to the sealed profile."""

    resolved = require_owned_path(config_dir, must_exist=True)
    if resolved != B200_CONFIG_DIR:
        raise B200ContractError(
            f"B200 config directory mismatch: expected={B200_CONFIG_DIR} "
            f"actual={resolved}"
        )
    if not resolved.is_dir():
        raise B200ContractError(f"config path is not a directory: {resolved}")
    return resolved


def resolve_b200_config_paths(
    config_dir: str | os.PathLike[str],
) -> dict[str, Path]:
    """Return the three required absolute config paths."""

    directory = resolve_b200_config_dir(config_dir)
    paths = {
        filename.removesuffix(".json"): require_owned_path(
            directory / filename, must_exist=True
        )
        for filename in CONFIG_FILENAMES
    }
    for path in paths.values():
        if not path.is_file():
            raise B200ContractError(f"config is not a regular file: {path}")
    return paths


def _expect(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise B200ContractError(
            f"{label} mismatch: expected={expected!r} actual={actual!r}"
        )


def _validate_model_config(model: Mapping[str, Any]) -> None:
    _expect("model profile", model.get("profile"), "b200")
    architecture = model.get("architecture_contract")
    hardware = model.get("hardware")
    if not isinstance(architecture, Mapping):
        raise B200ContractError("model architecture_contract must be an object")
    if not isinstance(hardware, Mapping):
        raise B200ContractError("model hardware must be an object")
    _expect("model hidden layers", architecture.get("num_hidden_layers"), 48)
    _expect("model required GPUs", hardware.get("required_gpus"), 2)
    _expect(
        "model physical visibility",
        hardware.get("physical_cuda_visible_devices"),
        EXPECTED_CUDA_VISIBLE_DEVICES,
    )
    _expect(
        "model logical devices",
        hardware.get("logical_devices"),
        ["cuda:0", "cuda:1"],
    )
    _expect(
        "model boundary layers",
        hardware.get("boundary_layers"),
        list(BOUNDARY_LAYERS),
    )
    _expect(
        "model minimum GPU memory",
        hardware.get("minimum_memory_bytes_per_gpu"),
        MINIMUM_MEMORY_BYTES_PER_GPU,
    )
    _expect(
        "model compute capability",
        hardware.get("required_compute_capability"),
        {"major": 10, "minor": 0},
    )
    expected_layers = {
        str(layer): device for layer, device in EXPECTED_LAYER_DEVICE_MAP.items()
    }
    _expect(
        "model layer device map",
        hardware.get("layer_device_map"),
        expected_layers,
    )
    _expect(
        "model embedding device",
        hardware.get("embedding_device"),
        "cuda:0",
    )
    _expect("model final norm device", hardware.get("final_norm_device"), "cuda:1")
    _expect("model lm head device", hardware.get("lm_head_device"), "cuda:1")
    _expect(
        "CPU or disk offload policy",
        hardware.get("cpu_or_disk_offload_allowed"),
        False,
    )
    _expect("DDP policy", hardware.get("ddp_allowed"), False)
    training_runtime = model.get("training_runtime")
    inference_runtime = model.get("inference_runtime")
    software = model.get("software")
    if not isinstance(training_runtime, Mapping):
        raise B200ContractError("model training_runtime must be an object")
    if not isinstance(inference_runtime, Mapping):
        raise B200ContractError("model inference_runtime must be an object")
    if not isinstance(software, Mapping):
        raise B200ContractError("model software must be an object")
    _expect(
        "training BF16 source directory",
        training_runtime.get("source_dir"),
        str(BF16_SOURCE_DIR),
    )
    _expect(
        "inference BF16 source directory",
        inference_runtime.get("source_dir"),
        str(BF16_SOURCE_DIR),
    )
    _expect(
        "training expert directory",
        training_runtime.get("expert_dir"),
        str(TRAIN_EXPERT_DIR),
    )
    _expect(
        "inference expert directory",
        inference_runtime.get("expert_dir"),
        str(INFERENCE_EXPERT_DIR),
    )
    _expect(
        "Python environment path",
        software.get("environment_path"),
        str(VENV_DIR),
    )
    for label, path in (
        ("training BF16 source", BF16_SOURCE_DIR),
        ("training expert artifact", TRAIN_EXPERT_DIR),
        ("inference expert artifact", INFERENCE_EXPERT_DIR),
        ("Python environment", VENV_DIR),
    ):
        resolved = require_owned_path(path, must_exist=True)
        if not resolved.is_dir():
            raise B200ContractError(f"{label} is not a directory: {resolved}")


def _validate_recipe_config(recipe: Mapping[str, Any]) -> None:
    _expect("recipe profile", recipe.get("profile"), "b200")
    _expect("recipe epochs", recipe.get("epochs"), 1)
    _expect("recipe optimizer steps", recipe.get("optimizer_steps"), 9316)
    _expect("recipe effective batch", recipe.get("effective_batch_size"), 16)
    _expect("recipe max length", recipe.get("max_length"), 9600)
    performance = recipe.get("training_performance")
    if not isinstance(performance, Mapping):
        raise B200ContractError("recipe training_performance must be an object")
    _expect(
        "training performance",
        dict(performance),
        b200_training_performance_contract(),
    )
    runtime_gate = recipe.get("runtime_gate")
    if not isinstance(runtime_gate, Mapping):
        raise B200ContractError("recipe runtime_gate must be an object")
    for key, expected in (
        ("projection_required", True),
        ("safety_factor", 1.25),
        ("recommended_hours", 144.0),
        ("former_absolute_limit_hours", 168.0),
        ("hard_limit_enforced", False),
        ("override_scope", "runtime_projection_only_exact_one_epoch"),
        (
            "user_instruction_sha256",
            RUNTIME_OVERRIDE_USER_INSTRUCTION_SHA256,
        ),
        ("non_runtime_gates_overridden", False),
    ):
        _expect(f"runtime gate {key}", runtime_gate.get(key), expected)
    deck = recipe.get("training_deck")
    validation = recipe.get("validation")
    if not isinstance(deck, Mapping):
        raise B200ContractError("recipe training_deck must be an object")
    if not isinstance(validation, Mapping):
        raise B200ContractError("recipe validation must be an object")
    _expect("base unique rows", deck.get("base_unique_rows"), 142713)
    _expect("expert unique rows", deck.get("expert_unique_rows"), 905)
    _expect(
        "expert occurrences",
        deck.get("expert_occurrences_per_row_per_epoch"),
        8,
    )
    occurrences = deck.get("occurrences_per_epoch")
    if not isinstance(occurrences, Mapping):
        raise B200ContractError("recipe occurrences_per_epoch must be an object")
    _expect("augmented occurrences", occurrences.get("total"), 149048)
    exact_tail = recipe.get("exact_tail")
    if not isinstance(exact_tail, Mapping):
        raise B200ContractError("recipe exact_tail must be an object")
    for key, expected in (
        ("total_occurrences", 149048),
        ("nominal_step_occurrences", 16),
        ("full_steps", 9315),
        ("full_step_occurrences", 149040),
        ("final_step", 9316),
        ("final_step_occurrences", 8),
    ):
        _expect(f"exact tail {key}", exact_tail.get(key), expected)
    _expect(
        "exact tail padding/duplication",
        exact_tail.get("padding_or_duplication_allowed"),
        False,
    )
    _expect(
        "full validation steps",
        validation.get("full_validation_steps"),
        [4658, 9316],
    )
    _expect(
        "validation selection weights",
        validation.get("selection_weights"),
        {"translation": 0.65, "summarization": 0.35},
    )
    early_stopping = validation.get("early_stopping")
    if not isinstance(early_stopping, Mapping):
        raise B200ContractError("early_stopping must be an object")
    _expect("quality early stopping", early_stopping.get("enabled"), False)
    monitors = validation.get("monitor_datasets")
    _expect("validation monitor datasets", monitors, ["validation"])
    forbidden = {"test", "full_test", "human_gold"}
    if any(str(value) in forbidden for value in monitors):
        raise B200ContractError("sealed evaluation leaked into monitoring")
    prompt = recipe.get("prompt")
    if not isinstance(prompt, Mapping):
        raise B200ContractError("recipe prompt must be an object")
    _expect("prompt evaluator path", prompt.get("source"), str(REFERENCE_EVALUATOR))
    _expect(
        "prompt evaluator SHA-256",
        prompt.get("reference_sha256"),
        REFERENCE_EVALUATOR_SHA256,
    )


def b200_runtime_override_evidence() -> dict[str, Any]:
    """Return the exact, scoped user override bound into production approval."""

    observed = hashlib.sha256(
        RUNTIME_OVERRIDE_USER_INSTRUCTION.encode("utf-8")
    ).hexdigest()
    if observed != RUNTIME_OVERRIDE_USER_INSTRUCTION_SHA256:
        raise B200ContractError("runtime override user-instruction hash changed")
    return {
        "schema": "solar-open2-b200-runtime-limit-override",
        "schema_version": 1,
        "status": "RECORDED",
        "scope": "runtime_projection_only_exact_one_epoch",
        "decision": "PROCEED_WITHOUT_RUNTIME_HARD_LIMIT",
        "maximum_epochs": 1,
        "former_absolute_limit_hours": 168.0,
        "user_instruction": RUNTIME_OVERRIDE_USER_INSTRUCTION,
        "user_instruction_sha256": RUNTIME_OVERRIDE_USER_INSTRUCTION_SHA256,
        "non_runtime_gates_overridden": False,
    }


def _validate_data_config(data: Mapping[str, Any]) -> None:
    _expect("data profile", data.get("profile"), "b200")
    expected_data_dir = str(OWNED_ROOT / "data")
    _expect("prepared data directory", data.get("prepared_data_dir"), expected_data_dir)
    _expect(
        "prepared manifest",
        data.get("prepared_manifest"),
        str(OWNED_ROOT / "data/manifest.json"),
    )
    _expect(
        "owned evaluator",
        data.get("reference_evaluator"),
        str(REFERENCE_EVALUATOR),
    )
    _expect(
        "owned evaluator SHA-256",
        data.get("reference_evaluator_sha256"),
        REFERENCE_EVALUATOR_SHA256,
    )
    _expect(
        "raw human-gold path",
        data.get("human_gold_raw"),
        str(DATA_DIR / "raw/human_gold/test-human-infer-vocab-final.jsonl"),
    )
    _expect(
        "raw human-gold SHA-256",
        data.get("human_gold_raw_sha256"),
        "996a094e9abca028bf89a647a86645715b9a6b05331f85dead4f3f5197eb531b",
    )
    require_owned_path(str(data.get("human_gold_raw")), must_exist=True)
    files = data.get("files")
    if not isinstance(files, Mapping):
        raise B200ContractError("data files must be an object")
    expected_files = {
        "train": ("train.jsonl", 142713),
        "validation": ("validation.jsonl", 17850),
        "full_test": ("full_test_nodict.jsonl", 18557),
        "human_gold": ("human_gold_nodict.jsonl", 1221),
        "combined_eval": ("combined_test_and_gold_nodict.jsonl", 19778),
    }
    for key, (name, rows) in expected_files.items():
        entry = files.get(key)
        if not isinstance(entry, Mapping):
            raise B200ContractError(f"data file entry must be an object: {key}")
        _expect(f"{key} filename", entry.get("name"), name)
        _expect(f"{key} rows", entry.get("rows"), rows)
        _expect(f"{key} path", entry.get("path"), str(DATA_DIR / name))
        require_owned_path(str(entry.get("path")), must_exist=True)
    _expect(
        "selection datasets",
        data.get("selection_datasets"),
        ["validation"],
    )
    _expect(
        "sealed evaluation datasets",
        data.get("sealed_evaluation_datasets"),
        ["full_test", "human_gold"],
    )


def load_b200_config_bundle(
    config_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Load and validate the explicit B200 config profile."""

    paths = resolve_b200_config_paths(config_dir)
    model = read_json_no_duplicates(paths["model"])
    recipe = read_json_no_duplicates(paths["recipe"])
    data = read_json_no_duplicates(paths["data_contract"])
    _validate_model_config(model)
    _validate_recipe_config(recipe)
    _validate_data_config(data)
    hashes = {key: sha256_file(path) for key, path in paths.items()}
    return {
        "config_dir": resolve_b200_config_dir(config_dir),
        "paths": paths,
        "hashes": hashes,
        "model": model,
        "recipe": recipe,
        "data_contract": data,
    }


def b200_config_identity(
    config_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Return a JSON-serializable identity for a validated config bundle."""

    bundle = load_b200_config_bundle(config_dir)
    return {
        "config_dir": str(bundle["config_dir"]),
        "files": {
            key: {
                "path": str(bundle["paths"][key]),
                "sha256": bundle["hashes"][key],
            }
            for key in ("model", "recipe", "data_contract")
        },
    }


def require_owned_evaluator(
    path: str | os.PathLike[str] = REFERENCE_EVALUATOR,
) -> Path:
    """Require the single owned evaluator copy and its sealed identity."""

    resolved = require_owned_path(path, must_exist=True)
    if resolved != REFERENCE_EVALUATOR:
        raise B200ContractError(
            f"evaluator path mismatch: expected={REFERENCE_EVALUATOR} "
            f"actual={resolved}"
        )
    return require_file_sha256(resolved, REFERENCE_EVALUATOR_SHA256)


def require_physical_visibility(
    environ: Mapping[str, str] | None = None,
) -> str:
    """Require the exact physical-to-logical visibility string."""

    environment = os.environ if environ is None else environ
    actual = environment.get("CUDA_VISIBLE_DEVICES")
    if actual != EXPECTED_CUDA_VISIBLE_DEVICES:
        raise B200ContractError(
            "CUDA_VISIBLE_DEVICES mismatch: "
            f"expected={EXPECTED_CUDA_VISIBLE_DEVICES!r} actual={actual!r}"
        )
    return actual


def require_sealed_nf4_runtime(
    *,
    capability: tuple[int, int],
    manifest_status: str,
    bitsandbytes_version: str,
    native_library: str | os.PathLike[str],
) -> dict[str, Any]:
    """Allow B200 consumption only for the sealed, pinned NF4 runtime."""

    _expect("runtime compute capability", tuple(capability), (10, 0))
    _expect("NF4 manifest status", manifest_status, "SEALED")
    _expect(
        "bitsandbytes version",
        bitsandbytes_version,
        REQUIRED_BITSANDBYTES_VERSION,
    )
    library_name = Path(native_library).name
    _expect(
        "bitsandbytes native library",
        library_name,
        REQUIRED_BITSANDBYTES_LIBRARY,
    )
    return {
        "status": "PASS",
        "compute_capability": [10, 0],
        "manifest_status": "SEALED",
        "bitsandbytes_version": REQUIRED_BITSANDBYTES_VERSION,
        "native_library": library_name,
    }


def require_b200_gpu_runtime(
    torch_module: Any | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Strictly qualify the two visible B200 devices.

    This is intentionally separate from all CPU-safe helpers.  It checks the
    physical visibility string before importing PyTorch, then verifies logical
    device count, identity, compute capability, memory, and binary architecture.
    """

    visibility = require_physical_visibility(environ)
    torch = importlib.import_module("torch") if torch_module is None else torch_module
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not bool(cuda.is_available()):
        raise B200ContractError("CUDA runtime is unavailable")
    count = int(cuda.device_count())
    if count != EXPECTED_LOGICAL_GPU_COUNT:
        raise B200ContractError(
            f"logical GPU count mismatch: expected=2 actual={count}"
        )
    arch_list = list(cuda.get_arch_list())
    if REQUIRED_TORCH_ARCH not in arch_list:
        raise B200ContractError(
            f"PyTorch binary lacks {REQUIRED_TORCH_ARCH}: {arch_list}"
        )
    devices: list[dict[str, Any]] = []
    for logical_index, physical_index in enumerate(EXPECTED_PHYSICAL_GPU_INDICES):
        properties = cuda.get_device_properties(logical_index)
        name = str(properties.name)
        capability = tuple(cuda.get_device_capability(logical_index))
        total_memory = int(properties.total_memory)
        if EXPECTED_GPU_NAME_TOKEN not in name.upper():
            raise B200ContractError(
                f"logical cuda:{logical_index} is not B200: {name!r}"
            )
        if capability != EXPECTED_COMPUTE_CAPABILITY:
            raise B200ContractError(
                f"logical cuda:{logical_index} capability mismatch: "
                f"expected={EXPECTED_COMPUTE_CAPABILITY} actual={capability}"
            )
        if total_memory < MINIMUM_MEMORY_BYTES_PER_GPU:
            raise B200ContractError(
                f"logical cuda:{logical_index} has insufficient memory: "
                f"required={MINIMUM_MEMORY_BYTES_PER_GPU} actual={total_memory}"
            )
        devices.append(
            {
                "logical_index": logical_index,
                "logical_device": f"cuda:{logical_index}",
                "physical_index": physical_index,
                "name": name,
                "compute_capability": list(capability),
                "total_memory_bytes": total_memory,
            }
        )
    return {
        "status": "PASS",
        "physical_cuda_visible_devices": visibility,
        "logical_gpu_count": count,
        "torch_arch_list": arch_list,
        "logical_to_physical": devices,
        "boundary_layers": list(BOUNDARY_LAYERS),
    }
