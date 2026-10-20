#!/usr/bin/env python3
"""Fail-closed accounting for the experts-only NF4 sidecar checkpoint."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .bnb_nf4 import validate_packed_record_contract
from .checkpoint import TensorRecord, read_safetensors_header
from .contract import (
    ARCHITECTURE,
    BASE_MODEL_ID,
    BASE_REVISION,
    EXPERT_INDEX_FILENAME,
    MANIFEST_FILENAME,
    PACKED_STATE_SUFFIXES,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    ArchitectureContract,
    canonical_json_sha256,
    output_weight_key,
    sha256_file,
)


OUTPUT_SHARD_RE = re.compile(
    r"^experts-layer-(\d{3})-e(\d{4})-e(\d{4})\.safetensors$"
)


def read_json_no_duplicates(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
    )
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def expected_output_keys(
    contract: ArchitectureContract = ARCHITECTURE,
) -> set[str]:
    keys: set[str] = set()
    for layer in range(contract.num_layers):
        for expert in range(contract.num_experts):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                base = output_weight_key(layer, expert, projection)
                keys.update(base + suffix for suffix in PACKED_STATE_SUFFIXES)
    return keys


def _validate_manifest_identity(
    manifest: dict[str, Any],
    contract: ArchitectureContract,
) -> None:
    expected = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "SEALED",
        "complete_model_checkpoint": False,
        "base_model_id": BASE_MODEL_ID,
        "base_revision": BASE_REVISION,
        "architecture": contract.to_dict(),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"manifest identity mismatch: {key}={manifest.get(key)!r}, "
                f"expected={value!r}"
            )
    quantization = manifest.get("quantization")
    expected_quantization = {
        "method": "bitsandbytes",
        "bitsandbytes_version": "0.49.2",
        "quant_type": "nf4",
        "blocksize": 64,
        "double_quant": True,
        "quant_storage": "bfloat16",
        "compute_dtype": "bfloat16",
        "scope": "routed_experts_only",
    }
    if quantization != expected_quantization:
        raise RuntimeError(
            f"manifest quantization contract changed: {quantization!r}"
        )


def preflight_output_checkpoint(
    checkpoint_dir: Path,
    *,
    contract: ArchitectureContract = ARCHITECTURE,
    hash_shards: bool = True,
) -> dict[str, Any]:
    """Validate all output headers, index mappings, bounds, and digests."""

    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    manifest_path = checkpoint_dir / MANIFEST_FILENAME
    index_path = checkpoint_dir / EXPERT_INDEX_FILENAME
    if not manifest_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(
            f"NF4 sidecar is not sealed: {checkpoint_dir}"
        )
    manifest = read_json_no_duplicates(manifest_path)
    index = read_json_no_duplicates(index_path)
    _validate_manifest_identity(manifest, contract)
    if manifest.get("expert_index_sha256") != sha256_file(index_path):
        raise RuntimeError("expert index digest does not match sealed manifest")
    if set(index) != {"metadata", "weight_map"}:
        raise RuntimeError(f"unexpected expert index fields: {sorted(index)}")
    metadata = index["metadata"]
    weight_map = index["weight_map"]
    if not isinstance(metadata, dict) or set(metadata) != {"total_size"}:
        raise RuntimeError("expert index metadata must contain only total_size")
    if not isinstance(weight_map, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in weight_map.items()
    ):
        raise RuntimeError("expert index weight_map is malformed")
    expected_keys = expected_output_keys(contract)
    actual_keys = set(weight_map)
    if actual_keys != expected_keys:
        raise RuntimeError(
            "expert index key coverage mismatch: "
            f"missing={sorted(expected_keys - actual_keys)[:8]} "
            f"unexpected={sorted(actual_keys - expected_keys)[:8]}"
        )

    file_records = manifest.get("files")
    if not isinstance(file_records, list) or not file_records:
        raise RuntimeError("manifest files list is empty or malformed")
    records_by_name: dict[str, dict[str, Any]] = {}
    coverage: dict[int, set[int]] = defaultdict(set)
    for record in file_records:
        if not isinstance(record, dict):
            raise RuntimeError("manifest file record is not an object")
        required_fields = {
            "filename",
            "layer",
            "expert_start",
            "expert_end",
            "file_bytes",
            "payload_bytes",
            "tensor_count",
            "logical_parameter_elements",
            "sha256",
            "maximum_relative_rmse",
            "minimum_cosine_similarity",
            "maximum_quant_state_ratio",
        }
        if set(record) != required_fields:
            raise RuntimeError(
                f"manifest file fields changed: {sorted(set(record))}"
            )
        filename = record["filename"]
        match = (
            OUTPUT_SHARD_RE.fullmatch(filename)
            if isinstance(filename, str)
            else None
        )
        if match is None or Path(filename).name != filename:
            raise RuntimeError(f"unsafe NF4 output shard name: {filename!r}")
        layer, encoded_start, encoded_end = (
            int(value) for value in match.groups()
        )
        start = record["expert_start"]
        end = record["expert_end"]
        if (
            layer != record["layer"]
            or start != encoded_start
            or end != encoded_end
            or not 0 <= layer < contract.num_layers
            or not 0 <= start < end <= contract.num_experts
        ):
            raise RuntimeError(f"invalid NF4 shard range: {record}")
        if filename in records_by_name:
            raise RuntimeError(f"duplicate manifest shard: {filename}")
        if (
            not math.isfinite(record["maximum_relative_rmse"])
            or not 0 <= record["maximum_relative_rmse"] <= 0.15
            or not math.isfinite(record["minimum_cosine_similarity"])
            or not 0.98 <= record["minimum_cosine_similarity"] <= 1.0
            or not math.isfinite(record["maximum_quant_state_ratio"])
            or not 0 <= record["maximum_quant_state_ratio"] <= 0.10
        ):
            raise RuntimeError(f"NF4 shard error bounds changed: {filename}")
        records_by_name[filename] = record
        for expert in range(start, end):
            if expert in coverage[layer]:
                raise RuntimeError(
                    f"overlapping NF4 shard coverage: layer={layer} expert={expert}"
                )
            coverage[layer].add(expert)
    for layer in range(contract.num_layers):
        expected_experts = set(range(contract.num_experts))
        if coverage[layer] != expected_experts:
            raise RuntimeError(
                f"incomplete NF4 shard coverage for layer {layer}: "
                f"missing={sorted(expected_experts - coverage[layer])[:8]}"
            )

    indexed_shards = set(weight_map.values())
    if indexed_shards != set(records_by_name):
        raise RuntimeError(
            "manifest/index shard mismatch: "
            f"manifest_only={sorted(set(records_by_name) - indexed_shards)[:8]} "
            f"index_only={sorted(indexed_shards - set(records_by_name))[:8]}"
        )
    extra_shards = {
        path.name for path in checkpoint_dir.glob("experts-*.safetensors")
    } - indexed_shards
    if extra_shards:
        raise RuntimeError(f"unsealed NF4 shards present: {sorted(extra_shards)[:8]}")

    indexed_by_shard: dict[str, set[str]] = defaultdict(set)
    for key, filename in weight_map.items():
        indexed_by_shard[filename].add(key)
    all_tensors: dict[str, TensorRecord] = {}
    packed_weight_bytes = 0
    quant_state_bytes = 0
    logical_parameter_elements = 0
    for filename in sorted(indexed_shards):
        record = records_by_name[filename]
        path = checkpoint_dir / filename
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"NF4 shard is missing/symlinked: {path}")
        if path.stat().st_size != record["file_bytes"]:
            raise RuntimeError(f"NF4 shard size changed: {filename}")
        if hash_shards and sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"NF4 shard digest changed: {filename}")
        tensors = read_safetensors_header(path)
        if set(tensors) != indexed_by_shard[filename]:
            raise RuntimeError(
                f"NF4 shard/index key mismatch: {filename}"
            )
        if len(tensors) != record["tensor_count"]:
            raise RuntimeError(f"NF4 shard tensor count changed: {filename}")
        payload_bytes = sum(item.nbytes for item in tensors.values())
        if payload_bytes != record["payload_bytes"]:
            raise RuntimeError(f"NF4 shard payload size changed: {filename}")
        shard_logical_elements = 0
        for key, tensor in tensors.items():
            if key in all_tensors:
                raise RuntimeError(f"duplicate NF4 tensor across shards: {key}")
            all_tensors[key] = tensor
            base = key
            suffix = ""
            for candidate in sorted(
                (item for item in PACKED_STATE_SUFFIXES if item),
                key=len,
                reverse=True,
            ):
                if key.endswith(candidate):
                    base = key[: -len(candidate)]
                    suffix = candidate
                    break
            if suffix == "":
                validate_packed_record_contract(
                    tensors,
                    base=key,
                    logical_elements=contract.projection_elements,
                )
                packed_weight_bytes += tensor.nbytes
                shard_logical_elements += contract.projection_elements
                logical_parameter_elements += contract.projection_elements
            else:
                quant_state_bytes += tensor.nbytes
        if shard_logical_elements != record["logical_parameter_elements"]:
            raise RuntimeError(
                f"NF4 shard logical parameter count changed: {filename}"
            )

    total_payload = sum(item.nbytes for item in all_tensors.values())
    if metadata["total_size"] != total_payload:
        raise RuntimeError(
            f"expert index total_size mismatch: {metadata['total_size']} != "
            f"{total_payload}"
        )
    if logical_parameter_elements != contract.routed_parameter_elements:
        raise RuntimeError(
            "NF4 logical parameter preservation failed: "
            f"{logical_parameter_elements} != {contract.routed_parameter_elements}"
        )
    expected_packed_bytes = contract.routed_parameter_elements // 2
    if packed_weight_bytes != expected_packed_bytes:
        raise RuntimeError(
            f"NF4 packed weight accounting mismatch: {packed_weight_bytes} != "
            f"{expected_packed_bytes}"
        )
    if quant_state_bytes > packed_weight_bytes // 10:
        raise RuntimeError("aggregate NF4 quantization metadata exceeds 10% bound")
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise RuntimeError("manifest source provenance is missing")
    for key in (
        "source_dir",
        "index_sha256",
        "config_sha256",
        "weight_map_sha256",
        "tensor_count",
        "stored_elements",
        "stored_tensor_bytes",
        "shard_count",
        "routed_source_layout",
        "routed_source_tensor_count",
        "routed_parameter_elements",
    ):
        if key not in source:
            raise RuntimeError(f"manifest source provenance lacks {key}")
    if source["routed_source_layout"] != "per_expert_gate_up_down":
        raise RuntimeError("source routed layout changed in manifest")
    if (
        source["routed_source_tensor_count"]
        != contract.source_expert_tensor_count
    ):
        raise RuntimeError("source expert tensor count changed in manifest")
    if source["routed_parameter_elements"] != contract.routed_parameter_elements:
        raise RuntimeError("source routed parameter count changed in manifest")

    return {
        "status": "PASS",
        "checkpoint_dir": str(checkpoint_dir),
        "manifest_sha256": sha256_file(manifest_path),
        "expert_index_sha256": sha256_file(index_path),
        "weight_map_sha256": canonical_json_sha256(weight_map),
        "shard_count": len(indexed_shards),
        "serialized_tensor_count": len(all_tensors),
        "logical_linear_count": contract.output_linear_count,
        "logical_parameter_elements": logical_parameter_elements,
        "packed_weight_bytes": packed_weight_bytes,
        "quant_state_bytes": quant_state_bytes,
        "total_payload_bytes": total_payload,
        "source_index_sha256": source["index_sha256"],
        "production_eligible": True,
    }
