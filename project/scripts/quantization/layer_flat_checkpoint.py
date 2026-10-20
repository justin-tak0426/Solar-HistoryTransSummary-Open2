#!/usr/bin/env python3
"""Sealing, accounting, and loading for layer-flat NF4 expert sidecars."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from .bnb_nf4 import (
    linear4bit_from_packed,
    validate_packed_record_contract,
)
from .checkpoint import read_safetensors_header
from .contract import (
    ARCHITECTURE,
    BASE_MODEL_ID,
    BASE_REVISION,
    EXPERT_INDEX_FILENAME,
    LAYER_FLAT_EXPERTS_PER_CHUNK,
    LAYER_FLAT_SCHEMA_NAME,
    MANIFEST_FILENAME,
    PACKED_STATE_SUFFIXES,
    SCHEMA_VERSION,
    ArchitectureContract,
    canonical_json_sha256,
    layer_flat_chunk_ranges,
    layer_flat_weight_key,
    sha256_file,
)
from .layer_flat_module import SolarOpen2LayerFlatNF4Experts
from .output_checkpoint import read_json_no_duplicates


FLAT_SHARD_RE = re.compile(r"^experts-layer-(\d{3})-flat\.safetensors$")
FLAT_WEIGHT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\."
    r"(gate_up_proj|down_proj)\.chunks\.(\d+)\.weight$"
)


def expected_flat_keys(
    contract: ArchitectureContract = ARCHITECTURE,
) -> set[str]:
    return {
        layer_flat_weight_key(layer, projection, chunk, contract) + suffix
        for layer in range(contract.num_layers)
        for projection in ("gate_up_proj", "down_proj")
        for chunk in range(len(layer_flat_chunk_ranges(contract)))
        for suffix in PACKED_STATE_SUFFIXES
    }


def _quantization_contract() -> dict[str, Any]:
    return {
        "method": "bitsandbytes",
        "bitsandbytes_version": "0.49.2",
        "quant_type": "nf4",
        "blocksize": 64,
        "double_quant": True,
        "quant_storage": "bfloat16",
        "compute_dtype": "bfloat16",
        "scope": "routed_experts_only",
        "layout": "layer_flat_2d_expert_chunks",
        "experts_per_chunk": LAYER_FLAT_EXPERTS_PER_CHUNK,
        "runtime": "dequantize_full_layer_then_native_dispatch",
    }


def _validate_identity(
    manifest: dict[str, Any],
    contract: ArchitectureContract,
) -> None:
    exact = {
        "schema": LAYER_FLAT_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "SEALED",
        "complete_model_checkpoint": False,
        "base_model_id": BASE_MODEL_ID,
        "base_revision": BASE_REVISION,
        "architecture": contract.to_dict(),
        "quantization": _quantization_contract(),
    }
    for key, expected in exact.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"layer-flat manifest mismatch: {key}={manifest.get(key)!r}, "
                f"expected={expected!r}"
            )


def preflight_layer_flat_checkpoint(
    checkpoint_dir: Path,
    *,
    contract: ArchitectureContract = ARCHITECTURE,
    hash_shards: bool = True,
) -> dict[str, Any]:
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    manifest_path = checkpoint_dir / MANIFEST_FILENAME
    index_path = checkpoint_dir / EXPERT_INDEX_FILENAME
    if not manifest_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(f"unsealed layer-flat sidecar: {checkpoint_dir}")
    manifest = read_json_no_duplicates(manifest_path)
    index = read_json_no_duplicates(index_path)
    _validate_identity(manifest, contract)
    if manifest.get("expert_index_sha256") != sha256_file(index_path):
        raise RuntimeError("layer-flat expert index digest changed")
    if set(index) != {"metadata", "weight_map"}:
        raise RuntimeError("invalid layer-flat expert index fields")
    metadata, weight_map = index["metadata"], index["weight_map"]
    if not isinstance(metadata, dict) or set(metadata) != {"total_size"}:
        raise RuntimeError("invalid layer-flat index metadata")
    if not isinstance(weight_map, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in weight_map.items()
    ):
        raise RuntimeError("invalid layer-flat weight_map")
    expected = expected_flat_keys(contract)
    if set(weight_map) != expected:
        raise RuntimeError(
            "layer-flat index coverage mismatch: "
            f"missing={sorted(expected - set(weight_map))[:8]} "
            f"unexpected={sorted(set(weight_map) - expected)[:8]}"
        )
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != contract.num_layers:
        raise RuntimeError("layer-flat manifest must seal exactly one shard per layer")
    records: dict[str, dict[str, Any]] = {}
    observed_layers: set[int] = set()
    record_fields = {
        "filename",
        "layer",
        "file_bytes",
        "payload_bytes",
        "tensor_count",
        "logical_parameter_elements",
        "sha256",
        "maximum_relative_rmse",
        "minimum_cosine_similarity",
        "maximum_quant_state_ratio",
        "dequantized_runtime_bytes",
    }
    for record in files:
        if not isinstance(record, dict) or set(record) != record_fields:
            raise RuntimeError("layer-flat file record schema changed")
        filename = record["filename"]
        match = FLAT_SHARD_RE.fullmatch(filename) if isinstance(filename, str) else None
        if match is None or Path(filename).name != filename:
            raise RuntimeError(f"unsafe layer-flat shard name: {filename!r}")
        layer = int(match.group(1))
        if layer != record["layer"] or not 0 <= layer < contract.num_layers:
            raise RuntimeError(f"layer-flat shard layer mismatch: {record}")
        if layer in observed_layers or filename in records:
            raise RuntimeError(f"duplicate layer-flat shard: {filename}")
        expected_runtime_bytes = (
            contract.num_experts
            * 3
            * contract.hidden_size
            * contract.intermediate_size
            * 2
        )
        if record["dequantized_runtime_bytes"] != expected_runtime_bytes:
            raise RuntimeError(
                f"layer-flat runtime byte contract changed: {filename}"
            )
        expected_logical = (
            contract.num_experts
            * 3
            * contract.hidden_size
            * contract.intermediate_size
        )
        if record["logical_parameter_elements"] != expected_logical:
            raise RuntimeError(
                f"layer-flat logical parameter count changed: {filename}"
            )
        if (
            not math.isfinite(record["maximum_relative_rmse"])
            or not 0 <= record["maximum_relative_rmse"] <= 0.15
            or not math.isfinite(record["minimum_cosine_similarity"])
            or not 0.98 <= record["minimum_cosine_similarity"] <= 1.0
            or not math.isfinite(record["maximum_quant_state_ratio"])
            or not 0 <= record["maximum_quant_state_ratio"] <= 0.10
        ):
            raise RuntimeError(f"layer-flat error bounds changed: {filename}")
        observed_layers.add(layer)
        records[filename] = record
    if observed_layers != set(range(contract.num_layers)):
        raise RuntimeError("layer-flat manifest layer coverage is incomplete")
    indexed_shards = set(weight_map.values())
    if indexed_shards != set(records):
        raise RuntimeError("layer-flat index/manifest shard sets differ")

    by_shard: dict[str, set[str]] = defaultdict(set)
    for key, filename in weight_map.items():
        by_shard[filename].add(key)
    actual_all: set[str] = set()
    packed_bytes = 0
    state_bytes = 0
    payload_bytes = 0
    logical_elements = 0
    for filename in sorted(records):
        path = checkpoint_dir / filename
        record = records[filename]
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"layer-flat shard missing/symlinked: {path}")
        if path.stat().st_size != record["file_bytes"]:
            raise RuntimeError(f"layer-flat shard size changed: {filename}")
        if hash_shards and sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"layer-flat shard digest changed: {filename}")
        tensors = read_safetensors_header(path)
        if set(tensors) != by_shard[filename]:
            raise RuntimeError(f"layer-flat shard/index keys differ: {filename}")
        expected_layer_keys = {
            layer_flat_weight_key(
                record["layer"], projection, chunk, contract
            )
            + suffix
            for projection in ("gate_up_proj", "down_proj")
            for chunk in range(len(layer_flat_chunk_ranges(contract)))
            for suffix in PACKED_STATE_SUFFIXES
        }
        if set(tensors) != expected_layer_keys:
            raise RuntimeError(
                f"layer-flat shard contains another layer's tensors: {filename}"
            )
        if len(tensors) != record["tensor_count"]:
            raise RuntimeError(f"layer-flat shard tensor count changed: {filename}")
        shard_payload = sum(item.nbytes for item in tensors.values())
        if shard_payload != record["payload_bytes"]:
            raise RuntimeError(f"layer-flat shard payload changed: {filename}")
        layer_logical = 0
        for key, tensor in tensors.items():
            if key in actual_all:
                raise RuntimeError(f"duplicate layer-flat tensor: {key}")
            actual_all.add(key)
            suffix = next(
                (
                    candidate
                    for candidate in sorted(
                        (value for value in PACKED_STATE_SUFFIXES if value),
                        key=len,
                        reverse=True,
                    )
                    if key.endswith(candidate)
                ),
                "",
            )
            if suffix:
                state_bytes += tensor.nbytes
                continue
            if tensor.dtype != "BF16":
                raise RuntimeError(f"layer-flat packed weight is not BF16: {key}")
            match = FLAT_WEIGHT_RE.fullmatch(key)
            if match is None:
                raise RuntimeError(f"unknown layer-flat packed weight: {key}")
            key_layer, projection, chunk_text = match.groups()
            chunk = int(chunk_text)
            ranges = layer_flat_chunk_ranges(contract)
            if int(key_layer) != record["layer"] or not 0 <= chunk < len(ranges):
                raise RuntimeError(f"layer-flat packed weight bounds changed: {key}")
            start, end = ranges[chunk]
            chunk_experts = end - start
            if projection == "gate_up_proj":
                elements = (
                    chunk_experts
                    * 2
                    * contract.intermediate_size
                    * contract.hidden_size
                )
            elif projection == "down_proj":
                elements = (
                    chunk_experts
                    * contract.hidden_size
                    * contract.intermediate_size
                )
            else:
                raise RuntimeError(f"unknown layer-flat packed weight: {key}")
            validate_packed_record_contract(
                tensors,
                base=key,
                logical_elements=elements,
            )
            packed_bytes += tensor.nbytes
            layer_logical += elements
            logical_elements += elements
        if layer_logical != record["logical_parameter_elements"]:
            raise RuntimeError(
                f"layer-flat logical count changed: {filename}"
            )
        payload_bytes += shard_payload
    if metadata["total_size"] != payload_bytes:
        raise RuntimeError("layer-flat index total_size differs from payload")
    if logical_elements != contract.routed_parameter_elements:
        raise RuntimeError("layer-flat logical parameter preservation failed")
    if packed_bytes != contract.routed_parameter_elements // 2:
        raise RuntimeError("layer-flat packed weight accounting failed")
    if state_bytes > packed_bytes // 10:
        raise RuntimeError("layer-flat quant metadata exceeds 10% bound")
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise RuntimeError("layer-flat manifest source provenance is missing")
    required_source = {
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
    }
    if not required_source.issubset(source):
        raise RuntimeError(
            f"layer-flat source provenance lacks "
            f"{sorted(required_source - set(source))}"
        )
    if source["routed_source_layout"] != "per_expert_gate_up_down":
        raise RuntimeError("layer-flat source routed layout changed")
    if (
        source["routed_source_tensor_count"]
        != contract.source_expert_tensor_count
    ):
        raise RuntimeError("layer-flat source expert tensor count changed")
    if source["routed_parameter_elements"] != contract.routed_parameter_elements:
        raise RuntimeError("layer-flat source routed parameter count changed")
    return {
        "status": "PASS",
        "checkpoint_dir": str(checkpoint_dir),
        "manifest_sha256": sha256_file(manifest_path),
        "expert_index_sha256": sha256_file(index_path),
        "weight_map_sha256": canonical_json_sha256(weight_map),
        "shard_count": len(records),
        "serialized_tensor_count": len(actual_all),
        "logical_parameter_elements": logical_elements,
        "packed_weight_bytes": packed_bytes,
        "quant_state_bytes": state_bytes,
        "total_payload_bytes": payload_bytes,
        "production_eligible": True,
    }


def load_layer_flat_experts(
    checkpoint_dir: Path,
    *,
    layer: int,
    device: str,
    contract: ArchitectureContract = ARCHITECTURE,
    verify_hash: bool = True,
) -> SolarOpen2LayerFlatNF4Experts:
    if not 0 <= layer < contract.num_layers:
        raise ValueError(f"layer out of bounds: {layer}")
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    manifest = read_json_no_duplicates(checkpoint_dir / MANIFEST_FILENAME)
    index = read_json_no_duplicates(checkpoint_dir / EXPERT_INDEX_FILENAME)
    _validate_identity(manifest, contract)
    if manifest.get("expert_index_sha256") != sha256_file(
        checkpoint_dir / EXPERT_INDEX_FILENAME
    ):
        raise RuntimeError("layer-flat index digest changed")
    records = {record["layer"]: record for record in manifest["files"]}
    if set(records) != set(range(contract.num_layers)):
        raise RuntimeError("layer-flat manifest does not cover every layer")
    record = records[layer]
    path = checkpoint_dir / record["filename"]
    if path.is_symlink() or path.stat().st_size != record["file_bytes"]:
        raise RuntimeError(f"layer-flat shard changed: {path}")
    if verify_hash and sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"layer-flat shard digest changed: {path}")
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("safetensors is required for layer-flat loading") from error
    weight_map = index["weight_map"]
    expected = {
        layer_flat_weight_key(layer, projection, chunk, contract) + suffix
        for projection in ("gate_up_proj", "down_proj")
        for chunk in range(len(layer_flat_chunk_ranges(contract)))
        for suffix in PACKED_STATE_SUFFIXES
    }
    if {key for key, value in weight_map.items() if value == path.name} != expected:
        raise RuntimeError(f"layer-flat index mapping changed for layer {layer}")
    with safe_open(path, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != expected:
            raise RuntimeError(f"layer-flat shard contents changed: {path}")
        tensors = {key: handle.get_tensor(key) for key in sorted(expected)}
    modules: dict[str, list[Any]] = {
        "gate_up_proj": [],
        "down_proj": [],
    }
    ranges = layer_flat_chunk_ranges(contract)
    for projection in ("gate_up_proj", "down_proj"):
        for chunk, (start, end) in enumerate(ranges):
            base = layer_flat_weight_key(
                layer, projection, chunk, contract
            )
            packed = {
                suffix: tensors.pop(base + suffix)
                for suffix in PACKED_STATE_SUFFIXES
            }
            chunk_experts = end - start
            if projection == "gate_up_proj":
                in_features = contract.hidden_size
                out_features = (
                    chunk_experts * 2 * contract.intermediate_size
                )
            else:
                in_features = contract.intermediate_size
                out_features = chunk_experts * contract.hidden_size
            modules[projection].append(
                linear4bit_from_packed(
                    tensors=packed,
                    in_features=in_features,
                    out_features=out_features,
                    device=device,
                )
            )
    if tensors:
        raise AssertionError("unconsumed layer-flat tensors")
    return SolarOpen2LayerFlatNF4Experts(
        modules["gate_up_proj"],
        modules["down_proj"],
        num_experts=contract.num_experts,
        hidden_size=contract.hidden_size,
        intermediate_size=contract.intermediate_size,
    )
