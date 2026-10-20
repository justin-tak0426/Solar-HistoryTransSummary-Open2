#!/usr/bin/env python3
"""Production-default Solar routed-expert conversion to layer-flat NF4.

The command is read-only unless ``--execute`` is present.  It never downloads
weights and only accepts the pinned Hugging Face snapshot under the assigned
runtime root.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import platform
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .bnb_nf4 import quantize_nf4_matrix, require_bitsandbytes
from .checkpoint import (
    SourceInventory,
    inventory_source_checkpoint,
    read_safetensors_header,
    validate_transformers_schema,
)
from .contract import (
    ARCHITECTURE,
    BASE_MODEL_ID,
    BASE_REVISION,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE_DIR,
    EXPERT_INDEX_FILENAME,
    LAYER_FLAT_EXPERTS_PER_CHUNK,
    LAYER_FLAT_SCHEMA_NAME,
    MANIFEST_FILENAME,
    OWNER_FILENAME,
    PACKED_STATE_SUFFIXES,
    PROGRESS_FILENAME,
    SCHEMA_VERSION,
    atomic_json,
    canonical_json_sha256,
    fsync_directory,
    layer_flat_chunk_ranges,
    layer_flat_weight_key,
    require_output_path,
    sha256_file,
    source_expert_key,
)
from .layer_flat_checkpoint import (
    _quantization_contract,
    expected_flat_keys,
    preflight_layer_flat_checkpoint,
)
from .layer_flat_module import dequantized_layer_bytes
from .output_checkpoint import read_json_no_duplicates


def _filename(layer: int) -> str:
    return f"experts-layer-{layer:03d}-flat.safetensors"


def _plan() -> list[dict[str, int | str]]:
    return [
        {"filename": _filename(layer), "layer": layer}
        for layer in range(ARCHITECTURE.num_layers)
    ]


def _owner(inventory: SourceInventory) -> dict[str, Any]:
    plan = _plan()
    return {
        "schema": LAYER_FLAT_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "owner_user": "mk03",
        "base_model_id": BASE_MODEL_ID,
        "base_revision": BASE_REVISION,
        "source_dir": str(inventory.source_dir),
        "source_index_sha256": inventory.index_sha256,
        "source_config_sha256": inventory.config_sha256,
        "architecture": ARCHITECTURE.to_dict(),
        "quantization": _quantization_contract(),
        "plan_sha256": canonical_json_sha256(plan),
    }


def _initialize(
    output_dir: Path,
    *,
    owner: dict[str, Any],
) -> dict[str, Any]:
    output_dir = require_output_path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
        fsync_directory(output_dir.parent)
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise RuntimeError(f"invalid output directory: {output_dir}")
    output_dir = require_output_path(output_dir)
    owner_path = output_dir / OWNER_FILENAME
    if owner_path.exists():
        if read_json_no_duplicates(owner_path) != owner:
            raise RuntimeError("existing output owner differs from this conversion")
        for partial in output_dir.glob(".*.partial"):
            if partial.is_symlink() or not partial.is_file():
                raise RuntimeError(f"unsafe orphan conversion temporary: {partial}")
            partial.unlink()
        fsync_directory(output_dir)
    else:
        if any(output_dir.iterdir()):
            raise RuntimeError("refusing nonempty converter output without owner marker")
        atomic_json(owner_path, owner)
    progress_path = output_dir / PROGRESS_FILENAME
    if progress_path.exists():
        progress = read_json_no_duplicates(progress_path)
    else:
        progress = {
            "schema": LAYER_FLAT_SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "status": "BUILDING",
            "owner_sha256": sha256_file(owner_path),
            "plan_sha256": owner["plan_sha256"],
            "completed": {},
        }
        atomic_json(progress_path, progress)
    exact = {
        "schema": LAYER_FLAT_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "BUILDING",
        "owner_sha256": sha256_file(owner_path),
        "plan_sha256": owner["plan_sha256"],
    }
    for key, expected in exact.items():
        if progress.get(key) != expected:
            raise RuntimeError(f"layer-flat progress mismatch: {key}")
    if not isinstance(progress.get("completed"), dict):
        raise RuntimeError("layer-flat progress completed map is malformed")
    expected_names = {_filename(layer) for layer in range(ARCHITECTURE.num_layers)}
    if not set(progress["completed"]).issubset(expected_names):
        raise RuntimeError("layer-flat progress contains unknown shards")
    return progress


def _metadata(
    *,
    layer: int,
    owner: dict[str, Any],
    maximum_relative_rmse: float,
    minimum_cosine_similarity: float,
    maximum_quant_state_ratio: float,
) -> dict[str, str]:
    return {
        "schema": LAYER_FLAT_SCHEMA_NAME,
        "schema_version": str(SCHEMA_VERSION),
        "base_model_id": BASE_MODEL_ID,
        "base_revision": BASE_REVISION,
        "source_index_sha256": owner["source_index_sha256"],
        "plan_sha256": owner["plan_sha256"],
        "layer": str(layer),
        "quant_type": "nf4",
        "quant_storage": "bfloat16",
        "maximum_relative_rmse": repr(maximum_relative_rmse),
        "minimum_cosine_similarity": repr(minimum_cosine_similarity),
        "maximum_quant_state_ratio": repr(maximum_quant_state_ratio),
    }


def _expected_layer_keys(layer: int) -> set[str]:
    return {
        layer_flat_weight_key(layer, projection, chunk) + suffix
        for projection in ("gate_up_proj", "down_proj")
        for chunk in range(len(layer_flat_chunk_ranges()))
        for suffix in PACKED_STATE_SUFFIXES
    }


def _read_metadata(path: Path) -> dict[str, str]:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("safetensors is required") from error
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    if not isinstance(metadata, dict):
        raise RuntimeError(f"layer-flat shard lacks metadata: {path}")
    return metadata


def _record_from_file(
    path: Path,
    *,
    layer: int,
    owner: dict[str, Any],
) -> dict[str, Any]:
    metadata = _read_metadata(path)
    static = {
        "schema": LAYER_FLAT_SCHEMA_NAME,
        "schema_version": str(SCHEMA_VERSION),
        "base_model_id": BASE_MODEL_ID,
        "base_revision": BASE_REVISION,
        "source_index_sha256": owner["source_index_sha256"],
        "plan_sha256": owner["plan_sha256"],
        "layer": str(layer),
        "quant_type": "nf4",
        "quant_storage": "bfloat16",
    }
    for key, expected in static.items():
        if metadata.get(key) != expected:
            raise RuntimeError(f"layer-flat shard metadata mismatch: {path} {key}")
    metric_keys = {
        "maximum_relative_rmse",
        "minimum_cosine_similarity",
        "maximum_quant_state_ratio",
    }
    if set(metadata) != set(static) | metric_keys:
        raise RuntimeError(f"unexpected layer-flat shard metadata: {path}")
    try:
        maximum_rmse = float(metadata["maximum_relative_rmse"])
        minimum_cosine = float(metadata["minimum_cosine_similarity"])
        maximum_ratio = float(metadata["maximum_quant_state_ratio"])
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"invalid layer-flat error metrics: {path}") from error
    tensors = read_safetensors_header(path)
    if set(tensors) != _expected_layer_keys(layer):
        raise RuntimeError(f"layer-flat shard key mismatch: {path}")
    logical = (
        ARCHITECTURE.num_experts
        * 3
        * ARCHITECTURE.projection_elements
    )
    return {
        "filename": path.name,
        "layer": layer,
        "file_bytes": path.stat().st_size,
        "payload_bytes": sum(item.nbytes for item in tensors.values()),
        "tensor_count": len(tensors),
        "logical_parameter_elements": logical,
        "sha256": sha256_file(path),
        "maximum_relative_rmse": maximum_rmse,
        "minimum_cosine_similarity": minimum_cosine,
        "maximum_quant_state_ratio": maximum_ratio,
        "dequantized_runtime_bytes": dequantized_layer_bytes(
            num_experts=ARCHITECTURE.num_experts,
            hidden_size=ARCHITECTURE.hidden_size,
            intermediate_size=ARCHITECTURE.intermediate_size,
        ),
    }


def _validate_existing(
    path: Path,
    *,
    layer: int,
    owner: dict[str, Any],
    progress_record: dict[str, Any] | None,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"layer-flat shard missing/symlinked: {path}")
    rebuilt = _record_from_file(path, layer=layer, owner=owner)
    if progress_record is not None and rebuilt != progress_record:
        raise RuntimeError(f"layer-flat progress record differs from shard: {path}")
    # Unlike the fallback layout, error metrics are sealed in safetensors
    # metadata, so a crash after rename can be adopted without guessing.
    return rebuilt


def _open_layer_source_handles(
    inventory: SourceInventory,
    layer: int,
) -> tuple[ExitStack, dict[str, Any]]:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("safetensors is required") from error
    keys = {
        source_expert_key(layer, expert, projection)
        for expert in range(ARCHITECTURE.num_experts)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    shard_names = {inventory.tensors[key].shard for key in keys}
    stack = ExitStack()
    handles = {
        shard: stack.enter_context(
            safe_open(
                inventory.source_dir / shard,
                framework="pt",
                device="cpu",
            )
        )
        for shard in sorted(shard_names)
    }
    return stack, handles


def _write_layer(
    output_dir: Path,
    *,
    layer: int,
    inventory: SourceInventory,
    owner: dict[str, Any],
    device: str,
    max_relative_rmse: float,
    min_cosine_similarity: float,
) -> dict[str, Any]:
    try:
        import torch
        from safetensors.torch import save_file
    except ImportError as error:
        raise RuntimeError("torch and safetensors are required") from error
    tensors: dict[str, torch.Tensor] = {}
    metrics = []
    stack, handles = _open_layer_source_handles(inventory, layer)
    try:
        for chunk, (start, end) in enumerate(layer_flat_chunk_ranges()):
            chunk_experts = end - start
            for projection in ("gate_up_proj", "down_proj"):
                if projection == "gate_up_proj":
                    source = torch.empty(
                        (
                            chunk_experts,
                            2 * ARCHITECTURE.intermediate_size,
                            ARCHITECTURE.hidden_size,
                        ),
                        dtype=torch.bfloat16,
                        device="cpu",
                    )
                    for offset, expert in enumerate(range(start, end)):
                        for source_projection, row_start in (
                            ("gate_proj", 0),
                            (
                                "up_proj",
                                ARCHITECTURE.intermediate_size,
                            ),
                        ):
                            key = source_expert_key(
                                layer, expert, source_projection
                            )
                            handle = handles[inventory.tensors[key].shard]
                            value = handle.get_tensor(key)
                            if (
                                tuple(value.shape)
                                != (
                                    ARCHITECTURE.intermediate_size,
                                    ARCHITECTURE.hidden_size,
                                )
                                or value.dtype is not torch.bfloat16
                            ):
                                raise RuntimeError(
                                    f"source expert tensor changed: {key}"
                                )
                            source[
                                offset,
                                row_start : row_start
                                + ARCHITECTURE.intermediate_size,
                                :,
                            ].copy_(value)
                            del value
                    flattened_shape = (
                        chunk_experts * 2 * ARCHITECTURE.intermediate_size,
                        ARCHITECTURE.hidden_size,
                    )
                else:
                    source = torch.empty(
                        (
                            chunk_experts,
                            ARCHITECTURE.hidden_size,
                            ARCHITECTURE.intermediate_size,
                        ),
                        dtype=torch.bfloat16,
                        device="cpu",
                    )
                    for offset, expert in enumerate(range(start, end)):
                        key = source_expert_key(
                            layer, expert, "down_proj"
                        )
                        handle = handles[inventory.tensors[key].shard]
                        value = handle.get_tensor(key)
                        if (
                            tuple(value.shape)
                            != (
                                ARCHITECTURE.hidden_size,
                                ARCHITECTURE.intermediate_size,
                            )
                            or value.dtype is not torch.bfloat16
                        ):
                            raise RuntimeError(
                                f"source expert tensor changed: {key}"
                            )
                        source[offset].copy_(value)
                        del value
                    flattened_shape = (
                        chunk_experts * ARCHITECTURE.hidden_size,
                        ARCHITECTURE.intermediate_size,
                    )
                flattened = source.view(*flattened_shape)
                if not flattened.is_contiguous():
                    raise RuntimeError(
                        f"source row-major flatten is not contiguous: "
                        f"layer={layer} projection={projection} chunk={chunk}"
                    )
                if flattened.numel() > 2_147_483_647:
                    raise RuntimeError(
                        f"bnb CUDA INT_MAX bound exceeded: layer={layer} "
                        f"projection={projection} chunk={chunk} "
                        f"numel={flattened.numel()}"
                    )
                packed = quantize_nf4_matrix(
                    flattened,
                    device=device,
                    max_relative_rmse=max_relative_rmse,
                    min_cosine_similarity=min_cosine_similarity,
                )
                base = layer_flat_weight_key(layer, projection, chunk)
                for suffix, tensor in packed.tensors.items():
                    key = base + suffix
                    if key in tensors:
                        raise RuntimeError(f"layer-flat output collision: {key}")
                    tensors[key] = tensor
                metrics.append(packed.metrics)
                del flattened, source, packed
                torch.cuda.empty_cache()
    finally:
        stack.close()
    expected_keys = _expected_layer_keys(layer)
    if set(tensors) != expected_keys:
        raise RuntimeError("layer-flat converter produced wrong keys")
    maximum_rmse = max(item.relative_rmse for item in metrics)
    minimum_cosine = min(item.cosine_similarity for item in metrics)
    maximum_ratio = max(
        item.quant_state_bytes / item.packed_weight_bytes for item in metrics
    )
    path = output_dir / _filename(layer)
    temporary = output_dir / f".{path.name}.{os.getpid()}.partial"
    if path.exists() or temporary.exists():
        raise FileExistsError(path if path.exists() else temporary)
    try:
        save_file(
            tensors,
            str(temporary),
            metadata=_metadata(
                layer=layer,
                owner=owner,
                maximum_relative_rmse=maximum_rmse,
                minimum_cosine_similarity=minimum_cosine,
                maximum_quant_state_ratio=maximum_ratio,
            ),
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        if set(read_safetensors_header(temporary)) != expected_keys:
            raise RuntimeError("temporary layer-flat shard failed header check")
        os.replace(temporary, path)
        fsync_directory(output_dir)
    finally:
        if temporary.exists():
            temporary.unlink()
    del tensors
    torch.cuda.empty_cache()
    return _record_from_file(path, layer=layer, owner=owner)


def _source_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    names = (
        "contract.py",
        "checkpoint.py",
        "bnb_nf4.py",
        "layer_flat_module.py",
        "layer_flat_checkpoint.py",
        "convert_layer_flat.py",
        "full_model_loader.py",
    )
    result = {}
    for name in names:
        path = directory / name
        if not path.is_file():
            raise RuntimeError(f"converter source is missing: {path}")
        result[name] = sha256_file(path)
    return result


def _seal(
    output_dir: Path,
    *,
    inventory: SourceInventory,
    schema_receipt: dict[str, Any],
    progress: dict[str, Any],
    owner: dict[str, Any],
) -> dict[str, Any]:
    expected_names = {_filename(layer) for layer in range(ARCHITECTURE.num_layers)}
    if set(progress["completed"]) != expected_names:
        raise RuntimeError("cannot seal incomplete layer-flat conversion")
    weight_map: dict[str, str] = {}
    files = []
    total_size = 0
    for layer in range(ARCHITECTURE.num_layers):
        path = output_dir / _filename(layer)
        record = _validate_existing(
            path,
            layer=layer,
            owner=owner,
            progress_record=progress["completed"][_filename(layer)],
        )
        files.append(record)
        tensors = read_safetensors_header(path)
        total_size += sum(item.nbytes for item in tensors.values())
        for key in sorted(tensors):
            if key in weight_map:
                raise RuntimeError(f"duplicate layer-flat tensor: {key}")
            weight_map[key] = path.name
    if set(weight_map) != expected_flat_keys():
        raise RuntimeError("layer-flat final weight map is incomplete")
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(weight_map.items())),
    }
    index_path = output_dir / EXPERT_INDEX_FILENAME
    if index_path.exists():
        if read_json_no_duplicates(index_path) != index:
            raise RuntimeError("existing layer-flat index differs")
    else:
        atomic_json(index_path, index)
    source = inventory.summary()
    source["shards"] = [
        asdict(inventory.shards[name])
        for name in sorted(inventory.shards)
    ]
    manifest = {
        "schema": LAYER_FLAT_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "SEALED",
        "complete_model_checkpoint": False,
        "base_model_id": BASE_MODEL_ID,
        "base_revision": BASE_REVISION,
        "architecture": ARCHITECTURE.to_dict(),
        "quantization": _quantization_contract(),
        "source": source,
        "official_transformers_schema": schema_receipt,
        "mapping": {
            "gate_up_proj": (
                "concatenate source expert gate_proj then up_proj row-wise, "
                "expert-aligned chunks viewed as "
                "[chunk_experts*2*intermediate_size,hidden_size]"
            ),
            "down_proj": (
                "stack source expert down_proj tensors, expert-aligned chunks viewed as "
                "[chunk_experts*hidden_size,intermediate_size]"
            ),
            "experts_per_chunk": LAYER_FLAT_EXPERTS_PER_CHUNK,
            "transpose": False,
        },
        "runtime_memory": {
            "dequantized_bytes_per_active_decoder_layer": (
                dequantized_layer_bytes(
                    num_experts=ARCHITECTURE.num_experts,
                    hidden_size=ARCHITECTURE.hidden_size,
                    intermediate_size=ARCHITECTURE.intermediate_size,
                )
            ),
            "requires_decoder_gradient_checkpointing": True,
            "full_dequantized_layers_must_not_be_cached": True,
        },
        "plan_sha256": owner["plan_sha256"],
        "converter_source_sha256": _source_hashes(),
        "files": files,
        "expert_index_sha256": sha256_file(index_path),
        "host": platform.node(),
        "created_by_user": getpass.getuser(),
        "logical_parameter_elements": ARCHITECTURE.routed_parameter_elements,
        "serialized_tensor_count": len(weight_map),
    }
    manifest_path = output_dir / MANIFEST_FILENAME
    if manifest_path.exists():
        if read_json_no_duplicates(manifest_path) != manifest:
            raise RuntimeError("existing layer-flat manifest differs")
    else:
        atomic_json(manifest_path, manifest)
    return preflight_layer_flat_checkpoint(output_dir, hash_shards=True)


def convert_layer_flat(
    *,
    source_dir: Path,
    output_dir: Path,
    device: str,
    execute: bool,
    max_relative_rmse: float,
    min_cosine_similarity: float,
) -> dict[str, Any]:
    if getpass.getuser() != "mk03":
        raise PermissionError("Solar conversion is restricted to mk03")
    if not 0 < max_relative_rmse <= 0.15:
        raise ValueError("production maximum relative RMSE must be in (0,0.15]")
    if not 0.98 <= min_cosine_similarity <= 1.0:
        raise ValueError("production minimum cosine similarity must be in [0.98,1]")
    inventory = inventory_source_checkpoint(
        source_dir,
        hash_shards=execute,
        require_revision_path=True,
    )
    schema_receipt = validate_transformers_schema(inventory)
    result = {
        "status": "PLAN_ONLY" if not execute else "BUILDING",
        "source": inventory.summary(),
        "official_transformers_schema": schema_receipt,
        "output_dir": str(output_dir),
        "layout": "layer_flat_2d_expert_chunks",
        "experts_per_chunk": LAYER_FLAT_EXPERTS_PER_CHUNK,
        "output_shard_count": ARCHITECTURE.num_layers,
        "serialized_tensor_count": (
            ARCHITECTURE.num_layers
            * 2
            * len(layer_flat_chunk_ranges())
            * len(PACKED_STATE_SUFFIXES)
        ),
        "logical_parameter_elements": ARCHITECTURE.routed_parameter_elements,
        "minimum_packed_weight_bytes": (
            ARCHITECTURE.routed_parameter_elements // 2
        ),
        "dequantized_runtime_bytes_per_layer": dequantized_layer_bytes(
            num_experts=ARCHITECTURE.num_experts,
            hidden_size=ARCHITECTURE.hidden_size,
            intermediate_size=ARCHITECTURE.intermediate_size,
        ),
        "writes_performed": False,
    }
    if not execute:
        return result
    require_bitsandbytes(require_cuda=True)
    output_dir = require_output_path(output_dir)
    owner = _owner(inventory)
    progress = _initialize(output_dir, owner=owner)
    progress_path = output_dir / PROGRESS_FILENAME
    allowed = {
        OWNER_FILENAME,
        PROGRESS_FILENAME,
        EXPERT_INDEX_FILENAME,
        MANIFEST_FILENAME,
        *[_filename(layer) for layer in range(ARCHITECTURE.num_layers)],
    }
    unexpected = {path.name for path in output_dir.iterdir()} - allowed
    if unexpected:
        raise RuntimeError(
            f"unexpected layer-flat output files: {sorted(unexpected)[:8]}"
        )
    if (output_dir / MANIFEST_FILENAME).exists():
        return preflight_layer_flat_checkpoint(output_dir, hash_shards=True)
    for layer in range(ARCHITECTURE.num_layers):
        name = _filename(layer)
        path = output_dir / name
        prior = progress["completed"].get(name)
        if path.exists():
            rebuilt = _validate_existing(
                path, layer=layer, owner=owner, progress_record=prior
            )
            if prior is None:
                progress["completed"][name] = rebuilt
                atomic_json(progress_path, progress)
            continue
        if prior is not None:
            raise RuntimeError(f"progress records missing layer-flat shard: {name}")
        record = _write_layer(
            output_dir,
            layer=layer,
            inventory=inventory,
            owner=owner,
            device=device,
            max_relative_rmse=max_relative_rmse,
            min_cosine_similarity=min_cosine_similarity,
        )
        progress["completed"][name] = record
        atomic_json(progress_path, progress)
        print(
            json.dumps(
                {
                    "event": "LAYER_FLAT_NF4_COMPLETE",
                    "layer": layer,
                    "total_layers": ARCHITECTURE.num_layers,
                    **record,
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
            flush=True,
        )
    receipt = _seal(
        output_dir,
        inventory=inventory,
        schema_receipt=schema_receipt,
        progress=progress,
        owner=owner,
    )
    receipt["writes_performed"] = True
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-relative-rmse", type=float, default=0.15)
    parser.add_argument("--min-cosine-similarity", type=float, default=0.98)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 0 < args.max_relative_rmse <= 0.15:
        parser.error("--max-relative-rmse must be in (0,0.15]")
    if not 0.98 <= args.min_cosine_similarity <= 1.0:
        parser.error("--min-cosine-similarity must be in [0.98,1.0]")
    result = convert_layer_flat(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        device=args.device,
        execute=args.execute,
        max_relative_rmse=args.max_relative_rmse,
        min_cosine_similarity=args.min_cosine_similarity,
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
