#!/usr/bin/env python3
"""Stream official Solar routed experts into a sealed bnb NF4 sidecar.

Run as a module:

    /data/home/mk03/workspace/model_training/Solar-HistoryTransSummary-Open2/solar-open-2/bin/python \
        -m scripts.quantization.convert [--execute]

Without ``--execute`` this command performs read-only source accounting and
prints the deterministic conversion plan.
"""

from __future__ import annotations

import argparse
import getpass
import json
import math
import os
import platform
from contextlib import ExitStack
from dataclasses import asdict, dataclass
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
    DEFAULT_PER_EXPERT_OUTPUT_DIR,
    DEFAULT_SOURCE_DIR,
    EXPERT_INDEX_FILENAME,
    MANIFEST_FILENAME,
    OWNER_FILENAME,
    PACKED_STATE_SUFFIXES,
    PROGRESS_FILENAME,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    ArchitectureContract,
    atomic_json,
    canonical_json_sha256,
    fsync_directory,
    output_weight_key,
    require_output_path,
    sha256_file,
    source_expert_key,
)
from .output_checkpoint import (
    OUTPUT_SHARD_RE,
    preflight_output_checkpoint,
    read_json_no_duplicates,
)


@dataclass(frozen=True)
class ShardPlan:
    filename: str
    layer: int
    expert_start: int
    expert_end: int

    @property
    def expert_count(self) -> int:
        return self.expert_end - self.expert_start

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def make_plan(
    *,
    experts_per_shard: int,
    contract: ArchitectureContract = ARCHITECTURE,
) -> list[ShardPlan]:
    if not isinstance(experts_per_shard, int) or not (
        1 <= experts_per_shard <= contract.num_experts
    ):
        raise ValueError(
            f"experts_per_shard must be in [1,{contract.num_experts}]"
        )
    plan = []
    for layer in range(contract.num_layers):
        for start in range(0, contract.num_experts, experts_per_shard):
            end = min(start + experts_per_shard, contract.num_experts)
            plan.append(
                ShardPlan(
                    filename=(
                        f"experts-layer-{layer:03d}-"
                        f"e{start:04d}-e{end:04d}.safetensors"
                    ),
                    layer=layer,
                    expert_start=start,
                    expert_end=end,
                )
            )
    if len({item.filename for item in plan}) != len(plan):
        raise AssertionError("conversion plan contains duplicate filenames")
    return plan


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
    }


def _owner_payload(
    inventory: SourceInventory,
    plan: list[ShardPlan],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "owner_user": "mk03",
        "base_model_id": BASE_MODEL_ID,
        "base_revision": BASE_REVISION,
        "source_dir": str(inventory.source_dir),
        "source_index_sha256": inventory.index_sha256,
        "source_config_sha256": inventory.config_sha256,
        "architecture": ARCHITECTURE.to_dict(),
        "quantization": _quantization_contract(),
        "plan_sha256": canonical_json_sha256(
            [item.to_dict() for item in plan]
        ),
    }


def _initialize_output(
    output_dir: Path,
    *,
    owner: dict[str, Any],
    plan: list[ShardPlan],
) -> dict[str, Any]:
    output_dir = require_output_path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
        fsync_directory(output_dir.parent)
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise RuntimeError(f"output path is not a regular directory: {output_dir}")
    output_dir = require_output_path(output_dir)
    owner_path = output_dir / OWNER_FILENAME
    progress_path = output_dir / PROGRESS_FILENAME
    if owner_path.exists():
        existing_owner = read_json_no_duplicates(owner_path)
        if existing_owner != owner:
            raise RuntimeError(
                "existing output belongs to a different source/plan; refusing it"
            )
        for partial in output_dir.glob(".*.partial"):
            if partial.is_symlink() or not partial.is_file():
                raise RuntimeError(f"unsafe orphan conversion temporary: {partial}")
            partial.unlink()
        fsync_directory(output_dir)
    else:
        if any(output_dir.iterdir()):
            raise RuntimeError(
                f"refusing nonempty unowned output directory: {output_dir}"
            )
        atomic_json(owner_path, owner)

    if progress_path.exists():
        progress = read_json_no_duplicates(progress_path)
    else:
        progress = {
            "schema": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "status": "BUILDING",
            "owner_sha256": sha256_file(owner_path),
            "plan_sha256": owner["plan_sha256"],
            "completed": {},
        }
        atomic_json(progress_path, progress)
    expected_progress = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "BUILDING",
        "owner_sha256": sha256_file(owner_path),
        "plan_sha256": owner["plan_sha256"],
    }
    for key, expected in expected_progress.items():
        if progress.get(key) != expected:
            raise RuntimeError(
                f"progress contract mismatch: {key}={progress.get(key)!r}"
            )
    if not isinstance(progress.get("completed"), dict):
        raise RuntimeError("progress completed field is malformed")
    plan_names = {item.filename for item in plan}
    unknown_completed = set(progress["completed"]) - plan_names
    if unknown_completed:
        raise RuntimeError(
            f"progress lists unknown shards: {sorted(unknown_completed)[:8]}"
        )
    return progress


def _expected_shard_keys(
    item: ShardPlan,
    contract: ArchitectureContract = ARCHITECTURE,
) -> set[str]:
    result = set()
    for expert in range(item.expert_start, item.expert_end):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            base = output_weight_key(item.layer, expert, projection)
            result.update(base + suffix for suffix in PACKED_STATE_SUFFIXES)
    return result


def _safetensors_metadata(
    item: ShardPlan,
    owner: dict[str, Any],
    *,
    maximum_relative_rmse: float,
    minimum_cosine_similarity: float,
    maximum_quant_state_ratio: float,
) -> dict[str, str]:
    return {
        "schema": SCHEMA_NAME,
        "schema_version": str(SCHEMA_VERSION),
        "base_model_id": BASE_MODEL_ID,
        "base_revision": BASE_REVISION,
        "source_index_sha256": owner["source_index_sha256"],
        "plan_sha256": owner["plan_sha256"],
        "layer": str(item.layer),
        "expert_start": str(item.expert_start),
        "expert_end": str(item.expert_end),
        "quant_type": "nf4",
        "quant_storage": "bfloat16",
        "maximum_relative_rmse": format(maximum_relative_rmse, ".17g"),
        "minimum_cosine_similarity": format(
            minimum_cosine_similarity, ".17g"
        ),
        "maximum_quant_state_ratio": format(
            maximum_quant_state_ratio, ".17g"
        ),
    }


def _validate_shard_metadata(
    path: Path,
    item: ShardPlan,
    owner: dict[str, Any],
) -> tuple[float, float, float]:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("safetensors is required for conversion") from error
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    metric_keys = {
        "maximum_relative_rmse",
        "minimum_cosine_similarity",
        "maximum_quant_state_ratio",
    }
    if not isinstance(metadata, dict) or set(metadata) != {
        "schema",
        "schema_version",
        "base_model_id",
        "base_revision",
        "source_index_sha256",
        "plan_sha256",
        "layer",
        "expert_start",
        "expert_end",
        "quant_type",
        "quant_storage",
        *metric_keys,
    }:
        raise RuntimeError(f"NF4 shard provenance metadata changed: {path}")
    try:
        maximum_relative_rmse = float(metadata["maximum_relative_rmse"])
        minimum_cosine_similarity = float(
            metadata["minimum_cosine_similarity"]
        )
        maximum_quant_state_ratio = float(
            metadata["maximum_quant_state_ratio"]
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"NF4 shard metric metadata is invalid: {path}") from error
    metrics = (
        maximum_relative_rmse,
        minimum_cosine_similarity,
        maximum_quant_state_ratio,
    )
    if not all(math.isfinite(value) for value in metrics):
        raise RuntimeError(f"NF4 shard metric metadata is non-finite: {path}")
    expected_static = _safetensors_metadata(
        item,
        owner,
        maximum_relative_rmse=maximum_relative_rmse,
        minimum_cosine_similarity=minimum_cosine_similarity,
        maximum_quant_state_ratio=maximum_quant_state_ratio,
    )
    if metadata != expected_static:
        raise RuntimeError(f"NF4 shard provenance metadata changed: {path}")
    if not 0.0 <= maximum_relative_rmse <= 0.15:
        raise RuntimeError(f"NF4 shard RMSE metadata is out of range: {path}")
    if not 0.98 <= minimum_cosine_similarity <= 1.0:
        raise RuntimeError(f"NF4 shard cosine metadata is out of range: {path}")
    if not 0.0 < maximum_quant_state_ratio <= 0.10:
        raise RuntimeError(
            f"NF4 shard quant-state ratio metadata is out of range: {path}"
        )
    return (
        maximum_relative_rmse,
        minimum_cosine_similarity,
        maximum_quant_state_ratio,
    )


def _make_file_record(
    path: Path,
    item: ShardPlan,
    *,
    maximum_relative_rmse: float,
    minimum_cosine_similarity: float,
    maximum_quant_state_ratio: float,
) -> dict[str, Any]:
    tensors = read_safetensors_header(path)
    expected_keys = _expected_shard_keys(item)
    if set(tensors) != expected_keys:
        raise RuntimeError(
            f"written NF4 shard key mismatch: {path.name} "
            f"missing={sorted(expected_keys - set(tensors))[:8]} "
            f"unexpected={sorted(set(tensors) - expected_keys)[:8]}"
        )
    logical_elements = (
        item.expert_count * 3 * ARCHITECTURE.projection_elements
    )
    return {
        "filename": item.filename,
        "layer": item.layer,
        "expert_start": item.expert_start,
        "expert_end": item.expert_end,
        "file_bytes": path.stat().st_size,
        "payload_bytes": sum(record.nbytes for record in tensors.values()),
        "tensor_count": len(tensors),
        "logical_parameter_elements": logical_elements,
        "sha256": sha256_file(path),
        "maximum_relative_rmse": maximum_relative_rmse,
        "minimum_cosine_similarity": minimum_cosine_similarity,
        "maximum_quant_state_ratio": maximum_quant_state_ratio,
    }


def _adopt_or_validate_existing(
    path: Path,
    item: ShardPlan,
    owner: dict[str, Any],
    progress_record: dict[str, Any] | None,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"existing NF4 shard is missing/symlinked: {path}")
    (
        maximum_relative_rmse,
        minimum_cosine_similarity,
        maximum_quant_state_ratio,
    ) = _validate_shard_metadata(path, item, owner)
    rebuilt = _make_file_record(
        path,
        item,
        maximum_relative_rmse=maximum_relative_rmse,
        minimum_cosine_similarity=minimum_cosine_similarity,
        maximum_quant_state_ratio=maximum_quant_state_ratio,
    )
    if progress_record is not None:
        if set(progress_record) != {
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
        }:
            raise RuntimeError(f"progress shard record schema changed: {path.name}")
        if rebuilt != progress_record:
            raise RuntimeError(
                f"existing NF4 shard differs from progress receipt: {path}"
            )
        return progress_record
    # A crash can occur after atomic rename but before the progress fsync.
    # Metrics are sealed into safetensors metadata, so the complete progress
    # record can be reconstructed without requantizing or weakening bounds.
    return rebuilt


def _source_handles(
    inventory: SourceInventory,
    layer: int,
    expert_start: int,
    expert_end: int,
) -> tuple[ExitStack, dict[str, Any]]:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("safetensors is required for conversion") from error
    keys = {
        source_expert_key(layer, expert, projection)
        for expert in range(expert_start, expert_end)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    shard_names = {inventory.tensors[key].shard for key in keys}
    stack = ExitStack()
    handles = {}
    for shard_name in sorted(shard_names):
        handles[shard_name] = stack.enter_context(
            safe_open(
                inventory.source_dir / shard_name,
                framework="pt",
                device="cpu",
            )
        )
    return stack, handles


def _write_one_shard(
    output_dir: Path,
    item: ShardPlan,
    *,
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
        raise RuntimeError("torch and safetensors are required for conversion") from error

    tensors: dict[str, torch.Tensor] = {}
    maximum_rmse = 0.0
    minimum_cosine = 1.0
    maximum_state_ratio = 0.0
    stack, handles = _source_handles(
        inventory,
        item.layer,
        item.expert_start,
        item.expert_end,
    )
    try:
        for expert in range(item.expert_start, item.expert_end):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                source_key = source_expert_key(
                    item.layer, expert, projection
                )
                handle = handles[inventory.tensors[source_key].shard]
                source = handle.get_tensor(source_key)
                expected_shape = (
                    (
                        ARCHITECTURE.intermediate_size,
                        ARCHITECTURE.hidden_size,
                    )
                    if projection in {"gate_proj", "up_proj"}
                    else (
                        ARCHITECTURE.hidden_size,
                        ARCHITECTURE.intermediate_size,
                    )
                )
                if (
                    tuple(source.shape) != expected_shape
                    or source.dtype is not torch.bfloat16
                ):
                    raise RuntimeError(
                        f"source expert tensor contract changed: {source_key}"
                    )
                packed = quantize_nf4_matrix(
                    source.contiguous(),
                    device=device,
                    max_relative_rmse=max_relative_rmse,
                    min_cosine_similarity=min_cosine_similarity,
                )
                base = output_weight_key(
                    item.layer, expert, projection
                )
                for suffix, tensor in packed.tensors.items():
                    key = base + suffix
                    if key in tensors:
                        raise RuntimeError(f"output tensor collision: {key}")
                    tensors[key] = tensor
                metric = packed.metrics
                maximum_rmse = max(maximum_rmse, metric.relative_rmse)
                minimum_cosine = min(
                    minimum_cosine, metric.cosine_similarity
                )
                ratio = metric.quant_state_bytes / metric.packed_weight_bytes
                maximum_state_ratio = max(maximum_state_ratio, ratio)
                del source, packed
    finally:
        stack.close()

    expected_keys = _expected_shard_keys(item)
    if set(tensors) != expected_keys:
        raise RuntimeError("converter did not produce the planned shard keys")
    target = output_dir / item.filename
    temporary = output_dir / f".{item.filename}.{os.getpid()}.partial"
    if target.exists() or temporary.exists():
        raise FileExistsError(target if target.exists() else temporary)
    try:
        save_file(
            tensors,
            str(temporary),
            metadata=_safetensors_metadata(
                item,
                owner,
                maximum_relative_rmse=maximum_rmse,
                minimum_cosine_similarity=minimum_cosine,
                maximum_quant_state_ratio=maximum_state_ratio,
            ),
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        _validate_shard_metadata(temporary, item, owner)
        if set(read_safetensors_header(temporary)) != expected_keys:
            raise RuntimeError("temporary NF4 shard failed header verification")
        os.replace(temporary, target)
        fsync_directory(output_dir)
    finally:
        if temporary.exists():
            temporary.unlink()
    record = _make_file_record(
        target,
        item,
        maximum_relative_rmse=maximum_rmse,
        minimum_cosine_similarity=minimum_cosine,
        maximum_quant_state_ratio=maximum_state_ratio,
    )
    del tensors
    torch.cuda.empty_cache()
    return record


def _converter_source_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    names = (
        "__init__.py",
        "contract.py",
        "checkpoint.py",
        "bnb_nf4.py",
        "expert_module.py",
        "output_checkpoint.py",
        "convert.py",
    )
    return {name: sha256_file(directory / name) for name in names}


def _finalize(
    output_dir: Path,
    *,
    inventory: SourceInventory,
    schema_receipt: dict[str, Any],
    progress: dict[str, Any],
    plan: list[ShardPlan],
    owner: dict[str, Any],
) -> dict[str, Any]:
    completed = progress["completed"]
    plan_names = {item.filename for item in plan}
    if set(completed) != plan_names:
        raise RuntimeError(
            f"cannot seal incomplete conversion: "
            f"missing={sorted(plan_names - set(completed))[:8]}"
        )
    weight_map: dict[str, str] = {}
    files = []
    total_size = 0
    for item in plan:
        record = completed[item.filename]
        path = output_dir / item.filename
        validated = _adopt_or_validate_existing(
            path, item, owner, record
        )
        files.append(validated)
        tensors = read_safetensors_header(path)
        total_size += sum(tensor.nbytes for tensor in tensors.values())
        for key in sorted(tensors):
            if key in weight_map:
                raise RuntimeError(f"duplicate output tensor while sealing: {key}")
            weight_map[key] = item.filename

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(weight_map.items())),
    }
    index_path = output_dir / EXPERT_INDEX_FILENAME
    if index_path.exists():
        if read_json_no_duplicates(index_path) != index:
            raise RuntimeError("existing expert index differs from rebuilt index")
    else:
        atomic_json(index_path, index)

    source = inventory.summary()
    source["shards"] = [
        asdict(inventory.shards[name])
        for name in sorted(inventory.shards)
    ]
    manifest = {
        "schema": SCHEMA_NAME,
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
            "gate_proj": "source per-expert gate_proj weight identity",
            "up_proj": "source per-expert up_proj weight identity",
            "down_proj": "source per-expert down_proj weight identity",
            "transpose": False,
        },
        "plan_sha256": owner["plan_sha256"],
        "converter_source_sha256": _converter_source_hashes(),
        "files": files,
        "expert_index_sha256": sha256_file(index_path),
        "host": platform.node(),
        "created_by_user": getpass.getuser(),
        "logical_parameter_elements": ARCHITECTURE.routed_parameter_elements,
        "logical_linear_count": ARCHITECTURE.output_linear_count,
        "serialized_tensor_count": len(weight_map),
    }
    manifest_path = output_dir / MANIFEST_FILENAME
    if manifest_path.exists():
        existing = read_json_no_duplicates(manifest_path)
        # Host and creation identity are deterministic for a resumed run on the
        # same node.  A mismatch blocks rather than rewrites a sealed artifact.
        if existing != manifest:
            raise RuntimeError("existing sealed manifest differs from rebuilt one")
    else:
        # The sealed manifest is intentionally the final write.
        atomic_json(manifest_path, manifest)
    return preflight_output_checkpoint(
        output_dir,
        contract=ARCHITECTURE,
        hash_shards=True,
    )


def convert(
    *,
    source_dir: Path,
    output_dir: Path,
    device: str,
    experts_per_shard: int,
    execute: bool,
    max_relative_rmse: float,
    min_cosine_similarity: float,
) -> dict[str, Any]:
    if getpass.getuser() != "mk03":
        raise PermissionError("Solar NF4 conversion is restricted to mk03")
    plan = make_plan(experts_per_shard=experts_per_shard)
    inventory = inventory_source_checkpoint(
        source_dir,
        contract=ARCHITECTURE,
        hash_shards=execute,
        require_revision_path=True,
    )
    schema_receipt = validate_transformers_schema(
        inventory, contract=ARCHITECTURE
    )
    result: dict[str, Any] = {
        "status": "PLAN_ONLY" if not execute else "BUILDING",
        "source": inventory.summary(),
        "official_transformers_schema": schema_receipt,
        "output_dir": str(output_dir),
        "experts_per_shard": experts_per_shard,
        "output_shard_count": len(plan),
        "logical_linear_count": ARCHITECTURE.output_linear_count,
        "logical_parameter_elements": ARCHITECTURE.routed_parameter_elements,
        "minimum_packed_weight_bytes": (
            ARCHITECTURE.routed_parameter_elements // 2
        ),
        "plan_sha256": canonical_json_sha256(
            [item.to_dict() for item in plan]
        ),
        "writes_performed": False,
    }
    if not execute:
        return result

    require_bitsandbytes(require_cuda=True)
    output_dir = require_output_path(output_dir)
    owner = _owner_payload(inventory, plan)
    progress = _initialize_output(
        output_dir, owner=owner, plan=plan
    )
    progress_path = output_dir / PROGRESS_FILENAME
    allowed_names = {
        OWNER_FILENAME,
        PROGRESS_FILENAME,
        EXPERT_INDEX_FILENAME,
        MANIFEST_FILENAME,
        *[item.filename for item in plan],
    }
    unexpected = {
        path.name for path in output_dir.iterdir()
    } - allowed_names
    if unexpected:
        raise RuntimeError(
            f"unexpected files in converter-owned output: {sorted(unexpected)[:8]}"
        )
    if (output_dir / MANIFEST_FILENAME).exists():
        return preflight_output_checkpoint(output_dir, hash_shards=True)

    for position, item in enumerate(plan, 1):
        target = output_dir / item.filename
        existing_record = progress["completed"].get(item.filename)
        if target.exists():
            validated = _adopt_or_validate_existing(
                target, item, owner, existing_record
            )
            if existing_record is None:
                progress["completed"][item.filename] = validated
                atomic_json(progress_path, progress)
            continue
        if existing_record is not None:
            raise RuntimeError(
                f"progress records a missing output shard: {item.filename}"
            )
        record = _write_one_shard(
            output_dir,
            item,
            inventory=inventory,
            owner=owner,
            device=device,
            max_relative_rmse=max_relative_rmse,
            min_cosine_similarity=min_cosine_similarity,
        )
        progress["completed"][item.filename] = record
        atomic_json(progress_path, progress)
        print(
            json.dumps(
                {
                    "event": "NF4_SHARD_COMPLETE",
                    "position": position,
                    "total": len(plan),
                    **record,
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
            flush=True,
        )

    receipt = _finalize(
        output_dir,
        inventory=inventory,
        schema_receipt=schema_receipt,
        progress=progress,
        plan=plan,
        owner=owner,
    )
    receipt["writes_performed"] = True
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_PER_EXPERT_OUTPUT_DIR
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--experts-per-shard", type=int, default=16)
    parser.add_argument("--max-relative-rmse", type=float, default=0.15)
    parser.add_argument("--min-cosine-similarity", type=float, default=0.98)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="perform the irreversible-size conversion; otherwise plan only",
    )
    args = parser.parse_args()
    if not 0 < args.max_relative_rmse <= 0.15:
        parser.error("--max-relative-rmse must be in (0,0.15]")
    if not 0.98 <= args.min_cosine_similarity <= 1.0:
        parser.error("--min-cosine-similarity must be in [0.98,1.0]")
    result = convert(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        device=args.device,
        experts_per_shard=args.experts_per_shard,
        execute=args.execute,
        max_relative_rmse=args.max_relative_rmse,
        min_cosine_similarity=args.min_cosine_similarity,
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
