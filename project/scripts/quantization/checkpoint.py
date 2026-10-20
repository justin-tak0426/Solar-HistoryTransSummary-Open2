#!/usr/bin/env python3
"""Header-only safetensors accounting for the official Solar checkpoint."""

from __future__ import annotations

import json
import os
import pwd
import re
import struct
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .contract import (
    ARCHITECTURE,
    BASE_MODEL_ID,
    BASE_REVISION,
    DEFAULT_ARTIFACT_ROOT,
    EXPECTED_SOURCE_SHARD_COUNT,
    EXPECTED_SOURCE_INDEX_SHA256,
    EXPECTED_SOURCE_TENSOR_COUNT,
    EXPECTED_SOURCE_TOTAL_SIZE,
    INDEX_FILENAME,
    NATIVE_PACKED_EXPERT_RE,
    SOURCE_EXPERT_RE,
    ArchitectureContract,
    canonical_json_sha256,
    native_shape,
    output_shape,
    sha256_file,
)


DTYPE_BYTES: dict[str, int] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
HEX_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
SOURCE_SHARD_RE = re.compile(r"^model-(\d{5})-of-(\d{5})\.safetensors$")
CONVERSION_SOURCE_PROFILE = "conversion_mk03"
B200_RUNTIME_SOURCE_PROFILE = "b200_owned_runtime"
_SOURCE_PROFILES = {
    CONVERSION_SOURCE_PROFILE,
    B200_RUNTIME_SOURCE_PROFILE,
}


@dataclass(frozen=True)
class TensorRecord:
    key: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    numel: int
    nbytes: int
    data_start: int
    data_end: int


@dataclass(frozen=True)
class ShardRecord:
    filename: str
    file_bytes: int
    payload_bytes: int
    tensor_count: int
    sha256: str | None


@dataclass
class SourceInventory:
    source_dir: Path
    index_sha256: str
    config_sha256: str
    index_metadata: dict[str, Any]
    tensors: dict[str, TensorRecord]
    shards: dict[str, ShardRecord]
    weight_map_sha256: str

    @property
    def tensor_count(self) -> int:
        return len(self.tensors)

    @property
    def stored_elements(self) -> int:
        return sum(record.numel for record in self.tensors.values())

    @property
    def stored_tensor_bytes(self) -> int:
        return sum(record.nbytes for record in self.tensors.values())

    def summary(self) -> dict[str, Any]:
        dtype_counts = Counter(record.dtype for record in self.tensors.values())
        dtype_bytes = Counter()
        for record in self.tensors.values():
            dtype_bytes[record.dtype] += record.nbytes
        routed = [
            record
            for key, record in self.tensors.items()
            if SOURCE_EXPERT_RE.fullmatch(key)
        ]
        return {
            "source_dir": str(self.source_dir),
            "index_sha256": self.index_sha256,
            "config_sha256": self.config_sha256,
            "weight_map_sha256": self.weight_map_sha256,
            "tensor_count": self.tensor_count,
            "stored_elements": self.stored_elements,
            "stored_tensor_bytes": self.stored_tensor_bytes,
            "shard_count": len(self.shards),
            "shard_file_bytes": sum(item.file_bytes for item in self.shards.values()),
            "shards": [
                asdict(self.shards[name])
                for name in sorted(self.shards)
            ],
            "dtype_tensor_counts": dict(sorted(dtype_counts.items())),
            "dtype_bytes": dict(sorted(dtype_bytes.items())),
            "routed_source_layout": "per_expert_gate_up_down",
            "routed_source_tensor_count": len(routed),
            "routed_native_packed_tensor_count_equivalent": (
                ARCHITECTURE.native_tensor_count
            ),
            "routed_parameter_elements": sum(item.numel for item in routed),
            "index_metadata": self.index_metadata,
        }


def _checked_numel(shape: Iterable[int]) -> int:
    result = 1
    for dimension in shape:
        if not isinstance(dimension, int) or dimension < 0:
            raise RuntimeError(f"invalid tensor shape: {tuple(shape)!r}")
        result *= dimension
    return result


def read_safetensors_header(path: Path) -> dict[str, TensorRecord]:
    """Parse and validate a safetensors header without materializing tensors."""

    file_size = path.stat().st_size
    if file_size < 8:
        raise RuntimeError(f"safetensors file is shorter than its header: {path}")
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise RuntimeError(f"cannot read safetensors header length: {path}")
        (header_length,) = struct.unpack("<Q", prefix)
        if header_length <= 1 or header_length > 256 * 1024 * 1024:
            raise RuntimeError(
                f"implausible safetensors header length {header_length}: {path}"
            )
        if 8 + header_length > file_size:
            raise RuntimeError(f"truncated safetensors header: {path}")
        raw_header = handle.read(header_length)
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(
                    f"duplicate safetensors header key {key!r}: {path}"
                )
            result[key] = value
        return result

    try:
        header = json.loads(
            raw_header.decode("utf-8").rstrip(" "),
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid safetensors JSON header: {path}") from error
    if not isinstance(header, dict):
        raise RuntimeError(f"safetensors header is not an object: {path}")

    payload_bytes = file_size - 8 - header_length
    records: dict[str, TensorRecord] = {}
    intervals: list[tuple[int, int, str]] = []
    for key, value in header.items():
        if key == "__metadata__":
            if value is not None and not isinstance(value, dict):
                raise RuntimeError(f"invalid __metadata__ in {path}")
            continue
        if not isinstance(key, str) or not key:
            raise RuntimeError(f"invalid tensor key in {path}: {key!r}")
        if not isinstance(value, dict):
            raise RuntimeError(f"invalid tensor descriptor for {key} in {path}")
        if set(value) != {"dtype", "shape", "data_offsets"}:
            raise RuntimeError(
                f"unexpected descriptor fields for {key} in {path}: "
                f"{sorted(value)}"
            )
        dtype = value["dtype"]
        shape_value = value["shape"]
        offsets = value["data_offsets"]
        if dtype not in DTYPE_BYTES:
            raise RuntimeError(f"unsupported safetensors dtype {dtype!r}: {key}")
        if not isinstance(shape_value, list):
            raise RuntimeError(f"shape is not a list: {key}")
        shape = tuple(shape_value)
        numel = _checked_numel(shape)
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(item, int) for item in offsets)
        ):
            raise RuntimeError(f"invalid data_offsets: {key}")
        start, end = offsets
        if start < 0 or end < start or end > payload_bytes:
            raise RuntimeError(f"out-of-range data_offsets for {key}: {offsets}")
        nbytes = end - start
        expected_bytes = numel * DTYPE_BYTES[dtype]
        if nbytes != expected_bytes:
            raise RuntimeError(
                f"tensor byte count mismatch for {key}: "
                f"header={nbytes} shape_dtype={expected_bytes}"
            )
        records[key] = TensorRecord(
            key=key,
            shard=path.name,
            dtype=dtype,
            shape=shape,
            numel=numel,
            nbytes=nbytes,
            data_start=start,
            data_end=end,
        )
        intervals.append((start, end, key))

    cursor = 0
    for start, end, key in sorted(intervals):
        if start != cursor:
            raise RuntimeError(
                f"safetensors payload has a gap/overlap before {key}: "
                f"expected={cursor} actual={start}"
            )
        cursor = end
    if cursor != payload_bytes:
        raise RuntimeError(
            f"unaccounted safetensors payload bytes: file={path} "
            f"accounted={cursor} payload={payload_bytes}"
        )
    return records


def _read_object(path: Path) -> dict[str, Any]:
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
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def validate_pinned_index_key_contract(
    index_path: Path,
    *,
    contract: ArchitectureContract = ARCHITECTURE,
) -> dict[str, Any]:
    """Validate the small official index without requiring any weight shard."""

    index = _read_object(index_path)
    digest = sha256_file(index_path)
    if digest != EXPECTED_SOURCE_INDEX_SHA256:
        raise RuntimeError(
            f"pinned source index digest changed: {digest} != "
            f"{EXPECTED_SOURCE_INDEX_SHA256}"
        )
    if set(index) != {"metadata", "weight_map"}:
        raise RuntimeError(f"unexpected source index fields: {sorted(index)}")
    metadata, weight_map = index["metadata"], index["weight_map"]
    if not isinstance(metadata, dict) or not isinstance(weight_map, dict):
        raise RuntimeError("malformed pinned source index")
    if metadata.get("total_size") != EXPECTED_SOURCE_TOTAL_SIZE:
        raise RuntimeError("pinned source index total_size changed")
    if len(weight_map) != EXPECTED_SOURCE_TENSOR_COUNT:
        raise RuntimeError("pinned source index tensor count changed")
    shard_names = set(weight_map.values())
    if len(shard_names) != EXPECTED_SOURCE_SHARD_COUNT:
        raise RuntimeError("pinned source index shard count changed")
    shard_numbers = set()
    for shard in shard_names:
        match = SOURCE_SHARD_RE.fullmatch(shard)
        if match is None:
            raise RuntimeError(f"unsafe pinned source shard name: {shard!r}")
        number, total = (int(value) for value in match.groups())
        if total != EXPECTED_SOURCE_SHARD_COUNT:
            raise RuntimeError(f"wrong total in pinned shard name: {shard}")
        shard_numbers.add(number)
    if shard_numbers != set(range(1, EXPECTED_SOURCE_SHARD_COUNT + 1)):
        raise RuntimeError("pinned source shard sequence changed")
    routed: set[tuple[int, int, str]] = set()
    for key in weight_map:
        if NATIVE_PACKED_EXPERT_RE.fullmatch(key):
            raise RuntimeError(f"pinned source unexpectedly uses packed key: {key}")
        match = SOURCE_EXPERT_RE.fullmatch(key)
        if match:
            routed.add(
                (
                    int(match.group("layer")),
                    int(match.group("expert")),
                    match.group("projection"),
                )
            )
    expected_routed = {
        (layer, expert, projection)
        for layer in range(contract.num_layers)
        for expert in range(contract.num_experts)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    if routed != expected_routed:
        raise RuntimeError(
            "pinned index routed coverage changed: "
            f"missing={sorted(expected_routed - routed)[:8]} "
            f"unexpected={sorted(routed - expected_routed)[:8]}"
        )
    for key in ("model.embed_tokens.weight", "lm_head.weight"):
        if key not in weight_map:
            raise RuntimeError(f"pinned source index lacks {key}")
    if (
        weight_map["model.embed_tokens.weight"]
        == weight_map["lm_head.weight"]
    ):
        raise RuntimeError(
            "pinned source embed/lm_head unexpectedly share a shard; "
            "re-audit tie_word_embeddings=false"
        )
    return {
        "status": "PASS",
        "index_sha256": digest,
        "tensor_count": len(weight_map),
        "shard_count": len(shard_names),
        "routed_source_tensor_count": len(routed),
        "routed_source_layout": "per_expert_gate_up_down",
        "embed_shard": weight_map["model.embed_tokens.weight"],
        "lm_head_shard": weight_map["lm_head.weight"],
        "total_size": metadata["total_size"],
    }


def validate_revision_provenance(
    source_dir: Path,
    *,
    expected_revision: str = BASE_REVISION,
) -> dict[str, str]:
    """Require a Hugging Face snapshot path pinned to the official commit."""

    if not HEX_REVISION_RE.fullmatch(expected_revision):
        raise ValueError(f"revision is not a 40-character commit: {expected_revision}")
    resolved = source_dir.expanduser().resolve()
    parts = resolved.parts
    observed: str | None = None
    for position, part in enumerate(parts[:-1]):
        if part == "snapshots":
            observed = parts[position + 1]
    if observed != expected_revision:
        raise RuntimeError(
            "source revision is not cryptographically anchored by its Hugging "
            f"Face snapshot path: expected snapshots/{expected_revision}, "
            f"resolved={resolved}"
        )
    return {
        "model_id": BASE_MODEL_ID,
        "expected_revision": expected_revision,
        "snapshot_revision": observed,
        "resolved_source_dir": str(resolved),
    }


def _source_shard_access_policy(
    source_dir: Path,
    *,
    runtime_profile: str,
) -> tuple[Path, int | None]:
    """Select path/ownership rules without leaking mk03 into B200 runtime.

    Conversion keeps the original mk03 cache and ownership contract. The B200
    profile instead anchors symlinks to the relocated snapshot's sibling blob
    directory and requires targets to be owned by the effective runtime uid.
    """

    if runtime_profile not in _SOURCE_PROFILES:
        raise ValueError(
            f"unknown source checkpoint runtime profile: {runtime_profile!r}"
        )
    if runtime_profile == CONVERSION_SOURCE_PROFILE:
        mk03_uid = pwd.getpwnam("mk03").pw_uid
        allowed_blob_root = (
            DEFAULT_ARTIFACT_ROOT
            / "hf_cache/models--upstage--Solar-Open2-250B/blobs"
        ).resolve(strict=False)
        # Preserve the source converter's historical behavior. Its entry point
        # separately requires mk03; when it is mk03, every target must be too.
        expected_owner_uid = mk03_uid if os.geteuid() == mk03_uid else None
        return allowed_blob_root, expected_owner_uid

    if source_dir.parent.name != "snapshots":
        raise RuntimeError(
            "B200 source checkpoint must remain under the pinned snapshots tree"
        )
    allowed_blob_root = (source_dir.parent.parent / "blobs").resolve(
        strict=True
    )
    return allowed_blob_root, os.geteuid()


def _validate_config(
    config: dict[str, Any],
    contract: ArchitectureContract,
) -> None:
    exact = {
        "model_type": "solar_open2",
        "hidden_size": contract.hidden_size,
        "num_hidden_layers": contract.num_layers,
        # ``num_local_experts`` is only a Transformers attribute alias.  The
        # official serialized config uses the canonical field below.
        "n_routed_experts": contract.num_experts,
        "moe_intermediate_size": contract.intermediate_size,
    }
    for key, expected in exact.items():
        actual = config.get(key)
        if actual != expected:
            raise RuntimeError(
                f"official config mismatch: {key}={actual!r}, expected={expected!r}"
            )
    architectures = config.get("architectures")
    if architectures != ["SolarOpen2ForCausalLM"]:
        raise RuntimeError(
            f"unexpected Solar architecture declaration: {architectures!r}"
        )
    dtype = config.get("dtype", config.get("torch_dtype"))
    if dtype not in {"bfloat16", "bf16"}:
        raise RuntimeError(f"official source dtype is not bfloat16: {dtype!r}")
    if config.get("tie_word_embeddings") is not False:
        raise RuntimeError(
            "official source requires distinct embed/lm_head weights: "
            f"tie_word_embeddings={config.get('tie_word_embeddings')!r}"
        )


def inventory_source_checkpoint(
    source_dir: Path,
    *,
    contract: ArchitectureContract = ARCHITECTURE,
    hash_shards: bool = False,
    require_revision_path: bool = True,
    runtime_profile: str = CONVERSION_SOURCE_PROFILE,
) -> SourceInventory:
    """Fully account every indexed source tensor using only shard headers."""

    contract.validate()
    source_dir = source_dir.expanduser().resolve()
    if require_revision_path:
        validate_revision_provenance(source_dir)
    allowed_blob_root, expected_owner_uid = _source_shard_access_policy(
        source_dir,
        runtime_profile=runtime_profile,
    )
    index_path = source_dir / INDEX_FILENAME
    config_path = source_dir / "config.json"
    if not index_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            f"source checkpoint lacks {INDEX_FILENAME} or config.json: {source_dir}"
        )
    index = _read_object(index_path)
    index_sha256 = sha256_file(index_path)
    if index_sha256 != EXPECTED_SOURCE_INDEX_SHA256:
        raise RuntimeError(
            f"pinned source index digest changed: {index_sha256} != "
            f"{EXPECTED_SOURCE_INDEX_SHA256}"
        )
    if set(index) != {"metadata", "weight_map"}:
        raise RuntimeError(f"unexpected source index fields: {sorted(index)}")
    metadata = index["metadata"]
    weight_map = index["weight_map"]
    if not isinstance(metadata, dict) or not isinstance(weight_map, dict):
        raise RuntimeError("invalid source safetensors index")
    if not weight_map or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in weight_map.items()
    ):
        raise RuntimeError("source weight_map is empty or malformed")
    if len(weight_map) != len(set(weight_map)):
        raise RuntimeError("duplicate source weight_map keys")

    config = _read_object(config_path)
    _validate_config(config, contract)
    shard_names = sorted(set(weight_map.values()))
    if len(shard_names) != EXPECTED_SOURCE_SHARD_COUNT:
        raise RuntimeError(
            f"official source shard count changed: {len(shard_names)} != "
            f"{EXPECTED_SOURCE_SHARD_COUNT}"
        )
    shard_numbers: set[int] = set()
    for name in shard_names:
        match = SOURCE_SHARD_RE.fullmatch(name)
        if match is None:
            raise RuntimeError(f"unsafe/non-safetensors shard name: {name!r}")
        number, total = (int(value) for value in match.groups())
        if total != EXPECTED_SOURCE_SHARD_COUNT:
            raise RuntimeError(f"wrong shard total encoded in {name}")
        shard_numbers.add(number)
    if shard_numbers != set(range(1, EXPECTED_SOURCE_SHARD_COUNT + 1)):
        raise RuntimeError("official source shard sequence is not consecutive")

    indexed_by_shard: dict[str, set[str]] = defaultdict(set)
    for key, shard in weight_map.items():
        indexed_by_shard[shard].add(key)

    tensors: dict[str, TensorRecord] = {}
    shards: dict[str, ShardRecord] = {}
    for shard_name in shard_names:
        shard_path = source_dir / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(shard_path)
        resolved_shard = shard_path.resolve()
        if shard_path.is_symlink():
            if allowed_blob_root not in resolved_shard.parents:
                raise RuntimeError(
                    f"snapshot shard symlink escapes the pinned model cache: "
                    f"{shard_path} -> {resolved_shard}"
                )
        elif source_dir not in resolved_shard.parents:
            raise RuntimeError(f"source shard escapes snapshot root: {shard_path}")
        if not resolved_shard.is_file():
            raise RuntimeError(f"source shard target is not regular: {resolved_shard}")
        if (
            expected_owner_uid is not None
            and resolved_shard.stat().st_uid != expected_owner_uid
        ):
            raise PermissionError(
                "source shard target ownership violates profile "
                f"{runtime_profile!r}: {resolved_shard}"
            )
        records = read_safetensors_header(shard_path)
        actual_keys = set(records)
        expected_keys = indexed_by_shard[shard_name]
        if actual_keys != expected_keys:
            raise RuntimeError(
                f"index/header key mismatch for {shard_name}: "
                f"missing={sorted(expected_keys - actual_keys)[:8]} "
                f"unexpected={sorted(actual_keys - expected_keys)[:8]}"
            )
        for key, record in records.items():
            if key in tensors:
                raise RuntimeError(f"tensor appears in multiple shards: {key}")
            tensors[key] = record
        file_bytes = shard_path.stat().st_size
        shards[shard_name] = ShardRecord(
            filename=shard_name,
            file_bytes=file_bytes,
            payload_bytes=sum(record.nbytes for record in records.values()),
            tensor_count=len(records),
            sha256=sha256_file(shard_path) if hash_shards else None,
        )

    if set(tensors) != set(weight_map):
        raise RuntimeError("source checkpoint accounting did not cover weight_map")
    if len(tensors) != EXPECTED_SOURCE_TENSOR_COUNT:
        raise RuntimeError(
            f"official source tensor count changed: {len(tensors)} != "
            f"{EXPECTED_SOURCE_TENSOR_COUNT}"
        )
    dtype_errors = []
    for key, record in tensors.items():
        expected_dtype = (
            "F32"
            if key.endswith(".mlp.gate.e_score_correction_bias")
            else "BF16"
        )
        if record.dtype != expected_dtype:
            dtype_errors.append((key, record.dtype, expected_dtype))
    if dtype_errors:
        raise RuntimeError(
            f"official source tensor dtype contract changed: {dtype_errors[:8]}"
        )
    extra_model_shards = {
        path.name for path in source_dir.glob("model-*.safetensors")
    } - set(shard_names)
    if extra_model_shards:
        raise RuntimeError(
            f"unindexed model shards present: {sorted(extra_model_shards)[:8]}"
        )

    packed_source_keys = [
        key for key in tensors if NATIVE_PACKED_EXPERT_RE.fullmatch(key)
    ]
    if packed_source_keys:
        raise RuntimeError(
            "official source unexpectedly contains native packed experts: "
            f"{packed_source_keys[:8]}"
        )
    routed_records: list[TensorRecord] = []
    observed_source: set[tuple[int, int, str]] = set()
    for key, record in tensors.items():
        match = SOURCE_EXPERT_RE.fullmatch(key)
        if not match:
            continue
        layer = int(match.group("layer"))
        expert = int(match.group("expert"))
        projection = match.group("projection")
        if not 0 <= layer < contract.num_layers:
            raise RuntimeError(f"source routed-expert layer out of range: {key}")
        if not 0 <= expert < contract.num_experts:
            raise RuntimeError(f"source routed-expert index out of range: {key}")
        identity = (layer, expert, projection)
        if identity in observed_source:
            raise RuntimeError(f"duplicate source routed-expert tensor: {key}")
        observed_source.add(identity)
        expected_shape = output_shape(projection, contract)
        if record.shape != expected_shape:
            raise RuntimeError(
                f"source routed-expert shape mismatch: {key} "
                f"actual={record.shape} expected={expected_shape}"
            )
        if record.dtype != "BF16":
            raise RuntimeError(
                f"source routed-expert tensor is not BF16: {key}={record.dtype}"
            )
        routed_records.append(record)
    expected_source = {
        (layer, expert, projection)
        for layer in range(contract.num_layers)
        for expert in range(contract.num_experts)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    if observed_source != expected_source:
        raise RuntimeError(
            "source routed-expert coverage mismatch: "
            f"missing={sorted(expected_source - observed_source)[:8]} "
            f"unexpected={sorted(observed_source - expected_source)[:8]}"
        )
    routed_elements = sum(record.numel for record in routed_records)
    if routed_elements != contract.routed_parameter_elements:
        raise RuntimeError(
            f"routed parameter accounting mismatch: {routed_elements} != "
            f"{contract.routed_parameter_elements}"
        )
    total_size = metadata.get("total_size")
    stored_tensor_bytes = sum(record.nbytes for record in tensors.values())
    if not isinstance(total_size, int) or total_size != stored_tensor_bytes:
        raise RuntimeError(
            f"index metadata.total_size mismatch: {total_size!r} != "
            f"{stored_tensor_bytes}"
        )
    if total_size != EXPECTED_SOURCE_TOTAL_SIZE:
        raise RuntimeError(
            f"pinned source total_size changed: {total_size} != "
            f"{EXPECTED_SOURCE_TOTAL_SIZE}"
        )

    return SourceInventory(
        source_dir=source_dir,
        index_sha256=index_sha256,
        config_sha256=sha256_file(config_path),
        index_metadata=metadata,
        tensors=tensors,
        shards=shards,
        weight_map_sha256=canonical_json_sha256(weight_map),
    )


def validate_transformers_schema(
    inventory: SourceInventory,
    *,
    contract: ArchitectureContract = ARCHITECTURE,
) -> dict[str, Any]:
    """Compare every checkpoint key/shape to an official-fork meta model.

    This is deliberately strict.  A missing or incompatible official
    Transformers fork blocks production conversion instead of silently
    accepting an inferred schema.
    """

    try:
        import torch
        import transformers
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as error:
        raise RuntimeError(
            "official Transformers/Accelerate environment is required for "
            "full model schema accounting"
        ) from error

    version = str(getattr(transformers, "__version__", "unknown"))
    if not version.startswith("5.14.1"):
        raise RuntimeError(
            "production conversion requires the Upstage "
            f"v5.14.1-solar-open2 fork, found transformers={version}"
        )
    config = AutoConfig.from_pretrained(
        inventory.source_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    if config.__class__.__name__ != "SolarOpen2Config":
        raise RuntimeError(f"unexpected config class: {config.__class__.__name__}")
    with init_empty_weights(include_buffers=True):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
    if model.__class__.__name__ != "SolarOpen2ForCausalLM":
        raise RuntimeError(f"unexpected model class: {model.__class__.__name__}")

    state = model.state_dict()
    native_expected_shapes = {
        key: tuple(value.shape) for key, value in state.items()
    }
    expected_shapes: dict[str, tuple[int, ...]] = {}
    for key, shape in native_expected_shapes.items():
        match = NATIVE_PACKED_EXPERT_RE.fullmatch(key)
        if match is None:
            expected_shapes[key] = shape
            continue
        layer = int(match.group("layer"))
        projection = match.group("projection")
        if projection == "gate_up_proj":
            projections = ("gate_proj", "up_proj")
        else:
            projections = ("down_proj",)
        for expert in range(contract.num_experts):
            for source_projection in projections:
                source_key = (
                    f"model.layers.{layer}.mlp.experts.{expert}."
                    f"{source_projection}.weight"
                )
                if source_key in expected_shapes:
                    raise RuntimeError(
                        f"translated source schema collision: {source_key}"
                    )
                expected_shapes[source_key] = output_shape(
                    source_projection, contract
                )
    actual_keys = set(inventory.tensors)
    expected_keys = set(expected_shapes)
    missing = expected_keys - actual_keys
    unexpected = actual_keys - expected_keys
    if missing or unexpected:
        raise RuntimeError(
            "official model/checkpoint key mismatch: "
            f"missing={sorted(missing)[:8]} "
            f"unexpected={sorted(unexpected)[:8]}"
        )
    shape_errors = []
    for key in sorted(actual_keys & expected_keys):
        actual = inventory.tensors[key].shape
        expected = expected_shapes[key]
        if actual != expected:
            shape_errors.append((key, actual, expected))
    if shape_errors:
        raise RuntimeError(f"official model shape mismatch: {shape_errors[:4]}")

    parameter_entries = list(model.named_parameters(remove_duplicate=False))
    raw_parameter_elements = sum(parameter.numel() for _, parameter in parameter_entries)
    logical_parameter_elements = raw_parameter_elements
    if (
        contract.total_parameters is not None
        and logical_parameter_elements != contract.total_parameters
    ):
        raise RuntimeError(
            "official meta-model parameter count mismatch: "
            f"{logical_parameter_elements} != {contract.total_parameters}"
        )

    for layer in range(contract.num_layers):
        experts = model.model.layers[layer].mlp.experts
        if experts.__class__.__name__ != "SolarOpen2Experts":
            raise RuntimeError(
                f"layer {layer} does not use native SolarOpen2Experts: "
                f"{experts.__class__.__name__}"
            )
        if tuple(experts.gate_up_proj.shape) != native_shape(
            "gate_up_proj", contract
        ):
            raise RuntimeError(f"layer {layer} gate_up_proj schema changed")
        if tuple(experts.down_proj.shape) != native_shape("down_proj", contract):
            raise RuntimeError(f"layer {layer} down_proj schema changed")

    del state, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "status": "PASS",
        "transformers_version": version,
        "model_class": "SolarOpen2ForCausalLM",
        "config_class": "SolarOpen2Config",
        "checkpoint_tensor_count": inventory.tensor_count,
        "meta_native_state_tensor_count": len(native_expected_shapes),
        "translated_source_tensor_count": len(expected_shapes),
        "source_expert_layout": "per_expert_gate_up_down",
        "tie_word_embeddings": False,
        "logical_parameter_elements": logical_parameter_elements,
        "schema_shapes_sha256": canonical_json_sha256(expected_shapes),
    }


def record_to_dict(record: TensorRecord | ShardRecord) -> dict[str, Any]:
    return asdict(record)
