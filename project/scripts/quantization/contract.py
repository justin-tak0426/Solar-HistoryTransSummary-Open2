#!/usr/bin/env python3
"""Dependency-light contracts shared by the Solar routed-expert converter."""

from __future__ import annotations

import hashlib
import json
import os
import pwd
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_NAME = "solar-open2-routed-experts-nf4"
SCHEMA_VERSION = 1
LAYER_FLAT_SCHEMA_NAME = "solar-open2-layer-flat-routed-experts-nf4"
BASE_MODEL_ID = "upstage/Solar-Open2-250B"
BASE_REVISION = "8d1dda5ad2af58207c65348d825772d2f4d72481"
SUPPORTED_BNB_VERSION = "0.49.2"
EXPECTED_SOURCE_TENSOR_COUNT = 47_019
EXPECTED_SOURCE_SHARD_COUNT = 94
EXPECTED_SOURCE_INDEX_SHA256 = (
    "2ba6aa72c31d2d758f747a546df3d671b5e1d5f1fa82985e789117b9cb148430"
)
EXPECTED_SOURCE_TOTAL_SIZE = 500_575_651_328
LAYER_FLAT_EXPERTS_PER_CHUNK = 64

DEFAULT_ARTIFACT_ROOT = Path("/data1/jhtak/Solar-HistoryTransSummary-Open2")
DEFAULT_SOURCE_DIR = (
    DEFAULT_ARTIFACT_ROOT
    / "hf_cache/models--upstage--Solar-Open2-250B/snapshots"
    / BASE_REVISION
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_ARTIFACT_ROOT / "models/Solar-Open2-250B-NF4-layer-flat-experts"
)
DEFAULT_PER_EXPERT_OUTPUT_DIR = (
    DEFAULT_ARTIFACT_ROOT / "models/Solar-Open2-250B-NF4-per-expert-fallback"
)

INDEX_FILENAME = "model.safetensors.index.json"
EXPERT_INDEX_FILENAME = "expert_nf4.safetensors.index.json"
MANIFEST_FILENAME = "expert_nf4_manifest.json"
PROGRESS_FILENAME = "_expert_nf4_progress.json"
OWNER_FILENAME = "_solar_nf4_owner.json"

NATIVE_PACKED_EXPERT_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<projection>gate_up_proj|down_proj)$"
)
SOURCE_EXPERT_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)
OUTPUT_WEIGHT_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)

PACKED_STATE_SUFFIXES = frozenset(
    {
        "",
        ".absmax",
        ".quant_map",
        ".nested_absmax",
        ".nested_quant_map",
        ".quant_state.bitsandbytes__nf4",
    }
)


@dataclass(frozen=True)
class ArchitectureContract:
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_experts: int
    total_parameters: int | None = None

    def validate(self) -> None:
        for name in (
            "hidden_size",
            "intermediate_size",
            "num_layers",
            "num_experts",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.total_parameters is not None and self.total_parameters <= 0:
            raise ValueError("total_parameters must be positive when provided")

    @property
    def projection_elements(self) -> int:
        return self.hidden_size * self.intermediate_size

    @property
    def routed_parameter_elements(self) -> int:
        return (
            self.num_layers
            * self.num_experts
            * 3
            * self.projection_elements
        )

    @property
    def native_tensor_count(self) -> int:
        return self.num_layers * 2

    @property
    def output_linear_count(self) -> int:
        return self.num_layers * self.num_experts * 3

    @property
    def source_expert_tensor_count(self) -> int:
        return self.output_linear_count

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ARCHITECTURE = ArchitectureContract(
    hidden_size=4096,
    intermediate_size=1280,
    num_layers=48,
    num_experts=320,
    total_parameters=250_287_794_944,
)


def native_gate_up_key(
    layer: int,
    contract: ArchitectureContract = ARCHITECTURE,
) -> str:
    require_layer(layer, contract)
    return f"model.layers.{layer}.mlp.experts.gate_up_proj"


def native_down_key(
    layer: int,
    contract: ArchitectureContract = ARCHITECTURE,
) -> str:
    require_layer(layer, contract)
    return f"model.layers.{layer}.mlp.experts.down_proj"


def output_weight_key(layer: int, expert: int, projection: str) -> str:
    require_layer(layer)
    require_expert(expert)
    if projection not in {"gate_proj", "up_proj", "down_proj"}:
        raise ValueError(f"unsupported projection: {projection}")
    return (
        f"model.layers.{layer}.mlp.experts.{expert}."
        f"{projection}.weight"
    )


def source_expert_key(layer: int, expert: int, projection: str) -> str:
    return output_weight_key(layer, expert, projection)


def layer_flat_weight_key(
    layer: int,
    projection: str,
    chunk: int = 0,
    contract: ArchitectureContract = ARCHITECTURE,
) -> str:
    require_layer(layer, contract)
    chunk_count = (
        contract.num_experts + LAYER_FLAT_EXPERTS_PER_CHUNK - 1
    ) // LAYER_FLAT_EXPERTS_PER_CHUNK
    if not isinstance(chunk, int) or not 0 <= chunk < chunk_count:
        raise ValueError(f"layer-flat chunk out of bounds: {chunk}")
    if projection == "gate_up_proj":
        base = native_gate_up_key(layer, contract)
    elif projection == "down_proj":
        base = native_down_key(layer, contract)
    else:
        raise ValueError(f"unsupported layer-flat projection: {projection}")
    return f"{base}.chunks.{chunk}.weight"


def layer_flat_chunk_ranges(
    contract: ArchitectureContract = ARCHITECTURE,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (
            start,
            min(start + LAYER_FLAT_EXPERTS_PER_CHUNK, contract.num_experts),
        )
        for start in range(
            0,
            contract.num_experts,
            LAYER_FLAT_EXPERTS_PER_CHUNK,
        )
    )


def native_shape(projection: str, contract: ArchitectureContract = ARCHITECTURE) -> tuple[int, ...]:
    if projection == "gate_up_proj":
        return (
            contract.num_experts,
            2 * contract.intermediate_size,
            contract.hidden_size,
        )
    if projection == "down_proj":
        return (
            contract.num_experts,
            contract.hidden_size,
            contract.intermediate_size,
        )
    raise ValueError(f"unsupported native projection: {projection}")


def output_shape(projection: str, contract: ArchitectureContract = ARCHITECTURE) -> tuple[int, int]:
    if projection in {"gate_proj", "up_proj"}:
        return contract.intermediate_size, contract.hidden_size
    if projection == "down_proj":
        return contract.hidden_size, contract.intermediate_size
    raise ValueError(f"unsupported output projection: {projection}")


def require_layer(layer: int, contract: ArchitectureContract = ARCHITECTURE) -> None:
    if not isinstance(layer, int) or not 0 <= layer < contract.num_layers:
        raise ValueError(f"layer out of bounds: {layer!r}")


def require_expert(expert: int, contract: ArchitectureContract = ARCHITECTURE) -> None:
    if not isinstance(expert, int) or not 0 <= expert < contract.num_experts:
        raise ValueError(f"expert out of bounds: {expert!r}")


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(
        path,
        (
            json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2)
            + "\n"
        ).encode("utf-8"),
    )


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def require_output_path(path: Path) -> Path:
    """Restrict production artifacts to the newly assigned project root."""

    mk03_uid = pwd.getpwnam("mk03").pw_uid
    if os.geteuid() != mk03_uid:
        raise PermissionError(
            f"converter must run as mk03 uid={mk03_uid}, found uid={os.geteuid()}"
        )
    root = DEFAULT_ARTIFACT_ROOT.resolve(strict=False)
    expanded = path.expanduser()
    absolute = expanded if expanded.is_absolute() else Path.cwd() / expanded
    # Resolve only after rejecting symlinks in every existing component.
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.exists() or current.is_symlink():
            stat_result = current.lstat()
            if current.is_symlink():
                raise RuntimeError(f"output path traverses a symlink: {current}")
            if current == Path("/data1"):
                continue
            if current == Path("/data1/jhtak") or (
                Path("/data1/jhtak") in current.parents
            ):
                if stat_result.st_uid != mk03_uid:
                    raise PermissionError(
                        f"refusing path not owned by mk03: {current} "
                        f"uid={stat_result.st_uid}"
                    )
    nearest = absolute
    while not nearest.exists():
        if nearest.parent == nearest:
            raise RuntimeError(f"no existing parent for output: {absolute}")
        nearest = nearest.parent
    if nearest != Path("/data1") and nearest.lstat().st_uid != mk03_uid:
        raise PermissionError(
            f"nearest output parent is not owned by mk03: {nearest}"
        )
    resolved = absolute.resolve(strict=False)
    if resolved == root or root not in resolved.parents:
        raise ValueError(
            f"output must be below {root}; refusing {resolved}"
        )
    return resolved


def validate_exact_keys(
    actual: Iterable[str],
    expected: Iterable[str],
    *,
    label: str,
) -> None:
    actual_set = set(actual)
    expected_set = set(expected)
    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)
    if missing or unexpected:
        raise RuntimeError(
            f"{label} keys differ: missing={missing[:8]} "
            f"unexpected={unexpected[:8]}"
        )


ARCHITECTURE.validate()
assert ARCHITECTURE.routed_parameter_elements == 241_591_910_400
assert ARCHITECTURE.native_tensor_count == 96
assert ARCHITECTURE.output_linear_count == 46_080
assert ARCHITECTURE.source_expert_tensor_count == 46_080
