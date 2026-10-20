#!/usr/bin/env python3
"""Pinned bitsandbytes NF4 serialization and loading primitives."""

from __future__ import annotations

import importlib.metadata
import inspect
import math
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

try:
    from ..b200_runtime import require_sealed_nf4_runtime
except ImportError:  # Direct scripts/ entry points expose quantization top-level.
    from b200_runtime import require_sealed_nf4_runtime
from .contract import PACKED_STATE_SUFFIXES, SUPPORTED_BNB_VERSION


CONVERSION_RUNTIME_PROFILE = "conversion_sm120"
B200_SEALED_RUNTIME_PROFILE = "b200_sealed_artifact"
_RUNTIME_PROFILES = {
    CONVERSION_RUNTIME_PROFILE,
    B200_SEALED_RUNTIME_PROFILE,
}
_ACTIVE_RUNTIME_PROFILE: ContextVar[tuple[str, str | None]] = ContextVar(
    "solar_nf4_runtime_profile",
    default=(CONVERSION_RUNTIME_PROFILE, None),
)


@dataclass(frozen=True)
class QuantizationMetrics:
    relative_rmse: float
    cosine_similarity: float
    source_rms: float
    maximum_absolute_error: float
    packed_weight_bytes: int
    quant_state_bytes: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "relative_rmse": self.relative_rmse,
            "cosine_similarity": self.cosine_similarity,
            "source_rms": self.source_rms,
            "maximum_absolute_error": self.maximum_absolute_error,
            "packed_weight_bytes": self.packed_weight_bytes,
            "quant_state_bytes": self.quant_state_bytes,
        }


@dataclass
class PackedNF4:
    tensors: dict[str, Any]
    metrics: QuantizationMetrics


def validate_packed_record_contract(
    records: Mapping[str, Any],
    *,
    base: str,
    logical_elements: int,
) -> None:
    """Validate the fixed bnb 0.49.2 nested-NF4 companion tensor schema."""

    if not isinstance(logical_elements, int) or logical_elements <= 0:
        raise ValueError("logical_elements must be positive")
    expected_keys = {base + suffix for suffix in PACKED_STATE_SUFFIXES}
    actual = {key for key in records if key in expected_keys}
    if actual != expected_keys:
        raise RuntimeError(f"packed NF4 companion coverage changed: {base}")
    block_count = math.ceil(logical_elements / 64)
    nested_block_count = math.ceil(block_count / 256)
    expected = {
        "": ("BF16", None),
        ".absmax": ("U8", (block_count,)),
        ".quant_map": ("F32", (16,)),
        ".nested_absmax": ("F32", (nested_block_count,)),
        ".nested_quant_map": ("F32", (256,)),
    }
    for suffix, (dtype, shape) in expected.items():
        record = records[base + suffix]
        if record.dtype != dtype:
            raise RuntimeError(
                f"packed NF4 companion dtype changed: {base + suffix}"
            )
        if shape is not None and tuple(record.shape) != shape:
            raise RuntimeError(
                f"packed NF4 companion shape changed: {base + suffix}"
            )
    packed = records[base]
    if packed.nbytes != logical_elements // 2:
        raise RuntimeError(f"packed NF4 weight byte count changed: {base}")
    encoded = records[base + ".quant_state.bitsandbytes__nf4"]
    if (
        encoded.dtype != "U8"
        or len(encoded.shape) != 1
        or not 0 < encoded.numel <= 4096
    ):
        raise RuntimeError(f"packed NF4 JSON state changed: {base}")


def bitsandbytes_native_library_path(bnb: Any) -> Path:
    """Return bnb's actually loaded native library, rejecting CPU fallback."""

    cextension = getattr(bnb, "cextension", None)
    native = getattr(cextension, "lib", None)
    loaded_library = getattr(native, "_lib", None)
    name = getattr(loaded_library, "_name", None)
    if not isinstance(name, str) or not name:
        raise RuntimeError("bitsandbytes native library identity is unavailable")
    path = Path(name).expanduser().resolve(strict=True)
    if not path.is_file():
        raise RuntimeError(f"bitsandbytes native library is not regular: {path}")
    if not bool(getattr(native, "compiled_with_cuda", False)):
        raise RuntimeError(
            f"bitsandbytes loaded a CPU fallback library: {path}"
        )
    return path


@contextmanager
def use_nf4_runtime_profile(
    runtime_profile: str,
    *,
    manifest_status: str | None = None,
) -> Any:
    """Explicitly scope prequantized loads to one audited runtime profile."""

    if runtime_profile not in _RUNTIME_PROFILES:
        raise ValueError(f"unknown NF4 runtime profile: {runtime_profile!r}")
    if (
        runtime_profile == B200_SEALED_RUNTIME_PROFILE
        and manifest_status != "SEALED"
    ):
        raise RuntimeError(
            "B200 NF4 runtime requires an artifact manifest with status SEALED"
        )
    token = _ACTIVE_RUNTIME_PROFILE.set((runtime_profile, manifest_status))
    try:
        yield
    finally:
        _ACTIVE_RUNTIME_PROFILE.reset(token)


def require_bitsandbytes(
    *,
    require_cuda: bool = True,
    runtime_profile: str | None = None,
    manifest_status: str | None = None,
) -> tuple[Any, Any]:
    """Load bnb under an explicit conversion or sealed-runtime profile.

    The default remains the source conversion contract and therefore requires
    SM120+. B200 SM100 is accepted only while consuming a SEALED sidecar.
    """

    active_profile, active_manifest_status = _ACTIVE_RUNTIME_PROFILE.get()
    profile = active_profile if runtime_profile is None else runtime_profile
    status = (
        active_manifest_status
        if manifest_status is None
        else manifest_status
    )
    if profile not in _RUNTIME_PROFILES:
        raise ValueError(f"unknown NF4 runtime profile: {profile!r}")
    if profile == B200_SEALED_RUNTIME_PROFILE and not require_cuda:
        raise RuntimeError("B200 sealed-artifact runtime cannot disable CUDA")

    try:
        import torch
        import bitsandbytes as bnb
    except ImportError as error:
        raise RuntimeError(
            f"bitsandbytes=={SUPPORTED_BNB_VERSION} and CUDA PyTorch are required"
        ) from error
    try:
        version = importlib.metadata.version("bitsandbytes")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("bitsandbytes package metadata is unavailable") from error
    if version != SUPPORTED_BNB_VERSION:
        raise RuntimeError(
            "refusing an unaudited bitsandbytes serialization API: "
            f"found={version}, required={SUPPORTED_BNB_VERSION}"
        )

    required_objects = (
        (bnb.functional, "quantize_4bit"),
        (bnb.functional, "dequantize_4bit"),
        (bnb.functional, "QuantState"),
        (bnb.nn, "Linear4bit"),
        (bnb.nn, "Params4bit"),
    )
    for owner, name in required_objects:
        if not hasattr(owner, name):
            raise RuntimeError(f"bitsandbytes API lacks {owner.__name__}.{name}")
    quantize_parameters = inspect.signature(
        bnb.functional.quantize_4bit
    ).parameters
    for name in (
        "A",
        "blocksize",
        "compress_statistics",
        "quant_type",
        "quant_storage",
    ):
        if name not in quantize_parameters:
            raise RuntimeError(
                f"bitsandbytes quantize_4bit lacks audited argument {name!r}"
            )
    prequantized_parameters = inspect.signature(
        bnb.nn.Params4bit.from_prequantized
    ).parameters
    for name in ("data", "quantized_stats", "device", "module"):
        if name not in prequantized_parameters:
            raise RuntimeError(
                f"Params4bit.from_prequantized lacks audited argument {name!r}"
            )
    if require_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is required for NF4 profile {profile!r}")
        capability = torch.cuda.get_device_capability()
        if (
            profile == CONVERSION_RUNTIME_PROFILE
            and capability[0] < 12
        ):
            raise RuntimeError(
                f"production artifact must be quantized on SM120+, found={capability}"
            )
        if profile == B200_SEALED_RUNTIME_PROFILE:
            native_library = bitsandbytes_native_library_path(bnb)
            require_sealed_nf4_runtime(
                capability=tuple(capability),
                manifest_status=status or "",
                bitsandbytes_version=version,
                native_library=native_library,
            )
    return torch, bnb


def require_b200_sealed_bitsandbytes(
    *,
    manifest_status: str,
) -> tuple[Any, Any, dict[str, Any]]:
    """Qualify bnb for a sealed SM100 load and return receipt evidence."""

    torch, bnb = require_bitsandbytes(
        require_cuda=True,
        runtime_profile=B200_SEALED_RUNTIME_PROFILE,
        manifest_status=manifest_status,
    )
    native_library = bitsandbytes_native_library_path(bnb)
    contract = require_sealed_nf4_runtime(
        capability=tuple(torch.cuda.get_device_capability()),
        manifest_status=manifest_status,
        bitsandbytes_version=importlib.metadata.version("bitsandbytes"),
        native_library=native_library,
    )
    contract["native_library_path"] = str(native_library)
    contract["runtime_profile"] = B200_SEALED_RUNTIME_PROFILE
    return torch, bnb, contract


def _cpu_clone(tensor: Any) -> Any:
    return tensor.detach().to(device="cpu").contiguous().clone()


def _tensor_bytes(tensor: Any) -> int:
    return tensor.numel() * tensor.element_size()


def quantize_nf4_matrix(
    weight: Any,
    *,
    device: str,
    max_relative_rmse: float = 0.15,
    min_cosine_similarity: float = 0.98,
) -> PackedNF4:
    """Quantize one BF16 matrix and serialize bnb's own packed QuantState."""

    torch, bnb = require_bitsandbytes(require_cuda=True)
    if not isinstance(device, str) or not device.startswith("cuda"):
        raise ValueError(f"production NF4 device must be CUDA, got {device!r}")
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise TypeError("weight must be a rank-2 torch.Tensor")
    if weight.dtype is not torch.bfloat16 or weight.device.type != "cpu":
        raise RuntimeError(
            "source matrix must be a CPU BF16 tensor loaded from safetensors"
        )
    if weight.numel() == 0 or weight.numel() % 4:
        raise RuntimeError(
            f"matrix element count must be positive and divisible by four: "
            f"{weight.numel()}"
        )
    if not bool(torch.isfinite(weight).all().item()):
        raise RuntimeError("source matrix contains NaN or Inf")

    source = weight.to(device=device, dtype=torch.bfloat16).contiguous()
    packed, quant_state = bnb.functional.quantize_4bit(
        source,
        blocksize=64,
        compress_statistics=True,
        quant_type="nf4",
        quant_storage=torch.bfloat16,
    )
    torch.cuda.synchronize(device)
    if packed.dtype is not torch.bfloat16:
        raise RuntimeError(
            f"bnb returned wrong quant_storage dtype: {packed.dtype}"
        )
    expected_weight_bytes = weight.numel() // 2
    packed_weight_bytes = _tensor_bytes(packed)
    if packed_weight_bytes != expected_weight_bytes:
        raise RuntimeError(
            "packed NF4 weight is not exactly four bits per source element: "
            f"{packed_weight_bytes} != {expected_weight_bytes}"
        )
    if (
        quant_state.quant_type != "nf4"
        or quant_state.blocksize != 64
        or not quant_state.nested
        or quant_state.dtype is not torch.bfloat16
        or tuple(quant_state.shape) != tuple(weight.shape)
    ):
        raise RuntimeError("bitsandbytes returned an unexpected QuantState")

    restored = bnb.functional.dequantize_4bit(packed, quant_state)
    torch.cuda.synchronize(device)
    if tuple(restored.shape) != tuple(source.shape):
        raise RuntimeError("NF4 dequantization changed matrix shape")
    if not bool(torch.isfinite(restored).all().item()):
        raise RuntimeError("NF4 dequantization produced NaN or Inf")
    source_float = source.float()
    restored_float = restored.float()
    difference = restored_float - source_float
    source_rms_tensor = source_float.square().mean().sqrt()
    if source_rms_tensor.item() == 0.0:
        relative_rmse = 0.0 if difference.abs().max().item() == 0.0 else math.inf
    else:
        relative_rmse = (
            difference.square().mean().sqrt() / source_rms_tensor
        ).item()
    cosine_similarity = torch.nn.functional.cosine_similarity(
        source_float.reshape(1, -1),
        restored_float.reshape(1, -1),
        dim=1,
        eps=1e-12,
    ).item()
    maximum_absolute_error = difference.abs().max().item()
    if not math.isfinite(relative_rmse) or relative_rmse > max_relative_rmse:
        raise RuntimeError(
            f"NF4 relative RMSE exceeds bound: {relative_rmse:.8f} > "
            f"{max_relative_rmse:.8f}"
        )
    if (
        not math.isfinite(cosine_similarity)
        or cosine_similarity < min_cosine_similarity
    ):
        raise RuntimeError(
            f"NF4 cosine similarity below bound: {cosine_similarity:.8f} < "
            f"{min_cosine_similarity:.8f}"
        )

    packed_state = quant_state.as_dict(packed=True)
    expected_state_keys = {
        suffix.removeprefix(".")
        for suffix in PACKED_STATE_SUFFIXES
        if suffix
    }
    if set(packed_state) != expected_state_keys:
        raise RuntimeError(
            "bitsandbytes packed QuantState schema changed: "
            f"actual={sorted(packed_state)} expected={sorted(expected_state_keys)}"
        )
    tensors: dict[str, Any] = {"": _cpu_clone(packed)}
    for name, tensor in packed_state.items():
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(
                f"packed QuantState value is not a tensor: {name}"
            )
        tensors[f".{name}"] = _cpu_clone(tensor)
    if set(tensors) != PACKED_STATE_SUFFIXES:
        raise AssertionError("internal packed-state key mismatch")
    state_bytes = sum(
        _tensor_bytes(tensor)
        for suffix, tensor in tensors.items()
        if suffix
    )
    # Nested NF4 metadata should be small.  This catches accidental master
    # weights or an incompatible serialization format.
    if state_bytes > math.ceil(expected_weight_bytes * 0.10):
        raise RuntimeError(
            f"NF4 metadata exceeds 10% safety bound: {state_bytes} bytes"
        )

    metrics = QuantizationMetrics(
        relative_rmse=float(relative_rmse),
        cosine_similarity=float(cosine_similarity),
        source_rms=float(source_rms_tensor.item()),
        maximum_absolute_error=float(maximum_absolute_error),
        packed_weight_bytes=packed_weight_bytes,
        quant_state_bytes=state_bytes,
    )
    del source, restored, source_float, restored_float, difference, packed
    torch.cuda.empty_cache()
    return PackedNF4(tensors=tensors, metrics=metrics)


def linear4bit_from_packed(
    *,
    tensors: Mapping[str, Any],
    in_features: int,
    out_features: int,
    device: str,
) -> Any:
    """Build a frozen Linear4bit directly from the converter's packed state."""

    torch, bnb = require_bitsandbytes(require_cuda=True)
    if set(tensors) != PACKED_STATE_SUFFIXES:
        raise RuntimeError(
            "prequantized projection has missing/unexpected tensors: "
            f"{sorted(tensors)}"
        )
    packed = tensors[""]
    if not isinstance(packed, torch.Tensor) or packed.dtype is not torch.bfloat16:
        raise RuntimeError("prequantized weight must use BF16 quant storage")
    expected_bytes = (in_features * out_features) // 2
    if _tensor_bytes(packed) != expected_bytes:
        raise RuntimeError(
            f"prequantized weight bytes mismatch: {_tensor_bytes(packed)} != "
            f"{expected_bytes}"
        )
    stats = {
        suffix.removeprefix("."): tensor
        for suffix, tensor in tensors.items()
        if suffix
    }
    linear = bnb.nn.Linear4bit(
        in_features,
        out_features,
        bias=False,
        compute_dtype=torch.bfloat16,
        compress_statistics=True,
        quant_type="nf4",
        quant_storage=torch.bfloat16,
        device="meta",
    )
    parameter = bnb.nn.Params4bit.from_prequantized(
        data=packed,
        quantized_stats=stats,
        requires_grad=False,
        device=device,
        module=linear,
    )
    linear.weight = parameter
    linear.quant_state = parameter.quant_state
    quant_state = parameter.quant_state
    if (
        quant_state.quant_type != "nf4"
        or quant_state.blocksize != 64
        or not quant_state.nested
        or quant_state.dtype is not torch.bfloat16
        or tuple(quant_state.shape) != (out_features, in_features)
        or quant_state.state2 is None
        or quant_state.state2.blocksize != 256
        or quant_state.state2.dtype is not torch.float32
    ):
        raise RuntimeError("restored Linear4bit QuantState violates NF4 contract")
    if linear.weight.requires_grad:
        raise RuntimeError("routed expert base weight must remain frozen")
    if (
        linear.weight.quant_storage is not torch.bfloat16
        or linear.weight.quant_type != "nf4"
        or not linear.weight.compress_statistics
    ):
        raise RuntimeError("loaded Linear4bit violates NF4 storage contract")
    return linear
