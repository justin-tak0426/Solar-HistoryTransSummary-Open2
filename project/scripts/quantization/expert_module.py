#!/usr/bin/env python3
"""Per-expert Solar module and strict NF4 sidecar loader."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from torch.nn import functional as F

from .bnb_nf4 import linear4bit_from_packed
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
    output_shape,
    output_weight_key,
    sha256_file,
)


def resolve_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if name in {"silu", "swish"}:
        return F.silu
    raise RuntimeError(
        f"unsupported Solar expert activation {name!r}; only silu is audited"
    )


class SolarOpen2PerExpertMLP(nn.Module):
    """One routed expert with the exact native gate/up/down orientation."""

    def __init__(
        self,
        gate_proj: nn.Module,
        up_proj: nn.Module,
        down_proj: nn.Module,
        *,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.act_fn = resolve_activation(activation)
        _validate_linear_shape(
            gate_proj, hidden_size, intermediate_size, "gate_proj"
        )
        _validate_linear_shape(
            up_proj, hidden_size, intermediate_size, "up_proj"
        )
        _validate_linear_shape(
            down_proj, intermediate_size, hidden_size, "down_proj"
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states))
            * self.up_proj(hidden_states)
        )


class SolarOpen2PerExpertRoutedExperts(nn.Module):
    """Drop-in replacement for the official packed SolarOpen2Experts module."""

    def __init__(
        self,
        experts: list[SolarOpen2PerExpertMLP] | nn.ModuleList,
        *,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        super().__init__()
        if not experts:
            raise ValueError("at least one routed expert is required")
        self.experts = (
            experts if isinstance(experts, nn.ModuleList) else nn.ModuleList(experts)
        )
        self.num_experts = len(self.experts)
        self.hidden_dim = hidden_size
        self.intermediate_dim = intermediate_size
        for position, expert in enumerate(self.experts):
            if not isinstance(expert, SolarOpen2PerExpertMLP):
                raise TypeError(
                    f"expert {position} has unexpected type {type(expert).__name__}"
                )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.ndim != 2 or hidden_states.shape[-1] != self.hidden_dim:
            raise RuntimeError(
                f"hidden_states must be [tokens,{self.hidden_dim}], "
                f"got={tuple(hidden_states.shape)}"
            )
        if top_k_index.ndim != 2 or top_k_index.shape[0] != hidden_states.shape[0]:
            raise RuntimeError("top_k_index shape is incompatible with hidden_states")
        if top_k_weights.shape != top_k_index.shape:
            raise RuntimeError("top_k_weights shape must equal top_k_index shape")
        if top_k_index.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise RuntimeError(
                f"top_k_index must use an integer dtype, got={top_k_index.dtype}"
            )
        if top_k_weights.device != hidden_states.device:
            raise RuntimeError("routing weights and hidden states are on different devices")
        if top_k_index.device != hidden_states.device:
            raise RuntimeError("routing indices and hidden states are on different devices")

        final_hidden_states = torch.zeros_like(hidden_states)
        # Python dispatch is intentional: each expert is an independently
        # prequantized Linear4bit module.  The unique list also provides one
        # bounded runtime validation point for router output.
        active_experts = torch.unique(top_k_index.detach()).to("cpu").tolist()
        for expert_index in active_experts:
            if not isinstance(expert_index, int) or not (
                0 <= expert_index < self.num_experts
            ):
                raise RuntimeError(
                    f"router selected out-of-range expert {expert_index}; "
                    f"num_experts={self.num_experts}"
                )
            token_index, top_k_position = torch.where(
                top_k_index == expert_index
            )
            current_state = hidden_states[token_index]
            current_hidden_states = self.experts[expert_index](current_state)
            current_hidden_states = current_hidden_states * top_k_weights[
                token_index, top_k_position, None
            ]
            final_hidden_states.index_add_(
                0,
                token_index,
                current_hidden_states.to(final_hidden_states.dtype),
            )
        return final_hidden_states

    @classmethod
    def from_native_tensors(
        cls,
        gate_up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        *,
        activation: str = "silu",
    ) -> "SolarOpen2PerExpertRoutedExperts":
        """Dense reference constructor used for exact split-fidelity tests."""

        if gate_up_proj.ndim != 3 or down_proj.ndim != 3:
            raise ValueError("native expert tensors must both be rank three")
        experts, doubled_intermediate, hidden = gate_up_proj.shape
        if doubled_intermediate % 2:
            raise ValueError("gate_up_proj dimension 1 must be even")
        intermediate = doubled_intermediate // 2
        if tuple(down_proj.shape) != (experts, hidden, intermediate):
            raise ValueError(
                f"down_proj shape mismatch: {tuple(down_proj.shape)} != "
                f"{(experts, hidden, intermediate)}"
            )
        if gate_up_proj.dtype != down_proj.dtype:
            raise ValueError("native expert tensors must have the same dtype")
        modules: list[SolarOpen2PerExpertMLP] = []
        for expert_index in range(experts):
            gate = nn.Linear(hidden, intermediate, bias=False)
            up = nn.Linear(hidden, intermediate, bias=False)
            down = nn.Linear(intermediate, hidden, bias=False)
            gate.to(dtype=gate_up_proj.dtype, device=gate_up_proj.device)
            up.to(dtype=gate_up_proj.dtype, device=gate_up_proj.device)
            down.to(dtype=gate_up_proj.dtype, device=gate_up_proj.device)
            with torch.no_grad():
                gate.weight.copy_(gate_up_proj[expert_index, :intermediate, :])
                up.weight.copy_(gate_up_proj[expert_index, intermediate:, :])
                down.weight.copy_(down_proj[expert_index, :, :])
            modules.append(
                SolarOpen2PerExpertMLP(
                    gate,
                    up,
                    down,
                    hidden_size=hidden,
                    intermediate_size=intermediate,
                    activation=activation,
                )
            )
        return cls(
            modules,
            hidden_size=hidden,
            intermediate_size=intermediate,
        )


def _validate_linear_shape(
    module: nn.Module,
    in_features: int,
    out_features: int,
    name: str,
) -> None:
    actual_in = getattr(module, "in_features", None)
    actual_out = getattr(module, "out_features", None)
    if (actual_in, actual_out) != (in_features, out_features):
        raise ValueError(
            f"{name} feature mismatch: {(actual_in, actual_out)} != "
            f"{(in_features, out_features)}"
        )
    weight = getattr(module, "weight", None)
    quant_state = getattr(weight, "quant_state", None)
    if quant_state is None and tuple(weight.shape) != (out_features, in_features):
        raise ValueError(
            f"{name}.weight shape mismatch: {tuple(weight.shape)} != "
            f"{(out_features, in_features)}"
        )
    if quant_state is not None and tuple(quant_state.shape) != (
        out_features,
        in_features,
    ):
        raise ValueError(
            f"{name} QuantState shape mismatch: {tuple(quant_state.shape)}"
        )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _validate_manifest(
    checkpoint_dir: Path,
    *,
    contract: ArchitectureContract,
) -> tuple[dict[str, Any], dict[str, str]]:
    manifest_path = checkpoint_dir / MANIFEST_FILENAME
    index_path = checkpoint_dir / EXPERT_INDEX_FILENAME
    manifest = _read_json(manifest_path)
    index = _read_json(index_path)
    exact_manifest = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "SEALED",
        "complete_model_checkpoint": False,
        "base_model_id": BASE_MODEL_ID,
        "base_revision": BASE_REVISION,
    }
    for key, expected in exact_manifest.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"NF4 manifest mismatch: {key}={manifest.get(key)!r}, "
                f"expected={expected!r}"
            )
    if manifest.get("architecture") != contract.to_dict():
        raise RuntimeError("NF4 manifest architecture contract changed")
    if manifest.get("expert_index_sha256") != sha256_file(index_path):
        raise RuntimeError("NF4 expert index changed after sealing")
    if set(index) != {"metadata", "weight_map"}:
        raise RuntimeError("invalid NF4 expert index fields")
    weight_map = index["weight_map"]
    if not isinstance(weight_map, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in weight_map.items()
    ):
        raise RuntimeError("invalid NF4 expert index weight_map")
    return manifest, weight_map


def load_nf4_layer(
    checkpoint_dir: Path,
    *,
    layer: int,
    device: str,
    contract: ArchitectureContract = ARCHITECTURE,
    verify_hashes: bool = True,
) -> SolarOpen2PerExpertRoutedExperts:
    """Load one complete routed-expert layer from sealed sidecar shards."""

    if not 0 <= layer < contract.num_layers:
        raise ValueError(f"layer out of range: {layer}")
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    manifest, weight_map = _validate_manifest(
        checkpoint_dir, contract=contract
    )
    files_by_name = {
        item["filename"]: item for item in manifest.get("files", [])
    }
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("safetensors is required to load NF4 experts") from error

    required_keys: set[str] = set()
    shard_names: set[str] = set()
    for expert in range(contract.num_experts):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            base = output_weight_key(layer, expert, projection)
            for suffix in PACKED_STATE_SUFFIXES:
                key = base + suffix
                required_keys.add(key)
                shard = weight_map.get(key)
                if shard is None:
                    raise RuntimeError(f"NF4 expert index is missing {key}")
                shard_names.add(shard)
    foreign_layer_keys = {
        key
        for key in weight_map
        if key.startswith(f"model.layers.{layer}.mlp.experts.")
    } - required_keys
    if foreign_layer_keys:
        raise RuntimeError(
            f"NF4 index has unexpected layer {layer} keys: "
            f"{sorted(foreign_layer_keys)[:8]}"
        )

    loaded: dict[str, torch.Tensor] = {}
    for shard_name in sorted(shard_names):
        if Path(shard_name).name != shard_name:
            raise RuntimeError(f"unsafe NF4 shard name: {shard_name!r}")
        shard_path = checkpoint_dir / shard_name
        record = files_by_name.get(shard_name)
        if record is None:
            raise RuntimeError(f"manifest does not seal shard {shard_name}")
        if shard_path.stat().st_size != record.get("file_bytes"):
            raise RuntimeError(f"NF4 shard size changed: {shard_name}")
        if verify_hashes and sha256_file(shard_path) != record.get("sha256"):
            raise RuntimeError(f"NF4 shard digest changed: {shard_name}")
        needed_from_shard = {
            key for key in required_keys if weight_map[key] == shard_name
        }
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            actual = set(handle.keys())
            if not needed_from_shard.issubset(actual):
                raise RuntimeError(
                    f"NF4 shard lacks indexed tensors: {shard_name}"
                )
            for key in sorted(needed_from_shard):
                if key in loaded:
                    raise RuntimeError(f"duplicate loaded NF4 tensor: {key}")
                loaded[key] = handle.get_tensor(key)
    if set(loaded) != required_keys:
        raise RuntimeError("NF4 layer load did not account every required tensor")

    experts: list[SolarOpen2PerExpertMLP] = []
    for expert in range(contract.num_experts):
        projections: dict[str, nn.Module] = {}
        for projection in ("gate_proj", "up_proj", "down_proj"):
            base = output_weight_key(layer, expert, projection)
            packed = {
                suffix: loaded.pop(base + suffix)
                for suffix in PACKED_STATE_SUFFIXES
            }
            out_features, in_features = output_shape(projection, contract)
            projections[projection] = linear4bit_from_packed(
                tensors=packed,
                in_features=in_features,
                out_features=out_features,
                device=device,
            )
        experts.append(
            SolarOpen2PerExpertMLP(
                projections["gate_proj"],
                projections["up_proj"],
                projections["down_proj"],
                hidden_size=contract.hidden_size,
                intermediate_size=contract.intermediate_size,
                activation="silu",
            )
        )
    if loaded:
        raise AssertionError("internal NF4 layer tensor leak")
    return SolarOpen2PerExpertRoutedExperts(
        experts,
        hidden_size=contract.hidden_size,
        intermediate_size=contract.intermediate_size,
    )


def install_nf4_experts(
    model: nn.Module,
    checkpoint_dir: Path,
    *,
    layer_device_map: Mapping[int, str],
    contract: ArchitectureContract = ARCHITECTURE,
    verify_hashes: bool = True,
) -> nn.Module:
    """Replace native meta experts on every layer with sealed NF4 modules."""

    expected_layers = set(range(contract.num_layers))
    if set(layer_device_map) != expected_layers:
        raise RuntimeError(
            "layer_device_map must cover every decoder layer exactly: "
            f"missing={sorted(expected_layers - set(layer_device_map))} "
            f"unexpected={sorted(set(layer_device_map) - expected_layers)}"
        )
    try:
        layers = model.model.layers
    except AttributeError as error:
        raise RuntimeError("model does not expose model.layers") from error
    if len(layers) != contract.num_layers:
        raise RuntimeError(
            f"model layer count mismatch: {len(layers)} != {contract.num_layers}"
        )
    for layer_index, layer_module in enumerate(layers):
        native = getattr(getattr(layer_module, "mlp", None), "experts", None)
        if native is None:
            raise RuntimeError(f"layer {layer_index} lacks routed experts")
        parameters = list(native.parameters(recurse=False))
        if {name for name, _ in native.named_parameters(recurse=False)} != {
            "gate_up_proj",
            "down_proj",
        }:
            raise RuntimeError(
                f"layer {layer_index} does not have native packed experts"
            )
        if any(parameter.device.type != "meta" for parameter in parameters):
            raise RuntimeError(
                "native routed weights are already materialized; refusing a "
                f"transient BF16 allocation at layer {layer_index}. Instantiate "
                "the official model under init_empty_weights first."
            )
        replacement = load_nf4_layer(
            checkpoint_dir,
            layer=layer_index,
            device=layer_device_map[layer_index],
            contract=contract,
            verify_hashes=verify_hashes,
        )
        layer_module.mlp.experts = replacement
    return model
