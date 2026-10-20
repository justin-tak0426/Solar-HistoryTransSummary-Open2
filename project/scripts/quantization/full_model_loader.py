#!/usr/bin/env python3
"""Stream the complete two-B200 Solar model with sealed NF4 experts."""

from __future__ import annotations

import importlib.metadata
import json
import re
from collections import defaultdict
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

try:
    from ..b200_runtime import (
        B200_CONFIG_DIR,
        BF16_SOURCE_DIR,
        BOUNDARY_LAYERS,
        EXPECTED_HF_DEVICE_MAP,
        EXPECTED_LAYER_DEVICE_MAP,
        INHERITED_LARGE_HASH_MODE,
        TRAIN_EXPERT_DIR,
        inherit_b200_source_shard_digests,
        load_b200_config_bundle,
        require_b200_gpu_runtime,
        require_expected_user,
        require_owned_path,
        require_owned_root,
        validate_b200_large_asset_receipt,
    )
except ImportError:  # Direct scripts/ entry points expose quantization top-level.
    from b200_runtime import (
        B200_CONFIG_DIR,
        BF16_SOURCE_DIR,
        BOUNDARY_LAYERS,
        EXPECTED_HF_DEVICE_MAP,
        EXPECTED_LAYER_DEVICE_MAP,
        INHERITED_LARGE_HASH_MODE,
        TRAIN_EXPERT_DIR,
        inherit_b200_source_shard_digests,
        load_b200_config_bundle,
        require_b200_gpu_runtime,
        require_expected_user,
        require_owned_path,
        require_owned_root,
        validate_b200_large_asset_receipt,
    )
from .bnb_nf4 import (
    B200_SEALED_RUNTIME_PROFILE,
    require_b200_sealed_bitsandbytes,
    use_nf4_runtime_profile,
)
from .checkpoint import (
    B200_RUNTIME_SOURCE_PROFILE,
    SourceInventory,
    inventory_source_checkpoint,
    validate_transformers_schema,
)
from .contract import (
    ARCHITECTURE,
    BASE_REVISION,
    MANIFEST_FILENAME,
    SOURCE_EXPERT_RE,
)
from .layer_flat_checkpoint import (
    load_layer_flat_experts,
    preflight_layer_flat_checkpoint,
)
from .layer_flat_module import SolarOpen2LayerFlatNF4Experts
from .output_checkpoint import read_json_no_duplicates


LAYER_KEY_RE = re.compile(r"^model\.layers\.(\d+)\.")
LORA_TARGET_RE = re.compile(
    r"^model\.layers\.\d+\.(?:"
    r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj|g_proj|b_proj|"
    r"f_a_proj|f_b_proj|g_a_proj|g_b_proj)|"
    r"mlp\.shared_experts\.(?:gate_proj|up_proj|down_proj))$"
)


def default_layer_device_map() -> dict[int, str]:
    """Return the shared 24/24 two-B200 layer placement contract."""

    return dict(EXPECTED_LAYER_DEVICE_MAP)


def _require_exact_layer_device_map(
    layer_device_map: Mapping[int, str],
) -> dict[int, str]:
    observed = dict(layer_device_map)
    expected = dict(EXPECTED_LAYER_DEVICE_MAP)
    if observed != expected:
        raise RuntimeError(
            "B200 layer device map mismatch: " f"expected={expected} actual={observed}"
        )
    if "cuda:2" in observed.values():
        raise RuntimeError("B200 layer device map must never reference cuda:2")
    return observed


def _validate_hardware(layer_device_map: Mapping[int, str]) -> dict[str, Any]:
    _require_exact_layer_device_map(layer_device_map)
    require_owned_root()
    require_expected_user()
    return require_b200_gpu_runtime(torch_module=torch)


def _resolve_b200_loader_inputs(
    *,
    source_dir: Path | None,
    expert_checkpoint_dir: Path | None,
    config_dir: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    bundle = load_b200_config_bundle(config_dir)
    runtime = bundle["model"].get("training_runtime")
    if not isinstance(runtime, Mapping):
        raise RuntimeError("B200 model config lacks training_runtime")
    configured_source = Path(str(runtime.get("source_dir")))
    configured_experts = Path(str(runtime.get("expert_dir")))
    if configured_source != BF16_SOURCE_DIR:
        raise RuntimeError(
            "B200 training source path drifted from the pinned snapshot: "
            f"{configured_source} != {BF16_SOURCE_DIR}"
        )
    if configured_experts != TRAIN_EXPERT_DIR:
        raise RuntimeError(
            "B200 training NF4 path drifted: "
            f"{configured_experts} != {TRAIN_EXPERT_DIR}"
        )
    resolved_source = require_owned_path(
        BF16_SOURCE_DIR if source_dir is None else source_dir,
        must_exist=True,
    )
    resolved_experts = require_owned_path(
        TRAIN_EXPERT_DIR if expert_checkpoint_dir is None else expert_checkpoint_dir,
        must_exist=True,
    )
    if resolved_source != BF16_SOURCE_DIR:
        raise RuntimeError(
            f"loader source must equal pinned B200 source: {resolved_source}"
        )
    if resolved_experts != TRAIN_EXPERT_DIR:
        raise RuntimeError(
            f"loader NF4 path must equal sealed B200 artifact: {resolved_experts}"
        )
    identity = {
        "config_dir": str(bundle["config_dir"]),
        "hashes": dict(bundle["hashes"]),
    }
    return resolved_source, resolved_experts, identity


def _target_device_for_key(
    key: str,
    layer_device_map: Mapping[int, str],
) -> str:
    match = LAYER_KEY_RE.match(key)
    if match:
        layer = int(match.group(1))
        if layer not in layer_device_map:
            raise RuntimeError(f"checkpoint key has out-of-map layer: {key}")
        return layer_device_map[layer]
    if key == "model.embed_tokens.weight":
        return "cuda:0"
    if key == "lm_head.weight":
        return "cuda:1"
    if key.startswith("model.norm."):
        return layer_device_map[ARCHITECTURE.num_layers - 1]
    raise RuntimeError(f"cannot assign non-layer Solar tensor to a GPU: {key}")


def _stream_nonexpert_tensors(
    model: nn.Module,
    inventory: SourceInventory,
    *,
    layer_device_map: Mapping[int, str],
) -> dict[str, Any]:
    try:
        from accelerate.utils import set_module_tensor_to_device
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError(
            "accelerate and safetensors are required for full model loading"
        ) from error
    expected_nonexpert = {
        key for key in inventory.tensors if not SOURCE_EXPERT_RE.fullmatch(key)
    }
    by_shard: dict[str, list[str]] = defaultdict(list)
    for key in sorted(expected_nonexpert):
        by_shard[inventory.tensors[key].shard].append(key)
    loaded: set[str] = set()
    for shard_name in sorted(by_shard):
        path = inventory.source_dir / shard_name
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in by_shard[shard_name]:
                value = handle.get_tensor(key)
                record = inventory.tensors[key]
                if tuple(value.shape) != record.shape:
                    raise RuntimeError(f"streamed tensor shape changed: {key}")
                device = _target_device_for_key(key, layer_device_map)
                set_module_tensor_to_device(
                    model,
                    key,
                    device,
                    value=value,
                    dtype=value.dtype,
                )
                loaded.add(key)
                del value
    if loaded != expected_nonexpert:
        raise RuntimeError(
            "nonexpert streaming coverage mismatch: "
            f"missing={sorted(expected_nonexpert - loaded)[:8]} "
            f"unexpected={sorted(loaded - expected_nonexpert)[:8]}"
        )

    embed = model.model.embed_tokens.weight
    lm_head = model.lm_head.weight
    if embed.device.type == "meta" or lm_head.device.type == "meta":
        raise RuntimeError("distinct embed/lm_head source weights were not both loaded")
    if embed is lm_head:
        raise RuntimeError(
            "tie_word_embeddings=false but loader aliased embed and lm_head"
        )
    return {
        "expected_nonexpert_tensor_count": len(expected_nonexpert),
        "loaded_nonexpert_tensor_count": len(loaded),
        "loaded_nonexpert_elements": sum(
            inventory.tensors[key].numel for key in loaded
        ),
    }


def _move_tree(value: Any, device: str) -> Any:
    """Differentiable recursive tensor move used by layer boundary hooks."""

    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, tuple):
        return tuple(_move_tree(item, device) for item in value)
    if isinstance(value, list):
        return [_move_tree(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_tree(item, device) for key, item in value.items()}
    return value


GRADIENT_CHECKPOINTING_KWARGS = {
    "use_reentrant": False,
    "preserve_rng_state": True,
    "determinism_check": "default",
    "early_stop": True,
}
SELECTIVE_CHECKPOINT_LAYER_INDICES = tuple(
    layer for layer in range(ARCHITECTURE.num_layers) if layer % 4 != 3
)


def _checkpoint_on_layer_device(
    function: Any,
    *args: Any,
    device: str,
) -> Any:
    """Move checkpoint inputs before PyTorch snapshots the execution-device RNG."""

    moved_args = _move_tree(args, device)
    if (
        not moved_args
        or not isinstance(moved_args[0], torch.Tensor)
        or moved_args[0].device != torch.device(device)
    ):
        raise RuntimeError(
            f"decoder checkpoint input did not reach its execution device: {device}"
        )
    return checkpoint(
        function,
        *moved_args,
        **GRADIENT_CHECKPOINTING_KWARGS,
    )


def _decoder_layers(model: nn.Module) -> nn.ModuleList:
    base_model = (
        model.get_base_model()
        if callable(getattr(model, "get_base_model", None))
        else model
    )
    decoder = getattr(base_model, "model", None)
    layers = getattr(decoder, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise RuntimeError("Solar decoder layers are unavailable")
    if len(layers) != ARCHITECTURE.num_layers:
        raise RuntimeError(
            f"Solar decoder layer count changed: {len(layers)} "
            f"!= {ARCHITECTURE.num_layers}"
        )
    return layers


def configure_model_parallel_gradient_checkpointing(
    model: nn.Module,
    *,
    layer_device_map: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Install non-reentrant checkpoints whose RNG device matches each layer."""

    if layer_device_map is None:
        layer_device_map = default_layer_device_map()
    _require_exact_layer_device_map(layer_device_map)
    if not bool(getattr(model, "is_gradient_checkpointing", False)):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=dict(GRADIENT_CHECKPOINTING_KWARGS)
        )
    layers = _decoder_layers(model)
    for layer_index, layer in enumerate(layers):
        device = str(torch.device(layer_device_map[layer_index]))
        layer.gradient_checkpointing = True
        layer._gradient_checkpointing_func = partial(
            _checkpoint_on_layer_device,
            device=device,
        )
        layer._solar_checkpoint_input_device = device

    boundaries = [
        layer_index
        for layer_index in range(1, ARCHITECTURE.num_layers)
        if layer_device_map[layer_index] != layer_device_map[layer_index - 1]
    ]
    if boundaries != list(BOUNDARY_LAYERS):
        raise RuntimeError(f"B200 checkpoint boundaries changed: {boundaries}")
    contract = {
        "status": "PASS",
        "implementation": "device_aware_non_reentrant",
        **GRADIENT_CHECKPOINTING_KWARGS,
        "decoder_layer_count": len(layers),
        "boundary_layers": boundaries,
        "layer_input_devices": {
            str(layer): str(torch.device(device))
            for layer, device in sorted(layer_device_map.items())
        },
    }
    model._solar_gradient_checkpointing_contract = contract
    return contract


def disable_model_parallel_gradient_checkpointing(
    model: nn.Module,
    *,
    layer_device_map: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Disable recomputation while preserving the audited two-device routing."""

    if layer_device_map is None:
        layer_device_map = default_layer_device_map()
    _require_exact_layer_device_map(layer_device_map)
    model.gradient_checkpointing_disable()
    layers = _decoder_layers(model)
    for layer_index, layer in enumerate(layers):
        layer.gradient_checkpointing = False
        layer._solar_checkpoint_input_device = str(
            torch.device(layer_device_map[layer_index])
        )
    if bool(getattr(model, "is_gradient_checkpointing", False)):
        raise RuntimeError("B200 activation-checkpoint disabling did not take effect")
    boundaries = [
        layer_index
        for layer_index in range(1, ARCHITECTURE.num_layers)
        if layer_device_map[layer_index] != layer_device_map[layer_index - 1]
    ]
    if boundaries != list(BOUNDARY_LAYERS):
        raise RuntimeError(f"B200 activation-policy boundaries changed: {boundaries}")
    contract = {
        "status": "PASS",
        "implementation": "disabled_b200_memory_optimized",
        "enabled": False,
        "decoder_layer_count": len(layers),
        "boundary_layers": boundaries,
        "layer_input_devices": {
            str(layer): str(torch.device(device))
            for layer, device in sorted(layer_device_map.items())
        },
        "qualification": "longest_row_and_all_bucket_backward_smoke_required",
    }
    model._solar_gradient_checkpointing_contract = contract
    return contract


def configure_selective_model_parallel_gradient_checkpointing(
    model: nn.Module,
    *,
    layer_device_map: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Checkpoint three of every four layers, balanced across both B200s."""

    if layer_device_map is None:
        layer_device_map = default_layer_device_map()
    _require_exact_layer_device_map(layer_device_map)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs=dict(GRADIENT_CHECKPOINTING_KWARGS)
    )
    layers = _decoder_layers(model)
    checkpointed = set(SELECTIVE_CHECKPOINT_LAYER_INDICES)
    for layer_index, layer in enumerate(layers):
        device = str(torch.device(layer_device_map[layer_index]))
        layer.gradient_checkpointing = layer_index in checkpointed
        layer._gradient_checkpointing_func = partial(
            _checkpoint_on_layer_device,
            device=device,
        )
        layer._solar_checkpoint_input_device = device
    if not bool(getattr(model, "is_gradient_checkpointing", False)):
        raise RuntimeError(
            "B200 selective activation checkpointing did not take effect"
        )
    boundaries = [
        layer_index
        for layer_index in range(1, ARCHITECTURE.num_layers)
        if layer_device_map[layer_index] != layer_device_map[layer_index - 1]
    ]
    if boundaries != list(BOUNDARY_LAYERS):
        raise RuntimeError(
            f"B200 selective checkpoint boundaries changed: {boundaries}"
        )
    selected_layers = sorted(checkpointed)
    expected_selected_layers = [
        layer for layer in range(ARCHITECTURE.num_layers) if layer % 4 != 3
    ]
    if selected_layers != expected_selected_layers:
        raise RuntimeError("B200 selective checkpoint layer set changed")
    uncheckpointed_layers = [
        layer for layer in range(ARCHITECTURE.num_layers) if layer not in checkpointed
    ]
    contract = {
        "status": "PASS",
        "implementation": "device_aware_non_reentrant_selective",
        "enabled": True,
        **GRADIENT_CHECKPOINTING_KWARGS,
        "decoder_layer_count": len(layers),
        "checkpointed_decoder_layer_count": len(selected_layers),
        "uncheckpointed_decoder_layer_count": len(layers) - len(selected_layers),
        "uncheckpoint_stride": 4,
        "uncheckpoint_offset": 3,
        "checkpointed_decoder_layers": selected_layers,
        "uncheckpointed_decoder_layers": uncheckpointed_layers,
        "boundary_layers": boundaries,
        "layer_input_devices": {
            str(layer): str(torch.device(device))
            for layer, device in sorted(layer_device_map.items())
        },
        "qualification": "longest_row_and_all_bucket_backward_smoke_required",
    }
    model._solar_gradient_checkpointing_contract = contract
    return contract


def attach_differentiable_model_parallel_hooks(
    model: nn.Module,
    *,
    layer_device_map: Mapping[int, str],
) -> list[Any]:
    """Move layer inputs without inference-mode/no-grad Accelerate hooks."""

    _require_exact_layer_device_map(layer_device_map)
    handles = []

    def attach(module: nn.Module, device: str) -> None:
        def move_inputs(
            _module: nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> tuple[tuple[Any, ...], dict[str, Any]]:
            return _move_tree(args, device), _move_tree(kwargs, device)

        handles.append(module.register_forward_pre_hook(move_inputs, with_kwargs=True))

    attach(model.model.embed_tokens, "cuda:0")
    for layer, module in enumerate(model.model.layers):
        attach(module, layer_device_map[layer])
    attach(
        model.model.norm,
        layer_device_map[ARCHITECTURE.num_layers - 1],
    )
    attach(model.lm_head, "cuda:1")
    model._solar_model_parallel_hook_handles = handles
    model.hf_device_map = {
        "model.embed_tokens": "cuda:0",
        **{
            f"model.layers.{layer}": device
            for layer, device in layer_device_map.items()
        },
        "model.norm": layer_device_map[ARCHITECTURE.num_layers - 1],
        "lm_head": "cuda:1",
    }
    if dict(model.hf_device_map) != dict(EXPECTED_HF_DEVICE_MAP):
        raise RuntimeError("constructed hf_device_map differs from B200 contract")
    return handles


def _expected_tensor_device(
    name: str,
    layer_device_map: Mapping[int, str],
) -> str:
    if name.startswith("model.embed_tokens."):
        return "cuda:0"
    if name.startswith("model.rotary_emb."):
        return "cuda:0"
    match = LAYER_KEY_RE.match(name)
    if match:
        return layer_device_map[int(match.group(1))]
    if name.startswith("model.norm.") or name.startswith("lm_head."):
        return "cuda:1"
    raise RuntimeError(f"cannot audit unexpected model tensor: {name}")


def audit_parameter_and_buffer_placement(
    model: nn.Module,
    *,
    layer_device_map: Mapping[int, str],
) -> dict[str, Any]:
    """Require every registered tensor on its exact logical B200 device."""

    _require_exact_layer_device_map(layer_device_map)
    hf_device_map = getattr(model, "hf_device_map", None)
    if not isinstance(hf_device_map, Mapping) or dict(hf_device_map) != dict(
        EXPECTED_HF_DEVICE_MAP
    ):
        raise RuntimeError(
            f"model hf_device_map differs from B200 contract: {hf_device_map!r}"
        )

    counts: defaultdict[str, int] = defaultdict(int)
    violations: list[tuple[str, str, str, str]] = []
    for kind, tensors in (
        ("parameter", model.named_parameters()),
        ("buffer", model.named_buffers()),
    ):
        for name, tensor in tensors:
            expected = _expected_tensor_device(name, layer_device_map)
            actual = str(tensor.device)
            counts[f"{kind}_count"] += 1
            counts[f"{kind}_{actual}"] += 1
            if tensor.device.type != "cuda" or actual != expected:
                violations.append((kind, name, actual, expected))
    if violations:
        raise RuntimeError(
            "CPU/meta/disk or misplaced model tensors are forbidden: "
            f"{violations[:8]}"
        )
    if any("cuda:2" in key for key in counts):
        raise RuntimeError("placement audit found forbidden cuda:2 tensors")
    return {
        "status": "PASS",
        "parameter_count": counts["parameter_count"],
        "buffer_count": counts["buffer_count"],
        "parameter_device_counts": {
            device: counts[f"parameter_{device}"] for device in ("cuda:0", "cuda:1")
        },
        "buffer_device_counts": {
            device: counts[f"buffer_{device}"] for device in ("cuda:0", "cuda:1")
        },
        "cpu_tensor_count": 0,
        "meta_tensor_count": 0,
        "disk_offload_count": 0,
        "forbidden_cuda2_tensor_count": 0,
    }


def audit_loaded_model(
    model: nn.Module,
    *,
    layer_device_map: Mapping[int, str],
) -> dict[str, Any]:
    if not bool(getattr(model, "is_gradient_checkpointing", False)):
        raise RuntimeError("layer-flat experts require decoder gradient checkpointing")
    placement = audit_parameter_and_buffer_placement(
        model,
        layer_device_map=layer_device_map,
    )
    routed_layers = 0
    device_aware_checkpoint_layers = 0
    for layer_index, layer in enumerate(model.model.layers):
        experts = layer.mlp.experts
        if not isinstance(experts, SolarOpen2LayerFlatNF4Experts):
            raise RuntimeError(
                f"layer {layer_index} routed expert class changed: "
                f"{type(experts).__name__}"
            )
        expected_device = torch.device(layer_device_map[layer_index])
        devices = {parameter.device for parameter in experts.parameters()}
        if devices != {expected_device}:
            raise RuntimeError(f"layer {layer_index} expert devices differ: {devices}")
        checkpoint_function = getattr(layer, "_gradient_checkpointing_func", None)
        if (
            not isinstance(checkpoint_function, partial)
            or checkpoint_function.func is not _checkpoint_on_layer_device
            or checkpoint_function.keywords.get("device") != str(expected_device)
            or getattr(layer, "_solar_checkpoint_input_device", None)
            != str(expected_device)
        ):
            raise RuntimeError(f"layer {layer_index} lacks its device-aware checkpoint")
        routed_layers += 1
        device_aware_checkpoint_layers += 1
    target_modules = {
        name: module
        for name, module in model.named_modules()
        if LORA_TARGET_RE.fullmatch(name)
    }
    if len(target_modules) != 528:
        raise RuntimeError(
            f"LoRA target module count changed: {len(target_modules)} != 528"
        )
    wrong_target_types = {
        name: type(module).__name__
        for name, module in target_modules.items()
        if type(module) is not nn.Linear
    }
    if wrong_target_types:
        raise RuntimeError(
            f"LoRA targets are no longer BF16 nn.Linear: "
            f"{list(wrong_target_types.items())[:8]}"
        )
    trainable = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if trainable:
        raise RuntimeError(f"base model has trainable parameters: {trainable[:8]}")
    handles = getattr(model, "_solar_model_parallel_hook_handles", None)
    if not isinstance(handles, list) or len(handles) != 51:
        raise RuntimeError("differentiable model-parallel hooks are incomplete")
    return {
        "status": "PASS",
        "routed_nf4_layers": routed_layers,
        "lora_target_bf16_linear_count": len(target_modules),
        "meta_parameter_count": 0,
        "meta_buffer_count": 0,
        "placement_audit": placement,
        "base_trainable_parameter_count": 0,
        "model_parallel_hook_count": len(handles),
        "gradient_checkpointing": True,
        "device_aware_checkpoint_layer_count": (device_aware_checkpoint_layers),
        "device_map": dict(model.hf_device_map),
    }


def load_full_model(
    *,
    source_dir: Path | None = None,
    expert_checkpoint_dir: Path | None = None,
    config_dir: Path = B200_CONFIG_DIR,
    layer_device_map: Mapping[int, str] | None = None,
    verify_hashes: bool = True,
    asset_integrity_receipt: Path | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a complete frozen base without ever materializing native experts."""

    if layer_device_map is None:
        layer_device_map = default_layer_device_map()
    source_dir, expert_checkpoint_dir, config_identity = _resolve_b200_loader_inputs(
        source_dir=source_dir,
        expert_checkpoint_dir=expert_checkpoint_dir,
        config_dir=config_dir,
    )
    hardware = _validate_hardware(layer_device_map)
    try:
        import accelerate
        import transformers
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as error:
        raise RuntimeError(
            "official Transformers and Accelerate environments are required"
        ) from error
    if not str(transformers.__version__).startswith("5.14.1"):
        raise RuntimeError(f"wrong Transformers build: {transformers.__version__}")
    accelerate_version = importlib.metadata.version("accelerate")
    if accelerate_version != "1.14.0":
        raise RuntimeError(
            f"unaudited Accelerate version: {accelerate_version} != 1.14.0"
        )

    if verify_hashes:
        if asset_integrity_receipt is not None:
            raise RuntimeError(
                "direct payload hashing must not also claim inherited evidence"
            )
        asset_integrity_evidence = None
        hash_verification_mode = "direct_large_shard_hashes"
        payload_rehash_performed = True
    else:
        if asset_integrity_receipt is None:
            raise RuntimeError(
                "skipping B200 payload hashes requires the fixed large-hash "
                "asset receipt"
            )
        asset_integrity_evidence = validate_b200_large_asset_receipt(
            asset_integrity_receipt,
            role="training",
        )
        hash_verification_mode = INHERITED_LARGE_HASH_MODE
        payload_rehash_performed = False

    inventory = inventory_source_checkpoint(
        source_dir,
        hash_shards=verify_hashes,
        require_revision_path=True,
        runtime_profile=B200_RUNTIME_SOURCE_PROFILE,
    )
    source_schema = validate_transformers_schema(inventory)
    expert_preflight = preflight_layer_flat_checkpoint(
        expert_checkpoint_dir,
        hash_shards=verify_hashes,
    )
    inventory_summary = inventory.summary()
    if asset_integrity_evidence is not None:
        inventory_summary = inherit_b200_source_shard_digests(
            inventory_summary,
            asset_integrity_evidence,
        )
        expert_evidence = asset_integrity_evidence["expert"]
        for label, actual, expected in (
            (
                "checkpoint directory",
                expert_preflight.get("checkpoint_dir"),
                expert_evidence.get("path"),
            ),
            (
                "manifest SHA-256",
                expert_preflight.get("manifest_sha256"),
                expert_evidence.get("manifest", {}).get("sha256"),
            ),
            (
                "index SHA-256",
                expert_preflight.get("expert_index_sha256"),
                expert_evidence.get("index", {}).get("sha256"),
            ),
            (
                "payload file count",
                expert_preflight.get("shard_count"),
                expert_evidence.get("payload_file_count"),
            ),
        ):
            if actual != expected:
                raise RuntimeError(
                    f"B200 inherited training NF4 {label} mismatch: "
                    f"{actual!r} != {expected!r}"
                )
    expert_manifest = read_json_no_duplicates(
        Path(expert_checkpoint_dir) / MANIFEST_FILENAME
    )
    manifest_status = expert_manifest.get("status")
    if manifest_status != "SEALED":
        raise RuntimeError(f"training NF4 artifact is not SEALED: {manifest_status!r}")
    source_manifest = expert_manifest.get("source")
    if not isinstance(source_manifest, dict):
        raise RuntimeError("expert sidecar lacks source provenance")
    source_identity = {
        "index_sha256": inventory.index_sha256,
        "config_sha256": inventory.config_sha256,
        "weight_map_sha256": inventory.weight_map_sha256,
        "shards": inventory_summary["shards"],
    }
    for key, expected in source_identity.items():
        if source_manifest.get(key) != expected:
            raise RuntimeError(
                f"expert/source checkpoint mismatch: {key} "
                f"{source_manifest.get(key)!r} != {expected!r}"
            )

    config = AutoConfig.from_pretrained(
        inventory.source_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    if config.tie_word_embeddings is not False:
        raise RuntimeError("official pinned config must keep embed/lm_head distinct")
    config.use_cache = False
    with init_empty_weights(include_buffers=True):
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=False,
        )
    if model.__class__.__name__ != "SolarOpen2ForCausalLM":
        raise RuntimeError(f"wrong model class: {model.__class__.__name__}")

    _bnb_torch, _bnb, bnb_runtime = require_b200_sealed_bitsandbytes(
        manifest_status=manifest_status
    )
    with use_nf4_runtime_profile(
        B200_SEALED_RUNTIME_PROFILE,
        manifest_status=manifest_status,
    ):
        for layer_index in range(ARCHITECTURE.num_layers):
            native = model.model.layers[layer_index].mlp.experts
            if {name for name, _ in native.named_parameters(recurse=False)} != {
                "gate_up_proj",
                "down_proj",
            }:
                raise RuntimeError(f"layer {layer_index} native expert schema changed")
            if any(
                parameter.device.type != "meta"
                for parameter in native.parameters(recurse=False)
            ):
                raise RuntimeError(
                    "layer "
                    f"{layer_index} native experts materialized before replacement"
                )
            replacement = load_layer_flat_experts(
                expert_checkpoint_dir,
                layer=layer_index,
                device=layer_device_map[layer_index],
                verify_hash=False,
            )
            model.model.layers[layer_index].mlp.experts = replacement

    rotary_class = type(model.model.rotary_emb)
    model.model.rotary_emb = rotary_class(config=config, device="cuda:0")
    if any(buffer.is_meta for buffer in model.model.rotary_emb.buffers()):
        raise RuntimeError("official rotary buffers remained on meta device")

    nonexpert_load = _stream_nonexpert_tensors(
        model,
        inventory,
        layer_device_map=layer_device_map,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    attach_differentiable_model_parallel_hooks(
        model,
        layer_device_map=layer_device_map,
    )
    gradient_checkpointing = configure_model_parallel_gradient_checkpointing(
        model,
        layer_device_map=layer_device_map,
    )
    audit = audit_loaded_model(
        model,
        layer_device_map=layer_device_map,
    )
    receipt = {
        "status": "PASS",
        "base_revision": BASE_REVISION,
        "b200_config": config_identity,
        "hardware": hardware,
        "bitsandbytes_runtime": bnb_runtime,
        "source_inventory": inventory_summary,
        "source_schema": source_schema,
        "expert_preflight": expert_preflight,
        "nonexpert_stream_load": nonexpert_load,
        "model_audit": audit,
        "transformers_version": transformers.__version__,
        "accelerate_version": accelerate_version,
        "use_cache": model.config.use_cache,
        "gradient_checkpointing": True,
        "gradient_checkpointing_contract": gradient_checkpointing,
        "hash_verification": True,
        "hash_verification_mode": hash_verification_mode,
        "direct_large_shard_hashes": verify_hashes,
        "payload_rehash_performed": payload_rehash_performed,
        "asset_integrity_evidence": asset_integrity_evidence,
        "production_eligible": True,
        "production_authorized": False,
        "next_gate": "max-length one-update forward/backward and save/reload smoke",
    }
    return model, receipt


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=B200_CONFIG_DIR)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--expert-checkpoint-dir", type=Path)
    parser.add_argument(
        "--no-hash-verification",
        action="store_true",
        help="inherit payload digests from --asset-integrity-receipt",
    )
    parser.add_argument("--asset-integrity-receipt", type=Path)
    args = parser.parse_args()
    _model, receipt = load_full_model(
        source_dir=args.source_dir,
        expert_checkpoint_dir=args.expert_checkpoint_dir,
        config_dir=args.config_dir,
        verify_hashes=not args.no_hash_verification,
        asset_integrity_receipt=args.asset_integrity_receipt,
    )
    print(json.dumps(receipt, ensure_ascii=True, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
