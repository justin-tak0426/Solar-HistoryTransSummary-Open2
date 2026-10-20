#!/usr/bin/env python3
"""Load Solar Open 2 on two B200s with sealed per-expert NF4."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from b200_runtime import (
    B200_CONFIG_DIR,
    BF16_SOURCE_DIR,
    INFERENCE_EXPERT_DIR,
    INHERITED_LARGE_HASH_MODE,
    b200_config_identity,
    inherit_b200_source_shard_digests,
    load_b200_config_bundle,
    require_owned_path,
    validate_b200_large_asset_receipt,
)
from common import canonical_sha256, expected_lora_module_names
from quantization.bnb_nf4 import (
    B200_SEALED_RUNTIME_PROFILE,
    require_b200_sealed_bitsandbytes,
    use_nf4_runtime_profile,
)
from quantization.checkpoint import (
    B200_RUNTIME_SOURCE_PROFILE,
    inventory_source_checkpoint,
    validate_transformers_schema,
)
from quantization.contract import (
    ARCHITECTURE,
    BASE_REVISION,
    MANIFEST_FILENAME,
)
from quantization.expert_module import (
    SolarOpen2PerExpertRoutedExperts,
    install_nf4_experts,
)

# Reuse the audited base-weight streamer and transfer hooks so training and
# inference cannot diverge in non-expert placement.
from quantization.full_model_loader import (
    _stream_nonexpert_tensors as stream_nonexpert_tensors,
)
from quantization.full_model_loader import (
    _validate_hardware as validate_hardware,
)
from quantization.full_model_loader import (
    attach_differentiable_model_parallel_hooks,
    audit_parameter_and_buffer_placement,
    default_layer_device_map,
)
from quantization.output_checkpoint import (
    preflight_output_checkpoint,
    read_json_no_duplicates,
)


def _validate_source_binding(
    inventory: Any,
    expert_manifest: dict[str, Any],
    inventory_summary: dict[str, Any] | None = None,
) -> None:
    source = expert_manifest.get("source")
    if not isinstance(source, dict):
        raise RuntimeError("per-expert NF4 sidecar lacks source provenance")
    summary = inventory.summary() if inventory_summary is None else inventory_summary
    actual = {
        "index_sha256": inventory.index_sha256,
        "config_sha256": inventory.config_sha256,
        "weight_map_sha256": inventory.weight_map_sha256,
        "shards": summary["shards"],
    }
    for key, value in actual.items():
        if source.get(key) != value:
            raise RuntimeError(
                f"expert/source checkpoint mismatch: {key}="
                f"{source.get(key)!r}, expected={value!r}"
            )


def _resolve_b200_inference_inputs(
    *,
    source_dir: Path | None,
    expert_checkpoint_dir: Path | None,
    config_dir: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    bundle = load_b200_config_bundle(config_dir)
    runtime = bundle["model"].get("inference_runtime")
    if not isinstance(runtime, Mapping):
        raise RuntimeError("B200 model config lacks inference_runtime")
    configured_source = Path(str(runtime.get("source_dir")))
    configured_experts = Path(str(runtime.get("expert_dir")))
    if configured_source != BF16_SOURCE_DIR:
        raise RuntimeError(
            "B200 inference source path drifted from the pinned snapshot: "
            f"{configured_source} != {BF16_SOURCE_DIR}"
        )
    if configured_experts != INFERENCE_EXPERT_DIR:
        raise RuntimeError(
            "B200 inference NF4 path drifted: "
            f"{configured_experts} != {INFERENCE_EXPERT_DIR}"
        )
    resolved_source = require_owned_path(
        BF16_SOURCE_DIR if source_dir is None else source_dir,
        must_exist=True,
    )
    resolved_experts = require_owned_path(
        (
            INFERENCE_EXPERT_DIR
            if expert_checkpoint_dir is None
            else expert_checkpoint_dir
        ),
        must_exist=True,
    )
    if resolved_source != BF16_SOURCE_DIR:
        raise RuntimeError(
            f"inference source must equal pinned B200 source: {resolved_source}"
        )
    if resolved_experts != INFERENCE_EXPERT_DIR:
        raise RuntimeError(
            "inference NF4 path must equal sealed B200 artifact: " f"{resolved_experts}"
        )
    identity = b200_config_identity(config_dir)
    return resolved_source, resolved_experts, identity


def audit_inference_model(
    model: nn.Module,
    *,
    layer_device_map: Mapping[int, str],
) -> dict[str, Any]:
    """Fail closed on meta/offloaded tensors and the per-expert layout."""

    if bool(getattr(model, "is_gradient_checkpointing", False)):
        raise RuntimeError("inference model unexpectedly uses gradient checkpointing")
    if model.config.use_cache is not True:
        raise RuntimeError("inference model must enable the generation cache")
    placement = audit_parameter_and_buffer_placement(
        model,
        layer_device_map=layer_device_map,
    )

    routed_layers = 0
    for layer_index, layer in enumerate(model.model.layers):
        experts = layer.mlp.experts
        if not isinstance(experts, SolarOpen2PerExpertRoutedExperts):
            raise RuntimeError(
                f"layer {layer_index} uses {type(experts).__name__}, "
                "not the sealed per-expert NF4 implementation"
            )
        expected_device = torch.device(layer_device_map[layer_index])
        devices = {parameter.device for parameter in experts.parameters()}
        if devices != {expected_device}:
            raise RuntimeError(f"layer {layer_index} expert devices differ: {devices}")
        if len(experts.experts) != ARCHITECTURE.num_experts:
            raise RuntimeError(
                f"layer {layer_index} expert count changed: " f"{len(experts.experts)}"
            )
        routed_layers += 1

    expected_lora_targets = expected_lora_module_names(
        {
            "num_hidden_layers": ARCHITECTURE.num_layers,
            "gqa_layers": list(range(0, ARCHITECTURE.num_layers, 4)),
            "kda_use_full_proj": False,
        }
    )
    modules = dict(model.named_modules())
    missing_targets = sorted(expected_lora_targets - set(modules))
    wrong_target_types = {
        name: type(modules[name]).__name__
        for name in expected_lora_targets
        if name in modules and type(modules[name]) is not nn.Linear
    }
    if missing_targets or wrong_target_types:
        raise RuntimeError(
            "base LoRA target contract changed: "
            f"missing={missing_targets[:8]} "
            f"wrong_types={list(wrong_target_types.items())[:8]}"
        )
    trainable = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if trainable:
        raise RuntimeError(f"base model has trainable parameters: {trainable[:8]}")
    handles = getattr(model, "_solar_model_parallel_hook_handles", None)
    if not isinstance(handles, list) or len(handles) != 51:
        raise RuntimeError("model-parallel transfer hooks are incomplete")
    return {
        "status": "PASS",
        "routed_nf4_layout": "per_expert",
        "routed_nf4_layers": routed_layers,
        "routed_experts_per_layer": ARCHITECTURE.num_experts,
        "lora_target_bf16_linear_count": len(expected_lora_targets),
        "lora_target_names_sha256": canonical_sha256(sorted(expected_lora_targets)),
        "meta_parameter_count": 0,
        "meta_buffer_count": 0,
        "non_cuda_parameter_count": 0,
        "non_cuda_buffer_count": 0,
        "placement_audit": placement,
        "base_trainable_parameter_count": 0,
        "model_parallel_hook_count": len(handles),
        "gradient_checkpointing": False,
        "use_cache": True,
        "device_map": dict(model.hf_device_map),
    }


def load_inference_model(
    *,
    source_dir: Path | None = None,
    expert_checkpoint_dir: Path | None = None,
    config_dir: Path = B200_CONFIG_DIR,
    layer_device_map: Mapping[int, str] | None = None,
    verify_hashes: bool = True,
    asset_integrity_receipt: Path | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a frozen two-B200 model without native BF16 expert materialization."""

    if layer_device_map is None:
        layer_device_map = default_layer_device_map()
    source_dir, expert_checkpoint_dir, config_identity = _resolve_b200_inference_inputs(
        source_dir=source_dir,
        expert_checkpoint_dir=expert_checkpoint_dir,
        config_dir=config_dir,
    )
    hardware = validate_hardware(layer_device_map)
    try:
        import accelerate
        import transformers
        from accelerate import init_empty_weights
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            GenerationConfig,
        )
    except ImportError as error:
        raise RuntimeError(
            "the pinned Solar Transformers and Accelerate environment is required"
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
            role="inference",
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
    expert_preflight = preflight_output_checkpoint(
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
                    f"B200 inherited inference NF4 {label} mismatch: "
                    f"{actual!r} != {expected!r}"
                )
    expert_manifest = read_json_no_duplicates(
        Path(expert_checkpoint_dir) / MANIFEST_FILENAME
    )
    manifest_status = expert_manifest.get("status")
    if manifest_status != "SEALED":
        raise RuntimeError(f"inference NF4 artifact is not SEALED: {manifest_status!r}")
    _validate_source_binding(inventory, expert_manifest, inventory_summary)

    config = AutoConfig.from_pretrained(
        inventory.source_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    if config.tie_word_embeddings is not False:
        raise RuntimeError("official pinned config must keep embed/lm_head distinct")
    config.use_cache = True
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
        install_nf4_experts(
            model,
            expert_checkpoint_dir,
            layer_device_map=layer_device_map,
            verify_hashes=False,
        )
    rotary_class = type(model.model.rotary_emb)
    model.model.rotary_emb = rotary_class(config=config, device="cuda:0")
    if any(buffer.is_meta for buffer in model.model.rotary_emb.buffers()):
        raise RuntimeError("official rotary buffers remained on the meta device")

    nonexpert_load = stream_nonexpert_tensors(
        model,
        inventory,
        layer_device_map=layer_device_map,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    generation_config_path = inventory.source_dir / "generation_config.json"
    if generation_config_path.is_file():
        model.generation_config = GenerationConfig.from_pretrained(
            inventory.source_dir,
            local_files_only=True,
        )
    else:
        model.generation_config = GenerationConfig.from_model_config(config)
    attach_differentiable_model_parallel_hooks(
        model,
        layer_device_map=layer_device_map,
    )
    model.eval()
    audit = audit_inference_model(
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
        "hash_verification": True,
        "hash_verification_mode": hash_verification_mode,
        "direct_large_shard_hashes": verify_hashes,
        "payload_rehash_performed": payload_rehash_performed,
        "asset_integrity_evidence": asset_integrity_evidence,
        "production_eligible": True,
    }
    return model, receipt
