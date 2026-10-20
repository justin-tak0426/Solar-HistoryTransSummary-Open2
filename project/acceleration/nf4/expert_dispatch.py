"""Opt-in micro-optimization for the sealed per-expert NF4 dispatcher.

The canonical dispatcher calls ``torch.unique`` on CUDA and then copies the
small result to the host.  Decode batches contain at most ``batch * top_k``
router entries, so copying that tiny vector and deduplicating on the host avoids
one CUDA unique/sort kernel while retaining the same ascending expert order.

This does *not* eliminate the required CUDA-to-host synchronization.  Python
must still learn which independently stored ``Linear4bit`` modules to invoke.
Eliminating that synchronization requires a packed/fused MoE kernel or weight
conversion, both outside this NF4 no-conversion experiment.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HostDedupInstallReceipt:
    patched_layers: int
    implementation: str = "copy_small_router_vector_then_sorted_host_dedup_v1"
    cuda_host_sync_eliminated: bool = False
    cuda_unique_kernel_eliminated: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "patched_layers": self.patched_layers,
            "implementation": self.implementation,
            "cuda_host_sync_eliminated": self.cuda_host_sync_eliminated,
            "cuda_unique_kernel_eliminated": self.cuda_unique_kernel_eliminated,
        }


def sorted_active_experts_host_dedup(top_k_index: Any, num_experts: int) -> list[int]:
    values = top_k_index.detach().reshape(-1).to("cpu").tolist()
    active = sorted(set(values))
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < num_experts
        for index in active
    ):
        raise RuntimeError(
            f"router selected an invalid expert; active={active[:16]} "
            f"num_experts={num_experts}"
        )
    return active


def _host_dedup_forward(
    self: Any,
    hidden_states: Any,
    top_k_index: Any,
    top_k_weights: Any,
) -> Any:
    import torch

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
        raise RuntimeError(f"top_k_index must use an integer dtype, got={top_k_index.dtype}")
    if top_k_weights.device != hidden_states.device:
        raise RuntimeError("routing weights and hidden states are on different devices")
    if top_k_index.device != hidden_states.device:
        raise RuntimeError("routing indices and hidden states are on different devices")

    final_hidden_states = torch.zeros_like(hidden_states)
    active_experts = sorted_active_experts_host_dedup(
        top_k_index, self.num_experts
    )
    for expert_index in active_experts:
        token_index, top_k_position = torch.where(top_k_index == expert_index)
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


def install_host_dedup_dispatch(model: Any) -> HostDedupInstallReceipt:
    """Patch only already-loaded canonical routed-expert instances in place."""

    patched = 0
    for module in model.modules():
        if type(module).__name__ != "SolarOpen2PerExpertRoutedExperts":
            continue
        required = ("experts", "num_experts", "hidden_dim", "intermediate_dim")
        if any(not hasattr(module, name) for name in required):
            raise RuntimeError("canonical routed expert shape contract changed")
        module.forward = types.MethodType(_host_dedup_forward, module)
        patched += 1
    if patched != 48:
        raise RuntimeError(f"expected 48 routed layers, patched {patched}")
    return HostDedupInstallReceipt(patched_layers=patched)
