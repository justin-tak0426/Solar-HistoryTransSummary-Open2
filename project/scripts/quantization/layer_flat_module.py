#!/usr/bin/env python3
"""Layer-flat NF4 routed experts with native Solar dispatch semantics."""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn
from torch.nn import functional as F

from .bnb_nf4 import require_bitsandbytes
from .contract import LAYER_FLAT_EXPERTS_PER_CHUNK


def _silu(value: torch.Tensor) -> torch.Tensor:
    return F.silu(value)


class SolarOpen2LayerFlatNF4Experts(nn.Module):
    """Expert-aligned NF4 chunks, dequantized once per decoder forward.

    ``gate_up_proj`` and ``down_proj`` are lists of Linear4bit containers. Their
    ``forward`` methods are not used because their logical output dimensions
    include the expert axis.  The packed weights are dequantized, reshaped to
    the official 3-D layout, and dispatched exactly like SolarOpen2Experts.
    """

    def __init__(
        self,
        gate_up_proj: list[nn.Module] | nn.ModuleList,
        down_proj: list[nn.Module] | nn.ModuleList,
        *,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        dequantize_4bit: Callable[..., torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_size
        self.intermediate_dim = intermediate_size
        self.gate_up_proj = (
            gate_up_proj
            if isinstance(gate_up_proj, nn.ModuleList)
            else nn.ModuleList(gate_up_proj)
        )
        self.down_proj = (
            down_proj
            if isinstance(down_proj, nn.ModuleList)
            else nn.ModuleList(down_proj)
        )
        self.act_fn = _silu
        self._finite_validated = False
        self._chunk_ranges = tuple(
            (
                start,
                min(start + LAYER_FLAT_EXPERTS_PER_CHUNK, num_experts),
            )
            for start in range(
                0, num_experts, LAYER_FLAT_EXPERTS_PER_CHUNK
            )
        )
        if not self._chunk_ranges:
            raise ValueError("at least one expert chunk is required")
        if len(self.gate_up_proj) != len(self._chunk_ranges) or len(
            self.down_proj
        ) != len(self._chunk_ranges):
            raise RuntimeError(
                "packed projection chunk count does not cover every expert"
            )
        if dequantize_4bit is None:
            _torch, bnb = require_bitsandbytes(require_cuda=True)
            dequantize_4bit = bnb.functional.dequantize_4bit
        self._dequantize_4bit = dequantize_4bit
        for chunk, (start, end) in enumerate(self._chunk_ranges):
            chunk_experts = end - start
            self._validate_packed_module(
                self.gate_up_proj[chunk],
                expected_shape=(
                    chunk_experts * 2 * intermediate_size,
                    hidden_size,
                ),
                name=f"gate_up_proj[{chunk}]",
            )
            self._validate_packed_module(
                self.down_proj[chunk],
                expected_shape=(
                    chunk_experts * hidden_size,
                    intermediate_size,
                ),
                name=f"down_proj[{chunk}]",
            )

    @staticmethod
    def _validate_packed_module(
        module: nn.Module,
        *,
        expected_shape: tuple[int, int],
        name: str,
    ) -> None:
        weight = getattr(module, "weight", None)
        quant_state = getattr(weight, "quant_state", None)
        if quant_state is None:
            raise RuntimeError(f"{name} lacks a prequantized bnb QuantState")
        if tuple(quant_state.shape) != expected_shape:
            raise RuntimeError(
                f"{name} QuantState shape mismatch: {tuple(quant_state.shape)} "
                f"!= {expected_shape}"
            )
        if getattr(weight, "requires_grad", True):
            raise RuntimeError(f"{name} routed base weight must be frozen")

    def _dequantized_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.gate_up_proj[0].weight.device
        dtype = self.gate_up_proj[0].weight.quant_state.dtype
        gate_up = torch.empty(
            (
                self.num_experts,
                2 * self.intermediate_dim,
                self.hidden_dim,
            ),
            device=device,
            dtype=dtype,
        )
        down = torch.empty(
            (
                self.num_experts,
                self.hidden_dim,
                self.intermediate_dim,
            ),
            device=device,
            dtype=dtype,
        )
        for chunk, (start, end) in enumerate(self._chunk_ranges):
            gate_module = self.gate_up_proj[chunk]
            down_module = self.down_proj[chunk]
            gate_chunk = self._dequantize_4bit(
                gate_module.weight,
                gate_module.weight.quant_state,
            ).view(
                end - start,
                2 * self.intermediate_dim,
                self.hidden_dim,
            )
            down_chunk = self._dequantize_4bit(
                down_module.weight,
                down_module.weight.quant_state,
            ).view(
                end - start,
                self.hidden_dim,
                self.intermediate_dim,
            )
            if not self._finite_validated and (
                not torch.isfinite(gate_chunk).all()
                or not torch.isfinite(down_chunk).all()
            ):
                raise RuntimeError(
                    f"dequantized routed expert chunk {chunk} contains NaN or Inf"
                )
            gate_up[start:end].copy_(gate_chunk)
            down[start:end].copy_(down_chunk)
            del gate_chunk, down_chunk
        self._finite_validated = True
        return gate_up, down

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        _validate_forward_inputs(
            hidden_states,
            top_k_index,
            top_k_weights,
            hidden_size=self.hidden_dim,
        )
        gate_up_weights, down_weights = self._dequantized_weights()
        final_hidden_states = torch.zeros_like(hidden_states)
        active_experts = torch.unique(top_k_index.detach()).to("cpu").tolist()
        for expert_index in active_experts:
            if not isinstance(expert_index, int) or not (
                0 <= expert_index < self.num_experts
            ):
                raise RuntimeError(
                    f"router selected out-of-range expert {expert_index}"
                )
            token_index, top_k_position = torch.where(
                top_k_index == expert_index
            )
            current_state = hidden_states[token_index]
            gate, up = F.linear(
                current_state, gate_up_weights[expert_index]
            ).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(
                current_hidden_states, down_weights[expert_index]
            )
            current_hidden_states = current_hidden_states * top_k_weights[
                token_index, top_k_position, None
            ]
            final_hidden_states.index_add_(
                0,
                token_index,
                current_hidden_states.to(final_hidden_states.dtype),
            )
        return final_hidden_states


class SolarOpen2LayerFlatDenseReference(nn.Module):
    """Dense equivalent used to prove mapping and input-gradient fidelity."""

    def __init__(
        self,
        gate_up_proj: torch.Tensor,
        down_proj: torch.Tensor,
    ) -> None:
        super().__init__()
        if gate_up_proj.ndim != 3 or down_proj.ndim != 3:
            raise ValueError("dense reference weights must be rank three")
        experts, doubled_intermediate, hidden = gate_up_proj.shape
        if doubled_intermediate % 2:
            raise ValueError("gate_up_proj intermediate dimension must be even")
        intermediate = doubled_intermediate // 2
        if tuple(down_proj.shape) != (experts, hidden, intermediate):
            raise ValueError("down_proj shape is incompatible with gate_up_proj")
        self.num_experts = experts
        self.hidden_dim = hidden
        self.intermediate_dim = intermediate
        self.gate_up_proj = nn.Parameter(
            gate_up_proj.detach().clone(), requires_grad=False
        )
        self.down_proj = nn.Parameter(
            down_proj.detach().clone(), requires_grad=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        _validate_forward_inputs(
            hidden_states,
            top_k_index,
            top_k_weights,
            hidden_size=self.hidden_dim,
        )
        final_hidden_states = torch.zeros_like(hidden_states)
        for expert_index in torch.unique(top_k_index.detach()).to("cpu").tolist():
            if not 0 <= expert_index < self.num_experts:
                raise RuntimeError(
                    f"router selected out-of-range expert {expert_index}"
                )
            token_index, top_k_position = torch.where(
                top_k_index == expert_index
            )
            gate, up = F.linear(
                hidden_states[token_index], self.gate_up_proj[expert_index]
            ).chunk(2, dim=-1)
            current = F.silu(gate) * up
            current = F.linear(current, self.down_proj[expert_index])
            current = current * top_k_weights[
                token_index, top_k_position, None
            ]
            final_hidden_states.index_add_(0, token_index, current)
        return final_hidden_states


def _validate_forward_inputs(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    *,
    hidden_size: int,
) -> None:
    if hidden_states.ndim != 2 or hidden_states.shape[-1] != hidden_size:
        raise RuntimeError(
            f"hidden_states must be [tokens,{hidden_size}], "
            f"got={tuple(hidden_states.shape)}"
        )
    if top_k_index.ndim != 2 or top_k_index.shape[0] != hidden_states.shape[0]:
        raise RuntimeError("top_k_index shape is incompatible with hidden states")
    if top_k_weights.shape != top_k_index.shape:
        raise RuntimeError("top_k_weights shape differs from top_k_index")
    if top_k_index.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise RuntimeError(f"top_k_index is not integral: {top_k_index.dtype}")
    if not (
        hidden_states.device
        == top_k_index.device
        == top_k_weights.device
    ):
        raise RuntimeError("routed expert inputs are on different devices")


def dequantized_layer_bytes(
    *,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    element_size: int = 2,
) -> int:
    return (
        num_experts
        * (
            2 * intermediate_size * hidden_size
            + hidden_size * intermediate_size
        )
        * element_size
    )


def audit_input_gradient(
    module: nn.Module,
    *,
    hidden_size: int,
    num_experts: int,
    device: str,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Any]:
    """Run a finite backward that requires only the routed input gradient."""

    hidden = torch.randn(
        7,
        hidden_size,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    indices = torch.tensor(
        [[row % num_experts, (row + 1) % num_experts] for row in range(7)],
        device=device,
        dtype=torch.long,
    )
    weights = torch.softmax(
        torch.randn(7, 2, device=device, dtype=torch.float32), dim=-1
    ).to(dtype)
    output = module(hidden, indices, weights)
    loss = output.float().square().mean()
    loss.backward()
    if hidden.grad is None or not torch.isfinite(hidden.grad).all():
        raise RuntimeError("layer-flat routed expert did not produce finite input grad")
    norm = hidden.grad.float().norm().item()
    if norm == 0.0:
        raise RuntimeError("layer-flat routed expert input gradient is zero")
    return {
        "status": "PASS",
        "loss": loss.item(),
        "input_grad_norm": norm,
        "output_shape": list(output.shape),
    }
