#!/usr/bin/env python3
"""CPU tests for model-parallel gradient-checkpoint configuration."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch import nn

from . import full_model_loader
from .contract import ARCHITECTURE
from .full_model_loader import (
    GRADIENT_CHECKPOINTING_KWARGS,
    _checkpoint_on_layer_device,
    _target_device_for_key,
    attach_differentiable_model_parallel_hooks,
    audit_parameter_and_buffer_placement,
    configure_model_parallel_gradient_checkpointing,
    configure_selective_model_parallel_gradient_checkpointing,
    default_layer_device_map,
    disable_model_parallel_gradient_checkpointing,
)

try:
    from ..b200_runtime import EXPECTED_HF_DEVICE_MAP
except ImportError:
    from b200_runtime import EXPECTED_HF_DEVICE_MAP


class DummyLayer(nn.Module):
    gradient_checkpointing = False

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


class DummyDecoder(nn.Module):
    def __init__(self, count: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([DummyLayer() for _ in range(count)])


class DummyBase(nn.Module):
    def __init__(self, count: int) -> None:
        super().__init__()
        self.model = DummyDecoder(count)


class DummyPeftModel(nn.Module):
    def __init__(self, count: int = ARCHITECTURE.num_layers) -> None:
        super().__init__()
        self.base = DummyBase(count)
        self.enable_kwargs = None

    def get_base_model(self) -> DummyBase:
        return self.base

    def gradient_checkpointing_enable(
        self,
        *,
        gradient_checkpointing_kwargs: dict,
    ) -> None:
        self.enable_kwargs = gradient_checkpointing_kwargs
        for layer in self.base.model.layers:
            layer.gradient_checkpointing = True
            layer._gradient_checkpointing_func = "generic"

    def gradient_checkpointing_disable(self) -> None:
        for layer in self.base.model.layers:
            layer.gradient_checkpointing = False

    @property
    def is_gradient_checkpointing(self) -> bool:
        return any(layer.gradient_checkpointing for layer in self.base.model.layers)


class ModelParallelCheckpointTests(unittest.TestCase):
    def test_configures_every_layer_after_generic_enable(self) -> None:
        model = DummyPeftModel()
        contract = configure_model_parallel_gradient_checkpointing(model)

        self.assertEqual(model.enable_kwargs, GRADIENT_CHECKPOINTING_KWARGS)
        self.assertEqual(contract["boundary_layers"], [24])
        self.assertEqual(contract["decoder_layer_count"], 48)
        self.assertEqual(
            contract["layer_input_devices"],
            {
                str(layer): "cuda:0" if layer < 24 else "cuda:1"
                for layer in range(ARCHITECTURE.num_layers)
            },
        )
        for layer_index, layer in enumerate(model.base.model.layers):
            self.assertTrue(layer.gradient_checkpointing)
            self.assertEqual(
                layer._gradient_checkpointing_func.keywords["device"],
                default_layer_device_map()[layer_index],
            )
            self.assertEqual(
                layer._solar_checkpoint_input_device,
                default_layer_device_map()[layer_index],
            )

    def test_disables_recomputation_without_changing_device_routing(self) -> None:
        model = DummyPeftModel()
        configure_model_parallel_gradient_checkpointing(model)
        contract = disable_model_parallel_gradient_checkpointing(model)

        self.assertFalse(model.is_gradient_checkpointing)
        self.assertEqual(contract["implementation"], "disabled_b200_memory_optimized")
        self.assertFalse(contract["enabled"])
        self.assertEqual(contract["boundary_layers"], [24])
        for layer_index, layer in enumerate(model.base.model.layers):
            self.assertFalse(layer.gradient_checkpointing)
            self.assertEqual(
                layer._solar_checkpoint_input_device,
                default_layer_device_map()[layer_index],
            )

    def test_selective_checkpointing_is_balanced_and_device_aware(self) -> None:
        model = DummyPeftModel()
        contract = configure_selective_model_parallel_gradient_checkpointing(model)

        expected = [layer for layer in range(ARCHITECTURE.num_layers) if layer % 4 != 3]
        uncheckpointed = [
            layer for layer in range(ARCHITECTURE.num_layers) if layer % 4 == 3
        ]
        self.assertTrue(model.is_gradient_checkpointing)
        self.assertEqual(contract["checkpointed_decoder_layers"], expected)
        self.assertEqual(contract["uncheckpointed_decoder_layers"], uncheckpointed)
        self.assertEqual(contract["checkpointed_decoder_layer_count"], 36)
        self.assertEqual(contract["uncheckpointed_decoder_layer_count"], 12)
        self.assertEqual(sum(layer < 24 for layer in expected), 18)
        self.assertEqual(sum(layer >= 24 for layer in expected), 18)
        self.assertEqual(sum(layer < 24 for layer in uncheckpointed), 6)
        self.assertEqual(sum(layer >= 24 for layer in uncheckpointed), 6)
        for layer_index, layer in enumerate(model.base.model.layers):
            self.assertEqual(layer.gradient_checkpointing, layer_index in expected)
            self.assertEqual(
                layer._gradient_checkpointing_func.keywords["device"],
                default_layer_device_map()[layer_index],
            )

    def test_default_map_is_exact_24_24_without_cuda2(self) -> None:
        device_map = default_layer_device_map()
        self.assertEqual(len(device_map), 48)
        self.assertEqual(
            [device_map[layer] for layer in range(24)],
            ["cuda:0"] * 24,
        )
        self.assertEqual(
            [device_map[layer] for layer in range(24, 48)],
            ["cuda:1"] * 24,
        )
        self.assertNotIn("cuda:2", device_map.values())

    def test_embedding_and_final_endpoints_are_exact(self) -> None:
        device_map = default_layer_device_map()
        self.assertEqual(
            _target_device_for_key("model.embed_tokens.weight", device_map),
            "cuda:0",
        )
        self.assertEqual(
            _target_device_for_key(
                "model.layers.23.self_attn.q_proj.weight", device_map
            ),
            "cuda:0",
        )
        self.assertEqual(
            _target_device_for_key(
                "model.layers.24.self_attn.q_proj.weight", device_map
            ),
            "cuda:1",
        )
        self.assertEqual(
            _target_device_for_key("model.norm.weight", device_map),
            "cuda:1",
        )
        self.assertEqual(
            _target_device_for_key("lm_head.weight", device_map),
            "cuda:1",
        )

    def test_transfer_hooks_publish_exact_hf_device_map(self) -> None:
        model = nn.Module()
        model.model = nn.Module()
        model.model.embed_tokens = nn.Module()
        model.model.layers = nn.ModuleList(
            nn.Module() for _ in range(ARCHITECTURE.num_layers)
        )
        model.model.norm = nn.Module()
        model.lm_head = nn.Module()
        handles = attach_differentiable_model_parallel_hooks(
            model,
            layer_device_map=default_layer_device_map(),
        )
        self.assertEqual(len(handles), 51)
        self.assertEqual(model.hf_device_map, dict(EXPECTED_HF_DEVICE_MAP))
        self.assertNotIn("cuda:2", model.hf_device_map.values())

    def test_placement_audit_rejects_cpu_buffers(self) -> None:
        class BufferOnlyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = nn.Module()
                self.model.embed_tokens = nn.Module()
                self.model.embed_tokens.register_buffer("probe", torch.ones(1))
                self.hf_device_map = dict(EXPECTED_HF_DEVICE_MAP)

        with self.assertRaisesRegex(RuntimeError, "CPU/meta/disk"):
            audit_parameter_and_buffer_placement(
                BufferOnlyModel(),
                layer_device_map=default_layer_device_map(),
            )

    def test_rejects_wrong_decoder_layer_count(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "layer count changed"):
            configure_model_parallel_gradient_checkpointing(DummyPeftModel(count=47))

    def test_wrapper_moves_input_before_checkpoint_and_pins_safety_flags(
        self,
    ) -> None:
        observed: dict[str, object] = {}

        def fake_checkpoint(function, *args, **kwargs):
            observed["device"] = str(args[0].device)
            observed["kwargs"] = kwargs
            return function(*args)

        value = torch.tensor([2.0], requires_grad=True)
        with patch.object(
            full_model_loader,
            "checkpoint",
            side_effect=fake_checkpoint,
        ):
            result = _checkpoint_on_layer_device(
                lambda tensor: tensor.square(),
                value,
                device="cpu",
            )
        result.sum().backward()

        self.assertEqual(observed["device"], "cpu")
        self.assertEqual(observed["kwargs"], GRADIENT_CHECKPOINTING_KWARGS)
        self.assertEqual(value.grad.item(), 4.0)

    def test_wrapper_requires_positional_hidden_tensor(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "checkpoint input"):
            _checkpoint_on_layer_device(lambda: None, device="cpu")
        with self.assertRaisesRegex(RuntimeError, "checkpoint input"):
            _checkpoint_on_layer_device(
                lambda value: value,
                "not-a-tensor",
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
