#!/usr/bin/env python3
"""CPU contract tests and optional SM120 bitsandbytes integration smoke."""

from __future__ import annotations

import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

try:
    from ..b200_runtime import BF16_SOURCE_DIR
except ImportError:
    from b200_runtime import BF16_SOURCE_DIR

from .bnb_nf4 import linear4bit_from_packed, quantize_nf4_matrix
from .checkpoint import (
    _validate_config,
    read_safetensors_header,
    validate_pinned_index_key_contract,
)
from .contract import (
    ARCHITECTURE,
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE_DIR,
    EXPECTED_SOURCE_TENSOR_COUNT,
    LAYER_FLAT_EXPERTS_PER_CHUNK,
    PACKED_STATE_SUFFIXES,
    ArchitectureContract,
    layer_flat_chunk_ranges,
    layer_flat_weight_key,
    native_shape,
    output_shape,
    output_weight_key,
    sha256_file,
)
from .convert import (
    ShardPlan,
    _safetensors_metadata,
    _validate_shard_metadata,
    make_plan,
)
from .expert_module import SolarOpen2PerExpertRoutedExperts
from .layer_flat_checkpoint import expected_flat_keys
from .layer_flat_checkpoint import (
    _quantization_contract as flat_quantization_contract,
    preflight_layer_flat_checkpoint,
)
from .layer_flat_module import (
    SolarOpen2LayerFlatDenseReference,
    SolarOpen2LayerFlatNF4Experts,
    audit_input_gradient,
    dequantized_layer_bytes,
)
from .output_checkpoint import expected_output_keys, read_json_no_duplicates


class ContractTests(unittest.TestCase):
    def test_production_counts(self) -> None:
        self.assertEqual(ARCHITECTURE.native_tensor_count, 96)
        self.assertEqual(ARCHITECTURE.output_linear_count, 46_080)
        self.assertEqual(
            ARCHITECTURE.routed_parameter_elements,
            241_591_910_400,
        )
        self.assertEqual(
            ARCHITECTURE.routed_parameter_elements // 2,
            120_795_955_200,
        )
        self.assertEqual(len(expected_flat_keys()), 2_880)
        self.assertEqual(len(expected_output_keys()), 276_480)
        self.assertEqual(EXPECTED_SOURCE_TENSOR_COUNT, 47_019)

    def test_runtime_paths_are_under_assigned_root(self) -> None:
        self.assertIn(DEFAULT_ARTIFACT_ROOT, DEFAULT_SOURCE_DIR.parents)
        self.assertIn(DEFAULT_ARTIFACT_ROOT, DEFAULT_OUTPUT_DIR.parents)
        self.assertIn("hf_cache/models--upstage", str(DEFAULT_SOURCE_DIR))

    def test_downloaded_pinned_index_contract(self) -> None:
        path = BF16_SOURCE_DIR / "model.safetensors.index.json"
        self.assertTrue(path.is_file(), path)
        receipt = validate_pinned_index_key_contract(path)
        self.assertEqual(receipt["tensor_count"], 47_019)
        self.assertEqual(receipt["routed_source_tensor_count"], 46_080)
        self.assertEqual(
            receipt["routed_source_layout"],
            "per_expert_gate_up_down",
        )
        self.assertNotEqual(receipt["embed_shard"], receipt["lm_head_shard"])

    def test_key_mapping_and_shapes(self) -> None:
        self.assertEqual(
            native_shape("gate_up_proj"),
            (320, 2560, 4096),
        )
        self.assertEqual(native_shape("down_proj"), (320, 4096, 1280))
        self.assertEqual(output_shape("gate_proj"), (1280, 4096))
        self.assertEqual(output_shape("down_proj"), (4096, 1280))
        self.assertEqual(
            output_weight_key(3, 7, "up_proj"),
            "model.layers.3.mlp.experts.7.up_proj.weight",
        )
        self.assertEqual(
            layer_flat_weight_key(3, "gate_up_proj"),
            "model.layers.3.mlp.experts.gate_up_proj.chunks.0.weight",
        )
        self.assertEqual(
            layer_flat_chunk_ranges(),
            ((0, 64), (64, 128), (128, 192), (192, 256), (256, 320)),
        )
        maximum_chunk_elements = (
            LAYER_FLAT_EXPERTS_PER_CHUNK
            * 2
            * ARCHITECTURE.intermediate_size
            * ARCHITECTURE.hidden_size
        )
        self.assertEqual(maximum_chunk_elements, 671_088_640)
        self.assertLess(maximum_chunk_elements, 2_147_483_647)

    def test_raw_config_uses_canonical_expert_field(self) -> None:
        config = {
            "model_type": "solar_open2",
            "hidden_size": ARCHITECTURE.hidden_size,
            "num_hidden_layers": ARCHITECTURE.num_layers,
            "n_routed_experts": ARCHITECTURE.num_experts,
            "moe_intermediate_size": ARCHITECTURE.intermediate_size,
            "architectures": ["SolarOpen2ForCausalLM"],
            "dtype": "bfloat16",
            "tie_word_embeddings": False,
        }
        _validate_config(config, ARCHITECTURE)
        alias_only = dict(config)
        alias_only["num_local_experts"] = alias_only.pop(
            "n_routed_experts"
        )
        with self.assertRaisesRegex(RuntimeError, "n_routed_experts"):
            _validate_config(alias_only, ARCHITECTURE)

    def test_deterministic_per_expert_plan(self) -> None:
        contract = ArchitectureContract(
            hidden_size=4,
            intermediate_size=2,
            num_layers=2,
            num_experts=5,
        )
        first = make_plan(experts_per_shard=2, contract=contract)
        second = make_plan(experts_per_shard=2, contract=contract)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        self.assertEqual(
            (first[-1].layer, first[-1].expert_start, first[-1].expert_end),
            (1, 4, 5),
        )


class DenseMappingTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.experts = 3
        self.hidden = 4
        self.intermediate = 2
        self.gate_up = torch.arange(
            self.experts * 2 * self.intermediate * self.hidden,
            dtype=torch.float32,
        ).reshape(
            self.experts,
            2 * self.intermediate,
            self.hidden,
        ) / 100.0
        self.down = torch.arange(
            self.experts * self.hidden * self.intermediate,
            dtype=torch.float32,
        ).reshape(
            self.experts,
            self.hidden,
            self.intermediate,
        ) / 50.0
        self.indices = torch.tensor(
            [[0, 2], [1, 0], [2, 1], [0, 1]],
            dtype=torch.long,
        )
        self.weights = torch.tensor(
            [[0.7, 0.3], [0.4, 0.6], [0.8, 0.2], [0.1, 0.9]],
            dtype=torch.float32,
        )

    def test_split_has_no_transpose_or_reordering(self) -> None:
        split = SolarOpen2PerExpertRoutedExperts.from_native_tensors(
            self.gate_up,
            self.down,
        )
        for expert in range(self.experts):
            self.assertTrue(
                torch.equal(
                    split.experts[expert].gate_proj.weight,
                    self.gate_up[expert, : self.intermediate, :],
                )
            )
            self.assertTrue(
                torch.equal(
                    split.experts[expert].up_proj.weight,
                    self.gate_up[expert, self.intermediate :, :],
                )
            )
            self.assertTrue(
                torch.equal(
                    split.experts[expert].down_proj.weight,
                    self.down[expert, :, :],
                )
            )
            recombined = torch.cat(
                [
                    split.experts[expert].gate_proj.weight,
                    split.experts[expert].up_proj.weight,
                ],
                dim=0,
            )
            self.assertTrue(torch.equal(recombined, self.gate_up[expert]))

    def test_split_forward_and_input_gradient_match_native(self) -> None:
        native = SolarOpen2LayerFlatDenseReference(self.gate_up, self.down)
        split = SolarOpen2PerExpertRoutedExperts.from_native_tensors(
            self.gate_up,
            self.down,
        )
        hidden_native = torch.randn(4, self.hidden, requires_grad=True)
        hidden_split = hidden_native.detach().clone().requires_grad_(True)
        native_output = native(hidden_native, self.indices, self.weights)
        split_output = split(hidden_split, self.indices, self.weights)
        # Individual nn.Linear calls and functional slices can select slightly
        # different CPU GEMM kernels; the underlying split weights are asserted
        # bitwise above.
        torch.testing.assert_close(
            split_output,
            native_output,
            rtol=1e-6,
            atol=2e-9,
        )
        native_output.square().sum().backward()
        split_output.square().sum().backward()
        torch.testing.assert_close(
            hidden_split.grad,
            hidden_native.grad,
            rtol=1e-6,
            atol=1e-7,
        )

    def test_layer_flat_dense_input_gradient_is_finite(self) -> None:
        native = SolarOpen2LayerFlatDenseReference(self.gate_up, self.down)
        receipt = audit_input_gradient(
            native,
            hidden_size=self.hidden,
            num_experts=self.experts,
            device="cpu",
            dtype=torch.float32,
        )
        self.assertEqual(receipt["status"], "PASS")
        self.assertGreater(receipt["input_grad_norm"], 0)

    def test_router_bounds_fail_closed(self) -> None:
        native = SolarOpen2LayerFlatDenseReference(self.gate_up, self.down)
        hidden = torch.randn(1, self.hidden)
        with self.assertRaisesRegex(RuntimeError, "out-of-range"):
            native(
                hidden,
                torch.tensor([[self.experts]]),
                torch.ones(1, 1),
            )
        with self.assertRaisesRegex(RuntimeError, "shape differs"):
            native(
                hidden,
                torch.tensor([[0]]),
                torch.ones(1, 2),
            )

    def test_wrong_native_shape_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "down_proj shape"):
            SolarOpen2PerExpertRoutedExperts.from_native_tensors(
                self.gate_up,
                torch.empty(self.experts, self.intermediate, self.hidden),
            )

    def test_layer_flat_runtime_bytes(self) -> None:
        self.assertEqual(
            dequantized_layer_bytes(
                num_experts=320,
                hidden_size=4096,
                intermediate_size=1280,
            ),
            10_066_329_600,
        )


class HeaderTests(unittest.TestCase):
    def test_per_expert_metrics_survive_atomic_shard_receipt_gap(self) -> None:
        with tempfile.TemporaryDirectory(prefix="solar-nf4-test-") as temporary:
            path = Path(temporary) / "toy.safetensors"
            item = ShardPlan(
                filename=path.name,
                layer=2,
                expert_start=16,
                expert_end=32,
            )
            owner = {
                "source_index_sha256": "1" * 64,
                "plan_sha256": "2" * 64,
            }
            expected = (0.12345678901234567, 0.9876543210987654, 0.03125)
            save_file(
                {"a": torch.ones(1)},
                str(path),
                metadata=_safetensors_metadata(
                    item,
                    owner,
                    maximum_relative_rmse=expected[0],
                    minimum_cosine_similarity=expected[1],
                    maximum_quant_state_ratio=expected[2],
                ),
            )
            self.assertEqual(
                _validate_shard_metadata(path, item, owner),
                expected,
            )

    def test_header_accounts_every_payload_byte(self) -> None:
        with tempfile.TemporaryDirectory(prefix="solar-nf4-test-") as temporary:
            path = Path(temporary) / "toy.safetensors"
            tensors = {
                "a": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
                "b": torch.arange(7, dtype=torch.uint8),
            }
            save_file(tensors, str(path), metadata={"test": "true"})
            records = read_safetensors_header(path)
            self.assertEqual(set(records), {"a", "b"})
            self.assertEqual(records["a"].shape, (3, 4))
            self.assertEqual(records["a"].dtype, "BF16")
            self.assertEqual(records["a"].nbytes, 24)
            self.assertEqual(records["b"].nbytes, 7)

    def test_duplicate_json_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="solar-nf4-test-") as temporary:
            path = Path(temporary) / "duplicate.json"
            path.write_text('{"a": 1, "a": 2}', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "duplicate JSON key"):
                read_json_no_duplicates(path)

    def test_duplicate_safetensors_header_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="solar-nf4-test-") as temporary:
            path = Path(temporary) / "duplicate.safetensors"
            header = b'{"__metadata__":{},"__metadata__":{}}'
            path.write_bytes(struct.pack("<Q", len(header)) + header)
            with self.assertRaisesRegex(
                RuntimeError,
                "duplicate safetensors header key",
            ):
                read_safetensors_header(path)

    def test_toy_layer_flat_checkpoint_accounts_all_weights(self) -> None:
        contract = ArchitectureContract(
            hidden_size=128,
            intermediate_size=64,
            num_layers=2,
            num_experts=8,
        )
        with tempfile.TemporaryDirectory(prefix="solar-nf4-test-") as temporary:
            directory = Path(temporary)
            weight_map = {}
            file_records = []
            total_size = 0
            for layer in range(contract.num_layers):
                filename = f"experts-layer-{layer:03d}-flat.safetensors"
                path = directory / filename
                tensors = {}
                logical = 0
                for projection in ("gate_up_proj", "down_proj"):
                    if projection == "gate_up_proj":
                        elements = (
                            contract.num_experts
                            * 2
                            * contract.intermediate_size
                            * contract.hidden_size
                        )
                    else:
                        elements = (
                            contract.num_experts
                            * contract.hidden_size
                            * contract.intermediate_size
                        )
                    logical += elements
                    base = layer_flat_weight_key(
                        layer, projection, contract=contract
                    )
                    tensors[base] = torch.zeros(
                        elements // 4, dtype=torch.bfloat16
                    )
                    block_count = (elements + 63) // 64
                    tensors[base + ".absmax"] = torch.zeros(
                        block_count, dtype=torch.uint8
                    )
                    tensors[base + ".quant_map"] = torch.zeros(
                        16, dtype=torch.float32
                    )
                    tensors[base + ".nested_absmax"] = torch.zeros(
                        (block_count + 255) // 256,
                        dtype=torch.float32,
                    )
                    tensors[base + ".nested_quant_map"] = torch.zeros(
                        256, dtype=torch.float32
                    )
                    tensors[
                        base + ".quant_state.bitsandbytes__nf4"
                    ] = torch.zeros(16, dtype=torch.uint8)
                save_file(tensors, str(path))
                header = read_safetensors_header(path)
                payload = sum(record.nbytes for record in header.values())
                total_size += payload
                for key in header:
                    weight_map[key] = filename
                file_records.append(
                    {
                        "filename": filename,
                        "layer": layer,
                        "file_bytes": path.stat().st_size,
                        "payload_bytes": payload,
                        "tensor_count": len(header),
                        "logical_parameter_elements": logical,
                        "sha256": sha256_file(path),
                        "maximum_relative_rmse": 0.10,
                        "minimum_cosine_similarity": 0.99,
                        "maximum_quant_state_ratio": 0.02,
                        "dequantized_runtime_bytes": logical * 2,
                    }
                )
            index_path = directory / "expert_nf4.safetensors.index.json"
            index_path.write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": total_size},
                        "weight_map": weight_map,
                    }
                ),
                encoding="utf-8",
            )
            index_sha = sha256_file(index_path)
            manifest = {
                "schema": "solar-open2-layer-flat-routed-experts-nf4",
                "schema_version": 1,
                "status": "SEALED",
                "complete_model_checkpoint": False,
                "base_model_id": "upstage/Solar-Open2-250B",
                "base_revision": (
                    "8d1dda5ad2af58207c65348d825772d2f4d72481"
                ),
                "architecture": contract.to_dict(),
                "quantization": flat_quantization_contract(),
                "files": file_records,
                "expert_index_sha256": index_sha,
                "source": {
                    "source_dir": "/pinned/snapshot",
                    "index_sha256": "a" * 64,
                    "config_sha256": "b" * 64,
                    "weight_map_sha256": "c" * 64,
                    "tensor_count": 7,
                    "stored_elements": 11,
                    "stored_tensor_bytes": 22,
                    "shard_count": 1,
                    "routed_source_layout": "per_expert_gate_up_down",
                    "routed_source_tensor_count": (
                        contract.source_expert_tensor_count
                    ),
                    "routed_parameter_elements": (
                        contract.routed_parameter_elements
                    ),
                },
            }
            (directory / "expert_nf4_manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            receipt = preflight_layer_flat_checkpoint(
                directory,
                contract=contract,
                hash_shards=True,
            )
            self.assertEqual(receipt["status"], "PASS")
            self.assertEqual(
                receipt["logical_parameter_elements"],
                contract.routed_parameter_elements,
            )


def _bnb_sm120_available() -> bool:
    if importlib.util.find_spec("bitsandbytes") is None:
        return False
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] >= 12
    )


@unittest.skipUnless(
    _bnb_sm120_available(),
    "requires bitsandbytes==0.49.2 on an allocated SM120 GPU",
)
class BitsAndBytesSM120Tests(unittest.TestCase):
    def test_layer_flat_nf4_forward_and_input_backward(self) -> None:
        torch.manual_seed(23)
        experts, hidden, intermediate = 2, 16, 8
        gate_native = torch.randn(
            experts,
            2 * intermediate,
            hidden,
            dtype=torch.bfloat16,
        )
        down_native = torch.randn(
            experts,
            hidden,
            intermediate,
            dtype=torch.bfloat16,
        )
        gate_packed = quantize_nf4_matrix(
            gate_native.view(experts * 2 * intermediate, hidden),
            device="cuda:0",
        )
        down_packed = quantize_nf4_matrix(
            down_native.view(experts * hidden, intermediate),
            device="cuda:0",
        )
        gate_linear = linear4bit_from_packed(
            tensors=gate_packed.tensors,
            in_features=hidden,
            out_features=experts * 2 * intermediate,
            device="cuda:0",
        )
        down_linear = linear4bit_from_packed(
            tensors=down_packed.tensors,
            in_features=intermediate,
            out_features=experts * hidden,
            device="cuda:0",
        )
        module = SolarOpen2LayerFlatNF4Experts(
            [gate_linear],
            [down_linear],
            num_experts=experts,
            hidden_size=hidden,
            intermediate_size=intermediate,
        )
        receipt = audit_input_gradient(
            module,
            hidden_size=hidden,
            num_experts=experts,
            device="cuda:0",
            dtype=torch.bfloat16,
        )
        self.assertEqual(receipt["status"], "PASS")
        self.assertGreater(receipt["input_grad_norm"], 0)
        for parameter in module.parameters():
            self.assertFalse(parameter.requires_grad)


if __name__ == "__main__":
    unittest.main()
