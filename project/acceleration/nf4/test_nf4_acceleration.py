from __future__ import annotations

import unittest

import torch
from torch import nn

from adaptive_batch import (
    ContiguousTokenBudgetPolicy,
    flatten_planned_rows,
    run_with_contiguous_oom_backoff,
)
from expert_dispatch import _host_dedup_forward, sorted_active_experts_host_dedup
from lora_merge import merge_and_validate_for_disposable_process


class AdaptiveBatchTest(unittest.TestCase):
    def test_plan_preserves_order_and_budget(self) -> None:
        rows = list("abcdefg")
        lengths = [100, 110, 900, 120, 125, 130, 140]
        policy = ContiguousTokenBudgetPolicy(
            max_batch_size=4,
            max_padded_prompt_tokens=1_000,
        )
        plan = policy.plan(lengths)
        self.assertEqual(flatten_planned_rows(rows, plan), rows)
        self.assertTrue(all(batch.size <= 4 for batch in plan))
        self.assertTrue(
            all(
                batch.size == 1 or batch.padded_prompt_tokens <= 1_000
                for batch in plan
            )
        )

    def test_oom_backoff_is_contiguous_and_order_preserving(self) -> None:
        class FakeOOM(RuntimeError):
            pass

        calls: list[list[int]] = []
        clears: list[None] = []

        def generate(rows: list[int]) -> list[int]:
            calls.append(list(rows))
            if len(rows) > 2:
                raise FakeOOM("oom")
            return [value * 10 for value in rows]

        results, receipt = run_with_contiguous_oom_backoff(
            [1, 2, 3, 4, 5],
            generate,
            is_oom=lambda error: isinstance(error, FakeOOM),
            clear_after_oom=lambda: clears.append(None),
        )
        self.assertEqual(results, [10, 20, 30, 40, 50])
        self.assertEqual(receipt.oom_batch_sizes, [5, 3])
        self.assertEqual(receipt.successful_batch_sizes, [2, 1, 2])
        self.assertEqual(len(clears), 2)
        self.assertEqual(calls, [[1, 2, 3, 4, 5], [1, 2], [3, 4, 5], [3], [4, 5]])

    def test_non_oom_propagates_without_clear(self) -> None:
        clears: list[None] = []
        with self.assertRaisesRegex(ValueError, "bad input"):
            run_with_contiguous_oom_backoff(
                [1, 2],
                lambda _rows: (_ for _ in ()).throw(ValueError("bad input")),
                is_oom=lambda _error: False,
                clear_after_oom=lambda: clears.append(None),
            )
        self.assertEqual(clears, [])


class _ToyExpert(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("weight", weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value @ self.weight


class _ToyRouted(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden_dim = 3
        self.intermediate_dim = 3
        self.experts = nn.ModuleList(
            [
                _ToyExpert(torch.eye(3) * 1.0),
                _ToyExpert(torch.eye(3) * 2.0),
                _ToyExpert(torch.eye(3) * 3.0),
            ]
        )
        self.num_experts = 3


class ExpertDispatchTest(unittest.TestCase):
    def test_host_dedup_matches_sorted_unique(self) -> None:
        indices = torch.tensor([[2, 0], [1, 2]], dtype=torch.int64)
        self.assertEqual(sorted_active_experts_host_dedup(indices, 3), [0, 1, 2])
        self.assertEqual(
            sorted_active_experts_host_dedup(indices, 3),
            torch.unique(indices).tolist(),
        )

    def test_forward_matches_canonical_algorithm(self) -> None:
        module = _ToyRouted()
        hidden = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        indices = torch.tensor([[2, 0], [1, 2]], dtype=torch.int64)
        weights = torch.tensor([[0.7, 0.3], [0.4, 0.6]])

        expected = torch.zeros_like(hidden)
        for expert_index in torch.unique(indices).tolist():
            token_index, position = torch.where(indices == expert_index)
            values = module.experts[expert_index](hidden[token_index])
            expected.index_add_(
                0, token_index, values * weights[token_index, position, None]
            )
        actual = _host_dedup_forward(module, hidden, indices, weights)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_invalid_router_index_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "invalid expert"):
            sorted_active_experts_host_dedup(torch.tensor([[3]]), 3)


class _Config:
    use_cache = False


class _FakeMerged(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))
        self.config = _Config()
        self.hf_device_map: dict[str, str] = {}


class _FakePeft(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": nn.Linear(1, 1, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(1, 1, bias=False)})
        self.hf_device_map = {"layer": "cuda:0"}
        self.config = _Config()
        self.merge_args: tuple[bool, bool] | None = None

    def merge_and_unload(self, *, safe_merge: bool, progressbar: bool) -> nn.Module:
        self.merge_args = (safe_merge, progressbar)
        merged = _FakeMerged()
        merged.hf_device_map = dict(self.hf_device_map)
        return merged


class LoraMergeTest(unittest.TestCase):
    def test_merge_requires_exact_probe_parity_and_preserves_map(self) -> None:
        model = _FakePeft()
        merged, receipt = merge_and_validate_for_disposable_process(
            model,
            parity_probe=lambda _model: [[1, 2, 129]],
            expected_lora_parameter_count=2,
        )
        self.assertEqual(model.merge_args, (True, False))
        self.assertEqual(merged.hf_device_map, {"layer": "cuda:0"})
        self.assertTrue(merged.config.use_cache)
        self.assertEqual(receipt.status, "PASS")
        self.assertTrue(receipt.exact_probe_parity)

    def test_merge_parity_failure_is_fatal(self) -> None:
        model = _FakePeft()
        calls = 0

        def probe(_model: nn.Module) -> list[int]:
            nonlocal calls
            calls += 1
            return [calls]

        with self.assertRaisesRegex(RuntimeError, "changed exact greedy probe"):
            merge_and_validate_for_disposable_process(model, parity_probe=probe)


if __name__ == "__main__":
    unittest.main()
