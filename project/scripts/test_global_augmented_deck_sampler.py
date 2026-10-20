#!/usr/bin/env python3
"""CPU-only contract tests for the finite global augmented deck sampler."""

from __future__ import annotations

import json
import sys
import unittest
from collections import Counter
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import GlobalAugmentedDeckSampler  # noqa: E402


EXPERT_LABEL = "gold_expert_corrected"
SILVER_LABEL = "silver_model_generated"


def identity(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["id"]), str(row["task"])


def encoded_draws(
    sampler: GlobalAugmentedDeckSampler,
    count: int,
) -> bytes:
    return b"\n".join(
        json.dumps(
            sampler.draw(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        for _ in range(count)
    )


def small_rows() -> list[dict[str, Any]]:
    return [
        {
            "id": "shared-expert",
            "task": "translation",
            "quality": EXPERT_LABEL,
            "length": 1_000_000,
            "response": "translation expert",
        },
        {
            "id": "translation-silver",
            "task": "translation",
            "quality": SILVER_LABEL,
            "length": 1,
            "response": "translation silver",
        },
        {
            "id": "shared-expert",
            "task": "summarization",
            "quality": EXPERT_LABEL,
            "length": 2_000_000,
            "response": "summarization expert",
        },
        {
            "id": "summarization-silver",
            "task": "summarization",
            "quality": SILVER_LABEL,
            "length": 2,
            "response": "summarization silver",
        },
    ]


def production_contract_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index in range(97_044):
        expert = index < 740
        rows.append(
            {
                "id": (
                    f"shared-expert-{index}"
                    if expert
                    else f"translation-silver-{index}"
                ),
                "task": "translation",
                "quality": EXPERT_LABEL if expert else SILVER_LABEL,
            }
        )
    for index in range(45_669):
        expert = index < 165
        rows.append(
            {
                "id": (
                    f"shared-expert-{index}"
                    if expert
                    else f"summarization-silver-{index}"
                ),
                "task": "summarization",
                "quality": EXPERT_LABEL if expert else SILVER_LABEL,
            }
        )
    return rows


class GlobalAugmentedDeckSamplerTest(unittest.TestCase):
    def make_small_sampler(
        self,
        *,
        seed: int = 20260727,
    ) -> GlobalAugmentedDeckSampler:
        return GlobalAugmentedDeckSampler(
            small_rows(),
            expert_label=EXPERT_LABEL,
            expert_boost=8,
            epochs=2,
            seed=seed,
        )

    def test_seeded_epoch_shuffle_and_exact_global_cursor_resume(self) -> None:
        first = self.make_small_sampler()
        same_seed = self.make_small_sampler()
        different_seed = self.make_small_sampler(seed=20260728)

        first_sequence = [
            identity(first.draw()) for _ in range(first.total_examples)
        ]
        same_sequence = [
            identity(same_seed.draw())
            for _ in range(same_seed.total_examples)
        ]
        different_sequence = [
            identity(different_seed.draw())
            for _ in range(different_seed.total_examples)
        ]
        self.assertEqual(first_sequence, same_sequence)
        self.assertNotEqual(first_sequence, different_sequence)
        per_epoch = first.examples_per_epoch
        self.assertNotEqual(
            first_sequence[:per_epoch],
            first_sequence[per_epoch:],
        )

        interrupted = self.make_small_sampler()
        prefix_count = interrupted.examples_per_epoch + 3
        encoded_draws(interrupted, prefix_count)
        state = interrupted.state_dict()
        self.assertEqual(state["cursor"], prefix_count)
        expected_remaining = encoded_draws(interrupted, interrupted.remaining)

        resumed = self.make_small_sampler()
        resumed.load_state_dict(state)
        self.assertEqual(resumed.cursor, prefix_count)
        self.assertEqual(
            encoded_draws(resumed, resumed.remaining),
            expected_remaining,
        )
        self.assertEqual(resumed.cursor, resumed.total_examples)
        with self.assertRaises(StopIteration):
            resumed.draw()

    def test_resume_rejects_changed_sampling_contract(self) -> None:
        state = self.make_small_sampler().state_dict()
        changed_seed = self.make_small_sampler(seed=7)
        with self.assertRaisesRegex(ValueError, "contract changed"):
            changed_seed.load_state_dict(state)

        changed_rows = small_rows()
        changed_rows[0]["quality"] = SILVER_LABEL
        changed_data = GlobalAugmentedDeckSampler(
            changed_rows,
            expert_label=EXPERT_LABEL,
            expert_boost=8,
            epochs=2,
            seed=20260727,
        )
        with self.assertRaisesRegex(ValueError, "contract changed"):
            changed_data.load_state_dict(state)

    def test_full_one_epoch_counts_and_row_exposure(self) -> None:
        rows = production_contract_rows()
        sampler = GlobalAugmentedDeckSampler(
            rows,
            expert_label=EXPERT_LABEL,
            expert_boost=8,
            epochs=1,
            seed=42,
        )
        self.assertEqual(len(rows), 142_713)
        self.assertEqual(sampler.examples_per_epoch, 149_048)
        self.assertEqual(len(sampler), 149_048)
        self.assertEqual(
            sampler.stats(),
            {
                "base_rows": 142_713,
                "expert_rows": 905,
                "expert_boost": 8,
                "epochs": 1,
                "examples_per_epoch": 149_048,
                "total_examples": 149_048,
                "base_task_counts": {
                    "translation": 97_044,
                    "summarization": 45_669,
                },
                "expert_task_counts": {
                    "translation": 740,
                    "summarization": 165,
                },
                "task_examples_per_epoch": {
                    "translation": 102_224,
                    "summarization": 46_824,
                },
                "task_total_examples": {
                    "translation": 102_224,
                    "summarization": 46_824,
                },
                "occurrence_identity_schema": (
                    "id/task/quality/expert_ordinal/epoch/position"
                ),
            },
        )

        all_exposure: Counter[tuple[str, str]] = Counter()
        for epoch in range(1):
            epoch_exposure: Counter[tuple[str, str]] = Counter()
            epoch_tasks: Counter[str] = Counter()
            for _ in range(sampler.examples_per_epoch):
                row = sampler.draw()
                row_identity = identity(row)
                epoch_exposure[row_identity] += 1
                all_exposure[row_identity] += 1
                epoch_tasks[str(row["task"])] += 1
            self.assertEqual(
                epoch_tasks,
                {"translation": 102_224, "summarization": 46_824},
            )
            self.assertEqual(len(epoch_exposure), 142_713)
            self.assertEqual(
                Counter(epoch_exposure.values()),
                {1: 141_808, 8: 905},
                msg=f"unexpected row exposure in epoch {epoch}",
            )

        self.assertEqual(sampler.cursor, 149_048)
        self.assertEqual(sampler.remaining, 0)
        self.assertEqual(len(all_exposure), 142_713)
        self.assertEqual(
            Counter(all_exposure.values()),
            {1: 141_808, 8: 905},
        )
        total_tasks = Counter(
            {
                task: sum(
                    count
                    for (_, row_task), count in all_exposure.items()
                    if row_task == task
                )
                for task in ("translation", "summarization")
            }
        )
        self.assertEqual(
            total_tasks,
            {"translation": 102_224, "summarization": 46_824},
        )
        with self.assertRaises(StopIteration):
            sampler.draw()

    def test_exact_tail_occurrence_identities_are_transactional(self) -> None:
        sampler = GlobalAugmentedDeckSampler(
            production_contract_rows(),
            expert_label=EXPERT_LABEL,
            expert_boost=8,
            epochs=1,
            seed=42,
        )
        tail = sampler.occurrences(149_040, 8)
        self.assertEqual(sampler.cursor, 0)
        self.assertEqual(
            [item["global_position"] for item in tail],
            list(range(149_040, 149_048)),
        )
        occurrence_keys = {
            (
                item["id"],
                item["task"],
                item["quality"],
                item["expert_ordinal"],
                item["epoch"],
                item["position"],
            )
            for item in tail
        }
        self.assertEqual(len(occurrence_keys), 8)
        self.assertEqual(len({item["identity_sha256"] for item in tail}), 8)

        state = sampler.state_dict()
        state["cursor"] = 149_040
        sampler.load_state_dict(state)
        drawn = [sampler.draw_occurrence() for _ in range(8)]
        self.assertEqual(drawn, tail)
        self.assertEqual(sampler.cursor, 149_048)
        self.assertEqual(sampler.remaining, 0)

    def test_row_identity_is_id_and_task_and_length_is_not_filtered(self) -> None:
        sampler = self.make_small_sampler()
        exposure = Counter(
            identity(sampler.draw()) for _ in range(sampler.total_examples)
        )
        self.assertEqual(exposure[("shared-expert", "translation")], 16)
        self.assertEqual(exposure[("shared-expert", "summarization")], 16)
        self.assertEqual(exposure[("translation-silver", "translation")], 2)
        self.assertEqual(
            exposure[("summarization-silver", "summarization")],
            2,
        )


if __name__ == "__main__":
    unittest.main()
