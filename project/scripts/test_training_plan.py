#!/usr/bin/env python3
"""CPU tests for the pure Solar training-plan helpers."""

from __future__ import annotations

import math
import copy
import json
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from training_plan import (
    early_stopping_decision,
    group_update_rows,
    occurrences_in_step,
    occurrences_through_step,
    planned_microbatch_counts,
    positive_cosine_learning_rate,
    select_best_full_validation,
    token_length_contract,
    validate_b200_early_stopping_policy,
    validate_early_stopping_policy,
    validate_microbatch_buckets,
)


class ExactTailOccurrenceTests(unittest.TestCase):
    PER_STEP = 16
    TOTAL = 149_048

    def through(self, step: int) -> int:
        return occurrences_through_step(
            step,
            occurrences_per_step=self.PER_STEP,
            total_occurrences=self.TOTAL,
        )

    def in_step(self, step: int) -> int:
        return occurrences_in_step(
            step,
            occurrences_per_step=self.PER_STEP,
            total_occurrences=self.TOTAL,
        )

    def test_exact_one_epoch_tail_is_not_padded(self) -> None:
        self.assertEqual(self.through(0), 0)
        self.assertEqual(self.through(9_315), 149_040)
        self.assertEqual(self.in_step(9_315), 16)
        self.assertEqual(self.through(9_316), 149_048)
        self.assertEqual(self.in_step(9_316), 8)
        self.assertNotEqual(self.through(9_316), 149_056)

    def test_resume_boundaries_before_and_after_tail_are_exact(self) -> None:
        boundaries = {
            9_315: {
                "consumed": 149_040,
                "remaining": 8,
                "next_step_occurrences": 8,
            },
            9_316: {
                "consumed": 149_048,
                "remaining": 0,
                "next_step_occurrences": None,
            },
        }
        for completed_step, expected in boundaries.items():
            with self.subTest(completed_step=completed_step):
                consumed = self.through(completed_step)
                self.assertEqual(consumed, expected["consumed"])
                self.assertEqual(self.TOTAL - consumed, expected["remaining"])
                if completed_step < 9_316:
                    self.assertEqual(
                        self.in_step(completed_step + 1),
                        expected["next_step_occurrences"],
                    )

    def test_helpers_fail_closed_outside_exact_horizon(self) -> None:
        for invalid in (-1, 9_317, True, 1.0):
            with self.subTest(helper="through", invalid=invalid):
                with self.assertRaises(ValueError):
                    self.through(invalid)  # type: ignore[arg-type]
        for invalid in (0, -1, 9_317, True, 1.0):
            with self.subTest(helper="in_step", invalid=invalid):
                with self.assertRaises(ValueError):
                    self.in_step(invalid)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "occurrences_per_step"):
            occurrences_through_step(
                0,
                occurrences_per_step=0,
                total_occurrences=self.TOTAL,
            )
        with self.assertRaisesRegex(ValueError, "total_occurrences"):
            occurrences_through_step(
                0,
                occurrences_per_step=self.PER_STEP,
                total_occurrences=True,
            )


class LearningRateTests(unittest.TestCase):
    def test_every_data_bearing_update_has_positive_lr(self) -> None:
        total = 18_631
        warmup = 558
        values = [
            positive_cosine_learning_rate(
                update=step,
                total_updates=total,
                warmup_updates=warmup,
                peak_learning_rate=1e-4,
                minimum_learning_rate_ratio=0.1,
            )
            for step in range(1, total + 1)
        ]
        self.assertTrue(all(math.isfinite(value) and value > 0 for value in values))
        self.assertAlmostEqual(values[0], 1e-4 / warmup)
        self.assertAlmostEqual(values[warmup - 1], 1e-4)
        self.assertGreaterEqual(values[-1], 1e-5)
        self.assertLess(values[-1], values[-2])


class MicrobatchPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.buckets = validate_microbatch_buckets(
            [
                {"max_length": 1024, "batch_size": 8},
                {"max_length": 2048, "batch_size": 4},
                {"max_length": 4096, "batch_size": 2},
                {"max_length": 9600, "batch_size": 1},
            ],
            max_length=9600,
        )

    def test_grouping_is_deterministic_and_lossless(self) -> None:
        lengths = [900, 5000, 1000, 3000, 1800, 700]
        groups = group_update_rows(
            list(range(len(lengths))),
            length_of=lambda index: lengths[index],
            buckets=self.buckets,
        )
        flattened = [index for group in groups for index in group]
        self.assertCountEqual(flattened, range(len(lengths)))
        self.assertEqual([len(group) for group in groups], [3, 1, 1, 1])

    def test_planned_counts_include_partial_final_update(self) -> None:
        lengths = [900] * 17
        result = planned_microbatch_counts(
            range(17),
            lengths=lengths,
            rows_per_update=16,
            buckets=self.buckets,
        )
        self.assertEqual(result["optimizer_updates"], 2)
        self.assertEqual(result["occurrences"], 17)
        self.assertEqual(result["microbatches"], 3)


class LengthAndSelectionTests(unittest.TestCase):
    def test_length_contract_reports_overflow_without_truncating(self) -> None:
        rows = [
            {"id": "same", "task": "translation", "quality": "silver"},
            {"id": "same", "task": "summarization", "quality": "expert"},
        ]
        result = token_length_contract(rows, [9599, 9601], max_length=9600)
        self.assertEqual(result["over_max_count"], 1)
        self.assertEqual(result["over_max_rows"][0]["task"], "summarization")

    def test_best_selection_uses_full_candidates_and_earlier_tie(self) -> None:
        history = [
            {
                "step": 9316,
                "scope": "full",
                "translation_loss": 0.4,
                "summarization_loss": 0.6,
                "selection_loss": 0.47,
            },
            {
                "step": 13973,
                "scope": "full",
                "translation_loss": 0.4,
                "summarization_loss": 0.6,
                "selection_loss": 0.47,
            },
            {
                "step": 18631,
                "scope": "full",
                "translation_loss": 0.5,
                "summarization_loss": 0.6,
                "selection_loss": 0.535,
            },
        ]
        best = select_best_full_validation(
            history,
            candidate_steps=[9316, 13973, 18631],
            translation_weight=0.65,
            summarization_weight=0.35,
        )
        self.assertEqual(best["step"], 9316)


class B200EarlyStoppingTests(unittest.TestCase):
    def setUp(self) -> None:
        recipe = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "config"
                / "b200"
                / "recipe.json"
            ).read_text(encoding="utf-8")
        )
        self.policy = recipe["validation"]["early_stopping"]
        self.full_steps = recipe["validation"]["full_validation_steps"]

    def test_quality_early_stopping_is_exactly_disabled(self) -> None:
        result = validate_b200_early_stopping_policy(
            self.policy,
            optimizer_steps=9_316,
            full_validation_steps=self.full_steps,
        )
        self.assertEqual(result, self.policy)
        self.assertFalse(result["enabled"])
        self.assertEqual(result["earliest_terminal_step"], 9_316)
        self.assertEqual(result["latest_terminal_step"], 9_316)

    def test_quality_early_stopping_cannot_be_silently_enabled(self) -> None:
        mutations = (
            ("enabled", True),
            ("minimum_coverage_epochs", 0.5),
            ("earliest_terminal_step", 6_987),
            ("subset_patience_used_for_stopping", True),
            ("sealed_evaluation_not_monitor", False),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                changed = copy.deepcopy(self.policy)
                changed[field] = value
                with self.assertRaisesRegex(ValueError, field):
                    validate_b200_early_stopping_policy(
                        changed,
                        optimizer_steps=9_316,
                        full_validation_steps=self.full_steps,
                    )


class EarlyStoppingTests(unittest.TestCase):
    def setUp(self) -> None:
        recipe = json.loads(
            (
                Path(__file__).resolve().parents[1] / "config" / "recipe.json"
            ).read_text(encoding="utf-8")
        )
        self.policy = recipe["validation"]["early_stopping"]
        self.full_steps = recipe["validation"]["full_validation_steps"]

    @staticmethod
    def record(
        step: int,
        scope: str,
        selection: float,
        translation: float | None = None,
        summarization: float | None = None,
    ) -> dict[str, object]:
        return {
            "step": step,
            "scope": scope,
            "selection_loss": selection,
            "translation_loss": (
                selection if translation is None else translation
            ),
            "summarization_loss": (
                selection if summarization is None else summarization
            ),
        }

    def decision(
        self,
        history: list[dict[str, object]],
        completed_step: int,
    ) -> dict[str, object]:
        return early_stopping_decision(
            history,
            completed_step=completed_step,
            policy=self.policy,
            optimizer_steps=18_631,
            full_validation_steps=self.full_steps,
            occurrences_per_step=16,
            occurrences_per_epoch=149_048,
        )

    def plateau_history(
        self,
        *,
        confirmation_selection: float = 0.8,
        confirmation_translation: float = 0.7,
        confirmation_summarization: float = 0.9,
    ) -> list[dict[str, object]]:
        return [
            self.record(9000, "fixed_subset", 0.8),
            self.record(9316, "full", 0.8, 0.7, 0.9),
            self.record(10000, "fixed_subset", 0.7995),
            self.record(11000, "fixed_subset", 0.7994),
            self.record(12000, "fixed_subset", 0.7993),
            self.record(13000, "fixed_subset", 0.7992),
            self.record(
                13973,
                "full",
                confirmation_selection,
                confirmation_translation,
                confirmation_summarization,
            ),
        ]

    def test_approved_policy_validates_exactly(self) -> None:
        result = validate_early_stopping_policy(
            self.policy,
            optimizer_steps=18_631,
            full_validation_steps=self.full_steps,
        )
        self.assertEqual(result, self.policy)
        changed = copy.deepcopy(self.policy)
        changed["subset_patience_evaluations"] = 2
        with self.assertRaisesRegex(ValueError, "patience"):
            validate_early_stopping_policy(
                changed,
                optimizer_steps=18_631,
                full_validation_steps=self.full_steps,
            )

    def test_policy_value_types_are_exact(self) -> None:
        mutations = (
            ("enabled", 1),
            ("minimum_epochs", 1),
            ("minimum_step", 9316.0),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                changed = copy.deepcopy(self.policy)
                changed[field] = value
                with self.assertRaisesRegex(ValueError, field):
                    validate_early_stopping_policy(
                        changed,
                        optimizer_steps=18_631,
                        full_validation_steps=self.full_steps,
                    )
        changed = copy.deepcopy(self.policy)
        changed["subset_min_delta"]["absolute"] = "0.001"
        with self.assertRaisesRegex(ValueError, "subset_min_delta"):
            validate_early_stopping_policy(
                changed,
                optimizer_steps=18_631,
                full_validation_steps=self.full_steps,
            )

    def test_decision_is_pure_and_deterministic(self) -> None:
        history = self.plateau_history()
        original = copy.deepcopy(history)
        first = self.decision(history, 13_973)
        second = self.decision(history, 13_973)
        self.assertEqual(first, second)
        self.assertEqual(history, original)

    def test_three_subset_non_improvements_and_full_plateau_trigger(self) -> None:
        result = self.decision(self.plateau_history(), 13_973)
        self.assertEqual(result["status"], "TRIGGERED")
        self.assertTrue(result["triggered"])
        self.assertEqual(result["subset_baseline_step"], 9000)
        self.assertEqual(
            result["considered_subset_steps"],
            [10000, 11000, 12000, 13000],
        )
        self.assertEqual(result["consecutive_non_improvements"], 4)
        self.assertTrue(result["patience_reached"])
        self.assertEqual(result["prior_full_step"], 9316)
        self.assertEqual(result["confirmation_step"], 13973)
        self.assertEqual(
            result["material_improvements"],
            {
                "selection_loss": False,
                "translation_loss": False,
                "summarization_loss": False,
            },
        )
        self.assertEqual(result["completed_candidate_steps"], [9316, 13973])
        self.assertAlmostEqual(
            result["completed_epochs"],
            223_568 / 149_048,
        )

    def test_subset_material_improvement_resets_patience(self) -> None:
        history = self.plateau_history()
        history[3] = self.record(11000, "fixed_subset", 0.797)
        result = self.decision(history, 13_973)
        self.assertEqual(result["status"], "CONTINUE")
        self.assertFalse(result["triggered"])
        self.assertEqual(result["consecutive_non_improvements"], 2)
        self.assertFalse(result["patience_reached"])
        self.assertEqual(result["subset_best_step"], 11000)

    def test_any_material_full_metric_forces_continue(self) -> None:
        cases = (
            (0.798, 0.7, 0.9, "selection_loss"),
            (0.8, 0.698, 0.9, "translation_loss"),
            (0.8, 0.7, 0.898, "summarization_loss"),
        )
        for selection, translation, summary, metric in cases:
            with self.subTest(metric=metric):
                result = self.decision(
                    self.plateau_history(
                        confirmation_selection=selection,
                        confirmation_translation=translation,
                        confirmation_summarization=summary,
                    ),
                    13_973,
                )
                self.assertEqual(result["status"], "CONTINUE")
                self.assertFalse(result["triggered"])
                self.assertTrue(result["material_improvements"][metric])

    def test_confirmation_is_pending_until_full_record_is_durable(self) -> None:
        history = self.plateau_history()[:-1]
        result = self.decision(history, 13_973)
        self.assertEqual(result["status"], "PENDING_CONFIRMATION")
        self.assertFalse(result["triggered"])

    def test_triggered_history_cannot_advance(self) -> None:
        history = self.plateau_history()
        with self.assertRaisesRegex(ValueError, "advanced beyond"):
            self.decision(history, 14_000)

    def test_full_completion_is_maximum_reached(self) -> None:
        history = self.plateau_history(
            confirmation_selection=0.79,
            confirmation_translation=0.69,
            confirmation_summarization=0.89,
        )
        history.extend(
            [
                self.record(14000, "fixed_subset", 0.79),
                self.record(15000, "fixed_subset", 0.78),
                self.record(16000, "fixed_subset", 0.77),
                self.record(17000, "fixed_subset", 0.76),
                self.record(18000, "fixed_subset", 0.75),
                self.record(18631, "full", 0.78, 0.68, 0.88),
            ]
        )
        result = self.decision(history, 18_631)
        self.assertEqual(result["status"], "MAXIMUM_REACHED")
        self.assertFalse(result["triggered"])
        self.assertEqual(
            result["completed_candidate_steps"],
            [9316, 13973, 18631],
        )
        self.assertEqual(result["completed_epochs"], 2.0)


if __name__ == "__main__":
    unittest.main()
