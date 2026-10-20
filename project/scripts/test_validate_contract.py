#!/usr/bin/env python3
"""CPU tests for the approved Solar recipe contract."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

import validate_contract  # noqa: E402
from common import CONFIG_DIR, read_json  # noqa: E402


B200_CONFIG_DIR = CONFIG_DIR / "b200"


class RecipeContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recipe = read_json(CONFIG_DIR / "recipe.json")
        self.b200_recipe = read_json(B200_CONFIG_DIR / "recipe.json")

    def test_approved_two_epoch_recipe_passes(self) -> None:
        evidence = validate_contract.validate_recipe_contract(self.recipe)
        self.assertEqual(evidence["epochs"], 2)
        self.assertEqual(evidence["optimizer_steps"], 18631)
        self.assertEqual(evidence["max_length"], 9600)
        self.assertEqual(evidence["total_occurrences"], 298096)
        self.assertEqual(
            evidence["full_validation_steps"],
            [9316, 13973, 18631],
        )
        self.assertEqual(
            evidence["early_stopping_policy"],
            self.recipe["validation"]["early_stopping"],
        )

    def test_recipe_name_is_fail_closed(self) -> None:
        recipe = copy.deepcopy(self.recipe)
        recipe["recipe_name"] = "solar_open2_hardterm_fancy_v2_two_epoch"
        with self.assertRaisesRegex(RuntimeError, "recipe name"):
            validate_contract.validate_recipe_contract(recipe)

    def test_early_stopping_policy_is_exact_and_fail_closed(self) -> None:
        mutations = {
            "missing_monitor": lambda policy: policy.pop("monitor"),
            "patience": lambda policy: policy.update(
                {"subset_patience_evaluations": 2}
            ),
            "confirmation_step": lambda policy: policy.update(
                {"confirmation_step": 13_000}
            ),
            "sealed_monitor": lambda policy: policy.update(
                {"sealed_evaluation_not_monitor": False}
            ),
            "subset_delta": lambda policy: policy[
                "subset_min_delta"
            ].update({"absolute": 0.002}),
            "full_delta": lambda policy: policy[
                "full_min_delta"
            ].update({"relative": 0.002}),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                recipe = copy.deepcopy(self.recipe)
                mutate(recipe["validation"]["early_stopping"])
                with self.assertRaisesRegex(
                    RuntimeError,
                    "early-stopping policy",
                ):
                    validate_contract.validate_recipe_contract(recipe)

    def test_old_fixed_final_recipe_is_rejected(self) -> None:
        recipe = copy.deepcopy(self.recipe)
        recipe["optimizer_steps"] = 4000
        recipe["max_length"] = 3072
        recipe["validation"]["selection_policy"] = "fixed_final_step"
        recipe["validation"]["selected_step"] = 4000
        with self.assertRaises(RuntimeError):
            validate_contract.validate_recipe_contract(recipe)

    def test_minimum_learning_rate_ratio_is_fail_closed(self) -> None:
        recipe = copy.deepcopy(self.recipe)
        recipe["minimum_learning_rate_ratio"] = 0.0
        with self.assertRaisesRegex(
            RuntimeError,
            "minimum_learning_rate_ratio",
        ):
            validate_contract.validate_recipe_contract(recipe)

    def test_deck_arithmetic_is_fail_closed(self) -> None:
        recipe = copy.deepcopy(self.recipe)
        recipe["training_deck"]["total_occurrences"]["translation"] -= 1
        with self.assertRaisesRegex(RuntimeError, "occurrence counts"):
            validate_contract.validate_recipe_contract(recipe)

    def test_candidate_metadata_is_fail_closed(self) -> None:
        recipe = copy.deepcopy(self.recipe)
        recipe["validation"]["full_validation_candidates"][1][
            "occurrence_offset"
        ] = 0
        with self.assertRaisesRegex(RuntimeError, "candidate metadata"):
            validate_contract.validate_recipe_contract(recipe)

    def test_training_and_selection_weights_cannot_be_conflated(self) -> None:
        recipe = copy.deepcopy(self.recipe)
        recipe["training_deck"]["natural_task_fraction"] = {
            "translation": 0.65,
            "summarization": 0.35,
        }
        with self.assertRaisesRegex(RuntimeError, "natural training-deck fraction"):
            validate_contract.validate_recipe_contract(recipe)

    def test_evaluation_contract_is_unchanged(self) -> None:
        recipe = copy.deepcopy(self.recipe)
        recipe["evaluation"]["max_model_len"] = 9600
        with self.assertRaisesRegex(RuntimeError, "evaluation contract"):
            validate_contract.validate_recipe_contract(recipe)

    def test_b200_exact_one_epoch_recipe_passes(self) -> None:
        evidence = validate_contract.validate_recipe_contract(
            self.b200_recipe
        )
        self.assertEqual(evidence["profile"], "b200")
        self.assertEqual(evidence["epochs"], 1)
        self.assertEqual(evidence["optimizer_steps"], 9_316)
        self.assertEqual(evidence["total_occurrences"], 149_048)
        self.assertEqual(evidence["nominal_step_occurrences"], 16)
        self.assertEqual(evidence["final_step_occurrences"], 8)
        self.assertEqual(evidence["resume_completed_steps"], [9_315, 9_316])
        self.assertEqual(
            evidence["fixed_subset_validation_steps"],
            [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000],
        )
        self.assertEqual(evidence["full_validation_steps"], [4_658, 9_316])
        self.assertEqual(
            evidence["full_validation_consumed_occurrences"],
            [74_528, 149_048],
        )
        self.assertFalse(evidence["early_stopping_policy"]["enabled"])

    def test_b200_exact_tail_is_fail_closed(self) -> None:
        mutations = {
            "two_epochs": lambda recipe: recipe.update({"epochs": 2}),
            "padded_total": lambda recipe: recipe["exact_tail"].update(
                {"total_occurrences": 149_056}
            ),
            "padded_final_step": lambda recipe: recipe["exact_tail"].update(
                {"final_step_occurrences": 16}
            ),
            "padding_allowed": lambda recipe: recipe["exact_tail"].update(
                {"padding_or_duplication_allowed": True}
            ),
            "missing_resume_boundary": lambda recipe: recipe[
                "exact_tail"
            ].update({"resume_completed_steps": [9_316]}),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                recipe = copy.deepcopy(self.b200_recipe)
                mutate(recipe)
                with self.assertRaises(RuntimeError):
                    validate_contract.validate_recipe_contract(recipe)

    def test_b200_validation_schedule_is_fail_closed(self) -> None:
        mutations = {
            "no_step_zero": lambda recipe: recipe["validation"].update(
                {"step_zero_baseline": False}
            ),
            "wrong_cadence": lambda recipe: recipe["validation"].update(
                {"eval_every_steps": 500}
            ),
            "wrong_full_steps": lambda recipe: recipe["validation"].update(
                {"full_validation_steps": [4_658, 9_315]}
            ),
            "padded_candidate": lambda recipe: recipe["validation"][
                "full_validation_candidates"
            ][1].update({"consumed_occurrences": 149_056}),
            "training_weights": lambda recipe: recipe["validation"].update(
                {
                    "selection_weights": {
                        "translation": 0.6858461703612259,
                        "summarization": 0.3141538296387741,
                    }
                }
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                recipe = copy.deepcopy(self.b200_recipe)
                mutate(recipe)
                with self.assertRaisesRegex(RuntimeError, "validation"):
                    validate_contract.validate_recipe_contract(recipe)

    def test_b200_quality_early_stop_and_sealed_monitors_are_forbidden(self) -> None:
        mutations = {
            "quality_early_stop": lambda recipe: recipe["validation"][
                "early_stopping"
            ].update({"enabled": True}),
            "test_monitor": lambda recipe: recipe["validation"].update(
                {"monitor_datasets": ["validation", "full_test"]}
            ),
            "human_selection": lambda recipe: recipe["validation"].update(
                {"monitor_datasets": ["human_gold"]}
            ),
            "sealed_eval_training_use": lambda recipe: recipe[
                "evaluation"
            ].update({"used_for_training_monitor_or_selection": True}),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                recipe = copy.deepcopy(self.b200_recipe)
                mutate(recipe)
                with self.assertRaises(RuntimeError):
                    validate_contract.validate_recipe_contract(recipe)

    def test_explicit_config_dir_selects_b200_without_changing_default(self) -> None:
        self.assertEqual(
            validate_contract._resolve_config_dir(None),
            CONFIG_DIR.resolve(),
        )
        b200 = validate_contract.validate_recipe(B200_CONFIG_DIR)
        self.assertEqual(b200["profile"], "b200")
        self.assertEqual(b200["epochs"], 1)
        self.assertEqual(
            Path(b200["config_dir"]),
            B200_CONFIG_DIR.resolve(),
        )

    def test_b200_evaluator_identity_is_owned_only(self) -> None:
        evidence = validate_contract.validate_evaluator_identity(
            B200_CONFIG_DIR
        )
        self.assertTrue(evidence["owned_only"])
        self.assertNotIn("source_path", evidence)
        self.assertEqual(
            evidence["sha256"],
            validate_contract.REFERENCE_SHA256,
        )


if __name__ == "__main__":
    unittest.main()
