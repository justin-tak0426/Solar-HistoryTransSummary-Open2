#!/usr/bin/env python3
"""Static fail-closed tests for the B200 training-to-evaluation dependency."""

from __future__ import annotations

import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
LAUNCH = PROJECT / "launch" / "b200"


class B200EvaluationDependencyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.training = (LAUNCH / "train_production.sh").read_text(encoding="utf-8")
        cls.evaluation = (LAUNCH / "run_post_training_evaluation.sh").read_text(
            encoding="utf-8"
        )

    def test_training_terminal_success_is_the_only_evaluation_edge(self) -> None:
        train_call = self.training.index(
            '"$B200_PYTHON" "$SOLAR_PROJECT/scripts/train.py"'
        )
        success_gate = self.training.index(
            'b200_require_existing_file "$B200_PRODUCTION_RUN_DIR/_SUCCESS"'
        )
        manifest_gate = self.training.index(
            'b200_require_existing_file "$B200_PRODUCTION_RUN_DIR/run_manifest.json"'
        )
        evaluation_exec = self.training.index(
            'exec "$B200_LAUNCH_DIR/run_post_training_evaluation.sh"'
        )
        self.assertLess(train_call, success_gate)
        self.assertLess(success_gate, manifest_gate)
        self.assertLess(manifest_gate, evaluation_exec)
        self.assertIn("b200_release_stage_lock", self.training)
        self.assertIn("export WANDB_MODE=online", self.training)

    def test_training_launcher_preserves_train_and_wandb_resume(self) -> None:
        self.assertIn(
            'b200_prepare_resumable_dir "$B200_PRODUCTION_RUN_DIR"',
            self.training,
        )
        self.assertIn('b200_prepare_resumable_dir "$WANDB_DIR"', self.training)
        self.assertNotIn(
            'b200_require_fresh_path "$B200_PRODUCTION_RUN_DIR"',
            self.training,
        )
        self.assertNotIn('b200_require_fresh_path "$WANDB_DIR"', self.training)
        self.assertIn("b200_1epoch_len9600_v1.resume-", self.training)

    def test_only_project_evaluation_py_scores(self) -> None:
        self.assertEqual(
            self.evaluation.count(
                '"$B200_PYTHON" "$SOLAR_PROJECT/scripts/evaluation.py" score'
            ),
            2,
        )
        self.assertNotIn("shared_nodict_evaluation.py", self.evaluation)
        self.assertNotIn("score-and-seal", self.evaluation)
        self.assertEqual(self.evaluation.count("seal-metrics"), 2)
        for value in (
            "--bert_model xlm-roberta-large",
            "--bert_batch_size 8",
            "--bleu_tokenizer ko-mecab",
            "--device cuda",
        ):
            self.assertEqual(self.evaluation.count(value), 2)
        self.assertNotIn(
            '"$SOLAR_PROJECT/scripts/evaluation.py" infer', self.evaluation
        )

    def test_bertscore_uses_only_the_sealed_metric_cache(self) -> None:
        common_env = (LAUNCH / "common_env.sh").read_text(encoding="utf-8")
        gate = (LAUNCH / "launcher_gate.py").read_text(encoding="utf-8")
        self.assertEqual(self.evaluation.count("b200_run_metric_scorer"), 2)
        for value in (
            'HF_HOME="$B200_METRIC_HF_HOME"',
            'HF_HUB_CACHE="$B200_METRIC_HUB_CACHE"',
            'HUGGINGFACE_HUB_CACHE="$B200_METRIC_HUB_CACHE"',
            'TRANSFORMERS_CACHE="$B200_METRIC_HUB_CACHE"',
        ):
            self.assertIn(value, common_env)
        for value in (
            "metric_model",
            "xlm-roberta-large",
            "large_weight_hash_checked",
            "validate_metric_cache()",
        ):
            self.assertIn(value, gate)

    def test_base_and_selected_best_adapter_are_the_only_models(self) -> None:
        self.assertEqual(self.evaluation.count("--mode production-evaluation"), 2)
        self.assertEqual(self.evaluation.count("--resume"), 2)
        self.assertEqual(self.evaluation.count("--adapter-dir"), 2)
        self.assertIn('--adapter-dir "$selected_adapter"', self.evaluation)
        self.assertIn('--model-label "$B200_EVALUATION_BASE_LABEL"', self.evaluation)
        self.assertIn('--model-label "$B200_EVALUATION_TUNED_LABEL"', self.evaluation)
        self.assertEqual(self.evaluation.count('--eval-jsonl "$B200_COMBINED_EVAL"'), 3)

    def test_raw_output_audit_precedes_any_scoring(self) -> None:
        audit = self.evaluation.index('"$SOLAR_PROJECT/scripts/audit_outputs.py"')
        first_split = self.evaluation.index(
            '"$SOLAR_PROJECT/scripts/finalize.py" split'
        )
        first_score = self.evaluation.index(
            '"$SOLAR_PROJECT/scripts/evaluation.py" score'
        )
        report = self.evaluation.index('"$SOLAR_PROJECT/scripts/finalize.py" report')
        upload = self.evaluation.index('"$SOLAR_PROJECT/scripts/upload_wandb.py"')
        self.assertLess(audit, first_split)
        self.assertLess(first_split, first_score)
        self.assertLess(first_score, report)
        self.assertLess(report, upload)
        self.assertIn("set -euo pipefail", self.evaluation)
        self.assertIn("outer", (PROJECT / "scripts" / "audit_outputs.py").read_text())

    def test_gpu_gate_covers_inference_and_bertscore_stages(self) -> None:
        self.assertGreaterEqual(self.evaluation.count("b200_gpu_idle_gate"), 4)
        self.assertIn("b200_use_gpu_6_7", self.evaluation)
        self.assertIn("b200_use_cpu_only", self.evaluation)

    def test_evaluation_wandb_is_separate_and_linked(self) -> None:
        self.assertIn('--run-id "$B200_EVALUATION_WANDB_RUN_ID"', self.evaluation)
        self.assertIn(
            "--training-run-id solar_open2_b200_1epoch_20260802_v2",
            self.evaluation,
        )
        self.assertIn(
            '--success-marker "$B200_EVALUATION_DIR/_SUCCESS"', self.evaluation
        )
        self.assertNotEqual(
            "solar_open2_b200_evaluation_20260802_v2",
            "solar_open2_b200_1epoch_20260802_v2",
        )


if __name__ == "__main__":
    unittest.main()
