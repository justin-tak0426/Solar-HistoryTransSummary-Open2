#!/usr/bin/env python3
"""CPU tests for raw decoding and exact-score provenance checks."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_outputs  # noqa: E402
import evaluation  # noqa: E402
import finalize  # noqa: E402
import infer  # noqa: E402
import upload_wandb  # noqa: E402
from b200_runtime import (  # noqa: E402
    B200_CONFIG_DIR,
    b200_config_identity,
    load_b200_config_bundle,
)
from common import SOLAR_OUTPUT_ONLY_GENERATION_PREAMBLE  # noqa: E402


TEST_TEMP_ROOT = Path(__file__).resolve().parents[2] / "runtime/tmp/tests/eval_chain"


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def training_manifest_fixture(
    losses: dict[int, float],
    *,
    selected_step: int | None = None,
    early_stopped: bool = False,
) -> dict[str, object]:
    recipe = finalize.read_json(finalize.CONFIG_DIR / "recipe.json")
    validation = recipe["validation"]
    target_step = int(recipe["optimizer_steps"])
    final_step = (
        int(validation["early_stopping"]["confirmation_step"])
        if early_stopped
        else target_step
    )
    status = "EARLY_STOPPED" if early_stopped else "COMPLETED"
    configured_candidate_steps = list(validation["full_validation_steps"])
    candidate_steps = [
        step for step in configured_candidate_steps if step <= final_step
    ]
    candidates = [
        {
            "step": step,
            "selection_loss": losses[step],
            "translation_loss": losses[step],
            "summarization_loss": losses[step],
            "adapter_dir": str(TEST_TEMP_ROOT / f"candidates/step-{step:05d}"),
            "files_sha256": {"adapter.safetensors": f"{step:064x}"[-64:]},
        }
        for step in candidate_steps
    ]
    wanted_step = selected_step
    if wanted_step is None:
        wanted_step = min(candidate_steps, key=lambda step: (losses[step], step))
    wanted_candidate = next(
        candidate for candidate in candidates if candidate["step"] == wanted_step
    )
    occurrences_per_step = recipe["microbatch_policy"]["optimizer_step_occurrences"]
    occurrences_per_epoch = recipe["training_deck"]["occurrences_per_epoch"]["total"]
    occurrences = final_step * occurrences_per_step
    scheduled_steps = sorted(
        set(
            range(
                validation["eval_every_steps"],
                final_step + 1,
                validation["eval_every_steps"],
            )
        )
        | set(candidate_steps)
    )
    history = []
    for step in scheduled_steps:
        scope = "full" if step in candidate_steps else "fixed_subset"
        if scope == "full":
            loss = losses[step]
        elif early_stopped:
            loss = 1.0 if step < validation["early_stopping"]["minimum_step"] else 1.1
        else:
            loss = 2.0 - step / 100_000
        history.append(
            {
                "step": step,
                "scope": scope,
                "selection_loss": loss,
                "translation_loss": loss,
                "summarization_loss": loss,
            }
        )
    early_stopping = finalize.early_stopping_decision(
        history,
        completed_step=final_step,
        policy=validation["early_stopping"],
        optimizer_steps=target_step,
        full_validation_steps=configured_candidate_steps,
        occurrences_per_step=occurrences_per_step,
        occurrences_per_epoch=occurrences_per_epoch,
    )
    return {
        "status": status,
        "config": recipe,
        "completed_updates": final_step,
        "target_updates": target_step,
        "consumed_occurrences": occurrences,
        "training_completion": {
            "status": status,
            "epochs": recipe["epochs"],
            "completed_epochs": occurrences / occurrences_per_epoch,
            "optimizer_steps": final_step,
            "consumed_occurrences": occurrences,
            "expected_optimizer_steps": target_step,
            "expected_occurrences": recipe["training_deck"]["total_occurrences"][
                "total"
            ],
            "early_stopping_triggered": early_stopping["triggered"],
        },
        "early_stopping_decision": early_stopping,
        "training_coverage": {
            "total_occurrences": occurrences,
            "skipped_occurrences": 0,
            "unconsumed_planned_occurrences": (
                recipe["training_deck"]["total_occurrences"]["total"] - occurrences
            ),
            "unique_raw_rows": recipe["training_deck"]["base_unique_rows"],
            "task_occurrences": (
                {"translation": 150000, "summarization": 73568}
                if early_stopped
                else {
                    task: recipe["training_deck"]["total_occurrences"][task]
                    for task in ("translation", "summarization")
                }
            ),
            "quality_occurrences": {"fixture": occurrences},
        },
        "history": history,
        "split_roles": copy.deepcopy(finalize.EXPECTED_SPLIT_ROLES),
        "selection_record": {
            "policy": recipe["validation"]["selection_policy"],
            "tie_break": recipe["validation"]["selection_tie_break"],
            "candidate_steps": candidate_steps,
            "candidates": candidates,
            "selected_step": wanted_step,
            "selected_selection_loss": wanted_candidate["selection_loss"],
        },
        "selected_checkpoint": {
            "policy": recipe["validation"]["selection_policy"],
            "tie_break": recipe["validation"]["selection_tie_break"],
            "step": wanted_step,
            "selection_loss": wanted_candidate["selection_loss"],
            "source_checkpoint_dir": wanted_candidate["adapter_dir"],
            "adapter_dir": str(TEST_TEMP_ROOT / "run/best_adapter"),
            "files_sha256": {"adapter.safetensors": "a" * 64},
        },
        "final_checkpoint": {
            "step": final_step,
            "adapter_dir": str(TEST_TEMP_ROOT / "run/final_adapter"),
            "files_sha256": {"adapter.safetensors": "b" * 64},
        },
    }


class RawDecodeTest(unittest.TestCase):
    def test_inference_requires_exact_output_only_preamble(self) -> None:
        messages = [{"role": "user", "content": "source"}]
        row = {
            "id": "row",
            "task": "translation",
            "origin": "source",
            "messages": messages,
        }

        class Reference:
            @staticmethod
            def clean_text(value: str) -> str:
                return value

            @staticmethod
            def build_messages_no_dict(
                task: str,
                origin: str,
            ) -> list[dict[str, str]]:
                return messages

        class Tokenizer:
            def __init__(self, prompt: str) -> None:
                self.prompt = prompt
                self.tokenized = False

            def apply_chat_template(self, *_args: object, **_kwargs: object) -> str:
                return self.prompt

            def __call__(self, *_args: object, **_kwargs: object) -> dict[str, object]:
                self.tokenized = True
                return {"input_ids": [1, 2, 3]}

        valid = Tokenizer("rendered" + SOLAR_OUTPUT_ONLY_GENERATION_PREAMBLE)
        self.assertEqual(
            infer.render_prompt_ids(
                valid,
                Reference(),
                row,
                max_prompt_tokens=32,
            ),
            [1, 2, 3],
        )
        self.assertTrue(valid.tokenized)

        invalid = Tokenizer("rendered<|im:start|>assistant")
        with self.assertRaisesRegex(RuntimeError, "preamble changed"):
            infer.render_prompt_ids(
                invalid,
                Reference(),
                row,
                max_prompt_tokens=32,
            )
        self.assertFalse(invalid.tokenized)

    def test_decode_preserves_non_eos_special_tokens(self) -> None:
        class Tokenizer:
            kwargs: dict[str, object] | None = None

            def decode(self, token_ids: list[int], **kwargs: object) -> str:
                self.kwargs = kwargs
                self.token_ids = token_ids
                return "  <|think:start|>비공개 추론<|think:end|>정답  "

        tokenizer = Tokenizer()
        decoded = infer.decode_raw_continuation(tokenizer, [7, 8, 9])
        self.assertEqual(
            decoded,
            "<|think:start|>비공개 추론<|think:end|>정답",
        )
        self.assertEqual(tokenizer.token_ids, [7, 8, 9])
        self.assertEqual(
            tokenizer.kwargs,
            {
                "skip_special_tokens": False,
                "clean_up_tokenization_spaces": False,
            },
        )

    def test_output_only_signals_are_detected(self) -> None:
        signals = audit_outputs.detect_output_signals(
            "<|think:start|>추론<|think:end|>\n답변: 결과\n```"
        )
        self.assertIn("special_control_token", signals)
        self.assertIn("result_label", signals)
        self.assertIn("markdown_fence", signals)
        self.assertEqual(audit_outputs.detect_output_signals("번역 본문"), {})
        model = {"confirmed_output_failures": {"count": 1}}
        self.assertEqual(
            audit_outputs.audit_status(
                [model],
                [],
                [{"model": "base", "violating_rows": 1}],
            ),
            {
                "status": "FAIL",
                "evaluation_integrity_status": "PASS",
                "output_cleanliness_status": "FAIL",
            },
        )

    def test_every_output_contamination_class_is_hard_failed(self) -> None:
        examples = {
            "explanation_framing": "설명: 이 문장은 관청의 통지를 뜻한다.",
            "reasoning_label": "분석: 먼저 날짜를 확인한다.",
            "reasoning_prose_framing": "먼저 원문을 분석해 보겠습니다.",
            "chat_control_marker": "[INST] 번역해 주세요 [/INST]",
            "explicit_markdown": "## 번역\n현대 한국어 본문",
            "prompt_echo_extended": (
                "Use a formal documentary style. 현대 한국어 본문"
            ),
            "result_label": "최종 답변: 현대 한국어 본문",
            "meta_framing": "Here is the translation: 현대 한국어 본문",
            "refusal_framing": "I cannot translate this.",
            "blockquote_framing": "> 현대 한국어 본문",
            "non_korean_body": "This is not a Korean answer body.",
        }
        for category, prediction in examples.items():
            with self.subTest(category=category):
                signals = audit_outputs.detect_output_signals(prediction)
                self.assertIn(category, signals)
                model = {"confirmed_output_failures": {"count": 1 if signals else 0}}
                self.assertEqual(
                    audit_outputs.audit_status(
                        [model],
                        [],
                        [
                            {
                                "model": "base",
                                "violating_rows": 1,
                            }
                        ],
                    )["status"],
                    "FAIL",
                )

    def test_output_only_known_wrapper_bypasses_are_hard_failed(self) -> None:
        examples = (
            ("**번역:** 현대 한국어 본문", "result_label"),
            ("번역문\n현대 한국어 본문", "result_label"),
            ("- 현대 한국어 번역 본문", "result_label"),
            (
                "제가 번역한 내용은 다음과 같습니다.\n현대 한국어 본문",
                "meta_framing",
            ),
            ("요약하면, 현대 한국어 본문", "meta_framing"),
            ("죄송하지만 번역할 수 없습니다.", "refusal_framing"),
            ("純粹漢字本文", "non_korean_body"),
        )
        for prediction, category in examples:
            with self.subTest(prediction=prediction):
                self.assertIn(
                    category,
                    audit_outputs.detect_output_signals(prediction),
                )

    def test_grounded_body_structure_is_not_framing(self) -> None:
        for prediction in (
            "비고: 풍속조사는 자료를 다시 구분하여 조사하기로 하였다.",
            "2) [동학당의 동북 진출 저지] 병력을 먼저 진행하게 하였다.",
            "1. 첫째 조항은 관할 구역을 정한다.",
        ):
            with self.subTest(prediction=prediction):
                self.assertEqual(
                    audit_outputs.detect_output_signals(prediction),
                    {},
                )

    def test_generation_integrity_requires_clean_eos_roundtrip(self) -> None:
        clean = {
            "task": "translation",
            "prediction": "현대 한국어 번역",
            "prompt_tokens": 128,
            "prediction_tokens": 16,
            "generation_stop_reason": "eos",
            "tokenizer_roundtrip_exact": True,
            "decoding": {"translation_max_new_tokens": 512},
        }
        self.assertEqual(
            audit_outputs.generation_integrity_failures(clean),
            [],
        )
        corrupted = {
            **clean,
            "prediction": "깨진\ufffd문자\ud800\u0001",
            "prediction_tokens": 512,
            "generation_stop_reason": "max_new_tokens",
            "tokenizer_roundtrip_exact": False,
        }
        self.assertEqual(
            set(audit_outputs.generation_integrity_failures(corrupted)),
            {
                "unicode_replacement_character",
                "unicode_surrogate",
                "unicode_control_character",
                "generation_limit_exhausted",
                "generation_did_not_stop_at_eos",
                "tokenizer_roundtrip_mismatch",
            },
        )

    def test_all_sealed_targets_have_zero_false_positives(self) -> None:
        contract = json.loads(
            (B200_CONFIG_DIR / "data_contract.json").read_text(encoding="utf-8")
        )
        scan = audit_outputs.scan_reference_outputs(B200_CONFIG_DIR)
        self.assertEqual(scan["status"], "PASS")
        self.assertEqual(
            set(scan["roles"]),
            {"train", "validation", "full_test", "human_gold"},
        )
        for role, values in scan["roles"].items():
            self.assertEqual(values["rows"], contract["files"][role]["rows"])
            self.assertEqual(values["output_signal_rows"], 0)
            self.assertEqual(values["output_signal_categories"], {})
        self.assertEqual(
            scan["roles"]["train"]["normalized_answer_prefix_count"],
            3,
        )
        self.assertEqual(
            scan["roles"]["full_test"]["normalized_answer_prefix_count"],
            3,
        )
        self.assertEqual(
            scan["roles"]["validation"]["normalized_answer_prefix_count"],
            0,
        )
        self.assertEqual(
            scan["roles"]["human_gold"]["normalized_answer_prefix_count"],
            0,
        )
        finalize.validate_reference_scan_receipt(
            {"reference_output_only_scan": scan}, config_dir=B200_CONFIG_DIR
        )
        tampered = json.loads(json.dumps(scan))
        tampered["roles"]["human_gold"]["output_signal_rows"] = 1
        with self.assertRaisesRegex(RuntimeError, "human_gold"):
            finalize.validate_reference_scan_receipt(
                {"reference_output_only_scan": tampered},
                config_dir=B200_CONFIG_DIR,
            )

    def test_evaluation_uses_a_separate_wandb_run(self) -> None:
        sbatch = (infer.PROJECT / "sbatch/05_eval.sbatch").read_text(encoding="utf-8")
        self.assertIn(
            "--run-id solar_open2_hardterm_eval_earlystop_20260727_v2",
            sbatch,
        )
        self.assertIn(
            "--training-run-id " "solar_open2_hardterm_2epoch_earlystop_20260727_v2",
            sbatch,
        )

    def test_training_wandb_reference_is_manifest_bound(self) -> None:
        identity = {
            "mode": "online",
            "entity": "entity",
            "project": "project",
            "run_id": "training",
            "run_name": "Solar training",
            "resume": "allow",
        }
        training = {
            "wandb": identity,
            "state_contract": {"wandb": identity},
        }
        self.assertEqual(
            upload_wandb.validate_training_wandb_identity(
                training,
                entity="entity",
                project="project",
                training_run_id="training",
            ),
            identity,
        )
        with self.assertRaisesRegex(RuntimeError, "run_id changed"):
            upload_wandb.validate_training_wandb_identity(
                training,
                entity="entity",
                project="project",
                training_run_id="arbitrary",
            )
        with self.assertRaisesRegex(RuntimeError, "run_name changed"):
            upload_wandb.validate_training_wandb_identity(
                training,
                entity="entity",
                project="project",
                training_run_id="training",
                expected_run_name="different production name",
            )


class TrainingSelectionContractTest(unittest.TestCase):
    def test_each_full_validation_candidate_can_be_selected(self) -> None:
        candidate_steps = [9316, 13973, 18631]
        for selected_step in candidate_steps:
            with self.subTest(selected_step=selected_step):
                losses = {step: 0.8 for step in candidate_steps}
                losses[selected_step] = 0.2
                manifest = training_manifest_fixture(losses)
                selection = finalize.validate_training_selection(manifest)
                self.assertEqual(selection["final_step"], 18631)
                self.assertEqual(selection["selected_step"], selected_step)
                self.assertIn(selection["selected_step"], candidate_steps)

    def test_selected_step_is_not_assumed_to_equal_final_step(self) -> None:
        manifest = training_manifest_fixture({9316: 0.2, 13973: 0.3, 18631: 0.4})
        selection = finalize.validate_training_selection(manifest)
        self.assertEqual(selection["final_step"], 18631)
        self.assertEqual(selection["selected_step"], 9316)
        self.assertNotEqual(selection["selected_step"], selection["final_step"])

    def test_manifest_selected_step_must_be_the_argmin(self) -> None:
        manifest = training_manifest_fixture(
            {9316: 0.2, 13973: 0.3, 18631: 0.4},
            selected_step=18631,
        )
        with self.assertRaisesRegex(RuntimeError, "candidate argmin"):
            finalize.validate_training_selection(manifest)

    def test_exact_loss_tie_selects_the_earlier_candidate(self) -> None:
        manifest = training_manifest_fixture({9316: 0.2, 13973: 0.2, 18631: 0.4})
        self.assertEqual(
            finalize.validate_training_selection(manifest)["selected_step"],
            9316,
        )
        tampered = copy.deepcopy(manifest)
        tampered["selection_record"]["selected_step"] = 13973
        tampered["selected_checkpoint"]["step"] = 13973
        with self.assertRaisesRegex(RuntimeError, "candidate argmin"):
            finalize.validate_training_selection(tampered)

    def test_non_candidate_selection_is_rejected(self) -> None:
        manifest = training_manifest_fixture({9316: 0.2, 13973: 0.3, 18631: 0.4})
        manifest["selection_record"]["candidate_steps"] = [9316, 10000, 18631]
        with self.assertRaisesRegex(RuntimeError, "policy or candidates"):
            finalize.validate_training_selection(manifest)

    def test_approved_early_stop_uses_only_completed_candidates(self) -> None:
        manifest = training_manifest_fixture(
            {9316: 0.4, 13973: 0.5, 18631: 0.1},
            early_stopped=True,
        )
        selection = finalize.validate_training_selection(manifest)
        self.assertEqual(selection["training_status"], "EARLY_STOPPED")
        self.assertEqual(selection["final_step"], 13973)
        self.assertEqual(selection["target_step"], 18631)
        self.assertEqual(selection["candidate_steps"], [9316, 13973])
        self.assertEqual(selection["selected_step"], 9316)
        self.assertEqual(
            manifest["consumed_occurrences"],
            223568,
        )
        self.assertEqual(
            selection["early_stopping_decision"]["status"],
            "TRIGGERED",
        )

    def test_early_stop_requires_exact_recomputed_trigger_evidence(self) -> None:
        manifest = training_manifest_fixture(
            {9316: 0.4, 13973: 0.5, 18631: 0.1},
            early_stopped=True,
        )
        manifest["early_stopping_decision"]["triggered"] = False
        with self.assertRaisesRegex(RuntimeError, "evidence changed"):
            finalize.validate_training_selection(manifest)

    def test_early_stop_cannot_move_off_confirmation_step(self) -> None:
        manifest = training_manifest_fixture(
            {9316: 0.4, 13973: 0.5, 18631: 0.1},
            early_stopped=True,
        )
        manifest["completed_updates"] = 13972
        with self.assertRaisesRegex(RuntimeError, "completion step changed"):
            finalize.validate_training_selection(manifest)

    def test_test_and_human_gold_cannot_be_selection_monitors(self) -> None:
        manifest = training_manifest_fixture(
            {9316: 0.4, 13973: 0.5, 18631: 0.1},
            early_stopped=True,
        )
        manifest["history"][0]["scope"] = "human_gold"
        with self.assertRaisesRegex(RuntimeError, "cannot be an early-stopping"):
            finalize.validate_training_selection(manifest)

        manifest = training_manifest_fixture(
            {9316: 0.4, 13973: 0.5, 18631: 0.1},
            early_stopped=True,
        )
        manifest["split_roles"]["full_test"] = "monitoring"
        with self.assertRaisesRegex(RuntimeError, "sealed test/human"):
            finalize.validate_training_selection(manifest)


class ExactScoreBindingTest(unittest.TestCase):
    def setUp(self) -> None:
        TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT)
        self.root = Path(self.temporary.name)
        self.labels = ["base", "tuned"]
        self.prediction_paths = [
            self.root / "base.jsonl",
            self.root / "tuned.jsonl",
        ]
        rows_by_model = []
        for label, prediction_path in zip(self.labels, self.prediction_paths):
            rows = [
                {
                    "id": f"{label}-translation",
                    "task": "translation",
                    "model_label": label,
                    "prediction": "현대 한국어 번역",
                    "reference": "현대 한국어 번역",
                },
                {
                    "id": f"{label}-summarization",
                    "task": "summarization",
                    "model_label": label,
                    "prediction": "한국어 요약",
                    "reference": "한국어 요약",
                },
            ]
            write_jsonl(prediction_path, rows)
            rows_by_model.extend(rows)
        self.metric_rows, self.summary = self.make_scorer_outputs(rows_by_model)
        self.rows_path = self.root / "metrics.rows.jsonl"
        self.summary_path = self.root / "metrics.summary.json"
        write_jsonl(self.rows_path, self.metric_rows)
        self.summary_path.write_text(
            json.dumps(self.summary, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_scorer_outputs(
        self,
        rows: list[dict[str, object]],
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        recipe = finalize.read_json(finalize.CONFIG_DIR / "recipe.json")["evaluation"]
        metric_rows = []
        groups: dict[tuple[str, str], list[dict[str, object]]] = {}
        for row in rows:
            observed = {
                **row,
                "sentence_bleu": 100.0,
                "rougeL_f1": 1.0,
                "bertscore_precision": 0.5,
                "bertscore_recall": 0.5,
                "bertscore_f1": 0.5,
            }
            metric_rows.append(observed)
            groups.setdefault(
                (str(row["model_label"]), str(row["task"])),
                [],
            ).append(observed)
        summaries = {}
        for (label, task), group in groups.items():
            summaries[f"{label}/{task}"] = {
                "count": len(group),
                "corpus_bleu": 100.0,
                "sentence_bleu_mean": 100.0,
                "rougeL_f1_mean": 1.0,
                "bertscore_f1_mean": 0.5,
            }
        return metric_rows, {
            "bleu_tokenizer": recipe["bleu_tokenizer"],
            "bert_model": recipe["bert_model"],
            "by_model_task": summaries,
        }

    def validate(self) -> dict[str, object]:
        return finalize.validate_scorer_outputs(
            prediction_paths=self.prediction_paths,
            summary_path=self.summary_path,
            rows_path=self.rows_path,
            expected_model_labels=self.labels,
        )

    def test_exact_scorer_outputs_pass(self) -> None:
        self.assertEqual(
            set(self.validate()["by_model_task"]),
            {
                "base/translation",
                "base/summarization",
                "tuned/translation",
                "tuned/summarization",
            },
        )

    def test_out_of_range_summary_is_rejected(self) -> None:
        self.summary["by_model_task"]["tuned/translation"]["corpus_bleu"] = 999.0
        self.summary_path.write_text(
            json.dumps(self.summary, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "invalid tuned/translation/corpus_bleu",
        ):
            self.validate()

    def test_numeric_strings_and_booleans_are_rejected(self) -> None:
        self.metric_rows[0]["sentence_bleu"] = "100.0"
        write_jsonl(self.rows_path, self.metric_rows)
        with self.assertRaisesRegex(RuntimeError, "invalid sentence_bleu"):
            self.validate()

        write_jsonl(
            self.rows_path,
            self.make_scorer_outputs(
                [
                    row
                    for path in self.prediction_paths
                    for row in finalize.read_jsonl(path)
                ]
            )[0],
        )
        self.summary["by_model_task"]["base/translation"]["count"] = True
        self.summary_path.write_text(
            json.dumps(self.summary, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(RuntimeError, "scorer count changed"):
            self.validate()

    def test_validator_does_not_recompute_evaluation_metrics(self) -> None:
        self.summary["by_model_task"]["tuned/translation"]["corpus_bleu"] = 42.0
        self.summary_path.write_text(
            json.dumps(self.summary, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self.assertEqual(
            self.validate()["by_model_task"]["tuned/translation"]["corpus_bleu"],
            42.0,
        )

    def test_metric_rows_cannot_change_prediction_input(self) -> None:
        self.metric_rows[0]["prediction"] = "바뀐 출력"
        write_jsonl(self.rows_path, self.metric_rows)
        with self.assertRaisesRegex(RuntimeError, "prediction input field"):
            self.validate()

    def test_project_evaluation_py_is_the_only_recorded_scorer(self) -> None:
        direct_dir = self.root / "direct"
        paths = [self.root / "cli.base.jsonl", self.root / "cli.tuned.jsonl"]
        for label, path in zip(self.labels, paths):
            write_jsonl(
                path,
                [
                    {
                        "id": "translation-1",
                        "task": "translation",
                        "model_label": label,
                        "origin": "原文",
                        "reference": "현대 한국어 번역",
                        "prediction": (
                            "현대 한국어 번역" if label == "base" else "한국어 번역"
                        ),
                        "prompt_variant": "no_dictionary_context",
                    },
                    {
                        "id": "summarization-1",
                        "task": "summarization",
                        "model_label": label,
                        "origin": "原文 二",
                        "reference": "한국어 요약",
                        "prediction": "한국어 요약",
                        "prompt_variant": "no_dictionary_context",
                    },
                ],
            )

        class Scores(list[float]):
            def tolist(self) -> list[float]:
                return list(self)

        fake_bert_score = types.ModuleType("bert_score")

        def score(
            candidates: list[str],
            _references: list[str],
            **_kwargs: object,
        ) -> tuple[Scores, Scores, Scores]:
            count = len(candidates)
            return (
                Scores([0.8] * count),
                Scores([0.7] * count),
                Scores([0.75] * count),
            )

        fake_bert_score.score = score
        recipe = json.loads(
            (B200_CONFIG_DIR / "recipe.json").read_text(encoding="utf-8")
        )["evaluation"]
        argv = [
            str(finalize.EVALUATION_ENTRYPOINT),
            "score",
            "--prediction_jsonl",
            *(str(path) for path in paths),
            "--out_dir",
            str(direct_dir),
            "--bert_model",
            recipe["bert_model"],
            "--bert_batch_size",
            str(recipe["bert_batch_size"]),
            "--device",
            "cuda",
            "--bleu_tokenizer",
            recipe["bleu_tokenizer"],
        ]
        with mock.patch.dict(
            sys.modules,
            {"bert_score": fake_bert_score},
        ), mock.patch.object(sys, "argv", argv):
            self.assertEqual(evaluation.main(), 0)
        execution = finalize.scorer_execution_identity(
            prediction_paths=paths,
            model_labels=self.labels,
            model_roles=["base", "tuned"],
            out_dir=direct_dir,
            config_dir=B200_CONFIG_DIR,
        )
        self.assertEqual(
            execution["method"],
            "external_project_evaluation_py_score",
        )
        self.assertEqual(
            execution["reference_evaluator"],
            {
                "path": str(finalize.EVALUATION_ENTRYPOINT),
                "sha256": finalize.REFERENCE_SHA256,
            },
        )
        self.assertEqual(
            execution["command"],
            [str(Path(sys.executable).resolve()), *argv],
        )
        finalize.validate_scorer_outputs(
            prediction_paths=paths,
            summary_path=direct_dir / "metrics.summary.json",
            rows_path=direct_dir / "metrics.rows.jsonl",
            expected_model_labels=self.labels,
            config_dir=B200_CONFIG_DIR,
        )


class SplitReceiptBindingTest(unittest.TestCase):
    def test_split_receipt_rejects_post_audit_tampering(self) -> None:
        TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            root = Path(temporary)
            full_eval = root / "full.jsonl"
            gold_eval = root / "gold.jsonl"
            combined = root / "combined.jsonl"
            full_output = root / "full.predictions.jsonl"
            gold_output = root / "gold.predictions.jsonl"
            receipt = root / "split.receipt.json"
            audit_path = root / "audit.json"
            manifest_path = root / "combined.jsonl.manifest.json"
            full_rows = [
                {
                    "id": "full-1",
                    "task": "translation",
                    "origin": "원문 1",
                    "response": "정답 1",
                }
            ]
            gold_rows = [
                {
                    "id": "gold-1",
                    "task": "summarization",
                    "origin": "원문 2",
                    "response": "정답 2",
                }
            ]
            write_jsonl(full_eval, full_rows)
            write_jsonl(gold_eval, gold_rows)
            predictions = [
                {
                    "id": row["id"],
                    "task": row["task"],
                    "model_label": "base",
                    "origin": row["origin"],
                    "reference": row["response"],
                    "prediction": row["response"],
                }
                for row in [*full_rows, *gold_rows]
            ]
            write_jsonl(combined, predictions)
            manifest_path.write_text("{}\n", encoding="utf-8")
            audit = {
                "status": "PASS",
                "evaluation_integrity_status": "PASS",
                "output_cleanliness_status": "PASS",
                "pipeline_code_sha256": finalize.pipeline_code_identity()["sha256"],
                "integrity_failures": [],
                "output_failures": [],
                "models": [
                    {
                        "model_label": "base",
                        "prediction_path": str(combined),
                        "prediction_seal": {
                            "output_jsonl_sha256": finalize.sha256(combined),
                            "manifest_path": str(manifest_path),
                            "manifest_sha256": finalize.sha256(manifest_path),
                            "contract_sha256": "contract",
                        },
                        "confirmed_output_failures": {"count": 0},
                        "structural_failures": {},
                    }
                ],
            }
            audit_path.write_text(
                json.dumps(audit, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            data_contract = {
                "prepared_data_dir": str(root),
                "files": {
                    "full_test": {
                        "name": full_eval.name,
                        "sha256": finalize.sha256(full_eval),
                        "rows": 1,
                        "by_task": {
                            "translation": 1,
                            "summarization": 0,
                        },
                    },
                    "human_gold": {
                        "name": gold_eval.name,
                        "sha256": finalize.sha256(gold_eval),
                        "rows": 1,
                        "by_task": {
                            "translation": 0,
                            "summarization": 1,
                        },
                    },
                },
            }
            original_read_json = finalize.read_json

            def fixture_read_json(path: Path) -> dict[str, object]:
                if path == finalize.CONFIG_DIR / "data_contract.json":
                    return data_contract
                return original_read_json(path)

            args = Namespace(
                combined_predictions=combined,
                model_label="base",
                audit_json=audit_path,
                full_eval=full_eval,
                human_gold_eval=gold_eval,
                full_predictions=full_output,
                gold_predictions=gold_output,
                receipt=receipt,
            )
            with mock.patch.object(
                finalize,
                "read_json",
                side_effect=fixture_read_json,
            ), mock.patch.object(
                finalize,
                "validate_reference_scan_receipt",
            ):
                finalize.split_predictions(args)
                validated = finalize.validate_split_receipt(
                    receipt,
                    expected_audit_path=audit_path,
                    expected_model_label="base",
                )
                self.assertEqual(validated["status"], "PASS")
                full_output.write_text("{}\n", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "checksum changed"):
                    finalize.validate_split_receipt(
                        receipt,
                        expected_audit_path=audit_path,
                        expected_model_label="base",
                    )


class B200FinalizeSelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_b200_config_bundle(B200_CONFIG_DIR)
        cls.recipe = cls.bundle["recipe"]
        cls.identity = b200_config_identity(B200_CONFIG_DIR)

    def manifest(self) -> dict[str, object]:
        schedule = [
            (0, "fixed_subset"),
            (1_000, "fixed_subset"),
            (2_000, "fixed_subset"),
            (3_000, "fixed_subset"),
            (4_000, "fixed_subset"),
            (4_658, "full"),
            (5_000, "fixed_subset"),
            (6_000, "fixed_subset"),
            (7_000, "fixed_subset"),
            (8_000, "fixed_subset"),
            (9_000, "fixed_subset"),
            (9_316, "full"),
        ]
        history: list[dict[str, object]] = []
        for step, scope in schedule:
            translation = 0.8 if step == 4_658 else 0.6
            summarization = 0.7 if step == 4_658 else 0.5
            history.append(
                {
                    "step": step,
                    "scope": scope,
                    "translation_loss": translation,
                    "summarization_loss": summarization,
                    "selection_loss": 0.65 * translation + 0.35 * summarization,
                }
            )
        full = {item["step"]: item for item in history if item["scope"] == "full"}
        run_dir = B200_CONFIG_DIR.parents[2] / "runtime/runs/test-b200-terminal"
        candidates = [
            {
                "step": step,
                "selection_loss": full[step]["selection_loss"],
                "translation_loss": full[step]["translation_loss"],
                "summarization_loss": full[step]["summarization_loss"],
                "adapter_dir": str(run_dir / "candidates" / f"step-{step:05d}"),
                "files_sha256": {},
            }
            for step in (4_658, 9_316)
        ]
        selected = candidates[1]
        return {
            "status": "COMPLETED",
            "profile": "b200",
            "config_identity": self.identity,
            "config": self.recipe,
            "split_roles": finalize.EXPECTED_SPLIT_ROLES,
            "completed_updates": 9_316,
            "target_updates": 9_316,
            "consumed_occurrences": 149_048,
            "training_completion": {
                "status": "COMPLETED",
                "epochs": 1,
                "completed_epochs": 1.0,
                "early_stopping_triggered": False,
                "optimizer_steps": 9_316,
                "consumed_occurrences": 149_048,
                "expected_optimizer_steps": 9_316,
                "expected_occurrences": 149_048,
            },
            "training_coverage": {
                "total_occurrences": 149_048,
                "skipped_occurrences": 0,
                "unconsumed_planned_occurrences": 0,
                "unique_raw_rows": 142_713,
                "task_occurrences": {
                    "translation": 102_224,
                    "summarization": 46_824,
                },
                "quality_occurrences": {
                    "gold_expert_corrected": 7_240,
                    "silver_model_generated": 141_808,
                },
            },
            "history": history,
            "early_stopping_decision": {
                "profile": "b200",
                "status": "MAXIMUM_REACHED",
                "evaluated_step": 9_316,
                "triggered": False,
                "reason": "exact_epoch_coverage_reached",
                "quality_early_stopping_enabled": False,
                "completed_candidate_steps": [4_658, 9_316],
                "completed_occurrences": 149_048,
                "completed_epochs": 1.0,
            },
            "selection_record": {
                "policy": "argmin_full_validation_selection_loss",
                "tie_break": "earlier_optimizer_step",
                "candidate_steps": [4_658, 9_316],
                "candidates": candidates,
                "selected_step": 9_316,
                "selected_selection_loss": selected["selection_loss"],
            },
            "selected_checkpoint": {
                "policy": "argmin_full_validation_selection_loss",
                "tie_break": "earlier_optimizer_step",
                "step": 9_316,
                "selection_loss": selected["selection_loss"],
                "source_checkpoint_dir": selected["adapter_dir"],
                "adapter_dir": str(run_dir / "best_adapter"),
                "files_sha256": {},
            },
            "final_checkpoint": {
                "step": 9_316,
                "adapter_dir": str(run_dir / "final_adapter"),
                "files_sha256": {},
            },
        }

    def test_b200_exact_tail_selection_passes(self) -> None:
        result = finalize.validate_training_selection(
            self.manifest(), config_dir=B200_CONFIG_DIR
        )
        self.assertEqual(result["training_status"], "COMPLETED")
        self.assertEqual(result["final_step"], 9_316)
        self.assertEqual(result["candidate_steps"], [4_658, 9_316])

    def test_b200_tail_coverage_and_selection_fail_closed(self) -> None:
        mutations = (
            lambda value: value.update({"completed_updates": 9_315}),
            lambda value: value["training_coverage"].update(
                {"unconsumed_planned_occurrences": 8}
            ),
            lambda value: value["early_stopping_decision"].update({"triggered": True}),
            lambda value: value["selection_record"].update({"selected_step": 4_658}),
        )
        for mutate in mutations:
            manifest = self.manifest()
            mutate(manifest)
            with self.assertRaises(RuntimeError):
                finalize.validate_training_selection(
                    manifest, config_dir=B200_CONFIG_DIR
                )


if __name__ == "__main__":
    unittest.main()
