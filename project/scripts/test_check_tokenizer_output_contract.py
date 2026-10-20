#!/usr/bin/env python3
"""CPU-only fake-tokenizer tests for the Solar output-only tokenizer gate."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_tokenizer_output_contract as gate  # noqa: E402
from common import (  # noqa: E402
    SOLAR_ASSISTANT_TEMPLATE_TERMINATION,
    SOLAR_OUTPUT_ONLY_GENERATION_PREAMBLE,
    TRAINING_RESULT_LABEL_IDS,
    load_reference_module,
)


class FakeTokenizer:
    def __init__(
        self,
        *,
        termination: str = SOLAR_ASSISTANT_TEMPLATE_TERMINATION,
        require_body_label: bool = False,
    ) -> None:
        self.termination = termination
        self.require_body_label = require_body_label
        self.generation_prompt_calls = 0
        self.chat_template = "fake reasoning_effort template"
        self.name_or_path = "fake-solar"
        self.is_fast = True
        self.eos_token_id = ord(SOLAR_ASSISTANT_TEMPLATE_TERMINATION[0])
        # Solar control IDs are atomic but omitted from all_special_ids by the
        # Transformers backend; generation_config supplies the stop contract.
        self.all_special_ids = [0, 1, 2]

    def _assistant(self, content: str) -> str:
        return f"{SOLAR_OUTPUT_ONLY_GENERATION_PREAMBLE}" f"{content}{self.termination}"

    def _messages(self, messages: list[dict[str, str]]) -> str:
        rendered = ""
        for message in messages:
            if message["role"] == "assistant":
                rendered += self._assistant(message["content"])
            else:
                rendered += (
                    f"\n<{message['role']}>{message['content']}" f"</{message['role']}>"
                )
        return rendered

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        reasoning_effort: str,
        think_render_option: str,
        provider_system_prompt: bool,
    ) -> str:
        if tokenize:
            raise AssertionError("fixture expects tokenize=False")
        if (
            reasoning_effort != "none"
            or think_render_option != "preserved"
            or provider_system_prompt is not False
        ):
            raise AssertionError("fixture received wrong template kwargs")
        rendered = self._messages(messages)
        if add_generation_prompt:
            self.generation_prompt_calls += 1
            rendered += SOLAR_OUTPUT_ONLY_GENERATION_PREAMBLE
            if self.require_body_label and self.generation_prompt_calls >= 2:
                rendered += "\n답변:"
        return rendered

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
    ) -> dict[str, list[int]]:
        if add_special_tokens:
            raise AssertionError("fixture expects add_special_tokens=False")
        return {"input_ids": [ord(character) for character in text]}

    def decode(
        self,
        input_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        if skip_special_tokens or clean_up_tokenization_spaces:
            raise AssertionError("fixture expects exact raw decoding")
        return "".join(chr(token) for token in input_ids)


def fixture_rows() -> list[dict[str, Any]]:
    rows = [
        {
            "id": "translation-clean",
            "task": "translation",
            "origin": "古文",
            "response": "이 문서를 현대 한국어로 옮겼다.",
            "hard_term_readings": ["한국어"],
            "prompt_variant": "no_dictionary_context",
            "_contract_split": "train",
        },
        {
            "id": "summarization-clean",
            "task": "summarization",
            "origin": "긴 역사 기록",
            "response": "역사 기록을 요약한다.",
            "hard_term_readings": [],
            "prompt_variant": "no_dictionary_context",
            "_contract_split": "train",
        },
    ]
    for index, row_id in enumerate(sorted(TRAINING_RESULT_LABEL_IDS), 1):
        rows.append(
            {
                "id": row_id,
                "task": "translation",
                "origin": f"원문 {index}",
                "response": f"답변: 접두어를 제거한 번역 본문 {index}",
                "hard_term_readings": ["번역"],
                "prompt_variant": "no_dictionary_context",
                "_contract_split": "train",
            }
        )
    rows.extend(
        [
            {
                "id": "validation-translation-clean",
                "task": "translation",
                "origin": "검증 번역 원문",
                "response": "검증용 번역 본문",
                "hard_term_readings": [],
                "prompt_variant": "no_dictionary_context",
                "_contract_split": "validation",
            },
            {
                "id": "validation-summarization-clean",
                "task": "summarization",
                "origin": "검증 요약 원문",
                "response": "검증용 요약 본문",
                "hard_term_readings": [],
                "prompt_variant": "no_dictionary_context",
                "_contract_split": "validation",
            },
        ]
    )
    return rows


def empty_b200_input_unicode_inventory(rows: int) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    return {
        "scope": "sealed validation origin before reference cleaning",
        "rows_scanned": rows,
        "utf8_encoding_failures": 0,
        "surrogate_codepoint_occurrences": 0,
        "replacement_character_rows": 0,
        "replacement_character_occurrences": 0,
        "c0_non_whitespace_control_rows": 0,
        "c0_non_whitespace_control_occurrences": 0,
        "c1_control_rows": 0,
        "c1_control_occurrences": 0,
        "anomalous_rows": 0,
        "anomalous_row_records": records,
        "anomalous_row_records_sha256": gate.canonical_sha256(records),
        "tokenizer_roundtrip": {
            "rows_checked": rows,
            "exact_roundtrip_rows": rows,
            "failures": 0,
        },
    }


class TokenizerOutputContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reference = load_reference_module(
            config_dir=gate.b200_runtime.B200_CONFIG_DIR,
            owned_only=True,
        )

    def test_valid_output_only_contract(self) -> None:
        result = gate.validate_tokenizer_output_contract(
            FakeTokenizer(),
            self.reference,
            fixture_rows(),
            hard_term_weight=2.5,
        )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["coverage"]["rows"], 7)
        self.assertEqual(
            result["coverage"]["pinned_answer_prefix_ids"],
            sorted(TRAINING_RESULT_LABEL_IDS),
        )
        self.assertFalse(result["inference_requires_response_field"])
        self.assertFalse(result["inference_requires_body_label"])
        self.assertTrue(
            all(
                row["assistant_termination_token_count"] == 1
                and row["supervised_labels_are_body_plus_single_eos"]
                and row["train_inference_prompt_identical"]
                and not row["inference_requires_body_label"]
                for row in result["rows"]
            )
        )

    def test_combined_eval_prompt_budget_scan(self) -> None:
        rows = [
            {
                "id": f"{task}-eval",
                "task": task,
                "origin": f"{task} source",
                "messages": self.reference.build_messages_no_dict(
                    task,
                    f"{task} source",
                ),
            }
            for task in ("translation", "summarization")
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "combined.jsonl"
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            result = gate.scan_evaluation_prompt_budget(
                FakeTokenizer(),
                self.reference,
                path,
                expected_rows=2,
                expected_by_task={
                    "translation": 1,
                    "summarization": 1,
                },
                prompt_budget_tokens={
                    "translation": 100_000,
                    "summarization": 100_000,
                },
            )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["truncation_count"], 0)
        self.assertEqual(result["exact_output_only_preamble_count"], 2)

    def test_rejects_combined_eval_prompt_over_budget(self) -> None:
        task = "translation"
        row = {
            "id": "translation-eval",
            "task": task,
            "origin": "source",
            "messages": self.reference.build_messages_no_dict(task, "source"),
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "combined.jsonl"
            path.write_text(
                json.dumps(row, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "requires prompt truncation",
            ):
                gate.scan_evaluation_prompt_budget(
                    FakeTokenizer(),
                    self.reference,
                    path,
                    expected_rows=1,
                    expected_by_task={"translation": 1},
                    prompt_budget_tokens={
                        "translation": 1,
                        "summarization": 1,
                    },
                )

    def test_rejects_missing_assistant_termination(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "termination rendering|termination supervision",
        ):
            gate.validate_tokenizer_output_contract(
                FakeTokenizer(termination=""),
                self.reference,
                fixture_rows(),
                hard_term_weight=2.5,
            )

    def test_rejects_inference_body_label_requirement(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "generation preamble|requires a body label",
        ):
            gate.validate_tokenizer_output_contract(
                FakeTokenizer(require_body_label=True),
                self.reference,
                fixture_rows(),
                hard_term_weight=2.5,
            )

    def test_rejects_forbidden_target_framing(self) -> None:
        rows = fixture_rows()
        rows[0]["response"] = "번역 결과: 현대 한국어 번역문"
        with self.assertRaisesRegex(RuntimeError, "forbidden framing"):
            gate.validate_tokenizer_output_contract(
                FakeTokenizer(),
                self.reference,
                rows,
                hard_term_weight=2.5,
            )

    def test_cli_requires_pinned_inputs_and_receipt(self) -> None:
        options = {
            option
            for action in gate.build_parser()._actions
            for option in action.option_strings
        }
        self.assertTrue(
            {
                "--config-dir",
                "--source-dir",
                "--data-dir",
                "--receipt",
            }.issubset(options)
        )

    def test_b200_full_validation_scan_is_output_only_and_untruncated(self) -> None:
        rows = [
            {
                "id": f"validation-{task}",
                "task": task,
                "origin": f"{task} 원문",
                "response": f"{task} 검증 본문",
                "prompt_variant": "no_dictionary_context",
            }
            for task in ("translation", "summarization")
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "validation.jsonl"
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            result = gate.scan_b200_validation_prompt_budget(
                FakeTokenizer(),
                self.reference,
                path,
                expected_rows=2,
                expected_by_task={
                    "translation": 1,
                    "summarization": 1,
                },
                prompt_budget_tokens={
                    "translation": 100_000,
                    "summarization": 100_000,
                },
            )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["truncation_count"], 0)
        self.assertEqual(result["invalid_unicode_count"], 0)
        self.assertEqual(result["forbidden_output_framing_count"], 0)
        self.assertEqual(
            result["input_unicode_inventory"]["tokenizer_roundtrip"],
            {
                "rows_checked": 2,
                "exact_roundtrip_rows": 2,
                "failures": 0,
            },
        )
        self.assertEqual(
            result["input_unicode_inventory"]["anomalous_rows"],
            0,
        )

    def test_b200_validation_identity_is_task_plus_id(self) -> None:
        rows = [
            {
                "id": "shared-source-id",
                "task": task,
                "origin": "동일 원문",
                "response": f"{task} 검증 본문",
                "prompt_variant": "no_dictionary_context",
            }
            for task in ("translation", "summarization")
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "validation.jsonl"
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            result = gate.scan_b200_validation_prompt_budget(
                FakeTokenizer(),
                self.reference,
                path,
                expected_rows=2,
                expected_by_task={"translation": 1, "summarization": 1},
                prompt_budget_tokens={
                    "translation": 100_000,
                    "summarization": 100_000,
                },
            )
        self.assertEqual(result["rows"], 2)

    def test_b200_validation_rejects_duplicate_task_plus_id(self) -> None:
        row = {
            "id": "duplicate-id",
            "task": "translation",
            "origin": "검증 원문",
            "response": "검증 본문",
            "prompt_variant": "no_dictionary_context",
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "validation.jsonl"
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for _ in range(2)),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "duplicate ID\\+task"):
                gate.scan_b200_validation_prompt_budget(
                    FakeTokenizer(),
                    self.reference,
                    path,
                    expected_rows=2,
                    expected_by_task={"translation": 2},
                    prompt_budget_tokens={
                        "translation": 100_000,
                        "summarization": 100_000,
                    },
                )

    def test_b200_validation_source_unicode_is_inventoried_and_roundtrips(
        self,
    ) -> None:
        rows = [
            {
                "id": "validation-translation",
                "task": "translation",
                "origin": "손상 표식 \ufffd 및 C1 \u0080 원문",
                "response": "정상 검증 본문",
                "prompt_variant": "no_dictionary_context",
            },
            {
                "id": "validation-summarization",
                "task": "summarization",
                "origin": "정상 요약 원문",
                "response": "정상 요약 본문",
                "prompt_variant": "no_dictionary_context",
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "validation.jsonl"
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            result = gate.scan_b200_validation_prompt_budget(
                FakeTokenizer(),
                self.reference,
                path,
                expected_rows=2,
                expected_by_task={"translation": 1, "summarization": 1},
                prompt_budget_tokens={
                    "translation": 100_000,
                    "summarization": 100_000,
                },
            )
        inventory = result["input_unicode_inventory"]
        self.assertEqual(inventory["anomalous_rows"], 1)
        self.assertEqual(inventory["replacement_character_rows"], 1)
        self.assertEqual(inventory["replacement_character_occurrences"], 1)
        self.assertEqual(inventory["c1_control_rows"], 1)
        self.assertEqual(inventory["c1_control_occurrences"], 1)
        self.assertEqual(inventory["surrogate_codepoint_occurrences"], 0)
        self.assertEqual(inventory["tokenizer_roundtrip"]["failures"], 0)
        self.assertEqual(
            inventory["anomalous_row_records"],
            [
                {
                    "id": "validation-translation",
                    "task": "translation",
                    "replacement_character_occurrences": 1,
                    "c1_control_occurrences": 1,
                }
            ],
        )

    def test_b200_validation_source_roundtrip_change_is_rejected(self) -> None:
        class ChangingTokenizer(FakeTokenizer):
            def decode(
                self,
                input_ids: list[int],
                *,
                skip_special_tokens: bool,
                clean_up_tokenization_spaces: bool,
            ) -> str:
                return (
                    super()
                    .decode(
                        input_ids,
                        skip_special_tokens=skip_special_tokens,
                        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                    )
                    .replace("\u0080", "")
                )

        row = {
            "id": "validation-translation",
            "task": "translation",
            "origin": "C1 \u0080 원문",
            "response": "정상 검증 본문",
            "prompt_variant": "no_dictionary_context",
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "validation.jsonl"
            path.write_text(
                json.dumps(row, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "round-trip changed"):
                gate.scan_b200_validation_prompt_budget(
                    ChangingTokenizer(),
                    self.reference,
                    path,
                    expected_rows=1,
                    expected_by_task={"translation": 1},
                    prompt_budget_tokens={
                        "translation": 100_000,
                        "summarization": 100_000,
                    },
                )

    def test_b200_input_unicode_inventory_rejects_boolean_count(self) -> None:
        inventory = empty_b200_input_unicode_inventory(2)
        inventory["utf8_encoding_failures"] = False
        with self.assertRaisesRegex(RuntimeError, "is not a count"):
            gate.validate_b200_input_unicode_inventory(
                inventory,
                expected_rows=2,
            )

    def test_b200_input_unicode_inventory_pins_expected_anomalies(self) -> None:
        inventory = empty_b200_input_unicode_inventory(2)
        expected = {
            key: 0
            for key in gate.B200_VALIDATION_UNICODE_EXPECTED
            if key != "anomalous_row_records_sha256"
        }
        expected["anomalous_row_records_sha256"] = gate.canonical_sha256([])
        gate.validate_b200_input_unicode_inventory(
            inventory,
            expected_rows=2,
            expected_anomalies=expected,
        )
        expected["anomalous_rows"] = 1
        with self.assertRaisesRegex(RuntimeError, "anomalous_rows changed"):
            gate.validate_b200_input_unicode_inventory(
                inventory,
                expected_rows=2,
                expected_anomalies=expected,
            )

    def test_b200_validation_scan_rejects_invalid_unicode(self) -> None:
        row = {
            "id": "validation-translation",
            "task": "translation",
            "origin": "검증 원문",
            "response": "손상된 \ufffd 본문",
            "prompt_variant": "no_dictionary_context",
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "validation.jsonl"
            path.write_text(
                json.dumps(row, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "replacement character"):
                gate.scan_b200_validation_prompt_budget(
                    FakeTokenizer(),
                    self.reference,
                    path,
                    expected_rows=1,
                    expected_by_task={"translation": 1},
                    prompt_budget_tokens={
                        "translation": 100_000,
                        "summarization": 100_000,
                    },
                )

    def test_b200_data_gate_hashes_only_train_and_validation(self) -> None:
        bundle = gate.b200_runtime.load_b200_config_bundle(
            gate.b200_runtime.B200_CONFIG_DIR
        )
        contract = bundle["data_contract"]
        expected = {
            Path(contract["prepared_manifest"]): contract["prepared_manifest_sha256"],
            Path(contract["files"]["train"]["path"]): contract["files"]["train"][
                "sha256"
            ],
            Path(contract["files"]["validation"]["path"]): contract["files"][
                "validation"
            ]["sha256"],
        }
        opened: list[Path] = []

        def fake_sha256(path: Path) -> str:
            resolved = Path(path)
            opened.append(resolved)
            if resolved not in expected:
                raise AssertionError(f"sealed dataset was accessed: {resolved}")
            return expected[resolved]

        with mock.patch.object(
            gate.b200_runtime, "sha256_file", side_effect=fake_sha256
        ), mock.patch.object(gate, "select_contract_rows", return_value=fixture_rows()):
            _rows, evidence = gate.validate_b200_data_dir(
                gate.b200_runtime.DATA_DIR,
                bundle,
            )
        self.assertEqual(set(opened), set(expected))
        self.assertEqual(set(evidence["files"]), {"train", "validation"})
        self.assertEqual(evidence["sealed_evaluation_dataset_roles_opened"], [])

    def test_b200_run_binds_config_and_never_selects_sealed_data(self) -> None:
        tokenizer = FakeTokenizer()
        bundle = gate.b200_runtime.load_b200_config_bundle(
            gate.b200_runtime.B200_CONFIG_DIR
        )
        validation_entry = bundle["data_contract"]["files"]["validation"]
        validation_rows = int(validation_entry["rows"])
        expected_anomalies = {
            key: 0
            for key in gate.B200_VALIDATION_UNICODE_EXPECTED
            if key != "anomalous_row_records_sha256"
        }
        expected_anomalies["anomalous_row_records_sha256"] = gate.canonical_sha256([])

        def source_validator(_path: Path, _bundle: dict[str, Any]) -> dict[str, Any]:
            return {
                "status": "PASS",
                "generation_eos_token_ids": [tokenizer.eos_token_id],
            }

        def data_validator(_path: Path, _bundle: dict[str, Any]) -> tuple[Any, Any]:
            return fixture_rows(), {
                "files": {
                    "train": {"path": "train.jsonl"},
                    "validation": {"path": "validation.jsonl"},
                },
                "opened_dataset_roles": ["train", "validation"],
                "sealed_evaluation_dataset_roles_opened": [],
            }

        def scanner(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {
                "status": "PASS",
                "path": "validation.jsonl",
                "rows": validation_rows,
                "by_task": validation_entry["by_task"],
                "prompt_budget_tokens": gate.PROMPT_BUDGET_TOKENS,
                "truncation_count": 0,
                "input_unicode_inventory": empty_b200_input_unicode_inventory(
                    validation_rows
                ),
                "invalid_unicode_count": 0,
                "target_invalid_unicode_count": 0,
                "forbidden_output_framing_count": 0,
                "target_prefix_removal_count": 0,
                "exact_output_only_preamble_count": validation_rows,
            }

        with mock.patch.object(
            gate,
            "B200_VALIDATION_UNICODE_EXPECTED",
            expected_anomalies,
        ):
            result = gate.run_b200(
                config_dir=gate.b200_runtime.B200_CONFIG_DIR,
                source_dir=gate.b200_runtime.BF16_SOURCE_DIR,
                data_dir=gate.b200_runtime.DATA_DIR,
                tokenizer=tokenizer,
                source_validator=source_validator,
                data_validator=data_validator,
                validation_scanner=scanner,
            )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["profile"], "b200")
        self.assertEqual(
            set(result["config"]["files"]),
            {
                "model",
                "recipe",
                "data_contract",
            },
        )
        self.assertEqual(result["checks"]["sealed_evaluation_not_opened"], "PASS")

    def test_b200_receipt_refuses_overwrite_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory(
            dir=gate.b200_runtime.OWNED_ROOT / "runtime/tmp"
        ) as temporary:
            directory = Path(temporary)
            existing = directory / "existing.json"
            existing.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                gate.require_fresh_b200_receipt(existing)
            target = directory / "target.json"
            link = directory / "link.json"
            link.symlink_to(target)
            with self.assertRaisesRegex(RuntimeError, "symlink"):
                gate.require_fresh_b200_receipt(link)


if __name__ == "__main__":
    unittest.main()
