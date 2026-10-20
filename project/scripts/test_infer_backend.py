#!/usr/bin/env python3
"""CPU-only contract tests for the Solar inference backend."""

from __future__ import annotations

import json
import copy
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

import infer  # noqa: E402
from b200_runtime import B200_ASSET_INTEGRITY_RECEIPT  # noqa: E402
from quantization.contract import DEFAULT_SOURCE_DIR  # noqa: E402
from solar_peft_compat import solar_open2_peft_v5_native  # noqa: E402


class InferenceContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {
                "id": "row-1",
                "task": "translation",
                "origin": "source",
                "response": "reference",
                "reference_source": "fixture",
            }
        ]
        decoding = {
            "temperature": 0.0,
            "top_p": 1.0,
            "do_sample": False,
        }
        self.contract = {
            "schema": infer.PROVENANCE_SCHEMA,
            "schema_version": infer.PROVENANCE_SCHEMA_VERSION,
            "model_label": "Solar-base",
            "source": {
                "model_id": "upstage/Solar-Open2-250B",
                "revision": infer.BASE_REVISION,
                "index_sha256": "source-index",
                "config_sha256": "source-config",
            },
            "per_expert_nf4": {
                "schema": infer.SCHEMA_NAME,
                "manifest_sha256": "expert-manifest",
                "expert_index_sha256": "expert-index",
            },
            "adapter": None,
            "evaluation": {"sha256": "eval-sha"},
            "decoding": decoding,
            "decoding_sha256": infer.canonical_sha256(decoding),
        }

    def prediction(self) -> dict[str, object]:
        digest = infer.canonical_sha256(self.contract)
        return {
            "id": "row-1",
            "task": "translation",
            **infer.expected_row_provenance(self.contract, digest),
            "origin": "source",
            "reference": "reference",
            "reference_source": "fixture",
            "prompt_variant": "no_dictionary_context",
            "reasoning_effort": "none",
            "prediction": "prediction",
        }

    def test_manifest_name_and_exact_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "predictions.jsonl"
            manifest_path, _manifest, previous, done = infer.initialize_or_resume(
                output_jsonl=output,
                contract=self.contract,
                selected_rows=self.rows,
                resume=True,
            )
            self.assertEqual(
                manifest_path.name,
                "predictions.jsonl.manifest.json",
            )
            self.assertEqual(previous, [])
            self.assertEqual(done, set())
            output.write_text(
                json.dumps(self.prediction(), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            _path, _manifest, previous, done = infer.initialize_or_resume(
                output_jsonl=output,
                contract=self.contract,
                selected_rows=self.rows,
                resume=True,
            )
            self.assertEqual(len(previous), 1)
            self.assertEqual(done, {("row-1", "translation")})

    def test_resume_rejects_mixed_contract_and_duplicate_keys(self) -> None:
        digest = infer.canonical_sha256(self.contract)
        prediction = self.prediction()
        prediction["inference_contract_sha256"] = "wrong"
        with self.assertRaisesRegex(RuntimeError, "provenance mismatch"):
            infer.validate_previous_rows(
                [prediction],
                selected_rows=self.rows,
                contract=self.contract,
                contract_sha256=digest,
            )
        prediction = self.prediction()
        with self.assertRaisesRegex(RuntimeError, "duplicate key"):
            infer.validate_previous_rows(
                [prediction, prediction],
                selected_rows=self.rows,
                contract=self.contract,
                contract_sha256=digest,
            )

    def test_base_row_has_null_adapter_file_map_and_full_decoding(self) -> None:
        values = infer.expected_row_provenance(
            self.contract,
            infer.canonical_sha256(self.contract),
        )
        self.assertIsNone(values["adapter_file_sha256"])
        self.assertIsNone(values["provenance"]["adapter_file_sha256"])
        self.assertEqual(values["decoding"], self.contract["decoding"])
        self.assertEqual(
            values["provenance"]["decoding"],
            self.contract["decoding"],
        )

    def test_cpu_padding_and_stop_trimming(self) -> None:
        input_ids, attention_mask = infer.left_pad(
            [[1, 2], [3]],
            pad_token_id=0,
        )
        self.assertEqual(input_ids.tolist(), [[1, 2], [0, 3]])
        self.assertEqual(attention_mask.tolist(), [[1, 1], [0, 1]])
        self.assertEqual(
            infer.trim_generated_ids(
                [7, 8, 2, 0],
                eos_token_ids={2},
                pad_token_id=0,
            ),
            [7, 8],
        )
        self.assertEqual(
            infer.classify_generation_stop(
                [7, 8, 2, 0],
                eos_token_ids={2, 129},
                max_new_tokens=4,
            ),
            "eos",
        )
        self.assertEqual(
            infer.classify_generation_stop(
                [7, 8, 9, 10],
                eos_token_ids={2, 129},
                max_new_tokens=4,
            ),
            "max_new_tokens",
        )
        self.assertEqual(
            infer.classify_generation_stop(
                [7, 8],
                eos_token_ids={2, 129},
                max_new_tokens=4,
            ),
            "stopped_without_eos",
        )

    def test_inference_uses_strict_shared_output_only_contract(self) -> None:
        examples = (
            ("**번역:** 현대 한국어 본문", "result_label"),
            ("요약하면, 현대 한국어 본문", "meta_framing"),
            ("I cannot translate this.", "refusal_framing"),
            ("> 현대 한국어 본문", "blockquote_framing"),
            ("純粹漢字本文", "non_korean_body"),
        )
        for prediction, category in examples:
            with self.subTest(prediction=prediction):
                self.assertIn(
                    category,
                    infer.output_only_violations(prediction),
                )

    def test_sealing_requires_verified_runtime_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "predictions.jsonl"
            manifest_path, manifest, _previous, _done = infer.initialize_or_resume(
                output_jsonl=output,
                contract=self.contract,
                selected_rows=self.rows,
                resume=True,
            )
            output.write_text(
                json.dumps(self.prediction(), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "runtime audit"):
                infer.seal_predictions(
                    manifest_path,
                    manifest,
                    output_jsonl=output,
                    completed_rows=1,
                )
            infer.record_runtime_audit(
                manifest_path,
                manifest,
                loader_receipt={
                    "model_audit": {"status": "PASS"},
                    "hash_verification": True,
                    "production_eligible": True,
                },
                adapter_audit=None,
            )
            infer.seal_predictions(
                manifest_path,
                manifest,
                output_jsonl=output,
                completed_rows=1,
            )
            sealed = infer.read_json(manifest_path)
            self.assertEqual(sealed["status"], "COMPLETE")
            self.assertIsNone(sealed["runtime_audit"]["adapter_audit"])


class _FakeReference:
    @staticmethod
    def clean_text(value: object) -> str:
        return str(value).strip()

    @classmethod
    def build_messages_no_dict(
        cls,
        task: str,
        origin: str,
    ) -> list[dict[str, str]]:
        return [{"role": "user", "content": f"{task}:{origin}"}]

    @classmethod
    def validate_eval_rows(
        cls,
        rows: list[dict[str, object]],
        _source: str,
    ) -> dict[str, object]:
        for row in rows:
            expected = cls.build_messages_no_dict(
                str(row["task"]),
                cls.clean_text(row["origin"]),
            )
            if row.get("messages") != expected:
                raise ValueError("noncanonical messages")
        return {"status": "PASS", "rows": len(rows)}


class B200InferenceContractTest(unittest.TestCase):
    def test_cli_defaults_are_exact_two_gpu_b200_contract(self) -> None:
        parser = infer.build_argument_parser()
        args = parser.parse_args(
            [
                "--config-dir",
                str(infer.B200_CONFIG_DIR),
                "--mode",
                infer.B200_VALIDATION_SMOKE_MODE,
                "--source-dir",
                str(infer.BF16_SOURCE_DIR),
                "--expert-nf4-dir",
                str(infer.INFERENCE_EXPERT_DIR),
                "--asset-integrity-receipt",
                str(B200_ASSET_INTEGRITY_RECEIPT),
                "--model-label",
                "base",
                "--eval-jsonl",
                "data/validation.jsonl",
                "--output-jsonl",
                "runtime/tmp/base-smoke.jsonl",
            ]
        )
        bundle = {
            "recipe": {
                "evaluation": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_model_len": 16_384,
                    "translation_max_new_tokens": 8_192,
                    "summarization_max_new_tokens": 6_144,
                }
            }
        }
        infer.validate_b200_arguments(args, bundle)
        self.assertEqual(args.tensor_parallel_size, 2)
        self.assertEqual(args.limit_per_task, 0)
        self.assertEqual(
            args.asset_integrity_receipt,
            B200_ASSET_INTEGRITY_RECEIPT,
        )
        args.tensor_parallel_size = 3
        with self.assertRaisesRegex(ValueError, "tensor_parallel_size=2"):
            infer.validate_b200_arguments(args, bundle)

    def test_loader_call_uses_receipt_backed_hash_mode(self) -> None:
        source = Path(infer.__file__).read_text(encoding="utf-8")
        self.assertIn(
            "verify_hashes=False,\n"
            "        asset_integrity_receipt=asset_integrity_receipt,",
            source,
        )

    def test_deterministic_validation_smoke_selects_first_one_per_task(self) -> None:
        rows = [
            {
                "id": "t0",
                "task": "translation",
                "origin": " first translation ",
                "response": "reference",
                "source_split": "validation",
                "prompt_variant": "no_dictionary_context",
            },
            {
                "id": "t1",
                "task": "translation",
                "origin": "second translation",
                "response": "reference",
                "source_split": "validation",
                "prompt_variant": "no_dictionary_context",
            },
            {
                "id": "s0",
                "task": "summarization",
                "origin": " first summary ",
                "response": "reference",
                "source_split": "validation",
                "prompt_variant": "no_dictionary_context",
            },
            {
                "id": "s1",
                "task": "summarization",
                "origin": "second summary",
                "response": "reference",
                "source_split": "validation",
                "prompt_variant": "no_dictionary_context",
            },
        ]
        selected, positions = infer.deterministic_validation_smoke_rows(
            rows,
            _FakeReference,
        )
        self.assertEqual(
            [(row["id"], row["task"]) for row in selected],
            [("t0", "translation"), ("s0", "summarization")],
        )
        self.assertEqual(positions, [0, 2])
        self.assertNotIn("messages", rows[0])

    def test_smoke_rejects_every_sealed_evaluation_path(self) -> None:
        bundle = infer.load_b200_config_bundle(infer.B200_CONFIG_DIR)
        for role in ("full_test", "human_gold", "combined_eval"):
            path = Path(bundle["data_contract"]["files"][role]["path"])
            with self.subTest(role=role), self.assertRaisesRegex(
                RuntimeError,
                "forbids sealed evaluation path",
            ):
                infer.reject_b200_smoke_sealed_path(path, bundle)

    def test_render_prompt_overflow_fails_without_truncation(self) -> None:
        row = {
            "id": "row",
            "task": "translation",
            "origin": "source",
        }
        row["messages"] = _FakeReference.build_messages_no_dict(
            "translation",
            "source",
        )

        class Tokenizer:
            @staticmethod
            def apply_chat_template(*_args: object, **_kwargs: object) -> str:
                return infer.SOLAR_OUTPUT_ONLY_GENERATION_PREAMBLE

            @staticmethod
            def __call__(*_args: object, **_kwargs: object) -> dict[str, list[int]]:
                return {"input_ids": [1, 2, 3, 4]}

        with self.assertRaisesRegex(RuntimeError, "truncation is forbidden"):
            infer.render_prompt_ids(
                Tokenizer(),
                _FakeReference,
                row,
                max_prompt_tokens=3,
            )

    @staticmethod
    def clean_generation() -> dict[str, object]:
        return {
            "prediction": "현대 한국어 본문",
            "raw_continuation": "  현대 한국어 본문\n",
            "prompt_token_ids": [10, 11],
            "raw_generation_token_ids": [20, 21, 2, 0],
            "prediction_token_ids": [20, 21],
            "prompt_tokens": 2,
            "prediction_tokens": 2,
            "generation_stop_reason": "eos",
            "eos_position": 2,
            "eos_token_ids": [2],
            "special_token_ids_in_prediction": [],
            "tokenizer_roundtrip_exact": True,
            "unicode_roundtrip_exact": True,
            "output_violation_audit": {
                "status": "PASS",
                "violations": [],
                "postprocessing": "outer_strip_only",
            },
        }

    def test_generation_audit_accepts_clean_raw_continuation(self) -> None:
        infer.validate_generation_record(
            self.clean_generation(),
            task="translation",
            max_new_tokens=8_192,
        )

    def test_generation_audit_rejects_eos_unicode_and_output_pollution(self) -> None:
        cases = {
            "control_token": "<|im:end|>",
            "result_label": "답변: 현대 한국어 본문",
            "markdown": "```text\n현대 한국어 본문\n```",
            "prompt_echo": "Return only the translated Korean text",
            "replacement": "본문\ufffd",
            "surrogate": "본문\ud800",
            "control": "본문\r다음",
        }
        for label, prediction in cases.items():
            record = self.clean_generation()
            record["prediction"] = prediction
            record["raw_continuation"] = prediction
            with self.subTest(label=label), self.assertRaisesRegex(
                RuntimeError,
                "violates B200 output contract",
            ):
                infer.validate_generation_record(
                    record,
                    task="translation",
                    max_new_tokens=8_192,
                )
        no_eos = self.clean_generation()
        no_eos["generation_stop_reason"] = "max_new_tokens"
        no_eos["eos_position"] = None
        with self.assertRaisesRegex(RuntimeError, "generation_did_not_stop_at_eos"):
            infer.validate_generation_record(
                no_eos,
                task="translation",
                max_new_tokens=8_192,
            )

    def test_terminal_training_success_is_exact_and_hash_bound(self) -> None:
        bundle = infer.load_b200_config_bundle(infer.B200_CONFIG_DIR)
        identity = infer.b200_config_identity(infer.B200_CONFIG_DIR)
        recipe = bundle["recipe"]
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            records = {}
            for label, dirname in (
                ("final_checkpoint", "final_adapter"),
                ("selected_checkpoint", "best_adapter"),
            ):
                directory = run_dir / dirname
                directory.mkdir()
                (directory / "adapter_config.json").write_text(
                    "{}\n",
                    encoding="utf-8",
                )
                (directory / "_SUCCESS").write_text("PASS\n", encoding="utf-8")
                files = {
                    child.name: infer.sha256(child)
                    for child in sorted(directory.iterdir())
                    if child.is_file()
                }
                records[label] = {
                    "step": 9_316,
                    "adapter_dir": str(directory),
                    "files_sha256": files,
                }
            records["selected_checkpoint"].update(
                {
                    "policy": recipe["validation"]["selection_policy"],
                    "tie_break": recipe["validation"]["selection_tie_break"],
                    "selection_loss": 1.0,
                }
            )
            selection = {"selected_step": 9_316}
            manifest = {
                "status": "COMPLETED",
                "profile": "b200",
                "config_identity": identity,
                "completed_updates": 9_316,
                "target_updates": 9_316,
                "consumed_occurrences": 149_048,
                "state_contract": {
                    "profile": "b200",
                    "smoke": False,
                    "config_identity": identity,
                },
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
                },
                "wandb_reporting": {"completion_status": "COMMITTED"},
                "selection_record": selection,
                **records,
            }
            (run_dir / "run_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            (run_dir / "_SUCCESS").write_text("PASS\n", encoding="utf-8")
            receipt = infer.validate_terminal_training_success(
                run_dir,
                config_identity=identity,
                recipe=recipe,
            )
            self.assertEqual(receipt["status"], "PASS")
            manifest["completed_updates"] = 9_315
            (run_dir / "run_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "completed_updates"):
                infer.validate_terminal_training_success(
                    run_dir,
                    config_identity=identity,
                    recipe=recipe,
                )


@unittest.skipUnless(
    (DEFAULT_SOURCE_DIR / "config.json").is_file(),
    "official Solar config has not been downloaded",
)
class SolarPeftCompatibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from transformers import AutoConfig

        cls.official_config = AutoConfig.from_pretrained(
            DEFAULT_SOURCE_DIR,
            local_files_only=True,
            trust_remote_code=False,
        )
        cls.target_regex = infer.read_json(infer.CONFIG_DIR / "recipe.json")["lora"][
            "target_regex"
        ]

    def test_official_meta_injection_preserves_528_target_regex(self) -> None:
        from accelerate import init_empty_weights
        from peft import LoraConfig, get_peft_model
        from peft.utils.transformers_weight_conversion import (
            get_checkpoint_conversion_mapping,
        )
        from transformers import AutoModelForCausalLM

        with init_empty_weights(include_buffers=True):
            model = AutoModelForCausalLM.from_config(
                copy.deepcopy(self.official_config),
                trust_remote_code=False,
            )
            with solar_open2_peft_v5_native():
                model = get_peft_model(
                    model,
                    LoraConfig(
                        r=32,
                        lora_alpha=64,
                        lora_dropout=0.05,
                        bias="none",
                        task_type="CAUSAL_LM",
                        target_modules=self.target_regex,
                    ),
                )
        audit = infer.audit_lora_modules(
            model,
            {
                "num_hidden_layers": 48,
                "gqa_layers": self.official_config.gqa_layers,
                "kda_use_full_proj": False,
            },
        )
        adapter_parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if ".lora_A." in name or ".lora_B." in name
        }
        self.assertEqual(audit["module_count"], 528)
        self.assertEqual(len(adapter_parameters), 1_056)
        self.assertEqual(
            sum(parameter.numel() for parameter in adapter_parameters.values()),
            133_177_344,
        )
        self.assertIsInstance(
            model.peft_config["default"].target_modules,
            str,
        )
        self.assertEqual(
            model.peft_config["default"].target_modules,
            self.target_regex,
        )
        self.assertIsNotNone(get_checkpoint_conversion_mapping("solar_open2"))

    def test_tiny_adapter_inject_save_and_fresh_reload(self) -> None:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from peft.utils import get_peft_model_state_dict
        from transformers import AutoModelForCausalLM

        config = copy.deepcopy(self.official_config)
        config._name_or_path = ""
        config.num_hidden_layers = 1
        config.hidden_size = 32
        config.num_attention_heads = 4
        config.num_key_value_heads = 2
        config.head_dim = 8
        config.intermediate_size = 64
        config.moe_intermediate_size = 16
        config.vocab_size = 128
        config.max_position_embeddings = 128
        config.n_routed_experts = 4
        config.n_shared_experts = 1
        config.num_experts_per_tok = 2
        config.gqa_layers = [0]
        config.layer_types = ["full_attention"]
        config.linear_attn_config = {
            "short_conv_kernel_size": 2,
            "head_dim": 8,
            "num_heads": 4,
            "num_kv_heads": None,
        }
        model = AutoModelForCausalLM.from_config(config)
        with solar_open2_peft_v5_native():
            model = get_peft_model(
                model,
                LoraConfig(
                    r=2,
                    lora_alpha=4,
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=self.target_regex,
                ),
            )
        for index, (name, parameter) in enumerate(model.named_parameters()):
            if ".lora_A." in name or ".lora_B." in name:
                parameter.data.fill_((index + 1) / 1000)
        expected = {
            key: value.detach().clone()
            for key, value in get_peft_model_state_dict(model).items()
        }
        with tempfile.TemporaryDirectory() as temporary:
            # Saving/get_peft_model_state_dict does not invoke the v5 conversion;
            # only injection and state loading require the compatibility guard.
            model.save_pretrained(
                temporary,
                safe_serialization=True,
                save_embedding_layers=False,
            )
            saved_config = LoraConfig.from_pretrained(
                temporary,
                local_files_only=True,
            )
            self.assertEqual(saved_config.target_modules, self.target_regex)
            fresh_base = AutoModelForCausalLM.from_config(config)
            with solar_open2_peft_v5_native():
                reloaded = PeftModel.from_pretrained(
                    fresh_base,
                    temporary,
                    local_files_only=True,
                    is_trainable=False,
                    autocast_adapter_dtype=True,
                )
            actual = get_peft_model_state_dict(reloaded)
        self.assertEqual(
            reloaded.peft_config["default"].target_modules, self.target_regex
        )
        self.assertEqual(expected.keys(), actual.keys())
        self.assertTrue(
            all(torch.equal(expected[key], actual[key]) for key in expected)
        )
        self.assertEqual(
            {parameter.dtype for parameter in actual.values()},
            {torch.float32},
        )


if __name__ == "__main__":
    unittest.main()
