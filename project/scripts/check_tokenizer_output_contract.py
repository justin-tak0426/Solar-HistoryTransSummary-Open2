#!/usr/bin/env python3
"""Certify the real Solar tokenizer's train/inference output-only boundary."""

from __future__ import annotations

import argparse
import getpass
import json
from collections import Counter
from pathlib import Path
from typing import Any

import b200_runtime

from common import (
    CONFIG_DIR,
    PROJECT,
    REFERENCE_SHA256,
    RESULT_LABEL_PREFIX_RE,
    SOLAR_ASSISTANT_TEMPLATE_TERMINATION,
    SOLAR_OUTPUT_ONLY_GENERATION_PREAMBLE,
    TRAINING_RESULT_LABEL_IDS,
    atomic_json,
    canonical_sha256,
    encode_row,
    iter_jsonl,
    load_reference_module,
    output_only_violations,
    pipeline_code_identity,
    read_json,
    require_owned_path,
    sha256,
)
from infer import (
    PINNED_COMBINED_EVAL_ROWS,
    PINNED_COMBINED_EVAL_TASK_COUNTS,
    TEMPLATE_KWARGS,
    render_prompt_ids,
)
from quantization.contract import BASE_MODEL_ID, BASE_REVISION


EXPECTED_TEMPLATE_KWARGS = {
    "reasoning_effort": "none",
    "think_render_option": "preserved",
    "provider_system_prompt": False,
}
EXPECTED_SOURCE_DIR = Path(
    "/data1/jhtak/Solar-HistoryTransSummary-Open2/hf_cache/"
    f"models--upstage--Solar-Open2-250B/snapshots/{BASE_REVISION}"
)
SOURCE_TOKENIZER_FILES = (
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
)
PROMPT_BUDGET_TOKENS = {
    "translation": 8_192,
    "summarization": 10_240,
}
B200_VALIDATION_UNICODE_EXPECTED = {
    "replacement_character_rows": 34,
    "replacement_character_occurrences": 44,
    "c0_non_whitespace_control_rows": 0,
    "c0_non_whitespace_control_occurrences": 0,
    "c1_control_rows": 38,
    "c1_control_occurrences": 58,
    "anomalous_rows": 66,
    "anomalous_row_records_sha256": (
        "52ce2074d6c48967be966098f2d56f51f3d7303b6addf981f5861190798570c4"
    ),
}


class RecordingTokenizer:
    """Delegate to a tokenizer while retaining exact chat-template calls."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.template_calls: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.tokenizer, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.tokenizer(*args, **kwargs)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        rendered = self.tokenizer.apply_chat_template(
            messages,
            *args,
            **kwargs,
        )
        self.template_calls.append(
            {
                "messages": messages,
                "args": args,
                "kwargs": dict(kwargs),
                "rendered": rendered,
            }
        )
        return rendered

    def clear_calls(self) -> None:
        self.template_calls.clear()


def _expect_template_call(
    call: dict[str, Any],
    *,
    messages: list[dict[str, str]],
    add_generation_prompt: bool,
    label: str,
) -> str:
    if call["args"]:
        raise RuntimeError(f"{label} passed positional chat-template options")
    expected_kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
        **EXPECTED_TEMPLATE_KWARGS,
    }
    if call["kwargs"] != expected_kwargs:
        raise RuntimeError(
            f"{label} template kwargs changed: "
            f"{call['kwargs']!r} != {expected_kwargs!r}"
        )
    if call["messages"] != messages:
        raise RuntimeError(f"{label} did not use the exact reference messages")
    rendered = call["rendered"]
    if not isinstance(rendered, str) or not rendered:
        raise RuntimeError(f"{label} rendered an empty/non-text prompt")
    return rendered


def _answer_body(row: dict[str, Any]) -> tuple[str, bool]:
    row_id = str(row["id"])
    raw = str(row["response"])
    match = RESULT_LABEL_PREFIX_RE.match(raw)
    expected_prefix = row_id in TRAINING_RESULT_LABEL_IDS
    if (match is not None) != expected_prefix:
        raise RuntimeError(
            f"pinned answer-prefix state changed for {row_id}: "
            f"observed={match is not None} expected={expected_prefix}"
        )
    body = raw[match.end() :] if match is not None else raw
    _validate_unicode_text(body, label=f"target body {row_id}")
    if not body or body != body.strip():
        raise RuntimeError(f"target body has invalid outer whitespace: {row_id}")
    violations = output_only_violations(body)
    if violations:
        raise RuntimeError(
            f"target body contains forbidden framing {violations}: {row_id}"
        )
    return body, expected_prefix


def _validate_unicode_text(text: str, *, label: str) -> None:
    if "\ufffd" in text:
        raise RuntimeError(f"{label} contains the replacement character")
    for character in text:
        codepoint = ord(character)
        if 0xD800 <= codepoint <= 0xDFFF:
            raise RuntimeError(f"{label} contains a surrogate codepoint")
        if (
            codepoint < 0x20 or 0x7F <= codepoint <= 0x9F
        ) and character not in "\n\r\t":
            raise RuntimeError(f"{label} contains a forbidden control character")
    try:
        text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise RuntimeError(f"{label} is not valid Unicode") from exc


def _source_unicode_inventory(text: str, *, label: str) -> dict[str, int]:
    """Inventory sealed source anomalies without rewriting historical text."""

    try:
        text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise RuntimeError(f"{label} is not valid UTF-8 encodable Unicode") from exc
    counts = {
        "replacement_character_occurrences": 0,
        "c0_non_whitespace_control_occurrences": 0,
        "c1_control_occurrences": 0,
        "surrogate_codepoint_occurrences": 0,
    }
    for character in text:
        codepoint = ord(character)
        if 0xD800 <= codepoint <= 0xDFFF:
            counts["surrogate_codepoint_occurrences"] += 1
        elif character == "\ufffd":
            counts["replacement_character_occurrences"] += 1
        elif codepoint < 0x20 and character not in "\n\r\t":
            counts["c0_non_whitespace_control_occurrences"] += 1
        elif 0x7F <= codepoint <= 0x9F:
            counts["c1_control_occurrences"] += 1
    if counts["surrogate_codepoint_occurrences"]:
        raise RuntimeError(f"{label} contains a surrogate codepoint")
    return counts


def validate_b200_input_unicode_inventory(
    value: Any,
    *,
    expected_rows: int,
    expected_anomalies: dict[str, Any] | None = None,
) -> dict[str, Any]:
    def require_count(container: dict[str, Any], key: str) -> int:
        count = container.get(key)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise RuntimeError(f"B200 input Unicode {key} is not a count")
        return count

    if not isinstance(value, dict):
        raise RuntimeError("B200 input Unicode inventory is not an object")
    if (
        isinstance(expected_rows, bool)
        or not isinstance(expected_rows, int)
        or expected_rows <= 0
    ):
        raise RuntimeError("B200 input Unicode expected row count is invalid")
    expected_fields = {
        "scope",
        "rows_scanned",
        "utf8_encoding_failures",
        "surrogate_codepoint_occurrences",
        "replacement_character_rows",
        "replacement_character_occurrences",
        "c0_non_whitespace_control_rows",
        "c0_non_whitespace_control_occurrences",
        "c1_control_rows",
        "c1_control_occurrences",
        "anomalous_rows",
        "anomalous_row_records",
        "anomalous_row_records_sha256",
        "tokenizer_roundtrip",
    }
    if set(value) != expected_fields:
        raise RuntimeError("B200 input Unicode inventory fields changed")
    if value.get("scope") != "sealed validation origin before reference cleaning":
        raise RuntimeError("B200 input Unicode inventory scope changed")
    for key, expected in (
        ("rows_scanned", expected_rows),
        ("utf8_encoding_failures", 0),
        ("surrogate_codepoint_occurrences", 0),
    ):
        actual = require_count(value, key)
        if actual != expected:
            raise RuntimeError(
                f"B200 input Unicode inventory {key} changed: "
                f"{actual!r} != {expected!r}"
            )
    records = value.get("anomalous_row_records")
    if not isinstance(records, list):
        raise RuntimeError("B200 anomalous source rows are not a list")
    if len(records) > expected_rows:
        raise RuntimeError("B200 anomalous source rows exceed scanned rows")
    allowed_occurrence_fields = {
        "replacement_character_occurrences",
        "c0_non_whitespace_control_occurrences",
        "c1_control_occurrences",
    }
    occurrence_totals: Counter[str] = Counter()
    row_totals: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    for record in records:
        if not isinstance(record, dict):
            raise RuntimeError("B200 anomalous source row is not an object")
        keys = set(record)
        if not {"id", "task"} <= keys or not (
            (keys - {"id", "task"}) <= allowed_occurrence_fields
        ):
            raise RuntimeError("B200 anomalous source row fields changed")
        row_id = record.get("id")
        task = record.get("task")
        if (
            not isinstance(row_id, str)
            or not row_id
            or not isinstance(task, str)
            or task not in {"translation", "summarization"}
        ):
            raise RuntimeError("B200 anomalous source row identity is invalid")
        identity = (row_id, task)
        if identity in seen:
            raise RuntimeError("B200 anomalous source row identity is invalid")
        seen.add(identity)
        anomaly_fields = keys - {"id", "task"}
        if not anomaly_fields:
            raise RuntimeError("B200 anomalous source row contains no anomaly")
        for field in anomaly_fields:
            count = record[field]
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise RuntimeError("B200 anomalous source count is invalid")
            occurrence_totals[field] += count
            row_totals[field] += 1
    if require_count(value, "anomalous_rows") != len(records):
        raise RuntimeError("B200 anomalous source row count changed")
    records_sha256 = value.get("anomalous_row_records_sha256")
    if not isinstance(records_sha256, str) or records_sha256 != canonical_sha256(
        records
    ):
        raise RuntimeError("B200 anomalous source row inventory hash changed")
    for field in allowed_occurrence_fields:
        row_field = field.removesuffix("_occurrences") + "_rows"
        if require_count(value, field) != occurrence_totals[field]:
            raise RuntimeError(f"B200 input Unicode {field} total changed")
        if require_count(value, row_field) != row_totals[field]:
            raise RuntimeError(f"B200 input Unicode {row_field} total changed")
    if expected_anomalies is not None:
        expected_anomaly_fields = {
            "replacement_character_rows",
            "replacement_character_occurrences",
            "c0_non_whitespace_control_rows",
            "c0_non_whitespace_control_occurrences",
            "c1_control_rows",
            "c1_control_occurrences",
            "anomalous_rows",
            "anomalous_row_records_sha256",
        }
        if set(expected_anomalies) != expected_anomaly_fields:
            raise RuntimeError("B200 expected Unicode anomaly fields changed")
        for key, expected in expected_anomalies.items():
            actual = (
                records_sha256
                if key == "anomalous_row_records_sha256"
                else require_count(value, key)
            )
            if actual != expected:
                raise RuntimeError(
                    f"B200 sealed validation Unicode {key} changed: "
                    f"{actual!r} != {expected!r}"
                )
    roundtrip = value.get("tokenizer_roundtrip")
    if not isinstance(roundtrip, dict):
        raise RuntimeError("B200 sealed validation tokenizer round-trip is invalid")
    expected_roundtrip = {
        "rows_checked": expected_rows,
        "exact_roundtrip_rows": expected_rows,
        "failures": 0,
    }
    if set(roundtrip) != set(expected_roundtrip):
        raise RuntimeError("B200 sealed validation tokenizer round-trip fields changed")
    for key, expected in expected_roundtrip.items():
        if require_count(roundtrip, key) != expected:
            raise RuntimeError(
                "B200 sealed validation input tokenizer round-trip is not exact"
            )
    return value


def _with_split(row: dict[str, Any], split: str) -> dict[str, Any]:
    return {**row, "_contract_split": split}


def select_contract_rows(
    train_path: Path,
    validation_path: Path,
) -> list[dict[str, Any]]:
    representatives: dict[str, dict[str, Any]] = {}
    pinned: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(train_path):
        row_id = str(row.get("id") or "")
        task = str(row.get("task") or "")
        if row_id in TRAINING_RESULT_LABEL_IDS:
            if row_id in pinned:
                raise RuntimeError(f"duplicate pinned training row: {row_id}")
            pinned[row_id] = row
        if (
            task in {"translation", "summarization"}
            and task not in representatives
            and row_id not in TRAINING_RESULT_LABEL_IDS
            and RESULT_LABEL_PREFIX_RE.match(str(row.get("response") or "")) is None
        ):
            representatives[task] = row
        if set(representatives) == {"translation", "summarization"} and set(
            pinned
        ) == set(TRAINING_RESULT_LABEL_IDS):
            break
    if set(representatives) != {"translation", "summarization"}:
        raise RuntimeError("train data lacks clean translation/summarization rows")
    if set(pinned) != set(TRAINING_RESULT_LABEL_IDS):
        raise RuntimeError(
            "pinned answer-prefix rows changed: "
            f"{sorted(pinned)} != {sorted(TRAINING_RESULT_LABEL_IDS)}"
        )
    validation_representatives: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(validation_path):
        task = str(row.get("task") or "")
        if (
            task in {"translation", "summarization"}
            and task not in validation_representatives
        ):
            if RESULT_LABEL_PREFIX_RE.match(str(row.get("response") or "")):
                raise RuntimeError(
                    f"validation target unexpectedly starts with a label: "
                    f"{row.get('id')}"
                )
            validation_representatives[task] = row
        if set(validation_representatives) == {"translation", "summarization"}:
            break
    if set(validation_representatives) != {"translation", "summarization"}:
        raise RuntimeError("validation data lacks translation/summarization rows")
    return [
        _with_split(representatives["translation"], "train"),
        _with_split(representatives["summarization"], "train"),
        *(
            _with_split(pinned[row_id], "train")
            for row_id in sorted(TRAINING_RESULT_LABEL_IDS)
        ),
        _with_split(validation_representatives["translation"], "validation"),
        _with_split(validation_representatives["summarization"], "validation"),
    ]


def _validate_row(
    tokenizer: RecordingTokenizer,
    reference: Any,
    row: dict[str, Any],
    *,
    hard_term_weight: float,
) -> dict[str, Any]:
    row_id = str(row["id"])
    task = str(row["task"])
    if task not in {"translation", "summarization"}:
        raise RuntimeError(f"unsupported contract task: {task}")
    if row.get("prompt_variant") != "no_dictionary_context":
        raise RuntimeError(f"non-canonical prompt variant: {row_id}")
    body, prefix_removed = _answer_body(row)
    messages = reference.build_messages_no_dict(task, str(row["origin"]))
    if messages[-1].get("role") != "user":
        raise RuntimeError(f"reference prompt does not end at a user turn: {row_id}")
    body_instruction = (
        "Return only the translated Korean text."
        if task == "translation"
        else "Return only the summarized Korean text."
    )
    final_instruction = str(messages[-1].get("content") or "")
    if (
        body_instruction not in final_instruction
        or "Do not add labels" not in final_instruction
    ):
        raise RuntimeError(f"reference body-only instruction changed: {row_id}")

    tokenizer.clear_calls()
    item = encode_row(
        tokenizer,
        reference,
        row,
        hard_term_weight=hard_term_weight,
    )
    if len(tokenizer.template_calls) != 2:
        raise RuntimeError(f"training rendered an unexpected prompt count: {row_id}")
    prompt_text = _expect_template_call(
        tokenizer.template_calls[0],
        messages=messages,
        add_generation_prompt=True,
        label="training prompt",
    )
    _expect_template_call(
        tokenizer.template_calls[1],
        messages=[*messages, {"role": "assistant", "content": body}],
        add_generation_prompt=False,
        label="training full sequence",
    )

    evidence = item["output_only_supervision"]
    prompt_count = int(evidence["prompt_token_count"])
    answer_count = int(evidence["answer_token_count"])
    termination_count = int(evidence["assistant_termination_token_count"])
    template_suffix_count = int(evidence["template_suffix_token_count"])
    if prompt_count <= 0 or answer_count <= 0 or termination_count != 1:
        raise RuntimeError(
            f"prompt/body must be nonempty and supervision must end at "
            f"exactly one EOS: {row_id}"
        )
    if template_suffix_count <= 0:
        raise RuntimeError(f"assistant template suffix is empty: {row_id}")
    if any(value != -100 for value in item["labels"][:prompt_count]):
        raise RuntimeError(f"prompt labels entered the loss: {row_id}")
    if any(value != 0.0 for value in item["weights"][:prompt_count]):
        raise RuntimeError(f"prompt weights entered the loss: {row_id}")
    answer_ids = tokenizer(body, add_special_tokens=False)["input_ids"]
    supervised_prefix = item["input_ids"][prompt_count : prompt_count + answer_count]
    if supervised_prefix != answer_ids:
        raise RuntimeError(f"supervised prefix is not the raw answer body: {row_id}")
    if item["labels"][prompt_count : prompt_count + answer_count] != answer_ids:
        raise RuntimeError(f"answer labels do not equal raw body tokens: {row_id}")
    termination_ids = tokenizer(
        SOLAR_ASSISTANT_TEMPLATE_TERMINATION,
        add_special_tokens=False,
    )["input_ids"]
    if not termination_ids:
        raise RuntimeError(f"tokenizer has no assistant termination ID: {row_id}")
    termination_token_id = int(termination_ids[0])
    expected_supervised = answer_ids + [termination_token_id]
    if item["labels"][prompt_count:] != expected_supervised:
        raise RuntimeError(f"supervised labels are not body plus one EOS: {row_id}")
    if item["input_ids"][prompt_count:] != expected_supervised:
        raise RuntimeError(f"supervised input is not body plus one EOS: {row_id}")
    if item["weights"][-1] != 1.0:
        raise RuntimeError(f"assistant EOS loss weight changed: {row_id}")
    expected_evidence = {
        "assistant_termination_token_id": termination_token_id,
        "template_suffix_starts_with_eos": True,
        "supervised_sequence_ends_at_eos": True,
        "prompt_labels_all_masked": True,
        "prompt_weights_all_zero": True,
        "supervised_prefix_is_answer_body": True,
        "reasoning_effort": "none",
        "answer_label_prefix_removed": prefix_removed,
    }
    for key, expected in expected_evidence.items():
        if evidence.get(key) != expected:
            raise RuntimeError(
                f"output-only evidence changed for {row_id}: "
                f"{key}={evidence.get(key)!r}"
            )

    tokenizer.clear_calls()
    inference_row = {
        "id": row_id,
        "task": task,
        "origin": str(row["origin"]),
        "messages": messages,
    }
    inference_prompt_ids = render_prompt_ids(
        tokenizer,
        reference,
        inference_row,
        max_prompt_tokens=1_000_000,
    )
    if len(tokenizer.template_calls) != 1:
        raise RuntimeError(f"inference rendered an unexpected prompt count: {row_id}")
    inference_prompt_text = _expect_template_call(
        tokenizer.template_calls[0],
        messages=messages,
        add_generation_prompt=True,
        label="inference prompt",
    )
    if inference_prompt_text != prompt_text:
        raise RuntimeError(f"train/inference rendered prompts differ: {row_id}")
    if inference_prompt_ids != item["input_ids"][:prompt_count]:
        raise RuntimeError(f"train/inference prompt token IDs differ: {row_id}")
    if not inference_prompt_text.endswith(SOLAR_OUTPUT_ONLY_GENERATION_PREAMBLE):
        raise RuntimeError(
            f"inference generation prompt requires a body label: {row_id}"
        )

    return {
        "id": row_id,
        "task": task,
        "coverage_role": (
            "pinned_answer_prefix"
            if row_id in TRAINING_RESULT_LABEL_IDS
            else f"{row['_contract_split']}_{task}_representative"
        ),
        "split": str(row["_contract_split"]),
        "body_sha256": canonical_sha256(body),
        "prompt_ids_sha256": canonical_sha256(inference_prompt_ids),
        "prompt_token_count": prompt_count,
        "answer_token_count": answer_count,
        "assistant_termination_token_count": termination_count,
        "assistant_termination_token_id": termination_token_id,
        "template_suffix_token_count": template_suffix_count,
        "supervised_labels_are_body_plus_single_eos": True,
        "answer_label_prefix_removed": prefix_removed,
        "forbidden_body_framing": [],
        "train_inference_prompt_identical": True,
        "inference_requires_body_label": False,
    }


def validate_tokenizer_output_contract(
    tokenizer: Any,
    reference: Any,
    rows: list[dict[str, Any]],
    *,
    hard_term_weight: float,
) -> dict[str, Any]:
    if TEMPLATE_KWARGS != EXPECTED_TEMPLATE_KWARGS:
        raise RuntimeError(
            f"inference template kwargs changed: "
            f"{TEMPLATE_KWARGS!r} != {EXPECTED_TEMPLATE_KWARGS!r}"
        )
    row_ids = [str(row.get("id") or "") for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise RuntimeError("contract rows contain duplicate IDs")
    pinned = set(row_ids) & set(TRAINING_RESULT_LABEL_IDS)
    if pinned != set(TRAINING_RESULT_LABEL_IDS):
        raise RuntimeError("contract rows do not contain all pinned prefix examples")
    ordinary_tasks_by_split: dict[str, set[str]] = {
        "train": set(),
        "validation": set(),
    }
    for row in rows:
        if str(row["id"]) in TRAINING_RESULT_LABEL_IDS:
            continue
        split = str(row.get("_contract_split") or "")
        if split not in ordinary_tasks_by_split:
            raise RuntimeError(f"invalid contract split: {split!r}")
        ordinary_tasks_by_split[split].add(str(row["task"]))
    for split, tasks in ordinary_tasks_by_split.items():
        if tasks != {"translation", "summarization"}:
            raise RuntimeError(
                f"{split} contract rows require one row for each task: "
                f"{sorted(tasks)}"
            )
    recording = RecordingTokenizer(tokenizer)
    row_evidence = [
        _validate_row(
            recording,
            reference,
            row,
            hard_term_weight=hard_term_weight,
        )
        for row in rows
    ]
    counts = Counter(item["task"] for item in row_evidence)
    return {
        "status": "PASS",
        "template_kwargs": dict(EXPECTED_TEMPLATE_KWARGS),
        "training_add_generation_prompt": True,
        "training_full_sequence_add_generation_prompt": False,
        "inference_add_generation_prompt": True,
        "inference_requires_response_field": False,
        "inference_requires_body_label": False,
        "rows": row_evidence,
        "coverage": {
            "rows": len(row_evidence),
            "by_task": dict(counts),
            "pinned_answer_prefix_ids": sorted(pinned),
            "ordinary_task_representatives": {
                split: sorted(tasks) for split, tasks in ordinary_tasks_by_split.items()
            },
        },
    }


def scan_evaluation_prompt_budget(
    tokenizer: Any,
    reference: Any,
    evaluation_path: Path,
    *,
    expected_rows: int,
    expected_by_task: dict[str, int],
    prompt_budget_tokens: dict[str, int] = PROMPT_BUDGET_TOKENS,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    lengths: dict[str, list[int]] = {
        "translation": [],
        "summarization": [],
    }
    over_budget: list[dict[str, Any]] = []
    maximum: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(evaluation_path):
        row_id = str(row.get("id") or "")
        task = str(row.get("task") or "")
        if task not in prompt_budget_tokens:
            raise RuntimeError(f"combined eval contains an invalid task: {task!r}")
        messages = reference.build_messages_no_dict(
            task,
            reference.clean_text(row.get("origin")),
        )
        if row.get("messages") != messages:
            raise RuntimeError(
                f"combined eval prompt is not the exact reference prompt: "
                f"{row_id}/{task}"
            )
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **EXPECTED_TEMPLATE_KWARGS,
        )
        if not isinstance(prompt, str) or not prompt.endswith(
            SOLAR_OUTPUT_ONLY_GENERATION_PREAMBLE
        ):
            raise RuntimeError(
                f"combined eval prompt lacks exact output-only preamble: "
                f"{row_id}/{task}"
            )
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
        )["input_ids"]
        length = len(prompt_ids)
        if length <= 0:
            raise RuntimeError(f"combined eval rendered an empty prompt: {row_id}")
        counts[task] += 1
        lengths[task].append(length)
        if task not in maximum or length > maximum[task]["tokens"]:
            maximum[task] = {"id": row_id, "tokens": length}
        if length > int(prompt_budget_tokens[task]):
            over_budget.append(
                {
                    "id": row_id,
                    "task": task,
                    "tokens": length,
                    "budget": int(prompt_budget_tokens[task]),
                }
            )
    if sum(counts.values()) != expected_rows:
        raise RuntimeError(
            f"combined eval scan row count changed: "
            f"{sum(counts.values())} != {expected_rows}"
        )
    if dict(counts) != expected_by_task:
        raise RuntimeError(
            f"combined eval scan task counts changed: "
            f"{dict(counts)} != {expected_by_task}"
        )
    if over_budget:
        raise RuntimeError(
            f"combined eval requires prompt truncation: count={len(over_budget)} "
            f"examples={over_budget[:8]}"
        )
    return {
        "status": "PASS",
        "path": str(evaluation_path),
        "rows": expected_rows,
        "by_task": dict(counts),
        "prompt_budget_tokens": dict(prompt_budget_tokens),
        "truncation_count": 0,
        "exact_output_only_preamble_count": expected_rows,
        "prompt_tokens": {
            task: {
                "minimum": min(values),
                "maximum": max(values),
                "maximum_row": maximum[task],
            }
            for task, values in lengths.items()
        },
    }


def validate_source_dir(source_dir: Path) -> dict[str, Any]:
    if source_dir != EXPECTED_SOURCE_DIR.resolve():
        raise RuntimeError(
            f"source-dir is not the pinned snapshot: "
            f"{source_dir} != {EXPECTED_SOURCE_DIR.resolve()}"
        )
    config_path = source_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = read_json(config_path)
    if config.get("model_type") != "solar_open2":
        raise RuntimeError("pinned source model_type changed")
    if "SolarOpen2ForCausalLM" not in (config.get("architectures") or []):
        raise RuntimeError("pinned source architecture changed")
    hashes = {
        name: sha256(source_dir / name)
        for name in SOURCE_TOKENIZER_FILES
        if (source_dir / name).is_file()
    }
    if "tokenizer_config.json" not in hashes:
        raise FileNotFoundError(source_dir / "tokenizer_config.json")
    if not {"tokenizer.json", "tokenizer.model"} & set(hashes):
        raise FileNotFoundError("pinned source tokenizer vocabulary is incomplete")
    if "chat_template.jinja" not in hashes:
        raise FileNotFoundError(source_dir / "chat_template.jinja")
    generation = read_json(source_dir / "generation_config.json")
    generation_eos = generation.get("eos_token_id")
    if isinstance(generation_eos, int) and not isinstance(generation_eos, bool):
        generation_eos_ids = [generation_eos]
    elif (
        isinstance(generation_eos, list)
        and generation_eos
        and all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in generation_eos
        )
    ):
        generation_eos_ids = [int(value) for value in generation_eos]
    else:
        raise RuntimeError("pinned generation EOS token IDs are invalid")
    return {
        "model_id": BASE_MODEL_ID,
        "revision": BASE_REVISION,
        "path": str(source_dir),
        "config_sha256": sha256(config_path),
        "generation_eos_token_ids": generation_eos_ids,
        "tokenizer_files_sha256": hashes,
    }


def validate_data_dir(
    data_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contract_path = CONFIG_DIR / "data_contract.json"
    contract = read_json(contract_path)
    expected_dir = Path(contract["prepared_data_dir"]).resolve()
    if data_dir != expected_dir:
        raise RuntimeError(f"data-dir is not pinned: {data_dir} != {expected_dir}")
    if contract.get("reference_evaluator_sha256") != REFERENCE_SHA256:
        raise RuntimeError("data contract reference evaluator hash changed")
    manifest_path = Path(contract["prepared_manifest"]).resolve()
    if manifest_path != data_dir / "manifest.json":
        raise RuntimeError("prepared manifest is outside the pinned data directory")
    manifest_sha256 = sha256(manifest_path)
    if manifest_sha256 != contract["prepared_manifest_sha256"]:
        raise RuntimeError("prepared data manifest hash changed")
    files: dict[str, Any] = {}
    for role, entry in contract["files"].items():
        path = data_dir / entry["name"]
        actual = sha256(path)
        if actual != entry["sha256"]:
            raise RuntimeError(f"pinned data hash changed: {role}={actual}")
        files[role] = {
            "path": str(path),
            "sha256": actual,
            "rows": entry["rows"],
            "by_task": entry["by_task"],
        }
    return select_contract_rows(
        Path(files["train"]["path"]),
        Path(files["validation"]["path"]),
    ), {
        "path": str(data_dir),
        "data_contract_sha256": sha256(contract_path),
        "prepared_manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha256,
        },
        "files": files,
    }


def _b200_source_file(source_dir: Path, name: str) -> Path:
    path = source_dir / name
    if not path.exists():
        raise FileNotFoundError(path)
    resolved = path.resolve(strict=True)
    b200_runtime.require_owned_path(resolved, must_exist=True)
    if not resolved.is_file():
        raise RuntimeError(f"B200 source asset is not a regular file: {resolved}")
    return resolved


def validate_b200_source_dir(
    source_dir: Path,
    bundle: dict[str, Any],
) -> dict[str, Any]:
    source = b200_runtime.require_owned_path(source_dir, must_exist=True)
    if source != b200_runtime.BF16_SOURCE_DIR:
        raise RuntimeError(
            f"B200 source-dir is not pinned: "
            f"{source} != {b200_runtime.BF16_SOURCE_DIR}"
        )
    configured = Path(bundle["model"]["training_runtime"]["source_dir"])
    if configured != source:
        raise RuntimeError("B200 model config/source CLI paths differ")
    config_path = _b200_source_file(source, "config.json")
    config = b200_runtime.read_json_no_duplicates(config_path)
    if config.get("model_type") != "solar_open2":
        raise RuntimeError("B200 source model_type changed")
    if "SolarOpen2ForCausalLM" not in (config.get("architectures") or []):
        raise RuntimeError("B200 source architecture changed")
    hashes = {
        name: b200_runtime.sha256_file(_b200_source_file(source, name))
        for name in SOURCE_TOKENIZER_FILES
        if (source / name).is_file()
    }
    if "tokenizer_config.json" not in hashes:
        raise FileNotFoundError(source / "tokenizer_config.json")
    if not {"tokenizer.json", "tokenizer.model"} & set(hashes):
        raise FileNotFoundError("B200 source tokenizer vocabulary is incomplete")
    if "chat_template.jinja" not in hashes:
        raise FileNotFoundError(source / "chat_template.jinja")
    generation_path = _b200_source_file(source, "generation_config.json")
    generation = b200_runtime.read_json_no_duplicates(generation_path)
    generation_eos = generation.get("eos_token_id")
    if isinstance(generation_eos, int) and not isinstance(generation_eos, bool):
        generation_eos_ids = [generation_eos]
    elif (
        isinstance(generation_eos, list)
        and generation_eos
        and all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in generation_eos
        )
    ):
        generation_eos_ids = [int(value) for value in generation_eos]
    else:
        raise RuntimeError("B200 generation EOS token IDs are invalid")
    return {
        "model_id": BASE_MODEL_ID,
        "revision": BASE_REVISION,
        "path": str(source),
        "config_sha256": b200_runtime.sha256_file(config_path),
        "generation_eos_token_ids": generation_eos_ids,
        "tokenizer_files_sha256": hashes,
    }


def validate_b200_data_dir(
    data_dir: Path,
    bundle: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contract = bundle["data_contract"]
    expected_dir = Path(str(contract["prepared_data_dir"]))
    resolved = b200_runtime.require_owned_path(data_dir, must_exist=True)
    if resolved != expected_dir:
        raise RuntimeError(f"B200 data-dir is not pinned: {resolved}")
    manifest_path = Path(str(contract["prepared_manifest"]))
    manifest_sha256 = b200_runtime.sha256_file(manifest_path)
    if manifest_sha256 != contract["prepared_manifest_sha256"]:
        raise RuntimeError("B200 prepared data manifest hash changed")
    if contract.get("selection_datasets") != ["validation"]:
        raise RuntimeError("B200 selection dataset contract changed")
    if contract.get("test_and_human_gold_monitoring_allowed") is not False:
        raise RuntimeError("sealed evaluation entered B200 monitoring")
    files: dict[str, Any] = {}
    for role in ("train", "validation"):
        entry = contract["files"][role]
        path = Path(str(entry["path"]))
        expected_path = resolved / str(entry["name"])
        if path != expected_path:
            raise RuntimeError(f"B200 {role} path changed: {path}")
        actual = b200_runtime.sha256_file(path)
        if actual != entry["sha256"]:
            raise RuntimeError(f"B200 pinned data hash changed: {role}={actual}")
        files[role] = {
            "path": str(path),
            "sha256": actual,
            "rows": entry["rows"],
            "by_task": entry["by_task"],
        }
    rows = select_contract_rows(
        Path(files["train"]["path"]),
        Path(files["validation"]["path"]),
    )
    return rows, {
        "path": str(resolved),
        "data_contract_sha256": bundle["hashes"]["data_contract"],
        "prepared_manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha256,
        },
        "files": files,
        "opened_dataset_roles": ["train", "validation"],
        "sealed_evaluation_dataset_roles_opened": [],
    }


def scan_b200_validation_prompt_budget(
    tokenizer: Any,
    reference: Any,
    validation_path: Path,
    *,
    expected_rows: int,
    expected_by_task: dict[str, int],
    prompt_budget_tokens: dict[str, int] = PROMPT_BUDGET_TOKENS,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    seen_row_keys: set[tuple[str, str]] = set()
    lengths: dict[str, list[int]] = {
        "translation": [],
        "summarization": [],
    }
    target_lengths: dict[str, list[int]] = {
        "translation": [],
        "summarization": [],
    }
    maximum: dict[str, dict[str, Any]] = {}
    over_budget: list[dict[str, Any]] = []
    source_unicode_occurrences: Counter[str] = Counter()
    source_unicode_rows: Counter[str] = Counter()
    anomalous_source_rows: list[dict[str, Any]] = []
    tokenizer_roundtrip_failures: list[dict[str, str]] = []
    for row in iter_jsonl(validation_path):
        row_id = str(row.get("id") or "")
        task = str(row.get("task") or "")
        if task not in prompt_budget_tokens:
            raise RuntimeError(f"validation contains an invalid task: {task!r}")
        row_key = (row_id, task)
        if not row_id or row_key in seen_row_keys:
            raise RuntimeError(f"validation has missing/duplicate ID+task: {row_key!r}")
        seen_row_keys.add(row_key)
        if row.get("prompt_variant") != "no_dictionary_context":
            raise RuntimeError(f"validation prompt variant changed: {row_id}")
        origin = str(row.get("origin") or "")
        source_inventory = _source_unicode_inventory(
            origin,
            label=f"validation origin {row_id}",
        )
        anomalous_counts = {
            key: int(value) for key, value in source_inventory.items() if int(value) > 0
        }
        for key, value in source_inventory.items():
            source_unicode_occurrences[key] += int(value)
            if int(value) > 0:
                source_unicode_rows[key] += 1
        if anomalous_counts:
            anomalous_source_rows.append(
                {
                    "id": row_id,
                    "task": task,
                    **anomalous_counts,
                }
            )
        origin_ids = tokenizer(origin, add_special_tokens=False)["input_ids"]
        decoded_origin = tokenizer.decode(
            origin_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if decoded_origin != origin:
            tokenizer_roundtrip_failures.append(
                {
                    "id": row_id,
                    "task": task,
                }
            )
        body, prefix_removed = _answer_body(row)
        if prefix_removed:
            raise RuntimeError(f"validation target required prefix removal: {row_id}")
        messages = reference.build_messages_no_dict(
            task,
            reference.clean_text(origin),
        )
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **EXPECTED_TEMPLATE_KWARGS,
        )
        if not isinstance(prompt, str) or not prompt.endswith(
            SOLAR_OUTPUT_ONLY_GENERATION_PREAMBLE
        ):
            raise RuntimeError(
                f"validation prompt lacks exact output-only preamble: "
                f"{row_id}/{task}"
            )
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        body_ids = tokenizer(body, add_special_tokens=False)["input_ids"]
        prompt_length = len(prompt_ids)
        if prompt_length <= 0 or not body_ids:
            raise RuntimeError(
                f"validation prompt/body tokenization is empty: {row_id}"
            )
        counts[task] += 1
        lengths[task].append(prompt_length)
        target_lengths[task].append(len(body_ids))
        if task not in maximum or prompt_length > maximum[task]["tokens"]:
            maximum[task] = {"id": row_id, "tokens": prompt_length}
        if prompt_length > int(prompt_budget_tokens[task]):
            over_budget.append(
                {
                    "id": row_id,
                    "task": task,
                    "tokens": prompt_length,
                    "budget": int(prompt_budget_tokens[task]),
                }
            )
    if sum(counts.values()) != expected_rows:
        raise RuntimeError(
            f"validation scan row count changed: "
            f"{sum(counts.values())} != {expected_rows}"
        )
    if dict(counts) != expected_by_task:
        raise RuntimeError(
            f"validation scan task counts changed: "
            f"{dict(counts)} != {expected_by_task}"
        )
    if over_budget:
        raise RuntimeError(
            f"validation requires prompt truncation: count={len(over_budget)} "
            f"examples={over_budget[:8]}"
        )
    if tokenizer_roundtrip_failures:
        raise RuntimeError(
            "validation source tokenizer round-trip changed sealed text: "
            f"count={len(tokenizer_roundtrip_failures)} "
            f"examples={tokenizer_roundtrip_failures[:8]}"
        )
    anomalous_source_rows_sha256 = canonical_sha256(anomalous_source_rows)
    return {
        "status": "PASS",
        "path": str(validation_path),
        "rows": expected_rows,
        "by_task": dict(counts),
        "prompt_budget_tokens": dict(prompt_budget_tokens),
        "truncation_count": 0,
        "input_unicode_inventory": {
            "scope": "sealed validation origin before reference cleaning",
            "rows_scanned": expected_rows,
            "utf8_encoding_failures": 0,
            "surrogate_codepoint_occurrences": 0,
            "replacement_character_rows": source_unicode_rows[
                "replacement_character_occurrences"
            ],
            "replacement_character_occurrences": source_unicode_occurrences[
                "replacement_character_occurrences"
            ],
            "c0_non_whitespace_control_rows": source_unicode_rows[
                "c0_non_whitespace_control_occurrences"
            ],
            "c0_non_whitespace_control_occurrences": source_unicode_occurrences[
                "c0_non_whitespace_control_occurrences"
            ],
            "c1_control_rows": source_unicode_rows["c1_control_occurrences"],
            "c1_control_occurrences": source_unicode_occurrences[
                "c1_control_occurrences"
            ],
            "anomalous_rows": len(anomalous_source_rows),
            "anomalous_row_records": anomalous_source_rows,
            "anomalous_row_records_sha256": anomalous_source_rows_sha256,
            "tokenizer_roundtrip": {
                "rows_checked": expected_rows,
                "exact_roundtrip_rows": expected_rows,
                "failures": 0,
            },
        },
        "invalid_unicode_count": 0,
        "target_invalid_unicode_count": 0,
        "forbidden_output_framing_count": 0,
        "target_prefix_removal_count": 0,
        "exact_output_only_preamble_count": expected_rows,
        "prompt_tokens": {
            task: {
                "minimum": min(values),
                "maximum": max(values),
                "maximum_row": maximum[task],
            }
            for task, values in lengths.items()
        },
        "target_body_tokens": {
            task: {
                "minimum": min(values),
                "maximum": max(values),
            }
            for task, values in target_lengths.items()
        },
    }


def run(
    *,
    source_dir: Path,
    data_dir: Path,
) -> dict[str, Any]:
    source = validate_source_dir(source_dir)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        source_dir,
        use_fast=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    if not isinstance(getattr(tokenizer, "chat_template", None), str):
        raise RuntimeError("Solar tokenizer does not expose a chat template")
    rows, data = validate_data_dir(data_dir)
    reference = load_reference_module()
    recipe = read_json(CONFIG_DIR / "recipe.json")
    output_contract = validate_tokenizer_output_contract(
        tokenizer,
        reference,
        rows,
        hard_term_weight=float(recipe["hard_term_target_reading_loss_weight"]),
    )
    supervised_termination_ids = {
        int(row["assistant_termination_token_id"]) for row in output_contract["rows"]
    }
    generation_eos_ids = {int(value) for value in source["generation_eos_token_ids"]}
    if (
        len(supervised_termination_ids) != 1
        or not supervised_termination_ids <= generation_eos_ids
    ):
        raise RuntimeError(
            "supervised assistant termination is not a configured generation "
            f"EOS: supervised={sorted(supervised_termination_ids)} "
            f"configured={sorted(generation_eos_ids)}"
        )
    output_contract["supervised_termination_token_ids"] = sorted(
        supervised_termination_ids
    )
    output_contract["configured_generation_eos_token_ids"] = sorted(generation_eos_ids)
    combined_prompt_budget = scan_evaluation_prompt_budget(
        tokenizer,
        reference,
        Path(data["files"]["combined_eval"]["path"]),
        expected_rows=PINNED_COMBINED_EVAL_ROWS,
        expected_by_task=PINNED_COMBINED_EVAL_TASK_COUNTS,
    )
    return {
        "status": "PASS",
        "user": getpass.getuser(),
        "project": str(PROJECT),
        "source": source,
        "data": data,
        "reference": {
            "path": str(PROJECT / "scripts/reference/shared_nodict_evaluation.py"),
            "sha256": REFERENCE_SHA256,
        },
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
            "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        },
        "output_only_contract": output_contract,
        "combined_eval_prompt_budget": combined_prompt_budget,
        "pipeline_code": pipeline_code_identity(),
        "checks": {
            "owned_pinned_source_and_data": "PASS",
            "pinned_data_hashes": "PASS",
            "exact_reference_prompt": "PASS",
            "reasoning_effort_none": "PASS",
            "prompt_loss_masking": "PASS",
            "body_prefix_and_termination": "PASS",
            "termination_is_configured_generation_eos": "PASS",
            "forbidden_framing_absent": "PASS",
            "train_inference_prompt_identity": "PASS",
            "inference_body_label_not_required": "PASS",
            "combined_eval_prompt_budget_no_truncation": "PASS",
        },
    }


def _bind_supervised_eos(
    output_contract: dict[str, Any],
    source: dict[str, Any],
) -> None:
    supervised_termination_ids = {
        int(row["assistant_termination_token_id"]) for row in output_contract["rows"]
    }
    generation_eos_ids = {int(value) for value in source["generation_eos_token_ids"]}
    if (
        len(supervised_termination_ids) != 1
        or not supervised_termination_ids <= generation_eos_ids
    ):
        raise RuntimeError(
            "supervised assistant termination is not a configured generation "
            f"EOS: supervised={sorted(supervised_termination_ids)} "
            f"configured={sorted(generation_eos_ids)}"
        )
    output_contract["supervised_termination_token_ids"] = sorted(
        supervised_termination_ids
    )
    output_contract["configured_generation_eos_token_ids"] = sorted(generation_eos_ids)


def run_b200(
    *,
    config_dir: Path,
    source_dir: Path,
    data_dir: Path,
    tokenizer: Any | None = None,
    source_validator: Any = None,
    data_validator: Any = None,
    validation_scanner: Any = None,
) -> dict[str, Any]:
    b200_runtime.require_expected_user()
    b200_runtime.require_owned_root()
    bundle = b200_runtime.load_b200_config_bundle(config_dir)
    source = (source_validator or validate_b200_source_dir)(source_dir, bundle)
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            source_dir,
            use_fast=True,
            local_files_only=True,
            trust_remote_code=False,
        )
    if not isinstance(getattr(tokenizer, "chat_template", None), str):
        raise RuntimeError("Solar tokenizer does not expose a chat template")
    rows, data = (data_validator or validate_b200_data_dir)(data_dir, bundle)
    if set(data["files"]) != {"train", "validation"}:
        raise RuntimeError("B200 tokenizer opened a sealed evaluation dataset")
    reference_path = b200_runtime.require_owned_evaluator()
    reference = load_reference_module(
        config_dir=bundle["config_dir"],
        owned_only=True,
    )
    recipe = bundle["recipe"]
    output_contract = validate_tokenizer_output_contract(
        tokenizer,
        reference,
        rows,
        hard_term_weight=float(recipe["hard_term_target_reading_loss_weight"]),
    )
    _bind_supervised_eos(output_contract, source)
    validation_entry = bundle["data_contract"]["files"]["validation"]
    evaluation = recipe["evaluation"]
    max_model_len = int(evaluation["max_model_len"])
    prompt_budget_tokens = {
        "translation": max_model_len - int(evaluation["translation_max_new_tokens"]),
        "summarization": max_model_len
        - int(evaluation["summarization_max_new_tokens"]),
    }
    if prompt_budget_tokens != PROMPT_BUDGET_TOKENS:
        raise RuntimeError(
            "B200 recipe-derived prompt budgets changed: " f"{prompt_budget_tokens!r}"
        )
    full_validation = (validation_scanner or scan_b200_validation_prompt_budget)(
        tokenizer,
        reference,
        Path(data["files"]["validation"]["path"]),
        expected_rows=int(validation_entry["rows"]),
        expected_by_task={
            str(key): int(value) for key, value in validation_entry["by_task"].items()
        },
        prompt_budget_tokens=prompt_budget_tokens,
    )
    if not isinstance(full_validation, dict):
        raise RuntimeError("B200 full validation prompt-budget result is not an object")
    expected_by_task = {
        str(key): int(value) for key, value in validation_entry["by_task"].items()
    }
    for key, expected in (
        ("status", "PASS"),
        ("path", data["files"]["validation"]["path"]),
        ("rows", int(validation_entry["rows"])),
        ("by_task", expected_by_task),
        ("prompt_budget_tokens", prompt_budget_tokens),
        ("truncation_count", 0),
        ("invalid_unicode_count", 0),
        ("target_invalid_unicode_count", 0),
        ("forbidden_output_framing_count", 0),
        ("target_prefix_removal_count", 0),
        ("exact_output_only_preamble_count", int(validation_entry["rows"])),
    ):
        if full_validation.get(key) != expected:
            raise RuntimeError(
                f"B200 full validation {key} changed: "
                f"{full_validation.get(key)!r} != {expected!r}"
            )
    validate_b200_input_unicode_inventory(
        full_validation.get("input_unicode_inventory"),
        expected_rows=int(validation_entry["rows"]),
        expected_anomalies=B200_VALIDATION_UNICODE_EXPECTED,
    )
    return {
        "schema": "solar_open2_b200_tokenizer_contract",
        "schema_version": 1,
        "status": "PASS",
        "profile": "b200",
        "user": b200_runtime.EXPECTED_USER,
        "root": str(b200_runtime.OWNED_ROOT),
        "project": str(PROJECT),
        "config": b200_runtime.b200_config_identity(config_dir),
        "source": source,
        "data": data,
        "reference": {
            "path": str(reference_path),
            "sha256": b200_runtime.REFERENCE_EVALUATOR_SHA256,
        },
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
            "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        },
        "output_only_contract": output_contract,
        "full_validation_prompt_budget": full_validation,
        "pipeline_code": pipeline_code_identity(config_dir=bundle["config_dir"]),
        "checks": {
            "owned_pinned_source": "PASS",
            "train_and_validation_only": "PASS",
            "sealed_evaluation_not_opened": "PASS",
            "validation_task_representatives": "PASS",
            "full_validation_prompt_budget_no_truncation": "PASS",
            "sealed_input_unicode_inventory_and_roundtrip": "PASS",
            "output_only_targets": "PASS",
            "target_and_output_valid_unicode": "PASS",
            "single_supervised_eos_is_generation_eos": "PASS",
            "train_inference_prompt_identity": "PASS",
        },
        "production_authorized": False,
    }


def require_fresh_b200_receipt(path: Path) -> Path:
    raw = path.expanduser()
    if raw.is_symlink():
        raise RuntimeError(f"B200 receipt must not be a symlink: {raw}")
    resolved = b200_runtime.require_owned_path(raw)
    if resolved.exists() or resolved.is_symlink():
        raise FileExistsError(f"refusing to overwrite B200 receipt: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.config_dir is not None:
        receipt = require_fresh_b200_receipt(args.receipt)
        result = run_b200(
            config_dir=args.config_dir,
            source_dir=args.source_dir,
            data_dir=args.data_dir,
        )
        b200_runtime.atomic_json(receipt, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if getpass.getuser() != "mk03":
        raise RuntimeError("Solar tokenizer certification is restricted to mk03")
    source_dir = require_owned_path(args.source_dir)
    data_dir = require_owned_path(args.data_dir)
    receipt = require_owned_path(args.receipt, create_parent=True)
    result = run(source_dir=source_dir, data_dir=data_dir)
    atomic_json(receipt, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
