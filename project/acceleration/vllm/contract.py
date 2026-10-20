#!/usr/bin/env python3
"""Fail-closed contracts shared by the isolated vLLM evaluation path."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path("/NHNHOME/WORKSPACE/26mfds002_B/jhtak/solar-open-2")
PROJECT = ROOT / "project"
VLLM_ENV = PROJECT / "solar-open2-vllm"
MODEL_REVISION = "de88f6226788077e2d340204fd79d37720c9eda0"
MODEL_DIR = ROOT / f"runtime/models/Solar-Open2-250B-Nota-NVFP4-{MODEL_REVISION}"
TOKENIZER_DIR = (
    ROOT
    / "runtime/hf_cache/models--upstage--Solar-Open2-250B/snapshots"
    / "8d1dda5ad2af58207c65348d825772d2f4d72481"
)
ADAPTER_DIR = ROOT / "runtime/runs/b200_1epoch_len9600_v1/best_adapter"
DATA_CONTRACT_PATH = PROJECT / "config/b200/data_contract.json"
MODEL_CONTRACT_PATH = PROJECT / "config/b200/model.json"
RECIPE_PATH = PROJECT / "config/b200/recipe.json"
EVALUATION_PATH = PROJECT / "scripts/evaluation.py"
COMMON_PATH = PROJECT / "scripts/common.py"
ASSET_RECEIPT = ROOT / "runtime/receipts/b200_vllm_nvfp4_asset_v1.json"
ENVIRONMENT_RECEIPT = (
    ROOT / "runtime/receipts/b200_vllm_acceleration_environment_v1.json"
)

EVALUATION_SHA256 = "540e9746f34138cdae4eb766a61c0030aadab9bf0e39d96186d5fb4d6140ab24"
COMMON_SHA256 = "54716b81289dc8a3d2c025793822d34ca5d489c1d319634b83c99048a836e375"
PINNED_DATA_CONTRACT_SHA256 = (
    "ca2a73f35051d9d7530ff9c8c5761d14ef4cb2944a32650880680f83a483cfe4"
)
MODEL_INDEX_SHA256 = "ada21368497b36a82031d0674f99475f73cb025c7d2f3e5b7a31e5c08755581f"
ADAPTER_WEIGHTS_SHA256 = (
    "34de50b7dee9e8baa98d48c8d84be11c984f6d922ae871c1d9c320f266a83846"
)

SCHEMA = "solar-open2-b200-vllm-evaluation"
SCHEMA_VERSION = 7
TASK_ORDER = ("translation", "summarization")
TASK_MAX_NEW_TOKENS = {"translation": 8_192, "summarization": 6_144}
MAX_MODEL_LEN = 16_384
EOS_TOKEN_IDS = {2, 129}
TERMINAL_LIMIT_SIGNALS = [
    "generation_did_not_stop_at_eos",
    "generation_limit_exhausted",
    "invalid_eos_position",
]
TERMINAL_LIMIT_POLICY = "preserve_raw_without_repair"
OUTPUT_BODY_PREFIX_SIGNALS = ("role_header", "result_label", "meta_framing")
DOCUMENT_INTERNAL_LABEL_SIGNALS = (
    "result_label",
    "reasoning_label",
    "explanation_framing",
)
QUALITY_FAILURE_SIGNALS = ("non_korean_body",)
OUTPUT_BODY_POLICY = "deterministic_leading_wrapper_extraction_v4"
ENCLOSING_MARKDOWN_FENCE_RE = re.compile(
    r"\A\s*(?P<fence>```|~~~)(?:text|txt|korean|한국어)?[ \t]*\r?\n"
    r"(?P<body>[\s\S]*?)\r?\n(?P=fence)[ \t]*\s*\Z",
    re.IGNORECASE,
)
TEMPLATE_KWARGS = {
    "reasoning_effort": "none",
    "think_render_option": "preserved",
    "provider_system_prompt": False,
}
OUTPUT_PREAMBLE = "<|im:start|>assistant<|im:content|><|think:start|><|think:end|>"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise RuntimeError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
    )
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, object_pairs_hook=reject_duplicates)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"corrupt JSONL: {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise RuntimeError(f"JSONL row is not an object: {path}:{line_number}")
            rows.append(row)
    return rows


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_module(name: str, path: Path, expected_sha256: str) -> ModuleType:
    if path.is_symlink() or not path.is_file() or sha256(path) != expected_sha256:
        raise RuntimeError(f"pinned module identity changed: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import pinned module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def load_reference_modules() -> tuple[ModuleType, ModuleType]:
    evaluation = load_module(
        "b200_vllm_reference_evaluation", EVALUATION_PATH, EVALUATION_SHA256
    )
    common = load_module("b200_vllm_reference_common", COMMON_PATH, COMMON_SHA256)
    if common.SOLAR_OUTPUT_ONLY_GENERATION_PREAMBLE != OUTPUT_PREAMBLE:
        raise RuntimeError("Solar output-only generation preamble changed")
    return evaluation, common


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("id") or ""), str(row.get("task") or "")


def dataset_entry(path: Path, *, mode: str) -> tuple[str, dict[str, Any]]:
    if sha256(DATA_CONTRACT_PATH) != PINNED_DATA_CONTRACT_SHA256:
        raise RuntimeError("B200 data contract changed")
    contract = read_json(DATA_CONTRACT_PATH)
    roles = (
        ("validation",)
        if mode == "smoke"
        else ("human_gold", "full_test", "combined_eval")
    )
    resolved = path.resolve(strict=True)
    matches = [
        (role, contract["files"][role])
        for role in roles
        if Path(contract["files"][role]["path"]).resolve(strict=True) == resolved
    ]
    if len(matches) != 1:
        raise RuntimeError(f"{mode} accepts only pinned evaluation paths: {path}")
    return matches[0]


def prepare_rows(
    path: Path,
    *,
    mode: str,
    shard_index: int,
    shard_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    if mode not in {"smoke", "production"}:
        raise ValueError(f"unsupported mode: {mode}")
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard index/count")
    role, entry = dataset_entry(path, mode=mode)
    rows = read_jsonl(path)
    actual_sha = sha256(path)
    if actual_sha != entry["sha256"] or len(rows) != entry["rows"]:
        raise RuntimeError(f"pinned {role} dataset identity changed")
    counts = Counter(str(row.get("task")) for row in rows)
    if dict(counts) != entry["by_task"]:
        raise RuntimeError(f"pinned {role} task counts changed: {dict(counts)}")
    evaluation, _ = load_reference_modules()
    if mode == "smoke":
        selected, seen = [], set()
        for source_row in rows:
            row = dict(source_row)
            task = str(row["task"])
            if task not in seen:
                row["messages"] = evaluation.build_messages_no_dict(
                    task, evaluation.clean_text(row["origin"])
                )
                row["prompt_variant"] = "no_dictionary_context"
                selected.append(row)
                seen.add(task)
            if seen == set(TASK_ORDER):
                break
        rows = selected
        if [row["task"] for row in rows] != list(TASK_ORDER):
            raise RuntimeError("smoke selection must contain one row per task")
        evaluation.validate_eval_rows(rows, "vLLM validation smoke canonical prompts")
    else:
        evaluation.validate_eval_rows(rows, str(path))

    full_rows = rows
    if shard_count > 1:
        selected_keys: set[tuple[str, str]] = set()
        for task in TASK_ORDER:
            task_rows = [row for row in full_rows if row["task"] == task]
            selected_keys.update(
                row_key(row)
                for ordinal, row in enumerate(task_rows)
                if ordinal % shard_count == shard_index
            )
        rows = [row for row in full_rows if row_key(row) in selected_keys]
    if not rows:
        raise RuntimeError("shard selected no evaluation rows")

    split_by_key: dict[tuple[str, str], str] = {}
    if role == "combined_eval":
        data_contract = read_json(DATA_CONTRACT_PATH)
        for split in ("full_test", "human_gold"):
            component = Path(data_contract["files"][split]["path"])
            component_rows = read_jsonl(component)
            split_by_key.update({row_key(row): split for row in component_rows})
    else:
        split_by_key = {row_key(row): role for row in full_rows}
    identities = {
        row_key(row): {
            "split": split_by_key[row_key(row)],
            "id": row["id"],
            "task": row["task"],
            "row_sha256": canonical_sha256(row),
        }
        for row in rows
    }
    selected_keys = [list(row_key(row)) for row in rows]
    evidence = {
        "mode": mode,
        "role": role,
        "path": str(path.resolve(strict=True)),
        "sha256": actual_sha,
        "source_rows": len(full_rows),
        "selected_rows": len(rows),
        "selected_by_task": dict(Counter(row["task"] for row in rows)),
        "selected_keys_sha256": canonical_sha256(selected_keys),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "shard_policy": "stable_task_local_ordinal_modulo",
    }
    return rows, evidence, identities


def render_prompt_ids(
    tokenizer: Any, row: dict[str, Any], *, max_new_tokens: int
) -> list[int]:
    evaluation, _ = load_reference_modules()
    origin = evaluation.clean_text(row["origin"])
    messages = evaluation.build_messages_no_dict(row["task"], origin)
    if row.get("messages") != messages:
        raise RuntimeError(f"prompt reconstruction changed: {row_key(row)}")
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **TEMPLATE_KWARGS
    )
    if not prompt.endswith(OUTPUT_PREAMBLE):
        raise RuntimeError(f"output-only preamble changed: {row_key(row)}")
    ids = [
        int(value) for value in tokenizer(prompt, add_special_tokens=False)["input_ids"]
    ]
    maximum = MAX_MODEL_LEN - max_new_tokens
    if not ids or len(ids) > maximum:
        raise RuntimeError(
            f"prompt is empty or exceeds budget without truncation: {row_key(row)} "
            f"tokens={len(ids)} limit={maximum}"
        )
    return ids


def output_body_violations(text: str, common: ModuleType) -> list[str]:
    """Reject structural wrappers, not semantic model-quality failures."""

    violations = set(common.output_only_violations(text))
    for signal in DOCUMENT_INTERNAL_LABEL_SIGNALS:
        pattern = common.OUTPUT_ONLY_FORBIDDEN_PATTERNS[signal]
        if signal in violations and pattern.match(text) is None:
            violations.remove(signal)
    violations.difference_update(QUALITY_FAILURE_SIGNALS)
    return sorted(violations)


def output_quality_failures(text: str, common: ModuleType) -> list[str]:
    """Record scoreable model failures without repairing or dropping the output."""

    detected = set(common.output_only_violations(text))
    return sorted(detected.intersection(QUALITY_FAILURE_SIGNALS))


def extract_output_body(
    raw_continuation: str, common: ModuleType
) -> tuple[str, dict[str, Any]]:
    """Remove only deterministic outer wrappers; never rewrite answer content."""

    body = raw_continuation.strip()
    removed_signals: list[str] = []
    removed_fragments: list[str] = []
    fence_match = ENCLOSING_MARKDOWN_FENCE_RE.fullmatch(body)
    if fence_match is not None:
        removed_signals.append("enclosing_markdown_fence")
        removed_fragments.extend(
            [fence_match.group("fence"), fence_match.group("fence")]
        )
        body = fence_match.group("body").strip()

    for _ in range(8):
        matched = False
        for signal in OUTPUT_BODY_PREFIX_SIGNALS:
            match = common.OUTPUT_ONLY_FORBIDDEN_PATTERNS[signal].match(body)
            if match is None or match.end() <= 0:
                continue
            removed_signals.append(signal)
            fragment = body[: match.end()]
            removed_fragments.append(fragment)
            body = body[match.end() :].lstrip()
            if signal == "result_label":
                leading = fragment.lstrip()
                for marker in ("**", "__"):
                    if leading.startswith(marker) and body.startswith(marker):
                        removed_fragments.append(marker)
                        body = body[len(marker) :].lstrip()
                        break
            matched = True
            break
        if not matched:
            break

    residual = output_body_violations(body, common)
    quality_failures = output_quality_failures(body, common)
    removed_text = "".join(removed_fragments)
    audit = {
        "status": "PASS" if body and not residual else "FAIL",
        "policy": OUTPUT_BODY_POLICY,
        "applied": bool(removed_signals),
        "removed_signals": removed_signals,
        "removed_prefix_chars": len(removed_text),
        "removed_prefix_sha256": hashlib.sha256(
            removed_text.encode("utf-8")
        ).hexdigest(),
        "raw_output_violations": sorted(
            set(common.output_only_violations(raw_continuation))
        ),
        "prediction_output_violations": residual,
        "prediction_quality_failures": quality_failures,
        "preserved_internal_document_label_spans": {
            signal: [
                [match.start(), match.end()]
                for match in common.OUTPUT_ONLY_FORBIDDEN_PATTERNS[signal].finditer(
                    body
                )
                if match.start() > 0
            ]
            for signal in DOCUMENT_INTERNAL_LABEL_SIGNALS
            if any(
                match.start() > 0
                for match in common.OUTPUT_ONLY_FORBIDDEN_PATTERNS[signal].finditer(
                    body
                )
            )
        },
        "content_rewrite": False,
    }
    if not body:
        raise RuntimeError(
            "output body is empty after deterministic wrapper extraction"
        )
    if residual:
        diagnostic = {}
        for signal in residual:
            matches = list(common.OUTPUT_ONLY_FORBIDDEN_PATTERNS[signal].finditer(body))
            diagnostic[signal] = [
                {
                    "span": [match.start(), match.end()],
                    "context": body[max(0, match.start() - 80) : match.end() + 120],
                }
                for match in matches[:4]
            ]
        raise RuntimeError(
            "output body still contains forbidden framing after extraction: "
            f"{residual}; diagnostic={diagnostic!r}"
        )
    return body, audit


def audit_completion(
    tokenizer: Any,
    common: ModuleType,
    *,
    row: dict[str, Any],
    prompt_ids: list[int],
    token_ids: list[int],
    finish_reason: str | None,
    stop_reason: int | str | None,
    max_new_tokens: int,
) -> dict[str, Any]:
    raw_ids = [int(value) for value in token_ids]
    eos_position = next(
        (i for i, value in enumerate(raw_ids) if value in EOS_TOKEN_IDS), None
    )
    terminal_limit = (
        finish_reason == "length"
        and eos_position is None
        and len(raw_ids) == max_new_tokens
    )
    violations: list[str] = []
    if finish_reason != "stop" and not terminal_limit:
        violations.append("vllm_finish_reason_not_stop")
    if eos_position is None:
        prediction_ids = raw_ids
        if not terminal_limit:
            violations.append("generation_did_not_stop_at_eos")
    else:
        prediction_ids = raw_ids[:eos_position]
        if eos_position != len(raw_ids) - 1:
            violations.append("tokens_after_eos")
    if len(raw_ids) >= max_new_tokens and eos_position is None and not terminal_limit:
        violations.append("generation_limit_exhausted")
    special_ids = {int(value) for value in (tokenizer.all_special_ids or [])}
    special_in_prediction = sorted(set(prediction_ids) & special_ids)
    if special_in_prediction:
        violations.append("special_token_in_prediction")
    raw_continuation = tokenizer.decode(
        prediction_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    try:
        prediction, normalization_audit = extract_output_body(raw_continuation, common)
    except RuntimeError as error:
        raise RuntimeError(
            f"output-only generation audit failed for {row_key(row)}: {error}"
        ) from error
    if "\ufffd" in prediction or any(
        0xD800 <= ord(char) <= 0xDFFF for char in prediction
    ):
        violations.append("invalid_unicode")
    if any(
        unicodedata.category(char) == "Cc" and char not in {"\n", "\t"}
        for char in prediction
    ):
        violations.append("unicode_control_character")
    output_signals = output_body_violations(prediction, common)
    quality_failures = output_quality_failures(prediction, common)
    roundtrip_ids = [
        int(value) for value in tokenizer.encode(prediction, add_special_tokens=False)
    ]
    roundtrip_text = tokenizer.decode(
        roundtrip_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    tokenizer_roundtrip = roundtrip_text == prediction
    unicode_roundtrip = prediction.encode("utf-8").decode("utf-8") == prediction
    if not tokenizer_roundtrip:
        violations.append("tokenizer_roundtrip_mismatch")
    if not unicode_roundtrip:
        violations.append("unicode_roundtrip_mismatch")
    violations = sorted(set(violations))
    if violations:
        raise RuntimeError(
            f"output-only generation audit failed for {row_key(row)}: {violations}"
        )
    result = {
        "prediction": prediction,
        "raw_continuation": raw_continuation,
        "prompt_token_ids": prompt_ids,
        "raw_generation_token_ids": raw_ids,
        "prediction_token_ids": roundtrip_ids,
        "prompt_tokens": len(prompt_ids),
        "prediction_tokens": len(roundtrip_ids),
        "generation_stop_reason": ("max_new_tokens" if terminal_limit else "eos"),
        "eos_position": eos_position,
        "eos_token_ids": sorted(EOS_TOKEN_IDS),
        "special_token_ids_in_prediction": special_in_prediction,
        "tokenizer_roundtrip_exact": tokenizer_roundtrip,
        "unicode_roundtrip_exact": unicode_roundtrip,
        "vllm_finish_reason": finish_reason,
        "vllm_stop_reason": stop_reason,
        "output_violation_audit": {
            "status": ("PASS_WITH_QUALITY_FAILURE" if quality_failures else "PASS"),
            "violations": output_signals,
            "quality_failures": quality_failures,
            "postprocessing": OUTPUT_BODY_POLICY,
        },
        "output_normalization_audit": normalization_audit,
    }
    if terminal_limit:
        result["nonfatal_terminal_limit_audit"] = {
            "status": "FAIL",
            "violations": list(TERMINAL_LIMIT_SIGNALS),
            "policy": TERMINAL_LIMIT_POLICY,
        }
    return result


def build_contract(
    *,
    variant: str,
    model_label: str,
    evaluation: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    if variant not in {"base", "tuned"}:
        raise ValueError(f"invalid model variant: {variant}")
    adapter = None
    if variant == "tuned":
        adapter = {
            "path": str(ADAPTER_DIR),
            "config_sha256": sha256(ADAPTER_DIR / "adapter_config.json"),
            "weights_sha256": sha256(ADAPTER_DIR / "adapter_model.safetensors"),
            "rank": 32,
            "module_count": 528,
        }
        if adapter["weights_sha256"] != ADAPTER_WEIGHTS_SHA256:
            raise RuntimeError("selected training adapter changed")
    decoding = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "seed": 42,
        "max_model_len": MAX_MODEL_LEN,
        "translation_max_new_tokens": TASK_MAX_NEW_TOKENS["translation"],
        "summarization_max_new_tokens": TASK_MAX_NEW_TOKENS["summarization"],
        "eos_token_ids": sorted(EOS_TOKEN_IDS),
        "prompt_truncation": "forbidden",
        "postprocessing": (
            "deterministic leading wrapper extraction; no answer-content rewrite"
        ),
    }
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "backend": "upstage-vllm-solar-open2-nvfp4",
        "backend_version": "0.1.dev16959+g00907fc9b",
        "backend_commit": "00907fc9b982f973808a0193ad0f8c61fbb62854",
        "model_id": "nota-ai/Solar-Open2-250B-Nota-NVFP4",
        "model_revision": MODEL_REVISION,
        "model_dir": str(MODEL_DIR),
        "model_index_sha256": MODEL_INDEX_SHA256,
        "tokenizer_dir": str(TOKENIZER_DIR),
        "variant": variant,
        "model_label": model_label,
        "adapter": adapter,
        "evaluation": evaluation,
        "prompt": {"variant": "no_dictionary_context", **TEMPLATE_KWARGS},
        "decoding": decoding,
        "runtime": runtime,
        "methodology": {
            "inference_prompt_and_row_validator": str(EVALUATION_PATH),
            "evaluation_py_sha256": EVALUATION_SHA256,
            "final_scorer_only": "project/scripts/evaluation.py score",
            "output_audit": str(COMMON_PATH),
            "output_audit_sha256": COMMON_SHA256,
            "output_violation_policy": {
                "action": "score_extracted_body_only",
                "policy": OUTPUT_BODY_POLICY,
                "raw_generation_preserved": True,
                "answer_content_rewrite": "forbidden",
                "internal_document_label_policy": {
                    "signals": list(DOCUMENT_INTERNAL_LABEL_SIGNALS),
                    "action": "preserve after leading-wrapper extraction",
                    "reason": "document fields are answer content",
                },
                "quality_failure_policy": {
                    "signals": list(QUALITY_FAILURE_SIGNALS),
                    "action": "preserve_and_score_as_generated",
                    "raw_text_mutation": "none",
                    "reason": (
                        "semantic model failures must lower metrics, "
                        "not abort evaluation"
                    ),
                },
                "residual_violation_action": (
                    "fail_closed_except_recorded_quality_failures"
                ),
            },
            "terminal_limit_policy": {
                "action": "preserve_raw_and_continue",
                "raw_text_mutation": "none",
                "allowed_nonfatal_violations": list(TERMINAL_LIMIT_SIGNALS),
                "audit_policy": TERMINAL_LIMIT_POLICY,
            },
        },
        "receipts": {
            "asset": {"path": str(ASSET_RECEIPT), "sha256": sha256(ASSET_RECEIPT)},
            "environment": {
                "path": str(ENVIRONMENT_RECEIPT),
                "sha256": sha256(ENVIRONMENT_RECEIPT),
            },
        },
    }


def manifest_path(output_jsonl: Path) -> Path:
    return output_jsonl.with_name(output_jsonl.name + ".manifest.json")


def validate_saved_row(
    saved: dict[str, Any],
    source: dict[str, Any],
    *,
    contract_sha256: str,
    identity: dict[str, Any],
    common: ModuleType,
) -> None:
    wanted = {
        "id": source["id"],
        "task": source["task"],
        "origin": source["origin"],
        "reference": source["response"],
        "reference_source": source.get("reference_source"),
        "prompt_variant": "no_dictionary_context",
        "inference_contract_sha256": contract_sha256,
        "evaluation_split": identity["split"],
        "evaluation_row_identity": identity,
    }
    for field, value in wanted.items():
        if saved.get(field) != value:
            raise RuntimeError(
                f"resume row invariant changed: {row_key(source)} field={field}"
            )
    max_new_tokens = TASK_MAX_NEW_TOKENS[str(source["task"])]
    required_lists = (
        "prompt_token_ids",
        "raw_generation_token_ids",
        "prediction_token_ids",
    )
    if any(not isinstance(saved.get(field), list) for field in required_lists):
        raise RuntimeError(f"resume row lacks raw token evidence: {row_key(source)}")
    prediction = saved.get("prediction")
    raw = saved.get("raw_continuation")
    violations = [] if isinstance(prediction, str) else ["prediction_not_string"]
    expected_normalization = None
    if not isinstance(raw, str):
        violations.append("raw_continuation_not_string")
    else:
        try:
            expected_prediction, expected_normalization = extract_output_body(
                raw, common
            )
        except RuntimeError:
            violations.append("saved_raw_output_not_extractable")
        else:
            if prediction != expected_prediction:
                violations.append("deterministic_output_body_mismatch")
    output_signals = output_body_violations(
        prediction if isinstance(prediction, str) else "", common
    )
    quality_failures = output_quality_failures(
        prediction if isinstance(prediction, str) else "", common
    )
    terminal_limit = saved.get("generation_stop_reason") == "max_new_tokens"
    if terminal_limit:
        wanted_terminal_audit = {
            "status": "FAIL",
            "violations": list(TERMINAL_LIMIT_SIGNALS),
            "policy": TERMINAL_LIMIT_POLICY,
        }
        if (
            saved.get("eos_position") is not None
            or saved.get("vllm_finish_reason") != "length"
            or len(saved["raw_generation_token_ids"]) != max_new_tokens
            or bool(set(saved["raw_generation_token_ids"]) & EOS_TOKEN_IDS)
            or saved.get("special_token_ids_in_prediction") != []
            or saved.get("nonfatal_terminal_limit_audit") != wanted_terminal_audit
        ):
            violations.append("terminal_limit_evidence_mismatch")
    elif saved.get("generation_stop_reason") != "eos":
        violations.append("generation_did_not_stop_at_eos")
    elif "nonfatal_terminal_limit_audit" in saved:
        violations.append("unexpected_terminal_limit_audit")
    if saved.get("prediction_tokens") != len(saved["prediction_token_ids"]):
        violations.append("prediction_token_count_mismatch")
    if saved.get("prompt_tokens") != len(saved["prompt_token_ids"]):
        violations.append("prompt_token_count_mismatch")
    if len(saved["raw_generation_token_ids"]) > max_new_tokens:
        violations.append("generation_limit_exceeded")
    wanted_output_audit = {
        "status": ("PASS_WITH_QUALITY_FAILURE" if quality_failures else "PASS"),
        "violations": output_signals,
        "quality_failures": quality_failures,
        "postprocessing": OUTPUT_BODY_POLICY,
    }
    if saved.get("output_violation_audit") != wanted_output_audit:
        violations.append("output_audit_mismatch")
    if saved.get("output_normalization_audit") != expected_normalization:
        violations.append("output_normalization_audit_mismatch")
    if violations:
        raise RuntimeError(
            f"resume row output contract failed: {row_key(source)} {sorted(set(violations))}"
        )


def initialize_or_resume(
    output_jsonl: Path,
    *,
    contract: dict[str, Any],
    rows: list[dict[str, Any]],
    identities: dict[tuple[str, str], dict[str, Any]],
    resume: bool,
) -> tuple[Path, dict[str, Any], set[tuple[str, str]]]:
    path = manifest_path(output_jsonl)
    digest = canonical_sha256(contract)
    if not resume and (output_jsonl.exists() or path.exists()):
        raise FileExistsError(f"output exists; pass --resume: {output_jsonl}")
    if output_jsonl.exists() and not path.is_file():
        raise RuntimeError("prediction JSONL exists without manifest")
    if path.is_file():
        manifest = read_json(path)
        if (
            manifest.get("contract") != contract
            or manifest.get("contract_sha256") != digest
        ):
            raise RuntimeError("resume inference contract changed")
        if manifest.get("status") not in {"IN_PROGRESS", "COMPLETE"}:
            raise RuntimeError("resume manifest status is invalid")
    else:
        manifest = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "status": "IN_PROGRESS",
            "created_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
            "contract": contract,
            "contract_sha256": digest,
            "expected_rows": len(rows),
            "completed_rows": 0,
        }
        atomic_json(path, manifest)
    selected = {row_key(row): row for row in rows}
    done: set[tuple[str, str]] = set()
    _, common = load_reference_modules()
    if output_jsonl.is_file():
        for saved in read_jsonl(output_jsonl):
            key = row_key(saved)
            if key in done or key not in selected:
                raise RuntimeError(f"duplicate or unexpected resume key: {key}")
            validate_saved_row(
                saved,
                selected[key],
                contract_sha256=digest,
                identity=identities[key],
                common=common,
            )
            done.add(key)
    if manifest["status"] == "COMPLETE":
        if len(done) != len(rows) or manifest.get("output_jsonl_sha256") != sha256(
            output_jsonl
        ):
            raise RuntimeError("sealed prediction output changed or is incomplete")
    return path, manifest, done


def update_manifest(path: Path, manifest: dict[str, Any], completed: int) -> None:
    manifest["status"] = "IN_PROGRESS"
    manifest["completed_rows"] = completed
    manifest["updated_at_utc"] = utc_now()
    atomic_json(path, manifest)


def seal_manifest(
    path: Path,
    manifest: dict[str, Any],
    *,
    output_jsonl: Path,
    completed: int,
    benchmark: dict[str, Any],
) -> None:
    if completed != manifest["expected_rows"]:
        raise RuntimeError("cannot seal incomplete vLLM predictions")
    manifest.update(
        status="COMPLETE",
        completed_rows=completed,
        updated_at_utc=utc_now(),
        completed_at_utc=utc_now(),
        output_jsonl_sha256=sha256(output_jsonl),
        benchmark=benchmark,
    )
    atomic_json(path, manifest)
