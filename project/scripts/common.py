#!/usr/bin/env python3
"""Shared, dependency-light contracts for the Solar Open 2 pipeline."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator


PROJECT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT / "config"
B200_ROOT = PROJECT.parent
B200_CONFIG_DIR = PROJECT / "config" / "b200"
B200_TEMP_ROOT = B200_ROOT / "runtime" / "tmp"
REFERENCE_COPY = PROJECT / "scripts/reference/shared_nodict_evaluation.py"
REFERENCE_SOURCE = Path(
    "/data/home/mk01/NIPAHistoryTransSummary/scripts/shared_nodict_evaluation.py"
)
REFERENCE_SHA256 = "a6e68d29f3db569ff78283f15b10680c11c2ff9f6472453dc8fd6ef9a7320b23"
LICENSE_SHA256 = "de031be88349e197e9ae20b7051679f55eb50ca53f7ffb2362343f8516b02d66"
OWNED_ROOTS = (
    B200_ROOT,
    Path("/data/home/mk03"),
    Path("/data1/jhtak/Solar-HistoryTransSummary-Open2"),
)
TASKS = ("translation", "summarization")
RESULT_LABEL_PREFIX_RE = re.compile(r"^답변\s*[:：]\s*")
SOLAR_OUTPUT_ONLY_GENERATION_PREAMBLE = (
    "<|im:start|>assistant<|im:content|><|think:start|><|think:end|>"
)
SOLAR_ASSISTANT_TEMPLATE_TERMINATION = "<|im:end|>\n"
OUTPUT_ONLY_FORBIDDEN_PATTERNS = {
    "special_control_token": re.compile(r"<\|[^|>\r\n]{1,64}\|>", re.I),
    "reasoning_tag": re.compile(r"</?(?:think|analysis|reasoning)>", re.I),
    "role_header": re.compile(r"(?im)^\s*(?:assistant|system|user)\s*[:：]"),
    "result_label": re.compile(
        r"(?im)^\s*(?:[-*+>]\s*)?(?:\*\*|__)?\s*(?:최종\s*)?"
        r"(?:번역(?:문|본| 결과)?|요약(?:문|본| 결과)?|답변|결과|"
        r"현대어\s*(?:풀이|번역|요약)|"
        r"(?:현대\s*)?한국어\s*(?:번역|요약)(?:\s*(?:문|본문|결과))?|"
        r"translation|summary|answer|result)\s*(?:\*\*|__)?\s*"
        r"(?:[:：]|$)"
    ),
    "meta_framing": re.compile(
        r"(?im)^\s*(?:(?:다음은|아래는|요청하신).{0,64}"
        r"(?:번역|요약|풀이)(?:문|본| 결과)?"
        r"(?:입니다|문입니다|입니다만)[.:\s]*|"
        r"제가\s+(?:번역|요약)한\s+(?:내용|결과)(?:은|는)\s+"
        r"다음과\s+같습니다[.:\s]*|"
        r"(?:번역|요약)(?:하면|하자면)\s*[,，:]\s*|"
        r"물론입니다[.!]?\s*|"
        r"here(?:'s| is| are)\s+(?:(?:the|your|a)\s+)?"
        r"(?:translation|summary|answer)\b[.:\s]*)"
    ),
    "refusal_framing": re.compile(
        r"(?im)^\s*(?:(?:죄송(?:합니다|하지만)[,.!]?\s*)?"
        r"(?:(?:번역|요약|요청|답변).{0,32})?"
        r"(?:할|드릴|수행할)\s+수\s+없(?:습니다|어요)|"
        r"(?:i\s+)?(?:cannot|can't|can not|unable to)\s+"
        r"(?:translate|summari[sz]e|answer|comply)\b)"
    ),
    "explanation_framing": re.compile(
        r"(?im)^\s*(?:설명|해설|주석|노트|" r"explanation|commentary|note)\s*[:：]"
    ),
    "reasoning_label": re.compile(
        r"(?im)^\s*(?:추론|분석|사고\s*과정|생각\s*과정|이유|근거|"
        r"reasoning|analysis|rationale)\s*[:：]"
    ),
    "reasoning_prose_framing": re.compile(
        r"(?im)^\s*(?:(?:먼저\s+)?(?:원문|문장|내용)을?\s+"
        r"(?:분석|검토|살펴보)|(?:다음|아래)과?\s*같이\s+"
        r"(?:번역|요약|풀이)|let(?:'|’)s\s+"
        r"(?:analy[sz]e|translate|summari[sz]e)\b)"
    ),
    "chat_control_marker": re.compile(
        r"(?im)(?:\[/?INST\]|<</?SYS>>|</?s>|"
        r"^\s*###\s*(?:assistant|system|user)\b|"
        r"</?(?:assistant|system|user)>)"
    ),
    "explicit_markdown": re.compile(
        r"(?m)(?:^\s*#{1,6}\s+|" r"!?\[[^\]\r\n]+\]\((?:https?://|mailto:)[^)]+\))",
    ),
    "blockquote_framing": re.compile(r"(?m)^\s*>\s+"),
    "markdown_fence": re.compile(r"```|~~~"),
    "prompt_echo": re.compile(
        r"Return only the (?:translated|summarized) Korean text|"
        r"Translate the source into contemporary Korean|"
        r"Summarize the source in contemporary Korean",
        re.I,
    ),
    "prompt_echo_extended": re.compile(
        r"You are a professional specialist|"
        r"Use only information explicitly present in the source text|"
        r"Do not mention metadata, reasoning, or these instructions|"
        r"Return Korean body text only|"
        r"Use a formal documentary style|"
        r"Do not add labels, notes, explanations, markdown, bullets, "
        r"reasoning, or framing|"
        r"Compress only information explicitly present in the source|"
        r"^\s*Source\s*:",
        re.I | re.M,
    ),
    "non_korean_body": re.compile(r"^[^가-힣]*$", re.DOTALL),
}
TRAINING_RESULT_LABEL_IDS = frozenset(
    {
        "sa_100_0020_0020_0010_011",
        "sa_100_0020_0020_0010_162",
        "sa_100_0020_0020_0010_192",
    }
)
LORA_TARGET_REGEX = re.compile(
    r"^(?:model\.)?model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj|g_proj|b_proj|"
    r"f_a_proj|f_b_proj|g_a_proj|g_b_proj)|"
    r"mlp\.shared_experts\.(?:gate_proj|up_proj|down_proj))$"
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: expected JSON object")
            yield value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def output_only_violations(text: str) -> list[str]:
    """Return forbidden framing present in an otherwise raw answer body."""

    return [
        name
        for name, pattern in OUTPUT_ONLY_FORBIDDEN_PATTERNS.items()
        if pattern.search(text)
    ]


def pipeline_code_identity(
    *,
    config_dir: Path | None = None,
) -> dict[str, Any]:
    """Bind receipts and resumable state to the executable Solar source tree."""

    selected: set[Path] = {
        path
        for path in (PROJECT / "scripts").rglob("*.py")
        if "__pycache__" not in path.parts
    }
    if config_dir is not None:
        resolved_config = config_dir.resolve(strict=True)
        if resolved_config != B200_CONFIG_DIR.resolve(strict=True):
            raise ValueError(
                "B200 code identity requires exact config directory: "
                f"{resolved_config} != {B200_CONFIG_DIR.resolve(strict=True)}"
            )
        selected.update(resolved_config.glob("*.json"))
        launch_dir = PROJECT / "launch" / "b200"
        if launch_dir.is_dir():
            selected.update(
                path
                for path in launch_dir.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix != ".pyc"
            )
        migration_dir = PROJECT / "migration" / "b200"
        selected.update(
            path
            for path in migration_dir.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        )
        selected.update(
            path for path in (PROJECT / "requirements").iterdir() if path.is_file()
        )
    files = {}
    for path in sorted(selected):
        resolved = path.resolve(strict=True)
        if resolved != B200_ROOT and B200_ROOT not in resolved.parents:
            raise RuntimeError(f"pipeline identity path escapes B200 root: {path}")
        files[str(path.relative_to(PROJECT))] = sha256(path)
    return {
        "files": files,
        "sha256": canonical_sha256(files),
    }


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def require_owned_path(
    path: Path,
    *,
    create_parent: bool = False,
    owned_roots: Iterable[Path] | None = None,
) -> Path:
    resolved = path.expanduser().resolve()
    roots = tuple(
        root.expanduser().resolve()
        for root in (OWNED_ROOTS if owned_roots is None else owned_roots)
    )
    if not roots:
        raise ValueError("at least one owned root is required")
    if not any(resolved == root or root in resolved.parents for root in roots):
        raise ValueError(f"refusing path outside owned roots: {resolved}")
    if create_parent:
        resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def load_reference_module(
    *,
    config_dir: Path | None = None,
    owned_only: bool = True,
) -> Any:
    """Load the sealed owned evaluator; legacy source comparison is opt-in."""

    if config_dir is not None:
        resolved_config = config_dir.resolve(strict=True)
        if resolved_config != B200_CONFIG_DIR.resolve(strict=True):
            raise ValueError(
                f"unsupported evaluator config directory: {resolved_config}"
            )
        owned_only = True
    checked_paths = (
        (REFERENCE_COPY,)
        if owned_only
        else (
            REFERENCE_COPY,
            REFERENCE_SOURCE,
        )
    )
    for path in checked_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != REFERENCE_SHA256:
            raise RuntimeError(
                f"reference evaluator changed: path={path} "
                f"expected={REFERENCE_SHA256} actual={actual}"
            )
    if not owned_only and REFERENCE_COPY.read_bytes() != REFERENCE_SOURCE.read_bytes():
        raise RuntimeError("owned evaluator copy is not byte-identical to source")
    spec = importlib.util.spec_from_file_location(
        "solar_official_nodict_reference", REFERENCE_COPY
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import reference evaluator: {REFERENCE_COPY}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_lora_module_names(model_config: dict[str, Any]) -> set[str]:
    layers = int(model_config["num_hidden_layers"])
    gqa_layers = {int(value) for value in model_config["gqa_layers"]}
    use_full_proj = bool(model_config["kda_use_full_proj"])
    names: set[str] = set()
    for layer in range(layers):
        prefix = f"model.layers.{layer}"
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            names.add(f"{prefix}.self_attn.{name}")
        if layer in gqa_layers:
            names.add(f"{prefix}.self_attn.g_proj")
        elif use_full_proj:
            names.add(f"{prefix}.self_attn.f_proj")
            names.add(f"{prefix}.self_attn.g_proj")
        else:
            for name in ("b_proj", "f_a_proj", "f_b_proj", "g_a_proj", "g_b_proj"):
                names.add(f"{prefix}.self_attn.{name}")
        for name in ("gate_proj", "up_proj", "down_proj"):
            names.add(f"{prefix}.mlp.shared_experts.{name}")
    return names


def audit_lora_modules(model: Any, model_config: dict[str, Any]) -> dict[str, Any]:
    expected = expected_lora_module_names(model_config)
    available = {
        name.removeprefix("base_model.model.")
        for name, _module in model.named_modules()
        if LORA_TARGET_REGEX.fullmatch(name)
        or LORA_TARGET_REGEX.fullmatch(name.removeprefix("base_model.model."))
    }
    missing = sorted(expected - available)
    unexpected = sorted(available - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"LoRA module contract mismatch: missing={missing[:8]} "
            f"unexpected={unexpected[:8]}"
        )
    return {
        "status": "PASS",
        "module_count": len(available),
        "module_names_sha256": canonical_sha256(sorted(available)),
    }


def token_subsequence_positions(
    supervised_ids: list[int],
    answer_end: int,
    readings: Iterable[str],
    tokenizer: Any,
) -> set[int]:
    weighted: set[int] = set()
    answer = supervised_ids[:answer_end]
    for reading in readings:
        reading_ids = tokenizer(str(reading), add_special_tokens=False)["input_ids"]
        if not reading_ids:
            continue
        width = len(reading_ids)
        for start in range(0, len(answer) - width + 1):
            if answer[start : start + width] == reading_ids:
                weighted.update(range(start, start + width))
    return weighted


def encode_row(
    tokenizer: Any,
    prompt_module: Any,
    row: dict[str, Any],
    *,
    hard_term_weight: float,
) -> dict[str, Any]:
    row_id = str(row["id"])
    raw_response = str(row["response"])
    prefix_match = RESULT_LABEL_PREFIX_RE.match(raw_response)
    if prefix_match is not None:
        if row_id not in TRAINING_RESULT_LABEL_IDS:
            raise RuntimeError(
                f"unexpected answer-label prefix in training target: {row_id}"
            )
        response = raw_response[prefix_match.end() :]
    else:
        if row_id in TRAINING_RESULT_LABEL_IDS:
            raise RuntimeError(
                f"pinned training answer-label prefix disappeared: {row_id}"
            )
        response = raw_response
    if not response or response != response.strip():
        raise RuntimeError(
            f"assistant answer must be nonempty body text without outer whitespace: "
            f"{row['id']}"
        )
    violations = output_only_violations(response)
    if violations:
        raise RuntimeError(
            f"assistant answer contains non-body output {violations}: {row['id']}"
        )
    messages = prompt_module.build_messages_no_dict(row["task"], row["origin"])
    template_kwargs = {
        "reasoning_effort": "none",
        "think_render_option": "preserved",
        "provider_system_prompt": False,
    }
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    full_text = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": response}],
        tokenize=False,
        add_generation_prompt=False,
        **template_kwargs,
    )
    if not prompt_text.endswith(SOLAR_OUTPUT_ONLY_GENERATION_PREAMBLE):
        raise RuntimeError(
            f"Solar output-only generation preamble changed: {row['id']}"
        )
    expected_full_text = prompt_text + response + SOLAR_ASSISTANT_TEMPLATE_TERMINATION
    if full_text != expected_full_text:
        raise RuntimeError(
            f"Solar assistant body/termination rendering changed: {row['id']}"
        )
    if not full_text.startswith(prompt_text):
        raise RuntimeError(f"assistant rendering is not prompt-prefixed: {row['id']}")
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    template_full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if template_full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError(f"prompt token prefix mismatch: {row['id']}")
    answer_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    template_supervised_ids = template_full_ids[len(prompt_ids) :]
    if template_supervised_ids[: len(answer_ids)] != answer_ids:
        raise RuntimeError(f"answer token alignment mismatch: {row['id']}")
    termination_ids = tokenizer(
        SOLAR_ASSISTANT_TEMPLATE_TERMINATION,
        add_special_tokens=False,
    )["input_ids"]
    if not answer_ids:
        raise RuntimeError(f"assistant answer supervision is empty: {row['id']}")
    if not termination_ids:
        raise RuntimeError(f"assistant termination is empty: {row['id']}")
    if template_supervised_ids != answer_ids + termination_ids:
        raise RuntimeError(
            f"assistant template contains non-body/non-termination tokens: "
            f"{row['id']}"
        )
    termination_token_id = int(termination_ids[0])
    # Solar exposes <|im:end|> as atomic ID 129 and a generation EOS, while
    # Transformers only reports the semantic BOS/EOS/UNK IDs in all_special_ids.
    # The real-tokenizer gate binds this ID to generation_config.json below.
    supervised_ids = answer_ids + [termination_token_id]
    full_ids = prompt_ids + supervised_ids
    labels = [-100] * len(prompt_ids) + supervised_ids
    weights = [0.0] * len(prompt_ids) + [1.0] * len(supervised_ids)
    if any(label != -100 for label in labels[: len(prompt_ids)]):
        raise AssertionError("prompt tokens unexpectedly entered the training loss")
    if any(weight != 0.0 for weight in weights[: len(prompt_ids)]):
        raise AssertionError("prompt tokens unexpectedly received loss weights")
    weighted = token_subsequence_positions(
        supervised_ids,
        len(answer_ids),
        row.get("hard_term_readings") or [],
        tokenizer,
    )
    for position in weighted:
        weights[len(prompt_ids) + position] = hard_term_weight
    return {
        "input_ids": full_ids,
        "labels": labels,
        "weights": weights,
        "length": len(full_ids),
        "hard_term_weighted_tokens": len(weighted),
        "output_only_supervision": {
            "prompt_token_count": len(prompt_ids),
            "answer_token_count": len(answer_ids),
            "assistant_termination_token_count": (
                len(supervised_ids) - len(answer_ids)
            ),
            "assistant_termination_token_id": termination_token_id,
            "template_suffix_token_count": len(termination_ids),
            "template_suffix_starts_with_eos": True,
            "supervised_sequence_ends_at_eos": True,
            "prompt_labels_all_masked": True,
            "prompt_weights_all_zero": True,
            "supervised_prefix_is_answer_body": True,
            "reasoning_effort": "none",
            "answer_label_prefix_removed": prefix_match is not None,
        },
    }


class TaskDeckSampler:
    """Task-weighted, without-replacement decks with explicit expert duplication."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        task_weights: dict[str, float],
        expert_label: str,
        expert_boost: int,
        seed: int,
    ) -> None:
        if expert_boost < 1:
            raise ValueError("expert_boost must be >= 1")
        if set(task_weights) != set(TASKS):
            raise ValueError(f"task_weights must cover {TASKS}")
        if abs(sum(task_weights.values()) - 1.0) > 1e-9:
            raise ValueError("task weights must sum to 1")
        self.rows = rows
        self.task_weights = dict(task_weights)
        self.expert_label = expert_label
        self.expert_boost = expert_boost
        self.rng = random.Random(seed)
        self.pools: dict[str, list[int]] = {}
        self.decks: dict[str, list[int]] = {}
        self.positions: dict[str, int] = {}
        for task in TASKS:
            base = [index for index, row in enumerate(rows) if row["task"] == task]
            experts = [
                index for index in base if rows[index].get("quality") == expert_label
            ]
            if not base or not experts:
                raise ValueError(f"task lacks base or expert rows: {task}")
            self.pools[task] = base + experts * (expert_boost - 1)
            self._refill(task)

    def _refill(self, task: str) -> None:
        deck = list(self.pools[task])
        self.rng.shuffle(deck)
        self.decks[task] = deck
        self.positions[task] = 0

    def draw(self) -> dict[str, Any]:
        threshold = self.rng.random()
        cumulative = 0.0
        selected = TASKS[-1]
        for task in TASKS:
            cumulative += self.task_weights[task]
            if threshold <= cumulative:
                selected = task
                break
        if self.positions[selected] >= len(self.decks[selected]):
            self._refill(selected)
        index = self.decks[selected][self.positions[selected]]
        self.positions[selected] += 1
        return self.rows[index]

    def state_dict(self) -> dict[str, Any]:
        return {
            "rng_state": self.rng.getstate(),
            "decks": self.decks,
            "positions": self.positions,
            "pool_counts": {task: len(pool) for task, pool in self.pools.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {task: len(pool) for task, pool in self.pools.items()}
        if state["pool_counts"] != expected:
            raise ValueError("sampler pool changed across resume")
        self.rng.setstate(state["rng_state"])
        self.decks = {task: list(state["decks"][task]) for task in TASKS}
        self.positions = {task: int(state["positions"][task]) for task in TASKS}

    def stats(self) -> dict[str, Any]:
        result = {}
        for task in TASKS:
            quality = Counter(
                self.rows[index].get("quality") for index in self.pools[task]
            )
            result[task] = {
                "pool_rows": len(self.pools[task]),
                "quality": dict(quality),
            }
        return result


class GlobalAugmentedDeckSampler:
    """Finite, deterministic epoch decks with explicit expert duplication."""

    STATE_SCHEMA = "global_augmented_deck_sampler"
    STATE_VERSION = 2
    OCCURRENCE_IDENTITY_SCHEMA = "id/task/quality/expert_ordinal/epoch/position"

    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        expert_label: str,
        expert_boost: int,
        epochs: int,
        seed: int,
    ) -> None:
        if not rows:
            raise ValueError("rows must not be empty")
        if expert_boost < 1:
            raise ValueError("expert_boost must be >= 1")
        if epochs < 1:
            raise ValueError("epochs must be >= 1")

        identities: set[tuple[str, str]] = set()
        expert_indices: list[int] = []
        base_task_counts = Counter()
        expert_task_counts = Counter()
        contract_rows: list[list[str]] = []
        for index, row in enumerate(rows):
            row_id = str(row.get("id", "")).strip()
            task = str(row.get("task", ""))
            if not row_id:
                raise ValueError(f"row {index} lacks a non-empty id")
            if task not in TASKS:
                raise ValueError(f"row {index} has unsupported task: {task!r}")
            identity = (row_id, task)
            if identity in identities:
                raise ValueError(f"duplicate row identity: {identity!r}")
            identities.add(identity)
            quality = str(row.get("quality", ""))
            contract_rows.append([row_id, task, quality])
            base_task_counts[task] += 1
            if quality == expert_label:
                expert_indices.append(index)
                expert_task_counts[task] += 1

        self.rows = rows
        self.expert_label = expert_label
        self.expert_boost = int(expert_boost)
        self.epochs = int(epochs)
        self.seed = int(seed)
        occurrence_pool: list[tuple[int, int]] = [
            (index, 1) for index in range(len(rows))
        ]
        for ordinal in range(2, self.expert_boost + 1):
            occurrence_pool.extend((index, ordinal) for index in expert_indices)
        if len(occurrence_pool) != len(set(occurrence_pool)):
            raise AssertionError("augmented occurrence identities are not unique")
        self._occurrence_pool = occurrence_pool
        self._pool = list(range(len(self._occurrence_pool)))
        self.examples_per_epoch = len(self._pool)
        self.total_examples = self.examples_per_epoch * self.epochs
        self._base_task_counts = {task: int(base_task_counts[task]) for task in TASKS}
        self._expert_task_counts = {
            task: int(expert_task_counts[task]) for task in TASKS
        }
        self._task_examples_per_epoch = {
            task: (
                self._base_task_counts[task]
                + (self.expert_boost - 1) * self._expert_task_counts[task]
            )
            for task in TASKS
        }
        self._contract_sha256 = canonical_sha256(
            {
                "rows": contract_rows,
                "expert_label": self.expert_label,
                "expert_boost": self.expert_boost,
                "epochs": self.epochs,
                "seed": self.seed,
                "occurrence_identity_schema": self.OCCURRENCE_IDENTITY_SCHEMA,
            }
        )
        self._cursor = 0
        self._cached_epoch: int | None = None
        self._cached_deck: list[int] = []

    @property
    def cursor(self) -> int:
        """Number of examples already drawn across all epochs."""

        return self._cursor

    @property
    def epoch(self) -> int:
        """Zero-based epoch containing the next draw."""

        return min(self._cursor // self.examples_per_epoch, self.epochs)

    @property
    def position(self) -> int:
        """Position of the next draw within its epoch deck."""

        if self._cursor >= self.total_examples:
            return self.examples_per_epoch
        return self._cursor % self.examples_per_epoch

    @property
    def remaining(self) -> int:
        return self.total_examples - self._cursor

    def __len__(self) -> int:
        return self.total_examples

    def _epoch_seed(self, epoch: int) -> int:
        payload = (
            f"{self.STATE_SCHEMA}:{self.STATE_VERSION}:" f"{self.seed}:{epoch}"
        ).encode("ascii")
        return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")

    def _deck_for_epoch(self, epoch: int) -> list[int]:
        if not 0 <= epoch < self.epochs:
            raise IndexError(f"epoch out of range: {epoch}")
        if self._cached_epoch != epoch:
            deck = list(self._pool)
            random.Random(self._epoch_seed(epoch)).shuffle(deck)
            self._cached_epoch = epoch
            self._cached_deck = deck
        return self._cached_deck

    def occurrence_at(self, global_position: int) -> dict[str, Any]:
        """Return one deterministic occurrence identity without moving the cursor."""

        if isinstance(global_position, bool) or not isinstance(global_position, int):
            raise TypeError("global_position must be an integer")
        if not 0 <= global_position < self.total_examples:
            raise IndexError(
                f"global occurrence position out of range: {global_position}"
            )
        epoch, position = divmod(global_position, self.examples_per_epoch)
        occurrence_index = self._deck_for_epoch(epoch)[position]
        row_index, expert_ordinal = self._occurrence_pool[occurrence_index]
        row = self.rows[row_index]
        result = {
            "global_position": global_position,
            "epoch": epoch,
            "position": position,
            "row_index": row_index,
            "id": str(row["id"]),
            "task": str(row["task"]),
            "quality": str(row.get("quality", "")),
            "expert_ordinal": expert_ordinal,
        }
        result["identity_sha256"] = canonical_sha256(
            {
                key: result[key]
                for key in (
                    "id",
                    "task",
                    "quality",
                    "expert_ordinal",
                    "epoch",
                    "position",
                )
            }
        )
        return result

    def occurrences(
        self,
        start: int,
        count: int,
    ) -> list[dict[str, Any]]:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("count must be a non-negative integer")
        if isinstance(start, bool) or not isinstance(start, int) or start < 0:
            raise ValueError("start must be a non-negative integer")
        if start + count > self.total_examples:
            raise ValueError("occurrence slice exceeds the deterministic deck")
        return [
            self.occurrence_at(position) for position in range(start, start + count)
        ]

    def draw_occurrence(self) -> dict[str, Any]:
        if self._cursor >= self.total_examples:
            raise StopIteration("global augmented deck is exhausted")
        occurrence = self.occurrence_at(self._cursor)
        self._cursor += 1
        return occurrence

    def draw_index(self) -> int:
        return int(self.draw_occurrence()["row_index"])

    def draw(self) -> dict[str, Any]:
        return self.rows[self.draw_index()]

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": self.STATE_SCHEMA,
            "version": self.STATE_VERSION,
            "contract_sha256": self._contract_sha256,
            "cursor": self._cursor,
            "examples_per_epoch": self.examples_per_epoch,
            "total_examples": self.total_examples,
            "task_examples_per_epoch": dict(self._task_examples_per_epoch),
            "occurrence_identity_schema": self.OCCURRENCE_IDENTITY_SCHEMA,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("schema") != self.STATE_SCHEMA:
            raise ValueError("sampler state schema mismatch")
        if state.get("version") != self.STATE_VERSION:
            raise ValueError("sampler state version mismatch")
        if state.get("contract_sha256") != self._contract_sha256:
            raise ValueError("sampler contract changed across resume")
        expected = {
            "examples_per_epoch": self.examples_per_epoch,
            "total_examples": self.total_examples,
            "task_examples_per_epoch": self._task_examples_per_epoch,
            "occurrence_identity_schema": self.OCCURRENCE_IDENTITY_SCHEMA,
        }
        for name, value in expected.items():
            if state.get(name) != value:
                raise ValueError(f"sampler {name} changed across resume")
        cursor = state.get("cursor")
        if isinstance(cursor, bool) or not isinstance(cursor, int):
            raise ValueError("sampler cursor must be an integer")
        if not 0 <= cursor <= self.total_examples:
            raise ValueError("sampler cursor is out of range")
        self._cursor = cursor
        self._cached_epoch = None
        self._cached_deck = []

    def stats(self) -> dict[str, Any]:
        return {
            "base_rows": len(self.rows),
            "expert_rows": sum(self._expert_task_counts.values()),
            "expert_boost": self.expert_boost,
            "epochs": self.epochs,
            "examples_per_epoch": self.examples_per_epoch,
            "total_examples": self.total_examples,
            "base_task_counts": dict(self._base_task_counts),
            "expert_task_counts": dict(self._expert_task_counts),
            "task_examples_per_epoch": dict(self._task_examples_per_epoch),
            "task_total_examples": {
                task: count * self.epochs
                for task, count in self._task_examples_per_epoch.items()
            },
            "occurrence_identity_schema": self.OCCURRENCE_IDENTITY_SCHEMA,
        }
