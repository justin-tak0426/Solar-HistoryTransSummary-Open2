"""Disposable-process LoRA merge qualification for faster tuned inference."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _lora_parameter_names(model: Any) -> list[str]:
    return sorted(
        name
        for name, _parameter in model.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    )


@dataclass(frozen=True)
class MergeQualificationReceipt:
    status: str
    safe_merge: bool
    premerge_lora_parameter_count: int
    postmerge_lora_parameter_count: int
    parity_probe_sha256: str
    merge_seconds: float
    hf_device_map_preserved: bool
    exact_probe_parity: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def merge_and_validate_for_disposable_process(
    model: Any,
    *,
    parity_probe: Callable[[Any], Any],
    expected_lora_parameter_count: int | None = None,
) -> tuple[Any, MergeQualificationReceipt]:
    """Merge a PEFT adapter and require exact caller-defined probe parity.

    The operation is intentionally irreversible.  Run it only in a disposable
    smoke/serving process, never in the process producing an existing evaluation
    segment.  ``parity_probe`` should use greedy generation and return JSON-safe
    token-id data, not decoded strings.
    """

    if not callable(getattr(model, "merge_and_unload", None)):
        raise TypeError("model is not a merge-capable PEFT model")
    before_names = _lora_parameter_names(model)
    if not before_names:
        raise RuntimeError("model contains no LoRA parameters to merge")
    if (
        expected_lora_parameter_count is not None
        and len(before_names) != expected_lora_parameter_count
    ):
        raise RuntimeError(
            "LoRA parameter tensor count changed: "
            f"{len(before_names)} != {expected_lora_parameter_count}"
        )
    before_probe = parity_probe(model)
    before_digest = _canonical_digest(before_probe)
    device_map = dict(getattr(model, "hf_device_map", {}))

    started = time.perf_counter()
    merged = model.merge_and_unload(safe_merge=True, progressbar=False)
    elapsed = time.perf_counter() - started
    if merged is None:
        raise RuntimeError("PEFT merge_and_unload unexpectedly returned None")
    if device_map:
        merged.hf_device_map = device_map
    if hasattr(merged, "config"):
        merged.config.use_cache = True
    merged.eval()
    for parameter in merged.parameters():
        parameter.requires_grad_(False)

    after_names = _lora_parameter_names(merged)
    if after_names:
        raise RuntimeError(f"LoRA parameters remain after merge: {after_names[:8]}")
    after_probe = parity_probe(merged)
    after_digest = _canonical_digest(after_probe)
    if before_probe != after_probe:
        raise RuntimeError(
            "merged adapter changed exact greedy probe token IDs; "
            f"before={before_digest} after={after_digest}"
        )
    receipt = MergeQualificationReceipt(
        status="PASS",
        safe_merge=True,
        premerge_lora_parameter_count=len(before_names),
        postmerge_lora_parameter_count=0,
        parity_probe_sha256=before_digest,
        merge_seconds=elapsed,
        hf_device_map_preserved=(
            not device_map or dict(getattr(merged, "hf_device_map", {})) == device_map
        ),
        exact_probe_parity=True,
    )
    if not receipt.hf_device_map_preserved:
        raise RuntimeError("two-GPU device map changed during LoRA merge")
    return merged, receipt
