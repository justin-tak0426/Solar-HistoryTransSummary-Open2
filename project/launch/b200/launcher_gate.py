#!/usr/bin/env python3
"""Fail-closed prerequisite receipt validation for B200 bash launchers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pwd
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/NHNHOME/WORKSPACE/26mfds002_B/jhtak/solar-open-2")
PROJECT = ROOT / "project"
SCRIPTS = PROJECT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import b200_runtime  # noqa: E402
from check_tokenizer_output_contract import (  # noqa: E402
    B200_VALIDATION_UNICODE_EXPECTED,
    validate_b200_input_unicode_inventory,
)
from common import canonical_sha256, pipeline_code_identity  # noqa: E402


RECEIPTS = {
    "asset": ROOT / "runtime/receipts/b200_asset_integrity_large_v1.json",
    "environment": ROOT / "runtime/receipts/b200_environment_v1.json",
    "kernel": ROOT / "runtime/receipts/b200_kernel_probe_v1.json",
    "tokenizer": ROOT / "runtime/receipts/b200_tokenizer_contract_v1.json",
    "preflight": ROOT / "runtime/receipts/b200_model_preflight_v1.json",
    "smoke": ROOT / "runtime/receipts/b200_smoke_2gpu_len9600_v1.json",
    "approval": ROOT / "runtime/receipts/b200_production_approval_v1.json",
}
METRIC_CACHE_ROOT = ROOT / "runtime/metric_hf/hub/models--xlm-roberta-large"
METRIC_REVISION = "c23d21b0620b635a76227c604d44e43a9f0ee389"
METRIC_FILES = {
    "config.json": ("8e5fb14e1352fd8fc678a7b293b63cfb5cf091f6", 616),
    "model.safetensors": (
        "2dfa19f172412917cab174da04b46e2134811b723666965fd0aabd97caa6e23b",
        2_244_817_354,
    ),
    "sentencepiece.bpe.model": (
        "db9af13bf09fd3028ca32be90d3fb66d5e470399",
        5_069_051,
    ),
    "tokenizer.json": ("463f3414782c1c9405828c9b31bfa36dda1f45c5", 9_096_718),
    "tokenizer_config.json": (
        "34ddbd64a4cd3f2d9d8a9120d3662d0bf91baead",
        25,
    ),
}
PRODUCTION_OUTPUT = ROOT / "runtime/runs/b200_1epoch_len9600_v1"
STAGE_REQUIREMENTS = {
    "cpu-tests": ("environment",),
    "probe": ("asset", "environment"),
    "tokenizer": ("asset", "environment", "kernel"),
    "preflight": ("asset", "environment", "kernel", "tokenizer"),
    "train-smoke": ("asset", "environment", "kernel", "tokenizer", "preflight"),
    "infer-smoke": ("asset", "environment", "kernel", "tokenizer", "preflight"),
    "certify-smoke": ("asset", "environment", "kernel", "tokenizer", "preflight"),
    "production": (
        "asset",
        "environment",
        "kernel",
        "tokenizer",
        "preflight",
        "smoke",
        "approval",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expect(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise RuntimeError(f"{label} mismatch: expected={expected!r} actual={actual!r}")


def regular_fixed(path: Path, expected: Path, label: str) -> Path:
    if path != expected:
        raise RuntimeError(f"{label} path changed: {path} != {expected}")
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be a regular non-symlink file: {path}")
    resolved = path.resolve(strict=True)
    if resolved != expected or ROOT not in resolved.parents:
        raise RuntimeError(f"{label} escaped the owned root: {resolved}")
    return resolved


def read_receipt(kind: str) -> tuple[Path, dict[str, Any]]:
    path = regular_fixed(RECEIPTS[kind], RECEIPTS[kind], f"{kind} receipt")
    value = b200_runtime.read_json_no_duplicates(path)
    if not isinstance(value, dict):
        raise RuntimeError(f"{kind} receipt is not a JSON object")
    return path, value


def identity(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256(path)}


def current_config() -> dict[str, Any]:
    return b200_runtime.b200_config_identity(b200_runtime.B200_CONFIG_DIR)


def current_pipeline() -> dict[str, Any]:
    return pipeline_code_identity(config_dir=b200_runtime.B200_CONFIG_DIR)


def validate_metric_cache() -> None:
    """Validate the sealed cache structure without rehashing the 2.2 GB weight."""

    directories = {
        "blobs",
        "refs",
        "snapshots",
        f"snapshots/{METRIC_REVISION}",
    }
    regular_files = {"refs/main"} | {
        f"blobs/{blob}" for blob, _size in METRIC_FILES.values()
    }
    links = {
        f"snapshots/{METRIC_REVISION}/{name}": f"../../blobs/{blob}"
        for name, (blob, _size) in METRIC_FILES.items()
    }
    if METRIC_CACHE_ROOT.is_symlink() or not METRIC_CACHE_ROOT.is_dir():
        raise RuntimeError(f"metric cache root is invalid: {METRIC_CACHE_ROOT}")
    actual_directories: set[str] = set()
    actual_regular: set[str] = set()
    actual_links: dict[str, str] = {}
    for path in METRIC_CACHE_ROOT.rglob("*"):
        relative = path.relative_to(METRIC_CACHE_ROOT).as_posix()
        if path.is_symlink():
            actual_links[relative] = os.readlink(path)
        elif path.is_dir():
            actual_directories.add(relative)
        elif path.is_file():
            actual_regular.add(relative)
        else:
            raise RuntimeError(f"metric cache contains unsupported entry: {path}")
    expect("metric cache directories", actual_directories, directories)
    expect("metric cache regular files", actual_regular, regular_files)
    expect("metric cache links", actual_links, links)
    metric_ref = METRIC_CACHE_ROOT / "refs/main"
    expect("metric cache refs/main", metric_ref.read_text().strip(), METRIC_REVISION)
    snapshot = METRIC_CACHE_ROOT / "snapshots" / METRIC_REVISION
    for name, (blob, size) in METRIC_FILES.items():
        blob_path = METRIC_CACHE_ROOT / "blobs" / blob
        link_path = snapshot / name
        if blob_path.is_symlink() or not blob_path.is_file():
            raise RuntimeError(f"metric cache blob is invalid: {blob_path}")
        expect(f"metric cache {name} bytes", blob_path.stat().st_size, size)
        if link_path.resolve(strict=True) != blob_path:
            raise RuntimeError(f"metric cache snapshot target changed: {link_path}")


def validate_asset(value: dict[str, Any]) -> None:
    for key, expected in (
        ("schema", "solar_open2_b200_asset_integrity"),
        ("schema_version", 1),
        ("status", "PASS"),
        ("mode", "full"),
        ("root", str(ROOT)),
        ("user", "e202"),
        ("large_hashes_checked", True),
        ("b200_production_authorized", False),
    ):
        expect(f"asset receipt {key}", value.get(key), expected)
    for key in ("bf16", "training_nf4", "inference_nf4"):
        record = value.get(key)
        if not isinstance(record, dict) or record.get("status") != "PASS":
            raise RuntimeError(f"asset receipt {key} did not pass")
    for key in ("training_nf4", "inference_nf4"):
        record = value[key]
        expect(
            f"asset receipt {key} large hashes",
            record.get("large_hashes_checked"),
            True,
        )
    metric_model = value.get("metric_model")
    if not isinstance(metric_model, dict):
        raise RuntimeError("asset receipt metric_model is missing")
    for key, expected in (
        ("status", "PASS"),
        ("model", "xlm-roberta-large"),
        ("revision", METRIC_REVISION),
        ("files", len(METRIC_FILES)),
        ("payload_file_bytes", sum(size for _blob, size in METRIC_FILES.values())),
        ("large_weight_hash_checked", True),
    ):
        expect(f"asset receipt metric_model {key}", metric_model.get(key), expected)
    validate_metric_cache()


def validate_environment(value: dict[str, Any]) -> None:
    for key, expected in (
        ("schema", "solar_open2_b200_environment"),
        ("schema_version", 1),
        ("status", "PASS"),
        ("root", str(ROOT)),
        ("user", "e202"),
        ("secrets_recorded", False),
        ("b200_production_authorized", False),
    ):
        expect(f"environment receipt {key}", value.get(key), expected)
    python = value.get("python")
    lock = value.get("dependency_lock")
    torch = value.get("torch")
    libraries = value.get("native_libraries")
    if not all(isinstance(item, dict) for item in (python, lock, torch, libraries)):
        raise RuntimeError("environment receipt nested contract changed")
    expect(
        "environment sys.prefix", python.get("sys_prefix"), str(b200_runtime.VENV_DIR)
    )
    lock_path = regular_fixed(
        Path(str(lock.get("path") or "")),
        PROJECT / "requirements/solar-open-2.lock.txt",
        "dependency lock",
    )
    expect("environment lock sha256", lock.get("sha256"), sha256(lock_path))
    if "sm_100" not in (torch.get("compiled_arch_flags") or []):
        raise RuntimeError("environment receipt lacks sm_100")
    cuda128 = libraries.get("libbitsandbytes_cuda128")
    if not isinstance(cuda128, dict):
        raise RuntimeError("environment receipt lacks CUDA 12.8 bitsandbytes")
    library = Path(str(cuda128.get("path") or ""))
    if library.name != "libbitsandbytes_cuda128.so":
        raise RuntimeError("environment receipt references the wrong bnb library")
    if (
        library.is_symlink()
        or not library.is_file()
        or b200_runtime.VENV_DIR not in library.parents
    ):
        raise RuntimeError("environment bnb library is not an owned venv file")
    expect("environment bnb sha256", cuda128.get("sha256"), sha256(library))


def validate_config_pipeline(value: dict[str, Any], label: str) -> None:
    expect(f"{label} config", value.get("config"), current_config())
    expect(f"{label} pipeline", value.get("pipeline_code"), current_pipeline())


def validate_hardware(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} hardware is missing")
    for key, expected in (
        ("status", "PASS"),
        ("physical_cuda_visible_devices", "6,7"),
        ("logical_gpu_count", 2),
        ("boundary_layers", [24]),
    ):
        expect(f"{label} hardware {key}", value.get(key), expected)
    if "sm_100" not in (value.get("torch_arch_list") or []):
        raise RuntimeError(f"{label} hardware lacks sm_100")
    devices = value.get("logical_to_physical")
    if not isinstance(devices, list) or len(devices) != 2:
        raise RuntimeError(f"{label} hardware mapping changed")
    for logical, physical in enumerate((6, 7)):
        device = devices[logical]
        if not isinstance(device, dict):
            raise RuntimeError(f"{label} device record changed")
        for key, expected in (
            ("logical_index", logical),
            ("logical_device", f"cuda:{logical}"),
            ("physical_index", physical),
            ("compute_capability", [10, 0]),
        ):
            expect(f"{label} device {logical} {key}", device.get(key), expected)
        if "B200" not in str(device.get("name") or "").upper():
            raise RuntimeError(f"{label} logical cuda:{logical} is not B200")
        if int(device.get("total_memory_bytes") or 0) < 160 * 1024**3:
            raise RuntimeError(
                f"{label} logical cuda:{logical} has insufficient memory"
            )


def validate_kernel(value: dict[str, Any]) -> None:
    for key, expected in (
        ("schema", "solar_open2_b200_kernel_probe"),
        ("schema_version", 1),
        ("status", "PASS"),
        ("profile", "b200"),
        ("root", str(ROOT)),
        ("user", "e202"),
        ("production_authorized", False),
        ("physical_cuda_visible_devices", "6,7"),
        ("logical_device_mapping", {"cuda:0": 6, "cuda:1": 7}),
    ):
        expect(f"kernel receipt {key}", value.get(key), expected)
    validate_config_pipeline(value, "kernel receipt")
    expect(
        "kernel environment identity",
        {key: value.get("environment", {}).get(key) for key in ("path", "sha256")},
        identity(RECEIPTS["environment"]),
    )
    validate_hardware(value.get("hardware"), "kernel receipt")
    bnb = value.get("bitsandbytes_runtime")
    if not isinstance(bnb, dict):
        raise RuntimeError("kernel receipt lacks bitsandbytes runtime")
    for key, expected in (
        ("status", "PASS"),
        ("compute_capability", [10, 0]),
        ("manifest_status", "SEALED"),
        ("bitsandbytes_version", "0.49.2"),
        ("native_library", "libbitsandbytes_cuda128.so"),
    ):
        expect(f"kernel bnb {key}", bnb.get(key), expected)


def validate_tokenizer(value: dict[str, Any]) -> None:
    for key, expected in (
        ("schema", "solar_open2_b200_tokenizer_contract"),
        ("schema_version", 1),
        ("status", "PASS"),
        ("profile", "b200"),
        ("root", str(ROOT)),
        ("user", "e202"),
    ):
        expect(f"tokenizer receipt {key}", value.get(key), expected)
    validate_config_pipeline(value, "tokenizer receipt")
    data = value.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("tokenizer receipt lacks data evidence")
    expect(
        "tokenizer opened roles",
        data.get("opened_dataset_roles"),
        ["train", "validation"],
    )
    expect(
        "tokenizer sealed roles", data.get("sealed_evaluation_dataset_roles_opened"), []
    )
    if value.get("full_validation_prompt_budget", {}).get("truncation_count") != 0:
        raise RuntimeError("tokenizer receipt reports validation truncation")
    prompt_budget = value.get("full_validation_prompt_budget")
    if not isinstance(prompt_budget, dict):
        raise RuntimeError("tokenizer receipt lacks full validation evidence")
    validate_b200_input_unicode_inventory(
        prompt_budget.get("input_unicode_inventory"),
        expected_rows=17_850,
        expected_anomalies=B200_VALIDATION_UNICODE_EXPECTED,
    )
    for field in ("invalid_unicode_count", "target_invalid_unicode_count"):
        expect(f"tokenizer receipt {field}", prompt_budget.get(field), 0)


def validate_preflight(value: dict[str, Any]) -> None:
    for key, expected in (
        ("schema", "solar_open2_b200_preflight"),
        ("schema_version", 1),
        ("status", "PASS"),
        ("profile", "b200"),
        ("root", str(ROOT)),
        ("user", "e202"),
        ("physical_cuda_visible_devices", "6,7"),
        ("production_authorized", False),
    ):
        expect(f"preflight receipt {key}", value.get(key), expected)
    validate_config_pipeline(value, "preflight receipt")
    expect(
        "preflight environment identity",
        {key: value.get("environment", {}).get(key) for key in ("path", "sha256")},
        identity(RECEIPTS["environment"]),
    )
    if value.get("transfer_integrity", {}).get("large_hashes_checked") is not True:
        raise RuntimeError("preflight lacks full transfer hash evidence")
    validate_hardware(value.get("hardware"), "preflight receipt")


def validate_smoke(value: dict[str, Any]) -> None:
    for key, expected in (
        ("schema", "solar_open2_b200_smoke_certification"),
        ("schema_version", 2),
        ("status", "PASS"),
        ("technical_production_gate_passed", True),
        ("authorization_scope", "technical_smoke_gate_only"),
        ("human_production_approval_required", True),
        ("human_production_approval_recorded", False),
        ("production_authorized", False),
        ("profile", "b200"),
        ("root", str(ROOT)),
        ("user", "e202"),
    ):
        expect(f"smoke receipt {key}", value.get(key), expected)
    validate_config_pipeline(value, "smoke receipt")
    inputs = value.get("inputs")
    if not isinstance(inputs, dict):
        raise RuntimeError("smoke receipt lacks input identities")
    for key, kind in (
        ("environment_receipt", "environment"),
        ("asset_integrity_receipt", "asset"),
        ("kernel_probe_receipt", "kernel"),
        ("tokenizer_contract_receipt", "tokenizer"),
        ("preflight_receipt", "preflight"),
    ):
        expect(f"smoke input {key}", inputs.get(key), identity(RECEIPTS[kind]))
    performance = b200_runtime.b200_training_performance_contract()
    activation = b200_runtime.b200_selective_gradient_checkpointing_contract()
    expect("smoke training performance", value.get("training_performance"), performance)
    expect(
        "smoke training performance SHA",
        value.get("training_performance_sha256"),
        canonical_sha256(performance),
    )
    expect(
        "smoke activation contract",
        value.get("gradient_checkpointing_contract"),
        activation,
    )
    expect(
        "smoke activation contract SHA",
        value.get("gradient_checkpointing_contract_sha256"),
        canonical_sha256(activation),
    )
    checks = value.get("checks")
    if not isinstance(checks, dict):
        raise RuntimeError("smoke receipt lacks checks")
    expect(
        "smoke selective training activation check",
        checks.get("selective_training_activation_policy"),
        "PASS",
    )
    for field in ("smoke_peak_allocated_bytes", "smoke_peak_reserved_bytes"):
        peak = value.get(field)
        if (
            not isinstance(peak, dict)
            or set(peak) != {"0", "1"}
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0
                for item in peak.values()
            )
        ):
            raise RuntimeError(f"smoke receipt has invalid {field}: {peak!r}")


def validate_approval(value: dict[str, Any]) -> None:
    smoke_path = RECEIPTS["smoke"]
    approval_path = RECEIPTS["approval"]
    if approval_path.stat().st_mtime_ns <= smoke_path.stat().st_mtime_ns:
        raise RuntimeError("production approval is not newer than the smoke receipt")
    config = current_config()
    pipeline = current_pipeline()
    expected = {
        "schema": "solar-open2-b200-production-approval",
        "schema_version": 2,
        "status": "PASS",
        "authorization_scope": "b200_production_training",
        "human_production_approval_recorded": True,
        "technical_smoke_receipt": identity(smoke_path),
        "config_identity": config,
        "pipeline_code_sha256": pipeline["sha256"],
        "model_contract_sha256": config["files"]["model"]["sha256"],
        "recipe_sha256": config["files"]["recipe"]["sha256"],
        "data_contract_sha256": config["files"]["data_contract"]["sha256"],
        "training_performance": b200_runtime.b200_training_performance_contract(),
        "training_performance_sha256": canonical_sha256(
            b200_runtime.b200_training_performance_contract()
        ),
        "gradient_checkpointing_contract_sha256": canonical_sha256(
            b200_runtime.b200_selective_gradient_checkpointing_contract()
        ),
        "output_dir": str(PRODUCTION_OUTPUT),
        "wandb": {
            "mode": "online",
            "entity": "junhyuk-tak-markr",
            "project": "modern-korean-lm",
            "run_id": "solar_open2_b200_1epoch_20260802_v2",
            "run_name": ("Solar Open2 250B 2xB200 1epoch len9600 optimized v2"),
            "resume": "allow",
        },
        "runtime_override": b200_runtime.b200_runtime_override_evidence(),
    }
    expect("production approval", value, expected)


VALIDATORS = {
    "asset": validate_asset,
    "environment": validate_environment,
    "kernel": validate_kernel,
    "tokenizer": validate_tokenizer,
    "preflight": validate_preflight,
    "smoke": validate_smoke,
    "approval": validate_approval,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGE_REQUIREMENTS), required=True)
    args = parser.parse_args()
    expect("launcher user", pwd.getpwuid(os.geteuid()).pw_name, "e202")
    expect("launcher root", b200_runtime.require_owned_root(), ROOT)
    results: dict[str, Any] = {}
    for kind in STAGE_REQUIREMENTS[args.stage]:
        path, value = read_receipt(kind)
        VALIDATORS[kind](value)
        results[kind] = identity(path)
    print(
        json.dumps(
            {"status": "PASS", "stage": args.stage, "receipts": results}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"B200 launcher receipt gate FAIL: {error}", file=sys.stderr)
        raise SystemExit(1) from None
