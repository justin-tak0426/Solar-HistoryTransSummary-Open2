#!/usr/bin/env python3
"""Offline verification for the pinned Solar-Open2 Nota NVFP4 asset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/NHNHOME/WORKSPACE/26mfds002_B/jhtak/solar-open-2")
REVISION = "de88f6226788077e2d340204fd79d37720c9eda0"
MODEL_ID = "nota-ai/Solar-Open2-250B-Nota-NVFP4"
MODEL_DIR = ROOT / f"runtime/models/Solar-Open2-250B-Nota-NVFP4-{REVISION}"
BASE_DIR = (
    ROOT
    / "runtime/hf_cache/models--upstage--Solar-Open2-250B/snapshots"
    / "8d1dda5ad2af58207c65348d825772d2f4d72481"
)
ADAPTER_DIR = ROOT / "runtime/runs/b200_1epoch_len9600_v1/best_adapter"
TREE_PATH = MODEL_DIR / f".cache/huggingface/trees/{REVISION}.json"

EXPECTED = {
    "tree_sha256": "5ebc91e71c5e2b0b8ddee9baf8425712c2625ef7cccc5abeb6537a7482959596",
    "config_sha256": "a5045fecbefc924382b821edb362532881c80876c70acb3c9239b8891e0de2dc",
    "index_sha256": "ada21368497b36a82031d0674f99475f73cb025c7d2f3e5b7a31e5c08755581f",
    "quantization_sha256": "5de240e7a6f0178b7aadccc64cca40875eb0a14dfce5269e8fe3f4f1dcd2d3ca",
    "tokenizer_sha256": "c7b7cb72447bbe1f8d856adaf34933acfdedfa89c7fe1d95d0fdabedf9d19a94",
    "adapter_config_sha256": "7d8ffd34fc3737a69644aa923f3bbc53360d675a02fa651f287c3114f7a94b39",
    "adapter_weights_sha256": "34de50b7dee9e8baa98d48c8d84be11c984f6d922ae871c1d9c320f266a83846",
    "weight_files": 29,
    "indexed_tensor_bytes": 153_288_243_200,
    "weight_file_bytes": 153_311_254_120,
    "adapter_tensor_count": 1_056,
    "adapter_module_count": 528,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise RuntimeError(f"duplicate JSON key {key!r}: {path}")
            value[key] = item
        return value

    result = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
    )
    if not isinstance(result, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return result


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


def adapter_modules() -> tuple[list[str], int]:
    from safetensors import safe_open

    weights = ADAPTER_DIR / "adapter_model.safetensors"
    with safe_open(weights, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    modules: set[str] = set()
    for key in keys:
        prefix = "base_model.model."
        if not key.startswith(prefix) or not key.endswith(
            (".lora_A.weight", ".lora_B.weight")
        ):
            raise RuntimeError(f"unexpected adapter tensor key: {key}")
        name = key[len(prefix) :]
        name = name.removesuffix(".lora_A.weight").removesuffix(".lora_B.weight")
        modules.add(name)
    return sorted(modules), len(keys)


def verify() -> dict[str, Any]:
    for path in (MODEL_DIR, BASE_DIR, ADAPTER_DIR):
        if path.is_symlink() or not path.is_dir():
            raise RuntimeError(f"required directory missing or symlinked: {path}")
    if sha256(TREE_PATH) != EXPECTED["tree_sha256"]:
        raise RuntimeError("pinned Hugging Face repository tree changed")
    tree = read_json(TREE_PATH)
    files = tree.get("files")
    if not isinstance(files, dict) or len(files) != 42:
        raise RuntimeError("repository tree must contain exactly 42 files")
    missing, wrong_size, symlinks = [], {}, []
    for name, metadata in files.items():
        path = MODEL_DIR / name
        if not path.is_file():
            missing.append(name)
            continue
        if path.is_symlink():
            symlinks.append(name)
        expected_size = int(metadata["size"])
        if path.stat().st_size != expected_size:
            wrong_size[name] = {
                "expected": expected_size,
                "actual": path.stat().st_size,
            }
        download_metadata = (
            MODEL_DIR / ".cache/huggingface/download" / f"{name}.metadata"
        )
        if not download_metadata.is_file():
            raise RuntimeError(f"download provenance missing: {name}")
        recorded_revision = download_metadata.read_text(encoding="utf-8").splitlines()[
            0
        ]
        if recorded_revision != REVISION:
            raise RuntimeError(
                f"download revision changed for {name}: {recorded_revision}"
            )
    if missing or wrong_size or symlinks:
        raise RuntimeError(
            f"asset coverage failed: missing={missing} wrong_size={wrong_size} symlinks={symlinks}"
        )

    checks = {
        "config_sha256": sha256(MODEL_DIR / "config.json"),
        "index_sha256": sha256(MODEL_DIR / "model.safetensors.index.json"),
        "quantization_sha256": sha256(MODEL_DIR / "quantization_config.json"),
        "tokenizer_sha256": sha256(MODEL_DIR / "tokenizer.json"),
        "adapter_config_sha256": sha256(ADAPTER_DIR / "adapter_config.json"),
        "adapter_weights_sha256": sha256(ADAPTER_DIR / "adapter_model.safetensors"),
    }
    for key, actual in checks.items():
        if actual != EXPECTED[key]:
            raise RuntimeError(f"{key} changed: {actual} != {EXPECTED[key]}")
    if sha256(BASE_DIR / "tokenizer.json") != EXPECTED["tokenizer_sha256"]:
        raise RuntimeError("NVFP4 and certified BF16 tokenizer identities differ")

    config = read_json(MODEL_DIR / "config.json")
    quantization = read_json(MODEL_DIR / "quantization_config.json")
    if config.get("architectures") != ["SolarOpen2ForCausalLM"]:
        raise RuntimeError("NVFP4 architecture changed")
    wanted_config = {
        "model_type": "solar_open2",
        "vocab_size": 196_608,
        "num_hidden_layers": 48,
        "n_routed_experts": 320,
        "num_experts_per_tok": 8,
    }
    for key, wanted in wanted_config.items():
        if config.get(key) != wanted:
            raise RuntimeError(f"NVFP4 config changed: {key}={config.get(key)!r}")
    if quantization.get("quant_method") != "compressed-tensors":
        raise RuntimeError("NVFP4 quantization method changed")
    if quantization.get("format") != "nvfp4-pack-quantized":
        raise RuntimeError("NVFP4 packed format changed")

    index = read_json(MODEL_DIR / "model.safetensors.index.json")
    weight_files = sorted(set(index.get("weight_map", {}).values()))
    weight_bytes = sum((MODEL_DIR / name).stat().st_size for name in weight_files)
    if len(weight_files) != EXPECTED["weight_files"]:
        raise RuntimeError("NVFP4 weight shard count changed")
    if index.get("metadata", {}).get("total_size") != EXPECTED["indexed_tensor_bytes"]:
        raise RuntimeError("NVFP4 indexed tensor byte count changed")
    if weight_bytes != EXPECTED["weight_file_bytes"]:
        raise RuntimeError("NVFP4 weight file byte count changed")

    modules, tensor_count = adapter_modules()
    ignored = set(quantization.get("ignore") or [])
    unquantized = sorted(set(modules) - ignored)
    if (
        tensor_count != EXPECTED["adapter_tensor_count"]
        or len(modules) != EXPECTED["adapter_module_count"]
    ):
        raise RuntimeError("adapter tensor/module coverage changed")
    if unquantized:
        raise RuntimeError(
            f"LoRA target modules were quantized by NVFP4: {unquantized[:8]}"
        )

    return {
        "schema": "solar-open2-b200-vllm-asset",
        "schema_version": 1,
        "status": "PASS",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "revision": REVISION,
        "model_dir": str(MODEL_DIR),
        "repository_tree": {
            "path": str(TREE_PATH),
            "sha256": EXPECTED["tree_sha256"],
            "files": len(files),
        },
        "hashes": checks,
        "weight_files": len(weight_files),
        "indexed_tensor_bytes": EXPECTED["indexed_tensor_bytes"],
        "weight_file_bytes": weight_bytes,
        "tokenizer_identical_to_certified_bf16": True,
        "quantization": {
            "method": "compressed-tensors",
            "format": "nvfp4-pack-quantized",
        },
        "adapter": {
            "path": str(ADAPTER_DIR),
            "tensor_count": tensor_count,
            "module_count": len(modules),
            "all_target_modules_unquantized": True,
        },
        "verification_policy": "HF pinned-tree revision plus exact sizes and pinned small-file/adapter SHA-256",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = verify()
    if args.receipt:
        atomic_json(args.receipt, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
