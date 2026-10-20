#!/usr/bin/env python3
"""Resume and verify the pinned official BF16 snapshot in the HF cache."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from common import CONFIG_DIR, atomic_json, read_json, require_owned_path, sha256
from quantization.checkpoint import read_safetensors_header
from quantization.contract import (
    BASE_MODEL_ID,
    BASE_REVISION,
    EXPECTED_SOURCE_SHARD_COUNT,
    EXPECTED_SOURCE_TENSOR_COUNT,
)

MAX_STABLE_HUB_WORKERS = 4
XET_FIXED_DOWNLOAD_CONCURRENCY = 8
XET_MAX_CONCURRENT_FILE_DOWNLOADS = 4


def configure_stable_download(requested_workers: int) -> int:
    if requested_workers < 1:
        raise ValueError("--max-workers must be positive")
    effective_workers = min(requested_workers, MAX_STABLE_HUB_WORKERS)
    # Avoid multiplying Hub file workers by Xet's high-performance adaptive
    # concurrency. The 250B snapshot otherwise creates enough simultaneous
    # signed-URL requests to leave every transfer waiting after URL refreshes.
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "0"
    os.environ["HF_XET_FIXED_DOWNLOAD_CONCURRENCY"] = str(
        XET_FIXED_DOWNLOAD_CONCURRENCY
    )
    os.environ["HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS"] = str(
        XET_MAX_CONCURRENT_FILE_DOWNLOADS
    )
    return effective_workers


def unique_regular_file_usage(root: Path) -> tuple[int, int]:
    seen: set[tuple[int, int]] = set()
    total = 0
    count = 0
    for directory, _subdirs, filenames in os.walk(root):
        for filename in filenames:
            path = Path(directory) / filename
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity in seen:
                continue
            seen.add(identity)
            total += stat.st_size
            count += 1
    return total, count


def validate_snapshot(snapshot: Path) -> dict[str, object]:
    model_contract = read_json(CONFIG_DIR / "model.json")
    if snapshot.name != BASE_REVISION or snapshot.parent.name != "snapshots":
        raise RuntimeError(f"snapshot is not pinned to {BASE_REVISION}: {snapshot}")
    required = (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "LICENSE",
    )
    for name in required:
        if not (snapshot / name).is_file():
            raise FileNotFoundError(snapshot / name)
    if sha256(snapshot / "LICENSE") != (
        "de031be88349e197e9ae20b7051679f55eb50ca53f7ffb2362343f8516b02d66"
    ):
        raise RuntimeError("downloaded Upstage Solar License checksum changed")
    config = read_json(snapshot / "config.json")
    architecture = model_contract["architecture_contract"]
    exact = {
        "model_type": "solar_open2",
        "architectures": ["SolarOpen2ForCausalLM"],
        **{
            key: architecture[key]
            for key in (
                "hidden_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "vocab_size",
                "max_position_embeddings",
                "n_routed_experts",
                "n_shared_experts",
                "num_experts_per_tok",
                "moe_intermediate_size",
                "gqa_layers",
                "kda_use_full_proj",
            )
        },
    }
    for key, expected in exact.items():
        if config.get(key) != expected:
            raise RuntimeError(
                f"official config changed: {key}={config.get(key)!r}, "
                f"expected={expected!r}"
            )
    index = read_json(snapshot / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    metadata = index.get("metadata")
    if not isinstance(weight_map, dict) or not isinstance(metadata, dict):
        raise RuntimeError("official model index is malformed")
    if len(weight_map) != EXPECTED_SOURCE_TENSOR_COUNT:
        raise RuntimeError(
            f"source tensor count changed: {len(weight_map)} "
            f"!= {EXPECTED_SOURCE_TENSOR_COUNT}"
        )
    shards = sorted(set(weight_map.values()))
    if len(shards) != EXPECTED_SOURCE_SHARD_COUNT:
        raise RuntimeError(
            f"source shard count changed: {len(shards)} "
            f"!= {EXPECTED_SOURCE_SHARD_COUNT}"
        )
    indexed_by_shard: dict[str, set[str]] = defaultdict(set)
    for key, shard in weight_map.items():
        if Path(shard).name != shard:
            raise RuntimeError(f"unsafe shard name in index: {shard!r}")
        indexed_by_shard[shard].add(key)
    logical_weight_bytes = 0
    observed_keys: set[str] = set()
    source_shards = []
    for position, shard in enumerate(shards, 1):
        path = snapshot / shard
        if not path.is_file():
            raise FileNotFoundError(path)
        records = read_safetensors_header(path)
        if set(records) != indexed_by_shard[shard]:
            raise RuntimeError(f"index/header mismatch: {shard}")
        if observed_keys & set(records):
            raise RuntimeError(f"duplicate tensor keys across shards: {shard}")
        observed_keys.update(records)
        payload_bytes = sum(record.nbytes for record in records.values())
        logical_weight_bytes += payload_bytes
        digest = sha256(path)
        if path.is_symlink():
            blob_name = path.resolve().name
            if re.fullmatch(r"[0-9a-f]{64}", blob_name) and digest != blob_name:
                raise RuntimeError(
                    f"Hugging Face LFS blob digest mismatch: {shard}"
                )
        source_shards.append(
            {
                "filename": shard,
                "file_bytes": path.stat().st_size,
                "payload_bytes": payload_bytes,
                "tensor_count": len(records),
                "sha256": digest,
            }
        )
        if position % 10 == 0:
            print(f"verified_shards={position}/{len(shards)}", flush=True)
    if observed_keys != set(weight_map):
        raise RuntimeError("header verification did not cover the complete index")
    if metadata.get("total_size") != logical_weight_bytes:
        raise RuntimeError(
            "index total_size differs from verified tensor payload: "
            f"{metadata.get('total_size')} != {logical_weight_bytes}"
        )
    if logical_weight_bytes != model_contract["base"]["indexed_tensor_bytes"]:
        raise RuntimeError(
            f"official BF16 size changed: {logical_weight_bytes} != "
            f"{model_contract['base']['indexed_tensor_bytes']}"
        )
    snapshot_files = [path for path in snapshot.iterdir() if path.is_file()]
    snapshot_logical_bytes = sum(path.stat().st_size for path in snapshot_files)
    return {
        "repo_id": BASE_MODEL_ID,
        "requested_revision": BASE_REVISION,
        "repo_sha": snapshot.name,
        "snapshot_dir": str(snapshot),
        "snapshot_file_count": len(snapshot_files),
        "snapshot_logical_bytes": snapshot_logical_bytes,
        "indexed_shards": len(shards),
        "indexed_tensors": len(weight_map),
        "verified_tensor_bytes": logical_weight_bytes,
        "source_shards": source_shards,
        "contract_hashes": {
            name: sha256(snapshot / name)
            for name in required
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(
            "/data1/jhtak/Solar-HistoryTransSummary-Open2/hf_cache"
        ),
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path(
            "/data1/jhtak/Solar-HistoryTransSummary-Open2/receipts/bf16_download.json"
        ),
    )
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    if getpass.getuser() != "mk03":
        raise RuntimeError("BF16 download is restricted to mk03")
    cache_dir = require_owned_path(args.cache_dir)
    receipt = require_owned_path(args.receipt, create_parent=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    effective_workers = configure_stable_download(args.max_workers)
    print(
        json.dumps(
            {
                "download_runtime": {
                    "requested_hub_workers": args.max_workers,
                    "effective_hub_workers": effective_workers,
                    "xet_high_performance": False,
                    "xet_fixed_download_concurrency": (
                        XET_FIXED_DOWNLOAD_CONCURRENCY
                    ),
                    "xet_max_concurrent_file_downloads": (
                        XET_MAX_CONCURRENT_FILE_DOWNLOADS
                    ),
                }
            },
            sort_keys=True,
        ),
        flush=True,
    )
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=BASE_MODEL_ID,
            revision=BASE_REVISION,
            repo_type="model",
            cache_dir=cache_dir,
            local_dir=None,
            max_workers=effective_workers,
        )
    ).resolve()
    expected = (
        cache_dir
        / "models--upstage--Solar-Open2-250B/snapshots"
        / BASE_REVISION
    ).resolve()
    if snapshot != expected:
        raise RuntimeError(
            f"snapshot_download used an unexpected path: {snapshot} != {expected}"
        )
    result = validate_snapshot(snapshot)
    used_bytes, cache_file_count = unique_regular_file_usage(cache_dir)
    result.update(
        {
            "status": "PASS",
            "cache_dir": str(cache_dir.resolve()),
            "cache_used_bytes": used_bytes,
            "cache_regular_file_count": cache_file_count,
            "local_dir_copy": False,
            "resumable_hf_cache": True,
        }
    )
    atomic_json(receipt, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
