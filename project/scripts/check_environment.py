#!/usr/bin/env python3
"""Fail closed unless the complete locked Solar environment is local and healthy."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from common import PROJECT, atomic_json, canonical_sha256, require_owned_path, sha256


ENVIRONMENT = (PROJECT / "solar-open-2").resolve()
LOCK = PROJECT / "requirements/solar-open-2.lock.txt"
CORE_MODULES = (
    "torch",
    "transformers",
    "bitsandbytes",
    "accelerate",
    "peft",
    "fla",
    "wandb",
    "bert_score",
    "sacrebleu",
    "rouge_score",
    "mecab_ko",
    "datasets",
    "trl",
)
TRANSFORMERS_COMMIT = "ef5c37485091057070b36dd0620798f74988eb01"


def check(*, required_cuda_count: int) -> dict[str, Any]:
    prefix = Path(sys.prefix).resolve()
    base_prefix = Path(sys.base_prefix).resolve()
    if prefix != ENVIRONMENT or base_prefix != ENVIRONMENT:
        raise RuntimeError(
            f"Solar environment is not self-contained: "
            f"prefix={prefix} base_prefix={base_prefix} expected={ENVIRONMENT}"
        )
    module_paths: dict[str, str] = {}
    versions: dict[str, str] = {}
    for name in CORE_MODULES:
        module = importlib.import_module(name)
        path = Path(str(module.__file__)).resolve()
        if ENVIRONMENT not in path.parents:
            raise RuntimeError(f"external module import is forbidden: {name}={path}")
        module_paths[name] = str(path)
        try:
            versions[name] = importlib.metadata.version(
                "fla-core" if name == "fla" else name.replace("_", "-")
            )
        except importlib.metadata.PackageNotFoundError:
            versions[name] = str(getattr(module, "__version__", "unknown"))

    transformers_distribution = importlib.metadata.distribution("transformers")
    direct_url_text = transformers_distribution.read_text("direct_url.json")
    if not direct_url_text:
        raise RuntimeError("Upstage Transformers direct_url provenance is missing")
    direct_url = json.loads(direct_url_text)
    if (
        direct_url.get("url") != "https://github.com/upstageAI/transformers.git"
        or direct_url.get("vcs_info", {}).get("commit_id") != TRANSFORMERS_COMMIT
    ):
        raise RuntimeError("Upstage Transformers commit provenance changed")

    freeze = subprocess.run(
        [str(ENVIRONMENT / "bin/python"), "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    locked = LOCK.read_text(encoding="utf-8").splitlines()
    if sorted(freeze) != sorted(locked):
        missing = sorted(set(locked) - set(freeze))
        unexpected = sorted(set(freeze) - set(locked))
        raise RuntimeError(
            f"environment differs from lock: missing={missing[:8]} "
            f"unexpected={unexpected[:8]}"
        )
    pip_check = subprocess.run(
        [str(ENVIRONMENT / "bin/python"), "-m", "pip", "check"],
        check=True,
        capture_output=True,
        text=True,
    )

    torch = importlib.import_module("torch")
    observed_cuda_count = int(torch.cuda.device_count())
    if observed_cuda_count != required_cuda_count:
        raise RuntimeError(
            f"CUDA device count changed: {observed_cuda_count} "
            f"!= {required_cuda_count}"
        )
    return {
        "status": "PASS",
        "environment": str(ENVIRONMENT),
        "sys_prefix": str(prefix),
        "sys_base_prefix": str(base_prefix),
        "python_version": sys.version,
        "lock_path": str(LOCK),
        "lock_sha256": sha256(LOCK),
        "freeze_sha256": canonical_sha256(sorted(freeze)),
        "pip_check": pip_check.stdout.strip(),
        "module_paths": module_paths,
        "versions": versions,
        "transformers_direct_url": direct_url,
        "required_cuda_count": required_cuda_count,
        "observed_cuda_count": observed_cuda_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda-count", type=int, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.require_cuda_count < 0:
        parser.error("--require-cuda-count must be non-negative")
    result = check(required_cuda_count=args.require_cuda_count)
    receipt = require_owned_path(args.receipt, create_parent=True)
    atomic_json(receipt, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
