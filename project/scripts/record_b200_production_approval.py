#!/usr/bin/env python3
"""Atomically record the user's exact B200 production and runtime approval."""

from __future__ import annotations

import argparse
from pathlib import Path

from b200_runtime import B200_CONFIG_DIR, atomic_json, require_owned_path
from common import sha256
from train import (
    b200_wandb_identity,
    expected_b200_production_approval,
    load_training_config,
    training_pipeline_code_identity,
    validate_production_smoke,
)


ROOT = Path("/NHNHOME/WORKSPACE/26mfds002_B/jhtak/solar-open-2")
EXPECTED_OUTPUT = ROOT / "runtime/runs/b200_1epoch_len9600_v1"
EXPECTED_RECEIPT = ROOT / "runtime/receipts/b200_production_approval_v1.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=EXPECTED_RECEIPT)
    args = parser.parse_args()

    smoke_path = require_owned_path(args.smoke_receipt, must_exist=True)
    destination = require_owned_path(args.output)
    if destination != EXPECTED_RECEIPT:
        raise RuntimeError(f"approval receipt path changed: {destination}")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    config = load_training_config(B200_CONFIG_DIR)
    validate_production_smoke(
        smoke_path,
        source_dir=Path(config["model"]["training_runtime"]["source_dir"]),
        expert_nf4_dir=Path(config["model"]["training_runtime"]["expert_dir"]),
        training_config=config,
    )
    value = expected_b200_production_approval(
        smoke_receipt_path=smoke_path,
        smoke_receipt_sha256=sha256(smoke_path),
        training_config=config,
        pipeline_identity=training_pipeline_code_identity(config),
        output_dir=EXPECTED_OUTPUT,
        wandb_identity=b200_wandb_identity(smoke=False, mode="online"),
    )
    atomic_json(destination, value)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
