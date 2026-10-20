#!/usr/bin/env python3
"""Read-only source and NF4 sidecar accounting CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .checkpoint import (
    inventory_source_checkpoint,
    validate_transformers_schema,
)
from .contract import DEFAULT_OUTPUT_DIR, DEFAULT_SOURCE_DIR
from .layer_flat_checkpoint import preflight_layer_flat_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--expert-checkpoint-dir", type=Path)
    parser.add_argument("--hash-source-shards", action="store_true")
    parser.add_argument("--hash-output-shards", action="store_true")
    parser.add_argument(
        "--header-only",
        action="store_true",
        help="skip official-fork meta schema; never production eligible",
    )
    parser.add_argument(
        "--default-expert-checkpoint",
        action="store_true",
        help=f"also validate {DEFAULT_OUTPUT_DIR}",
    )
    args = parser.parse_args()
    inventory = inventory_source_checkpoint(
        args.source_dir,
        hash_shards=args.hash_source_shards,
        require_revision_path=True,
    )
    result = {
        "status": "PASS",
        "source": inventory.summary(),
        "source_shard_hashes_verified": args.hash_source_shards,
        "production_eligible": False,
    }
    if not args.header_only:
        result["official_transformers_schema"] = validate_transformers_schema(
            inventory
        )
    checkpoint = args.expert_checkpoint_dir
    if args.default_expert_checkpoint:
        if checkpoint is not None:
            parser.error(
                "use either --expert-checkpoint-dir or "
                "--default-expert-checkpoint"
            )
        checkpoint = DEFAULT_OUTPUT_DIR
    if checkpoint is not None:
        result["layer_flat_expert_checkpoint"] = (
            preflight_layer_flat_checkpoint(
                checkpoint,
                hash_shards=args.hash_output_shards,
            )
        )
        result["output_shard_hashes_verified"] = args.hash_output_shards
    result["production_eligible"] = bool(
        not args.header_only
        and args.hash_source_shards
        and checkpoint is not None
        and args.hash_output_shards
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
