"""Verify that isolated acceleration work did not mutate baseline files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_baseline(root: Path, receipt_path: Path) -> dict[str, object]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    changed: list[dict[str, object]] = []
    for record in receipt.get("files", []):
        relative = record.get("path")
        expected = record.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise RuntimeError("baseline receipt contains an invalid file record")
        path = root / relative
        actual = sha256_file(path) if path.is_file() else None
        if actual != expected:
            changed.append({"path": relative, "expected": expected, "actual": actual})
    return {
        "status": "PASS" if not changed else "FAIL",
        "checked_files": len(receipt.get("files", [])),
        "baseline_inventory_sha256": receipt.get("inventory_sha256"),
        "changed": changed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = verify_baseline(args.root.resolve(), args.receipt.resolve())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
