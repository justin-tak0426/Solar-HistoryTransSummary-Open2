#!/usr/bin/env python3
"""CPU-only B200 branch tests for shared path and evaluator contracts."""

from __future__ import annotations

import unittest
from pathlib import Path

from common import (
    B200_CONFIG_DIR,
    B200_ROOT,
    OWNED_ROOTS,
    REFERENCE_COPY,
    load_reference_module,
    require_owned_path,
)


class B200SharedContractTest(unittest.TestCase):
    def test_b200_root_is_owned_and_legacy_tmp_is_not(self) -> None:
        legacy_tmp = Path("/").joinpath("tmp", "mk03")
        self.assertIn(B200_ROOT.resolve(), tuple(OWNED_ROOTS))
        self.assertNotIn(legacy_tmp, tuple(OWNED_ROOTS))
        candidate = B200_ROOT / "runtime/tmp/common-contract-test"
        self.assertEqual(require_owned_path(candidate), candidate.resolve())
        with self.assertRaisesRegex(ValueError, "outside owned roots"):
            require_owned_path(legacy_tmp / "not-owned")

    def test_b200_evaluator_loads_owned_copy_only(self) -> None:
        module = load_reference_module(config_dir=B200_CONFIG_DIR)
        self.assertIsNotNone(module)
        self.assertTrue(REFERENCE_COPY.is_file())

    def test_default_evaluator_no_longer_requires_source_server(self) -> None:
        module = load_reference_module()
        self.assertIsNotNone(module)


if __name__ == "__main__":
    unittest.main()
