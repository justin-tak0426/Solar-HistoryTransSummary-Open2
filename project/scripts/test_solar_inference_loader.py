#!/usr/bin/env python3
"""CPU-only B200 config contracts for the Solar inference loader."""

from __future__ import annotations

import unittest

from b200_runtime import (
    B200_CONFIG_DIR,
    BF16_SOURCE_DIR,
    INFERENCE_EXPERT_DIR,
)
from solar_inference_loader import _resolve_b200_inference_inputs


class B200InferenceLoaderTests(unittest.TestCase):
    def test_defaults_match_pinned_source_and_per_expert_artifact(self) -> None:
        source, experts, identity = _resolve_b200_inference_inputs(
            source_dir=None,
            expert_checkpoint_dir=None,
            config_dir=B200_CONFIG_DIR,
        )
        self.assertEqual(source, BF16_SOURCE_DIR)
        self.assertEqual(experts, INFERENCE_EXPERT_DIR)
        self.assertEqual(identity["config_dir"], str(B200_CONFIG_DIR))


if __name__ == "__main__":
    unittest.main()
