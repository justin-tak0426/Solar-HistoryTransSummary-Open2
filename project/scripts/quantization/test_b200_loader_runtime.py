#!/usr/bin/env python3
"""CPU-mocked contracts for B200 NF4 loading and relocated checkpoints."""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import torch

from . import bnb_nf4, checkpoint, full_model_loader

try:
    from ..b200_runtime import (
        B200_CONFIG_DIR,
        BF16_SOURCE_DIR,
        TRAIN_EXPERT_DIR,
    )
except ImportError:
    from b200_runtime import (
        B200_CONFIG_DIR,
        BF16_SOURCE_DIR,
        TRAIN_EXPERT_DIR,
    )


def _quantize_4bit(
    A,
    blocksize,
    compress_statistics,
    quant_type,
    quant_storage,
):
    return A


def _dequantize_4bit(A, quant_state):
    return A


class _FakeParams4bit:
    @staticmethod
    def from_prequantized(data, quantized_stats, device, module):
        return data


class _FakeLinear4bit:
    pass


def _fake_bnb(native_library: Path, *, compiled_with_cuda: bool = True):
    functional = types.SimpleNamespace(
        quantize_4bit=_quantize_4bit,
        dequantize_4bit=_dequantize_4bit,
        QuantState=object,
    )
    nn = types.SimpleNamespace(
        Linear4bit=_FakeLinear4bit,
        Params4bit=_FakeParams4bit,
    )
    native = types.SimpleNamespace(
        compiled_with_cuda=compiled_with_cuda,
        _lib=types.SimpleNamespace(_name=str(native_library)),
    )
    return types.SimpleNamespace(
        functional=functional,
        nn=nn,
        cextension=types.SimpleNamespace(lib=native),
    )


@contextmanager
def _mock_bnb_runtime(
    native_library: Path,
    *,
    capability: tuple[int, int],
    compiled_with_cuda: bool = True,
):
    fake_bnb = _fake_bnb(
        native_library,
        compiled_with_cuda=compiled_with_cuda,
    )
    with ExitStack() as stack:
        stack.enter_context(patch.dict(sys.modules, {"bitsandbytes": fake_bnb}))
        stack.enter_context(
            patch.object(
                bnb_nf4.importlib.metadata,
                "version",
                return_value="0.49.2",
            )
        )
        stack.enter_context(patch.object(torch.cuda, "is_available", return_value=True))
        stack.enter_context(
            patch.object(
                torch.cuda,
                "get_device_capability",
                return_value=capability,
            )
        )
        yield fake_bnb


class NF4RuntimeProfileTests(unittest.TestCase):
    def test_conversion_profile_still_rejects_sm100(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            native = Path(directory) / "libbitsandbytes_cuda128.so"
            native.touch()
            with _mock_bnb_runtime(native, capability=(10, 0)):
                with self.assertRaisesRegex(RuntimeError, "SM120"):
                    bnb_nf4.require_bitsandbytes(require_cuda=True)

    def test_conversion_profile_accepts_sm120(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            native = Path(directory) / "unused.so"
            native.touch()
            with _mock_bnb_runtime(native, capability=(12, 0)):
                loaded_torch, _loaded_bnb = bnb_nf4.require_bitsandbytes(
                    require_cuda=True
                )
            self.assertIs(loaded_torch, torch)

    def test_sealed_profile_accepts_exact_sm100_cuda128(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            native = Path(directory) / "libbitsandbytes_cuda128.so"
            native.touch()
            with _mock_bnb_runtime(native, capability=(10, 0)):
                with bnb_nf4.use_nf4_runtime_profile(
                    bnb_nf4.B200_SEALED_RUNTIME_PROFILE,
                    manifest_status="SEALED",
                ):
                    _torch, _bnb = bnb_nf4.require_bitsandbytes(
                        require_cuda=True
                    )

    def test_sealed_profile_rejects_cpu_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            native = Path(directory) / "libbitsandbytes_cpu.so"
            native.touch()
            with _mock_bnb_runtime(
                native,
                capability=(10, 0),
                compiled_with_cuda=False,
            ):
                with self.assertRaisesRegex(RuntimeError, "CPU fallback"):
                    bnb_nf4.require_bitsandbytes(
                        require_cuda=True,
                        runtime_profile=bnb_nf4.B200_SEALED_RUNTIME_PROFILE,
                        manifest_status="SEALED",
                    )

    def test_sealed_profile_requires_sealed_manifest(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "SEALED"):
            with bnb_nf4.use_nf4_runtime_profile(
                bnb_nf4.B200_SEALED_RUNTIME_PROFILE,
                manifest_status="IN_PROGRESS",
            ):
                pass


class SourceRuntimeProfileTests(unittest.TestCase):
    def test_b200_profile_does_not_lookup_mk03(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_root = Path(directory) / "models--upstage--Solar-Open2-250B"
            source = model_root / "snapshots" / ("a" * 40)
            source.mkdir(parents=True)
            blobs = model_root / "blobs"
            blobs.mkdir()
            with patch.object(
                checkpoint.pwd,
                "getpwnam",
                side_effect=AssertionError("B200 runtime looked up mk03"),
            ):
                allowed, owner_uid = checkpoint._source_shard_access_policy(
                    source,
                    runtime_profile=checkpoint.B200_RUNTIME_SOURCE_PROFILE,
                )
            self.assertEqual(allowed, blobs.resolve())
            self.assertEqual(owner_uid, os.geteuid())

    def test_conversion_profile_retains_mk03_ownership(self) -> None:
        account = types.SimpleNamespace(pw_uid=1234)
        with patch.object(checkpoint.pwd, "getpwnam", return_value=account) as lookup:
            with patch.object(checkpoint.os, "geteuid", return_value=1234):
                _allowed, owner_uid = checkpoint._source_shard_access_policy(
                    Path("/irrelevant"),
                    runtime_profile=checkpoint.CONVERSION_SOURCE_PROFILE,
                )
        lookup.assert_called_once_with("mk03")
        self.assertEqual(owner_uid, 1234)


class B200LoaderConfigTests(unittest.TestCase):
    def test_training_loader_defaults_match_sealed_config(self) -> None:
        source, experts, identity = full_model_loader._resolve_b200_loader_inputs(
            source_dir=None,
            expert_checkpoint_dir=None,
            config_dir=B200_CONFIG_DIR,
        )
        self.assertEqual(source, BF16_SOURCE_DIR)
        self.assertEqual(experts, TRAIN_EXPERT_DIR)
        self.assertEqual(identity["config_dir"], str(B200_CONFIG_DIR))


if __name__ == "__main__":
    unittest.main()
