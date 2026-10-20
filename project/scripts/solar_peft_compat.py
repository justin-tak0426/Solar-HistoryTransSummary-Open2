#!/usr/bin/env python3
"""Narrow PEFT compatibility guard for native Transformers-v5 Solar Open 2."""

from __future__ import annotations

import importlib.metadata
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


_PATCH_LOCK = threading.RLock()


@contextmanager
def solar_open2_peft_v5_native() -> Iterator[None]:
    """Prevent PEFT from treating native Solar experts as a v4 MoE checkpoint.

    PEFT 0.19.1 maps ``solar_open2`` to the legacy ``qwen2_moe`` conversion.
    That conversion calls ``set(regex_string)`` and corrupts a regex
    ``target_modules`` contract into a set of individual characters. Solar
    Open 2 is already instantiated in its native Transformers-v5 layout, so
    that compatibility conversion must not run during adapter injection/load.
    """

    version = importlib.metadata.version("peft")
    if version != "0.19.1":
        raise RuntimeError(f"unaudited PEFT version: {version} != 0.19.1")
    from peft.utils import transformers_weight_conversion as conversion

    with _PATCH_LOCK:
        original = conversion.get_checkpoint_conversion_mapping

        def guarded(model_type: str) -> Any:
            if model_type == "solar_open2":
                return None
            return original(model_type)

        conversion.get_checkpoint_conversion_mapping = guarded
        try:
            yield
        finally:
            conversion.get_checkpoint_conversion_mapping = original

