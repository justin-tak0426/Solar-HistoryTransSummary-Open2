"""Streaming NF4 conversion support for Solar Open 2 routed experts.

This package intentionally produces an experts-only sidecar checkpoint.  It is
not a standalone Hugging Face model checkpoint.
"""

from .contract import (
    ARCHITECTURE,
    BASE_MODEL_ID,
    BASE_REVISION,
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE_DIR,
    ArchitectureContract,
)

__all__ = [
    "ARCHITECTURE",
    "BASE_MODEL_ID",
    "BASE_REVISION",
    "DEFAULT_ARTIFACT_ROOT",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_SOURCE_DIR",
    "ArchitectureContract",
]
