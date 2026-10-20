"""Isolated, opt-in accelerators for the sealed Solar Open2 NF4 runtime.

Nothing in this package is imported by the production evaluation pipeline.  The
helpers are deliberately opt-in so they can be qualified in a disposable smoke
process before a new evaluation or serving run uses them.
"""

from .adaptive_batch import (
    ContiguousTokenBudgetPolicy,
    OOMBackoffReceipt,
    run_with_contiguous_oom_backoff,
)
from .expert_dispatch import install_host_dedup_dispatch
from .lora_merge import merge_and_validate_for_disposable_process

__all__ = [
    "ContiguousTokenBudgetPolicy",
    "OOMBackoffReceipt",
    "install_host_dedup_dispatch",
    "merge_and_validate_for_disposable_process",
    "run_with_contiguous_oom_backoff",
]
