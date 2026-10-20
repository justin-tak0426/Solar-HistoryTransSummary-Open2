"""Order-preserving batching and fail-closed OOM recovery for HF generation.

The production evaluator currently uses a fixed batch of two.  This module does
not alter that live run.  It provides deterministic building blocks for a new
run to try a larger batch while preserving row order and to split a batch before
any output is committed when CUDA reports an out-of-memory condition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Generic, Iterable, Sequence, TypeVar


RowT = TypeVar("RowT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class PlannedBatch:
    start: int
    stop: int
    size: int
    padded_prompt_tokens: int
    maximum_prompt_tokens: int


@dataclass(frozen=True)
class ContiguousTokenBudgetPolicy:
    """Build contiguous batches without changing canonical evaluation order.

    ``max_padded_prompt_tokens`` limits ``batch_size * max(prompt_length)``.
    It protects the prefill activation peak; decode growth remains guarded by
    OOM backoff because Transformers' dynamic cache grows to the *actual*, not
    merely configured, continuation length.
    """

    max_batch_size: int = 4
    max_padded_prompt_tokens: int = 16_384

    def __post_init__(self) -> None:
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        if self.max_padded_prompt_tokens < 1:
            raise ValueError("max_padded_prompt_tokens must be positive")

    def plan(self, prompt_lengths: Sequence[int]) -> list[PlannedBatch]:
        if any(isinstance(length, bool) or length < 1 for length in prompt_lengths):
            raise ValueError("prompt lengths must be positive integers")
        batches: list[PlannedBatch] = []
        start = 0
        while start < len(prompt_lengths):
            stop = start
            maximum = 0
            while stop < len(prompt_lengths) and stop - start < self.max_batch_size:
                candidate_maximum = max(maximum, prompt_lengths[stop])
                candidate_size = stop - start + 1
                candidate_tokens = candidate_maximum * candidate_size
                if candidate_size > 1 and candidate_tokens > self.max_padded_prompt_tokens:
                    break
                maximum = candidate_maximum
                stop += 1
            if stop == start:
                # A single canonical row cannot be split further.
                stop += 1
                maximum = prompt_lengths[start]
            size = stop - start
            batches.append(
                PlannedBatch(
                    start=start,
                    stop=stop,
                    size=size,
                    padded_prompt_tokens=maximum * size,
                    maximum_prompt_tokens=maximum,
                )
            )
            start = stop
        return batches


@dataclass
class OOMBackoffReceipt:
    attempted_batch_sizes: list[int]
    successful_batch_sizes: list[int]
    oom_batch_sizes: list[int]

    def as_dict(self) -> dict[str, list[int]]:
        return asdict(self)


def _validate_batch_result(rows: Sequence[RowT], results: Sequence[ResultT]) -> None:
    if len(results) != len(rows):
        raise RuntimeError(
            f"generation result count changed: {len(results)} != {len(rows)}"
        )


def run_with_contiguous_oom_backoff(
    rows: Sequence[RowT],
    generate: Callable[[Sequence[RowT]], Sequence[ResultT]],
    *,
    is_oom: Callable[[BaseException], bool],
    clear_after_oom: Callable[[], None],
) -> tuple[list[ResultT], OOMBackoffReceipt]:
    """Generate all rows, recursively halving only failed contiguous batches.

    Callers must invoke this before writing any result from the attempted batch.
    Non-OOM exceptions propagate unchanged.  A one-row OOM also propagates,
    avoiding an infinite retry loop and preserving fail-closed behavior.
    """

    receipt = OOMBackoffReceipt([], [], [])

    def run(part: Sequence[RowT]) -> list[ResultT]:
        receipt.attempted_batch_sizes.append(len(part))
        try:
            values = list(generate(part))
        except BaseException as error:
            if not is_oom(error) or len(part) == 1:
                raise
            receipt.oom_batch_sizes.append(len(part))
            clear_after_oom()
            midpoint = len(part) // 2
            return run(part[:midpoint]) + run(part[midpoint:])
        _validate_batch_result(part, values)
        receipt.successful_batch_sizes.append(len(part))
        return values

    if not rows:
        return [], receipt
    results = run(rows)
    _validate_batch_result(rows, results)
    return results, receipt


def flatten_planned_rows(
    rows: Sequence[RowT], batches: Iterable[PlannedBatch]
) -> list[RowT]:
    """Testing/audit helper proving that a plan retained exact input order."""

    flattened: list[RowT] = []
    for batch in batches:
        flattened.extend(rows[batch.start : batch.stop])
    return flattened
