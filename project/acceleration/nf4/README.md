# Existing NF4 inference acceleration experiment

This directory is deliberately outside the hashed production pipeline.  It does
not change or import itself into the running GPU 6/7 evaluation.

## What can be accelerated without converting weights

1. **Merge the trained LoRA in a disposable process.**  The tuned runtime
   currently executes 528 LoRA target modules separately.  Merging removes the
   low-rank branches from every decode token.  `lora_merge.py` requires PEFT's
   safe merge and exact greedy token-ID parity on a caller-provided probe before
   reporting PASS.  A fresh full evaluation must be started after qualification;
   merged and unmerged rows must never be mixed.
2. **Raise batch size conservatively.**  `adaptive_batch.py` builds contiguous,
   source-order-preserving batches and recursively halves an OOM batch before
   any rows are written.  For the first GPU smoke, try max batch 4 (not 8), keep
   the existing output limits, and compare rows/hour plus output-token/second.
3. **Avoid the CUDA unique kernel.**  `expert_dispatch.py` copies the tiny router
   vector and performs sorted host deduplication.  It preserves the canonical
   expert accumulation order and removes one CUDA unique/sort kernel per routed
   layer call.  It still has one host synchronization per routed layer.

## Hard blocker

The sealed fallback has 320 separately stored bitsandbytes `Linear4bit` experts
per layer.  Python must know the routed expert IDs to choose module objects, so
the GPU-to-CPU synchronization cannot be removed while retaining this layout.
A packed/fused MoE implementation (for example the separately investigated
Solar-aware vLLM path) or a weight conversion is required for the large gain.

## Qualification gate

- Run the CPU tests in this directory.
- Verify the pre-acceleration baseline receipt with `baseline_guard.py`.
- In a disposable two-GPU process, capture exact greedy token IDs for a fixed
  translation/summarization probe, safe-merge LoRA, rerun, and require equality.
- Benchmark batch 2 versus batch 4 on the same prompt/output token workload.
- Require output-contract audit PASS and no OOM before using a new complete run.
- Do not modify, append to, or resume the existing live result with a different
  runtime implementation.

Expected gains remain hypotheses until that GPU smoke: LoRA merge is likely the
largest safe improvement; host dedup is a micro-optimization; batch 4 can improve
throughput but does not reduce single-request autoregressive latency.
