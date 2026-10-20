#!/usr/bin/env python3
"""Resumable TP2 vLLM inference for the pinned base or epoch-1 adapter."""

from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import statistics
import sys
import time
from pathlib import Path
from typing import Any


if __package__:
    from . import contract
    from .verify_asset import verify as verify_asset
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import contract  # type: ignore[no-redef]
    from verify_asset import verify as verify_asset  # type: ignore[no-redef]


REQUEUE_EXIT_CODE = 75


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


def require_runtime(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "6,7":
        raise RuntimeError("vLLM evaluation requires CUDA_VISIBLE_DEVICES=6,7 exactly")
    if args.tensor_parallel_size != 2:
        raise RuntimeError("Solar-Open2 B200 vLLM evaluation requires TP=2")
    if Path(sys.prefix).resolve() != contract.VLLM_ENV.resolve():
        raise RuntimeError(
            f"vLLM evaluation requires {contract.VLLM_ENV}, found {Path(sys.prefix).resolve()}"
        )
    if hasattr(os, "getuid") and os.getuid() != 2943:
        raise RuntimeError(f"vLLM evaluation requires uid 2943, found {os.getuid()}")
    import torch

    if torch.cuda.device_count() != 2:
        raise RuntimeError(
            f"expected exactly two visible GPUs, found {torch.cuda.device_count()}"
        )
    devices = []
    for logical in range(2):
        props = torch.cuda.get_device_properties(logical)
        if props.name != "NVIDIA B200" or (props.major, props.minor) != (10, 0):
            raise RuntimeError(
                f"logical GPU {logical} is not the required B200: {props}"
            )
        if props.total_memory < 160 * 1024**3:
            raise RuntimeError(f"logical GPU {logical} has insufficient memory")
        devices.append(
            {
                "logical_index": logical,
                "physical_index": 6 + logical,
                "name": props.name,
                "compute_capability": [props.major, props.minor],
                "total_memory_bytes": props.total_memory,
            }
        )
    return {
        "physical_cuda_visible_devices": "6,7",
        "tensor_parallel_size": 2,
        "devices": devices,
    }


def engine_memory_snapshot() -> list[dict[str, int]]:
    import torch

    result = []
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        result.append(
            {
                "logical_index": index,
                "free_bytes": int(free),
                "used_bytes": int(total - free),
                "total_bytes": int(total),
            }
        )
    return result


def make_record(
    *,
    row: dict[str, Any],
    identity: dict[str, Any],
    result: dict[str, Any],
    contract_sha256: str,
    args: argparse.Namespace,
    timing: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": row["id"],
        "task": row["task"],
        "model_label": args.model_label,
        "model_dir": str(contract.MODEL_DIR),
        "source_dir": str(contract.MODEL_DIR),
        "adapter_dir": str(contract.ADAPTER_DIR) if args.variant == "tuned" else None,
        "origin": row["origin"],
        "reference": row["response"],
        "reference_source": row.get("reference_source"),
        "prompt_variant": "no_dictionary_context",
        "reasoning_effort": "none",
        "evaluation_split": identity["split"],
        "evaluation_row_identity": identity,
        "inference_contract_sha256": contract_sha256,
        "backend": "upstage-vllm-solar-open2-nvfp4",
        "backend_timing": timing,
        **result,
    }


def generate(args: argparse.Namespace) -> int:
    if args.max_model_len != contract.MAX_MODEL_LEN:
        raise RuntimeError("max model length contract changed")
    if args.translation_max_new_tokens != contract.TASK_MAX_NEW_TOKENS["translation"]:
        raise RuntimeError("translation generation cap changed")
    if (
        args.summarization_max_new_tokens
        != contract.TASK_MAX_NEW_TOKENS["summarization"]
    ):
        raise RuntimeError("summarization generation cap changed")
    if not 0.50 <= args.gpu_memory_utilization <= 0.95:
        raise ValueError("gpu-memory-utilization must be in [0.50, 0.95]")
    if args.request_chunk_size < args.max_num_seqs:
        raise ValueError("request-chunk-size must be at least max-num-seqs")

    runtime_identity = require_runtime(args)
    asset = verify_asset()
    if asset.get("status") != "PASS":
        raise RuntimeError("NVFP4 asset verification failed")
    receipt = contract.read_json(contract.ASSET_RECEIPT)
    if (
        receipt.get("status") != "PASS"
        or receipt.get("revision") != contract.MODEL_REVISION
    ):
        raise RuntimeError("NVFP4 asset receipt is missing or stale")

    eval_path = args.eval_jsonl.resolve(strict=True)
    output_path = args.output_jsonl.resolve(strict=False)
    try:
        output_path.relative_to(contract.ROOT)
    except ValueError as error:
        raise RuntimeError("output JSONL must remain under the project root") from error
    if output_path.is_symlink() or output_path.is_dir():
        raise RuntimeError(f"invalid output path: {output_path}")
    rows, evaluation, identities = contract.prepare_rows(
        eval_path,
        mode=args.mode,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        contract.TOKENIZER_DIR,
        use_fast=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    generation_config = contract.read_json(
        contract.TOKENIZER_DIR / "generation_config.json"
    )
    if set(generation_config.get("eos_token_id") or []) != contract.EOS_TOKEN_IDS:
        raise RuntimeError("Solar EOS token contract changed")

    prompt_ids: dict[tuple[str, str], list[int]] = {}
    for row in rows:
        maximum = contract.TASK_MAX_NEW_TOKENS[str(row["task"])]
        prompt_ids[contract.row_key(row)] = contract.render_prompt_ids(
            tokenizer, row, max_new_tokens=maximum
        )

    runtime = {
        **runtime_identity,
        "engine": {
            "runner": "generate",
            "dtype": "bfloat16",
            "quantization": "compressed-tensors/nvfp4-pack-quantized",
            "enable_expert_parallel": args.enable_expert_parallel,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "fully_sharded_loras": args.fully_sharded_loras,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "request_chunk_size": args.request_chunk_size,
            "enforce_eager": args.enforce_eager,
            "custom_all_reduce_enabled": True,
            "distributed_executor_backend": "mp",
        },
    }
    inference_contract = contract.build_contract(
        variant=args.variant,
        model_label=args.model_label,
        evaluation=evaluation,
        runtime=runtime,
    )
    contract_digest = contract.canonical_sha256(inference_contract)
    manifest_path, manifest, done = contract.initialize_or_resume(
        output_path,
        contract=inference_contract,
        rows=rows,
        identities=identities,
        resume=args.resume,
    )
    if manifest["status"] == "COMPLETE":
        print(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "already_done": len(done),
                    "output": str(output_path),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0

    remaining = [row for row in rows if contract.row_key(row) not in done]
    print(
        json.dumps(
            {
                "status": "PREFLIGHT_PASS",
                "variant": args.variant,
                "model_label": args.model_label,
                "rows": len(rows),
                "already_done": len(done),
                "remaining": len(remaining),
                "prompt_tokens_max": max(map(len, prompt_ids.values())),
                "contract_sha256": contract_digest,
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if not remaining:
        benchmark = manifest.get("benchmark") or {"resumed_complete": True}
        contract.seal_manifest(
            manifest_path,
            manifest,
            output_jsonl=output_path,
            completed=len(done),
            benchmark=benchmark,
        )
        return 0

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import RequestOutputKind

    load_started = time.monotonic()
    llm = LLM(
        model=str(contract.MODEL_DIR),
        tokenizer=str(contract.TOKENIZER_DIR),
        runner="generate",
        dtype="bfloat16",
        trust_remote_code=False,
        tensor_parallel_size=2,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_expert_parallel=args.enable_expert_parallel,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        enable_lora=args.variant == "tuned",
        max_loras=1,
        max_cpu_loras=1,
        max_lora_rank=32,
        lora_dtype="bfloat16",
        fully_sharded_loras=args.fully_sharded_loras,
        enforce_eager=args.enforce_eager,
        disable_custom_all_reduce=False,
        distributed_executor_backend="mp",
        seed=42,
    )
    load_seconds = time.monotonic() - load_started
    lora_request = None
    if args.variant == "tuned":
        lora_request = LoRARequest(
            "b200_epoch1",
            1,
            str(contract.ADAPTER_DIR),
            base_model_name=str(contract.MODEL_DIR),
        )
    print(
        json.dumps(
            {
                "status": "MODEL_READY",
                "load_seconds": load_seconds,
                "memory": engine_memory_snapshot(),
            }
        ),
        flush=True,
    )

    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(
            "[signal] graceful stop requested; finishing current request chunk",
            flush=True,
        )

    signal.signal(signal.SIGUSR1, request_stop)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_mode = "a" if output_path.exists() else "w"
    generation_seconds = 0.0
    generated_rows = 0
    generated_prompt_tokens = 0
    generated_output_tokens = 0
    ttft_seconds: list[float] = []
    tpot_seconds: list[float] = []
    _, common = contract.load_reference_modules()

    with output_path.open(output_mode, encoding="utf-8") as output:
        for task in contract.TASK_ORDER:
            task_rows = [row for row in remaining if row["task"] == task]
            maximum = contract.TASK_MAX_NEW_TOKENS[task]
            sampling = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                seed=42,
                max_tokens=maximum,
                stop_token_ids=sorted(contract.EOS_TOKEN_IDS),
                ignore_eos=False,
                detokenize=False,
                skip_special_tokens=False,
                output_kind=RequestOutputKind.FINAL_ONLY,
            )
            for start in range(0, len(task_rows), args.request_chunk_size):
                batch = task_rows[start : start + args.request_chunk_size]
                batch_prompts = [
                    {"prompt_token_ids": prompt_ids[contract.row_key(row)]}
                    for row in batch
                ]
                chunk_started = time.monotonic()
                outputs = llm.generate(
                    batch_prompts,
                    sampling_params=sampling,
                    lora_request=lora_request,
                    use_tqdm=True,
                )
                chunk_seconds = time.monotonic() - chunk_started
                if len(outputs) != len(batch):
                    raise RuntimeError(
                        "vLLM output coverage differs from request batch"
                    )
                generation_seconds += chunk_seconds
                for row, request_output in zip(batch, outputs, strict=True):
                    if len(request_output.outputs) != 1:
                        raise RuntimeError("vLLM returned non-single completion")
                    completion = request_output.outputs[0]
                    key = contract.row_key(row)
                    audited = contract.audit_completion(
                        tokenizer,
                        common,
                        row=row,
                        prompt_ids=prompt_ids[key],
                        token_ids=list(completion.token_ids),
                        finish_reason=completion.finish_reason,
                        stop_reason=completion.stop_reason,
                        max_new_tokens=maximum,
                    )
                    metrics = request_output.metrics
                    ttft = float(metrics.first_token_latency) if metrics else None
                    tpot = None
                    if metrics and len(completion.token_ids) > 1:
                        elapsed = float(metrics.last_token_ts - metrics.first_token_ts)
                        tpot = elapsed / (len(completion.token_ids) - 1)
                    if ttft is not None and ttft >= 0:
                        ttft_seconds.append(ttft)
                    if tpot is not None and tpot >= 0:
                        tpot_seconds.append(tpot)
                    timing = {
                        "ttft_seconds": ttft,
                        "mean_tpot_seconds": tpot,
                        "chunk_wall_seconds": chunk_seconds,
                        "chunk_rows": len(batch),
                        "num_cached_prompt_tokens": request_output.num_cached_tokens,
                    }
                    record = make_record(
                        row=row,
                        identity=identities[key],
                        result=audited,
                        contract_sha256=contract_digest,
                        args=args,
                        timing=timing,
                    )
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    done.add(key)
                    generated_rows += 1
                    generated_prompt_tokens += audited["prompt_tokens"]
                    generated_output_tokens += audited["prediction_tokens"]
                output.flush()
                os.fsync(output.fileno())
                contract.update_manifest(manifest_path, manifest, len(done))
                print(
                    json.dumps(
                        {
                            "status": "IN_PROGRESS",
                            "generated_this_run": generated_rows,
                            "completed": len(done),
                            "expected": len(rows),
                            "chunk_rows": len(batch),
                            "chunk_seconds": chunk_seconds,
                            "output_tokens_per_second": (
                                sum(len(item.outputs[0].token_ids) for item in outputs)
                                / chunk_seconds
                            ),
                        }
                    ),
                    flush=True,
                )
                if stop_requested:
                    return REQUEUE_EXIT_CODE

    benchmark = {
        "status": "PASS",
        "load_seconds": load_seconds,
        "generation_wall_seconds": generation_seconds,
        "generated_rows_this_run": generated_rows,
        "generated_prompt_tokens_this_run": generated_prompt_tokens,
        "generated_output_tokens_this_run": generated_output_tokens,
        "rows_per_second": generated_rows / generation_seconds,
        "output_tokens_per_second": generated_output_tokens / generation_seconds,
        "ttft_seconds_mean": statistics.mean(ttft_seconds) if ttft_seconds else None,
        "ttft_seconds_p50": percentile(ttft_seconds, 0.50),
        "ttft_seconds_p95": percentile(ttft_seconds, 0.95),
        "tpot_seconds_mean": statistics.mean(tpot_seconds) if tpot_seconds else None,
        "tpot_seconds_p50": percentile(tpot_seconds, 0.50),
        "tpot_seconds_p95": percentile(tpot_seconds, 0.95),
        "memory_at_completion": engine_memory_snapshot(),
    }
    contract.seal_manifest(
        manifest_path,
        manifest,
        output_jsonl=output_path,
        completed=len(done),
        benchmark=benchmark,
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "model_label": args.model_label,
                "completed": len(done),
                "output_jsonl": str(output_path),
                "manifest": str(manifest_path),
                "benchmark": benchmark,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    del llm
    gc.collect()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "production"), required=True)
    parser.add_argument("--variant", choices=("base", "tuned"), required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=contract.MAX_MODEL_LEN)
    parser.add_argument("--translation-max-new-tokens", type=int, default=8_192)
    parser.add_argument("--summarization-max-new-tokens", type=int, default=6_144)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32_768)
    parser.add_argument("--request-chunk-size", type=int, default=256)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--enable-expert-parallel", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--fully-sharded-loras", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    return generate(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
