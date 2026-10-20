#!/usr/bin/env python3
"""One-file no-dictionary inference and evaluation for Gemma4-compatible models.

The prompt contract is embedded in this file. Inference refuses any JSONL row
whose messages are not exactly reconstructible from only ``task`` and
``origin``. Run ``infer`` once per model and ``score`` on the resulting files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FORBIDDEN_NONEMPTY_FIELDS = (
    "vocab_text", "matched_label", "matched_term_pairs", "matched_hard_terms",
    "matched_full_terms", "matched_candidate_terms", "target_terms",
)
_SPACE_RE = re.compile(r"\s+")


def clean_text(value: Any) -> str:
    return _SPACE_RE.sub(" ", str(value or "").strip())


def nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, dict, tuple, set)):
        return bool(value)
    return str(value).strip() not in {"", "None", "null"}


def build_messages_no_dict(task: str, origin: str) -> list[dict[str, str]]:
    system = (
        "You are a professional specialist in translating and summarizing "
        "pre-modern East Asian historical records into contemporary Korean. "
        "Use only information explicitly present in the source text. "
        "Do not mention metadata, reasoning, or these instructions. "
        "Return Korean body text only."
    )
    if task == "translation":
        final_prompt = (
            "Translate the source into contemporary Korean.\n"
            "Return only the translated Korean text.\n"
            "Use a formal documentary style. Do not add labels, notes, explanations, markdown, bullets, reasoning, or framing.\n\n"
            f"Source:\n{origin}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": "Translate the source into contemporary Korean.\nReturn only the translated Korean text.\n\nSource:\n明治43年9月27日 西平 田中亨 小野篤次郞"},
            {"role": "assistant", "content": "명치 43년 9월 27일 서평 다나카 도루, 오노 도쿠지로"},
            {"role": "user", "content": "Translate the source into contemporary Korean.\nReturn only the translated Korean text.\n\nSource:\n釜山理事廳ニ於テ左記土地來十月一日午前十時競賣執行ノ旨通知アリタリ"},
            {"role": "assistant", "content": "부산이사청에서 아래 기재한 토지에 대하여 오는 10월 1일 오전 10시에 경매를 집행한다는 통지가 있었다."},
            {"role": "user", "content": final_prompt},
        ]
    if task == "summarization":
        final_prompt = (
            "Summarize the source in contemporary Korean.\n"
            "Return only the summarized Korean text.\n"
            "Use a formal documentary style. Do not add labels, notes, explanations, markdown, bullets, reasoning, or framing.\n"
            "Compress only information explicitly present in the source.\n\n"
            f"Source:\n{origin}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": "Summarize the source in contemporary Korean.\nReturn only the summarized Korean text.\n\nSource:\n正誤 第三條中「官吏」ハ「職員」ノ誤, 書式第二號ニ「備考」ヲ加フ"},
            {"role": "assistant", "content": "제3조의 「관리」를 「직원」으로 정정하고, 서식 제2호에 「비고」를 추가한다."},
            {"role": "user", "content": final_prompt},
        ]
    raise ValueError(f"unsupported task: {task}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_eval_rows(rows: list[dict[str, Any]], source: str) -> dict[str, Any]:
    seen: set[tuple[str, str]] = set()
    counts: Counter[str] = Counter()
    for index, row in enumerate(rows, 1):
        task, row_id = str(row.get("task") or ""), str(row.get("id") or "")
        key = (row_id, task)
        if not row_id or task not in {"translation", "summarization"}:
            raise ValueError(f"{source}:{index}: invalid id/task {key}")
        if key in seen:
            raise ValueError(f"{source}:{index}: duplicate id/task {key}")
        seen.add(key)
        origin = clean_text(row.get("origin"))
        if row.get("messages") != build_messages_no_dict(task, origin):
            raise ValueError(f"{source}:{index}: prompt is not canonical no-dictionary prompt")
        forbidden = [field for field in FORBIDDEN_NONEMPTY_FIELDS if nonempty(row.get(field))]
        if forbidden:
            raise ValueError(f"{source}:{index}: dictionary metadata is non-empty: {forbidden}")
        if row.get("prompt_variant") != "no_dictionary_context" or not nonempty(row.get("response")):
            raise ValueError(f"{source}:{index}: invalid prompt_variant/reference")
        counts[task] += 1
    return {"status": "PASS", "rows": len(rows), "by_task": dict(counts), "source": source}


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("id")), str(row.get("task"))


def initialize_focus_embeddings(model: Any, init_plan_path: Path) -> dict[str, Any]:
    import torch

    input_embeddings = model.get_input_embeddings().weight.data
    output_module = model.get_output_embeddings()
    output_embeddings = output_module.weight.data if output_module is not None else None
    shared_storage = output_embeddings is not None and output_embeddings.data_ptr() == input_embeddings.data_ptr()
    initialized = skipped = 0
    with init_plan_path.open("r", encoding="utf-8") as handle, torch.no_grad():
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            new_id = int(record.get("new_id", -1))
            old_ids = [int(value) for value in record.get("old_subtoken_ids") or []]
            old_ids = [value for value in old_ids if 0 <= value < input_embeddings.shape[0] and value != new_id]
            if not (0 <= new_id < input_embeddings.shape[0]) or not old_ids:
                skipped += 1
                continue
            input_embeddings[new_id].copy_(input_embeddings[old_ids].mean(dim=0))
            if output_embeddings is not None and not shared_storage and new_id < output_embeddings.shape[0]:
                output_embeddings[new_id].copy_(output_embeddings[old_ids].mean(dim=0))
            initialized += 1
    return {"initialized": initialized, "skipped": skipped, "init_plan": str(init_plan_path)}


def merge(args: argparse.Namespace) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"output_dir is not empty; pass --overwrite: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, local_files_only=True, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model_dir, dtype=torch.bfloat16, device_map="auto",
        local_files_only=True, trust_remote_code=True, low_cpu_mem_usage=True,
    )
    base_vocab_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) != base_vocab_size:
        if not args.focus_init_plan or not args.focus_init_plan.is_file():
            raise ValueError("--focus_init_plan is required for a FOCUS-expanded adapter")
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        initialization = initialize_focus_embeddings(model, args.focus_init_plan)
    else:
        initialization = {"initialized": 0, "reason": "base and tokenizer vocab sizes match"}
    merged = PeftModel.from_pretrained(model, args.adapter_dir, local_files_only=True).merge_and_unload(safe_merge=True)
    merged.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size=args.max_shard_size)
    tokenizer.save_pretrained(args.output_dir)
    for name in ("processor_config.json", "preprocessor_config.json", "chat_template.jinja"):
        source = Path(args.processor_source_dir or args.base_model_dir) / name
        if source.is_file():
            shutil.copy2(source, args.output_dir / name)
    summary = {"status": "PASS", "base_model_dir": args.base_model_dir, "adapter_dir": args.adapter_dir, "output_dir": str(args.output_dir), "focus_initialization": initialization}
    (args.output_dir / "merge_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def infer(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = read_jsonl(args.eval_jsonl)
    if args.limit_per_task > 0:
        counts: Counter[str] = Counter()
        selected = []
        for row in rows:
            task = str(row.get("task"))
            if counts[task] < args.limit_per_task:
                selected.append(row)
                counts[task] += 1
        rows = selected
    print(json.dumps({"no_dictionary_validation": validate_eval_rows(rows, str(args.eval_jsonl))}, ensure_ascii=False), flush=True)

    done = set()
    if args.resume and args.output_jsonl.exists():
        done = {row_key(row) for row in read_jsonl(args.output_jsonl)}
        rows = [row for row in rows if row_key(row) not in done]
    print(json.dumps({"model": args.model_dir, "model_label": args.model_label, "remaining": len(rows), "already_done": len(done), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "")}, ensure_ascii=False), flush=True)
    if not rows:
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = LLM(
        model=args.model_dir, tokenizer=args.model_dir, runner="generate", dtype="bfloat16",
        trust_remote_code=True, tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=args.enforce_eager, disable_custom_all_reduce=True,
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    generated = 0
    with args.output_jsonl.open(mode, encoding="utf-8") as out:
        for task, max_new_tokens in (("translation", args.translation_max_new_tokens), ("summarization", args.summarization_max_new_tokens)):
            task_rows = [row for row in rows if row["task"] == task]
            for start in range(0, len(task_rows), args.generation_batch_size):
                batch = task_rows[start:start + args.generation_batch_size]
                prompts, prompt_lengths = [], []
                max_prompt_tokens = args.max_model_len - max_new_tokens
                for row in batch:
                    prompt = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
                    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                    if len(ids) > max_prompt_tokens:
                        ids = ids[-max_prompt_tokens:]
                        prompt = tokenizer.decode(ids, skip_special_tokens=False)
                    prompts.append(prompt)
                    prompt_lengths.append(len(ids))
                outputs = llm.generate(prompts, SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_new_tokens))
                for row, prompt_tokens, result in zip(batch, prompt_lengths, outputs):
                    prediction = result.outputs[0].text.strip() if result.outputs else ""
                    prediction_tokens = len(result.outputs[0].token_ids) if result.outputs else 0
                    out.write(json.dumps({
                        "id": row["id"], "task": task, "model_label": args.model_label,
                        "model_dir": args.model_dir, "origin": row["origin"],
                        "reference": row["response"], "reference_source": row.get("reference_source"),
                        "prompt_variant": "no_dictionary_context", "prediction": prediction,
                        "prompt_tokens": prompt_tokens, "prediction_tokens": prediction_tokens,
                    }, ensure_ascii=False) + "\n")
                    generated += 1
                out.flush()
                print(f"generated={generated}/{len(rows)} model={args.model_label}", flush=True)


class UnicodeWordTokenizer:
    def tokenize(self, text: str) -> list[str]:
        normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
        return re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)


def score(args: argparse.Namespace) -> None:
    from bert_score import score as bert_score
    from rouge_score import rouge_scorer
    from sacrebleu.metrics import BLEU

    rows = [row for path in args.prediction_jsonl for row in read_jsonl(path)]
    if not rows:
        raise ValueError("no prediction rows")
    seen, references_by_key = set(), {}
    keys_by_model: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in rows:
        model_label = str(row.get("model_label"))
        model_key = (model_label,) + row_key(row)
        if model_key in seen:
            raise ValueError(f"duplicate model/id/task: {model_key}")
        seen.add(model_key)
        if row.get("prediction") is None or not nonempty(row.get("reference")):
            raise ValueError(f"missing prediction/reference: {model_key}")
        ref_key = row_key(row)
        ref_value = (str(row.get("origin") or ""), str(row["reference"]))
        if ref_key in references_by_key and references_by_key[ref_key] != ref_value:
            raise ValueError(f"reference mismatch across models: {ref_key}")
        references_by_key[ref_key] = ref_value
        keys_by_model[model_label].add(ref_key)
        if row.get("prompt_variant") != "no_dictionary_context":
            raise ValueError(f"prediction is not marked no_dictionary_context: {model_key}")
    expected_keys = set(references_by_key)
    unequal = {model: len(expected_keys - keys) for model, keys in keys_by_model.items() if keys != expected_keys}
    if unequal:
        raise ValueError(f"models were not evaluated on identical id/task keys: missing_by_model={unequal}")

    bleu = BLEU(tokenize=args.bleu_tokenizer, effective_order=True)
    bleu.sentence_score("토크나이저 확인", ["토크나이저 확인"])
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False, tokenizer=UnicodeWordTokenizer())
    candidates = [str(row["prediction"]) for row in rows]
    references = [str(row["reference"]) for row in rows]
    precision, recall, f1 = bert_score(candidates, references, model_type=args.bert_model, batch_size=args.bert_batch_size, device=args.device, verbose=True, rescale_with_baseline=False)
    for row, p, r, f in zip(rows, precision.tolist(), recall.tolist(), f1.tolist()):
        rouge_result = rouge.score(row["reference"], row["prediction"])["rougeL"]
        row.update(sentence_bleu=float(bleu.sentence_score(row["prediction"], [row["reference"]]).score), rougeL_f1=float(rouge_result.fmeasure), bertscore_precision=float(p), bertscore_recall=float(r), bertscore_f1=float(f))

    def mean(values: list[float]) -> float:
        return statistics.mean(values)

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        candidates = [str(row["prediction"]) for row in group]
        references = [str(row["reference"]) for row in group]
        return {"count": len(group), "corpus_bleu": float(bleu.corpus_score(candidates, [references]).score), "sentence_bleu_mean": mean([row["sentence_bleu"] for row in group]), "rougeL_f1_mean": mean([row["rougeL_f1"] for row in group]), "bertscore_f1_mean": mean([row["bertscore_f1"] for row in group])}

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["model_label"]), str(row["task"]))].append(row)
    summary = {"bleu_tokenizer": args.bleu_tokenizer, "bert_model": args.bert_model, "by_model_task": {f"{model}/{task}": summarize(group) for (model, task), group in sorted(groups.items())}}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.out_dir / "metrics.rows.jsonl").open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    merge_parser = subparsers.add_parser("merge", help="Merge a PEFT/DoRA adapter into a standalone vLLM model")
    merge_parser.add_argument("--base_model_dir", required=True)
    merge_parser.add_argument("--adapter_dir", required=True)
    merge_parser.add_argument("--focus_init_plan", type=Path)
    merge_parser.add_argument("--output_dir", type=Path, required=True)
    merge_parser.add_argument("--processor_source_dir")
    merge_parser.add_argument("--max_shard_size", default="4GB")
    merge_parser.add_argument("--overwrite", action="store_true")
    infer_parser = subparsers.add_parser("infer", help="Strict no-dictionary vLLM inference")
    infer_parser.add_argument("--model_dir", required=True)
    infer_parser.add_argument("--model_label", required=True)
    infer_parser.add_argument("--eval_jsonl", type=Path, required=True)
    infer_parser.add_argument("--output_jsonl", type=Path, required=True)
    infer_parser.add_argument("--tensor_parallel_size", type=int, default=1)
    infer_parser.add_argument("--limit_per_task", type=int, default=0)
    infer_parser.add_argument("--max_model_len", type=int, default=16384)
    infer_parser.add_argument("--translation_max_new_tokens", type=int, default=8192)
    infer_parser.add_argument("--summarization_max_new_tokens", type=int, default=6144)
    infer_parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    infer_parser.add_argument("--max_num_seqs", type=int, default=16)
    infer_parser.add_argument("--max_num_batched_tokens", type=int, default=32768)
    infer_parser.add_argument("--generation_batch_size", type=int, default=128)
    infer_parser.add_argument("--enforce_eager", action="store_true")
    infer_parser.add_argument("--resume", action="store_true")
    score_parser = subparsers.add_parser("score", help="BLEU, ROUGE-L, and BERTScore")
    score_parser.add_argument("--prediction_jsonl", nargs="+", type=Path, required=True)
    score_parser.add_argument("--out_dir", type=Path, required=True)
    score_parser.add_argument("--bert_model", default="xlm-roberta-large")
    score_parser.add_argument("--bert_batch_size", type=int, default=8)
    score_parser.add_argument("--device", default="cuda")
    score_parser.add_argument("--bleu_tokenizer", default="ko-mecab")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "merge":
        merge(args)
    elif args.command == "infer":
        infer(args)
    else:
        score(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
