#!/usr/bin/env python3
"""Portable BF16 inference example; requires memory for the full 250B backbone."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "project" / "scripts"))
from solar_peft_compat import solar_open2_peft_v5_native
from reference.shared_nodict_evaluation import build_messages_no_dict, clean_text

BASE = "upstage/Solar-Open2-250B"
REVISION = "8d1dda5ad2af58207c65348d825772d2f4d72481"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--task", choices=["translation", "summarization"], default="translation")
    parser.add_argument("--adapter", type=Path, default=ROOT / "adapter")
    parser.add_argument("--base", default=BASE, help="Pinned model ID or the same revision's local snapshot")
    parser.add_argument("--max-new-tokens", type=int)
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPUs with memory for the BF16 250B model are required")
    source = clean_text(args.source_file.read_text(encoding="utf-8"))
    if not source:
        parser.error("source file is empty")
    remote_kwargs = {"revision": REVISION} if args.base == BASE else {}
    tokenizer = AutoTokenizer.from_pretrained(str(args.adapter), local_files_only=True)
    prompt = tokenizer.apply_chat_template(
        build_messages_no_dict(args.task, source), tokenize=False,
        add_generation_prompt=True, reasoning_effort="none",
        think_render_option="preserved", provider_system_prompt=False,
    )
    preamble = "<|im:start|>assistant<|im:content|><|think:start|><|think:end|>"
    if not prompt.endswith(preamble):
        raise RuntimeError("Unexpected chat template / output-only preamble")
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    max_new = args.max_new_tokens or (8192 if args.task == "translation" else 6144)
    if max_new <= 0 or inputs.input_ids.shape[1] + max_new > 16384:
        parser.error("prompt + generation must fit 16384 tokens; source is never truncated")
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=False, **remote_kwargs,
    )
    with solar_open2_peft_v5_native():
        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
    model.eval()
    inputs = inputs.to(model.get_input_embeddings().weight.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs, do_sample=False, max_new_tokens=max_new,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im:end|>"),
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = output[0, inputs.input_ids.shape[1]:]
    print(tokenizer.decode(generated, skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
