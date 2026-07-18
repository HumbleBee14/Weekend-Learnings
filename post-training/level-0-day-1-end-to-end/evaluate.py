"""
Evaluate a model (optionally + a LoRA adapter) on the JSON-extraction test set.
Uses the SAME score() that becomes the RL reward in Level 4. Runs on CUDA / MPS / CPU.

    # base model, BEFORE training (watch it fail)
    python evaluate.py --model Qwen/Qwen3-0.6B --limit 100 --out reports/base.json

    # AFTER SFT (watch the number move)
    python evaluate.py --model Qwen/Qwen3-0.6B --adapter out/sft-lora --limit 100 --out reports/sft.json
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from task import score


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--adapter", default=None, help="PEFT LoRA adapter dir (omit for base model)")
    ap.add_argument("--data", default="data/test.jsonl")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--out", default=None, help="optional path to save the JSON report")
    a = ap.parse_args()

    device = pick_device()
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype="auto").to(device)
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter).to(device)
    model.eval()

    rows = [json.loads(line) for line in open(a.data)][: a.limit]
    agg = {"parse_ok": 0.0, "field_accuracy": 0.0, "exact_match": 0.0}
    examples = []
    for i, ex in enumerate(rows):
        gold = json.loads(ex["completion"])
        inp = tok(ex["prompt"], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=a.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        text = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        s = score(text, gold)
        for k in agg:
            agg[k] += s[k]
        if i < 3:
            examples.append({"output": text.strip()[:200], "score": s})

    n = len(rows)
    report = {
        "model": a.model, "adapter": a.adapter, "n": n, "device": device,
        "parse_rate": round(agg["parse_ok"] / n, 3),
        "field_accuracy": round(agg["field_accuracy"] / n, 3),
        "exact_match_rate": round(agg["exact_match"] / n, 3),
    }
    print(json.dumps(report, indent=2))
    print("\nfirst few outputs:")
    for e in examples:
        print(f"  out  : {e['output']}")
        print(f"  score: {e['score']}")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"report": report, "examples": examples}, open(a.out, "w"), indent=2)
        print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
