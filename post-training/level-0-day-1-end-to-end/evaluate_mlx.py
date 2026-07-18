"""
Evaluate on Apple Silicon via MLX — same score(), MLX backend, zero cloud cost.

    pip install mlx-lm
    python gen_data.py

    # train the adapter (CLI):
    mlx_lm.lora --model Qwen/Qwen3-0.6B --train --data data \
        --iters 400 --batch-size 4 --num-layers 8 --adapter-path out/mlx-adapters

    # eval base vs adapter:
    python evaluate_mlx.py --limit 100
    python evaluate_mlx.py --adapter out/mlx-adapters --limit 100
"""
import argparse
import json

from mlx_lm import load, generate

from task import score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--adapter", default=None, help="mlx-lm adapter dir (omit for base model)")
    ap.add_argument("--data", default="data/test.jsonl")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=96)
    a = ap.parse_args()

    model, tok = load(a.model, adapter_path=a.adapter)
    rows = [json.loads(line) for line in open(a.data)][: a.limit]
    agg = {"parse_ok": 0.0, "field_accuracy": 0.0, "exact_match": 0.0}
    for ex in rows:
        gold = json.loads(ex["completion"])
        text = generate(model, tok, prompt=ex["prompt"], max_tokens=a.max_tokens, verbose=False)
        s = score(text, gold)
        for k in agg:
            agg[k] += s[k]

    n = len(rows)
    print(json.dumps({
        "model": a.model, "adapter": a.adapter, "n": n,
        "parse_rate": round(agg["parse_ok"] / n, 3),
        "field_accuracy": round(agg["field_accuracy"] / n, 3),
        "exact_match_rate": round(agg["exact_match"] / n, 3),
    }, indent=2))


if __name__ == "__main__":
    main()
