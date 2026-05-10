"""
Turn a folder of plain-text / Markdown files into the mlx_lm JSONL format
that mlx_lm.lora expects. Applies the model's chat template so the
adapter learns the right boundary tokens.

Outputs:
    ./data/train.jsonl
    ./data/valid.jsonl

Each line is {"text": "<chat-templated turn pair>"}.
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

from mlx_lm import load


def chunk_text(text: str, max_chars: int = 2000) -> list[str]:
    """Split a doc into roughly-paragraph chunks bounded by max_chars."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 2 > max_chars and cur:
            chunks.append(cur)
            cur = p
        else:
            cur = (cur + "\n\n" + p) if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="folder of .md / .txt files")
    ap.add_argument("--output", default="./data")
    ap.add_argument("--model", required=True,
                    help="base model id, used only for tokenizer + chat template")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    _, tokenizer = load(args.model)
    random.seed(args.seed)

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    examples: list[dict] = []
    for path in Path(args.input).rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in (".md", ".txt"):
            continue
        body = path.read_text(errors="replace")
        for chunk in chunk_text(body):
            # Synthesize a (instruction, response) pair where the response is
            # the chunk in your voice. Adjust the instruction template per
            # what you're actually trying to teach.
            messages = [
                {"role": "user",
                 "content": "Write a passage in the author's voice on this topic."},
                {"role": "assistant", "content": chunk},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            examples.append({"text": text})

    random.shuffle(examples)
    n_val = max(1, int(len(examples) * args.val_frac))
    val, train = examples[:n_val], examples[n_val:]

    with (out_root / "train.jsonl").open("w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")
    with (out_root / "valid.jsonl").open("w") as f:
        for ex in val:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(train)} train, {len(val)} val examples to {out_root}")


if __name__ == "__main__":
    main()
