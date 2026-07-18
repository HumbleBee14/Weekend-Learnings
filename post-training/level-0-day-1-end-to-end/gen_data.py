"""
Generate the text -> JSON dataset. Pure Python, no GPU.

Writes train.jsonl / valid.jsonl / test.jsonl in prompt-completion format, which
BOTH backends read directly:
  - TRL (cloud)   : load_dataset("json", data_files=...)
  - mlx-lm (Mac)  : mlx_lm.lora --data <this dir>   (expects train.jsonl / valid.jsonl)

Usage:
    python gen_data.py                       # 600 / 100 / 200 by default
    python gen_data.py --n-train 1000 --seed 1
"""
import argparse
import json
import os
import random

from task import make_record, render_row, build_prompt, target_completion


def make_split(n: int, rng: random.Random) -> list[dict]:
    rows = []
    for _ in range(n):
        rec = make_record(rng)
        row = render_row(rec, rng)
        rows.append({"prompt": build_prompt(row), "completion": target_completion(rec)})
    return rows


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--n-valid", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="data")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    rng = random.Random(a.seed)
    for name, n in [("train", a.n_train), ("valid", a.n_valid), ("test", a.n_test)]:
        rows = make_split(n, rng)
        write_jsonl(os.path.join(a.out_dir, f"{name}.jsonl"), rows)
        print(f"wrote {n:>5}  ->  {a.out_dir}/{name}.jsonl")

    print("\nsample record:")
    print(json.dumps(make_split(1, rng)[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
