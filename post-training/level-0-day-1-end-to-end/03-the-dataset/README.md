# 03 — The Dataset

```bash
python gen_data.py                 # 600 train / 100 valid / 200 test
python gen_data.py --n-train 1000 --seed 1   # regenerate any size
```

Writes `data/train.jsonl`, `data/valid.jsonl`, `data/test.jsonl` in **prompt-completion** format — the one format both TRL and `mlx_lm.lora` read unchanged.

## Why *generate* instead of download

Two reasons, both about learning cleanly:

1. **Zero data-wrangling risk.** No broken download, no license, no schema surprises on Day 0. You own the data end-to-end.
2. **You see the data contract.** Open `task.py`: `render_row()` makes the *messy input* (six different phrasings of the same facts), `target_completion()` makes the *gold JSON*. SFT's job is to learn that mapping — and the six phrasings force it to learn the mapping, not memorize one layout.

## The one line that matters

```json
{"prompt": "Extract … Record: Ava Kim  41  Sales …\nJSON:", "completion": "{\"name\": \"Ava Kim\", …}"}
```

TRL (and MLX) compute the training loss on the **completion only** — the model is graded on the JSON it should produce, not on re-reading the instruction. That's "train on completion only," and it's the default here.

Peek at what you made:

```bash
head -2 data/train.jsonl
```

Next → [04 — baseline eval](../04-baseline-eval/).
