# Level 0 — Day 1: Train End-to-End

> Track map: [`post-training/README.md`](../README.md) · Artifact: **the JSON-extractor**, first touch
>
> Goal: complete the *whole loop once* — deliberately shallow, fully working — so every later deep-dive points back at something you already built and measured.

## The one thing to feel today

You take a small model that **can't** do a task, run **one** post-training method (SFT), and watch a **number** move. That is the entire mental model of post-training in one sitting: *base model → data → train → eval → it got better.* Everything after Level 0 is depth on one of those arrows.

Don't chase quality today. A tiny model (0.6B), a few hundred synthetic examples, a few minutes of training. The win condition is *"I saw `field_accuracy` jump,"* not *"I built a great model."*

## What you'll have by tonight

- A generated `text → JSON` dataset you can regenerate at any size.
- A one-command eval printing `parse_rate`, `field_accuracy`, `exact_match_rate`.
- **Two saved eval runs — base vs SFT — where the second is visibly better.**
- The ability to say, in one sentence, what SFT did to the model.

## The files in this folder

| File | What it is | Needs |
|---|---|---|
| `task.py` | The task: schema + messy-row renderer + **the JSON verifier**. Pure Python. The verifier is our eval metric now and the RL reward in Level 4. | nothing |
| `gen_data.py` | Generates `train/valid/test.jsonl` in prompt-completion format (both backends read it). | nothing |
| `evaluate.py` | Scores a model (± LoRA adapter) on the test set. CUDA / MPS / CPU. | transformers, torch |
| `train_cloud.py` | SFT + LoRA via TRL. The cloud/GPU path. | trl, peft, datasets |
| `evaluate_mlx.py` | The Mac eval, MLX backend. | mlx-lm |
| `requirements.txt` | Deps for both paths. | — |

Run everything **from this folder**, with the shared env active.

## Setup (once)

```bash
# from the repo ROOT:
python3 -m venv .venv            # skip if .venv already exists
source .venv/bin/activate
pip install -r post-training/level-0-day-1-end-to-end/requirements.txt   # cloud stack
pip install mlx-lm               # add for the Mac / MLX path (Apple Silicon)
cd post-training/level-0-day-1-end-to-end
```

Every later session: just `source .venv/bin/activate` from the repo root.

---

## Track A — Cloud (any NVIDIA GPU: Colab, RunPod, Lambda)

```bash
python gen_data.py                                              # make the data
python evaluate.py --limit 100 --out reports/base.json          # BEFORE — watch it fail
python train_cloud.py                                           # ~a few min on an L4/T4
python evaluate.py --adapter out/sft-lora --limit 100 --out reports/sft.json   # AFTER
```

*No local GPU? Open Colab, `pip install -r requirements.txt`, and run the same four commands — the free T4 is plenty for 0.6B.*

## Track B — Mac (Apple Silicon, zero cost)

```bash
python gen_data.py
python evaluate_mlx.py --limit 100                              # BEFORE
mlx_lm.lora --model Qwen/Qwen3-0.6B --train --data data \
    --iters 400 --batch-size 4 --num-layers 8 --adapter-path out/mlx-adapters
python evaluate_mlx.py --adapter out/mlx-adapters --limit 100   # AFTER
```

You picked **cloud + Mac** — do Track A first for the fast, guaranteed win, then reproduce on the Mac so you own the no-cloud, no-cost version too. *Same dataset, same verifier, same task — only the training backend differs.*

## What "moving the number" looks like

Exact values depend on the model and your sample, but the **shape** is unmistakable:

| metric | base `Qwen3-0.6B` | after SFT |
|---|---|---|
| `parse_rate` | ~0.5 – 0.9 (chatty, code-fenced, or drifting) | **~1.0** |
| `field_accuracy` | ~0.2 – 0.6 | **~0.9+** |
| `exact_match_rate` | ~0.05 – 0.3 | **~0.7 – 0.9** |

If `field_accuracy` roughly doubles (or better), you did it. That delta *is* the lesson.

## The topics, mapped to the steps

| # | Topic | Step it covers |
|---|-------|----------------|
| [01](01-the-30-minute-map/) | the-30-minute-map | Read first — the concepts, so today makes sense (and you're competent even if you stop here) |
| [02](02-environment-setup/) | environment-setup | `pip install …` |
| [03](03-the-dataset/) | the-dataset | `python gen_data.py` |
| [04](04-baseline-eval/) | baseline-eval | `python evaluate.py` (base) |
| [05](05-first-sft-run/) | first-sft-run | `python train_cloud.py` / `mlx_lm.lora` |
| [06](06-after-eval-and-reproduce/) | after-eval-and-reproduce | `python evaluate.py --adapter …` + the other backend |

## Troubleshooting

- **CUDA out of memory** → lower `--batch-size` (try 4 or 2) in `train_cloud.py`.
- **Mac eval is slow** → drop `--limit` to 30; MPS generation is unhurried but fine for a smoke check.
- **First run stalls** → it's downloading `Qwen/Qwen3-0.6B` (~1.2 GB) once; subsequent runs are cached.
- **Weird `<think>` text in output** → we prompt raw (no chat template), so thinking mode stays off; if you switch to the chat template later, pass `enable_thinking=False`.
- **`field_accuracy` didn't move** → confirm `out/sft-lora` (or `out/mlx-adapters`) exists and you passed `--adapter`; check the loss actually fell during training.

## What you can now say

> "I took a base model that couldn't reliably produce structured output, generated a task dataset, ran supervised fine-tuning with LoRA, and measured a real before/after improvement on a verifiable metric — on both a cloud GPU and my own Mac."

That sentence is the confidence anchor. Levels 1–5 turn each clause of it into depth.

## Teach-back

Explain to someone with zero background: *what did SFT change about the model, and how do you know it worked?* If it comes out clean, you've got Level 0. If it stumbles, we go again with a different framing before moving on.
