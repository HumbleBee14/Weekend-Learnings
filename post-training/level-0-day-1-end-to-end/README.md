# Level 0 — Day 1: Train End-to-End

> Track map: [`post-training/README.md`](../README.md) · Artifact: **the JSON-extractor**, first touch
>
> Goal: complete the *whole loop once* — deliberately shallow, fully working — so every later deep-dive points back at something you already built and measured.

**How to read this level** — three files, in this order:

1. [`THE-30-MINUTE-MAP.md`](THE-30-MINUTE-MAP.md) — concepts primer, *before* running (makes you conversational about post-training even if you stop after this level)
2. **This README** — the runbook; do the four steps
3. [`MEMORY-ANATOMY.md`](MEMORY-ANATOMY.md) — depth read, *after* your run

Everything else here is code you'll run or evidence you can inspect ([`runs/`](runs/)).

## The one thing to feel today

You take a small model that **can't** do a task, run **one** post-training method (SFT), and watch a **number** move. That is the entire mental model of post-training in one sitting: *base model → data → train → eval → it got better.* Everything after Level 0 is depth on one of those arrows.

Don't chase quality today. A tiny model (0.6B), a few hundred synthetic examples, ~40 seconds of training. The win condition is *"I saw `field_accuracy` jump,"* not *"I built a great model."*

## What's in this folder

| File | What it is |
|---|---|
| `README.md` | **This runbook** — the whole Day 0, start to finish |
| [`THE-30-MINUTE-MAP.md`](THE-30-MINUTE-MAP.md) | Concepts primer — read before running |
| [`MEMORY-ANATOMY.md`](MEMORY-ANATOMY.md) | Depth read for after: what the ~3 GB of training memory is made of |
| `task.py` | The task: schema + messy-row renderer + **the verifier** (`score()`) — eval metric today, RL reward in Level 4 |
| `gen_data.py` | Generates `data/{train,valid,test}.jsonl`; one format feeds both backends |
| `evaluate.py` / `evaluate_mlx.py` | Before/after scoring — CUDA/CPU (transformers) / Apple Silicon (MLX) |
| `train_cloud.py` | SFT+LoRA via TRL — the cloud/GPU path |
| `runs/` | **Git-tracked evidence** from noteworthy runs (raw log + results). Scratch dirs `data/ out/ reports/` are gitignored and regenerable |

## Setup (once)

```bash
# from the repo ROOT:
python3 -m venv .venv            # skip if .venv already exists
source .venv/bin/activate
pip install -r post-training/level-0-day-1-end-to-end/requirements.txt   # cloud stack
pip install mlx-lm               # add for the Mac / MLX path (Apple Silicon)
cd post-training/level-0-day-1-end-to-end
```

Verify it loaded:

```bash
python -c "import torch, trl, mlx_lm; print('torch', torch.__version__, '| mps', torch.backends.mps.is_available())"
```

Every later session: `source .venv/bin/activate` from the repo root, then cd here.

**Which backend you'll use:**

- **Cloud GPU** (Colab / RunPod / any NVIDIA): the TRL path — `train_cloud.py` + `evaluate.py`.
- **Your Mac** (Apple Silicon): the MLX path — `mlx_lm.lora` + `evaluate_mlx.py`.

The shared env has *both* toolchains installed, so you can switch freely. Each step below shows both commands.

**Why a tiny model:** `Qwen/Qwen3-0.6B` is deliberate — big enough to actually learn the task, small enough that a full train+eval loop is minutes, not hours, so you *iterate*, which is the whole point of Day 0. It's also the model TRL's own docs teach on, so the toolchain is well-trodden. Downloads once (~1.2 GB), cached after.

---

## The loop — four steps

### Step 1 — Generate the data

```bash
python gen_data.py                              # 600 train / 100 valid / 200 test
python gen_data.py --n-train 1000 --seed 1      # regenerate any size, any seed
head -2 data/train.jsonl                        # look at what you made
```

**Why generate instead of download:** zero data-wrangling risk, fully reproducible, and you *see the data contract*. Open `task.py`: `render_row()` makes the messy input, `target_completion()` the gold JSON. The input requires **transformation, not copying** — `"11/2/21"` → `2021-11-02`, `"$64K"` → `64000`, `"People Ops"` → `"HR"`. That's deliberate: a base model can already copy, so a copy-task teaches nothing (we proved it — first version of this task had verbatim fields and the baseline scored 93%).

**The one line that matters** — every record in the JSONL looks like this:

```json
{"prompt": "Extract … Record: Ava Kim  41  biz dev …\nJSON:", "completion": "{\"name\": \"Ava Kim\", …}"}
```

This **prompt-completion** format is the one format both TRL and `mlx_lm.lora` read unchanged — and both compute the training loss on the **completion only**: the model is graded on the JSON it should produce, not on re-reading the instruction. That's "train on completion only," and it's the default here.

### Step 2 — Baseline eval (watch it fail)

```bash
python evaluate.py --limit 30 --out reports/base.json     # cloud / CPU
python evaluate_mlx.py --limit 30                          # Mac
```

Measure **before** you touch anything — you can't claim an improvement without a baseline. Three numbers:

- `parse_rate` — outputs that are valid JSON at all
- `field_accuracy` — fraction of the 5 fields matching ground truth
- `exact_match_rate` — records with *all* fields right

A 0.6B base model typically wraps answers in prose or code fences, invents keys, fumbles the date normalizations, or guesses departments — so expect a **low, telling** baseline. Read the `first few outputs` the script prints; *seeing* the failure modes is the point.

**Why this eval is the same object as the reward:** `evaluate.py` calls `score()` from `task.py`. In Level 4 the **exact same `score()`** becomes the GRPO reward function. Eval and reward are one verifiable checker — a good baseline eval today is also the foundation of RL later.

Save `reports/base.json` — you'll diff it against the post-SFT run in Step 4.

### Step 3 — Train (SFT + LoRA)

```bash
# cloud / GPU
python train_cloud.py                           # -> adapter in out/sft-lora

# Mac (Apple Silicon)
mlx_lm.lora --model Qwen/Qwen3-0.6B --train --data data \
    --iters 400 --batch-size 4 --num-layers 8 --learning-rate 1e-4 \
    --mask-prompt --adapter-path out/mlx-adapters
```

What's actually happening, in 60 seconds:

- **LoRA, not full fine-tuning:** the 596M base weights stay **frozen**; you train ~1.4M small adapter matrices (0.24%) bolted onto the layers. That's why this fits on a laptop — the full memory story is in [`MEMORY-ANATOMY.md`](MEMORY-ANATOMY.md). Later methods (DPO in Level 3, GRPO in Level 4) reuse the same trick; the *why* of low-rank adapters is Level 2.
- **Loss on the completion only** (`--mask-prompt` on MLX; TRL's default for prompt-completion data): the model is scored on the JSON, not the instruction.
- **`eos_token="<|im_end|>"`** (cloud script): Qwen-specific — teaches the model to *stop* after the JSON.
- **Watch the loss fall.** Falling loss = learning to reproduce the gold completions. That's SFT's entire visible signal. (If it hits 0.000 within ~20 iters, your task is too easy for the model — that's a data-design smell, not a win.)

Knobs you'll understand deeply in Level 2: `r`/`lora_alpha` (adapter capacity), learning rate (~1e-4 for adapters), iters/epochs, `--num-layers` (MLX: how many layers get adapters). Defaults here just work; don't fiddle yet.

When it finishes you'll have a saved adapter (`out/sft-lora` or `out/mlx-adapters`) — **a few MB of learned deltas, not a new full model.** At inference you load base + adapter together.

### Step 4 — After eval + reproduce on the other backend

```bash
python evaluate.py --adapter out/sft-lora --limit 30 --out reports/sft.json   # cloud
python evaluate_mlx.py --adapter out/mlx-adapters --limit 30                  # Mac
```

Same eval, same records — only the adapter is new. **Put the two reports side by side** (`reports/base.json` vs `reports/sft.json`): that jump, from one method, on a metric you defined, is what "post-training worked" means. Not a vibe — a number.

Then **reproduce on the other backend** (you're doing cloud + Mac): same `data/`, same `task.py`, same eval — only the trainer changed (`train_cloud.py` ↔ `mlx_lm.lora`). The result holding across two frameworks proves the *method* did the work, not some quirk of one library. It's also your first taste of the `systems-for-ml` Level 8 lesson: same workload, different hardware substrate and bandwidth budget.

**Done with Level 0** when both reports are saved and the delta is real. Keep them — they're your evidence (see [`runs/`](runs/) for how we archive noteworthy ones).

---

## The real measured result (so you know what to expect)

Verified run — Qwen3-0.6B, MLX on an Apple Silicon Mac, 400 train examples, 400 LoRA iters, 30 test records (July 2026). Raw log + writeup: [`runs/2026-07-19-mlx-day0/`](runs/2026-07-19-mlx-day0/RESULTS.md).

| metric | base model | after SFT |
|---|---|---|
| `parse_rate` | 0.867 | **1.000** |
| `field_accuracy` | 0.673 | **1.000** |
| `exact_match_rate` | **0.133** | **1.000** |

The base model gets a complete record right 13% of the time — it copies names fine but fumbles the normalizations. After a ~40-second LoRA fine-tune: perfect on all 30. Footprint: 0.24% of params trained, ~2,400 tokens/sec, ~3 GB peak memory ([what that 3 GB is](MEMORY-ANATOMY.md)). Your numbers will vary a little with the seed; the shape won't.

## Troubleshooting

- **CUDA out of memory** → lower `--batch-size` (4 or 2) in `train_cloud.py`.
- **Mac eval is slow** → drop `--limit` to 30; MPS generation is unhurried but fine.
- **First run stalls** → it's downloading the model (~1.2 GB) once; cached after.
- **Weird `<think>` text in output** → we prompt raw (no chat template), so thinking mode stays off; if you switch to the chat template later, pass `enable_thinking=False`.
- **Metrics didn't move** → confirm the adapter dir exists and you passed `--adapter`; check the loss actually fell during training.

## What you can now say

> "I took a base model that couldn't reliably produce structured output, generated a task dataset, ran supervised fine-tuning with LoRA, and measured a real before/after improvement on a verifiable metric — on both a cloud GPU and my own Mac."

That sentence is the confidence anchor. Levels 1–5 turn each clause of it into depth.

**The honest caveat** (previews Levels 2 & 5): you measured the *target* skill. You have *not* checked whether the model got worse at everything else — that's **catastrophic forgetting**, and measuring it is exactly what the later levels add.

## Teach-back

Explain to someone with zero background: *what did SFT actually change about the model, and how do you know it worked?* If it comes out clean, you've got Level 0 — on to [Level 1](../level-1-post-training-landscape/). If it stumbles, we go again with a different framing.
