# 02 — Environment Setup

Everything runs in the **shared root `.venv`** — one environment for the whole repo. Set it up once.

## One-time setup (from the repo root)

```bash
python3 -m venv .venv            # skip if it already exists
source .venv/bin/activate
pip install -r post-training/level-0-day-1-end-to-end/requirements.txt   # transformers, trl, peft, torch, …
pip install mlx-lm               # Apple Silicon only — the Mac training path
```

Verify it loaded:

```bash
python -c "import torch, trl, mlx_lm; print('torch', torch.__version__, '| mps', torch.backends.mps.is_available())"
```

Every session after: `source .venv/bin/activate` from the repo root, then `cd post-training/level-0-day-1-end-to-end`.

## Which backend you'll use

- **Cloud GPU** (Colab / RunPod): the TRL path — `train_cloud.py` + `evaluate.py`.
- **Your Mac** (Apple Silicon): the MLX path — `mlx_lm.lora` + `evaluate_mlx.py`.

The shared env has *both* toolchains installed, so you can switch freely.

## Why a tiny model

`Qwen/Qwen3-0.6B` is deliberate: big enough to actually learn the task, small enough that a full train+eval loop is minutes, not hours — so you *iterate*, which is the whole point of Day 0. It's also the model TRL's own docs teach on. Downloads once (~1.2 GB), cached after.

Next → [03 — the dataset](../03-the-dataset/).
