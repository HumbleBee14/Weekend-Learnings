# Post-Training LLMs — Train Your Own Custom Local Models

Take a small open model and teach it a new, *verifiable* skill — starting from zero post-training knowledge, ending able to run the full **SFT → preference → online-RL** loop yourself, on a cheap rented GPU or your own Mac.

> Design doc: [`docs/superpowers/specs/2026-07-15-post-training-track-design.md`](../docs/superpowers/specs/2026-07-15-post-training-track-design.md)

## What this is — and what it isn't

**Post-training** is everything that happens to a model *after* pretraining to make it useful: following instructions, matching human preferences, and optimizing against a reward. Pretraining gives you a model that can *continue text*; post-training gives you one that can *do what you ask*.

- **Not pretraining** (`systems-for-ml/` L6 owns that).
- **Not serving/inference** (`systems-for-ml/` L1–L8).
- **Prerequisite:** `python-pytorch/level-5-llm-finetuning` teaches LoRA *mechanics*. This track teaches the *discipline* — which method, on what data, measured how.

You learn by carrying **one small model** through **every method**, on **one verifiable task**, judged by **one eval** — so you can see precisely what each method adds.

## Do you even need post-training?

Post-training is the most expensive, slowest-to-iterate tool in the box. Reach for the **cheapest rung that solves your problem**, and climb only when it stops working.

| If you need to… | Reach for | Why — and the catch |
|---|---|---|
| Nudge behavior, follow a few instructions, add a guardrail | **Prompting** (system prompt, few-shot) | Zero training, instant iteration. *Brittle* — the model may drop instructions under pressure, and a long prompt costs tokens on every call. |
| Answer over fresh, private, or fast-changing facts | **RAG / search** | Knowledge stays external and updatable — no retraining when facts change. But it changes only *what the model sees*, not what it can *do*. |
| Inject large-scale new domain knowledge (>1B tokens: medicine, law, a new language) | **Continual pre-training → then post-training** | Post-training alone can't teach knowledge the base model never saw. Add pre-training first, then post-train to make it usable. |
| Reliably follow many/strict instructions, or sharpen a targeted skill (SQL, function-calling, a reasoning or extraction model) | **Post-training** — *this track* | The only tool that *durably changes behaviour* and lifts a target capability. The catch: done carelessly it degrades everything else (catastrophic forgetting) — which is exactly why eval is a pillar. |

> **The one line to remember:** prompting and RAG change *what the model sees*; post-training changes *what the model is*.

### Once you're post-training — which method?

The same escalation logic applies inside the track. Start at the top; add a rung only when the one above can't express what you want:

| If you can provide… | Use | Lives on the map at… |
|---|---|---|
| **Demonstrations** of the right answer | **SFT** — always start here | offline · imitation |
| **Comparisons** — this answer is better than that one | **DPO** (or KTO/ORPO) | offline · preference |
| **A score** — you can *grade* an answer automatically (verifiable) or with a reward model | **Online RL — GRPO / RLVR** | online · reward |

> **Rule of thumb:** SFT first, *always*. Add DPO when *"better vs worse"* captures your goal better than *"right vs wrong."* Reach for RL only when you can **score** an answer but can't **demonstrate** it.

## The three pillars

A working post-training recipe requires three co-equal elements:

```
   post-training  =  (data ⊗ algorithm)   +   reliable & efficient library   +   evaluation suite
                     ─────────────────        ──────────────────────────         ────────────────
                     WHAT you train on         WHAT you train with                HOW you know it worked
                     and HOW                    (TRL / Unsloth / MLX)              (the steering wheel)
```

"Data ⊗ algorithm" is **one** pillar on purpose: the algorithm dictates the data contract. SFT wants demonstrations, DPO wants preference pairs, GRPO wants prompts + a reward. Pick the method and the data shape together — that's **co-design**.

## The mental model — two axes of methods

Every method sits at the intersection of two independent questions:

```
                    OFFLINE                          ONLINE
              (fixed pre-collected data)      (model generates its own
                                               data during training)
   IMITATION      SFT
   PREFERENCE     DPO, KTO, ORPO, IPO          Online DPO, XPO, Nash-MD
   REWARD / RL                                 REINFORCE, RLOO, PPO, GRPO, RLVR, DAPO, GSPO
```

- **online vs offline** — does the model roll out its *own* answers during training and learn from them, or learn from a frozen dataset?
- **the signal** — imitation (copy the target) → preference (chosen > rejected) → reward (push toward a scalar score).
- **"online RL"** = the bottom-right cell. Family tree: `REINFORCE → {PPO adds a critic} → {GRPO, RLOO drop the critic again}`. GRPO is the DeepSeek-R1 method; **RLVR** = RL with a machine-*verifiable* reward.

## The running artifact — one task, three lenses

**Task:** messy tabular text → strict, correct JSON. A real slice of "read a spreadsheet / extract structured data," with one magic property: **correctness is machine-verifiable** (does it parse? do the fields match?). That property is what lets the same task carry through every method and *become* the RL reward.

```
Day-0 base            ──SFT──►     ──DPO──►            ──GRPO/RLVR──►
rambles, invalid      emits clean  prefers correct     self-corrects to
JSON                  JSON         over broken JSON     a verified reward
        the SAME small model, the SAME eval, three lenses
```

## Setup & running

All tracks share one virtualenv at the repo root.

```bash
# from repo root, one time:
python3 -m venv .venv && source .venv/bin/activate
pip install -r post-training/level-0-day-1-end-to-end/requirements.txt   # + `pip install mlx-lm` on Apple Silicon
```

Then follow the **[Level 0 runbook](level-0-day-1-end-to-end/README.md)** — exact commands, both backends, end to end. Every session after: `source .venv/bin/activate`.

## The levels

| Level | Title | You walk out able to… |
|---|---|---|
| **[0](level-0-day-1-end-to-end/)** | **Day 1 — train end-to-end** | Run one full SFT loop cloud + Mac, and *see* the before/after on a real eval |
| **[1](level-1-post-training-landscape/)** | Landscape, two axes & three pillars | Place any method on the map and explain why post-training exists at all |
| **[2](level-2-sft/)** | SFT, deep | Explain and control what next-token loss on curated data actually changes |
| **[3](level-3-preference-optimization/)** | Preference optimization (offline) | Derive DPO's insight, build preference data, run DPO on your SFT model |
| **[4](level-4-online-rl/)** | Online RL: PPO → GRPO → RLVR | Explain the policy-gradient family and run GRPO with a verifiable reward |
| **[5](level-5-evaluation-and-pipeline/)** | Eval depth, data & the full pipeline | Judge a model honestly and reproduce the whole SFT→DPO→GRPO run |

Level 0 is the fast, complete hit. Levels 1–5 earn the theory against the artifact Level 0 already produced. **Levels 2–4 each end by running the full eval** (task + regression) to measure that method's delta.

## Toolchain — cloud + Mac (same concepts, two backends)

Day-0 model: **Qwen3-0.6B** (fallback `Qwen2.5-0.5B` — TRL's own teaching model).

| Stage | Cloud (CUDA) | Mac (Apple Silicon) |
|---|---|---|
| SFT | TRL `SFTTrainer` / Unsloth (QLoRA) | `mlx-lm` / `mlx-lm-lora` |
| Preference (DPO) | TRL `DPOTrainer` | `mlx-lm-lora` |
| Online RL (GRPO) | TRL `GRPOTrainer` / Unsloth (~5 GB VRAM) | `mlx-lm-lora` |
| Eval | `lm-eval-harness` / `lighteval` + task metric | same |

**Scale tier (named, not built):** OpenRLHF / veRL / NeMo-RL — Ray-based actors, fast rollout↔train weight-sync, multi-node. You graduate here when TRL stops fitting; that's the bridge back to `systems-for-ml`.

## Compute

```
  90% LOCAL   → design, code, generate data, test logic, Mac/MLX runs (zero cost)
  10% GPU     → real cloud runs. A single L4/A10 (~$0.40/hr) covers Day 0.
```

Whole track spend: single-digit dollars. Spin GPUs down between sessions.

## References — read fresh per topic (this space moves monthly)

- **RLHF Book** — Nathan Lambert, [rlhfbook.com](https://rlhfbook.com) — the canonical free text on post-training. Primary read.
- **TRL docs** (HuggingFace) — SFT/DPO/GRPO trainers, the how.
- **DPO paper** — Rafailov et al. 2023, *Direct Preference Optimization*.
- **GRPO** — DeepSeekMath (2024) introduced it; **DeepSeek-R1** (2025) scaled it.
- **Open recipes** — AI2 **Tulu 3** (incl. RLVR) and HuggingFace **SmolLM3** blueprint — full end-to-end post-training pipelines with published data mixes.
- **Local:** Unsloth docs; `mlx-lm-lora` (SFT/DPO/GRPO on Apple Silicon).
- **Reddi Vol 1** *Efficient AI / Model Optimizations* — light academic framing for the optimization side (`../systems-for-ml/references/`).

Per repo convention: every topic gets a fresh web-research pass before we write it — the stack (RLOO, GRPO, DAPO, GSPO, veRL…) changes fast.
