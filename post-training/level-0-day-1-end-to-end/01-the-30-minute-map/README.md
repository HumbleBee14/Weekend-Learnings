# 01 — The 30-Minute Map

Read this before touching code. It's the whole field compressed to what you need so today makes sense — and so that even if you never do Levels 1–5, you can hold a competent conversation about post-training.

## What post-training even is

**Pretraining** trains a model to predict the next token over the whole internet. The result *knows* a staggering amount but only *continues text* — it doesn't reliably answer, follow a format, refuse, or prefer the better of two replies.

**Post-training** is everything you do *after* that to install those behaviours — without paying to pretrain again. That's this whole track.

> One line: prompting and RAG change *what the model sees*; post-training changes *what the model is*.

## Do you even need it?

Reach for the cheapest tool that works; climb only when it breaks:

```
prompting  →  RAG / search  →  continual pre-training  →  POST-TRAINING
(few instrs)  (fresh facts)     (>1B tokens of new       (reliably change behaviour;
                                 domain knowledge)         sharpen a target skill)
```

You post-train when you need to *reliably change behaviour* or *sharpen a specific skill* (SQL, function-calling, JSON extraction, reasoning) — which is exactly today's task.

## The three methods, in one map

Every method sits at the crossing of two questions: **where does the training data come from** (a fixed set, or the model's own live outputs) and **what signal drives it** (copy the answer / prefer the better one / chase a reward).

```
                 OFFLINE (fixed data)         ONLINE (model's own rollouts)
   IMITATION     SFT  ← today
   PREFERENCE    DPO
   REWARD/RL                                  GRPO / RLVR
```

- **SFT** (Level 2) — show it `prompt → ideal answer`, it imitates. The foundation. *This is what you run today.*
- **DPO** (Level 3) — show it `chosen vs rejected`, it learns to prefer the better one. Use when "better vs worse" beats "right vs wrong."
- **GRPO / RLVR** (Level 4) — it generates answers, a reward function scores them, it optimizes the reward. Use when you can *score* an answer but can't *demonstrate* it. When the score is machine-checkable, that's **RLVR** — the DeepSeek-R1 engine.

Rule of thumb: **SFT first, always.** Add the next rung only when the one above can't express your goal.

## What we do today, and why this task

Teach `Qwen3-0.6B` to turn a messy record into strict JSON:

```
"Name: Ava Kim, Age 41, Dept=Sales, Started 2019-03-02, Salary $88000"
        │  SFT
        ▼
{"name":"Ava Kim","age":41,"department":"Sales","start_date":"2019-03-02","salary":88000}
```

Why this task: its correctness is **machine-verifiable** (does it parse? do the fields match?). That one property lets the *same task* carry through all three methods — and the verifier you'll use to eval today literally *becomes the reward* in Level 4. Data, algorithm, and eval, co-designed as one object.

## The three pillars you're touching today

```
(data ⊗ algorithm)      +      reliable library      +      evaluation
 gen_data.py + SFT              TRL / mlx-lm                  task.py's score()
```

## After 30 minutes you can say

- what pretraining leaves undone, and what post-training fixes;
- when to prompt / RAG / post-train;
- what SFT, DPO, and RL each do, and when to pick each;
- why a *verifiable* task is the good one to learn on.

Now go run it → [Level 0 runbook](../README.md).
