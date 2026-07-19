# The 30-Minute Map — read this first

A real crash course, assuming **zero** post-training knowledge. By the end you'll understand what today's run actually does at every step — not just which commands to type. (If a section is already obvious to you, skip it; the headers are honest.)

---

## 1. What a language model physically is

Strip away the mystique: a language model is a **giant mathematical function with adjustable knobs**, called *parameters* or *weights*. Our model today, `Qwen3-0.6B`, is 596 million floating-point numbers. That's it. No rules, no database of facts, no code that "understands" — just numbers that get multiplied and added.

The function does exactly one thing:

```
input:  a sequence of tokens (word-pieces)        "The capital of France is"
output: a probability for EVERY possible next token
            "Paris"  → 0.92
            "Lyon"   → 0.03
            "the"    → 0.001
            ...one probability per token in a ~150k-token vocabulary
```

Generation is just this function in a loop: predict next token → append it → predict again. Every chatbot you've used is this loop.

> SWE bridge: think of it as a pure function `f(context) -> prob_distribution`, where the 596M weights are the function's compiled constants. "Training" recompiles the constants; the function shape never changes.

## 2. Where the model comes from: pretraining

How do 596M numbers come to encode "Paris follows *the capital of France is*"? **Pretraining**: take trillions of tokens of internet text, and for every position in every document, ask the model to predict the next token. When it's wrong, nudge the weights to make the right token slightly more probable. Repeat ~trillions of times.

The nudging mechanism (this is all of deep learning in one paragraph): the model's error is measured by a **loss** — a single number that's low when the model put high probability on the actual next token and high when it didn't. Calculus (backpropagation) tells us, for each of the 596M weights, *which direction to nudge it* to lower the loss. Take a tiny step in that direction, and repeat. That's **gradient descent**. When you later watch a "loss curve" fall during your training run, this is what's happening: millions of knobs each getting nudged toward "produce the target text."

What pretraining yields is a **base model** — a phenomenal *autocomplete*. It has absorbed grammar, facts, code, reasoning patterns — everything needed to *continue text plausibly*. It knows staggering amounts.

## 3. The gap: why a base model doesn't do what you ask

Here's the catch, and it's the reason this whole field exists. Ask a base model:

```
You:  Extract the name from this record as JSON: "Ava Kim, 41, Sales"
Base: Extract the age from this record as JSON: "Tom Ali, 56, HR"
      Extract the department from this record as JSON: ...
```

It didn't answer — it **continued**. To an autocomplete, your question looks like the first item of a list of similar questions, so the most *probable continuation* is... more questions. Other classic base-model behaviors: answering and then rambling forever (nothing taught it to stop), wrapping answers in essay prose, or happily continuing harmful text (the internet contains everything).

The base model has the *capability* — somewhere in those weights it absolutely can map "Ava Kim, 41, Sales" to JSON. What it lacks is the *behavior*: the reliable habit of reading an instruction, doing the task, emitting the answer, and stopping.

**Post-training is how you install behaviors into a pretrained model.** Pretraining = years of reading the library. Post-training = the short apprenticeship that turns a well-read recluse into a useful colleague. It's cheap relative to pretraining (pretraining: millions of dollars; today's post-training: **40 seconds on your laptop**) precisely because the knowledge is already in there — you're shaping *how it's used*.

> The one-liner worth memorizing: prompting and RAG change *what the model sees*; post-training changes *what the model is*.

## 4. But do you even need it? The escalation ladder

Post-training is the most powerful tool in the box — and the slowest to iterate. Reach for the cheapest rung that solves your problem:

| You need to… | Reach for | Why / the catch |
|---|---|---|
| Nudge behavior, a few instructions | **Prompting** | Zero training, instant iteration. Brittle — instructions get dropped under pressure, and long prompts cost tokens on every single call. |
| Answer over fresh or private facts | **RAG / search** | Knowledge stays external and updatable. But it changes what the model *sees*, not what it can *do*. |
| Inject massive new domain knowledge (>1B tokens — medicine, a new language) | **Continual pretraining**, then post-train | Post-training can't teach knowledge the base never saw; it shapes behavior, not knowledge. |
| Reliably follow many/strict rules; sharpen a target skill (SQL, function-calling, extraction, reasoning) | **Post-training** ← today | The only tool that durably changes behavior. Catch: done carelessly, it degrades everything else (§8). |

Today's task — reliable, strict, always-parseable JSON extraction — is squarely the bottom rung. You *could* prompt for it, and you'd get 80-90% compliance. Post-training is how you get to ~100% with a model 100× smaller than the one you'd have to prompt.

## 5. Method 1 — SFT: teach by demonstration

**Supervised Fine-Tuning** is pretraining's mechanism pointed at curated data. Instead of "predict the next token of the internet," it's "predict the next token of *examples I wish you would imitate*":

```
prompt:      "Extract ... Record: Ava Kim, 41, biz dev, Mar 2 '19, 88k\nJSON:"
completion:  {"name": "Ava Kim", "age": 41, "department": "Sales",
              "start_date": "2019-03-02", "salary": 88000}
```

Same loss, same gradient descent, same knob-nudging as §2 — but on a few hundred hand-shaped examples instead of the internet, and (crucially) **the loss is computed only on the completion tokens**. The model isn't graded on re-predicting your instruction — only on producing the answer. After a few hundred nudges, "emit exactly this JSON shape, normalized exactly this way, then stop" becomes the most probable continuation. That's all "learning the behavior" means: *probability mass moving onto the behavior you demonstrated*.

**SFT is the workhorse.** It's how every chatbot first learned to chat. Its limit: it can only teach what you can *demonstrate*. You need to know the ideal answer to show it.

## 6. Method 2 — DPO: teach by comparison

Sometimes you can't write the ideal answer, but you can *rank* two candidates. "Answer A is better than answer B" — more helpful, better tone, more correct. That's **preference data**:

```
prompt:    "Summarize this thread"
chosen:    (the summary a human preferred)
rejected:  (the summary they passed over)
```

**DPO (Direct Preference Optimization)** trains directly on such pairs: nudge the weights so `chosen` becomes more probable and `rejected` less probable, with a leash (a *reference model*) that stops the model drifting too far from where it started. Historically this required training a separate "reward model" and running reinforcement learning against it (the original RLHF pipeline, 2022-era ChatGPT); DPO's insight was that the math collapses into a single supervised-style loss. Level 3 derives it.

Use it when *"better vs worse"* captures your goal more naturally than *"here's the right answer."*

## 7. Method 3 — RL (GRPO): teach by score

Now the most interesting case: you can't demonstrate the answer, you can't even rank pairs by hand — but you can **score** an answer automatically. Does the code pass the tests? Does the math check out? **Does the JSON parse and match the ground truth?**

Then flip the data flow. Instead of learning from a fixed dataset, the model learns from **its own attempts**:

```
loop:
  1. model generates several answers to a prompt   (e.g. 8 attempts)
  2. a reward function scores each                 (parse? fields correct? → 0.0 to 1.0)
  3. weights are nudged so higher-scoring attempts become more probable
     and lower-scoring ones less probable
```

This is **online reinforcement learning** — "online" because the training data is generated live by the current model, not collected in advance. **GRPO** is the 2024-25 algorithm that made it cheap (it scores each attempt *relative to the other attempts in its group*, killing the need for an expensive helper model — Level 4 unpacks this). When the reward is a machine-checkable verifier rather than a learned model, it's called **RLVR** — *RL with Verifiable Rewards* — and it's literally how DeepSeek-R1 learned to reason: generate solutions, check them, reinforce what worked.

Why bother, if SFT already works? Because RL can push **past** your demonstrations. SFT's ceiling is "as good as the examples you wrote." RL's ceiling is "as good as whatever scores well" — the model can *discover* strategies you never demonstrated. And here's the elegant part for us: the scorer we need is `task.py`'s `score()` function — **the same function you'll use as your eval today becomes the RL reward in Level 4.** One verifiable checker, doing data, algorithm, and eval duty. That's why we picked this task.

## 8. The map — and the fine print

All of it in one picture. Two questions place every method: *where does training data come from?* (fixed dataset = **offline**; the model's own live attempts = **online**) and *what's the signal?* (copy / compare / score):

```
                  OFFLINE (fixed data)          ONLINE (model's own attempts)
  IMITATION       SFT   ← today
  PREFERENCE      DPO                           (Online DPO — Level 4 on-ramp)
  REWARD / RL                                   GRPO, RLVR  ← Level 4
```

**Rule of thumb: SFT first, always.** Add DPO when "better vs worse" beats "right vs wrong." Reach for RL only when you can *score* but not *demonstrate*. Each rung costs more complexity than the one above.

Two pieces of fine print that Levels 2–5 exist for:

- **Catastrophic forgetting.** Nudging weights for *your* task can un-nudge them for everything else. Post-train carelessly and your JSON expert quietly gets worse at general tasks. Honest post-training always measures both: *did the target skill improve?* and *did anything else break?* (Today we do only the first — knowingly.)
- **Reward hacking.** Under RL, the model optimizes exactly what you score, not what you meant. Score only "valid JSON"? It may learn to emit `{}` — valid, useless. Metric design is a skill (Level 4/5).

## 9. Two practicalities in today's run: LoRA and the eval

**LoRA** — we won't actually nudge all 596M weights. LoRA freezes them and bolts small *adapter* matrices beside key layers — we train only those: **1.4M knobs, 0.24% of the model**. Works because a narrow behavior change doesn't need 596M degrees of freedom. Payoffs: ~3 GB of memory instead of a full fine-tune's ~9 (see [`MEMORY-ANATOMY.md`](MEMORY-ANATOMY.md) after your run), and your "trained model" is a few-MB adapter file loaded on top of the frozen base. DPO and GRPO reuse the same trick later.

**The eval** — you'll measure the model **before** training (expect it to fail: ~13% of records fully correct in our verified run) and **after** (100% in that run), on held-out test records it never trained on, with three numbers: `parse_rate` (valid JSON at all?), `field_accuracy` (fields correct?), `exact_match_rate` (whole record right?). No baseline, no claim — this is the discipline that separates *"I trained it and it feels better"* from *"exact-match went 0.13 → 1.00, n=30."*

## 10. Now map it onto what you're about to run

| Runbook step | What it is, in this document's terms |
|---|---|
| `gen_data.py` | Manufacture demonstration pairs (§5) requiring real *transformation* — dates `"Mar 2 '19"`→`2019-03-02`, salaries `"88k"`→`88000` — because a copy-task teaches nothing the base can't already do |
| `evaluate_mlx.py` (base) | Baseline: watch the §3 gap with your own eyes, quantified (§9) |
| `mlx_lm.lora --train` | SFT (§5) via LoRA adapters (§9): a few hundred gradient-descent nudges (§2) on 1.4M knobs |
| `evaluate_mlx.py --adapter` | Same eval, same records: the delta is the entire lesson |

And the arc from here: **Level 1** deepens this map · **2** SFT properly · **3** DPO · **4** GRPO/RLVR, where today's `score()` becomes the reward · **5** honest evaluation at depth.

If you can explain to someone else why the base model fails at JSON (§3), what a falling loss means physically (§2), and when you'd pick SFT vs DPO vs RL (§8) — you've extracted everything Day 0 has to give before running it.

→ Now go run it: [README.md](README.md), Setup then Step 1.
