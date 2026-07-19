# Memory Anatomy — what's in the 3 GB while a model trains?

*Depth read for after your first run. Prompted by a real question: the training log says `Peak mem 3.041 GB` — but the dataset was ~160 KB. So what is the 3 GB?*

Answering this precisely is the single most useful systems skill in ML: knowing what lives in RAM during inference vs training, and how each piece scales.

## First: which memory?

- **Apple Silicon (our run):** *unified memory* — one RAM pool shared by CPU and GPU. The 3 GB came out of ordinary system RAM.
- **NVIDIA cloud GPU:** the same bytes would live in the GPU's dedicated **VRAM**, and moving tensors between CPU RAM and VRAM is an extra cost that unified memory doesn't pay. (This is the Level 8 `systems-for-ml` story — same lesson, different bandwidth budget.)

Either way, "how much memory does training need" is a property of the **model and method**, not of the data.

## The anatomy of our 3 GB

```
  Peak ~3.0 GB during the Day-0 run (Qwen3-0.6B + LoRA, batch 4)
  ┌──────────────────────────────────────────────────────────────┐
  │ 1. Model weights (FROZEN)              ~1.2 GB               │
  │      596M params × 2 bytes (bf16)                            │
  │                                                              │
  │ 2. Activations (per forward pass)      ~1.5 GB               │
  │      every layer's intermediate outputs, kept until          │
  │      backprop consumes them; scales with batch × seq len     │
  │                                                              │
  │ 3. Trainable state (LoRA only!)        ~0.02 GB              │
  │      1.4M adapter params + their gradients                   │
  │      + Adam optimizer state (2 moments per param)            │
  │                                                              │
  │ 4. Framework overhead, buffers         ~0.3 GB               │
  └──────────────────────────────────────────────────────────────┘
```

(The split is an informed reconstruction — the log reports only the peak — but each line follows from arithmetic you can do yourself, below.)

## The four numbers behind the picture

**1. Weights: `params × bytes-per-param`.** The most-used equation in ML systems. 596M × 2 bytes (bf16) ≈ 1.19 GB. In fp32 it would be 2.4 GB; quantized to 4-bit, ~0.3 GB. This one multiplication tells you what fits on what hardware.

**2. Activations scale with batch and sequence, not with dataset size.** A forward pass stores every layer's outputs so the backward pass can compute gradients. Double the batch → roughly double the activations. This is the knob you turn when you hit out-of-memory (`--batch-size 2`), and it's why OOM errors appear *during* training, not at load time.

**3. Trainable state costs ~10 bytes per TRAINABLE parameter.** For each parameter being trained you hold: the gradient (2 bytes) + Adam's two moment estimates (often kept in fp32: 8 bytes). So:

| method | trainable params | extra training state |
|---|---|---|
| **LoRA (our run)** | 1.4M (0.24%) | **~15 MB** |
| full fine-tune of the same 0.6B | 596M | **~6 GB** |

That single row-to-row jump is *the entire reason* a laptop can post-train a model. LoRA freezes the base weights (line 1 stays read-only) and bolts small low-rank adapter matrices beside them; only those learn. QLoRA goes further — the frozen base is stored 4-bit quantized — which is how a 7B model fine-tunes in ~8 GB.

**4. Data is invisible in this budget.** 400 examples ≈ 160 KB of text — noise. Only the current batch is ever in memory. **Memory scales with the model; time scales with the data.** Training on 40,000 examples instead of 400 would need the same 3 GB and simply take 100× longer per epoch.

## The rules of thumb worth memorizing

```
inference          ≈  weights (+ KV cache + small activations)
LoRA fine-tune     ≈  inference + activations + ~10 B × adapter params   ← tiny add-on
full fine-tune     ≈  inference + activations + ~10 B × ALL params       ← the A100 tier
```

And from our actual log, the throughput numbers that make it concrete: **~2,400 tokens/sec** trained, **~10.5 iterations/sec** at batch 4, val loss 0.983 → 0.001 in 400 iterations, ~40 seconds wall clock.

## Where the analogy to systems you know holds (and breaks)

Weights are like a **memory-mapped read-only file** shared by all requests; activations are like **per-request scratch on the heap** — sized by concurrency (batch), freed after each pass; optimizer state is like an **index you maintain only on the columns you're updating** — LoRA is choosing to index 0.24% of the table. The analogy breaks at backprop: unlike request scratch, activations can't be freed as you go — the backward pass needs them in reverse order, which is why training (not inference) is the memory-hungry direction, and why tricks like *gradient checkpointing* (recompute instead of store) exist.

---

*Next depth on this: Level 2 (`lora-qlora-from-first-principles`) — why low-rank works at all, and what rank/alpha actually control.*
