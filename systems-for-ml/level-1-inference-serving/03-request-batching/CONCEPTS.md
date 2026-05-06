# 03 — Request Batching

## Why batching exists

GPUs hate small batches. A forward pass on batch=1 leaves most of the SMs (streaming multiprocessors) idle waiting on memory. The matmul reads weights from HBM, does a tiny amount of compute, writes back. Memory bandwidth is the bottleneck — adding more sequences to the batch costs almost nothing in extra time but multiplies useful work.

Concrete numbers on an A100 with a 7B model:
- Batch=1 decode: ~30 tokens/sec
- Batch=8 decode: ~210 tokens/sec
- Batch=32 decode: ~600 tokens/sec

Throughput grew 20× while per-step time grew ~3×. That's the entire reason every serving engine batches.

## The two batching strategies

### Static batching

Wait for N requests to arrive, run them as one batch, return all responses, done.

Problems:
1. **Padding waste.** If request A is 50 tokens and request B is 5000 tokens, you pad A to 5000. The GPU computes 4950 useless positions for A.
2. **Head-of-line blocking.** A batch finishes only when its slowest request finishes. Fast users wait for slow ones.
3. **Wait-vs-throughput trap.** Wait longer to fill the batch → bigger batches → higher throughput but worse TTFT. Wait less → smaller batches → better TTFT but lower throughput.

This topic builds static batching anyway. Why? Because you have to *feel* these problems to understand why continuous batching (Level 4) was invented.

### Continuous batching (preview, Level 4)

The fix: don't lock requests into a fixed batch. After every decode step, finished requests leave; new requests join. The batch is a *living set*, not a snapshot. vLLM and SGLang both do this. We'll build it ourselves in Level 4.

## The micro-batching architecture

The pattern in code:

```
┌─────────────────┐
│  HTTP request   │  → put on queue, await response future
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  Async queue    │  ← multiple requests pile up
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  Batcher loop   │  every 10ms (or 8 requests, whichever first):
│                 │     drain queue → batch them → run model → split outputs
└─────────────────┘
```

Two parameters control everything:
- **`max_wait_ms`** — how long the batcher waits before launching a partial batch
- **`max_batch_size`** — the cap

Tune these to the throughput-vs-latency curve you want.

## Padding and attention masks

Padding right-aligns the batch:

```
Request A (50 tokens):    [A_1, A_2, ..., A_50, PAD, PAD, ..., PAD]   length 5000
Request B (5000 tokens):  [B_1, B_2, B_3, ..., ......., B_5000]       length 5000
```

The model needs to know which positions are real and which are padding. That's the **attention mask**:

```
mask_A = [1,1,1,...,1,0,0,...,0]  (1=real, 0=padded)
mask_B = [1,1,1,...,1,1,1,...,1]
```

`tokenizer(prompts, padding=True, return_tensors="pt")` does this for you. Pass `inputs["attention_mask"]` to `model.generate()`.

## Splitting batched outputs back to per-request

After `model.generate()` returns, you have a `(batch_size, max_seq_length)` tensor. You need to:

1. Slice off the prompt tokens (different length per request — track original lengths)
2. Trim trailing padding/EOS tokens per request
3. Decode each row separately

`generate()` returns padded outputs. You must record `input_lengths_per_request` before generation so you can slice correctly afterward.

## Pitfalls

1. **Forgetting `attention_mask`.** Model attends to padding tokens, output is garbage. Always pass the mask.
2. **`return_tensors="pt"` without `padding=True`.** Tokenizer raises on variable-length input. Must enable padding.
3. **`pad_token_id` not set.** Some models (LLaMA, GPT-2) don't have a pad token by default. Set it to `eos_token_id` before tokenizing.
4. **Treating batch size as free.** Bigger batches use more memory linearly (KV cache) and can OOM. There's a hardware cap.
5. **Only running one experiment.** Throughput-vs-latency is a *curve*, not a point. Sweep batch sizes and record.

## What the measurements should show

Sweep batch sizes 1, 2, 4, 8, 16. Plot throughput (tokens/sec) and p99 latency (ms). You'll see:

- Throughput rises sharply at first, flattens at high batch sizes (memory-bound regime)
- p99 latency rises slowly at first, then explodes when the batch gets too big to finish in time

The "knee" — where throughput gain levels off — is your sweet spot for that workload. This is **G1 from the Level 1 project**.
