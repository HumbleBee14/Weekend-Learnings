# 10 — Multi-LoRA Serving

## Why this exists

Every company fine-tunes adapters: one per customer, one per feature, one per dialect, one per safety policy. Naive deployment — one server per LoRA — is unaffordable. A 7B base model is ~14GB in BF16; an L4 GPU is $0.40-0.80/hr; you can't run 50 of them per tenant.

Multi-LoRA serving fixes this. **Load the base model once. Hot-swap adapters per request.** All adapters share the same KV cache space, the same continuous batch, the same scheduler. Memory cost per additional LoRA is ~10-50 MB vs 14 GB for a full replica.

## What a LoRA actually is (one paragraph)

For each linear layer `Y = X @ W` you replace `W` with `W + (B @ A) * (alpha/r)`, where `A` is `r × d_in` and `B` is `d_out × r`. `r` is small (typically 8-64). At rank 16 on a 7B, the LoRA is ~30 MB. Inference: pre-merge `B @ A` into a single small matrix, or apply it as a separate kernel on top of the base GEMM.

## The two ways to apply LoRA at serving time

```
Option A — pre-merged
─────────────────────
  W_eff = W + B@A
  forward: Y = X @ W_eff
  pros: zero overhead at inference
  cons: switching adapters requires recomputing W_eff per request — expensive

Option B — additive (Punica / S-LoRA)
──────────────────────────────────────
  forward: Y = X @ W + (X @ A) @ B * scale
  pros: switch adapters by swapping (A, B); base GEMM unchanged
  cons: extra GEMM per layer; specialized kernels needed for batching
        across heterogeneous adapters in one batch
```

Production engines use Option B with batched LoRA kernels:

- **Punica kernels** (Chen et al., 2023) — group requests by which adapter they use; one specialized GEMM per group, fused with the base. Used in vLLM.
- **S-LoRA** (Sheng et al., 2023) — unified memory pool for adapters; serves thousands of LoRAs from one base. Heterogeneous-batch scheduling.
- **LoRAX** — Predibase's open-source multi-LoRA stack; similar approach.

The upshot: a single 7B-base server can serve a hundred LoRAs at once with throughput within ~10-20% of the no-LoRA baseline.

## What this looks like in vLLM

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --enable-lora \
    --max-loras 16 \
    --max-lora-rank 64 \
    --lora-modules \
        code-lora=./adapters/code \
        poetry-lora=./adapters/poetry \
        legal-lora=./adapters/legal
```

Per-request, choose the adapter via the `model` field:

```python
client.chat.completions.create(
    model="code-lora",  # routes to the LoRA
    messages=[...],
)
```

Flags:
- `--max-loras` — how many adapters can be *active in one batch*. Larger = more memory for adapter staging buffers + more kernel-grouping overhead.
- `--max-lora-rank` — max rank you'll allow. Larger = more memory per adapter.
- `--max-cpu-loras` — how many to keep in CPU memory ready to swap to GPU. Eviction-and-reload happens transparently if you exceed `--max-loras`.

## What you measure

```
1. Throughput vs single-LoRA baseline
   Expected: ~85-95% of baseline. Higher means your LoRAs are effectively free.

2. Memory cost per additional LoRA
   Roughly: rank * (d_model + d_proj) * 2 bytes per BF16 layer * n_layers
   For r=16 on a 7B: ~30 MB.

3. Switching latency
   First request to a cold adapter: pays the CPU→GPU swap (tens of ms).
   Steady state with many adapters: should be ~0 extra.

4. Heterogeneous-batch behavior
   Mixed batch (some requests with LoRA-A, some with LoRA-B, some without).
   Throughput should not collapse — Punica kernels handle this.
```

## When NOT to use multi-LoRA

- **One model per tenant with strong isolation requirements.** Multi-LoRA shares the base; if your threat model requires zero shared state, separate replicas.
- **Drastically different ranks per adapter.** Scheduler grouping cost grows.
- **Inference-time-merged for highest QPS, single LoRA.** Pre-merging the LoRA into base weights gives the no-LoRA throughput; you give up dynamism.

## The 2026 frontier

- **MoLA / heterogeneous LoRA fleets** — different ranks, different precisions in one base.
- **LoRA + spec decode interactions** — draft model with adapters, target with the same. Open research, partial support in vLLM.
- **Multi-LoRA + multi-tenancy** — per-tenant cache salting (RFC #16016) so two tenants never share a prefix-cache hit even if their prompts collide.

## Pitfalls

1. **`--max-loras` set too low.** You'll see latency spikes when the active set thrashes.
2. **Wildly different ranks across LoRAs in one batch.** Pad-to-max-rank waste. Group your fine-tunes around a small number of canonical ranks (e.g., r=16 and r=64 only).
3. **Forgetting LoRAs need to live in PEFT format.** vLLM accepts the standard `adapter_config.json` + `adapter_model.safetensors`. Your training output should already be in this shape.
4. **Per-request quality drift.** A LoRA trained against an older base doesn't always work on a new base. Pin the base model version.

## What to do this topic

1. Train two tiny LoRAs on top of your base model (PEFT + Trainer; 10 minutes each). Pick contrasting domains — code and poetry is the canonical demo.
2. Serve through vLLM with both adapters loaded.
3. Run `multi_lora_demo.py` — sends interleaved requests for both LoRAs and measures throughput vs single-LoRA baseline.
4. Try a third LoRA. Confirm memory and throughput scale linearly (or close to it).

## References

- vLLM multi-LoRA docs — https://docs.vllm.ai/en/stable/features/lora.html
- PEFT (LoRA training) — https://huggingface.co/docs/peft
- Punica paper — https://arxiv.org/abs/2310.18547
- S-LoRA paper — https://arxiv.org/abs/2311.03285
- LoRAX (Predibase) — https://github.com/predibase/lorax
- LoRA original paper — https://arxiv.org/abs/2106.09685
- Cache salting RFC — https://github.com/vllm-project/vllm/issues/16016
