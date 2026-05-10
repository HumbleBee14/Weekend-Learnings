# 07 — Expert Parallelism

## Files

- `CONCEPTS.md` — what MoE does, why EP is awkward, token-dropping vs no-dropping, DeepSeek-V3 frontier MoE
- `ep_demo.py` — toy EP=2 with 4 experts; two all-to-alls per layer; top-1 routing for clarity

## Quickstart

```bash
torchrun --standalone --nproc_per_node=2 ep_demo.py
```

## Expected output

```
rank0: sent [9, 7], recv [9, 8]
rank0: input  norm: 32.18
rank0: output norm: 28.51
each token visited exactly one expert via two all-to-alls
```

The asymmetric send/recv counts are the "load imbalance" that load-balance loss tries to fix in real MoE training.

## Try

- Set `n_tokens_per_device = 1024` and time the all-to-alls. Compare to the same total bytes via all-reduce (Topic 00). All-to-all is harder on the fabric than all-reduce at the same volume.
- Switch to top-K=2: each token visits 2 experts. The sent tensor doubles. Outputs combine via a weighted sum.
- Force routing imbalance (e.g., bias all logits toward expert 0). Watch the all-to-all fall apart — one device does all the work, the other waits.
- Compose with FSDP for the non-expert layers (router, attention) — see torchtitan's MoE recipe.

## Where this goes

- Topic 09 — composing EP with the other axes for MoE pretraining
- Topic 10 — torchtitan-or-megatron has the production version of this
