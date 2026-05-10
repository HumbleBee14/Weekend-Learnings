# 07 — Expert Parallelism

EP is the parallelism axis for Mixture-of-Experts (MoE) models. Distribute the *experts* across devices. Each token in a batch is routed to its top-K experts via all-to-all. Without EP, every device would store every expert — impossible at trillion-parameter scale.

## What MoE does

A standard transformer FFN: every token goes through one `[hidden → 4·hidden → hidden]` MLP.

A MoE FFN: a small router computes per-token logits over `E` experts, picks top-K (typically K=1 or 2), and the token's MLP forward runs only on its chosen experts. Each expert is an independent `[hidden → 4·hidden → hidden]` MLP.

```
without MoE                          with MoE (E=8, K=2)
─────────────                        ─────────────────
token → MLP → output                 token ──┬── router → [expert3, expert6]
                                              ├── expert3.forward(token)
                                              └── expert6.forward(token)
                                              + weighted sum
```

Effective parameter count: `E · expert_size`. Active parameters per token: `K · expert_size`. DeepSeek-V3: 671B total, 37B active. Mixtral 8×22B: 141B total, 39B active.

## Why MoE is awkward to parallelize

The router decides per token which experts run. Different tokens in the same batch go to different experts. With `E` experts and uniform routing, each expert receives ~`B·K/E` tokens.

Without EP, every device runs every expert on every token routed there — but every device must hold every expert. At E=256 with 2B-param experts, that's 512B of weights replicated everywhere. Won't fit.

With EP, each device owns a subset of experts. The token-routing pattern becomes:

```
1. Local router runs on all tokens local to this device.
2. all-to-all: send each token to the device that owns its chosen expert.
3. Local expert forward runs on the tokens that landed here.
4. all-to-all: send each token's output back to its origin device.
5. Combine top-K outputs (weighted sum) back at origin.
```

Two all-to-all collectives per MoE layer. That's the expensive bit. All-to-all bandwidth scales as `O(N²)` flows through the fabric — the worst-case for any collective.

## EP placement

```
ep_size = number of devices that share the expert pool
e_per_device = E / ep_size

device 0:  experts 0, 1, 2, 3
device 1:  experts 4, 5, 6, 7
device 2:  experts 8, 9, 10, 11
...
```

Composes with TP, FSDP, PP. Common pattern at frontier scale (DeepSeek-V3): EP=64 across nodes, with TP inside each node for the shared layers (attention, MoE router, embed).

## Token-dropping vs no-token-dropping

If router output is unbalanced — one expert gets many more tokens than its capacity — choices:

- **Token dropping** (Switch Transformer, GShard): cap each expert at `capacity = α · B·K/E` tokens. Excess tokens skip the MoE block (residual passes through). Faster, deterministic shape, slight quality loss.
- **No-dropping** (Megatron-Core 2026, DeepSeek): pad each expert to a worst-case capacity, run real all-to-all. Slower, wastes some compute on padding, no quality loss.

In 2026 the field has moved mostly to no-dropping with **load-balancing loss** to keep router output even, plus **expert capacity factor 1.0–1.25** to bound the all-to-all volume.

## DeepSeek-V3 — the frontier MoE

671B total, 256 experts, K=8 active per token (compared to K=2 in older MoE). Innovations:
- **Auxiliary-loss-free load balancing**: instead of a load-balance loss term, dynamically bias the router via expert "popularity" tracking.
- **Multi-Token Prediction (MTP)** during training: predicts more than the next token.
- **DualPipe** + heavy EP across nodes.

Worth reading the paper end-to-end. Reference: [arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437).

## Practical sizes

- 8B-class MoE: E=8 or 16, K=2. Single-node EP=2 or 4.
- 70B-class MoE: E=64, K=2. Multi-node EP=8 or 16.
- 400B+ MoE: E=128–256, K=2–8. Frontier-scale EP=32–64.

## Build steps

This topic is mostly conceptual at home-cluster scale. Two things you can do:

1. **Read the Megatron-Core EP docs**: [github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md).
2. **Run the small EP demo** in `ep_demo.py` — a 4-expert toy model with 2 GPUs and EP=2. The all-to-all token routing is the part to study.

If you have access to a small MoE checkpoint (Qwen2.5-MoE-A2.7B is small enough), run it with the torchtitan or vLLM EP backend on 2 GPUs.

## Reference

- Switch Transformer: [arxiv.org/abs/2101.03961](https://arxiv.org/abs/2101.03961)
- GShard: [arxiv.org/abs/2006.16668](https://arxiv.org/abs/2006.16668)
- DeepSeek-V3: [arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437)
- Megatron-Core MoE docs: [github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe)
- Mixtral 8×7B paper: [arxiv.org/abs/2401.04088](https://arxiv.org/abs/2401.04088)
