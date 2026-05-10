# 06 — Pipeline Parallelism

PP splits the model *depth-wise* across GPUs. GPU 0 has layers 0–7, GPU 1 has 8–15, etc. Microbatches flow through the pipeline. The key cost: the **bubble** — idle time when the pipeline is filling at the start of a step and draining at the end.

## The bubble

```
4 stages, 4 microbatches, naive (GPipe) schedule:

Stage 0:  F0  F1  F2  F3                          B3  B2  B1  B0
Stage 1:      F0  F1  F2  F3                  B3  B2  B1  B0
Stage 2:          F0  F1  F2  F3          B3  B2  B1  B0
Stage 3:              F0  F1  F2  F3  B3  B2  B1  B0

       └─ filling ─┘                      └─ draining ─┘
        (bubble)                            (bubble)
```

Bubble fraction for GPipe with `S` stages and `M` microbatches: `(S-1)/(M+S-1)`. To get bubble below 10% you need M ≥ 9·(S-1). For 8 stages that's M ≥ 63 microbatches per step. Practical, but the activation memory of all those microbatches in flight is the constraint.

## Schedules

### GPipe (2018)

Naive. All forwards then all backwards. Big bubble. Big activation memory (every microbatch's activations live until backward starts). Historical only.

### 1F1B / PipeDream-Flush

Alternate one forward with one backward as soon as the pipeline is full. Memory: bounded by the pipeline depth (only `S` microbatches' activations live at once). Bubble: same as GPipe theoretically, but the steady-state region is more compact.

```
4 stages, 4 microbatches, 1F1B:

Stage 0:  F0  F1  F2  F3  B0  B1  B2  B3
Stage 1:      F0  F1  F2  F3  B0  B1  B2  B3
Stage 2:          F0  F1  F2  B0  F3  B1  B2  B3
Stage 3:              F0  B0  F1  B1  F2  B2  F3  B3
```

Standard sync schedule. Implemented in PyTorch's `torch.distributed.pipelining` and Megatron-Core.

### Interleaved 1F1B (Megatron, 2021)

Each device owns multiple non-contiguous chunks of layers ("virtual stages"). E.g., GPU 0 owns layers 0, 1, 16, 17 if you have 8 stages × 2 chunks. More fine-grained microbatch flow → smaller bubble. Cost: 2× the comms (each microbatch crosses more boundaries).

Bubble fraction: `(S-1) / (M·V + S-1)` where V = chunks-per-device. With V=4 and S=8, M=8 microbatches gets you ~18% bubble vs 47% for plain 1F1B.

### Zero Bubble Pipeline (ZB-V, ICLR 2024)

The 2026 practical default. Insight: the backward pass actually computes two things — the gradient w.r.t. inputs (`dX`) and the gradient w.r.t. weights (`dW`). `dX` is needed by the previous stage's backward; `dW` is local. ZB-V splits these and schedules `dW` into bubble slots.

```
ZB-V (4 stages, 4 microbatches, simplified):

Stage 0:  F0  F1  F2  F3                  dW0/dX0  dW1/dX1  dW2/dX2  dW3/dX3
Stage 1:      F0  F1  F2  F3      dX3..dX0 woven into the pipeline as dW slots fill
...
```

Same memory as 1F1B. Near-zero bubble. Reference: [arxiv.org/abs/2401.10241](https://arxiv.org/abs/2401.10241), implemented in torchtitan and Megatron-Core.

### DualPipe (DeepSeek-V3, late 2024)

Bidirectional fwd/bwd overlap. The pipeline runs in two directions simultaneously, doubling effective utilization. Cited heavily in 2026 — the DeepSeek-V3 paper made this widely known. Reference: [arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437).

## When to reach for PP

PP communicates at *stage boundaries* — point-to-point send/recv of microbatch activations and gradients. Volume: per microbatch, `seq_len × hidden_dim × bytes`. For 2K-token, 4K-hidden, BF16: 16 MB per microbatch boundary.

That's *much* less than TP's per-layer all-reduce at the same scale. PP tolerates inter-node bandwidth. PP is what you reach for when FSDP+TP still doesn't fit the model on one node.

Decision rule:
- Model fits in one NVLink domain after FSDP+TP → don't use PP.
- Model needs to cross node boundaries → PP first, since it's the most bandwidth-friendly inter-node parallelism.

## Memory math

For `S` stages and `B` microbatches in flight under 1F1B:
- Per-stage activation memory: `B × layer_acts × layers_per_stage`
- Per-stage parameter memory: `params/S` (compose with FSDP for further reduction)

Frontier-scale runs typically use S=8–16, B=64–128, plus FSDP across the DP axis.

## torchtitan / `torch.distributed.pipelining`

```python
from torch.distributed.pipelining import pipeline, Schedule1F1B

# split the model into stages (manually or by tracing)
stage = pipeline(
    model,
    mb_args=(microbatch_input,),
    split_spec={"layer8": SplitPoint.BEGINNING, "layer16": SplitPoint.BEGINNING},
)
schedule = Schedule1F1B(stage, n_microbatches=8, loss_fn=loss_fn)

# step
schedule.step(input_microbatches, target_microbatches)
```

The `pipeline()` function traces the model, partitions it at the named split points, and produces a `PipelineStage` for the local rank. The `Schedule1F1B` (or `ScheduleZBVZeroBubble`, etc.) drives the microbatch flow.

## Build steps

1. Reuse the Topic 04 transformer (with N≥4 blocks).
2. Manually split into 2 stages: rank 0 owns blocks 0–N/2, rank 1 owns N/2–N.
3. Run a 1F1B schedule with 4 microbatches.
4. Measure step time. Calculate bubble fraction empirically: `1 - (compute_time / wallclock)`.
5. (Optional, if your torch version supports it) Switch to ZB-V and watch the bubble shrink.

## Reference

- 1F1B / PipeDream: [arxiv.org/abs/1806.03377](https://arxiv.org/abs/1806.03377)
- Interleaved 1F1B (Megatron): [arxiv.org/abs/2104.04473](https://arxiv.org/abs/2104.04473)
- Zero Bubble Pipeline: [arxiv.org/abs/2401.10241](https://arxiv.org/abs/2401.10241)
- DeepSeek-V3 DualPipe: [arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437)
- PyTorch pipelining: [pytorch.org/docs/stable/distributed.pipelining.html](https://pytorch.org/docs/stable/distributed.pipelining.html)
- torchtitan PP: [github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/pipeline_llama.py](https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/pipeline_llama.py)
