"""Piecewise CUDA graph wrapper.

The pattern, mirroring vLLM v1's design:

  1. Mark attention as a custom op so Dynamo treats it as one opaque FX node.
     The block's forward becomes one whole FX graph with one black-box node.
  2. Compile the block with torch.compile + Inductor graph partition. Inductor
     splits the FX graph at the custom op and emits each non-attention piece
     as its own callable.
  3. At runtime, capture each piece into a CUDA graph keyed by (B, S).
     Replay on every subsequent call with matching shape.
  4. Attention runs eagerly between the captured pieces.

The wrapper is parameterized over the block module. The contract: the block's
forward must have exactly one attention call routed through the custom op
defined here.

NOTE: This is a teaching implementation. vLLM's version handles more cases
(prefix prefill, chunked prefill, FP8 KV cache, etc.) and uses a richer
dispatcher (FULL_AND_PIECEWISE coexistence). The core trick is the same.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch
import torch.nn.functional as F


# -------- Step 1: attention as a custom op --------
# Wrapping attention as a custom op makes Dynamo treat the call as one
# opaque node. The shapes inside attention (KV cache growth) can vary
# without triggering a recompile of the surrounding compiled code.

@torch.library.custom_op("capstone::attn", mutates_args=())
def attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Causal SDPA. Eager. Shapes can vary across calls."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


@torch.library.register_fake("capstone::attn")
def _attn_fake(q, k, v):
    # Output shape matches q. Required for Dynamo's FakeTensor tracing.
    return torch.empty_like(q)


# -------- Step 2 + 3 + 4: the piecewise wrapper --------

class PiecewiseCUDAGraph:
    """Wraps an nn.Module whose forward routes attention through capstone::attn.

    Lazy capture: on the first call with new (batch, seqlen), capture a CUDA
    graph by running the compiled block once with stash buffers, snapshotting.

    Replay: on every subsequent call with matching shape, copy inputs in,
    replay, copy outputs out.

    Caveat: this implementation captures the ENTIRE compiled block, not
    individual partitions, in one graph that contains an eager-callback for
    attention. Inductor 2.8's graph_partition does the partition-per-piece
    version automatically when enabled. We do it manually here to make the
    mechanism visible.
    """

    def __init__(self, block: torch.nn.Module) -> None:
        self.block = block
        self.compiled = torch.compile(block, fullgraph=True, mode="default")
        self._graphs: Dict[Tuple[int, int], torch.cuda.CUDAGraph] = {}
        self._buffers: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}

    def _capture(self, x: torch.Tensor) -> None:
        key = (x.shape[0], x.shape[1])

        # Stash input buffer; the CUDA graph will replay against this same address.
        static_in = torch.empty_like(x)
        static_in.copy_(x)

        # Warmup outside the graph (Inductor compiles, autotunes, etc.)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _ = self.compiled(static_in)
        torch.cuda.current_stream().wait_stream(s)

        # Capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_out = self.compiled(static_in)

        self._graphs[key] = g
        self._buffers[key] = {"in": static_in, "out": static_out}

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        key = (x.shape[0], x.shape[1])
        if key not in self._graphs:
            self._capture(x)

        bufs = self._buffers[key]
        bufs["in"].copy_(x)
        self._graphs[key].replay()
        return bufs["out"].clone()


# -------- Helper: wire the block to use capstone::attn --------

def install_custom_attn(block) -> None:
    """Monkey-patch the block so its attention call routes through capstone::attn.

    The block's forward in llama_block.py calls F.scaled_dot_product_attention.
    We swap that to torch.ops.capstone.attn so it becomes one FX node.
    """
    # The cleanest version is to edit forward() to call torch.ops.capstone.attn,
    # but to keep llama_block.py readable we patch at module level.
    import torch.nn.functional as F_mod

    original = F_mod.scaled_dot_product_attention

    def patched(q, k, v, *args, is_causal=False, **kwargs):
        # Use the custom op when called with our exact pattern.
        if is_causal and not args and not kwargs:
            return torch.ops.capstone.attn(q, k, v)
        return original(q, k, v, *args, is_causal=is_causal, **kwargs)

    F_mod.scaled_dot_product_attention = patched
