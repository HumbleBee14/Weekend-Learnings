"""Break 04 — optional kwargs plumbing (the HF Transformers pattern).

Real-world models pass dozens of optional kwargs forward through every block.
Dynamo specializes on each combination of which kwargs are None and which
are not, and on the type of any non-None kwarg. Mix it with an optional Cache
object whose type can be None / DynamicCache / StaticCache, and you blow up
the guard space and recompile every call.

Run as-is, then apply the FIX HERE and re-run with FULLGRAPH=1.
"""

from __future__ import annotations

import os
from typing import Optional

import torch


class CacheA:
    def __init__(self) -> None:
        self.x = torch.zeros(4)


class CacheB:
    def __init__(self) -> None:
        self.x = torch.zeros(4)


def model_forward(
    x: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    cache: Optional[object] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    **kwargs,
) -> torch.Tensor:
    y = x * 2
    if attn_mask is not None:
        y = y + attn_mask
    if cache is not None and isinstance(cache, CacheA):
        y = y + cache.x
    # ===== FIX HERE =====
    # Option A: do not pass **kwargs through. Strip to the args the block
    #           actually uses. This is what Transformers v5 is moving toward.
    # Option B: pin the cache type. Always pass CacheA, never None / CacheB.
    # Option C: torch._dynamo.config.error_on_graph_break = False around just
    #           the kwargs plumbing; accept a single break there.
    # ====================
    return y.relu()


def main() -> None:
    fullgraph = os.environ.get("FULLGRAPH") == "1"
    compiled = torch.compile(model_forward, fullgraph=fullgraph)

    x = torch.randn(64)
    mask = torch.randn(64)

    # First call: cache=None
    out = compiled(x, attn_mask=mask, cache=None, use_cache=False)
    # Second call: cache=CacheA — different kwargs shape, may recompile
    out = compiled(x, attn_mask=mask, cache=CacheA(), use_cache=True)
    # Third: a different cache type — definitely recompile
    out = compiled(x, attn_mask=None, cache=CacheB(), use_cache=True)
    print(f"ok, last out mean: {out.mean().item():.4f}")


if __name__ == "__main__":
    main()
