"""Audit the LLaMA block for graph breaks.

Runs the block with fullgraph=True. If it fails, prints torch._dynamo.explain
output so you can find the offender. Once it passes cleanly with the custom
attention op installed, sub-module 03 lessons are now in your fingers.
"""

from __future__ import annotations

import torch

from llama_block import make_block
from piecewise_wrapper import install_custom_attn


def main() -> None:
    assert torch.cuda.is_available()
    install_custom_attn()

    block = make_block(hidden=2048)  # smaller hidden to fit on a T4
    x = torch.randn(1, 64, 2048, device="cuda", dtype=torch.bfloat16)

    # First: print Dynamo's explanation (won't error on breaks)
    print("=== torch._dynamo.explain ===")
    explanation = torch._dynamo.explain(block)(x)
    print(explanation)
    print()

    # Then: try fullgraph=True. This errors on any break.
    print("=== fullgraph=True ===")
    compiled = torch.compile(block, fullgraph=True)
    try:
        with torch.inference_mode():
            y = compiled(x)
        print(f"PASS — output mean: {y.float().mean().item():.4f}")
    except Exception as e:
        print(f"FAIL — {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
