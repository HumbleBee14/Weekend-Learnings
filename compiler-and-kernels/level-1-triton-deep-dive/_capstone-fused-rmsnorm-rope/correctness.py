"""
Correctness check for fused_rmsnorm_rope across multiple shapes.

If any of these fail, your kernel has a bug. Fix it before running benchmark.py.
The benchmark numbers of a wrong kernel are meaningless.
"""

import torch
from reference import rmsnorm_rope_reference, build_rope_tables
from fused_rmsnorm_rope import fused_naive, fused_autotuned, fused_persistent


def run_one(impl, name, B, S, H, dtype, max_seqlen=8192, atol=1e-2):
    torch.manual_seed(0)
    device = "cuda"
    x = torch.randn(B, S, H, device=device, dtype=dtype) * 0.1
    w = (torch.randn(H, device=device, dtype=dtype) * 0.5 + 1.0)
    cos, sin = build_rope_tables(max_seqlen, H, device=device)
    pos = torch.arange(S, device=device, dtype=torch.int64).expand(B, S).contiguous()

    out_t = impl(x, w, cos, sin, pos)
    out_r = rmsnorm_rope_reference(x, w, cos, sin, pos)

    diff = (out_t - out_r).abs().max().item()
    ok = "OK " if diff < atol else "WRONG"
    print(f"  {name:<20} B={B} S={S:<5} H={H:<5} {dtype}  max_diff={diff:.2e}  {ok}")
    if diff >= atol:
        # Show first mismatch for debugging
        idx = (out_t - out_r).abs().argmax()
        print(f"    first mismatch at flat-idx {idx.item()}")
        print(f"    triton={out_t.flatten()[idx].item():.4f}  ref={out_r.flatten()[idx].item():.4f}")
    return diff < atol


def main():
    print("Testing fused_naive...")
    cases = [
        (1, 8, 128),
        (2, 16, 256),
        (4, 128, 512),
        (8, 512, 4096),
        (32, 2048, 4096),
        (1, 4097, 4096),    # non-power-of-2 S — should still work
    ]
    for impl, name in [
        (fused_naive, "naive"),
        (fused_autotuned, "autotuned"),
        (fused_persistent, "persistent"),
    ]:
        all_ok = True
        for B, S, H in cases:
            ok = run_one(impl, name, B, S, H, torch.float16)
            all_ok = all_ok and ok
        print(f"  -> {name}: {'ALL OK' if all_ok else 'HAD FAILURES'}\n")


if __name__ == "__main__":
    main()
