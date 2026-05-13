"""FlashInfer paged-KV decode. Measure cold JIT vs warm run.

Install:
    pip install flashinfer

Run:
    python paged_decode_demo.py
"""
from __future__ import annotations

import time

import torch
import triton

try:
    import flashinfer
except ImportError:
    raise SystemExit("pip install flashinfer first.")


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    torch.manual_seed(0)

    num_qo_heads = 16
    num_kv_heads = 16
    head_dim = 128
    page_size = 16
    max_num_pages = 8192

    # 32 requests with context lengths drawn from a Zipf-like spread.
    batch_size = 32
    rng = torch.Generator().manual_seed(0)
    ctx_lens = torch.tensor(
        [int(x.item()) for x in (torch.rand(batch_size, generator=rng) * 7500 + 128)],
        dtype=torch.int32,
    )
    print(f"context lens (sorted): {sorted(ctx_lens.tolist())[:5]} ... {sorted(ctx_lens.tolist())[-5:]}")

    pages_per_req = [(L + page_size - 1) // page_size for L in ctx_lens]
    last_page_len = torch.tensor([L - (p - 1) * page_size if (p := pp) > 0 else 0
                                  for L, pp in zip(ctx_lens.tolist(), pages_per_req)],
                                 dtype=torch.int32, device="cuda")
    total_pages = sum(pages_per_req)
    assert total_pages <= max_num_pages

    kv_page_indices = torch.arange(total_pages, dtype=torch.int32, device="cuda")
    kv_page_indptr = torch.tensor([0] + list(__import__("itertools").accumulate(pages_per_req)),
                                   dtype=torch.int32, device="cuda")

    kv_cache = torch.randn(
        max_num_pages, 2, page_size, num_kv_heads, head_dim,
        device="cuda", dtype=torch.bfloat16,
    )
    # One query token per request (decode shape).
    q = torch.randn(batch_size, num_qo_heads, head_dim, device="cuda", dtype=torch.bfloat16)

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrap = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD")

    # Cold plan (JIT compile).
    t0 = time.time()
    wrap.plan(
        indptr=kv_page_indptr,
        indices=kv_page_indices,
        last_page_len=last_page_len,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    cold_plan_s = time.time() - t0

    # Warm run.
    ms = triton.testing.do_bench(lambda: wrap.run(q, kv_cache), warmup=25, rep=100)

    print(f"\ncold plan() (JIT compile + cache):  {cold_plan_s*1000:.1f} ms")
    print(f"warm run() (steady state):          {ms:.3f} ms")
    print(f"ratio: {cold_plan_s*1000/ms:.0f}x")
    print(f"\nTakeaway: a server warm-starts once, then runs at the warm cost forever.")
    print("Replanning is only triggered when the metadata signature changes.")


if __name__ == "__main__":
    main()
