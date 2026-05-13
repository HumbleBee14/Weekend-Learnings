"""FlashInfer ragged-batch prefill vs padded SDPA. A100 expected.

Install:
    pip install flashinfer

Run:
    python ragged_prefill_demo.py
"""
from __future__ import annotations

import math

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

    # Realistic mixed-length batch.
    seq_lens = [128, 4096, 512, 2048, 64, 8192, 256, 1024]
    total = sum(seq_lens)
    print(f"batch lens: {seq_lens}; total = {total}")

    # Ragged tensors.
    q = torch.randn(total, num_qo_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(total, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(total, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    indptr = torch.tensor([0] + list(__import__("itertools").accumulate(seq_lens)), dtype=torch.int32, device="cuda")

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrap = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, kv_layout="NHD")
    wrap.plan(
        qo_indptr=indptr, kv_indptr=indptr,
        num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim, head_dim_vo=head_dim,
        causal=True, q_data_type=torch.bfloat16,
    )

    def run_flashinfer():
        return wrap.run(q, k, v)

    # Padded SDPA baseline.
    B = len(seq_lens)
    Nmax = max(seq_lens)
    q_pad = torch.zeros(B, num_qo_heads, Nmax, head_dim, device="cuda", dtype=torch.bfloat16)
    k_pad = torch.zeros(B, num_kv_heads, Nmax, head_dim, device="cuda", dtype=torch.bfloat16)
    v_pad = torch.zeros(B, num_kv_heads, Nmax, head_dim, device="cuda", dtype=torch.bfloat16)
    for i, (s, e) in enumerate(zip([0] + list(__import__("itertools").accumulate(seq_lens))[:-1],
                                    list(__import__("itertools").accumulate(seq_lens)))):
        q_pad[i, :, : seq_lens[i]] = q[s:e].transpose(0, 1)
        k_pad[i, :, : seq_lens[i]] = k[s:e].transpose(0, 1)
        v_pad[i, :, : seq_lens[i]] = v[s:e].transpose(0, 1)

    def run_padded_sdpa():
        return torch.nn.functional.scaled_dot_product_attention(q_pad, k_pad, v_pad, is_causal=True)

    ms_fi = triton.testing.do_bench(run_flashinfer, warmup=25, rep=100)
    ms_sdpa = triton.testing.do_bench(run_padded_sdpa, warmup=25, rep=100)

    real_flops = sum(2 * 2 * num_qo_heads * L * L * head_dim * 0.5 for L in seq_lens)  # 0.5 for causal
    pad_flops = B * 2 * 2 * num_qo_heads * Nmax * Nmax * head_dim * 0.5
    print(f"\nFlashInfer ragged prefill: {ms_fi:.3f} ms  ({real_flops/ms_fi/1e9:.1f} eff TFLOPs/s)")
    print(f"Padded SDPA prefill:       {ms_sdpa:.3f} ms  ({pad_flops/ms_sdpa/1e9:.1f} TFLOPs/s nominal)")
    print(f"Wallclock speedup (ragged vs padded): {ms_sdpa/ms_fi:.2f}x")
    print(f"Wasted compute in padded path: {(pad_flops - real_flops)/pad_flops*100:.1f}%")


if __name__ == "__main__":
    main()
