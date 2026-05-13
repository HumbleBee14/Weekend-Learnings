"""Packed multi-document attention via mask_mod. Each token attends within its document.

Run on GPU:
    python document_mask.py
"""
from __future__ import annotations

import torch
import triton
from torch.nn.attention.flex_attention import flex_attention, create_block_mask


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    torch.manual_seed(0)

    # Four documents of length 2048, packed into N=8192.
    seq_lens = [2048, 2048, 2048, 2048]
    N = sum(seq_lens)
    doc_ids = torch.cat([torch.full((L,), i, dtype=torch.int32) for i, L in enumerate(seq_lens)]).to("cuda")

    def doc_mask_fn(b, h, q_idx, kv_idx):
        return doc_ids[q_idx] == doc_ids[kv_idx]

    block_mask = create_block_mask(doc_mask_fn, B=None, H=None, Q_LEN=N, KV_LEN=N, device="cuda")
    print(f"doc-mask BlockMask kept-fraction: {block_mask.sparsity():.4f}  (expected ~0.25)")

    B, H, D = 1, 8, 64
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)

    flex = torch.compile(flex_attention, dynamic=False)
    out = flex(q, k, v, block_mask=block_mask)

    # Sanity: rows from doc 0 should produce outputs that depend only on doc-0 V.
    # Replace doc-1..3 V with garbage; doc-0 rows of out should be unchanged.
    v_perturbed = v.clone()
    v_perturbed[:, :, seq_lens[0]:] = 1000.0
    out_perturbed = flex(q, k, v_perturbed, block_mask=block_mask)
    doc0_err = (out[:, :, :seq_lens[0]] - out_perturbed[:, :, :seq_lens[0]]).abs().max().item()
    print(f"doc-0 output stability under doc-1..3 V perturbation: max delta = {doc0_err:.2e}")
    assert doc0_err < 1e-2, "doc-0 attended to other documents; mask is not effective"

    ms = triton.testing.do_bench(lambda: flex(q, k, v, block_mask=block_mask), warmup=25, rep=100)
    flops = 4 * B * H * N * N * D * block_mask.sparsity()
    print(f"\nshape (B,H,N,D)=({B},{H},{N},{D}) bf16 doc-mask")
    print(f"  {ms:.3f} ms   ~{flops/ms/1e9:.1f} effective TFLOPs/s (post-sparsity)")


if __name__ == "__main__":
    main()
